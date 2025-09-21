# MIDAS_RADINT: High-Performance Azimuthal Integration Suite

This document describes a suite of two programs designed for high-throughput, real-time azimuthal integration of 2D detector data: `DetectorMapper` and `IntegratorFitPeaksGPUStream`.

## Overview

The integration process is split into two stages for maximum performance:

1.  **Offline Pre-computation (`DetectorMapper`):** A CPU-based program that runs **once** to analyze the experimental geometry. It creates a highly optimized lookup table that maps every detector pixel to its corresponding location in a polar coordinate system (`R`, `η`). This is the key to the system's speed.
2.  **Real-time Integration (`IntegratorFitPeaksGPUStream`):** A GPU-accelerated program that runs continuously. It listens for 2D detector data on a network socket, uses the pre-computed maps to perform the integration in microseconds, and optionally performs peak fitting on the resulting 1D profile.

This two-stage approach moves the most computationally expensive geometric calculations offline, allowing the GPU to focus solely on fast data processing and arithmetic during an experiment.

---

## 1. `DetectorMapper`

This command-line tool generates the essential `Map.bin` and `nMap.bin` files required by the integrator. It must be run anytime the experimental geometry (detector distance, tilts, etc.) or the desired binning scheme changes.

### Requirements

*   **System:** A standard Linux or macOS machine with a C compiler (like `gcc`).
*   **CPU:** A modern multi-core CPU is recommended, as the process is computationally intensive.
*   **RAM:** Requires enough RAM to hold the mapping data in memory before writing to disk. For a 2Kx2K detector, 8-16 GB of RAM is sufficient.

### Workflow & Mathematics

The `DetectorMapper` performs a rigorous mapping of the detector's Cartesian pixel grid to the desired final polar coordinate grid.

#### Mathematical Transformations

For each of the four corners of every single pixel on the detector, the program applies a chain of transformations to find its precise location in 3D space and, finally, its polar coordinates:

1.  **Distortion Correction (Optional):** Applies a user-provided distortion map to correct for physical imperfections in the detector, moving `(y, z)` to `(y_distorted, z_distorted)`.
2.  **Coordinate System Shift:** Converts pixel indices to physical units (e.g., mm) based on pixel size and shifts the origin to the beam center `(Ycen, Zcen)`.
3.  **3D Rotations:** Applies a 3x3 rotation matrix derived from the detector tilt angles (`tx`, `ty`, `tz`) to correctly orient the detector plane in the 3D laboratory frame.
4.  **Sample-to-Detector Projection:** Accounts for the sample-to-detector distance (`Lsd`).
5.  **Projection to Polar Coordinates:** The final 3D vector is projected back onto a virtual 2D plane to calculate the definitive polar coordinates:
    *   **Radius (`R`):** The radial distance from the beam center. `R = (Lsd / X_final) * sqrt(Y_final² + Z_final²)`
    *   **Azimuthal Angle (`η`):** The angle in the detector plane. `η = atan2(Y_final, Z_final)`
6.  **Radial Distortion Correction (Optional):** Applies a final polynomial correction to `R` based on the `p0, p1, p2, p3` parameters.

#### Geometric Clipping Algorithm

The core of the program is a complex geometric clipping algorithm that calculates the precise fractional contribution of each pixel to each polar bin.

*   **Broad Phase:** The program first determines a bounding box for each transformed pixel and quickly identifies a list of candidate polar bins that it *might* overlap with.
*   **Narrow Phase:** For each candidate bin, it calculates the exact intersection polygon between the (warped) Cartesian pixel and the (curved) polar bin.
*   **Area Calculation:** Using the **Shoelace formula**, it computes the precise area of this intersection polygon. This area becomes the `fractional contribution`.

#### Output Files

After processing every pixel, the program serializes the results into two binary files:
*   `Map.bin`: A large file containing a list of `(y, z, fraction)` structs for every pixel that contributes to any bin.
*   `nMap.bin`: A smaller index file that, for each polar bin, stores the number of contributing pixels and their starting location within `Map.bin`.

### How to Run

1.  **Prepare a parameter file** (e.g., `mapper.params`) containing all geometric and binning information. See the parameter list below.
2.  **Compile the code:**
    ```bash
    gcc src/DetectorMapper.c -o bin/DetectorMapper -lm -O3
    ```
3.  **Run from the command line:**
    ```bash
    ./bin/DetectorMapper mapper.params
    ```
4.  This will produce `Map.bin` and `nMap.bin` in the current directory.

#### Parameters for `DetectorMapper`

| Parameter | Example | Description |
| :--- | :--- | :--- |
| `tx`, `ty`, `tz`| `tx 0.1` | Detector tilt angles in degrees around X, Y, and Z axes. |
| `px` or `pxY`/`pxZ` | `px 200.0` | Pixel size in microns. Use `pxY` and `pxZ` for non-square pixels. |
| `BC` | `BC 1024.5 1023.8` | Beam Center coordinates (Y, Z) in pixels. |
| `Lsd` | `Lsd 150000.0` | Sample-to-detector distance in microns. |
| `RhoD` | `RhoD 200000.0`| A normalization factor for radial distortion, typically `Lsd`.|
| `p0`,`p1`,`p2`,`p3`|`p0 0.001`| Polynomial coefficients for radial distortion correction. |
| `RMin`, `RMax` | `RMin 50.0` | Minimum and maximum radius for integration, in pixels. |
| `RBinSize` | `RBinSize 0.5` | The width of each radial bin, in pixels. |
| `EtaMin`, `EtaMax`| `EtaMin -180.0`| Minimum and maximum azimuthal angle for integration, in degrees. |
| `EtaBinSize` | `EtaBinSize 1.0` | The width of each azimuthal bin, in degrees. |
| `NrPixels` or `NrPixelsY`/`NrPixelsZ`| `NrPixels 2048 2048` | The dimensions of the detector in pixels. |
| `DistortionFile`| `DistortionFile my_dist.bin`| (Optional) Path to a binary file containing detector distortion maps.|

---

## 2. `IntegratorFitPeaksGPUStream`

This is the high-performance, real-time processing engine. It runs continuously, ingesting data and producing results.

### Requirements

*   **System:** A Linux machine with an NVIDIA GPU.
*   **GPU:** An NVIDIA GPU with CUDA capability is **required**. Compute capability 6.0 (Pascal architecture) or newer is highly recommended.
*   **Software:** NVIDIA CUDA Toolkit (version 11.0 or newer) and the corresponding drivers must be installed.
*   **Dependencies:** Requires the `NLopt` library for peak fitting.
*   **Input Files:** Requires the `Map.bin` and `nMap.bin` files generated by `DetectorMapper` to be in the same directory where the program is executed.
*   **Network:** Listens on a TCP socket (default port `60439`) for incoming data.

### Workflow & Logic

The integrator is designed as a multi-threaded pipeline to maximize throughput and hide latency.

1.  **Network Threads:** A dedicated thread listens for incoming network connections. For each connection, it spawns a new `handle_client` thread.
2.  **`handle_client` Thread:**
    *   This thread continuously reads data from its client socket.
    *   It expects a simple protocol: a 2-byte header (`uint16_t` dataset number) followed by a contiguous block of `int64_t` image data.
    *   Upon receiving a full frame, it allocates a **pinned memory buffer** (for fast CPU-GPU transfers) and places the data into a thread-safe processing queue.
3.  **Main Processing Thread:** This is the core GPU pipeline. It runs in a tight loop:
    *   **Pop from Queue:** It pulls the next available data chunk from the queue. This is a blocking operation; the thread sleeps if no data is ready. The time spent waiting here is profiled as `QPop`.
    *   **GPU Processing (`ProcessImageGPU`):** The `int64_t` raw data is copied from pinned host memory to GPU device memory. A sequence of CUDA kernels then performs image transformations (flips, transposes) and dark frame subtraction. The output is a `double` precision, fully corrected 2D image residing on the GPU.
    *   **GPU Integration (`integrate_*` kernel):** This kernel uses the pre-computed `Map.bin` and `nMap.bin` to perform the 2D -> 2D polar integration. It launches one thread per `[R, Eta]` bin and "gathers" the corresponding pixel intensities from the corrected image. The output is an intermediate 2D polar representation (`R` vs `Eta`) on the GPU.
    *   **GPU Reduction (`calculate_1D_profile_kernel`):** This highly-optimized kernel collapses the 2D polar data into the final 1D profile. It launches one CUDA block per `R` bin, and all threads in the block work together to compute the area-weighted average intensity over all `Eta` bins. It uses advanced features like warp-level shuffles and shared memory to achieve maximum speed.
    *   **Async Data Copy:** The final 1D profile is copied asynchronously from the GPU back to pinned host memory.
    *   **CPU Work:** While the GPU is processing the next frame, the CPU can work on the results of the previous one:
        *   Saving the 1D lineout data to `lineout.bin`.
        *   Optionally, performing multi-peak fitting on the 1D profile using the NLopt library.
        *   Saving fit results to `fit.bin`.

### How to Run

1.  **Prepare a parameter file** (e.g., `integrator.params`) describing the detector geometry and processing options. **This must match the parameters used to generate the maps.**
2.  **Ensure `Map.bin` and `nMap.bin` are present** in the working directory.
3.  **(Optional) Provide a dark frame** as a command-line argument. This should be a binary file of `int64_t` values.
4.  **Compile the code** using `nvcc`. An example compile command is provided in the header of the `.cu` source file. You will need to adjust paths to your CUDA and NLopt installations.
5.  **Run the integrator:**
    ```bash
    # Without dark frame subtraction
    ./bin/IntegratorFitPeaksGPUStream integrator.params

    # With dark frame subtraction
    ./bin/IntegratorFitPeaksGPUStream integrator.params my_dark_frame.raw
    ```
6.  The program will start listening on port `60439` and will wait for a data sender (like the provided Python script) to connect and stream data.

#### Key Output Files

*   `lineout.bin`: A binary file containing the 1D integrated profiles. Each profile consists of `[R_1, I_1, R_2, I_2, ...]` as `double` values.
*   `fit.bin`: (If enabled) A binary file containing the results of the peak fitting.
*   `Int2D.bin`: (If enabled) A single large binary file containing all the intermediate 2D polar integration patterns.
*   `fit_curves.bin`: (If enabled) A binary file containing the calculated fit curves for visualization.