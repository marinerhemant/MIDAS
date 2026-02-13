# MIDAS_RADINT: High-Performance Azimuthal Integration Suite

This document describes a suite of two programs designed for high-throughput, real-time azimuthal integration of 2D detector data: `DetectorMapper` and `IntegratorFitPeaksGPUStream`.

## Overview

The integration process is split into two stages for maximum performance:

1.  **Offline Pre-computation (`DetectorMapper`):** A CPU-based program that runs **once** to analyze the experimental geometry. It creates a highly optimized lookup table that maps every detector pixel to its corresponding location in a polar coordinate system (`R`, `η`). This is the key to the system's speed.
2.  **Real-time Integration (`IntegratorFitPeaksGPUStream`):** A GPU-accelerated program that runs continuously. It listens for 2D detector data on a network socket, uses the pre-computed maps to perform the integration in microseconds, and optionally performs peak fitting on the resulting 1D profile.

This two-stage approach moves the most computationally expensive geometric calculations offline, allowing the GPU to focus solely on fast data processing and arithmetic during an experiment.

**THE BEST WAY TO RUN IS USING THE utils/integrator_batch_process.py script, which will start the appropriate codes in the right workflow order.**

---

# utils/integrator_batch_process.py User Manual

---

## 1. Introduction

`integrator_batch_process.py` is the master orchestration script for the real-time, GPU-accelerated diffraction integration pipeline. It is not a standalone processing tool, but rather a controller that manages a series of specialized components to create a complete, automated workflow.

The primary goal of this pipeline is to take raw 2D detector data, either from a folder of image files or a live PVA stream, and process it at high speed into reduced 1D lineouts, fitted peak parameters, and a final, consolidated HDF5 file.

This script automates the following sequence:
1.  **Pre-flight Check:** Automatically runs `DetectorMapper` to generate necessary pixel mapping files (`Map.bin`, `nMap.bin`) if they are not found.
2.  **Start Backend:** Launches the `IntegratorFitPeaksGPUStream` C++/CUDA application, which starts a server and waits for data.
3.  **Start Data Feeder:** Runs the `integrator_server.py` script, which finds the data (files or stream) and sends it frame by frame to the GPU backend.
4.  **Monitor Progress:** Actively monitors the processing, providing real-time feedback on the number of frames completed, processing rate, and estimated time remaining.
5.  **Shutdown:** Gracefully terminates the server and backend processes once processing is complete or if the user aborts.
6.  **Post-process:** Calls `integrator_stream_process_h5.py` to convert the raw binary output files (`lineout.bin`, `fit.bin`, etc.) into a single, well-structured HDF5 file for analysis.

---

## 2. Prerequisites

1.  **MIDAS Installation:** All components (`DetectorMapper`, `IntegratorFitPeaksGPUStream`, and the Python scripts) must be correctly compiled and located in their standard MIDAS directory structure (e.g., `~/opt/MIDAS/`).
2.  **Python Environment:** A Python environment with the `psutil` library installed (`pip install psutil`).
3.  **Input Data:**
    *   A **Parameter File** that configures the `IntegratorFitPeaksGPUStream` backend.
    *   Either a directory of image files or a running PVA stream.

---

## 3. Parameters Guide (Command-Line Arguments)

The entire pipeline is configured through the command-line arguments passed to `integrator_batch_process.py`.

### **Required Arguments**

These arguments are essential and must be provided for the script to run.

`--param-file <path>`
:   **Function:** Specifies the path to the main parameter file.
:   **Details:** This is the most crucial file, as it contains all the geometric, integration, and peak-fitting parameters that will be used by the `IntegratorFitPeaksGPUStream` backend. This script reads it to determine frame size and other metadata, and passes it directly to the backend.

A data source must be specified using one of the following two mutually exclusive options:

`--folder <path>`
:   **Function:** Specifies the directory containing the raw image files to be processed.
:   **Details:** The script will scan this folder for files matching the `--extension` and feed them one by one to the processing backend.

`--pva`
:   **Function:** Instructs the script to get data from a live PVA (Portable Virtual Application) stream instead of from files.
:   **Details:** This is used for real-time, on-the-fly processing during an experiment. The specific PVA server and channel are configured with the `--pva-ip` and `--pva-channel` options.

### **Optional Arguments**

These arguments allow you to customize the pipeline's behavior.

`--extension <ext>`
:   **Function:** Sets the file extension for the image files to be processed when using `--folder`.
:   **Default:** `tif`
:   **Details:** Use this if your image files have a different extension, for example, `h5` or `geX`.

`--dark <path>`
:   **Function:** Specifies the path to a dark-field image or a binary file containing averaged dark frames.
:   **Details:** This file is passed directly to the `IntegratorFitPeaksGPUStream` backend, which will use it for background subtraction during the GPU processing pipeline. If omitted, no dark correction is performed.

`--output-h5 <filename>`
:   **Function:** Sets the name for the final, consolidated HDF5 output file.
:   **Default:** `integrator_output.h5`
:   **Details:** This is the filename that will be passed to the `integrator_stream_process_h5.py` script at the end of the workflow.

`--mapping-file <filename>`
:   **Function:** Sets the name for the JSON file that maps frame numbers to their original filenames or timestamps.
:   **Default:** `frame_mapping.json`
:   **Details:** This file is crucial for correlating the processed data back to the source. The `integrator_server.py` script generates it during processing.

`--save-interval <integer>`
:   **Function:** Controls how often the mapping file is saved to disk.
:   **Default:** `10`
:   **Details:** The mapping file will be updated every N frames. A smaller number provides more frequent updates for progress monitoring but results in slightly more disk I/O.

`--h5-location <path>`
:   **Function:** Specifies the internal path within an HDF5 file where the image data is located.
:   **Default:** `exchange/data`
:   **Details:** This is only used when processing HDF5 files (i.e., `--extension h5`). It tells the server where to find the 2D image dataset inside each HDF5 file.

#### **PVA-Specific Arguments**
These are only used when the `--pva` flag is active.

`--pva-ip <ip_address>`
:   **Function:** The IP address of the PVA server.
:   **Default:** `10.54.105.139`

`--pva-channel <string>`
:   **Function:** The name of the PVA channel that is broadcasting the image data.
:   **Default:** `16pil-idb:Pva1:Image`

---

## 4. Execution Examples

### Example 1: Processing a Folder of TIFF Files
Process all `.tif` files in the `/data/my_scan_01` directory, using a dark file for correction.

```bash
python integrator_control.py \
    --param-file /params/setup_30keV.txt \
    --folder /data/my_scan_01 \
    --dark /data/darks/dark_avg.bin \
    --output-h5 my_scan_01_processed.h5
```

### Example 2: Processing a Live PVA Stream
Connect to a PVA stream for real-time processing and save the output to a different HDF5 file.

```bash
python integrator_control.py \
    --param-file /params/realtime_setup.txt \
    --pva \
    --pva-ip 10.54.105.150 \
    --pva-channel "MyDetector:PvaStream" \
    --output-h5 live_output.h5
```

### Example 3: Processing HDF5 Files
Process a folder of HDF5 files where the data is located at a custom internal path.

```bash
python integrator_control.py \
    --param-file /params/h5_setup.txt \
    --folder /data/hdf_scan \
    --extension h5 \
    --h5-location /entry/instrument/detector/data
```

---

## 5. Output Files

The script coordinates the creation of several files, but the most important one is the final HDF5 file.

*   **`<output-h5>` (e.g., `integrator_output.h5`):** The primary output. A structured HDF5 file containing all the processed data, including 1D lineouts, peak fit results, and metadata, generated by `integrator_stream_process_h5.py`.
*   **`lineout.bin`, `fit.bin`, etc.:** Intermediate raw binary files created by `IntegratorFitPeaksGPUStream`. These are consumed by the HDF5 conversion script.
*   **`<mapping-file>` (e.g., `frame_mapping.json`):** A JSON file linking each processed frame index to its source file or timestamp.
*   **`integrator.log`:** A log file containing the stdout and stderr from the `IntegratorFitPeaksGPUStream` backend. **Check this file first for debugging GPU or core processing errors.**
*   **`server.log`:** A log file from the `integrator_server.py` script. **Check this file for debugging data input errors (e.g., file not found, PVA connection issues).**
*   **`detector_mapper.log`:** (If run) The log file from the `DetectorMapper` pre-flight check.

---

## Detailed workflow

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

### 2.1. Introduction

`IntegratorFitPeaksGPUStream` is a high-performance, GPU-accelerated C++/CUDA application designed for real-time processing of 2D detector data. Its primary function is to receive a continuous stream of detector frames over a network socket, perform a series of transformations and integrations to reduce the 2D data to a 1D profile (a lineout), and optionally perform advanced peak fitting on the resulting profile.

This tool is built for speed and is intended for use in experimental environments like synchrotron beamlines, where low-latency feedback is crucial. It leverages a multi-threaded architecture for network communication, a multi-stage GPU pipeline for all heavy computation, and the NLopt library for robust, non-linear peak fitting.

#### Key Features
*   **Real-Time Socket Streaming:** Receives detector frames over a TCP/IP socket in a dedicated, multi-threaded server, allowing for live, on-the-fly processing.
*   **High-Performance GPU Pipeline:** All major computational steps are performed on the GPU:
    *   Image transformations (flips, transpose).
    *   Dark frame subtraction.
    *   2D-to-1D integration using pre-calculated pixel maps.
    *   Area-weighted or simple-mean 1D profile calculation.
*   **Advanced Peak Fitting:** Integrates the NLopt library to perform robust, non-linear least-squares fitting of Pseudo-Voigt peaks on the 1D profile, complete with a global background parameter.
*   **Memory Optimized:** Uses memory-mapped files (`mmap`) for efficient access to large pixel maps (`Map.bin`, `nMap.bin`) and pinned host memory for high-speed data transfers between the CPU and GPU.
*   **Flexible & Configurable:** Behavior is controlled by a comprehensive text-based parameter file, allowing users to fine-tune every aspect of the processing chain without recompiling.
*   **Rich Data Output:** Saves results in efficient binary formats, including 1D lineouts, detailed peak fit parameters, and optionally the full 2D integrated patterns or fitted curves for each frame.

---

### 2.2. Prerequisites & Compilation

#### Prerequisites
1.  **Hardware:** A server with a CUDA-capable NVIDIA GPU.
2.  **Dependencies:**
    *   NVIDIA CUDA Toolkit (compiler, runtime libraries).
    *   NLopt library (for peak fitting).
    *   Zlib, Pthread, and standard C++ build tools.
3.  **Input Files:**
    *   A **Parameter File** (text format).
    *   `Map.bin`: A binary file mapping each pixel on the 2D detector to its corresponding (R, Eta) bin.
    *   `nMap.bin`: A binary file containing the number of detector pixels contributing to each (R, Eta) bin.
    *   An optional **Dark Frame File** (binary format).

#### Compilation
The application must be compiled from source. A typical compile command is provided in the source file header. It requires linking against the CUDA, NLopt, and other system libraries.

**Example Compile Command:**
```bash
/path/to/cuda/bin/nvcc IntegratorFitPeaksGPUStream.cu \
  -o IntegratorFitPeaksGPUStream \
  -gencode=arch=compute_86,code=sm_86 \
  -Xcompiler -fopenmp \
  -I/path/to/nlopt/include \
  -L/path/to/nlopt/lib \
  -O3 -lnlopt -lz -ldl -lm -lpthread \
  -Xlinker "-rpath=/path/to/nlopt/lib"
```
*(Note: Adjust paths and GPU architecture flags (`-gencode`) for your specific system.)*

---

### 2.3. Workflow Overview

1.  **Initialization:**
    *   The application starts, initializes the GPU, and reads all settings from the specified parameter file.
    *   It memory-maps the `Map.bin` and `nMap.bin` files for fast access.
    *   It reads, transforms, and averages the dark frame file (if provided) and copies it to the GPU.
    *   It pre-calculates the area of each (R, Eta) bin on the GPU, accounting for any pixel masks.
    *   It starts a multi-threaded TCP server, listening for incoming data on port `60439`.

2.  **Data Streaming Loop:**
    *   A dedicated thread accepts client connections. For each connection, a new thread is spawned to handle data reception.
    *   The reception thread reads detector frames from the socket, places them into a thread-safe buffer queue, and immediately goes back to listening for the next frame.
    *   The main thread continuously pulls frames from this queue for processing.

3.  **GPU Processing Pipeline (per frame):**
    *   The raw frame is copied from CPU (pinned memory) to GPU.
    *   A sequence of GPU kernels performs image transformations (flips, etc.) and dark frame subtraction.
    *   A highly parallel GPU kernel uses the `Map.bin` data to integrate the 2D image into a 2D (R vs. Eta) array of intensities.
    *   A final set of GPU reduction kernels calculates the final 1D profile (Intensity vs. R) by averaging over all Eta bins. Two versions are calculated: an area-weighted average and a simple mean.

4.  **Peak Fitting (Optional):**
    *   The 1D profile is copied from the GPU back to the CPU.
    *   If peak fitting is enabled, the script identifies peak candidates based on specified locations or an auto-detection algorithm.
    *   It defines Regions of Interest (ROIs) around these peaks.
    *   Using OpenMP for parallelization, it runs the NLopt optimizer on each ROI to fit the peak parameters (Amplitude, Center, Width, Mixing) and a shared background.

5.  **Output:**
    *   The results (1D profiles, fit parameters, etc.) are written to binary output files (`lineout.bin`, `fit.bin`). This is done for every single frame received.
    *   The application continues this loop until it receives a shutdown signal (Ctrl+C).

---

### 2.4. Parameters Guide

The behavior of the application is controlled entirely by its command-line arguments and the keywords within the parameter file.

#### 2.4.1. Command-Line Arguments

The application takes one required and one optional command-line argument:

| Argument | Description | Required | Example |
| :--- | :--- | :--- | :--- |
| `ParamFN` | The path to the main parameter file that configures the entire workflow. | **Yes** | `./IntegratorFitPeaksGPUStream my_params.txt` |
| `DarkAvgFN` | The path to a binary file containing one or more dark frames. If provided, dark subtraction is enabled. | No | `... my_params.txt dark_frames.bin` |

#### 2.4.2. Parameter File Keywords

This is the primary way to configure the application. The file is a simple text file with `key value` pairs.

##### **Integration Binning Parameters**
These define the R-Eta grid for 2D-to-1D integration.

| Keyword | Type | Description |
| :--- | :--- | :--- |
| `RMin` | double | The minimum radial value (in pixels) to begin the integration. |
| `RMax` | double | The maximum radial value (in pixels) to end the integration. |
| `RBinSize` | double | The width of each radial bin (in pixels). The number of R bins is `(RMax - RMin) / RBinSize`. |
| `EtaMin` | double | The minimum azimuthal angle (in degrees, typically -180 to 180) to begin integration. |
| `EtaMax` | double | The maximum azimuthal angle (in degrees) to end integration. |
| `EtaBinSize` | double | The width of each azimuthal bin (in degrees). |

##### **Detector & Geometry Parameters**

| Keyword | Type | Description |
| :--- | :--- | :--- |
| `NrPixelsY` | int | The number of pixels on the detector in the Y dimension (fastest changing axis). |
| `NrPixelsZ` | int | The number of pixels on the detector in the Z dimension. |
| `NrPixels` | int | A shortcut to set both `NrPixelsY` and `NrPixelsZ` to the same value for square detectors. |
| `Lsd` | double | The sample-to-detector distance (in micrometers). Used for calculating 2-theta values. |
| `px` | double | The physical size of a detector pixel (in micrometers). Used for 2-theta calculations. |

##### **Image Processing & Masking Parameters**

| Keyword | Type | Description |
| :--- | :--- | :--- |
| `ImTransOpt`| int | Defines an image transformation. Can be specified multiple times to create a sequence. **0**: No-op, **1**: Flip Horizontal, **2**: Flip Vertical, **3**: Transpose. |
| `Normalize` | int | **1**: (Default) Normalize the intensity in each bin by its total area. This corrects for geometric distortions. **0**: Use the raw sum of intensities. |
| `GapIntensity`| long long| If a `DarkAvgFN` is provided, any pixel in the dark frame with this exact integer value will be masked (ignored) during integration. |
| `BadPxIntensity`| long long| A second intensity value for masking pixels from the dark frame. |

##### **Peak Fitting Parameters**
These control the optional peak fitting stage. Setting `DoPeakFit` or `PeakLocation` enables this stage.

| Keyword | Type | Description |
| :--- | :--- | :--- |
| `DoPeakFit` | int | **1**: Enable the peak fitting workflow. **0**: (Default) Disable peak fitting. |
| `PeakLocation`| double | Specifies the approximate radial position (in pixels) of a peak to be fitted. Can be used multiple times for multiple peaks. Automatically enables `DoPeakFit` and `MultiplePeaks`. |
| `MultiplePeaks`| int | **1**: Find and fit multiple peaks. Implied if `PeakLocation` is used. **0**: (Default) Fit only the single most intense peak. |
| `DoSmoothing`| int | **1**: Apply a Savitzky-Golay smoothing filter to the 1D profile *before* peak finding (not recommended if fitting specific `PeakLocation`s). **0**: (Default) Do not smooth. |
| `FitROIPadding`| int | The number of **bins** to include on either side of a peak's center to define its Region of Interest (ROI) for fitting. Default is 20. |
| `FitROIAuto`| int | **1**: Dynamically determine the ROI padding based on an initial estimate of the peak's Full Width at Half Maximum (FWHM). Overrides `FitROIPadding`. **0**: (Default) Use the fixed `FitROIPadding` value. |

##### **Output Control Parameters**

| Keyword | Type | Description |
| :--- | :--- | :--- |
| `SumImages` | int | **1**: Accumulate the 2D integrated patterns from all frames into a single sum. **0**: (Default) Process each frame independently. |
| `Write2D` | int | **1**: Write the full 2D (R vs. Eta) integrated pattern for *every single frame* to `Int2D.bin`. **Warning: Creates very large files.** **0**: (Default) Do not write this data. |

---

### 2.5. Input & Output Files

#### Input Files
*   **Parameter File (e.g., `my_params.txt`):** The main configuration file, as described above.
*   **`Map.bin`:** A binary file of `struct data {int y; int z; double frac;}`. Describes which detector pixels `(y, z)` contribute to each (R, Eta) bin, and with what fractional area `frac`. Must be in the working directory.
*   **`nMap.bin`:** A binary file of `int`s. Contains the number of pixels and the starting offset into `Map.bin` for each (R, Eta) bin. Must be in the working directory.
*   **Dark Frame (e.g., `dark.bin`, optional):** A binary file containing one or more `int64_t` raw detector frames.

#### Output Files
All output files are binary and are created in the working directory.
*   **`lineout.bin`:** The primary output. A continuous stream of `[R, Intensity]` pairs (doubles) for the area-weighted 1D profile. Contains `nRBins * 2` doubles per frame.
*   **`lineout_simple_mean.bin`:** Same format as `lineout.bin`, but for the 1D profile calculated using a simple average instead of an area-weighted average.
*   **`fit.bin`:** (If fitting enabled) A stream of fit results. Contains `7` doubles per fitted peak per frame: `[Amplitude, Background, Mixing, Center, Sigma, GoodnessOfFit, IntegratedArea]`.
*   **`fit_curves.bin`:** (If fitting enabled) A stream of the fitted peak shapes. Useful for visualization and debugging the fit quality.
*   **`Int2D.bin`:** (If `Write2D 1`) A stream of the full 2D integrated patterns. Contains `nRBins * nEtaBins` doubles per frame.
*   **`RTthEtaAreaMap.bin`:** A static map file written once at the start. Contains a flattened `[R, TTh, Eta, Area]` array of doubles for every bin in the integration grid.

---

### 2.6. Execution Example

1.  Ensure `Map.bin`, `nMap.bin`, and `my_params.txt` are in your current directory.
2.  Have your data source ready to stream to `localhost:60439`.
3.  Run the application from the command line:

```bash
# Example without dark subtraction
./IntegratorFitPeaksGPUStream my_params.txt

# Example with dark subtraction
./IntegratorFitPeaksGPUStream my_params.txt /path/to/darks.bin
```
The application will start, print its configuration, and wait for a data stream. Press `Ctrl+C` to shut it down gracefully.

### How to Run

0. **Use the provided Python script** (`utils/integrator_batch_process.py`) to automate the entire pipeline, including running `DetectorMapper` if needed, and then starting the integrator. This is the recommended approach.
   
