# NF-HEDM Analysis Script User Manual

**Contact:** hsharma@anl.gov

---

## 1. Introduction

This Python script is the primary driver for conducting Near-Field High-Energy Diffraction Microscopy (NF-HEDM) analysis using the MIDAS software suite. It orchestrates a series of computationally intensive binaries to reconstruct a 3D microstructure map from a series of 2D diffraction patterns taken at multiple sample-to-detector distances.

The script is heavily optimized for performance on HPC clusters. It makes extensive use of the Parsl parallel programming library to distribute tasks and leverages **shared memory (`/dev/shm`)** to dramatically accelerate the orientation fitting process by avoiding slow disk I/O.

A key feature of this workflow is its ability to use results from a Far-Field (FF) HEDM experiment as a starting point, or "seed," for the NF reconstruction. This can significantly reduce the search space and computational cost, making it a powerful tool for high-resolution characterization.

### Key Features
*   **End-to-End NF-HEDM Workflow:** Automates all steps from raw image processing to the final reconstructed microstructure.
*   **Performance Optimized:** Uses Parsl for parallel execution and shared memory for high-throughput data access during fitting.
*   **Far-Field Seeding:** Can use a `Grains.csv` file from an FF-HEDM analysis to create a list of seed orientations, drastically improving efficiency.
*   **Reconstruction Space Filtering:** Allows the user to limit the 3D reconstruction volume using either a prior tomographic reconstruction or a simple coordinate-based mask.
*   **Robust Pre-processing:** Includes parallel steps for median-filter-based background subtraction and image preparation.

---

## 2. Prerequisites & Configuration

1.  **MIDAS Installation:** The script assumes a **hardcoded installation path** for MIDAS at `~/opt/MIDAS/`. All binaries and utility scripts are expected to be in their standard subdirectories (`NF_HEDM/bin/`, `utils/`, etc.). This path must be adjusted in the script if your installation is located elsewhere.
2.  **Python Environment:** A Python environment with `parsl` and `numpy` installed.
3.  **Shared Memory (`/dev/shm`):** The compute nodes must have a sufficiently large shared memory partition (`/dev/shm`). The script copies several large binary files here for fast access. The script includes a check to prevent conflicts if multiple users are running jobs on the same node.
4.  **Input Files:**
    *   A **Parameter File** (`-paramFN`) that defines the entire experiment.
    *   **Raw Data Files:** The script expects raw data to be in TIFF format, located in the `DataDirectory` specified in the parameter file.
    *   **Far-Field Results (Optional):** If using the FF-seeding workflow (`-ffSeedOrientations 1`), a `Grains.csv` file from a prior FF-HEDM analysis is required.

---

## 3. Workflow Overview

The script follows a sequential, multi-stage process to perform the NF-HEDM reconstruction.

1.  **Setup and Configuration:**
    *   Parses command-line arguments and the main parameter file.
    *   Loads the appropriate Parsl configuration for the specified `-machineName`.
    *   Sets up the output and log directories inside the `DataDirectory`.

2.  **Template Generation (Simulation):**
    *   **HKL Generation:** Calls `GetHKLListNF` to calculate the theoretical Bragg reflections for the NF geometry.
    *   **Seed Orientations:** If `-ffSeedOrientations 1`, it calls `GenSeedOrientationsFF2NFHEDM` to convert the grain orientations from the FF `Grains.csv` file into a list of seeds for the NF analysis.
    *   **Reconstruction Grid:** Calls `MakeHexGrid` to create the 3D grid of points (voxels) where the reconstruction will be performed.
    *   **Grid Filtering (Optional):** If a tomogram or grid mask is provided in the parameter file, the reconstruction grid is filtered to reduce the number of points, saving significant computation time.
    *   **Simulated Spots:** Calls `MakeDiffrSpots` to generate a complete library of simulated diffraction spots for every seed orientation at every point on the reconstruction grid. This creates the templates that will be matched against the experimental data.

3.  **Experimental Image Processing (Optional):**
    *   This stage is controlled by the `-doImageProcessing` flag and can be skipped if previously completed.
    *   **Median Calculation:** Calls `MedianImageLibTiff` in parallel for each detector distance to create a median-filtered image for background subtraction.
    *   **Image Processing:** Calls `ImageProcessingLibTiffOMP` in parallel across all nodes to perform background subtraction and other corrections on the raw TIFF images.

4.  **Reconstruction (Fitting):**
    *   **Memory Mapping:** Calls `MMapImageInfo` to load the massive simulated spot library and processed experimental data into shared memory (`/dev/shm`). This is the most critical performance optimization.
    *   **Orientation Fitting:** Calls `FitOrientationOMP` in parallel across all nodes. This binary iterates through every point in the reconstruction grid, compares the experimental data at that location to all the simulated templates in shared memory, and finds the best-fitting orientation.
    *   **Result Parsing:** Calls `ParseMic` to consolidate the results from the fitting step into a single, human-readable `.mic` file.

5.  **Cleanup:**
    *   The script removes the large binary files from `/dev/shm` to free up resources on the compute node.

---

## 4. Command-Line Arguments

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-paramFN` | **(Required)** The main parameter file defining the experiment. | `''` | `-paramFN NF_experiment.txt` |
| `-machineName` | Execution environment. Options: `local`, `orthrosnew`, `orthrosall`, `umich`, `marquette`, `purdue`. | `local` | `-machineName purdue` |
| `-nNodes` | Number of compute nodes to use for the analysis. | `1` | `-nNodes 10` |
| `-nCPUs` | Number of CPU cores to use per task on each node. | `10` | `-nCPUs 128` |
| `-ffSeedOrientations` | `1`: Use orientations from an FF `Grains.csv` file as seeds. `0`: Use the default seed list. | `0` | `-ffSeedOrientations 1` |
| `-doImageProcessing` | `1`: Run the median and image processing steps. `0`: Skip these steps (if already done). | `1` | `-doImageProcessing 0` |

---

## 5. Execution Example

Submit an NF-HEDM analysis to the `Purdue` cluster using 4 nodes. The analysis will use a `Grains.csv` file from a previous far-field scan to seed the reconstruction.

```bash
python /path/to/nf_hedm_script.py \
    -paramFN /path/to/my_nf_params.txt \
    -machineName purdue \
    -nNodes 4 \
    -nCPUs 128 \
    -ffSeedOrientations 1
```

---

## 6. Output Directory Structure

All output is generated within the `DataDirectory` specified in the parameter file.

```
<DataDirectory>/
├── midas_log/              # Contains stdout/stderr logs for every binary executed
│   ├── median1_out.csv
│   ├── image0_err.csv
│   ├── fit0_out.csv
│   └── ...
├── grid.txt                # The final (potentially filtered) 3D reconstruction grid
├── SeedOrientations.txt    # The list of seed orientations used for the analysis
├── DiffractionSpots.bin    # The large binary file of simulated diffraction spots
├── SpotsInfo.bin           # The large binary file of processed experimental image data
├── Grains.mic              # **FINAL RECONSTRUCTION OUTPUT** in .mic format
└── ...                     # Other intermediate and configuration files
```

### Key Output Files

-   **`Grains.mic`**: This is the final output of the entire workflow. It is a text file containing the position, orientation, and other metrics for each successfully reconstructed voxel in the grid.
-   **`midas_log/`**: This directory is essential for debugging. Each step of the workflow logs its standard output and standard error to a file here. If a step fails, the corresponding `..._err.csv` file will contain the detailed error message from the MIDAS C++ backend.

---

## 7. Troubleshooting

-   **Hardcoded Paths:** The most likely initial error is a `FileNotFoundError` due to the hardcoded `~/opt/MIDAS/` paths in the script. Ensure these are updated to match your installation location.
-   **Shared Memory (`/dev/shm`) Errors:**
    *   **Permission Denied:** The script may fail if it cannot write to `/dev/shm`. Check permissions on the compute nodes.
    *   **No space left on device:** The simulated spot libraries can be very large. Ensure the `/dev/shm` partition on the nodes is large enough to hold `SpotsInfo.bin`, `DiffractionSpots.bin`, etc.
    *   **User Conflict:** The script attempts to detect `.bin` files from other users. If it reports a conflict, you must wait for the other user's job to finish and clean up, or ask a system administrator to clear the files.
-   **`FitOrientationOMP` Fails:** This is the most complex step. Failures can be due to:
    *   Mismatches between the simulated templates and experimental data (check your geometry in the parameter file).
    *   Insufficient signal in the processed images (check the output of the image processing stage).
    *   Errors in the FF-seeding (e.g., the FF reconstruction does not represent the NF sample well).
    Check the `fit*_err.csv` logs for specific error codes from the binary.