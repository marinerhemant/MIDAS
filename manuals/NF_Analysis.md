# nf_MIDAS.py User Manual (Updated Version)

**Contact:** hsharma@anl.gov

---

## 1. Introduction

This Python script is the primary, high-performance driver for conducting Near-Field High-Energy Diffraction Microscopy (NF-HEDM) analysis using the MIDAS software suite. It orchestrates a series of computationally intensive binaries to reconstruct a 3D microstructure map from a series of 2D diffraction patterns taken at multiple sample-to-detector distances.

This updated version has been significantly refactored for improved robustness, usability, and error handling. It retains the core high-performance features—parallel execution with Parsl and use of shared memory (`/dev/shm`)—while adding features like automatic retry logic, dynamic path detection, and a more resilient cleanup process.

The script supports two primary modes of operation:
1.  **Microstructure Reconstruction:** The main mode, which solves for the crystal orientation at every point in a 3D grid.
2.  **Parameter Refinement:** An advanced mode used to refine the experimental geometry parameters (e.g., detector distance, tilts) by fitting the data from one or more known grain locations.

### Key Features
*   **End-to-End NF-HEDM Workflow:** Automates all steps from raw image processing to the final reconstructed microstructure.
*   **Dynamic Path Detection:** Automatically locates the MIDAS installation directory, eliminating the need for hardcoded paths.
*   **Robust and Resilient:**
    *   Features automatic retries with exponential backoff for parallel tasks, handling transient cluster issues.
    *   Includes a robust cleanup mechanism that runs even if the script fails mid-workflow.
    *   Provides clearer, more detailed error messages.
*   **Performance Optimized:** Uses Parsl for parallel execution and shared memory for high-throughput data access during fitting.
*   **Far-Field Seeding:** Can use a `Grains.csv` file from an FF-HEDM analysis to create a list of seed orientations, drastically improving efficiency.
*   **Advanced Refinement Modes:** Supports both interactive single-point and file-based multi-point parameter refinement.

---

## 2. Prerequisites & Configuration

1.  **MIDAS Installation:** The script and its associated binaries must be part of a standard MIDAS installation. The script will automatically find the installation directory.
2.  **Python Environment:** A Python environment with `parsl` and `numpy` installed.
3.  **Shared Memory (`/dev/shm`):** The compute nodes must have a sufficiently large and writable shared memory partition. The script now performs a more specific check for conflicting files created by this exact workflow.
4.  **Input Files:**
    *   A **Parameter File** (`-paramFN`) that defines the entire experiment.
    *   **Raw Data Files:** The script expects raw data to be in TIFF format, located in the `DataDirectory` specified in the parameter file.
    *   **Far-Field Results (Optional):** If using the FF-seeding workflow (`-ffSeedOrientations 1`), a `Grains.csv` file from a prior FF-HEDM analysis is required.

---

## 3. Workflow Overview

The script's execution is modularized into several distinct stages. The exact path depends on the `-refineParameters` flag.

### Stage 1: Pre-processing (Simulation)
This stage prepares all the simulated data (templates) needed for the analysis.
1.  **HKL Generation:** Calls `GetHKLListNF` to calculate the theoretical Bragg reflections.
2.  **Seed Orientations:** If `-ffSeedOrientations 1`, it calls `GenSeedOrientationsFF2NFHEDM` to convert orientations from an FF `Grains.csv` file into a list of seeds. The script now automatically updates the parameter file with the number of orientations.
3.  **Reconstruction Grid:** Calls `MakeHexGrid` to create the 3D grid of points (voxels) for the analysis.
4.  **Grid Filtering (Optional):** If a tomogram or `GridMask` is provided in the parameter file, the reconstruction grid is filtered to reduce the number of points, saving significant computation time.
5.  **Simulated Spots:** Calls `MakeDiffrSpots` to generate a complete library of simulated diffraction spots for every seed orientation at every point on the grid.

### Stage 2: Experimental Image Processing
This stage, controlled by `-doImageProcessing`, prepares the experimental data.
1.  **Median Calculation:** Calls `MedianImageLibTiff` in parallel for each detector distance to create a median-filtered image for background subtraction. This step now uses `multiprocessing.Pool` for `local` execution for better performance.
2.  **Image Processing:** Calls `ImageProcessingLibTiffOMP` in parallel to perform background subtraction and other corrections on the raw TIFF images.

### Stage 3: Fitting and Post-processing
This is the core reconstruction phase.
1.  **Memory Mapping:** Calls `MMapImageInfo` to prepare the data and then copies the critical binary files (`DiffractionSpots.bin`, `SpotsInfo.bin`, etc.) to shared memory (`/dev/shm`).
2.  **Analysis Path Selection:** The workflow diverges based on the `-refineParameters` flag.

    #### Workflow A: Microstructure Reconstruction (`-refineParameters 0`)
    1.  **Orientation Fitting:** Calls `FitOrientationOMP` in parallel across all nodes. This binary iterates through every point in the grid, comparing experimental data to the simulated templates in shared memory to find the best-fitting orientation.
    2.  **Result Parsing:** Calls `ParseMic` to consolidate the results into a single `Grains.mic` file.

    #### Workflow B: Parameter Refinement (`-refineParameters 1`)
    This mode uses known grain locations to optimize the experimental geometry.
    1.  **Select Grid Points:**
        *   If `-multiGridPoints 0`, the script will **prompt you to enter the (x,y) coordinates** of a single, known grain. It finds the closest grid point to use for refinement.
        *   If `-multiGridPoints 1`, it uses a list of pre-defined points from the parameter file.
    2.  **Parameter Fitting:** It calls `FitOrientationParameters` or `FitOrientationParametersMultiPoint`. These binaries perform a least-squares optimization to find the geometric parameters that best explain the data at the specified location(s). The output is a refined parameter set printed to a log file.

### Final Stage: Cleanup
A `finally` block in the main function ensures that shared memory files are removed and Parsl is shut down cleanly, even if an error occurs during the workflow.

---

## 4. Command-Line Arguments

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-paramFN` | **(Required)** The main parameter file defining the experiment. | `''` | `-paramFN NF_exp.txt` |
| `-machineName`| Machine configuration to use. Options are `local`, `orthrosnew`, `orthrosall`, `umich`, `marquette`, `purdue`.| `local` | `-machineName purdue` |
| `-nNodes` | Number of compute nodes to use for the analysis. | `1` | `-nNodes 10` |
| `-nCPUs` | Number of CPU cores to use per task on each node. | `10` | `-nCPUs 128` |
| `-ffSeedOrientations` | `1`: Use orientations from an FF `Grains.csv` file as seeds. `0`: Use the default seed list. | `0` | `-ffSeedOrientations 1` |
| `-doImageProcessing` | `1`: Run the median and image processing steps. `0`: Skip these steps (if already done). | `1` | `-doImageProcessing 0` |
| `-refineParameters` | `1`: Run in parameter refinement mode. `0`: Run in standard microstructure reconstruction mode. | `0` | `-refineParameters 1` |
| `-multiGridPoints`| **Only used if `-refineParameters 1`**. `1`: Use multiple points from the parameter file for refinement. `0`: Prompt for a single (x,y) point. | `0` | `-multiGridPoints 1`|

---

## 5. Execution Examples

### Example 1: Standard Microstructure Reconstruction
Submit an NF-HEDM analysis to the `Purdue` cluster using 4 nodes, seeding the reconstruction with a far-field `Grains.csv` file.

```bash
python nf_MIDAS.py \
    -paramFN /path/to/my_nf_params.txt \
    -machineName purdue \
    -nNodes 4 \
    -nCPUs 128 \
    -ffSeedOrientations 1 \
    -refineParameters 0
```

### Example 2: Interactive Parameter Refinement
Run the script in parameter refinement mode on a local machine. The script will pause and ask you to input the coordinates of the grain you want to use for the fit.

```bash
python nf_MIDAS.py \
    -paramFN /path/to/my_nf_params.txt \
    -machineName local \
    -nCPUs 8 \
    -refineParameters 1 \
    -multiGridPoints 0

# The script will then print:
# "Enter the x,y coordinates to optimize (e.g., 1.2,3.4): "
# You would then type your coordinates, e.g., "150.5,210.2" and press Enter.
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
├── Grains.mic              # **FINAL RECONSTRUCTION OUTPUT** (only in reconstruction mode)
└── ...                     # Other intermediate and configuration files
```

### Key Output Files

-   **`Grains.mic`**: (Reconstruction Mode Only) This is the final output of the entire workflow. It is a text file containing the position, orientation, and other metrics for each successfully reconstructed voxel in the grid.
-   **`midas_log/`**: This directory is essential for debugging. Each step of the workflow logs its standard output and standard error to a file here. If a step fails, the corresponding `..._err.csv` file will contain the detailed error message from the MIDAS C++ backend.
-   **`midas_log/fit_..._out.csv`**: (Parameter Refinement Mode Only) The refined geometric parameters will be printed to these files.

---

## 7. Troubleshooting

-   **Automatic Retries:** If you see log messages about a Parsl app failing and retrying, this is the new resiliency feature at work. The workflow will only fail if all retries are exhausted.
-   **Shared Memory (`/dev/shm`) Errors:**
    *   **Permission Denied:** The script may fail if it cannot write to `/dev/shm`. Check permissions on the compute nodes.
    *   **No space left on device:** The simulated spot libraries can be very large. Ensure the `/dev/shm` partition on the nodes is large enough.
    *   **User Conflict:** The script now performs a more targeted check for the specific files it creates (`SpotsInfo.bin`, etc.). If a conflict with another user is detected, you must wait for their job to finish or ask an administrator to clear the files.
-   **`FitOrientationOMP` Fails:** This is the most complex step in reconstruction mode. Failures can be due to:
    *   Mismatches between the simulated templates and experimental data (check your geometry in the parameter file).
    *   Insufficient signal in the processed images.
    *   Poor FF-seeding.
    Check the `fit*_err.csv` logs for specific error codes from the binary.