# pf_MIDAS.py User Manual

**Contact:** hsharma@anl.gov

---

## 1. Introduction

`pf_MIDAS.py` is a comprehensive Python script for processing and analyzing scanning High-Energy Diffraction Microscopy (HEDM) data, often referred to as Point-Focus (PF) HEDM. It is designed to handle multiple far-field HEDM datasets collected at various positions across a sample, automating the entire workflow from raw data to a fully reconstructed 3D microstructure.

The script leverages the Parsl parallel programming library to efficiently distribute tasks, making it suitable for both multi-core workstations and large-scale HPC clusters. It orchestrates a complex, multi-stage analysis that includes parallel peak searching on individual scans followed by a combined indexing and refinement process on the aggregated data volume.

### Key Features
*   **Scanning HEDM Workflow:** Specifically designed to process a series of scans collected at different spatial positions.
*   **Two-Stage Parallel Processing:**
    1.  Processes each scan in parallel to find diffraction peaks.
    2.  Combines all peak data to perform a global indexing and refinement across the entire scanned volume.
*   **Flexible Analysis Modes:** Supports finding either a single best-fit orientation per voxel or multiple potential solutions.
*   **Tomographic Reconstruction:** Includes an optional, powerful feature to generate a 3D tomographic reconstruction of the grain map from the diffraction data (`doTomo`).
*   **Advanced Control:** Offers a rich set of command-line arguments to control every step of the workflow, from data conversion to intensity normalization and scan merging.
*   **Robust Logging and Error Handling:** Creates detailed logs for each step, facilitating monitoring and debugging of complex analysis runs.

---

## 2. Prerequisites

1.  **MIDAS Installation:** The script must be located within a functioning MIDAS installation, as it depends on binaries and utility scripts from the `FF_HEDM/bin/` and `utils/` directories.
2.  **Python Environment:** A Python environment with `parsl`, `numpy`, `pandas`, `scikit-image`, `Pillow`, `h5py`, `zarr`, and `numba` installed.
3.  **Input Files:**
    *   A **Parameter File** (`-paramFile`) containing the experimental geometry, material information, and processing settings.
    *   A **`positions.csv`** file located in the same directory as the parameter file. This is a crucial file that must contain a single column of y-motor positions for each scan. The script's help text notes: *"(negative positions with respect to actual motor position, motor position is normally position of the rotation axis, opposite to the voxel position)"*.

---

## 3. Workflow Overview

The script operates in a two-stage process:

### Stage 1: Parallel Peak Search (Per-Scan)
-   The script reads the list of scan positions from `positions.csv`.
-   For each scan, it launches a parallel task (`parallel_peaks`).
-   Each task performs a complete far-field analysis workflow on its assigned scan:
    1.  Creates a dedicated subdirectory for the scan.
    2.  Converts raw data to a `.MIDAS.zip` (Zarr) archive (if `-convertFiles=1`).
    3.  Generates an HKL list.
    4.  Performs peak searching (`PeaksFittingOMPZarrRefactor`).
    5.  Merges overlapping peaks.
    6.  Prepares data for indexing (`CalcRadiusAllZarr`, `FitSetupZarr`).
    7.  Outputs a consolidated peak list (`InputAllExtraInfoFittingAll<layerNr>.csv`).

### Stage 2: Combined Analysis (Full Volume)
-   After all peak search tasks are complete, the script transitions to the main directory.
-   It optionally merges data from adjacent scans if the `nMerges` parameter is set in the parameter file.
-   It bins all the peak data from all scans into a spatial grid (`SaveBinDataScanning`).
-   It runs a parallel indexing job (`IndexerScanningOMP`) on the entire binned dataset to find orientation candidates for each voxel.
-   Based on the `-oneSolPerVox` flag, it proceeds down one of two paths:

    #### Path A: Single Solution Per Voxel (`-oneSolPerVox 1`)
    1.  The script runs `findSingleSolutionPF` to identify the most likely grain orientation for each voxel from the candidates.
    2.  **Tomography (Optional, `-doTomo 1`):** If enabled, it generates sinograms from the intensity data of each unique grain orientation and performs an inverse Radon transform (`iradon`) to create a 3D tomographic reconstruction of the microstructure. The final output is a 3D grain map.
    3.  A final parallel refinement step (`FitOrStrainsScanningOMP`) is run on the selected unique solutions to determine the precise orientation, position, and strain tensor.

    #### Path B: Multiple Solutions Per Voxel (`-oneSolPerVox 0`)
    1.  The script runs `findMultipleSolutionsPF` to cluster and identify all plausible orientation solutions for each voxel.
    2.  A final refinement step is run on *all* found solutions. This mode is useful for analyzing regions with high ambiguity or complex microstructures.

---

## 4. Command-Line Arguments

The script's behavior is controlled via the following arguments:

### Basic Arguments

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-paramFile` | **(Required)** The main parameter file name. Must be in the current directory. | N/A | `-paramFile Ni_scan.txt` |
| `-resultDir` | Directory to save all results. Defaults to the current directory. | `''` | `-resultDir /path/to/analysis` |

### Execution Control

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-machineName` | Execution environment. Options: `local`, `orthrosall`, `orthrosnew`, `umich`, `marquette`. | `local` | `-machineName umich` |
| `-nNodes` | Number of compute nodes to request on a cluster. | `1` | `-nNodes 8` |
| `-nCPUs` | Number of CPU cores per task on a cluster node. | `32` | `-nCPUs 36` |
| `-nCPUsLocal`| Number of CPU cores for local helper tasks. | `4` | `-nCPUsLocal 8` |

### Workflow Control

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-doPeakSearch` | Controls Stage 1. `1`: Run peak search. `0`: Skip if already done. `-1`: Reprocess existing peak search output without re-running the search. | `1` | `-doPeakSearch 0` |
| `-runIndexing` | Controls Stage 2. `1`: Run binning and indexing. `0`: Skip Stage 2 (useful for stopping after peak search). | `1` | `-runIndexing 0` |
| `-oneSolPerVox` | `1`: Find a single best solution per voxel. `0`: Allow multiple solutions per voxel. | `1` | `-oneSolPerVox 0` |
| `-doTomo` | `1`: Perform tomographic reconstruction. **Requires `-oneSolPerVox 1`**. `0`: Do not perform tomography. | `1` | `-doTomo 0` |
| `-convertFiles` | `1`: Convert raw data to Zarr. `0`: Skip if Zarr files already exist. | `1` | `-convertFiles 0` |
| `-startScanNr` | The scan number to start processing from. Useful for resuming a failed run. | `1` | `-startScanNr 15` |

### Advanced & Data Handling Arguments

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-omegaFile` | Path to a file containing omega angle offsets to override values in the parameter file. | `''` | `-omegaFile omega_offsets.txt` |
| `-numFrameChunks`| Splits large datasets into chunks during conversion to save RAM. `-1` disables chunking. | `-1` | `-numFrameChunks 4` |
| `-preProcThresh`| Saves dark-corrected/thresholded data during conversion. `-1` disables. `0` just subtracts dark. | `-1` | `-preProcThresh 50` |
| `-normalizeIntensities` | Controls intensity use in tomography. `0`: Use equivalent grain radius. `1`: Normalize by powder ring intensity. `2`: Use raw integrated intensity. | `2` | `-normalizeIntensities 1` |
| `-minThresh` | Filters out peaks with a maximum intensity below this value during post-processing of peak search results. `-1` disables. | `-1` | `-minThresh 150` |

---

## 5. Execution Example

Submit an analysis to a `umich` cluster, requesting 4 nodes. The job will skip the file conversion and peak search steps (assuming they were done previously) and proceed directly to indexing and tomographic reconstruction.

```bash
python /path/to/pf_MIDAS.py \
    -paramFile Inconel_params.txt \
    -resultDir /scratch/user/Inconel_Tomo_Run \
    -machineName umich \
    -nNodes 4 \
    -convertFiles 0 \
    -doPeakSearch 0 \
    -runIndexing 1 \
    -oneSolPerVox 1 \
    -doTomo 1
```

---

## 6. Output Directory Structure

The script generates a highly structured output in the `-resultDir`.

```
<resultDir>/
├── 100000/                   # Subdirectory for the first scan (named by StartFileNr)
│   ├── output/               # Logs for the peak search on this scan
│   ├── paramstest.txt
│   └── ...
├── 100360/                   # Subdirectory for the second scan
│   └── ...
├── Recons/                   # **PRIMARY OUTPUT** for Tomography
│   ├── microstrFull.csv      # Detailed refinement results for all found grains
│   ├── microstructure.hdf    # HDF5 file with results and reconstructed image data
│   ├── Full_recon_max_project.tif # 2D max projection of the 3D reconstruction
│   └── recon_grNr_0001.tif   # Reconstructed slice for each individual grain
├── Sinos/                    # Sinograms for each grain (if doTomo=1)
├── Thetas/                   # Omega angles for each sinogram (if doTomo=1)
├── Output/                   # Intermediate indexing results for the full volume
├── Results/                  # Final refinement results (CSV file per grain)
├── output/                   # Logs for the combined analysis stage (indexing, etc.)
├── Grains.csv                # Final grain list
├── UniqueOrientations.csv    # List of unique orientations found
├── processing.log            # Main log file for the script's execution
└── paramstest.txt            # The master parameter file used for the combined analysis
```

### Key Output Files

-   **`Recons/microstructure.hdf`**: The primary scientific output when using tomography. It contains the refined grain data and a 23-channel image stack with properties like quaternion, position, strain, and completeness for every pixel.
-   **`Recons/microstrFull.csv`**: A human-readable CSV containing the full refined parameters for every successfully indexed grain.
-   **`microstrFull.csv`** (in root): The main output file when `-oneSolPerVox=0`.
-   **`processing.log`**: The main log file. Check this first for high-level progress and errors.
-   **`output/` directories**: Contain detailed stdout/stderr logs from the MIDAS binaries. Essential for debugging low-level failures.

---

## 7. Troubleshooting

-   **`positions.csv` not found:** Ensure the file exists in the same directory where you are running the script and contains the correct number of y-positions.
-   **Peak Search Fails:** Check the logs in the subdirectory for the failing scan (e.g., `<resultDir>/100000/output/processing_err0.csv`). This often points to issues with raw data paths or detector geometry in the parameter file.
-   **Indexing Fails:** Check the logs in the main `output/` directory (e.g., `output/indexing_err0.csv`). Failures here might indicate that too few peaks were found in Stage 1 or that indexing parameters (e.g., tolerances) are too strict.
-   **Tomography (`iradon`) Issues:** Ensure that a sufficient number of peaks were indexed for each grain to produce a clean sinogram. Poor reconstructions can result from sparse data.