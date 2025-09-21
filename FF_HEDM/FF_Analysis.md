# ff_MIDAS.py User Manual

**Version:** 7.0.0  
**Contact:** hsharma@anl.gov

---

## 1. Introduction

`ff_MIDAS.py` is a powerful command-line script designed to automate the far-field High-Energy Diffraction Microscopy (FF-HEDM) analysis pipeline using the MIDAS (Microstructure-Informed Design and Analysis Software) package. It orchestrates a series of complex data processing and analysis steps, from raw diffraction data to a final, refined list of crystal grains.

The script is built for efficiency and robustness, leveraging the Parsl parallel programming library to distribute computationally intensive tasks across multiple processors and compute nodes. This makes it suitable for execution on a local workstation, a multi-core server, or a high-performance computing (HPC) cluster.

### Key Features
*   **End-to-End Automation:** Runs the complete FF-HEDM workflow, including data conversion, peak finding, indexing, and strain refinement.
*   **Parallel Processing:** Uses Parsl to significantly speed up analysis by parallelizing key steps like peak search, indexing, and refinement.
*   **Machine-Aware Configuration:** Includes pre-built configurations for various computing environments (e.g., `local`, `purdue`, `umich`).
*   **Flexible Workflow Control:** Command-line arguments allow users to run the entire pipeline, execute specific parts (e.g., peak search only), or resume analysis from intermediate steps.
*   **Robust Error Handling:** Features safe command execution, automatic cleanup, and a retry mechanism for transient errors, ensuring more reliable runs.
*   **Detailed Logging:** Provides comprehensive logging of progress, timings, and errors to help monitor and debug the analysis process.

---

## 2. Prerequisites

Before running `ff_MIDAS.py`, ensure the following requirements are met:

1.  **MIDAS Installation:** The script must be located within the correct directory structure of a functioning MIDAS installation. It relies on various binaries (e.g., `PeaksFittingOMPZarrRefactor`, `IndexerOMP`) and utility scripts located in the `bin/` and `utils/` directories of the MIDAS package.
2.  **Python Environment:** A Python interpreter with the `parsl` and `numpy` libraries installed.
3.  **Machine Configuration Files:** If running on an HPC cluster (`umich`, `purdue`, etc.), the corresponding Python configuration files (e.g., `uMichConfig.py`) must be present and correctly configured.
4.  **Environment Variable (Optional):** The script automatically detects the MIDAS installation directory. However, for robustness, you can manually set the `MIDAS_INSTALL_DIR` environment variable to point to the root of your MIDAS installation.

---

## 3. Script Overview & Workflow

The script processes experimental data on a "layer-by-layer" basis. A layer typically corresponds to one complete scan or dataset. The core analysis workflow for each layer consists of the following steps:

1.  **Setup:** The script creates a dedicated result directory for the current layer and copies the main parameter file into it.
2.  **Data Conversion (Optional):** If starting from raw data (e.g., HDF5 or GE detector files), the script calls `ffGenerateZipRefactor.py` to convert and combine the data into a single, optimized `.MIDAS.zip` file (a Zarr dataset). This step can be skipped if a Zarr file is already available.
3.  **HKL List Generation:** A list of theoretical Bragg reflections (HKLs) is generated based on the crystal structure information in the parameter file. This is done by the `GetHKLListZarr` or `GetHKLList` binary.
4.  **Peak Search (Optional):** The `PeaksFittingOMPZarrRefactor` binary is run in parallel across the specified nodes/cores to find diffraction peaks in the detector images. This is one of the most time-consuming steps and can be skipped if peak data is already present.
5.  **Peak Merging:** Overlapping peaks found in adjacent frames are merged into a consolidated list using `MergeOverlappingPeaksAllZarr`.
6.  **Data Preparation for Indexing:** A series of preparatory steps (`CalcRadiusAllZarr`, `FitSetupZarr`) are executed to transform the peak data into a format suitable for indexing.
7.  **Data Binning:** The `SaveBinData` binary bins the spots to accelerate the indexing process.
8.  **Indexing:** The `IndexerOMP` binary is run in parallel to determine the crystallographic orientation (i.e., find potential grains) from the list of measured peaks.
9.  **Refinement:** The `FitPosOrStrainsOMP` binary is run in parallel to refine the orientation, position, and strain tensor for each indexed grain.
10. **Grain Processing:** Finally, `ProcessGrainsZarr` (or `ProcessGrains`) compiles the results from all refinement tasks into a final `Grains.csv` file, which contains the primary output of the analysis.

---

## 4. Command-Line Arguments

The script's behavior is controlled via a rich set of command-line arguments.

### Basic Arguments

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-resultFolder` | Folder where all analysis results will be saved. | Current directory | `-resultFolder /path/to/analysis` |
| `-paramFN` | The main parameter file. **Required** unless `-dataFN` for an existing ZIP is provided. | `''` | `-paramFN Ti_params.txt` |
| `-dataFN` | Path to the input data file (HDF5 or `.MIDAS.zip`). | `''` | `-dataFN my_scan.h5` |

### Execution Control

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-machineName` | Execution environment. Determines the Parsl configuration. Options: `local`, `orthrosnew`, `orthrosall`, `umich`, `marquette`, `purdue`. | `local` | `-machineName purdue` |
| `-nCPUs` | Number of CPU cores to use per parallel task (mainly for `local` execution). | `10` | `-nCPUs 16` |
| `-nNodes` | Number of compute nodes to request for the analysis on a cluster. | `1` | `-nNodes 4` |

### Layer and File Selection

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-startLayerNr` | The starting layer number to process. | `1` | `-startLayerNr 5` |
| `-endLayerNr` | The ending layer number to process (inclusive). | `1` | `-endLayerNr 10` |
| `-fileName` | Process a single, specific data file, overriding layer range arguments. | `''` | `-fileName sample_00123.h5` |

### Workflow Step Control

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-convertFiles` | Set to `0` to skip data conversion if `.MIDAS.zip` files already exist. | `1` | `-convertFiles 0` |
| `-doPeakSearch` | Set to `0` to skip the peak search step if results are already available. | `1` | `-doPeakSearch 0` |
| `-peakSearchOnly` | Set to `1` to run only data conversion and peak search steps. | `0` | `-peakSearchOnly 1` |
| `-provideInputAll`| Set to `1` for an advanced mode using a pre-existing `InputAllExtraInfoFittingAll.csv` file. | `0` | `-provideInputAll 1` |

### Advanced Data Handling

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-numFrameChunks` | Splits large datasets into chunks during conversion to save RAM. `-1` disables chunking. | `-1` | `-numFrameChunks 4` |
| `-preProcThresh` | Saves dark-corrected/thresholded data during conversion. `-1` disables. | `-1` | `-preProcThresh 100` |
| `-rawDir` | Overrides the `RawFolder` path specified in the parameter file. | `''` | `-rawDir /new/path/to/data` |
| `-grainsFile` | Optional input file with seed grains to guide indexing. | `''` | `-grainsFile seed_grains.csv` |

---

## 5. Execution Examples

### Example 1: Local Analysis
Run an analysis for layers 1 through 5 on a local machine using 16 cores.

```bash
python /path/to/ff_MIDAS.py \
    -resultFolder /home/user/analysis/Ti_sample \
    -paramFN /home/user/params/Ti_params.txt \
    -dataFN /home/user/raw_data/Ti_scan_001.h5 \
    -machineName local \
    -nCPUs 16 \
    -startLayerNr 1 \
    -endLayerNr 5
```

### Example 2: Cluster Analysis
Submit a job to the `Purdue` cluster, requesting 8 nodes to process layers 10-20, skipping the initial file conversion.

```bash
python /path/to/ff_MIDAS.py \
    -resultFolder /scratch/user/analysis/Inconel_run2 \
    -paramFN /home/user/params/Inconel_params.txt \
    -dataFN /scratch/user/zarr_data/Inconel_scan_010.MIDAS.zip \
    -machineName purdue \
    -nNodes 8 \
    -startLayerNr 10 \
    -endLayerNr 20 \
    -convertFiles 0
```
---

## 6. Output Directory Structure

The script generates a structured output in the specified `-resultFolder`. For each layer, a subdirectory is created with a consistent layout.

```
<resultFolder>/
└── LayerNr_1/
    ├── output/                 # Contains stdout/stderr logs for each binary
    │   ├── peaksearch_out0.csv
    │   ├── peaksearch_err0.csv
    │   ├── indexing_out0.csv
    │   └── ...
    ├── Temp/                   # Temporary file storage
    ├── Grains.csv              # FINAL OUTPUT: List of found grains and their properties
    ├── SpotsToIndex.csv        # List of spot IDs selected for indexing
    ├── hkls.csv                # List of theoretical reflections
    ├── paramstest.txt          # A copy/modified version of the parameter file used for this layer
    └── Ti_scan_000001.MIDAS.zip # The Zarr dataset created from the raw data
└── LayerNr_2/
    └── ...
```

-   **`LayerNr_*/`**: The main directory for each processed layer.
-   **`output/`**: **This is the most important directory for debugging.** It contains the standard output (`...out...`) and standard error (`...err...`) logs from every external program executed by the script. If a step fails, check the corresponding `err` file for detailed messages.
-   **`Grains.csv`**: This is the primary result file, containing information about each grain found, including its position, orientation (in Rodrigues vector form), and strain tensor components.
-   **`.MIDAS.zip`**: The converted Zarr data archive. Note that despite the `.zip` extension, this is a directory-based format and should not be unzipped manually.

---

## 7. Error Handling and Troubleshooting

-   **Ctrl+C Handling:** If you interrupt the script with `Ctrl+C`, it will attempt to gracefully shut down the Parsl Data Flow Kernel to prevent leaving zombie processes.
-   **Automatic Retries:** Computationally intensive parallel steps (peak search, index, refine) have a built-in retry mechanism. If a task fails due to a transient issue (like a network filesystem glitch), it will be automatically retried up to 3 times with an increasing delay.
-   **Log Files:** The script produces two types of logs:
    1.  **Console Log:** Progress, timings, and major status updates are printed to the console. It is highly recommended to redirect this to a file for later review: `python ff_MIDAS.py ... > run.log 2>&1`.
    2.  **Binary Logs:** As mentioned above, the `resultFolder/LayerNr_*/output/` directory contains detailed logs from each step of the MIDAS C++ backend. **Always check these files first when a run fails.**
-   **Common Errors:**
    *   **`FileNotFoundError`:** Usually caused by an incorrect path in the `-paramFN`, `-dataFN`, or the `RawFolder` parameter inside the parameter file. Double-check all paths.
    *   **Indexing or Refinement Fails:** If `IndexerOMP` or `FitPosOrStrainsOMP` fails, it often indicates a problem with the input data or parameters. Check for errors in peak finding or inspect the `paramstest.txt` file in the layer directory to ensure the geometry is correct.
    *   **Parsl Errors:** Errors related to Parsl often stem from misconfiguration of the cluster environment (e.g., incorrect partition name, walltime limits). Review the `*Config.py` file for your machine.