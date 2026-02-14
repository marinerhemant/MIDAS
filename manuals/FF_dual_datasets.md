# ff_dual_datasets.py User Manual

**Version:** 9.0  
**Contact:** hsharma@anl.gov

---

## 1. Introduction

`ff_dual_datasets.py` is a specialized and powerful script within the MIDAS suite designed for the combined analysis of **two separate but related far-field HEDM datasets**. Its primary purpose is to process two datasets independently through the initial stages, spatially map them into a common reference frame, and then perform a single, unified indexing and refinement on the combined data.

This script is ideal for scenarios such as:
-   Analyzing a sample **before and after** an in-situ experiment (e.g., heating or mechanical loading).
-   Combining two **overlapping scans** to create a larger, contiguous map.
-   Correlating datasets collected under slightly different experimental conditions.

By leveraging Parsl, the script efficiently parallelizes the workflow across multiple cores or HPC nodes, making it a robust tool for advanced, comparative HEDM analysis.

### Key Features
*   **Dual Dataset Processing:** Natively handles the entire workflow for two distinct datasets.
*   **Spatial Mapping:** Uses user-provided offsets (X, Y, Z, Omega) to align and merge the two datasets.
*   **Three-Stage Workflow:** Automates a complex process involving independent pre-processing, data mapping, and a final combined analysis.
*   **End-to-End Automation:** Manages all steps from raw data conversion to the final `Grains.csv` file for the combined volume.
*   **HPC-Ready:** Includes configurations for local workstations and various HPC clusters.

---

## 2. Prerequisites

1.  **MIDAS Installation:** The script must be located within a functioning MIDAS installation, as it depends on binaries like `MapDatasets` and `ProcessGrains` from the `FF_HEDM/bin/` directory.
2.  **Python Environment:** A Python environment with `parsl`, `numpy`, and other standard scientific libraries installed.
3.  **Input Files:**
    *   A **single Parameter File** (`-paramFN`) that is applicable to *both* datasets.
    *   A **Data File for Dataset 1** (`-dataFN`). This is the reference dataset.
    *   A **Data File for Dataset 2** (`-dataFN2`). This dataset will be mapped onto the first.
    *   **Four Offset Values:** You must provide the spatial (`-offsetX`, `-offsetY`, `-offsetZ`) and rotational (`-offsetOmega`) offsets required to align Dataset 2 with Dataset 1.

---

## 3. Workflow Overview

The script executes a sophisticated three-stage workflow:
 
 ```mermaid
 graph TD
     subgraph "Stage 1: Independent Pre-processing (Parallel)"
         direction TB
         D1[Dataset 1 Input] --> P1[Pre-process & Bin];
         D2[Dataset 2 Input] --> P2[Pre-process & Bin];
     end
 
     subgraph "Stage 2: Mapping"
         P2 --> M[MapDatasets];
         P1 --> M;
         M -- "Apply Offsets (X, Y, Z, Omega)" --> C[Combined Binned Data];
     end
 
     subgraph "Stage 3: Combined Analysis"
         C --> I[IndexerOMP];
         I --> R[FitPosOrStrainsDoubleDataset];
         R --> G[ProcessGrains];
         G --> F[Final Grains.csv];
     end
 ```

### Stage 1: Independent Pre-processing
The script first processes both datasets entirely separately and in parallel. For each dataset, it creates a dedicated analysis folder (`dataset_1_analysis` and `dataset_2_analysis`) and performs the following steps:
1.  **Data Conversion:** Converts raw data (e.g., HDF5) to a `.MIDAS.zip` (Zarr) archive.
2.  **HKL Generation:** Generates the list of theoretical Bragg reflections.
3.  **Peak Search:** Finds all diffraction peaks in the data.
4.  **Peak Merging & Prep:** Merges overlapping peaks and prepares the data for indexing.
5.  **Data Binning:** Runs `SaveBinData` to create a binned representation of the diffraction spots in 3D space.

At the end of this stage, you will have two folders, each containing the fully processed but un-indexed results for one dataset.

### Stage 2: Dataset Mapping
This is the core step that makes the script unique.
1.  The script takes the binned data from both datasets.
2.  It calls the `MapDatasets` MIDAS binary.
3.  Using the user-provided offsets, `MapDatasets` transforms the binned data from Dataset 2 into the coordinate system of Dataset 1 and merges them.
4.  The result is a new, larger set of binned data files stored within the `dataset_1_analysis` folder, representing the combined volume.

### Stage 3: Combined Analysis
Finally, the script performs the indexing and refinement steps on the single, merged dataset created in Stage 2. All work is now done inside the `dataset_1_analysis` folder.
1.  **Indexing:** `IndexerOMP` is run in parallel on the combined binned data to find grain orientation candidates.
2.  **Refinement:** `FitPosOrStrainsDoubleDataset` refines the orientation, position, and strain for each indexed grain.
3.  **Grain Processing:** `ProcessGrains` compiles the final results into a single `Grains.csv` file.

The final output is one consistent microstructure map derived from the information of both initial datasets.

---

## 4. Technical Implementation Details

### 4.1. Orchestration with Parsl
The workflow uses **Parsl**, a Python parallel scripting library, to manage concurrency.
*   **Parallel Execution:** `ff_dual_datasets.py` defines Parsl "apps" (`peaks`, `index`, `refine`) that wrap the C binaries. This allows the script to run pre-processing for both datasets simultaneously on available resources (e.g., 2 nodes on a cluster).
*   **Machine Configuration:** The script calculates optimal resource allocation (`num_procs`, `n_nodes`) based on the `-machineName` argument, loading pre-defined configurations for known clusters (e.g., `orthros`, `purdue`).

### 4.2. Dataset Mapping Logic
*   **Parameter Propagation:** The script appends a special key, `Dataset2Folder`, to the parameter file of the first dataset (`paramstest.txt`). This line contains the path to the second dataset's results and the 4 user-provided offsets (X, Y, Z, Omega).
*   **MapDatasets Binary:**
    *   This C tool loads the diffraction spots from both datasets (`Spots.bin`, `ExtraInfo.bin`).
    *   It parallelizes (OpenMP) over the spots in the second dataset.
    *   For each spot, it applies the rotational offset (`-offsetOmega`) and converts the detector coordinates to a **g-vector** (scattering vector in sample frame).
    *   It performs a fast grid search (hashed by Ring, Eta, Omega) to find the matching spot in Dataset 1 with the highest cosine similarity (dot product of g-vectors).
    *   The result is a mapping index file (`mapDatasets.txt`) that links observations across the two datasets.

### 4.3. Combined Indexing
*   **Binder:** The final indexing and refinement steps (`IndexerOMP`, `FitPosOrStrainsDoubleDataset`) read the `Dataset2Folder` info. They utilize the mapping from `mapDatasets.txt` to treat corresponding spots from both datasets as observations of the same grain, minimizing the global error across the combined volume.

---

## 5. Command-Line Arguments

The script's behavior is controlled via the following arguments.

### Dual-Dataset Specific Arguments

| Argument | Description | Required | Example |
| :--- | :--- | :--- | :--- |
| `-resultFolder` | Top-level folder where all analysis subdirectories will be saved. | **Yes** | `-resultFolder /path/to/analysis` |
| `-paramFN` | The main parameter file, used for both datasets. | **Yes** | `-paramFN Ti_params.txt` |
| `-dataFN` | Data file for the **first (reference)** dataset. | **Yes** | `-dataFN before_heat.h5` |
| `-dataFN2` | Data file for the **second** dataset to be mapped. | **Yes** | `-dataFN2 after_heat.h5` |
| `-offsetX` | Offset in **X** to map Dataset 2 to Dataset 1 (micrometers). | **Yes** | `-offsetX 10.5` |
| `-offsetY` | Offset in **Y** to map Dataset 2 to Dataset 1 (micrometers). | **Yes** | `-offsetY -5.2` |
| `-offsetZ` | Offset in **Z** to map Dataset 2 to Dataset 1 (micrometers). | **Yes** | `-offsetZ 0.0` |
| `-offsetOmega`| Rotational offset in **Omega** to map Dataset 2 to Dataset 1 (degrees).| **Yes** | `-offsetOmega 0.15` |

### Standard Configuration Arguments

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-machineName` | Execution environment. Options: `local`, `orthrosnew`, `orthrosall`, `umich`, `marquette`, `purdue`. | `local` | `-machineName purdue` |
| `-nNodes` | Number of compute nodes to request for the analysis on a cluster. | `1` | `-nNodes 4` |
| `-nCPUs` | Number of CPU cores to use per node/task. | `10` | `-nCPUs 128` |
| `-numFrameChunks`| Splits large datasets into chunks during conversion to save RAM. `-1` disables chunking. | `-1` | `-numFrameChunks 4` |
| `-preProcThresh`| Saves dark-corrected/thresholded data during conversion. `-1` disables. `0` just subtracts dark. | `-1` | `-preProcThresh 100` |

---

## 6. Execution Example

Submit an analysis to the `Purdue` cluster, requesting 2 nodes. The job will process two datasets, `set1.h5` and `set2.h5`, applying the specified offsets to align them before the final indexing.

```bash
python /path/to/ff_dual_datasets.py \
    -resultFolder /scratch/user/in_situ_heating_exp \
    -paramFN /home/user/params/Inconel_params.txt \
    -dataFN /raw_data/set1.h5 \
    -dataFN2 /raw_data/set2.h5 \
    -offsetX 15.0 \
    -offsetY -10.2 \
    -offsetZ 1.5 \
    -offsetOmega -0.25 \
    -machineName purdue \
    -nNodes 2 \
    -nCPUs 128
```

---

## 7. Output Directory Structure

The script generates two initial analysis directories within the main `-resultFolder`, but the final combined results are all consolidated into the first one.

```
<resultFolder>/
├── dataset_1_analysis/       # Primary analysis folder for Dataset 1
│   ├── output/               # Logs for all stages, including mapping and combined analysis
│   │   ├── peaksearch_out0.csv # Peak search logs for dataset 1
│   │   ├── map_out.txt         # Log for the MapDatasets binary
│   │   ├── indexing_out0.csv   # Logs for the combined indexing
│   │   └── ...
│   ├── Grains.csv            # **FINAL COMBINED OUTPUT**
│   ├── SpotsToIndex.csv      # Spots from the combined dataset
│   ├── paramstest.txt        # The parameter file, now with the "Dataset2Folder" line added
│   └── ...                   # All other intermediate files for the combined analysis
│
└── dataset_2_analysis/       # Analysis folder for Dataset 2
    ├── output/               # Logs for the independent pre-processing of Dataset 2
    │   ├── peaksearch_out0.csv
    │   └── ...
    ├── binnedData/           # Binned data for dataset 2 (used by the mapping step)
    └── ...                   # Other intermediate files for dataset 2
```

### Key Output Files

-   **`dataset_1_analysis/Grains.csv`**: This is the single, primary output file containing the final list of grains and their properties, derived from the combined information of both datasets.
-   **`dataset_1_analysis/output/map_err.txt`**: The error log for the crucial `MapDatasets` step. Check this file if the mapping stage fails or produces unexpected results.
-   **`dataset_*/output/`**: The `output` folders contain detailed logs from every MIDAS binary. The logs in `dataset_1_analysis` will cover both its own pre-processing and the entire combined analysis stage.

---

## 8. Troubleshooting

-   **Mapping Fails (`MapDatasets` error):** The most common issue is incorrect offsets. Double-check the signs and values of your `-offset*` arguments. Small errors in offsets can cause the algorithm to fail to find corresponding volumes. Check `dataset_1_analysis/output/map_err.txt` for details.
 
 > [!NOTE]
 > If `MapDatasets` produces an empty or very small combined dataset, verify that your provided offsets actually result in spatial overlap between the two scanned volumes.
-   **Pre-processing Fails:** If one of the initial stages fails, treat it as a standard `ff_MIDAS.py` failure. Check the `output` directory of the corresponding dataset (e.g., `dataset_2_analysis/output/`) to debug issues with peak finding, data conversion, etc.
-   **Poor Indexing Results:** If the final indexing yields few grains, it could be a sign of poor alignment during the mapping stage. This can happen if the offsets are not precise enough, leading to a "blurry" or inconsistent combined dataset.

---

## 9. See Also

- [FF_Analysis.md](FF_Analysis.md) — Standard single-dataset FF-HEDM analysis
- [FF_calibration.md](FF_calibration.md) — Geometry calibration
- [FF_Interactive_Plotting.md](FF_Interactive_Plotting.md) — Visualizing FF-HEDM results
- [ForwardSimulationManual.md](ForwardSimulationManual.md) — Forward simulation for validation
- [README.md](README.md) — High-level MIDAS overview and manual index

---

If you encounter any issues or have questions, please open an issue on this repository.