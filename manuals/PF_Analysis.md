# pf_MIDAS.py User Manual

**Version:** 11.0
**Contact:** hsharma@anl.gov

> [!NOTE]
> For **standard (box-beam)** FF-HEDM analysis, see [FF_Analysis.md](FF_Analysis.md).
> PF-HEDM (Point-Focus / Pencil-beam / Scanning FF-HEDM) processes data from **multiple sample positions** to reconstruct a spatially-resolved microstructure map.

---

## 1. Introduction

`pf_MIDAS.py` is the driver script for **Point-Focus (Scanning) FF-HEDM** analysis within MIDAS. Unlike standard box-beam FF-HEDM, scanning FF-HEDM translates the sample across a focused beam, collecting diffraction data at each position. This allows voxel-level orientation mapping similar to EBSD but using high-energy X-rays for non-destructive 3D characterization.

The script:
1. Runs peak search on each scan position in parallel.
2. Combines spot data across all positions.
3. Performs scanning-mode indexing and refinement.
4. Optionally reconstructs tomographic sinograms (inverse Radon transform) for each grain.
5. Produces a `microstrFull.csv` and `microstructure.hdf` with the spatially-resolved microstructure.

---

## 2. Prerequisites

-   A working MIDAS installation.
-   Raw diffraction data (GE, HDF5, or pre-built Zarr-ZIP).
-   A calibrant-derived parameter file (see [FF_Calibration.md](FF_Calibration.md)).
-   A `positions.csv` file listing Y-positions (one per line, negative w.r.t. motor position).
-   Parameter file and `positions.csv` must be in the **same directory**.
-   Python environment with: `parsl`, `numpy`, `pandas`, `scikit-image`, `Pillow`, `h5py`, `zarr`, `numba`.

---

## 3. Command-Line Arguments

```
python pf_MIDAS.py -paramFile <param.txt> [options]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `-paramFile` | `str` | **Required** | Parameter file name (basename only, not full path). |
| `-resultDir` | `str` | `''` (cwd) | Output directory for results. |
| `-nCPUs` | `int` | `32` | Number of CPUs per node for peak search and indexing. |
| `-nCPUsLocal` | `int` | `4` | Number of local CPUs for non-parallelized tasks. |
| `-nNodes` | `int` | `1` | Number of compute nodes. |
| `-machineName` | `str` | `local` | Execution target: `local`, `orthrosnew`, `orthrosall`, `umich`, `marquette`. |
| `-doPeakSearch` | `int` | `1` | `1` = run peak search; `0` = skip (already done); `-1` = re-process peak output without re-running search. |
| `-convertFiles` | `int` | `1` | `1` = convert raw to Zarr-ZIP; `0` = use existing. |
| `-runIndexing` | `int` | `1` | `1` = run indexing; `0` = skip indexing. |
| `-oneSolPerVox` | `int` | `1` | `1` = single orientation per voxel; `0` = allow multiple orientations per voxel. |
| `-doTomo` | `int` | `1` | `1` = reconstruct tomographic sinograms; `0` = skip. Only for `-oneSolPerVox 1`. |
| `-normalizeIntensities` | `int` | `2` | `0` = equivalent grain size; `1` = normalize by powder intensity; `2` = integrated intensity. |
| `-numFrameChunks` | `int` | `-1` | Chunk data for low-RAM systems. `-1` disables. |
| `-preProcThresh` | `int` | `-1` | Pre-processing threshold above dark. `-1` disables. |
| `-startScanNr` | `int` | `1` | First scan number to process (for partial peak search). |
| `-minThresh` | `int` | `-1` | Filter peaks with maxIntensity below this value. `-1` disables. |
| `-micFN` | `str` | `''` | Path to a `.mic` file for guided indexing. |
| `-grainsFN` | `str` | `''` | Path to a grains file for seed-based indexing. |
| `-omegaFile` | `str` | `''` | Override omega values (one per scan, text file). |
| `-sinoType` | `str` | `raw` | Sinogram variant to use for reconstruction. Valid: `raw`, `norm`, `abs`, `normabs`. See [Section 7](#sinogram-variants). |
| `-sinoSource` | `str` | `tolerance` | Sinogram spot source: `tolerance` = match all spots by angular tolerance (default); `indexing` = use only spots from per-voxel indexing results (cleaner). |
| `-reconMethod` | `str` | `fbp` | Reconstruction algorithm: `fbp` (Filtered Back-Projection via gridrec), `mlem` (Maximum Likelihood EM), `osem` (Ordered Subsets EM). See [Section 8.3](#83-reconstruction-methods-fbp-mlem-osem). |
| `-mlemIter` | `int` | `50` | Number of MLEM/OSEM iterations (only used when `-reconMethod` is `mlem` or `osem`). |
| `-osemSubsets` | `int` | `4` | Number of ordered subsets for OSEM (only used when `-reconMethod` is `osem`). |
| `-resume` | `str` | `''` | Path to a pipeline H5 to resume from. Auto-detects the last completed stage. |
| `-restartFrom` | `str` | `''` | Explicit stage to restart from. Valid stages: `hkl`, `peak_search`, `merge`, `params_rewrite`, `indexing`, `refinement`, `find_multiple_solutions`, `consolidation`. |

### Example

```bash
# Full analysis with tomographic reconstruction:
python pf_MIDAS.py -paramFile ps_pf.txt -resultDir ~/results/ -nCPUs 16 -doTomo 1

# Skip peak search (already done):
python pf_MIDAS.py -paramFile ps_pf.txt -doPeakSearch 0 -nCPUs 32

# Multiple orientations per voxel (no tomography):
python pf_MIDAS.py -paramFile ps_pf.txt -oneSolPerVox 0 -doTomo 0

# Resume from the last completed stage:
python pf_MIDAS.py -paramFile ps_pf.txt -resume /path/to/pipeline.h5

# Restart from indexing:
python pf_MIDAS.py -paramFile ps_pf.txt -restartFrom indexing

# Use MLEM reconstruction instead of FBP:
python pf_MIDAS.py -paramFile ps_pf.txt -doTomo 1 -reconMethod mlem -mlemIter 80

# Use OSEM (faster convergence) with normalized sinograms:
python pf_MIDAS.py -paramFile ps_pf.txt -doTomo 1 -reconMethod osem -osemSubsets 4 -sinoType norm
```

---

## 4. Parameter File Reference

The parameter file uses the same format as [FF_Analysis.md](FF_Analysis.md). The following **additional** keys are specific to PF-HEDM:

| Key | Type | Description |
|---|---|---|
| `nScans` | `int` | Number of scan positions (must match lines in `positions.csv`) |
| `BeamSize` | `float` | Beam size (μm). Multiplied by `nStepsToMerge` if merging is used. |
| `nStepsToMerge` | `int` | Number of adjacent scans to merge (0 = no merging) |
| `MaxAng` | `float` | Maximum misorientation angle for grain matching (degrees) |
| `TolOme` | `float` | Omega tolerance for spot matching (degrees) |
| `TolEta` | `float` | Eta tolerance for spot matching (degrees) |
| `OverAllRingToIndex` | `int` | Primary ring for indexing |
| `MicFile` | `str` | Path to `.mic` file for guided indexing (optional) |
| `GrainsFile` | `str` | Path to seed grains file (optional) |
| `MinMatchesToAcceptFrac` | `float` | Minimum completeness fraction to accept a solution (multi-solution mode) |

All core FF-HEDM parameters (`Lsd`, `Wavelength`, `BC`, tilts, `LatticeParameter`, `SpaceGroup`, etc.) apply identically — see [FF_Analysis.md § 4](FF_Analysis.md).

---

## 5. Input Files

### positions.csv

One Y-position per line (μm), **negative** with respect to the motor position. The number of lines must equal `nScans`.

```
-50.0
-40.0
-30.0
...
```

### Parameter File

Must be in the **same directory** as `positions.csv`. The parameter file name passed via `-paramFile` should be the **basename only** (not a full path).

---

## 6. Workflow Architecture

```mermaid
flowchart TD
    A[Parameter File + positions.csv] --> B[Per-position peak search<br/>parallel_peaks × nScans]
    B --> C{Merge scans?}
    C -->|Yes| D[mergeScansScanning<br/>nStepsToMerge]
    C -->|No| E[Combined spot data]
    D --> E
    E --> F[SaveBinDataScanning<br/>Bin combined data]
    F --> G[IndexerScanningOMP<br/>Find orientations]
    G --> H{One sol per voxel?}
    H -->|Yes| I[findSingleSolutionPF<br/>Select best orientation]
    H -->|No| J[findMultipleSolutionsPF<br/>Allow multiple orientations]
    I --> K{Do tomography?}
    K -->|Yes| L[Sinogram variants<br/>raw / norm / abs / normabs]
    K -->|No| M[FitOrStrainsScanningOMP<br/>Refine positions & strains]
    L --> L2{Recon method?}
    L2 -->|FBP| L3[gridrec via MIDAS_TOMO]
    L2 -->|MLEM| L4[mlem_recon.mlem]
    L2 -->|OSEM| L5[mlem_recon.osem]
    L3 --> N[microstructure.hdf + microstrFull.csv]
    L4 --> N
    L5 --> N
    J --> M
    L3 --> M
    L4 --> M
    L5 --> M
    M --> N

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style N fill:#1a1a2e,stroke:#00d4aa,color:#fff
```

### Stage Descriptions

| Stage | Binary / Function | Description |
|---|---|---|
| **Peak Search** | `parallel_peaks()` (Parsl) | Runs per-position pipeline: ZIP generation, HKL list, peak search, merge, radius, fit setup |
| **Scan Merging** | `mergeScansScanning` | Merges adjacent scan positions to increase signal (optional) |
| **Binning** | `SaveBinDataScanning` | Bins spots from all scan positions for efficient search |
| **Indexing** | `IndexerScanningOMP` | Scanning-mode indexing with position awareness |
| **Single Solution** | `findSingleSolutionPF` | Selects best unique orientation per voxel |
| **Multi Solution** | `findMultipleSolutionsPF` | Allows overlapping grains in a single voxel |
| **Tomography** | FBP / MLEM / OSEM | Reconstructs tomographic images per grain from sinograms (see [Section 8.3](#83-reconstruction-methods-fbp-mlem-osem)) |
| **Refinement** | `FitOrStrainsScanningOMP` | Refines orientations, positions, and strains |

### Pipeline Stage Order (for `-restartFrom`)

The internal stage order used by `-restartFrom` and `-resume` is:

1. `hkl`
2. `peak_search`
3. `merge`
4. `params_rewrite`
5. `indexing`
6. `refinement`
7. `find_multiple_solutions`
8. `consolidation`

Restarting from a given stage re-runs that stage and all subsequent stages.

---

## 7. Output Files

### Directory Structure

```
<resultDir>/
├── positions.csv                  # Input positions
├── paramstest.txt                 # Auto-generated parameter file
├── hkls.csv                       # HKL reflections
├── <startNr>/                     # Per-scan-position subdirectories
│   ├── paramstest.txt
│   ├── <filestem>_NNNNNN.MIDAS.zip
│   └── output/
├── InputAllExtraInfoFittingAll0.csv  # Combined spots (scan 0)
├── InputAllExtraInfoFittingAll1.csv  # Combined spots (scan 1)
├── ...
├── SpotsToIndex.csv               # Spots selected for indexing
├── UniqueOrientations.csv         # Unique grain orientations
├── Output/
│   ├── UniqueIndexSingleKey.bin   # Single-solution voxel map
│   └── IndexBest_voxNr_*.bin      # Best index per voxel
├── Results/
│   └── *.csv                      # Per-voxel refinement results
├── Sinos/                         # Tomographic sinograms (if -doTomo 1)
│   ├── sino_raw_grNr_NNNN.tif    # Raw intensity sinogram
│   ├── sino_norm_grNr_NNNN.tif   # Normalized (I/Imax) sinogram
│   ├── sino_abs_grNr_NNNN.tif    # Absorption (exp(-I)) sinogram
│   └── sino_normabs_grNr_NNNN.tif # Normalized absorption (exp(-I/Imax)) sinogram
├── Thetas/                        # Per-grain theta arrays
│   └── thetas_grNr_*.txt
├── Recons/                        # Reconstructed images
│   ├── recon_grNr_*.tif           # Per-grain reconstruction
│   ├── Full_recon_max_project.tif
│   ├── Full_recon_max_project_grID.tif
│   ├── all_recons_together.tif
│   ├── microstrFull.csv           # ★ Final microstructure CSV
│   └── microstructure.hdf         # ★ Final HDF5 output
├── output/
│   ├── mapping_out.csv / mapping_err.csv
│   ├── indexing_out*.csv / indexing_err*.csv
│   └── refining_out*.csv / refining_err*.csv
└── processing.log
```

### Sinogram Variants

When `-doTomo 1` is enabled, the workflow saves **four sinogram variants** per grain as TIF files in the `Sinos/` subdirectory. These are produced by `findSingleSolutionPFRefactored` and written out by `save_sinogram_variants()`:

| Variant | Filename Pattern | Description |
|---|---|---|
| `raw` | `sino_raw_grNr_NNNN.tif` | Raw intensity values from matched spots |
| `norm` | `sino_norm_grNr_NNNN.tif` | Normalized intensity (I / I_max per grain) |
| `abs` | `sino_abs_grNr_NNNN.tif` | Absorption-like transform: exp(-I) |
| `normabs` | `sino_normabs_grNr_NNNN.tif` | Normalized absorption: exp(-I / I_max) |

Each sinogram TIF has shape `(nScans, nSpots)` and stores double-precision values. The `-sinoType` argument controls which variant is fed into the reconstruction step.

### microstrFull.csv Column Format (43 columns)

The output CSV contains one row per indexed voxel with the following columns:

| Columns | Name | Description |
|---|---|---|
| 1 | `SpotID` | Spot/Voxel identifier |
| 2–10 | `O11`–`O33` | Orientation matrix (3×3, row-major) |
| 11 | `SpotID` | (repeated) |
| 12–14 | `x`, `y`, `z` | Position (μm) |
| 15 | `SpotID` | (repeated) |
| 16–21 | `a`, `b`, `c`, `alpha`, `beta`, `gamma` | Fitted lattice parameters |
| 22 | `SpotID` | (repeated) |
| 23 | `PosErr` | Position error |
| 24 | `OmeErr` | Omega error |
| 25 | `InternalAngle` | Internal angle metric |
| 26 | `Radius` | Equivalent grain radius |
| 27 | `Completeness` | Fraction of expected spots matched |
| 28–36 | `E11`–`E33` | Strain tensor (6 unique components) |
| 37–39 | `Eul1`, `Eul2`, `Eul3` | Euler angles (Bunge convention, degrees) |
| 40–43 | `Quat1`–`Quat4` | Quaternion (fundamental region) |

### microstructure.hdf

HDF5 file with two datasets:
-   **`microstr`**: Full results array (same as `microstrFull.csv`)
-   **`images`**: 3D array `(23, nScans, nScans)` with spatially-resolved data suitable for imaging — includes ID, quaternion, position, lattice parameters, strain, completeness.

---

## 8. Technical Implementation Details

### 8.1. Spatially-Aware Indexing (`IndexerScanningOMP`)
Unlike standard box-beam indexing, the scanning indexer accounts for the sample's translation across the beam.
*   **Dynamic Geometry:** For every candidate voxel at position $(x, y)$, the diffraction spot projection is recalculated. The expected detector $Y$ position ($Y_{det}$) is modified by the sample translation projected onto the detector plane:

$$Y_{det} = Y_{theor} + \frac{x \cdot \sin \omega + y \cdot \cos \omega}{px_{size}}$$


*   **Voxel Grid:** The software discretizes the sample space into a grid defined by the `BeamSize`. It systematically tests orientations at each grid point, effectively performing a "diffraction-based raster scan."

### 8.2. Sinogram Generation
When `-doTomo 1` is enabled, sinograms are constructed before reconstruction:
1.  **Per-grain aggregation:** For each identified grain, the script aggregates intensity metrics across all scan positions and rotation angles ($\omega$). This forms a sinogram where the vertical axis is the scan position and the horizontal axis is the projection angle.
2.  **Four variants:** The sinogram is saved in four processing combinations (raw, normalized, absorption, normalized-absorption) to allow flexibility in the downstream reconstruction. See [Sinogram Variants](#sinogram-variants) in Section 7.
3.  **Variant selection:** The `-sinoType` argument controls which variant is used for reconstruction. For samples with strong absorption contrast, `abs` or `normabs` may produce better grain shape reconstructions. For most diffraction-intensity cases, `raw` (default) or `norm` work well.

### 8.3. Reconstruction Methods (FBP, MLEM, OSEM)

The workflow supports three tomographic reconstruction algorithms, selected via `-reconMethod`:

#### FBP (Filtered Back-Projection) — default

The standard analytical reconstruction method. Uses the `gridrec` algorithm via `MIDAS_TOMO` (`midas_tomo_python.run_tomo_from_sinos`). FBP is fast and produces good results when the sinogram has dense, uniform angular coverage.

```bash
python pf_MIDAS.py -paramFile ps_pf.txt -doTomo 1 -reconMethod fbp
```

#### MLEM (Maximum Likelihood Expectation Maximization)

An iterative statistical reconstruction method (`utils/mlem_recon.mlem`) that is better suited for sparse or irregularly sampled sinograms. MLEM naturally handles:

-   **Missing projections:** Only sinogram rows with non-zero data are used; missing angles do not introduce artifacts.
-   **Non-uniform angular sampling:** No assumption of evenly spaced projection angles.
-   **Positivity constraint:** The multiplicative update inherently produces non-negative reconstructions.
-   **Poisson noise model:** More appropriate than FBP's implicit Gaussian assumption for photon-counting data.

Each iteration forward-projects the current estimate, computes the ratio of measured to projected data, back-projects the ratio, and applies a multiplicative correction. Convergence is controlled by `-mlemIter` (default: 50).

```bash
python pf_MIDAS.py -paramFile ps_pf.txt -doTomo 1 -reconMethod mlem -mlemIter 80
```

#### OSEM (Ordered Subsets Expectation Maximization)

An accelerated variant of MLEM (`utils/mlem_recon.osem`) that divides the projection angles into interleaved subsets and updates the estimate after each subset rather than after a full pass. This converges approximately `n_subsets` times faster than standard MLEM, making it practical for large datasets. Controlled by `-mlemIter` (iterations over all subsets) and `-osemSubsets` (number of subsets, default: 4).

```bash
python pf_MIDAS.py -paramFile ps_pf.txt -doTomo 1 -reconMethod osem -mlemIter 50 -osemSubsets 4
```

#### When to use each method

| Scenario | Recommended Method |
|---|---|
| Dense angular coverage, fast results needed | `fbp` |
| Sparse/irregular omega angles, missing projections | `mlem` or `osem` |
| Large number of grains, need speed | `osem` (faster convergence than `mlem`) |
| Best quality from limited data, time not critical | `mlem` with high iteration count |

---

## 9. Troubleshooting

| Issue | Likely Cause | Resolution |
|---|---|---|
| `positions.csv` not found | Wrong directory or missing file | Ensure `-paramFile` and `positions.csv` are in the same folder |
| `Failed at generateZip for layer N` | Raw data not found | Check `RawFolder`, `FileStem`, `StartFileNrFirstLayer` |
| Empty `InputAllExtraInfoFittingAll*.csv` | No peaks found | Lower `RingThresh`, check detector geometry |
| No sino files found | `findSingleSolutionPF` found no orientations | Lower `MinNrSpots`, check `MaxAng` |
| Out of memory | Large number of scans × rings | Use `-numFrameChunks` and fewer `-nCPUs` |
| `Error reading sino data` | Mismatched `nScans` vs. `positions.csv` | Verify `nScans` matches number of positions |
| `Invalid restart stage` | Wrong stage name in `-restartFrom` | Use one of: `hkl`, `peak_search`, `merge`, `params_rewrite`, `indexing`, `refinement`, `find_multiple_solutions`, `consolidation` |

---

## 10. GPU Acceleration & Consolidated I/O

### Consolidated Binary I/O

Per-voxel file I/O has been replaced with consolidated binary format:
- 3 binary files per scan: `IndexBest_all.bin`, `IndexKey_all.bin`, `IndexBest_IDs_all.bin`
- Reduces filesystem overhead from ~30K+ small files to 3 files
- Uses `IndexerConsolidatedIO.h` with `VoxelAccumulator` (writer) and `ConsolidatedReader` (mmap-based reader) for O(1) voxel access
- `pf_MIDAS.py` reads consolidated `IndexBest_all.bin` via numpy for `.mic` file generation

### GPU Acceleration

Enable GPU-accelerated scanning indexer and fitter:

```bash
python pf_MIDAS.py -paramFN params.txt -useGPU 1
```

`IndexerScanningGPU` supports three modes: spot-driven with beam proximity filter, MicFile-seeded, and GrainsFile-seeded. `FitOrStrainsScanningGPU` reads consolidated indexer output.

### Resume/Restart

`--resume` and `--restartFrom` flags are fully wired with `_should_run()` gates for proper stage-skipping.

See [GPU_Acceleration.md](GPU_Acceleration.md) for full GPU documentation.

---

## 11. See Also

- [FF_Analysis.md](FF_Analysis.md) — Standard (box-beam) FF-HEDM analysis
- [FF_Calibration.md](FF_Calibration.md) — Geometry calibration
- [PF_Interactive_Plotting.md](PF_Interactive_Plotting.md) — Interactive sinogram, intensity & tomo viewer for PF-HEDM
- [FF_Interactive_Plotting.md](FF_Interactive_Plotting.md) — Visualizing FF-HEDM results
- [Tomography_Reconstruction.md](Tomography_Reconstruction.md) — MIDAS tomography reconstruction
- [Forward_Simulation.md](Forward_Simulation.md) — Forward simulation for validation
- [NF_Analysis.md](NF_Analysis.md) — Near-field HEDM reconstruction
- [README.md](README.md) — High-level MIDAS overview and manual index

---

If you encounter any issues or have questions, please open an issue on this repository.
