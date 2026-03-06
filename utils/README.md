# utils/ — Python Utilities for MIDAS

Core data processing, calibration, and analysis scripts for the MIDAS pipeline.

> **Note:** As of 2026-03, several file groups have been reorganized:
> - **Tests** → [`tests/`](../tests/) (6 files)
> - **Viewers & plotting** → [`gui/viewers/`](../gui/viewers/) (14 files)
> - **Format converters** → [`converters/`](converters/) (12 files)
> - **Deprecated** → [`archive/`](archive/)

---

## Core Workflow & Calibration

| Script | Description |
|--------|-------------|
| `AutoCalibrateZarr.py` | **FF-HEDM auto-calibration.** Determines detector geometry from powder calibrant rings. Auto-detects calibrant (CeO2/LaB6) from filename. |
| `phase_id.py` | **Multi-phase identification.** Ring-matching, peak fitting, lattice parameter refinement. See [FF_Phase_Identification](../manuals/FF_Phase_Identification.md). |
| `extract_lineouts.py` | **Batch lineout extraction.** Direct-mode `IntegratorZarrOMP` with SNIP background, SavGol peak detection, and multiplet pseudo-Voigt fitting. |
| `gsas_ii_refine.py` | Import caked 1D profiles into GSAS-II for powder diffraction refinement. |
| `map_header.py` | `Map.bin` header reader — detects stale mapping files via parameter hashes. |
| `midas_config.py` | MIDAS configuration and environment variable management. |

Integrator scripts moved to [`FF_HEDM/workflows/`](../FF_HEDM/workflows/): `integrator.py`, `integrator_batch_process.py`, `integrator_server.py`, `integrator_stream_process_h5.py`, `integrate_and_refine.py`.

## Data Conversion & Format Utilities

| Script | Description |
|--------|-------------|
| `ffGenerateZipRefactor.py` | Convert raw detector data (HDF5 / GE / TIFF) into MIDAS Zarr-ZIP archives. |
| `updateZarrDset.py` | Update datasets/metadata within existing Zarr-ZIP archives. |

See [`converters/`](converters/) for standalone format converters: `GE2Tiff.py`, `ang2mic.py`, `esrf2zip_pf.py`, `esrf_to_ge.py`, `ff2midas.py`, `mergeGEfiles.py`, `mergeH5s.py`, `mergePeaks.py`, `midas2zip.py`, `SpotMatrixToSpotsHDF.py`, `GFF2Grains.py`, `vtkSimExportBin.py`.

## Grain Matching & Analysis

| Script | Description |
|--------|-------------|
| `match_grains.py` | **Grain matching and layer stitching.** Hungarian or greedy matching across load states. See [FF_Match_Stack_Reconstructions](../manuals/FF_Match_Stack_Reconstructions.md). |
| `calcMiso.py` | Crystallographic misorientation calculations. Used as a library by `match_grains.py`. |

## FF-HEDM Post-Processing

| Script | Description |
|--------|-------------|
| `ff_fitgrain.py` | Refines orientation, position, and strain for individual grains. |
| `ff_hdf_result.py` | Write FF-HEDM results (grain list + spot matrix) into HDF5. |
| `ffGenerateHDFResult.py` | Generate HDF5 result files from FF-HEDM outputs. |
| `extractPeaks.py` | Extract individual peak data from the spot table. |
| `processFilesParallel.py` | Parallel file processing for batch FF-HEDM operations. |
| `extract_grains_mic.py` | Extract grain data from `.mic` files. |

## NF-HEDM Post-Processing

| Script | Description |
|--------|-------------|
| `NFGrainCentroids.py` | Calculate grain centroids from NF-HEDM `.mic` files. |
| `nf_mic_to_grains.py` | Convert NF `.mic` files to `Grains.csv` format. |
| `nf_neighbor_calc.py` | Compute grain neighbor relationships and boundary statistics. |
| `nf_paraview_gen.py` | Generate ParaView VTK files from NF reconstructions. |
| `findUniqueOrientationsNF.py` | Identify unique grain orientations from NF data. |
| `hdf_gen_nf.py` | Generate HDF5 files from NF-HEDM inputs/outputs. |
| `nf_manipulation_codes.py` | Utility functions for NF `.mic` data. |

## Visualization

All viewers moved to [`gui/viewers/`](../gui/viewers/). Includes: `plot_lineout_results.py`, `plot_integrator_peaks.py`, `plot_calibrant_results.py`, `plot_phase_id_results.py`, `plot_lineout_comparison.py`, `live_viewer.py`, `interactiveFFplotting.py`, `pfIntensityViewer.py`, `peak_sigma_statistics.py`, `plotFFSpots3d.py`, `plotGrains3d.py`, `PlotFFNF.py`, `viz_caking.py`.

Remaining here: `sino_cleanup_tomo.py` — sinogram cleanup and tomo reconstruction. See [Tomography_Reconstruction](../manuals/Tomography_Reconstruction.md).

## Simulation

| Script | Description |
|--------|-------------|
| `simulatePeaks.py` | Simulate diffraction peak positions for testing. |
| `sim_ff_transformed.py` | Generate transformed FF-HEDM simulation data. |
| `compressedSimulationReader.py` | Read compressed forward simulation output. |

Tests moved to [`tests/`](../tests/): `test_ff_hedm.py`, `test_nf_hedm.py`, `test_ff_calibration.py`, `test_integrator_peaks.py`, `test_phase_id.py`, `test_live_viewer.py`.

## Scanning / Point-Focus HEDM

| Script | Description |
|--------|-------------|
| `runScanning.py` | Driver for scanning (point-focus) HEDM experiments. |
| `evalScanning.py` | Evaluate scanning HEDM results. |

## Specialized Tools

| Script | Description |
|--------|-------------|
| `blobPeaksearch.py` | Blob-based peak search on raw detector images. |
| `BatchCake.py` | Batch radial integration. |
| `batchImages.py` | Batch image processing (dark subtraction, normalization). |
| `DL2FF.py` | Convert deep-learning peak predictions to FF-HEDM input. |
| `run_full_images_ff.py` | Process full detector images through FF-HEDM. |
| `ff_peaks_raw_images.py` | Extract peaks from raw images for diagnostics. |
| `undistort_image.py` | Apply spatial distortion correction to detector images. |
| `generate_mask.py` | Create uint8 TIFF mask files. Supports Dioptas `.mask` input. |

## Deprecated (`archive/`)

| Script | Description |
|--------|-------------|
| `deprecated_AutoCalibrate.py` | Superseded by `AutoCalibrateZarr.py`. |
| `ffGenerateZip.py` | Superseded by `ffGenerateZipRefactor.py`. |
| `updateZarrDsetRefactor.py` | Unused refactored Zarr updater (original `updateZarrDset.py` is still active). |