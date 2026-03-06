# utils/ — Python Utilities for MIDAS

This directory contains Python scripts and tools used across the MIDAS analysis pipeline.

> **Note:** As of 2026-03, several file groups have been reorganized:
> - **Tests** → `tests/` (6 files)
> - **Viewers & plotting** → `gui/viewers/` (14 files)
> - **Format converters** → `utils/converters/` (12 files)

---

## Core Workflow & Calibration

| Script | Description |
|--------|-------------|
| `AutoCalibrateZarr.py` | **FF-HEDM auto-calibration.** Determines detector tilt angles, beam center, and sample-to-detector distance from powder calibrant diffraction rings. Auto-detects file format and calibrant material (CeO2/LaB6) from filename. Uses all CPU cores. |
| `integrator.py` | **Radial integrator (caking).** Converts 2D detector images to 1D intensity vs. 2θ profiles. Supports multiple detector geometries and azimuthal wedges. |
| `integrator_batch_process.py` | Batch-mode radial integration for processing large datasets non-interactively. |
| `integrator_server.py` | Long-running integration server for real-time data reduction during beamline operation. |
| `integrator_stream_process_h5.py` | Streaming radial integration from HDF5 files as they are written during acquisition. |
| `phase_id.py` | **Multi-phase identification.** Identifies crystallographic phases from diffraction images using ring-matching, peak fitting, and lattice parameter refinement. Outputs per-phase detection summary **and** a combined peak table (`peak_table.csv` / `peak_table.txt`) with R, 2θ, FWHM, intensity, and detection flag for all fitted peaks. Supports single/multi-file, `-startNr`/`-endNr`, `-dataFolder`, and `--multi-cpu N` parallel modes. See [FF_Phase_Identification](../manuals/FF_Phase_Identification.md). |
| `map_header.py` | **Map.bin header reader.** Python counterpart to `MapHeader.h` — reads 64-byte parameter-hash headers from `Map.bin`/`nMap.bin` to detect stale mapping files. Used by orchestrators. |
| `live_viewer.py` | **Real-time dashboard.** PyQtGraph-based live viewer that tails `lineout.bin` and `fit.bin` from the GPU integrator, showing 1D lineouts, heatmap waterfall, and peak evolution plots. See [FF_Radial_Integration](../manuals/FF_Radial_Integration.md) §6.3. |
| `integrate_and_refine.py` | Combined radial integration + GSAS-II peakfit refinement pipeline. |
| `gsas_ii_refine.py` | **GSAS-II integration.** Imports caked 1D profiles into GSAS-II for powder diffraction refinement. |
| `midas_config.py` | MIDAS configuration and environment variable management. |

## Grain Matching & Analysis

| Script | Description |
|--------|-------------|
| `match_grains.py` | **Grain matching and layer stitching.** Matches grains across load states (Hungarian or greedy) and stitches multi-layer scans. Supports affine deformation transforms, consolidated HDF5 input, and library-callable API. See [FF_Match_Stack_Reconstructions manual](../manuals/FF_Match_Stack_Reconstructions.md). |
| `calcMiso.py` | **Crystallographic misorientation calculations.** Quaternion symmetry operations, orientation matrix ↔ quaternion conversions, misorientation angles for all 230 space groups. Used as a library by `match_grains.py` and other tools. |

## Data Conversion & Format Utilities

Zarr/ZIP utilities remain here. Standalone converters moved to `utils/converters/`.

| Script | Description |
|--------|-------------|
| `ffGenerateZip.py` | Convert raw detector data (HDF5 / GE / TIFF) into MIDAS Zarr-ZIP archives. |
| `ffGenerateZipRefactor.py` | Refactored version with improved performance and memory management. |
| `updateZarrDset.py` | Update datasets/metadata within existing Zarr-ZIP archives. |
| `updateZarrDsetRefactor.py` | Refactored Zarr dataset updater. |

See `utils/converters/` for: `GE2Tiff.py`, `ang2mic.py`, `esrf2zip_pf.py`, `esrf_to_ge.py`, `ff2midas.py`, `mergeGEfiles.py`, `mergeH5s.py`, `midas2zip.py`, etc.

## FF-HEDM Post-Processing

| Script | Description |
|--------|-------------|
| `ff2midas.py` | Convert external FF-HEDM formats into MIDAS-compatible data structures. |
| `ff_fitgrain.py` | Python-based grain fitting: refines orientation, position, and strain for individual grains. |
| `ff_hdf_result.py` | Write FF-HEDM results (grain list + spot matrix) into HDF5 format. |
| `ffGenerateHDFResult.py` | Generate HDF5 result files from FF-HEDM analysis outputs. |
| `SpotMatrixToSpotsHDF.py` | Convert SpotMatrix.csv to HDF5 spots format for downstream analysis. |
| `GFF2Grains.py` | Convert GrainSpotter `.gff` output to MIDAS `Grains.csv` format. |
| `mergePeaks.py` | Merge peak lists from multiple frames or scans. |
| `extractPeaks.py` | Extract individual peak data from the spot table for inspection. |
| `peak_sigma_statistics.py` | Compute and report peak width (σ) statistics from FF-HEDM fitting results. |
| `processFilesParallel.py` | Parallel file processing utility for batch FF-HEDM operations. |
| `extract_grains_mic.py` | Extract grain data from `.mic` files for cross-referencing with FF-HEDM results. |

## NF-HEDM Post-Processing

| Script | Description |
|--------|-------------|
| `NFGrainCentroids.py` | Calculate grain centroids from NF-HEDM `.mic` reconstruction files. |
| `nf_mic_to_grains.py` | Convert NF-HEDM `.mic` files to `Grains.csv` format for comparison with FF-HEDM. |
| `nf_neighbor_calc.py` | Compute grain neighbor relationships and boundary statistics from NF reconstructions. |
| `nf_paraview_gen.py` | Generate ParaView-compatible VTK files from NF-HEDM reconstructions for 3D visualization. |
| `findUniqueOrientationsNF.py` | Identify unique grain orientations from NF reconstruction data. |
| `hdf_gen_nf.py` | Generate HDF5 files from NF-HEDM inputs and outputs. |
| `nf_manipulation_codes.py` | Utility functions for manipulating NF-HEDM `.mic` data. |

## Visualization

All visualization/plotting scripts have moved to `gui/viewers/`. See `gui/viewers/` for:
`interactiveFFplotting.py`, `plotFFSpots3d.py`, `plotGrains3d.py`, `PlotFFNF.py`,
`pfIntensityViewer.py`, `viz_caking.py`, `plot_integrator_peaks.py`,
`plot_calibrant_results.py`, `plot_lineout_results.py`, `plot_lineout_comparison.py`,
`live_viewer.py`, `peak_sigma_statistics.py`.

Data processing scripts that remain here:

| Script | Description |
|--------|-------------|
| `extract_lineouts.py` | **Batch lineout extraction.** Runs `IntegratorZarrOMP` in direct mode with SNIP background, SavGol peak detection, and multiplet fitting. |
| `sino_cleanup_tomo.py` | **Sinogram cleanup and tomo reconstruction.** See [Tomography_Reconstruction](../manuals/Tomography_Reconstruction.md). |

## Simulation & Testing

Simulation scripts remain here. Test scripts have moved to `tests/`.

| Script | Description |
|--------|-------------|
| `simulatePeaks.py` | Simulate diffraction peak positions for testing and validation. |
| `sim_ff_transformed.py` | Generate transformed FF-HEDM simulation data for testing deformation workflows. |
| `compressedSimulationReader.py` | Read compressed forward simulation output files. |

See `tests/` for: `test_ff_hedm.py`, `test_nf_hedm.py`, `test_ff_calibration.py`,
`test_integrator_peaks.py`, `test_phase_id.py`, `test_live_viewer.py`.

## Scanning / Point-Focus HEDM

| Script | Description |
|--------|-------------|
| `runScanning.py` | Driver for scanning (point-focus) HEDM experiments. |
| `evalScanning.py` | Evaluate scanning HEDM results and compute summary statistics. |

## Specialized Tools

| Script | Description |
|--------|-------------|
| `MIDAS_dig_tw.py` | Digital twin workflow for in situ experiments — couples MIDAS with simulation. |
| `ff_dig_tw.py` | FF-HEDM digital twin: forward model + comparison with experimental data. |
| `blobPeaksearch.py` | Blob-based peak search for alternative peak detection on raw detector images. |
| `BatchCake.py` | Batch radial integration using the caking engine. |
| `batchImages.py` | Batch image processing (dark subtraction, normalization). |
| `DL2FF.py` | Convert deep-learning peak predictions to FF-HEDM input format. |
| `run_full_images_ff.py` | Process full detector images through the FF-HEDM pipeline. |
| `ff_peaks_raw_images.py` | Extract peaks from raw images for diagnostic purposes. |
| `undistort_image.py` | Apply spatial distortion correction to a raw detector image using calibrated parameters. |
| `generate_mask.py` | **Mask generator.** Create uint8 TIFF mask files for detector pixel masking (convention: 0=valid, 1=masked). Supports Dioptas `.mask` format input. Used with `MaskFile` parameter. |

## Deprecated

| Script | Description |
|--------|-------------|
| `deprecated_AutoCalibrate.py` | Older auto-calibration script (superseded by `AutoCalibrateZarr.py`). |
