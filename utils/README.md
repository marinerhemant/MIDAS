# utils/ — Python Utilities for MIDAS

This directory contains Python scripts and tools used across the MIDAS analysis pipeline. Scripts fall into several categories: workflow drivers, data conversion, visualization, calibration, and analysis.

---

## Core Workflow & Calibration

| Script | Description |
|--------|-------------|
| `AutoCalibrateZarr.py` | **FF-HEDM auto-calibration.** Determines detector tilt angles, beam center, and sample-to-detector distance from powder calibrant diffraction rings. Reads Zarr-ZIP data directly. |
| `integrator.py` | **Radial integrator (caking).** Converts 2D detector images to 1D intensity vs. 2θ profiles. Supports multiple detector geometries and azimuthal wedges. |
| `integrator_batch_process.py` | Batch-mode radial integration for processing large datasets non-interactively. |
| `integrator_server.py` | Long-running integration server for real-time data reduction during beamline operation. |
| `integrator_stream_process_h5.py` | Streaming radial integration from HDF5 files as they are written during acquisition. |
| `integrate_and_refine.py` | Combined radial integration + GSAS-II peakfit refinement pipeline. |
| `gsas_ii_refine.py` | **GSAS-II integration.** Imports caked 1D profiles into GSAS-II for powder diffraction refinement. |
| `midas_config.py` | MIDAS configuration and environment variable management. |

## Grain Matching & Analysis

| Script | Description |
|--------|-------------|
| `match_grains.py` | **Grain matching and layer stitching.** Matches grains across load states (Hungarian or greedy) and stitches multi-layer scans. Supports affine deformation transforms, consolidated HDF5 input, and library-callable API. See [FF_Match_Stack_Reconstructions manual](../manuals/FF_Match_Stack_Reconstructions.md). |
| `calcMiso.py` | **Crystallographic misorientation calculations.** Quaternion symmetry operations, orientation matrix ↔ quaternion conversions, misorientation angles for all 230 space groups. Used as a library by `match_grains.py` and other tools. |

## Data Conversion & Format Utilities

| Script | Description |
|--------|-------------|
| `ffGenerateZip.py` | Convert raw detector data (HDF5 / GE / TIFF) into MIDAS Zarr-ZIP archives for FF-HEDM analysis. Handles dark subtraction, data reshaping, and metadata embedding. |
| `ffGenerateZipRefactor.py` | Refactored version of `ffGenerateZip.py` with improved performance and memory management. |
| `updateZarrDset.py` | Update individual datasets or metadata within an existing Zarr-ZIP archive without full reprocessing. |
| `updateZarrDsetRefactor.py` | Refactored Zarr dataset updater with support for additional operations. |
| `GE2Tiff.py` | Convert GE detector binary files to TIFF format. |
| `esrf2zip_pf.py` | Convert ESRF-format data to MIDAS Zarr-ZIP for point-focus HEDM. |
| `esrf_to_ge.py` | Convert ESRF-format detector images to GE binary format. |
| `mergeGEfiles.py` | Merge multiple GE binary files into a single file. |
| `mergeH5s.py` | Merge multiple HDF5 files into one. |
| `midas2zip.py` | Convert older MIDAS output formats to Zarr-ZIP archives. |
| `ang2mic.py` | Convert `.ang` (EBSD-style) files to `.mic` (MIDAS microstructure) format. |
| `ConvTiffToGE.c` | (Compiled separately) Convert TIFF images to GE binary format. |

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

| Script | Description |
|--------|-------------|
| `interactiveFFplotting.py` | **Interactive FF-HEDM browser app.** Dash-based visualization of grains, spots, and raw detector images. See [FF_Interactive_Plotting manual](../manuals/FF_Interactive_Plotting.md). |
| `plotFFSpots3d.py` | 3D scatter plot of FF-HEDM diffraction spots (Eta, Omega, 2Theta). |
| `plotFFSpots3dGrains.py` | 3D scatter plot of FF-HEDM spots, color-coded by grain assignment. |
| `plotGrains3d.py` | 3D scatter plot of grain centroids with orientation coloring. |
| `PlotFFNF.py` | Overlay FF-HEDM grain centroids on NF-HEDM orientation maps. |
| `pfIntensityViewer.py` | Interactive viewer for point-focus / scanning HEDM intensity data. |
| `viz_caking.py` | Visualize radial integration (caking) results as 2D plots. |
| `vtkSimExportBin.py` | Export simulation results to VTK binary format for ParaView. |

## Simulation & Testing

| Script | Description |
|--------|-------------|
| `simulatePeaks.py` | Simulate diffraction peak positions for testing and validation. |
| `sim_ff_transformed.py` | Generate transformed FF-HEDM simulation data for testing deformation workflows. |
| `compressedSimulationReader.py` | Read compressed forward simulation output files. |
| `test_ff_hedm.py` | **FF-HEDM benchmark test.** End-to-end test using simulated data to validate the full FF-HEDM pipeline (simulation → indexing → regression comparison). Includes automatic cleanup of generated files. |
| `test_nf_hedm.py` | **NF-HEDM benchmark test.** End-to-end test: runs `simulateNF`, reconstructs via `nf_MIDAS.py`, and compares orientations against a reference `.mic` file. |
| `test_ff_calibration.py` | **FF-HEDM calibration benchmark.** Runs `CalibrantPanelShiftsOMP` on example CeO2 data and validates mean strain ≤ threshold. See [FF_calibration manual §9](../manuals/FF_Calibration.md). |

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

## Deprecated

| Script | Description |
|--------|-------------|
| `deprecated_AutoCalibrate.py` | Older auto-calibration script (superseded by `AutoCalibrateZarr.py`). |
