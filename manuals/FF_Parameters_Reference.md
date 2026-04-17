# FF-HEDM Parameter Reference

Authoritative list of every key the MIDAS FF-HEDM pipeline reads from a text
parameter file. Defaults, types, units, and read-by paths are verified against
source:

- [FF_HEDM/src/MIDAS_ParamParser.c](../FF_HEDM/src/MIDAS_ParamParser.c) — the
  shared C parser (`midas_config_defaults`, `midas_parse_params`)
- [utils/ffGenerateZipRefactor.py](../utils/ffGenerateZipRefactor.py) — copies
  text params into the Zarr analysis file consumed by the indexing/fitting
  executables (`FORCE_DOUBLE_PARAMS`, `FORCE_INT_PARAMS`, `FORCE_STRING_PARAMS`,
  `RENAME_MAP`)
- [FF_HEDM/workflows/ff_MIDAS.py](../FF_HEDM/workflows/ff_MIDAS.py) — Python
  driver, reads a small set of keys directly

---

## How the parameter file is consumed

```
   Parameters.txt
        │
        ├─→ ff_MIDAS.py         (Python direct reads: file discovery, layer loop)
        │
        ├─→ ffGenerateZipRefactor.py
        │        │
        │        └─→ analysis.zip (Zarr)  ← most FF C executables read FROM HERE
        │                                  (IndexerOMP, FitPosOrStrainsOMP, etc.)
        │
        └─→ MIDAS_ParamParser.c
                 └─→ MIDASConfig struct   ← CalibrantPanelShiftsOMP,
                                            IntegratorZarrOMP, some utilities
```

Key implication: most "indexing" parameters (`Completeness`, `StepSizePos`,
`MarginRadial`, `Vsample`, etc.) are **not** read by any C file directly —
they ride through `ffGenerateZipRefactor.py` into the Zarr, which the
executables then open. Some keys are renamed on the way in (see
[Rename map](#zarr-rename-map)).

---

## Conventions

- **Units**: µm (lengths), deg (angles, except lattice α/β/γ which are also deg),
  Å (wavelength, lattice a/b/c), pixels (detector coords), counts (intensity),
  s (time), dimensionless fraction (0–1).
- **Default** = value set by `midas_config_defaults()` in
  `MIDAS_ParamParser.c`, or the value implied by
  `ffGenerateZipRefactor.py`. "—" means no default; user must set.
- **Required**: *yes* if the pipeline crashes or produces nonsense without it;
  *no* otherwise.
- **Stage** = where the value is consumed: `file-discovery`, `integration`,
  `peak-search`, `indexing`, `refinement`, `forward-sim`, `calibration`,
  `multi-panel`.

---

## 1. Data file discovery

These keys tell the pipeline where to find raw frames and how to name them.
Read primarily by `ff_MIDAS.py` and `ffGenerateZipRefactor.py`.

| Key                       | Type | Units   | Default  | Required | Stage          | Notes |
|---------------------------|------|---------|----------|----------|----------------|-------|
| `RawFolder`               | str  | path    | —        | yes      | file-discovery | Directory holding raw frame files. |
| `Folder`                  | str  | path    | —        | no       | file-discovery | Generic folder; some legacy executables look here. |
| `FileStem`                | str  | prefix  | —        | yes      | file-discovery | Filename stem before the zero-padded number. |
| `Ext`                     | str  | ext     | —        | yes      | file-discovery | Extension with leading dot (`.ge3`, `.tif`, `.h5`). |
| `Padding`                 | int  | digits  | 6        | no       | file-discovery | Zero-pad width (e.g. 6 → `_000001`). |
| `StartNr`                 | int  | —       | 0        | yes      | file-discovery | First file number (global). |
| `EndNr`                   | int  | —       | 0        | yes      | file-discovery | Last file number (global). |
| `StartFileNrFirstLayer`   | int  | —       | 1        | no       | file-discovery | Starting file number for layer 1 (multi-layer runs). |
| `NrFilesPerSweep`         | int  | count   | 1        | no       | file-discovery | Files per ω sweep (multi-wedge scans). |
| `Dark`                    | str  | path    | `""`     | no       | peak-search    | Dark/background frame (basename or full path). |
| `SkipFrame`               | int  | count   | 0        | no       | file-discovery | Frames to skip at start of each sweep. |
| `HeadSize`                | int  | bytes   | 0 (→8192 if `DataType=1`) | no | file-discovery | Header to skip in raw binary. Auto-set to 8192 for GE. |
| `DataType`                | int  | code    | 1        | no       | file-discovery | 1 = GE uint16 with 8192 header; other values for Pilatus/Dexela/etc. |
| `ScanStep`                | float | µm     | —        | no       | file-discovery | Translation step between scan points (PF/multi-scan). |

## 2. HDF5 / Zarr dataset names

For HDF5/Zarr data sources (non-raw-frame inputs).

| Key            | Type | Default           | Notes |
|----------------|------|-------------------|-------|
| `darkDataset`  | str  | `exchange/dark`   | HDF5 path for dark frames. Alias: `darkLoc`. |
| `dataDataset`  | str  | `exchange/data`   | HDF5 path for data frames. Alias: `dataLoc`. |

## 3. Detector geometry

| Key         | Type   | Units  | Default | Required | Notes |
|-------------|--------|--------|---------|----------|-------|
| `NrPixels`  | int    | pixels | —       | yes      | Shortcut: sets both `NrPixelsY` and `NrPixelsZ`. |
| `NrPixelsY` | int    | pixels | —       | yes      | Horizontal pixels. |
| `NrPixelsZ` | int    | pixels | —       | yes      | Vertical pixels. |
| `px`        | double | µm     | —       | yes      | Pixel size. Renamed to `PixelSize` in Zarr. |
| `Lsd`       | double | µm     | 0       | yes      | Sample-to-detector distance. Alias: `Distance`. |
| `BC`        | 2×double | pixels | 0 0   | yes      | Beam center `Y_px Z_px`. Split into `YCen`/`ZCen` in Zarr. |
| `tx`        | double | deg    | 0       | no       | Detector rotation about X-ray axis. |
| `ty`        | double | deg    | 0       | no       | Detector rotation about horizontal axis. |
| `tz`        | double | deg    | 0       | no       | Detector rotation about vertical axis. |
| `p0`–`p14`  | double | varies | 0 (see below) | no | Distortion coefficients. `p3` default=45, `p6`=90, `p8/p10/p12/p14`=180 (these are periodicity constants, not user-tuned — see `midas_apply_tol_defaults`). |
| `Wedge`     | double | deg    | 0       | no       | Deviation from 90° between rotation axis and beam. |
| `RhoD`      | double | µm     | 0       | no       | Max ring radius for distortion model. Alias: `MaxRingRad`. |
| `Parallax`  | double | —      | 0       | no       | Parallax correction term. |
| `ResidualCorrectionMap` | str | path | `""` | no     | Residual distortion correction map file. |

## 4. Crystallography / material

| Key                | Type     | Units | Default | Required | Notes |
|--------------------|----------|-------|---------|----------|-------|
| `LatticeConstant`  | 6×double | Å, deg | `0 0 0 0 0 0` | yes | `a b c α β γ`. Alias: `LatticeParameter`. Renamed to `LatticeParameter` in Zarr. |
| `SpaceGroup`       | int      | —     | 225     | yes      | Space group number. |
| `Wavelength`       | double   | Å     | 0       | yes      | X-ray wavelength. |
| `NumPhases`        | int      | count | 1       | no       | Total phases in sample. |
| `PhaseNr`          | int      | —     | 0       | no       | Phase currently being analyzed. |

## 5. Omega scan and analysis ranges

| Key                     | Type     | Units | Default  | Required | Notes |
|-------------------------|----------|-------|----------|----------|-------|
| `OmegaStart`            | double   | deg   | 0        | yes      | First frame ω. |
| `OmegaEnd`              | double   | deg   | 0        | yes      | Last frame ω. |
| `OmegaStep`             | double   | deg   | 0        | yes      | Δω per frame (negative = reverse). Renamed to `step` in Zarr. |
| `OmegaRange`            | 2×double | deg   | —        | no, multi | Analysis window. Repeatable → stacks into `OmegaRanges`. |
| `BoxSize`               | 4×double | µm    | —        | no, multi | `Ymin Ymax Zmin Zmax` virtual detector box. Repeatable → stacks into `BoxSizes`. Use huge values (e.g. ±1e6) to disable. |
| `MinOmeSpotIDsToIndex`  | double   | deg   | —        | no       | ω lower bound for spots used as indexing seeds. |
| `MaxOmeSpotIDsToIndex`  | double   | deg   | —        | no       | ω upper bound for indexing seeds. |
| `OmeBinSize`            | double   | deg   | 0        | no       | ω LUT bin size. Typical: 0.1. |
| `OmegaSigma`            | double   | deg   | 0        | no       | Simulated peak broadening in ω (forward-sim only). |

## 6. Ring selection

Ring indices refer to the `hkls.csv` output of `GetHKLList`/`GetHKLListZarr`.

| Key                   | Type | Units | Default | Required | Notes |
|-----------------------|------|-------|---------|----------|-------|
| `RingThresh`          | int int | ring_nr, counts | — | yes, multi | `RingThresh <ring_nr> <intensity_threshold>`. One line per ring used. |
| `RingsToExclude`      | int  | ring_nr | —     | no, multi | Rings to exclude from analysis. Alias: `RingsToReject`. |
| `RingNumbers`         | int  | ring_nr | —     | no, multi | Explicit ring list (alternate form). |
| `RingRadii`           | double | Å    | —     | no, multi | Corresponding radii (must match `RingNumbers` count). |
| `MaxRingNumber`       | int  | —     | 0       | no       | Upper cap on ring numbers to consider. |
| `OverAllRingToIndex`  | int  | ring_nr | 1     | yes (indexing) | Ring used to generate candidate orientations. Prefer low-multiplicity. Renamed to `OverallRingToIndex` (double L) in Zarr. |

## 7. Image transforms and masking

| Key                 | Type | Units | Default | Notes |
|---------------------|------|-------|---------|-------|
| `ImTransOpt`        | int  | code  | —, multi | Image transform code. Repeatable; values accumulate into `TransOpt[]`. 0 = no transform. |
| `MaskFile`          | str  | path  | `""`    | Binary mask (1=data, 0=masked). Also sets `useMask=1` for forward-sim. |
| `GapFile`           | str  | path  | `""`    | Gap / dead-zone mask. |
| `BadPxFile`         | str  | path  | `""`    | Bad-pixel mask. |
| `GapIntensity`      | int64 | counts | 0     | Fill value for gap pixels. |
| `BadPxIntensity`    | int64 | counts | 0     | Fill value for bad pixels. |

## 8. Peak search and integration

Most of these are calibration/integrator keys but a few apply to peak fitting.

| Key                          | Type | Units | Default | Notes |
|------------------------------|------|-------|---------|-------|
| `Width`                      | double | µm | 1500  | Ring half-width in radial (2θ) direction. |
| `EtaBinSize`                 | double | deg | 0    | η LUT bin size. Typical: 0.1. |
| `RBinDivisions`              | int  | —   | 4       | Radial bin divisions for fitting. |
| `MultFactor`                 | double | —  | 0      | Outlier detection multiplier. |
| `MinIndicesForFit`           | int  | count | 1     | Minimum matched indices to accept fit. |
| `FitOrWeightedMean`          | int  | code | 0      | 0 = fit, 1 = weighted mean. |
| `nIterations`                | int  | count | 1     | Calibration refinement iterations. |
| `IterOffset`                 | int  | —   | 0       | Iteration numbering offset. |
| `DoubletSeparation`          | double | µm | 0     | Minimum separation for resolving doublets. |
| `NormalizeRingWeights`       | int  | bool | 0      | Normalize fit weights across rings. |
| `OutlierIterations`          | int  | count | 1     | Outlier removal passes. |
| `RemoveOutliersBetweenIters` | int  | bool | 0      | Remove outliers between iterations. |
| `ReFitPeaks`                 | int  | bool | 0      | Re-fit after initial pass. |
| `TrimmedMeanFraction`        | double | frac | 1.0  | Fraction kept in trimmed mean. |
| `WeightByRadius`             | int  | bool | 0      | Weight by radius. |
| `WeightByFitSNR`             | int  | bool | 0      | Weight by fit SNR. |
| `WeightByPositionUncertainty`| int  | bool | 0      | Weight by position uncertainty. |
| `AdaptiveEtaBins`            | int  | bool | 0      | Adapt η bins to spot density. |
| `L2Objective`                | int  | bool | 0      | Use L2 norm instead of L1. |
| `PeakFitMode`                | int  | code | 0      | 0 = pseudo-Voigt, 1 = GSAS-II TCH. |
| `SubPixelLevel`              | int  | —   | 1       | Sub-pixel splitting at cardinal η (1=off, 4=default). |
| `SubPixelCardinalWidth`      | double | deg | 5.0  | Half-width for cardinal η sub-pixel splitting. |
| `ConvergenceThresholdPPM`    | double | ppm | 0    | Early-stop Δstrain threshold (0=disabled). |
| `SkipVerification`           | int  | bool | 0      | Skip final verification E-step. |
| `RingDiagnosticsCSV`         | str  | path | `""`   | Output path for ring-wise diagnostics. |
| `ResumeFromCheckpoint`       | int  | bool | 0      | Resume calibration from checkpoint. |
| `GradientCorrection`         | int  | bool | 0      | Apply beam gradient correction. |
| `UpperBoundThreshold`        | int  | counts | —    | Saturation cap; pixels above this are ignored in peak search. |

## 9. Indexing

These are passed to `IndexerOMP` via the Zarr analysis file (not via
`MIDAS_ParamParser.c`).

| Key                  | Type   | Units    | Default | Required | Notes |
|----------------------|--------|----------|---------|----------|-------|
| `Completeness`       | double | fraction | —       | yes      | Minimum match fraction to accept a grain. Typical: 0.8. Renamed to `MinMatchesToAcceptFrac` in Zarr. |
| `MinNrSpots`         | int    | count    | 1       | yes      | Minimum unique solutions before confirming grain. Typical: 3. |
| `MinMatchesToAcceptFrac` | double | fraction | —   | no       | Direct name used inside Zarr; set via `Completeness`. |
| `UseFriedelPairs`    | int    | bool     | 1       | no       | Use Friedel-pair speedup. Typical: 1. |
| `StepSizeOrient`     | double | deg      | 0       | yes      | Orientation search step. Typical: 0.2. Alias: `StepsizeOrient`. |
| `StepSizePos`        | double | µm       | —       | yes      | Position search step. Typical: 100. |
| `MarginOme`          | double | deg      | 0       | no       | ω tolerance for spot matching. Typical: 0.5. |
| `MarginEta`          | double | deg      | 0       | no       | η tolerance (deg) for spot matching. Typical: 500 (µm in some flows — verify per-executable). |
| `MarginRadial`       | double | µm       | —       | no       | 2θ (radial) spot-matching tolerance. |
| `MarginRadius`       | double | µm       | —       | no       | Equivalent grain radius filter. |
| `MinEta`             | double | deg      | 0       | no       | Azimuthal pole exclusion. Typical: 6. Alias: `ExcludePoleAngle`. |
| `MinConfidence`      | double | fraction | 0.5     | no       | Minimum confidence threshold. |
| `MinFracAccept`      | double | fraction | 0.5     | no       | Minimum acceptance fraction. |

## 10. Sample geometry and beam

| Key              | Type   | Units   | Default | Notes |
|------------------|--------|---------|---------|-------|
| `Rsample`        | double | µm      | 0       | Horizontal sample radius (grain positions limited to ±Rsample). |
| `Hbeam`          | double | µm      | 0       | Beam vertical size (grain positions limited to ±Hbeam/2). |
| `Vsample`        | double | µm³     | —       | Illuminated volume. Used by `CalcRadiusAllZarr`. |
| `BeamThickness`  | double | µm      | 0       | Beam vertical thickness. |
| `BeamSize`       | double | µm      | 0       | Horizontal beam size (PF-HEDM). |
| `GlobalPosition` | double | µm      | 0       | Sample starting position along beam. |
| `DiscModel`      | int    | code    | 0       | 0 = parallel beam, 1 = focused beam. |
| `DiscArea`       | double | µm²     | —       | Illuminated area for focused beam (`DiscModel=1`). |
| `tInt`           | double | s       | —       | Integration time per frame. |
| `tGap`           | double | s       | —       | Dead time between frames. |

## 11. Refinement — control flags

| Key                      | Type   | Units | Default | Notes |
|--------------------------|--------|-------|---------|-------|
| `MargABC`                | double | %     | 0.3     | Lattice `a,b,c` refinement tolerance. |
| `MargABG`                | double | %     | 0.3     | Lattice `α,β,γ` refinement tolerance. |
| `FitAllAtOnce`           | int    | bool  | 0       | Fit all grains simultaneously vs. sequentially. |
| `DoDynamicReassignment`  | int    | bool  | 0       | Dynamically reassign spots during refinement. |
| `TakeGrainMax`           | int    | bool  | 0       | Twin-analysis: take max-solution grain. |
| `LocalMaximaOnly`        | int    | bool  | 0       | Use only local maxima in peak detection (forces `doPeakFit=0`). |
| `Twins`                  | int    | bool  | 0       | Enable twin analysis. |
| `GBAngle`                | double | deg   | 0       | Grain-boundary angle tolerance (twin analysis). |
| `DebugMode`              | int    | bool  | 0       | Verbose diagnostic output. |
| `WeightMask`             | double | frac  | 0       | Weighting for masked regions. |
| `WeightFitRMSE`          | double | frac  | 0       | Weighting by fit RMSE. |

## 12. Refinement — tolerances (calibration)

Used by `CalibrantPanelShiftsOMP` and related calibration executables. All
default to 0 unless noted. `tolP0…tolP14` fall back to `tolP` if unset (see
`midas_apply_tol_defaults`).

| Key                 | Type   | Units | Default | Notes |
|---------------------|--------|-------|---------|-------|
| `tolTilts`          | double | deg   | 0       | Fallback for `tx/ty/tz` tolerances. |
| `tolTiltX`          | double | deg   | 0       | Individual `tx` tolerance. |
| `tolBC`             | double | pixels | 0      | Beam center refinement tolerance. |
| `tolLsd`            | double | µm    | 0       | Sample-detector distance tolerance. |
| `tolLsdPanel`       | double | µm    | 100     | Per-panel Lsd tolerance. |
| `tolP`              | double | —     | 0       | General distortion-coefficient fallback. |
| `tolP0`–`tolP14`    | double | varies | 0/45/90/180 | Per-coefficient tolerances; see `midas_apply_tol_defaults`. |
| `tolP2Panel`        | double | —     | 0.0001  | Per-panel `p2` tolerance. |
| `tolShifts`         | double | µm    | 1.0     | Panel shift refinement tolerance. |
| `tolRotation`       | double | deg   | 0       | Panel rotation tolerance. |
| `tolWavelength`     | double | Å     | 0.001   | Wavelength refinement tolerance. |
| `tolParallax`       | double | —     | 0       | Parallax tolerance. |
| `FitWavelength`     | int    | bool  | 0       | Refine wavelength. |
| `FitParallax`       | int    | bool  | 0       | Refine parallax. |
| `PerPanelLsd`       | int    | bool  | 0       | Independent Lsd per panel. |
| `PerPanelDistortion` | int   | bool  | 0       | Independent distortion per panel. |
| `FixPanelID`        | int    | panel | 0       | Hold this panel fixed (don't refine). |

## 13. Image processing (peak search filters)

Mostly NF-HEDM inheritance but available for FF peak search.

| Key                 | Type   | Units  | Default | Notes |
|---------------------|--------|--------|---------|-------|
| `Deblur`            | int    | bool   | 0       | Deblur overlapping peaks. |
| `DoLoGFilter`       | int    | bool   | 0       | Laplacian-of-Gaussian filter. |
| `GaussFiltRadius`   | int    | pixels | 0       | Gaussian filter radius. |
| `GaussWidth`        | double | pixels | 0       | Gaussian width (also used in forward-sim). |
| `LoGMaskRadius`     | int    | pixels | 0       | LoG mask radius. |
| `MedFiltRadius`     | int    | pixels | 0       | Median filter radius. |
| `BlanketSubtraction` | int   | bool   | 0       | Blanket background subtraction. |
| `SkipImageBinning`  | int    | bool   | 0       | Skip image binning step. |

## 14. Multi-panel detector

| Key                  | Type | Units | Default | Notes |
|----------------------|------|-------|---------|-------|
| `NPanelsY`           | int  | count | 0       | Panels in Y direction. |
| `NPanelsZ`           | int  | count | 0       | Panels in Z direction. |
| `PanelSizeY`         | int  | pixels | 0      | Pixels per panel in Y. |
| `PanelSizeZ`         | int  | pixels | 0      | Pixels per panel in Z. |
| `PanelGapsY`         | up to 10×int | pixels | 0 | Gap widths between Y-panels (space-separated on one line). |
| `PanelGapsZ`         | up to 10×int | pixels | 0 | Gap widths between Z-panels. |
| `PanelShiftsFile`    | str  | path  | `""`    | Per-panel geometry corrections file. |
| `BigDetSize`         | int  | pixels | 0      | Size of assembled detector (multi-detector mode). |
| `DetParams`          | up to 10×10×double | varies | 0 | Per-detector geometry matrix (repeatable). |

## 15. Multi-scan / multi-distance

| Key                   | Type | Units | Default | Notes |
|-----------------------|------|-------|---------|-------|
| `nScans`              | int  | count | 0       | Number of scan positions (1=FF, >1=PF-HEDM with `positions.csv`). |
| `nDistances`          | int  | count | 0       | Number of sample-detector distances (NF). |
| `NrFilesPerDistance`  | int  | count | 1       | Files per distance step. |
| `LsdMean`             | double | µm  | 0       | Mean Lsd for NF grids. |

## 16. Output paths

| Key              | Type | Default | Notes |
|------------------|------|---------|-------|
| `OutputFolder`   | str  | `""`    | Intermediate analysis output (`Temp/`, `Output/`). |
| `ResultFolder`   | str  | `""`    | Final results (`Grains.csv`, `Results/`). |
| `OutDirPath`     | str  | `""`    | Forward-sim output directory. |
| `DataDirectory`  | str  | `""`    | Generic data directory used by some flows. |

## 17. Seeded / tracked workflows

| Key                  | Type | Default | Notes |
|----------------------|------|---------|-------|
| `GrainsFile`         | str  | `""`    | Input `Grains.csv` for refinement/tracking. |
| `SeedOrientations`   | str  | `""`    | Seed/hint orientations file. |
| `RefinementFileName` | str  | `""`    | Refinement input file. Alias: `InFileName`. |
| `InputAllExtraInfoFittingAll` | str | `""` | Pre-computed spots input (used with `-provideInputAll 1`). |

## 18. Forward simulation only

Read by `ForwardSimulationCompressed` and related. **Not required** for
analyzing real experimental data.

| Key                   | Type   | Units  | Default | Notes |
|-----------------------|--------|--------|---------|-------|
| `InFileName`          | str    | path   | `""`    | Input grain orientations file. |
| `OutFileName`         | str    | stem   | `""`    | Output file stem (produces `{stem}_scanNr_0.zip`). |
| `IntensitiesFile`     | str    | path   | `""`    | Per-grain intensities file. |
| `PeakIntensity`       | double | counts | 2000    | Peak amplitude. |
| `MaxOutputIntensity`  | double | counts | 65000   | Saturation cap. |
| `GaussWidth`          | double | pixels | 0       | Gaussian width of simulated spots. |
| `SimNoiseSigma`       | double | frac   | 0       | Gaussian noise level. |
| `WriteSpots`          | int    | bool   | 1       | Write `SpotMatrixGen.csv`. |
| `WriteImage`          | int    | bool   | 1       | Write simulated detector images. |
| `IsBinary`            | int    | bool   | 0       | Input grain file is binary. |
| `LoadNr`              | int    | —      | 0       | Grain index to load from binary. |
| `UpdatedOrientations` | int    | bool   | 1       | Use refined orientations from prior run. |
| `NFOutput`            | int    | bool   | 0       | Output in NF stack format. |
| `NoSaveAll`           | int    | bool   | 0       | Skip writing all-spots file. |
| `WriteLegacyBin`      | int    | bool   | 0       | Write legacy binary output. |
| `SimulationBatches`   | int    | count  | 0       | Batches for large simulations. |
| `RingsToUse`          | int    | ring_nr | —, multi | Explicit rings to include. |
| `EResolution`         | double int | —, count | 0 1 | Energy resolution + `num_lambda_samples` on same line. |

## 19. NF-HEDM only (ignore for FF)

| Key                   | Type | Default | Notes |
|-----------------------|------|---------|-------|
| `NrOrientations`      | int  | 0       | Candidate orientations for NF indexing. |
| `OrientTol`           | double | 0     | NF orientation tolerance. |
| `GridSize`            | double | 0     | NF grid spacing. |
| `EdgeLength`          | double | 0     | Virtual voxel size. |
| `GridPoints`          | int  | 0       | Grid points per dimension. |
| `TopLayer`            | int  | 0       | Top layer index. |
| `GridFileName`        | str  | `""`    | Grid definition file. |

## 20. Lineout / calibration-only (unused in FF analysis)

These are consumed by `CalibrantIntegratorOMP` and `IntegratorZarrOMP` for
1D radial integration of calibrant data. **Not used by FF indexing/fitting.**

| Key              | Type   | Units | Default | Notes |
|------------------|--------|-------|---------|-------|
| `RMin`           | double | µm    | 10      | Min radius for integration. Stored as `lineoutRMin`. |
| `RMax`           | double | µm    | 0       | Max radius for integration. |
| `RBinSize`       | double | µm    | 0.25    | Radial bin size. Stored as `lineoutRBinSize`. |
| `EtaMin`         | double | deg   | —       | Min η for integration. Read by integrators only, not central parser. |
| `EtaMax`         | double | deg   | —       | Max η for integration. |
| `DoSmoothing`    | int    | bool  | —       | Smooth 1D lineout. Integrator-only. |
| `DoPeakFit`      | int    | bool  | —       | Fit peaks in 1D lineout. Integrator-only. |
| `MultiplePeaks`  | int    | bool  | —       | Allow multiple peaks per ring. |
| `PeakLocation`   | double | —     | —       | Expected peak location. Integrator-only. |

## 21. CLI arguments (ff_MIDAS.py)

Not in the parameter file — passed on the command line.

| Flag                  | Type | Default | Notes |
|-----------------------|------|---------|-------|
| `-resultFolder`       | str  | `""`    | Output directory. |
| `-paramFN`            | str  | `""`    | **Path to this parameter file.** |
| `-dataFN`             | str  | `""`    | Path to existing `.h5` / `.zip` (skips ffGenerateZip). |
| `-nCPUs`              | int  | 10      | Parallel workers. |
| `-machineName`        | str  | `local` | Execution target (`local`, `lcrc`, `purdue`, etc.). |
| `-numFrameChunks`     | int  | -1      | Chunk size for streaming large datasets. |
| `-preProcThresh`      | int  | -1      | Pre-processing threshold. |
| `-nNodes`             | int  | -1      | Nodes for distributed execution. |
| `-fileName`           | str  | `""`    | Override raw file name. |
| `-startLayerNr`       | int  | 1       | First layer to process. |
| `-endLayerNr`         | int  | 1       | Last layer to process. |
| `-convertFiles`       | int  | 1       | 1 = run ffGenerateZipRefactor. |
| `-peakSearchOnly`     | int  | 0       | Stop after peak search. |
| `-doPeakSearch`       | int  | 1       | 1 = run peak search; 0 = skip to indexing. |
| `-provideInputAll`    | int  | 0       | Skip peak search; use `InputAllExtraInfoFittingAll.csv`. |
| `-rawDir`             | str  | `""`    | Override `RawFolder`. |
| `-grainsFile`         | str  | `""`    | Override `GrainsFile`. |
| `-nfResultDir`        | str  | `""`    | NF seed directory. |
| `-reprocess`          | int  | 0       | Redo analysis on existing output. |
| `-batchMode`          | int  | 0       | Non-interactive mode. |
| `-resume`             | str  | `""`    | Resume from checkpoint. |
| `-restartFrom`        | str  | `""`    | Restart from named stage. |
| `-useGPU`             | int  | 0       | Use GPU executables where available. |
| `-generateH5`         | int  | 0       | Write analysis H5 after processing. |

## Zarr rename map

When `ffGenerateZipRefactor.py` copies text params into the Zarr analysis
file, some keys are renamed. Most C executables read the Zarr, not the text
file, so know the renamed name too.

| Text-file key          | Zarr dataset key              |
|------------------------|-------------------------------|
| `OmegaStep`            | `step`                        |
| `Completeness`         | `MinMatchesToAcceptFrac`      |
| `px`                   | `PixelSize`                   |
| `LatticeConstant`      | `LatticeParameter`            |
| `OverAllRingToIndex`   | `OverallRingToIndex`          |
| `resultFolder`         | `ResultFolder`                |
| `OmegaRange`           | `OmegaRanges` (2D array)      |
| `BoxSize`              | `BoxSizes` (2D array)         |
| `BC`                   | `YCen` + `ZCen` (split)       |

Unrecognized keys in the text file are silently skipped by both the C parser
and `ffGenerateZipRefactor.py` — there is no "unknown key" warning today.

## Defaults summary (verified against source)

The following defaults come directly from `midas_config_defaults()` in
[MIDAS_ParamParser.c](../FF_HEDM/src/MIDAS_ParamParser.c):

| Key                    | Default |
|------------------------|---------|
| `Padding`              | 6 |
| `DataType`             | 1 (GE, 8192-byte header) |
| `RBinDivisions`        | 4 |
| `tolShifts`            | 1.0 |
| `tolLsdPanel`          | 100 |
| `tolP2Panel`           | 0.0001 |
| `tolWavelength`        | 0.001 |
| `lineoutRBinSize`      | 0.25 |
| `lineoutRMin`          | 10.0 |
| `MargABC`              | 0.3 |
| `MargABG`              | 0.3 |
| `NumPhases`            | 1 |
| `NrFilesPerDistance`   | 1 |
| `MinFracAccept`        | 0.5 |
| `MinNrSpots`           | 1 |
| `SpaceGroup`           | 225 |
| `WriteSpots`           | 1 |
| `WriteImage`           | 1 |
| `UpdatedOrientations`  | 1 |
| `PeakIntensity`        | 2000.0 |
| `MaxOutputIntensity`   | 65000.0 |
| `SimNoiseSigma`        | 0.0 |
| `num_lambda_samples`   | 1 |
| `SubPixelLevel`        | 1 |
| `SubPixelCardinalWidth` | 5.0 |
| `Width`                | 1500 |
| `darkDataset`          | `exchange/dark` |
| `dataDataset`          | `exchange/data` |

Several values in [FF_HEDM/Example/Parameters.txt](../FF_HEDM/Example/Parameters.txt) are
recommended working values, **not** parser defaults — e.g. `StepSizeOrient 0.2`,
`StepSizePos 100`, `Completeness 0.8`, `MinNrSpots 3`, `MinEta 6`,
`MarginRadial 500`, `MarginEta 500`, `MarginOme 0.5`, `OmeBinSize 0.1`,
`EtaBinSize 0.1`. These are the numbers users should start from, not zeros.
