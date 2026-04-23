# NF-HEDM Parameter Reference

Authoritative list of every key the MIDAS NF-HEDM pipeline reads from a text
parameter file. Companion to [FF_Parameters_Reference.md](FF_Parameters_Reference.md).

## Architecture note (read this first)

Although `/Users/hsharma/opt/MIDAS/NF_HEDM/src/MIDAS_ParamParser.c` is
**byte-identical** to the FF copy, almost no NF executable uses it. Only
[GetHKLList.c](../NF_HEDM/src/GetHKLList.c) imports `midas_parse_params`.
Every other NF executable has its **own inline** `fgets`+`strncmp` parameter
loop, so the set of recognized keys and their defaults is per-executable.

This means:

- The shared-parser defaults from FF (`MargABC=0.3`, `SpaceGroup=225`, etc.) do
  **not** apply in NF unless that specific executable sets the same default
  inline.
- NF has several **multi-entry** keys that appear once per detector distance:
  `Lsd`, `BC`, `OmegaRange`, `BoxSize` — each line adds to an array indexed by
  distance. Order matters: the *i*th `Lsd` must match the *i*th `BC`.
- There is no Zarr-bridge step (no equivalent of `ffGenerateZipRefactor.py`).
  All executables read the text parameter file directly.

## How the parameter file is consumed

```
   Parameters.txt
        │
        ├─→ nf_MIDAS.py / nf_MIDAS_Multiple_Resolutions.py
        │       (Python direct reads: data paths, grid setup, seeding)
        │
        └─→ each NF C executable
              (inline param loop — own defaults per executable)
```

Data between executables flows through **binary files** (`SpotsInfo.bin`,
`DiffractionSpots.bin`, `Key.bin`, `OrientMat.bin`) mmap'd from
`DataDirectory`, not through a packaged Zarr.

## Conventions

- Units: µm (lengths), deg (angles), Å (wavelength, lattice a/b/c), pixels
  (detector coords), counts (intensity).
- `Default` comes from the C source's variable initialization. "—" means the
  executable expects it and will error or produce wrong output if missing.
- `Multi` means the key may appear multiple times and each line is appended
  to an array (typical count = `nDistances`).
- `Stage`: `file-discovery`, `image-preproc`, `seed-gen`, `spot-gen`,
  `indexing`, `calibration`, `forward-sim`, `post-analysis`.

---

## 1. Data paths and file discovery

| Key                | Type | Units | Default | Required | Stage          | Notes |
|--------------------|------|-------|---------|----------|----------------|-------|
| `DataDirectory`    | str  | path  | —       | yes      | all            | Input directory holding raw frames, reduced `.bin` files, and where `SpotsInfo.bin`/`DiffractionSpots.bin` live. Also mmap target. |
| `OutputDirectory`  | str  | path  | `DataDirectory` | no | all          | Where new outputs are written. Falls back to `DataDirectory`. |
| `OrigFileName`     | str  | stem  | —       | yes      | image-preproc  | Input raw-frame stem. `ProcessImagesCombined`/`MedianImageLibTiff` only. |
| `ReducedFileName`  | str  | stem  | —       | yes      | image-preproc + indexing | Output stem for median-subtracted/processed frames; then read by `FitOrientationOMP`, `MMapImageInfo`, `compareNF`. |
| `extOrig`          | str  | ext (no dot) | `tif` | no     | image-preproc  | Input file extension, e.g. `tif`. |
| `extReduced`       | str  | ext (no dot) | `bin` | no     | image-preproc  | Reduced output extension. |
| `RawStartNr`       | int  | —     | —       | yes      | image-preproc  | First raw frame number. |
| `StartNr`          | int  | —     | —       | yes      | indexing       | First frame number for NF mmap. |
| `EndNr`            | int  | —     | —       | yes      | indexing       | Last frame number for NF mmap. |
| `NrFilesPerDistance` | int | count | 1     | yes      | image-preproc  | Frames per detector distance. |
| `WFImages`         | int  | count | 0       | no       | image-preproc  | Number of wide-field calibration images per layer (skipped). |
| `logDir`           | str  | path  | `<resultFolder>/logs` | no | orchestration | Where `nf_MIDAS.py` writes per-layer logs. |
| `resultFolder`     | str  | path  | —       | yes      | orchestration  | Top-level output directory for the workflow driver. |

## 2. Detector geometry

| Key         | Type       | Units  | Default | Required | Notes |
|-------------|------------|--------|---------|----------|-------|
| `nDistances`| int        | count  | —       | yes      | Number of detector positions. Every multi-entry key below must appear exactly `nDistances` times. |
| `Lsd`       | double     | µm     | —       | yes, multi | Sample-to-detector distance. One line per distance, in order. |
| `BC`        | 2×double   | pixels | —       | yes, multi | Beam center `Y Z`. One line per distance. |
| `tx`        | double     | deg    | 0       | yes      | Detector rotation about beam axis. Shared across distances. |
| `ty`        | double     | deg    | 0       | yes      | Detector rotation about horizontal axis. |
| `tz`        | double     | deg    | 0       | yes      | Detector rotation about vertical axis. |
| `Wedge`     | double     | deg    | 0       | no       | Wedge angle. |
| `px`        | double     | µm     | —       | yes      | Pixel size (square). |
| `NrPixels`  | int        | pixels | 2048    | yes      | Square detector shortcut: sets both `NrPixelsY` and `NrPixelsZ`. |
| `NrPixelsY` | int        | pixels | `NrPixels` | no   | Detector Y pixel count. |
| `NrPixelsZ` | int        | pixels | `NrPixels` | no   | Detector Z pixel count. |
| `MaxRingRad`| double     | µm     | —       | yes      | Maximum ring radius to consider. Also aliased to `RhoD`. |

## 3. Crystallography

| Key                | Type       | Units | Default | Required | Notes |
|--------------------|------------|-------|---------|----------|-------|
| `LatticeParameter` | 6×double   | Å,deg | —       | yes      | `a b c α β γ`. Alias: `LatticeConstant`. |
| `Wavelength`       | double     | Å     | —       | yes      | X-ray wavelength. |
| `SpaceGroup`       | int        | —     | 225     | yes      | Space group number. Also read under name `SGNr` by some utilities (e.g. `ParseMic`). |
| `NumPhases`        | int        | count | 1       | no       | Number of phases in sample. |
| `PhaseNr`          | int        | —     | 1       | no       | Current phase being analyzed. |

## 4. Omega scan and integration bounds

NF reads `OmegaRange` and `BoxSize` as **multi-entry** keys — one line per
detector distance. Mismatched counts cause silent corruption.

| Key           | Type       | Units | Default | Required | Notes |
|---------------|------------|-------|---------|----------|-------|
| `OmegaStart`  | double     | deg   | —       | yes      | Starting ω. |
| `OmegaStep`   | double     | deg   | —       | yes      | Δω per frame (sign indicates direction). |
| `OmegaRange`  | 2×double   | deg   | —       | yes, multi | `ω_min ω_max` search range per distance. |
| `BoxSize`     | 4×double   | µm    | —       | yes, multi | `Y_min Y_max Z_min Z_max` detector clipping box per distance. |
| `ExcludePoleAngle` | double | deg   | 6       | no       | Azimuthal exclusion near poles. |

## 5. Sample geometry

| Key                       | Type   | Units | Default | Required | Notes |
|---------------------------|--------|-------|---------|----------|-------|
| `Rsample`                 | double | µm    | —       | yes      | Radius of inscribed circle in hexagonal grid. |
| `GridSize`                | double | µm    | —       | yes      | Voxel/triangle spacing. |
| `EdgeLength`              | double | µm    | `GridSize` | no    | Triangle edge length for mic grid. |
| `GridFileName`            | str    | path  | `grid.txt` | no    | Hexagonal grid file produced by `MakeHexGrid`. |
| `GridMask`                | 4×double | µm  | ±1000   | no       | `ymin ymax zmin zmax` grid bounding box. |
| `GlobalPosition`          | double | µm    | 0       | no       | Z position of current layer. |
| `GlobalPositionFirstLayer`| double | µm    | 0       | no       | Z position of layer 1 (multi-layer driver). |
| `LayerThickness`          | double | µm    | 0       | no       | Layer-to-layer Z step. |
| `TomoImage`               | str    | path  | `""`    | no       | Tomography reconstruction used to mask the grid. |
| `TomoPixelSize`           | double | µm    | —       | if `TomoImage` set | Pixel size of tomo image for scaling. |

## 5b. Step 0 — Image denoising (MIDAS-NF-preProc)

Optional self-supervised denoising of raw TIFF stacks performed **before**
`ProcessImagesCombined`. Provided by the standalone pip package
`MIDAS-NF-preProc` (`pip install MIDAS-NF-preProc`). When `Denoise=1`, the new
"step 0" stage writes denoised TIFFs to `{DataDirectory}/denoised/` and
re-points `DataDirectory` (in-memory and on-disk) for every downstream stage.
Raw data is never modified. See [NF_Denoising.md](NF_Denoising.md) for a
walkthrough.

| Key                  | Type   | Default                       | Notes |
|----------------------|--------|-------------------------------|-------|
| `Denoise`            | int    | `0`                           | Master switch. `0` = skip step 0 entirely; `1` = enable. |
| `DenoiseMethod`      | str    | `nlm`                         | `nlm` (scikit-image Non-Local Means, CPU, instant) or `n2v` (Noise2Void via CAREamics, **CUDA GPU required**). The workflow aborts at step 0 if `n2v` is selected and no GPU is detected. |
| `DenoisedDirectory`  | str    | `{DataDirectory}/denoised`    | Override output directory. |
| `DenoiseConfigFile`  | str    | —                             | Optional YAML config tuning N2V/NLM hyperparameters (see preProc README). |
| `DenoisePattern`     | str    | `*.tif`                       | Glob pattern for input frames. |
| `DenoiseTrainJointly`| int    | `0`                           | N2V-only: train one shared U-Net across the stack instead of one per image (faster prediction). |
| `DenoiseCheckpoint`  | str    | —                             | N2V-only: path to a `.ckpt`. When set, training is skipped and the checkpoint is used for prediction. |
| `DenoiseFinetune`    | int    | `0`                           | N2V-only: with `DenoiseCheckpoint`, fine-tune the model per image. |
| `DenoiseMaskThreshold` | double | —                           | If set, use the denoised image as a binary mask: pixels below the threshold are zeroed in the original. |
| `DenoiseNoMedian`    | int    | `0`                           | Disable the package's temporal pixel-wise median subtraction (default: enabled). |

**Resume / restart semantics**: `-restartFrom denoise` re-runs the denoise
step. Resuming from any later stage automatically re-points `DataDirectory` to
the existing `{DataDirectory}/denoised/` directory if a prior denoise output is
present.

## 6. Image processing (ProcessImagesCombined, MedianImageLibTiff)

| Key                | Type   | Units  | Default | Notes |
|--------------------|--------|--------|---------|-------|
| `BlanketSubtraction` | int  | counts | 0       | Flat DC offset subtracted from all pixels. |
| `MedFiltRadius`    | int    | pixels | 1       | Median filter radius. |
| `GaussFiltRadius`  | double | pixels | 4       | Gaussian filter σ. |
| `DoLoGFilter`      | int    | bool   | 0       | Enable Laplacian-of-Gaussian filter. |
| `LoGMaskRadius`    | int    | pixels | 10      | LoG mask radius (should be ≥ 2.5× `GaussFiltRadius`). |
| `Deblur`           | int    | bool   | 0       | Enable deblurring step (forces `WriteFinImage=1`). |
| `WriteFinImage`    | int    | bool   | 0       | Write uncompressed reduced images. |
| `WriteLegacyBin`   | int    | bool   | 0       | Write legacy per-frame `.bin` output (normally skipped). |
| `SkipImageBinning` | int    | bool   | 0       | Skip 2×2 binning in `MMapImageInfo`. |
| `PrecomputedSpotsInfo` | int | bool | 0       | `MMapImageInfo`: read existing `SpotsInfo.bin` rather than regenerate. |
| `Ice9Input`        | int    | bool   | 0       | Legacy Ice9-format compatibility flag. |

## 7. Orientation search (FitOrientationOMP / FitOrientationGPU)

| Key                    | Type   | Units    | Default | Required | Notes |
|------------------------|--------|----------|---------|----------|-------|
| `RingsToUse`           | int    | ring_nr  | —, multi | yes    | Rings to include. Repeatable. |
| `NrOrientations`       | int    | count    | —       | yes      | Number of candidate orientations in search space. |
| `SeedOrientations`     | str    | path     | —       | yes (indexing) | Seed orientations file (from FF or `GenSeedOrientationsFF2NFHEDM`). |
| `SeedOrientationsAll`  | str    | path     | —       | no       | Multi-resolution driver: union of seed sets across resolutions. |
| `OrientTol`            | double | deg      | 2       | no       | Optimization radius after first-pass candidates. |
| `MinFracAccept`        | double | fraction | 0.04    | no       | Minimum overlap fraction in candidate search. Typical: 0.1 seeded, 0.04 unseeded, 0.01 deformed. |
| `MinConfidence`        | double | fraction | 0.5     | no       | Threshold for filtering bad voxel solutions. |
| `SaveNSolutions`       | int    | count    | 1       | no       | Top-N solutions saved per voxel. |
| `NearestMisorientation`| int    | bool     | 0       | no       | Enforce nearest-neighbor misorientation constraint. |
| `MinMisoNSaves`        | double | deg      | 0       | no       | Minimum misorientation when `NearestMisorientation=1`. |

## 8. Calibration — FitOrientationParameters / MultiPoint

These keys are read **only** by `FitOrientationParameters.c` and
`FitOrientationParametersMultiPoint.c`. Exclude from the indexing wizard.

| Key               | Type     | Units  | Default | Notes |
|-------------------|----------|--------|---------|-------|
| `LsdTol`          | double   | µm     | —       | Lsd search range. |
| `LsdRelativeTol`  | double   | µm     | —       | Relative Lsd accuracy between distances. |
| `BCTol`           | 2×double | pixels | —       | Beam center tolerance `[y_tol z_tol]`. |
| `TiltsTol`        | double   | deg    | —       | Tilt angle tolerance. |
| `NumIterations`   | int      | count  | 3       | Calibration iterations (MultiPoint). |
| `GridPoints`      | 2×int    | count  | —       | Number of grid points `[nx ny]` to calibrate (MultiPoint). |

## 9. Output and post-processing

| Key              | Type | Units | Default | Notes |
|------------------|------|-------|---------|-------|
| `MicFileBinary`  | str  | path  | —       | Output binary `.mic` file with voxel orientations. Stale copy deleted on fresh run. |
| `MicFileText`    | str  | path  | —       | Text `.mic` file (`ParseMic` converts binary → text). |
| `OnlySpotsInfo`  | int  | bool  | 0       | `simulateNF`: skip image writing, produce `SpotsInfo.bin` only. |
| `SaveReducedOutput` | int | bool | 0      | `simulateNF`: also write reduced image files. |
| `WriteImage`     | int  | bool  | 1       | `simulateNF`: write simulated detector images. |
| `SimulationBatches` | int | count | 0 (auto) | `simulateNF`: memory-partition count for very large simulations. |

## 10. Grain post-analysis (Mic2GrainsList)

| Key           | Type | Units | Default | Notes |
|---------------|------|-------|---------|-------|
| `MaxAngle`    | double | deg | —       | Misorientation threshold for grain boundary detection. |
| `GBAngle`     | double | deg | 5.0     | Alias for grain-boundary tolerance (`ParseMic`). |

## 11. Multi-resolution workflow keys (nf_MIDAS_Multiple_Resolutions.py)

| Key               | Type | Units | Default | Notes |
|-------------------|------|-------|---------|-------|
| `GridRefactor`    | float | —    | 2       | Grid subdivision factor per resolution step. |
| `SeedOrientationsAll` | str | path | —      | Combined seed set across resolution levels. |
| `MinConfidence`   | double | frac | 0.5    | Per-iteration threshold; `nf_MIDAS_Multiple_Resolutions.py` also reads the CLI `-minConfidence` override. |

## 12. CLI arguments

### nf_MIDAS.py

| Flag                    | Type | Default | Notes |
|-------------------------|------|---------|-------|
| `-paramFN`              | str  | —       | Parameter file path. |
| `-nCPUs`                | int  | —       | Parallel workers (OpenMP threads). |
| `-nNodes`               | int  | 1       | For distributed execution. |
| `-machineName`          | str  | `local` | Execution target. |
| `-ffSeedOrientations`   | int  | 0       | 1 = use FF-derived seed orientations (via `GenSeedOrientationsFF2NFHEDM`). |
| `-gpuFit`               | int  | 0       | 1 = use `FitOrientationGPU`. |
| `-doImageProcessing`    | int  | 1       | 1 = run image processing; 0 = skip. |
| `-multiGridPoints`      | int  | 0       | 1 = multi-grid-point calibration. |
| `-refineParameters`     | int  | 0       | 1 = run detector calibration refinement. |
| `-restartFrom`          | str  | `""`    | Restart from a named stage. |
| `-resume`               | str  | `""`    | Resume from checkpoint. |

### nf_MIDAS_Multiple_Resolutions.py (in addition)

| Flag                 | Type | Default | Notes |
|----------------------|------|---------|-------|
| `-startLayerNr`      | int  | 1       | First layer to process. |
| `-endLayerNr`        | int  | 1       | Last layer to process. |
| `-resultFolder`      | str  | —       | Overrides `resultFolder`. |
| `-minConfidence`     | float | —      | Overrides `MinConfidence`. |

---

## NF-specific keys NOT in the shared `MIDASConfig` struct

Useful quick-reference for the validator/wizard: these keys are handled by
per-executable inline parsers, so the NF parser treats them as first-class
but the FF parser ignores them.

| Key                       | Readers                           |
|---------------------------|-----------------------------------|
| `nDistances`              | most NF executables               |
| `Lsd` (multi)             | MakeDiffrSpots, FitOrientationOMP, FitOrientationParameters*, simulateNF, compareNF |
| `BC` (multi)              | FitOrientationOMP, FitOrientationParameters*, simulateNF, compareNF |
| `OmegaRange` (multi)      | FitOrientationOMP, MakeDiffrSpots, simulateNF, compareNF |
| `BoxSize` (multi)         | FitOrientationOMP, MakeDiffrSpots, simulateNF, compareNF |
| `RingsToUse` (multi)      | MakeDiffrSpots, FitOrientationOMP, simulateNF, compareNF |
| `extOrig`, `extReduced`   | ProcessImagesCombined, MedianImageLibTiff |
| `RawStartNr`              | ProcessImagesCombined, MedianImageLibTiff |
| `OrigFileName`            | ProcessImagesCombined, MedianImageLibTiff |
| `ReducedFileName`         | ProcessImagesCombined, FitOrientationOMP, MMapImageInfo, compareNF |
| `WFImages`                | ProcessImagesCombined, MedianImageLibTiff |
| `MicFileBinary`           | FitOrientationOMP, ParseMic       |
| `MicFileText`             | ParseMic                          |
| `GridFileName`            | MakeHexGrid, FitOrientationOMP, compareNF |
| `GridSize`                | MakeHexGrid, FitOrientationParametersMultiPoint |
| `EdgeLength`              | MakeHexGrid                       |
| `Rsample`                 | MakeHexGrid                       |
| `GridMask`                | nf_MIDAS.py, nf_MIDAS_Multiple_Resolutions.py |
| `LayerThickness`          | nf_MIDAS.py, nf_MIDAS_Multiple_Resolutions.py |
| `GlobalPosition`, `GlobalPositionFirstLayer` | ParseMic, nf_MIDAS.py |
| `TomoImage`, `TomoPixelSize` | nf_MIDAS.py                    |
| `DataDirectory`, `OutputDirectory` | every NF executable       |
| `SaveNSolutions`          | FitOrientationOMP, ParseMic, compareNF, nf_MIDAS_Multiple_Resolutions.py |
| `LsdTol`, `LsdRelativeTol`, `BCTol`, `TiltsTol` | FitOrientationParameters* only |
| `NumIterations`, `GridPoints` | FitOrientationParametersMultiPoint only |
| `OnlySpotsInfo`, `SaveReducedOutput`, `SimulationBatches` | simulateNF only |
| `NearestMisorientation`, `MinMisoNSaves` | FitOrientationOMP, compareNF |
| `MaxAngle`                | Mic2GrainsList                    |
| `GridRefactor`, `SeedOrientationsAll` | nf_MIDAS_Multiple_Resolutions.py |
| `SkipImageBinning`, `PrecomputedSpotsInfo` | MMapImageInfo       |
| `Ice9Input`               | MMapImageInfo, FitOrientationOMP, FitOrientationParameters* |
| `Deblur`, `WriteFinImage`, `WriteLegacyBin` | ProcessImagesCombined |

## Shared `MIDASConfig` keys that ARE used in NF

Only `GetHKLList.c` uses the shared parser directly. However NF Example files
set keys that are *also* recognized by that parser when `GetHKLList` runs:
`Wavelength`, `LatticeParameter` (alias `LatticeConstant`), `SpaceGroup`,
`MaxRingNumber`, `MaxRingRad` (alias of `RhoD`), `RingsToExclude`. See
[FF_Parameters_Reference.md](FF_Parameters_Reference.md) for the full
`MIDASConfig` field list.

## Things that commonly break NF runs

A non-exhaustive list of validator candidates — these are the places where
scope creep into a full checker pays for itself:

- **Multi-entry count mismatch**: `Lsd`, `BC`, `OmegaRange`, `BoxSize` must
  each appear exactly `nDistances` times. Silent corruption otherwise.
- **`GridFileName` missing**: `FitOrientationOMP` errors if `MakeHexGrid`
  hasn't run or the grid path is wrong.
- **Stale `MicFileBinary`**: previous-run leftovers cause false seeded
  reconstruction. The workflow driver deletes these at startup (see
  `ProcessImagesCombined` comments); manual reruns skip that cleanup.
- **`StartNr`/`EndNr` vs `NrFilesPerDistance`**: `EndNr - StartNr + 1` must
  match `NrFilesPerDistance × nDistances` (plus any `WFImages` skipped).
- **`SeedOrientations` file missing or wrong format**: `MakeDiffrSpots`
  reads this silently; indexing later looks "just bad".
- **`ExcludePoleAngle = 0`**: causes wedge-corrupted spots near η=0,±180 to
  contaminate indexing. Default 6 is usually fine.
- **Sign of `OmegaStep`** vs order of frames in `RawFolder`: if reversed,
  reconstructed orientations will be mirrored.
- **`nDistances=1` with multi-entry keys**: older NF workflows still require
  a single `Lsd`/`BC` line even for single-distance runs.
- **`px` inconsistency**: the pixel size used at NF is often 1–2 µm, much
  smaller than FF's 200 µm. Accidentally copying an FF value silently breaks
  ring radii.

## Defaults summary (verified against source)

Unlike FF, NF has no central `midas_config_defaults()` governing most keys.
Defaults are inline, per executable. The table below is the union.

| Key                  | Default | Set in |
|----------------------|---------|--------|
| `ExcludePoleAngle`   | 6       | MakeDiffrSpots, FitOrientationOMP |
| `OrientTol`          | 2       | FitOrientationOMP |
| `MinFracAccept`      | 0.04    | FitOrientationOMP |
| `MinConfidence`      | 0.5     | nf_MIDAS.py (`-minConfidence` fallback) |
| `SaveNSolutions`     | 1       | FitOrientationOMP, ParseMic, compareNF |
| `MedFiltRadius`      | 1       | ProcessImagesCombined |
| `GaussFiltRadius`    | 4       | ProcessImagesCombined |
| `LoGMaskRadius`      | 10      | ProcessImagesCombined |
| `DoLoGFilter`        | 0       | ProcessImagesCombined |
| `BlanketSubtraction` | 0       | ProcessImagesCombined |
| `WriteFinImage`      | 0       | ProcessImagesCombined |
| `Deblur`             | 0       | ProcessImagesCombined |
| `WFImages`           | 0       | ProcessImagesCombined, MedianImageLibTiff |
| `NrPixels`           | 2048    | MMapImageInfo (fallback) |
| `extOrig`            | `tif`   | ProcessImagesCombined |
| `extReduced`         | `bin`   | ProcessImagesCombined |
| `GridFileName`       | `grid.txt` | MakeHexGrid, FitOrientationOMP |
| `GBAngle`            | 5.0     | ParseMic |
| `SpaceGroup` (as `SGNr`) | 225 | ParseMic |
| `NumIterations`      | 3       | FitOrientationParametersMultiPoint |
| `NearestMisorientation` | 0    | FitOrientationOMP, compareNF |
| `WriteImage` (sim)   | 1       | simulateNF |

Recommended working values from [Example/ps_au.txt](../NF_HEDM/Example/ps_au.txt)
for typical experiments: `MinFracAccept 0.04` (unseeded), `OrientTol 2`,
`MinConfidence 0.7`, `ExcludePoleAngle 6`, `NrOrientations ~243000`.

## FF/PF keys that do NOT apply to NF

NF-HEDM uses a direct pinhole + tilts geometry inversion. Several keys
common in FF parameter files are silently ignored by NF executables — do
not copy them into an NF configuration expecting them to take effect:

| Key family | Why it doesn't apply to NF |
|---|---|
| `p0`–`p14`, `tolP0`–`tolP14`, `tolP`, `tolP2Panel` | NF has no analytical distortion polynomial — the `(Lsd, BC, tx, ty, tz)` forward model is applied directly. |
| `DistortionFile` | Same: NF does not read a per-pixel distortion map. |
| `RingThresh`, `OverAllRingToIndex`, `MarginRadial`, `MarginEta`, `MarginOme`, `MarginRadius` | These are FF/PF indexer tolerances. NF uses `OrientTol`, `MinFracAccept`, `MinConfidence`, `ExcludePoleAngle` instead. |
| `StepSizeOrient`, `StepSizePos`, `Completeness`, `UseFriedelPairs` | FF/PF indexer parameters; NF's `NrOrientations` + `OrientTol` cover the equivalent search. |
| Forward-sim keys `PeakIntensity`, `MaxOutputIntensity`, `SimNoiseSigma`, `WriteSpots`, `PositionsFile`, `IntensitiesFile`, `EResolution` | FF/PF forward-simulation only (`ForwardSimulationCompressed`). NF's `simulateNF` reads a different subset. |
| Multi-panel keys `NPanelsY`, `NPanelsZ`, `PanelGapsY`, `PanelGapsZ`, `PanelShiftsFile` | Multi-panel detector support is FF/PF-only. |

If you see any of these in an NF parameter file, it is almost certainly a
copy-paste from a FF setup. Remove them to avoid confusion. Running
`midas-params validate <file> --path nf` surfaces them as warnings.
