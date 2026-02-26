# AutoCalibrateZarr.py User Manual

**Version:** 10.0  
**Contact:** hsharma@anl.gov

---

## 1. Introduction

`AutoCalibrateZarr.py` is a sophisticated utility script within the MIDAS framework designed to automatically determine the precise geometry of a powder X-ray diffraction experiment. It analyzes a 2D diffraction image containing Debye-Scherrer rings from a known calibrant material (e.g., CeO2) and iteratively refines a comprehensive set of geometric parameters until it converges on a stable, high-accuracy solution.

The script is a crucial first step for any HEDM analysis, as an accurate geometric calibration is fundamental to correctly interpreting diffraction data. It can process several input file formats (Zarr, HDF5, TIFF, GE) and produces a refined parameter file (`refined_MIDAS_params.txt`) ready for use in subsequent MIDAS analysis scripts like `ff_MIDAS.py`.

### Key Features
*   **Automated Iterative Refinement:** The script runs the `CalibrantOMP` binary in a loop, automatically identifying and excluding outlier rings and refining parameters until a user-defined strain tolerance is met.
*   **Comprehensive Parameter Fitting:** It refines a wide range of geometric parameters, including:
    *   Sample-to-detector distance (`Lsd`)
    *   Beam center coordinates (`BC`)
    *   Detector tilts (`ty`, `tz`)
    *   Detector distortion coefficients (`p0`, `p1`, `p2`, `p3`)
*   **Robust Image Processing:** Employs advanced techniques like median filtering for background subtraction, automatic thresholding, and an optimized algorithm for detecting the initial beam center from ring patterns.
*   **Flexible Input:** Can handle multiple common data formats and automatically convert them to the Zarr format required by the MIDAS backend.
*   **Detailed Output:** Generates a final, refined parameter file, along with optional plots and an HDF5 file containing all intermediate data for detailed inspection and debugging.

---

## 2. Prerequisites

1.  **MIDAS Installation:** The script must be located within a functioning MIDAS installation to access the `CalibrantOMP` and `GetHKLList` binaries in the `FF_HEDM/bin/` directory.
2.  **Python Environment:** A Python environment with the following libraries installed: `numpy`, `matplotlib`, `zarr`, `scikit-image`, `plotly`, `pandas`, `diplib`, `Pillow`, `h5py`, and `numba`.
3.  **Input Data:**
    *   A 2D diffraction image of a calibrant material showing clear Debye-Scherrer rings.
    *   If using a format other than Zarr (e.g., HDF5, TIFF), a basic parameter file (`-paramFN`) is required to provide essential metadata like `SpaceGroup`, `Wavelength`, and `LatticeConstant`.

---

## 3. Workflow Overview

The script follows a logical, multi-step process to achieve a converged geometric solution:
 
 ```mermaid
 graph TD
     A[Start] --> B{Input Format?};
     B -- Zarr --> C[Read Zarr];
     B -- TIFF/HDF5/GE --> D[Convert to Zarr using ffGenerateZipRefactor];
     D --> C;
     C --> E[Calculate Average Image];
     E --> F{NoMedian?};
     F -- No --> G[Apply Median Filter/Background Subtraction];
     F -- Yes --> H[Skip Background Subtraction];
     G --> I[Threshold Image];
     H --> I;
     I --> J[Detect Beam Center & Ring Radii];
     J --> K[Estimate Initial Lsd];
     K --> L[Refinement Loop Start];
     L --> M[Run CalibrantOMP];
     M --> N[Analyze Strain & Identify Outliers];
     N --> O{Converged?};
     O -- No --> P[Exclude Outliers];
     P --> M;
     O -- Yes --> Q[Save refined_MIDAS_params.txt];
     Q --> R[Save Optional HDF5 Data];
     R --> S[End];
 ```

1.  **File Input & Conversion:**
    *   The script takes a data file as input. If the format is TIFF, GE, or HDF5, it first calls `ffGenerateZipRefactor.py` to convert it into a standard `.MIDAS.zip` (Zarr) archive.
    *   It can also apply geometric transformations (flips, transposes) if specified via `-ImTransOpt`.
    *   It reads essential metadata (e.g., `SpaceGroup`, `Wavelength`, `PixelSize`) from the Zarr file or the provided parameter file.

2.  **Initial Image Processing:**
    *   It calculates an average 2D image from the input data (if it's a 3D stack).
    *   A heavy median filter (`diplib.MedianFilter`) is applied to create a robust model of the image background (unless `-NoMedian 1` is used).
    *   The background is subtracted, and an automatic or user-defined threshold is applied to create a clean, binary image of the diffraction rings.

3.  **Initial Guess Estimation:**
    *   **Beam Center:** It uses a highly optimized, parallel algorithm (`detect_beam_center_optimized`) to find the geometric center of the ring pattern from the thresholded image. Alternatively, a manual guess can be provided.
    *   **Ring Radii:** It detects the radii of the visible rings in the image.
    *   **Sample-to-Detector Distance (`Lsd`):** It compares the ratios of the detected ring radii to the theoretical ratios for the calibrant material to make an initial estimate of the `Lsd`. A manual guess can also be provided.

4.  **Iterative Refinement Loop:** This is the core of the script.
    *   The script enters a `while` loop that continues until the refinement converges.
    *   **Run `CalibrantOMP`:** In each iteration, it calls the `CalibrantOMP` MIDAS binary with the current best-fit parameters. This binary performs a least-squares fit of the ring data and calculates a new, more accurate set of geometric parameters (`Lsd`, `BC`, tilts, distortion). It also computes a "pseudo-strain" for each measured point on each ring.
    *   **Analyze Results:** The script parses the output of `CalibrantOMP`. The pseudo-strain should be close to zero for a perfect fit.
    *   **Outlier Rejection:** It calculates the mean pseudo-strain for each ring. If a ring's mean strain is significantly higher than the median (controlled by `-MultFactor`), it is flagged as an outlier and added to an exclusion list for the next iteration.
    *   **Check for Convergence:** The loop terminates when two conditions are met:
        1.  No new outlier rings are identified in an iteration.
        2.  The overall mean pseudo-strain falls below a specified tolerance (`-StoppingStrain`).

5.  **Final Output:**
    *   Once converged, the script writes the final, best-fit geometric parameters to `refined_MIDAS_params.txt`.
    *   If requested, it saves all intermediate data (raw images, backgrounds, thresholded images, strain plots per iteration) into a single HDF5 file for detailed analysis.
    *   It also generates a final report of the converged parameters on the console.

---

## 4. Technical Implementation Details

### 4.1. AutoCalibrateZarr.py (The Orchestrator)
*   **Beam Center Detection:** Uses `scikit-image` (`measure.label`) to identify potential ring arcs. A custom, JIT-compiled function (`numba`) then calculates the geometric center of these arcs. This process is parallelized using Python's `multiprocessing` module for speed.
*   **Initial Guess Logic:** The script attempts to identify rings by comparing the ratios of detected ring radii to the theoretical HKL spacing ratios of the calibrant.
*   **Iterative Refinement:**
    *   The script enters a convergence loop that calls the C binary `CalibrantOMP`.
    *   It parses the `calibrant_screen_out.csv` to get updated parameters.
    *   It calculates the mean pseudo-strain for each ring. Rings with strain > `MultFactor * median_strain` are flagged as outliers and excluded from the next iteration.

### 4.2. CalibrantOMP (The Optimization Engine)
*   **Optimization Algorithm:** Uses the **Nelder-Mead simplex algorithm** (via the `nlopt` library) to minimize the objective function.
*   **Objective Function:** The function calculates the "Mean Pseudo-Strain," which is the sum of differences between the measured ring radii (after geometric correction) and the theoretical ring radii.
*   **Sub-Pixel Precision:** For each azimuthal bin, the code extracts a radial lineout and fits a **Pseudo-Voigt** profile to find the peak position with sub-pixel accuracy.
*   **Parallelization:** The peak fitting process is parallelized using **OpenMP**, distributing the azimuthal bins across available CPU cores.

---

## 4. Command-Line Arguments

The script's behavior is controlled via the following arguments:

### Required Arguments

| Argument | Description | Example |
| :--- | :--- | :--- |
| `-dataFN` | The input data file. Can be `.zip` (Zarr), `.h5`, `.ge`, or `.tif`. | `-dataFN CeO2_scan.h5` |

### File Conversion & Input Parameters

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-ConvertFile` | Specifies the input file type. `0`: Zarr (default), `1`: HDF5, `2`: GE binary, `3`: TIFF. | `0` | `-ConvertFile 1` |
| `-paramFN` | A parameter file containing material/beamline info. **Required** if `-ConvertFile` is not 0. | `''` | `-paramFN initial_params.txt` |
| `-darkFN` | Path to a separate file containing the dark-field image. | `''` | `-darkFN dark_image.h5` |
| `-dataLoc` | Path to the dataset within an HDF5 file if not the standard location. | `''` | `-dataLoc /entry/data/data` |
 | `-ImTransOpt` | Image transformations: `0`: None, `1`: FlipLR, `2`: FlipUD, `3`: Transpose. Can be multiple. | `[0]` | `-ImTransOpt 1 3` |
 | `-BadPxIntensity` | Intensity value representing bad pixels in the input. | `NaN` | `-BadPxIntensity -2` |
 | `-GapIntensity` | Intensity value representing inter-module gaps. | `NaN` | `-GapIntensity -1` |

### Calibration & Refinement Control

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-StoppingStrain`| The mean pseudo-strain at which the refinement is considered converged. | `0.00004` | `-StoppingStrain 0.0001` |
| `-MultFactor` | A multiplier for outlier rejection. A ring is rejected if its mean strain is > `MultFactor` * median strain. | `2.5` | `-MultFactor 3.0` |
| `-FirstRingNr` | The index (1-based) of the first prominent ring visible in the data. Used for the initial `Lsd` estimate. | `1` | `-FirstRingNr 2` |
| `-LsdGuess` | An initial guess for the sample-to-detector distance (in µm). | `1000000` | `-LsdGuess 210000` |
| `-BCGuess` | An initial guess for the beam center `[y z]` (in pixels). If not provided, it is auto-detected. | `[0.0 0.0]` | `-BCGuess 1024.5 1021.0` |
| `-EtaBinSize` | The size of the azimuthal bins (in degrees) used by `CalibrantOMP` for fitting. | `5.0` | `-EtaBinSize 2.0` |
| `-Threshold` | A manual intensity threshold for segmenting the rings. If `0`, it's auto-calculated. | `0` | `-Threshold 500` |
 | `-NoMedian` | Set to `1` to skip the median filter calculation (faster, but less robust background subtraction). | `0` | `-NoMedian 1` |
 
 > [!TIP]
 > Using `-NoMedian 1` significantly speeds up processing, especially for large images, but requires a clean input or a pre-calculated background.
 
 > [!IMPORTANT]
 > If you provide either `-BadPxIntensity` or `-GapIntensity`, it is recommended to provide **both** if your data contains both types of artifacts.

### Input Parameter File (`-paramFN`)

When using `-ConvertFile` (i.e., input is not a pre-existing Zarr), you must provide a parameter file with initial metadata. `AutoCalibrateZarr.py` reads the following keys:

| Key | Description | Example |
| :--- | :--- | :--- |
| `SpaceGroup` | Space group number of the calibrant (e.g., 225 for CeO2). | `225` |
| `LatticeParameter` | Lattice constants (a b c alpha beta gamma). | `5.411 5.411 5.411 90 90 90` |
| `Wavelength` | X-ray wavelength in Angstroms. | `0.41328` |
| `px` | Pixel size in microns. | `200` |
| `RingsToExclude` | (Optional) Ring indices to exclude from the start. | `RingsToExclude 1` |
| `SkipFrame` | (Optional) Number of frames to skip in the data file (usually 0). | `0` |
| `tx` | (Optional) Initial variation in x (usually 0). | `0` |

### Output & Visualization

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `-MakePlots` | Set to `1` to display intermediate Matplotlib plots during the run. | `0` | `-MakePlots 1` |
| `-SavePlotsHDF` | Path to an HDF5 file where all intermediate data arrays will be saved for offline analysis. | `''` | `-SavePlotsHDF cal_data.h5` |

---

## 5. Execution Example

Calibrate the geometry using an HDF5 file, providing an initial parameter file and a good guess for the `Lsd`. Save all the intermediate data to an HDF5 file for later inspection.

```bash
python /path/to/AutoCalibrateZarr.py \
    -dataFN CeO2_30keV_210mm.h5 \
    -ConvertFile 1 \
    -paramFN initial_params.txt \
    -LsdGuess 210000 \
    -StoppingStrain 0.0005 \
    -SavePlotsHDF calibration_run.h5
```

---

## 6. Output Files

-   **`refined_MIDAS_params.txt`**: **This is the primary output.** A text file containing the final, converged geometric parameters in a format ready to be used by other MIDAS tools.
-   **`autocal.log`**: A log file containing detailed information about the script's execution, including parameter values at each iteration.
-   **`calibration_run.h5` (Optional)**: If `-SavePlotsHDF` is used, this HDF5 file contains a structured breakdown of the entire process, including:
    -   Raw, background, and thresholded images.
    -   Data for ring overlays.
    -   Strain vs. 2-theta plots for each iteration.
    -   The final converged strain data and results dataframe.
-   **`calibrant_screen_out.csv`**: The raw text output from the last run of the `CalibrantOMP` binary. Useful for debugging backend issues.

---

## 7. Manual Calibration (Panel Shifts & Rotation)

While `AutoCalibrateZarr.py` provides a robust calibration for most standard setups, it treats the detector as a single continuous surface. For multi-panel detectors (e.g., Pilatus, Eiger, or custom detector arrays) where individual panels may have slight independent translations and rotations, a dedicated refinement step is required using the `CalibrantPanelShiftsOMP` binary.

### When to use this
*   You are using a tiled or multi-panel detector.
*   You observe residuals or "kinks" in the Debye-Scherrer rings at panel boundaries.
*   You need to refine the translations and/or in-plane rotations of individual panels.

### Geometry Model

Each panel is defined by its pixel boundaries (`yMin`, `yMax`, `zMin`, `zMax`) and three per-panel correction parameters:

| Parameter | Description |
| :--- | :--- |
| `dY` | Translational shift in Y (pixels) |
| `dZ` | Translational shift in Z (pixels) |
| `dTheta` | In-plane rotation around the panel center (degrees) |
| `dLsd` | Per-panel sample-to-detector distance offset (μm) |
| `dP2` | Per-panel radial distortion (p2) offset |

The rotation is applied around the geometric center of the panel `(centerY, centerZ)`:

```
rawY = centerY + (Y - centerY)·cos(θ) - (Z - centerZ)·sin(θ)
rawZ = centerZ + (Y - centerY)·sin(θ) + (Z - centerZ)·cos(θ)
correctedY = rawY + dY
correctedZ = rawZ + dZ
```

> [!IMPORTANT]
> One panel must be held fixed (`FixPanelID`) with `dY=0`, `dZ=0`, `dTheta=0` to break the degeneracy with global parameters (beam center and tilts absorb any uniform shift/rotation of all panels).

### Workflow

1.  **Run Auto-Calibration First:**
    Run `AutoCalibrateZarr.py` as described above to obtain a good baseline geometry (`refined_MIDAS_params.txt`).

2.  **Prepare Parameter File:**
    Copy `refined_MIDAS_params.txt` and add the keys below. `AutoCalibrateZarr` does *not* write file-handling or panel parameters, so **you must add them manually**:

    **File I/O Parameters (Required)**
    ```text
    Folder /absolute/path/to/raw/data/
    FileStem CeO2_scan_
    Ext .tif
    StartNr 1
    EndNr 1
    Padding 6
    DataType 9                # 1=uint16, 2=double, 3=float, 4=uint32, 5=int32
                              # 6=tiff-uint32, 7=tiff-uint8, 8=hdf5, 9=tiff-uint16
    Dark /path/to/dark.tif
    HeadSize 8192             # Header size (bytes), for binary formats
    ```

    **Panel Configuration (Required for Multi-Panel)**
    ```text
    NPanelsY 6                # Number of panels in Y direction
    NPanelsZ 8                # Number of panels in Z direction
    PanelSizeY 195            # Pixels per panel in Y
    PanelSizeZ 487            # Pixels per panel in Z
    PanelGapsY 1 7 1 7 1      # Gap widths between Y panels (NPanelsY-1 values)
    PanelGapsZ 17 17 17 17 17 17 17  # Gap widths between Z panels (NPanelsZ-1 values)
    PanelShiftsFile panelshifts.txt  # File to save/load panel corrections
    FixPanelID 0              # Panel held fixed (anchor)
    ```

    **Optimization Tolerances**
    ```text
    tolShifts 1.0             # Search range for dY, dZ (pixels)
    tolRotation 1.0           # Search range for dTheta (degrees); 0 = disabled
    ```

    > [!TIP]
    > When `tolRotation` is `0` (the default), no rotation variables are added to the optimizer — the behavior is identical to the previous version. Set it to a non-zero value (e.g., `1.0` or `3.0`) to enable per-panel rotation optimization.

3.  **Run `CalibrantPanelShiftsOMP`:**

    ```bash
    CalibrantPanelShiftsOMP manual_params.txt 96
    ```

    The binary prints a boxed parameter summary at startup showing all parsed values, followed by optimization progress and results.

4.  **Review Output:**

    The program produces the following output:

    *   **Console:** Refined geometry (`Lsd`, `BC`, tilts, distortion), per-ring deviation tables, and the "Indices per Panel" coverage map.
    *   **`panelshifts.txt`** — Per-panel corrections in text format:
        ```text
        # ID dY dZ dTheta dLsd dP2
        0  0.0000000000  0.0000000000  0.0000000000  0.0000000000  0.0000000000
        1  0.3456789012 -0.1234567890  0.0876543210  12.3456789000  0.0000123456
        ...
        ```
    *   **`panelshifts.txt.shifts.tif`** — A **float32 TIFF image** (same dimensions as the detector) where each pixel contains the total shift magnitude in pixels. This combines translational shifts with position-dependent rotation contribution:

        ```
        magnitude = sqrt((dY + rotDY)² + (dZ + rotDZ)²)
        ```

        where `rotDY` and `rotDZ` are the displacement at that specific pixel due to the panel's in-plane rotation. Gap pixels are set to `-1`.

    > [!TIP]
    > Open the `.shifts.tif` in ImageJ, Python (`tifffile.imread`), or the `ff_asym.py` GUI to visually inspect the per-pixel shift pattern. Panels with large rotations will show a gradient pattern (increasing shift toward panel edges), while purely translational shifts appear as uniform color per panel.

5.  **Iterate if Needed:**
    If the fit hasn't fully converged (high mean strain), you can re-run the binary with the same parameter file. It will automatically read the previous panel shifts from `PanelShiftsFile` as starting values for the next optimization.

6.  **Use in Analysis:**
    The generated `panelshifts.txt` and refined global parameters are used by downstream MIDAS tools (`PeaksFittingOMPZarrRefactor`, `FitMultipleGrains`, etc.) to correct peak positions before indexing and refinement.

### Complete Parameter Reference

The following table lists all parameters recognized by `CalibrantPanelShiftsOMP`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **File I/O** | | | |
| `FileStem` | string | — | Base filename prefix |
| `Folder` | string | — | Data directory |
| `Ext` | string | — | File extension |
| `Dark` | string | — | Dark image path |
| `StartNr` / `EndNr` | int | — | File number range |
| `Padding` | int | 6 | Filename zero-padding width |
| `HeadSize` | int | 8192 | Binary file header (bytes) |
| `DataType` | int | 1 | Pixel data type |
| `SkipFrame` | int | 0 | Frames to skip at start |
| **Detector Geometry** | | | |
| `NrPixels` | int | — | Square detector size (sets Y=Z) |
| `NrPixelsY` / `NrPixelsZ` | int | — | Non-square detector dimensions |
| `px` | double | — | Pixel size (μm) |
| `Lsd` | double | — | Sample-to-detector distance (μm) |
| `BC` | 2×double | — | Beam center (Y Z) in pixels |
| `tx` | double | 0 | Detector tilt X (fixed) |
| `ty` / `tz` | double | 0 | Detector tilts Y, Z (initial) |
| `p0` – `p3` | double | 0 | Distortion coefficients (initial) |
| `ImTransOpt` | int | 0 | Image transform (repeatable) |
| **Crystallography** | | | |
| `SpaceGroup` | int | — | Space group number |
| `LatticeConstant` | 6×double | — | a b c α β γ |
| `Wavelength` | double | — | X-ray wavelength (Å) |
| `RhoD` | double | — | Max ring radius (μm) |
| **Masking** | | | |
| `BadPxIntensity` | int | 0 | Bad pixel intensity value |
| `GapIntensity` | int | 0 | Gap pixel intensity value |
| `MaskFile` | string | — | Binary mask file |
| **Optimization Tolerances** | | | |
| `tolTilts` | double | — | Search range for ty, tz (°) |
| `tolBC` | double | — | Search range for BC (pixels) |
| `tolLsd` | double | — | Search range for Lsd (μm) |
| `tolP` | double | — | Default range for p0–p2 |
| `tolP0` – `tolP3` | double | =tolP | Per-coefficient overrides |
| `tolShifts` | double | 1.0 | Search range for panel dY, dZ (pixels) |
| `tolRotation` | double | 0.0 | Search range for panel dTheta (°) |
| **Calibration Control** | | | |
| `Width` | double | — | Max ring search width (μm) |
| `EtaBinSize` | double | — | Azimuthal bin size (°) |
| `RingsToExclude` | int | — | Ring index to exclude (repeatable) |
| `MultFactor` | double | 0 | Outlier rejection multiplier |
| `MinIndicesForFit` | int | 1 | Min points per ring |
| **Multi-Panel** | | | |
| `NPanelsY` / `NPanelsZ` | int | 0 | Panel grid dimensions |
| `PanelSizeY` / `PanelSizeZ` | int | 0 | Panel size (pixels) |
| `PanelGapsY` / `PanelGapsZ` | int... | — | Inter-panel gap sizes |
| `PanelShiftsFile` | string | — | File for panel corrections |
| `FixPanelID` | int | 0 | Panel held fixed |
| **Iterative Refinement** | | | |
| `nIterations` | int | 1 | Number of refinement iterations (best result is kept) |
| **Doublet Fitting** | | | |
| `DoubletSeparation` | double | 0 | Max pixel separation for doublet ring detection; 0 = disabled |
| **Objective Function Weighting** | | | |
| `NormalizeRingWeights` | int | 0 | `1` = each ring contributes equally regardless of eta-bin count |
| `WeightByRadius` | int | 0 | `1` = weight points by R/Rmax (emphasizes outer rings) |
| **Outlier Rejection** | | | |
| `OutlierIterations` | int | 1 | Number of iterative sigma-clipping passes |
| **Distortion Model** | | | |
| `DistortionOrder` | int | 4 | `4` = standard (p0–p3); `6` = adds R⁶ term (p4) |
| `tolP4` | double | 0 | Search range for p4 coefficient |
| **Per-Panel Advanced** | | | |
| `PerPanelLsd` | int | 0 | `1` = fit per-panel Lsd offset (detector non-planarity) |
| `tolLsdPanel` | double | 100 | Search range for per-panel dLsd (μm) |
| `PerPanelDistortion` | int | 0 | `1` = fit per-panel p2 offset (panel flatness) |
| `tolP2Panel` | double | 0.0001 | Search range for per-panel dP2 |
 
### Advanced Optimization Features

The following optional features can be enabled to improve calibration accuracy, especially for multi-panel detectors:

#### Doublet Fitting
Closely spaced rings (within `DoubletSeparation` pixels) are automatically detected and fitted simultaneously using a 10-parameter dual pseudo-Voigt model with shared background and Lorentzian fraction. This eliminates bias from overlapping peaks that would otherwise require excluding those rings.

```text
DoubletSeparation 25
```

#### Ring Weight Normalization
By default, inner rings near the beam center have more eta bins on the detector and thus contribute more terms to the optimization objective. Enabling `NormalizeRingWeights` ensures each ring contributes equally:

```text
NormalizeRingWeights 1
```

#### Higher-Order Distortion
The standard distortion model uses 4 parameters (p0–p3). Enabling `DistortionOrder 6` adds an R⁶ radially-symmetric term (p4) that can dramatically reduce StdStrain (up to ~74% improvement observed):

```text
DistortionOrder 6
tolP4 0.0001
```

#### Per-Panel Corrections
For tiled detectors where panels may not be perfectly co-planar or uniformly flat:

- **`PerPanelLsd 1`** — Fits a per-panel Lsd offset to account for detector tile non-planarity.
- **`PerPanelDistortion 1`** — Fits a per-panel p2 (radial distortion) offset to account for per-tile flatness variations.

```text
PerPanelLsd 1
tolLsdPanel 100
PerPanelDistortion 1
tolP2Panel 0.0001
```

#### Iterative Sigma-Clipping
The standard outlier rejection uses a single pass with `MultFactor × MeanStrain` threshold. Setting `OutlierIterations > 1` iterates the rejection, recomputing the mean each pass for more robust rejection:

```text
OutlierIterations 3
```

#### Best-Iteration Tracking
When `nIterations > 1`, the optimizer tracks the best result (lowest MeanStrain) across all iterations and automatically restores those parameters at the end, preventing oscillation from degrading the final result:

```text
nIterations 10
```

> [!TIP]
> A recommended starting configuration for multi-panel detectors:
> ```text
> nIterations 10
> NormalizeRingWeights 1
> DistortionOrder 6
> tolP4 0.0001
> PerPanelDistortion 1
> OutlierIterations 3
> DoubletSeparation 25
> ```

---
 
 ## 8. See Also

- [FF_Analysis.md](FF_Analysis.md) — Standard FF-HEDM analysis using calibrated geometry
- [PF_Analysis.md](PF_Analysis.md) — Scanning/Point-Focus FF-HEDM analysis
- [FF_dual_datasets.md](FF_dual_datasets.md) — Dual-dataset FF-HEDM analysis
- [FF_RadialIntegration.md](FF_RadialIntegration.md) — Radial integration / caking using calibrated parameters
- [ForwardSimulationManual.md](ForwardSimulationManual.md) — Forward simulation for validation
- [GSAS-II_Integration.md](GSAS-II_Integration.md) — Importing caked output into GSAS-II
- [README.md](README.md) — High-level MIDAS overview and manual index

---

If you encounter any issues or have questions, please open an issue on this repository.