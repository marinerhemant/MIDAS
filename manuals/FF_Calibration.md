# AutoCalibrateZarr.py User Manual

**Version:** 12.0  
**Contact:** hsharma@anl.gov

---

## 1. Introduction

`AutoCalibrateZarr.py` is a utility script within the MIDAS framework designed to automatically determine the precise geometry of a powder X-ray diffraction experiment. It analyzes a 2D diffraction image containing Debye-Scherrer rings from a known calibrant material (e.g., CeO2, LaB6) and refines a comprehensive set of geometric parameters (including optional parallax correction) via a single call to the `CalibrantPanelShiftsOMP` C/OpenMP binary (40 iterations, doublet detection, SNR weighting).

The script is a crucial first step for any HEDM analysis, as an accurate geometric calibration is fundamental to correctly interpreting diffraction data. It can process several input file formats (Zarr, HDF5, TIFF, GE) and produces a refined parameter file (`refined_MIDAS_params.txt`) ready for use in subsequent MIDAS analysis scripts like `ff_MIDAS.py`.

### Key Features
*   **Single-Call C-Side Refinement:** The script calls `CalibrantPanelShiftsOMP` once with `nIterations=40` — all iteration, outlier rejection, doublet detection, and convergence are handled inside the optimized C/OpenMP binary.
*   **Auto-Detect File Format:** Automatically determines input format from the file extension (`.zip`, `.h5`, `.tif`, `.ge*`). No need to specify `-ConvertFile`.
*   **Comprehensive Parameter Fitting:** Refines `Lsd`, `BC`, tilts, distortion (`p0`–`p5`), optional parallax and wavelength, with doublet detection (`DoubletSeparation=25px`), ring weight normalization, and SNR-based weighting all enabled by default.
*   **Modern CLI:** Uses `--flag` convention (e.g., `--data`, `--params`, `--lsd-guess`) with backward-compatible aliases for legacy `-dataFN`, `-paramFN`, etc.
*   **Robust Image Processing:** Median filtering for background subtraction, automatic thresholding, and JIT-accelerated beam center detection.
*   **Flexible Input:** Handles Zarr .zip, HDF5, GE binary, and TIFF formats.
*   **Detailed Output:** Final refined parameter file, optional HDF5 data, and real-time terminal progress from the C binary.

---

## 2. Prerequisites

1.  **MIDAS Installation:** The script must be located within a functioning MIDAS installation to access the `CalibrantPanelShiftsOMP` and `GetHKLList` binaries in the `FF_HEDM/bin/` directory.
2.  **Python Environment:** A Python environment with the following libraries installed: `numpy`, `matplotlib`, `zarr`, `scikit-image`, `plotly`, `pandas`, `diplib`, `Pillow`, `h5py`, and `numba`.
3.  **Input Data:**
    *   A 2D diffraction image of a calibrant material showing clear Debye-Scherrer rings.
    *   If using a format other than Zarr (e.g., HDF5, TIFF), a parameter file (`--params`) **or** the `--px` flag plus energy-in-filename is needed.
    *   **SpaceGroup and LatticeParameter are auto-detected** from the filename (see §3.1 below) and do not need to be specified for CeO2 or LaB6.

---

### 2.1. Calibrant Auto-Detection

The script auto-detects the calibrant material from the data filename (case-insensitive):

| Filename Pattern | Detected Material | SpaceGroup | Lattice (Å) |
|---|---|---|---|
| `ceo2`, `CeO2`, `ceria`, `ceriumoxide` | CeO2 | 225 | 5.4116 |
| `lab6`, `LaB6`, `lab_6`, `lab-6`, `lanthanumhexaboride` | LaB6 | 221 | 4.1569 |

It also extracts **energy** and **distance** from the filename:

| Pattern | Example | Parsed Value |
|---|---|---|
| `<number>keV` (decimal: `.` or `p`) | `71p676keV` | Energy 71.676 keV → wavelength 0.17298 Å |
| `<number>mm` | `657mm` | Distance 657 mm → Lsd 657000 µm |

For example, `CeO2_Pil_100x100_att000_657mm_71p676keV_000062.tif` auto-sets:
- Calibrant: CeO2 (SG 225)
- Wavelength: 0.17298 Å (from 71.676 keV)
- Lsd guess: 657000 µm (from 657 mm)

**Priority:** Zarr metadata > param file > CLI argument > filename detection > defaults.

---

## 3. Workflow Overview

The script follows a logical, multi-step process to achieve a converged geometric solution:
 
 ```mermaid
 graph TD
     A[Start] --> B{Auto-detect Format};
     B -- Zarr --> C[Read Zarr];
     B -- TIFF/HDF5/GE --> D[Convert to Zarr];
     D --> C;
     C --> E[Calculate Average Image];
     E --> F{NoMedian?};
     F -- No --> G[Median Filter Background];
     F -- Yes --> H[Dark Subtraction];
     G --> I[Threshold Image];
     H --> I;
     I --> J[Detect Beam Center & Ring Radii];
     J --> K[Estimate Initial Lsd];
     K --> L["CalibrantPanelShiftsOMP<br/>(40 iterations, doublets, SNR weights)"];
     L --> Q[Save refined_MIDAS_params.txt];
     Q --> R[Save Optional HDF5 Data];
     R --> S[End];
 ```

1.  **File Input & Conversion:**
    *   The file format is auto-detected from the extension (`.zip`→Zarr, `.h5`→HDF5, `.ge*`→GE, `.tif`→TIFF).
    *   Non-Zarr formats are automatically converted via `ffGenerateZipRefactor.py`.
    *   Image transformations (flips, transposes) can be applied via `--im-trans`.

2.  **Initial Image Processing:**
    *   Average 2D image is computed from multi-frame input.
    *   Median filter background subtraction (or dark subtraction with `--no-median`).
    *   Automatic or manual thresholding for ring detection.

3.  **Initial Guess Estimation:**
    *   **Beam Center:** JIT-accelerated parallel algorithm, or manual guess via `--bc-guess`.
    *   **Ring Radii & Lsd:** Automatic matching of ring radius ratios to HKL spacings.

4.  **Single-Call Refinement:**
    *   A single call to `CalibrantPanelShiftsOMP` with all features enabled:
        -   `nIterations 40` — multi-iteration with stagnation detection and perturbation
        -   `DoubletSeparation 25` — automatic doublet ring fitting
        -   `OutlierIterations 3` — per-ring sigma-clipping
        -   `NormalizeRingWeights 1` — equal weight per ring
        -   `WeightByFitSNR 1` — SNR-based point weighting
        -   `MinIndicesForFit 5` — skip under-sampled rings
    *   Uses all available CPU cores.
    *   Progress (iteration strains, doublet detections) is streamed to the terminal in real-time.

5.  **Final Output:**
    *   `refined_MIDAS_params.txt` with converged geometry.
    *   Optional HDF5 with all intermediate data.
    *   Console summary of best-fit parameters.

---

## 4. Technical Implementation Details

### 4.1. AutoCalibrateZarr.py (The Orchestrator)
*   **Beam Center Detection:** Uses `scikit-image` (`measure.label`) to identify potential ring arcs. A custom, JIT-compiled function (`numba`) then calculates the geometric center of these arcs. This process is parallelized using Python's `multiprocessing` module.
*   **Initial Guess Logic:** Ring radii ratios are matched to HKL spacing ratios to estimate `Lsd`.
*   **Single C Call:** Instead of a Python iteration loop, the script makes a single call to `CalibrantPanelShiftsOMP` with `nIterations=40` and all advanced features (doublets, SNR weighting, ring normalization). The C code handles all iteration, outlier rejection, and convergence internally.
*   **State Management:** Uses a `CalibState` dataclass to manage all calibration parameters cleanly.

### 4.2. CalibrantPanelShiftsOMP (The Optimization Engine)
*   **Optimization Algorithm:** Uses the **Nelder-Mead simplex algorithm** (via the `nlopt` library) to minimize the objective function.
*   **Objective Function:** The function calculates the "Mean Pseudo-Strain," which is the sum of differences between the measured ring radii (after geometric correction) and the theoretical ring radii.
*   **Sub-Pixel Precision:** For each azimuthal bin, the code extracts a radial lineout and fits a **height-normalized Pseudo-Voigt** profile (Gaussian and Lorentzian sharing a single FWHM, with mixing parameter Mu) to find the peak position with sub-pixel accuracy.

    **Singlet Peak Shape** (5 parameters: $R_{cen}$, $\mu$, $\Gamma$, $I_{max}$, $BG$):

    $$L(R) = \frac{1}{1 + 4\,(R - R_{cen})^2 / \Gamma^2} \qquad G(R) = \exp\!\left(-\frac{4\ln 2\,(R - R_{cen})^2}{\Gamma^2}\right)$$

    $$I(R) = BG + I_{max}\bigl[\mu\,L(R) + (1-\mu)\,G(R)\bigr]$$

    Both $L$ and $G$ peak at 1.0 at $R = R_{cen}$ and share the same FWHM $\Gamma$.

    **Doublet Peak Shape** (8 parameters: $R_1$, $R_2$, $\mu$, $\Gamma_1$, $I_{max,1}$, $\Gamma_2$, $I_{max,2}$, $BG$):

    For closely-spaced ring pairs (within `DoubletSeparation` pixels), two peaks are fitted simultaneously with shared $\mu$ and $BG$:

    $$I(R) = BG + I_{max,1}\bigl[\mu\,L_1(R) + (1-\mu)\,G_1(R)\bigr] + I_{max,2}\bigl[\mu\,L_2(R) + (1-\mu)\,G_2(R)\bigr]$$

    The doublet fitter uses several safeguards to ensure robust results:
    - **Center constraints:** $R_1$ is constrained below the theoretical midpoint $R_{mid}$ and $R_2$ above it, preventing the optimizer from swapping peak assignments.
    - **Ideal-radius initialization:** Initial guesses use the theoretical ring radii rather than intensity-weighted means, which fail when peaks heavily overlap.
    - **Edge-clip fallback:** If the merged doublet window extends beyond the detector, the primary ring falls back to singlet fitting instead of discarding both rings.
    - **Pre-initialized partner slots:** Secondary doublet array slots are zeroed before the parallel region to avoid undefined values from uninitialized memory.

*   **Parallelization:** The peak fitting process is parallelized using **OpenMP**, distributing the azimuthal bins across available CPU cores.

---

## 4. Command-Line Arguments

The script uses `--flag` convention with backward-compatible aliases for all legacy `-Flag` names.

### Required Arguments

| New Flag | Legacy Alias | Description | Example |
| :--- | :--- | :--- | :--- |
| `--data` | `-dataFN`, `-d` | Input data file (format auto-detected from extension) | `--data CeO2.h5` |

### File Conversion & Input Parameters

| New Flag | Legacy Alias | Description | Default | Example |
| :--- | :--- | :--- | :--- | :--- |
| `--convert` | `-ConvertFile` | Force format: `0`=Zarr, `1`=HDF5, `2`=GE, `3`=TIFF. Default: auto-detect. | auto | `--convert 1` |
| `--params` | `-paramFN`, `-p` | Parameter file. Optional if `--px` + energy in filename. | `''` | `--params setup.txt` |
| `--dark` | `-darkFN` | Separate dark field image file | `''` | `--dark dark.h5` |
| `--data-loc` | `-dataLoc` | HDF5 dataset path (if non-standard) | `''` | `--data-loc /entry/data` |
| `--im-trans` | `-ImTransOpt` | Image transforms: `0`=none, `1`=flipLR, `2`=flipUD, `3`=transpose | `[0]` | `--im-trans 1 3` |
| `--bad-px` | `-BadPxIntensity` | Bad pixel intensity value | `NaN` | `--bad-px -2` |
| `--gap-px` | `-GapIntensity` | Gap pixel intensity value | `NaN` | `--gap-px -1` |

### Calibration & Refinement Control

| New Flag | Legacy Alias | Description | Default | Example |
| :--- | :--- | :--- | :--- | :--- |
| `--n-iterations` | — | Number of C-side calibration iterations | `40` | `--n-iterations 20` |
| `--mult-factor` | `-MultFactor` | Outlier ring rejection factor (× median strain) | `2.5` | `--mult-factor 3.0` |
| `--doublet-separation` | — | Doublet detection threshold (pixels) | `25` | `--doublet-separation 30` |
| `--outlier-iterations` | — | Per-ring outlier removal iterations | `3` | `--outlier-iterations 5` |
| `--first-ring` | `-FirstRingNr` | First ring number to use | `1` | `--first-ring 2` |
| `--eta-bin-size` | `-EtaBinSize` | Azimuthal bin size (degrees) | `5.0` | `--eta-bin-size 2.0` |
| `--lsd-guess` | `-LsdGuess` | Initial guess for detector distance (µm) | `1000000` | `--lsd-guess 210000` |
| `--bc-guess` | `-BCGuess` | Initial guess for beam center [Y Z] (pixels) | `[0 0]` | `--bc-guess 1024 1024` |
| `--threshold` | `-Threshold` | Manual threshold for ring detection (0=auto) | `0` | `--threshold 500` |
| `--no-median` | `-NoMedian` | Skip median filter (0=use, 1=skip) | `0` | `--no-median 1` |
| `--px` | — | Pixel size (µm). Enables param-file-free usage for non-Zarr. | `0` (auto) | `--px 172` |
| `--tx` | — | Detector tilt tx (radians). Not fitted, passed through to `CalibrantPanelShiftsOMP`. | `0.0` | `--tx 0.001` |
| `--mask` | `-MaskFile` | Mask TIFF for bad/gap pixels (passed as `MaskFile`). Convention: `0` = bad, `1` = good. | `''` | `--mask mask.tif` |
| `--fit-parallax` | — | Fit parallax correction (0=off, 1=on) | `0` | `--fit-parallax 1` |
| `--parallax-guess` | — | Initial guess for parallax value (µm) | `0.0` | `--parallax-guess 50.0` |
| `--tol-parallax` | — | Tolerance for parallax bounds (µm) | `200.0` | `--tol-parallax 100.0` |
| `--cpus` | — | Number of CPUs for CalibrantPanelShiftsOMP (0=all) | `0` | `--cpus 32` |

> [!TIP]
> The following C-side features are enabled by default and do not need explicit flags:
> `NormalizeRingWeights 1`, `WeightByFitSNR 1`, `MinIndicesForFit 5`

> [!IMPORTANT]
> `-StoppingStrain` has been removed. Convergence is now controlled entirely by `--n-iterations` and `--mult-factor` inside the C binary.

### Input Parameter File (`--params`)

When input is not a pre-existing Zarr file, provide a parameter file with initial metadata:

| Key | Description | Example |
| :--- | :--- | :--- |
| `SpaceGroup` | Space group number (e.g., 225 for CeO2) | `225` |
| `LatticeParameter` | Lattice constants (a b c α β γ) | `5.411 5.411 5.411 90 90 90` |
| `Wavelength` | X-ray wavelength (Å) | `0.41328` |
| `px` | Pixel size (µm) | `200` |
| `RingsToExclude` | (Optional) Ring indices to exclude | `RingsToExclude 1` |
| `SkipFrame` | (Optional) Frames to skip | `0` |
| `tx` | (Optional) Initial tx value | `0` |

### Output & Visualization

| New Flag | Legacy Alias | Description | Default | Example |
| :--- | :--- | :--- | :--- | :--- |
| `--plots` | `-MakePlots`, `-P` | Make plots: `0`=no, `1`=yes | `0` | `--plots 1` |
| `--save-hdf` | `-SavePlotsHDF` | Save arrays to HDF5 file | `''` | `--save-hdf cal.h5` |

---

## 5. Execution Examples

### Simple (auto-detect everything)
```bash
python AutoCalibrateZarr.py --data CeO2_00001.zip
```

### HDF5 with geometry hints
```bash
python AutoCalibrateZarr.py --data CeO2_30keV.h5 \
    --params initial_params.txt \
    --lsd-guess 210000 \
    --save-hdf calibration.h5
```

### TIFF with old-style flags (backward compatible)
```bash
python AutoCalibrateZarr.py -dataFN CeO2.tif -paramFN ps.txt \
    -BadPxIntensity -2 -GapIntensity -1 -MultFactor 3.0
```

### Zero-config TIFF (everything auto-detected from filename)
```bash
python AutoCalibrateZarr.py \
    --data CeO2_Pil_100x100_att000_657mm_71p676keV_000062.tif --px 172
```

### With custom iteration count and CPU control
```bash
python AutoCalibrateZarr.py --data CeO2.zip \
    --n-iterations 20 --cpus 64 --doublet-separation 30
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
-   **`calibrant_screen_out.csv`**: The raw text output from `CalibrantPanelShiftsOMP`, including per-iteration strain values. Contains all 40 iterations of the single C call.
-   **`.lineout.xy`**: Full-range 2θ vs intensity lineout (text, two-column) generated at the end of calibration. Useful for visual comparison with `IntegratorZarrOMP` output via `plot_lineout_comparison.py`.

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

### Geometry Model

The full detector geometry model describes how a pixel position `(Y, Z)` on the detector maps to a scattering angle `2θ` and azimuthal angle `η`. The model consists of several stages applied sequentially:

#### 1. Per-Panel Correction (Multi-Panel Detectors)

For each pixel, the code first determines which panel it belongs to and applies the per-panel corrections. The in-plane rotation is applied around the panel geometric center `(cY, cZ)`, followed by the translational shift:

```
Y' = cY + (Y - cY)·cos(dθ) - (Z - cZ)·sin(dθ) + dY
Z' = cZ + (Y - cY)·sin(dθ) + (Z - cZ)·cos(dθ) + dZ
```

#### 2. Detector Coordinate Transform

The corrected pixel position is converted to physical coordinates relative to the beam center `(ybc, zbc)`:

```
Yc = -(Y' - ybc) · px
Zc =  (Z' - zbc) · px
```

where `px` is the pixel size in μm.

#### 3. Tilt Correction

The detector may be tilted relative to the ideal normal-to-beam orientation. Three rotation matrices (about X, Y, Z axes) are applied:

```
Rx = [[1, 0, 0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]]
Ry = [[cos(ty), 0, sin(ty)], [0, 1, 0], [-sin(ty), 0, cos(ty)]]
Rz = [[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0, 0, 1]]

T = Rx · (Ry · Rz)

[X, Y, Z]_lab = T · [0, Yc, Zc]
```

The scattering vector in lab coordinates is then:

```
Lsd_eff = Lsd + dLsd  (per-panel Lsd offset, if PerPanelLsd enabled)
XYZ = [Lsd_eff + X_lab, Y_lab, Z_lab]
R = (Lsd_eff / XYZ[0]) · sqrt(Y_lab² + Z_lab²)
η = atan2(Y_lab, Z_lab)
```

#### 4. Spatial Distortion Correction

The distortion model corrects for non-ideal detector response. The normalized radius is `RNorm = R / Rmax`:

```
D(R,η) = p0·RNorm²·cos(2·(90-η)) + p1·RNorm⁴·cos(4·(90-η) + p3)   [azimuthal]
        + p2·RNorm²  +  p5·RNorm⁴  +  p4·RNorm⁶                    [isotropic radial]
```

The isotropic radial terms form a polynomial in RNorm: `p2·R² + p5·R⁴ + p4·R⁶`. The `p5·R⁴` term (new in v10) fills the gap between `p2·R²` and `p4·R⁶`, enabling the model to capture residual patterns with two zero crossings.

When `PerPanelDistortion` is enabled, `p2` is replaced by `p2 + dP2` (per-panel offset).

The corrected radius and ideal radius are:
```
R_corr = R · (1 + D(R,η))
R_ideal = Lsd_eff · tan(2θ_hkl) + parallax · sin(2θ_hkl)
```

The parallax term accounts for the depth-dependent offset of X-ray conversion in thick scintillator detectors. When `FitParallax` is enabled, the parallax value is optimized alongside the geometry; when a fixed non-zero `Parallax` value is provided, it is applied as a correction without fitting.

#### 5. Objective Function

The optimization minimizes the sum of fractional differences:
```
Objective = Σᵢ wᵢ · |1 - R_corr,i / R_ideal,i|
```

where the weight `wᵢ` is:
- `1.0` by default
- `1 / N_ring` if `NormalizeRingWeights` enabled (each ring contributes equally)
- Multiplied by `RNorm` if `WeightByRadius` enabled (outer rings weighted more)
- Multiplied by `min(1, SNR_i / median_SNR)` if `WeightByFitSNR` enabled (bins with poor peak fits contribute less; SNR = fitted amplitude / rms residual of the pseudo-Voigt fit)
- If `L2Objective` is enabled, the strain is squared before weighting (L2 norm); otherwise absolute value is used (L1 norm)


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

    *   **Console:** Refined geometry (`Lsd`, `BC`, tilts, distortion, parallax if fitted), per-ring deviation tables (with `MeanSNR` and `MeanStrain(µε)` columns), and the "Indices per Panel" coverage map.
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
| `p0` – `p5` | double | 0 | Distortion coefficients (initial). `p0`/`p1` are azimuthal, `p2`/`p5`/`p4` are isotropic radial, `p3` is azimuthal phase. |
| `ImTransOpt` | int | 0 | Image transform (repeatable) |
| **Crystallography** | | | |
| `SpaceGroup` | int | — | Space group number |
| `LatticeConstant` | 6×double | — | a b c α β γ |
| `Wavelength` | double | — | X-ray wavelength (Å) |
| `RhoD` | double | — | Max ring radius (μm) |
| **Masking** | | | |
| `BadPxIntensity` | int | 0 | Bad pixel intensity value |
| `GapIntensity` | int | 0 | Gap pixel intensity value |
| `MaskFile` | string | — | Path to a uint8 TIFF mask file. Convention: `0` = valid pixel, `1` = masked (bad) pixel. Can be generated from a dark frame using `utils/generate_mask.py`. |
| **Optimization Tolerances** | | | |
| `tolTilts` | double | — | Search range for ty, tz (°) |
| `tolBC` | double | — | Search range for BC (pixels) |
| `tolLsd` | double | — | Search range for Lsd (μm) |
| `tolP` | double | — | Default range for distortion coefficients |
| `tolP0` – `tolP3` | double | =tolP | Per-coefficient overrides |
| `tolP4` | double | =tolP | Search range for p4 (R⁶) coefficient |
| `tolP5` | double | =tolP | Search range for p5 (R⁴) coefficient |
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
| `nIterations` | int | 1 | Number of refinement iterations (best result is kept). Set to `0` to skip optimization and evaluate input parameters directly (strain + lineout output only). |
| **Doublet Fitting** | | | |
| `DoubletSeparation` | double | 0 | Max pixel separation for doublet ring detection; 0 = disabled |
| **Objective Function Weighting** | | | |
| `NormalizeRingWeights` | int | 0 | `1` = each ring contributes equally regardless of eta-bin count |
| `WeightByRadius` | int | 0 | `1` = weight points by R/Rmax (emphasizes outer rings) |
| `WeightByFitSNR` | int | 0 | `1` = weight points by peak-fit SNR (low-quality fits contribute less) |
| `L2Objective` | int | 0 | `1` = use squared strain (L2 norm) instead of absolute strain (L1) |
| **Outlier Rejection** | | | |
| `OutlierIterations` | int | 1 | Number of iterative sigma-clipping passes |
| **Parallax Correction** | | | |
| `FitParallax` | int | 0 | `1` = fit parallax correction alongside geometry |
| `Parallax` | double | 0 | Initial parallax value (µm). Applied even if `FitParallax` is 0 when non-zero. |
| `tolParallax` | double | 200 | Search range for parallax (µm). If `FitParallax=1` and `tolParallax` is unset, defaults to 200 µm. |
| **Wavelength Fitting** | | | |
| `FitWavelength` | int | 0 | `1` = refine wavelength alongside geometry |
| `tolWavelength` | double | 0.001 | Search range for wavelength (Å) |
| `PointDSpacing` | — | auto | Per-ring d-spacings computed from lattice parameters |
| **Per-Panel Advanced** | | | |
| `PerPanelLsd` | int | 0 | `1` = fit per-panel Lsd offset (detector non-planarity) |
| `tolLsdPanel` | double | 100 | Search range for per-panel dLsd (μm) |
| `PerPanelDistortion` | int | 0 | `1` = fit per-panel p2 offset (panel flatness) |
| `tolP2Panel` | double | 0.0001 | Search range for per-panel dP2 |

> [!TIP]
> To create a mask from a dark frame, use the included utility:
> ```bash
> python utils/generate_mask.py dark.tif -1 -2 -o mask.tif
> ```
> This marks all pixels with intensity -1 (gaps) or -2 (bad pixels) as masked.
 
### Advanced Optimization Features

The following optional features can be enabled to improve calibration accuracy, especially for multi-panel detectors:

#### Doublet Fitting
Closely spaced rings (within `DoubletSeparation` pixels) are automatically detected and fitted simultaneously using an 8-parameter dual height-normalized Pseudo-Voigt model with shared Mu and background, each peak having its own independent FWHM (Gamma) and peak height (Imax). This eliminates bias from overlapping peaks that would otherwise require excluding those rings.

The doublet fitter includes several robustness improvements:
- **Center constraints:** Each peak center is constrained to its side of the theoretical midpoint, preventing label swaps.
- **Ideal-radius initial guesses:** Uses theoretical ring positions instead of intensity-weighted means, which are unreliable for heavily overlapping peaks.
- **Edge-clip fallback:** When the merged window extends beyond the detector boundary, the primary ring falls back to singlet fitting rather than discarding both rings.

```text
DoubletSeparation 25
```

#### Ring Weight Normalization
By default, inner rings near the beam center have more eta bins on the detector and thus contribute more terms to the optimization objective. Enabling `NormalizeRingWeights` ensures each ring contributes equally:

```text
NormalizeRingWeights 1
```

#### Isotropic Radial Distortion (p2, p5, p4)
The distortion model includes three isotropic radial terms: `p2·R²`, `p5·R⁴` (new in v10), and `p4·R⁶`. All three are always active with tolerances defaulting to `tolP`. The R⁴ term fills the gap between R² and R⁶, enabling the model to capture residual patterns with two zero crossings. The R⁶ term can reduce StdStrain by up to ~74%.

To set per-coefficient search ranges:
```text
tolP2 0.002
tolP5 0.002
tolP4 0.002
```

> [!WARNING]
> The isotropic radial terms (p2, p5, p4) are correlated with Lsd. When opening their tolerances wider than ~0.002, consider constraining `tolLsd` to prevent the optimizer from finding spurious minima.

#### Parallax Correction
For thick-scintillator detectors, X-rays converting at different depths within the scintillator produce a 2θ-dependent radial offset. The parallax correction adds a term `parallax · sin(2θ)` to the ideal radius. When `FitParallax` is enabled, the parallax value is optimized alongside the standard geometry parameters.

```text
FitParallax 1
Parallax 50.0
tolParallax 200.0
```

A fixed (non-fitted) parallax can also be applied by setting `Parallax` to a non-zero value without enabling `FitParallax`.

> [!WARNING]
> Parallax is partially correlated with Lsd and the isotropic radial distortion terms. When fitting parallax, consider constraining `tolLsd` and `tolP2` to avoid degeneracy.

#### Wavelength Fitting
Optionally refine the X-ray wavelength alongside geometry to account for uncertainties in the incident energy. When enabled, the optimizer adjusts wavelength by recomputing per-point ideal 2θ from Bragg's law (λ/2d).

```text
FitWavelength 1
tolWavelength 0.01
```

> [!IMPORTANT]
> Wavelength fitting is degenerate with lattice parameter. Only enable this when the lattice parameter of the calibrant is well known and the energy/wavelength has measurable uncertainty.

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

#### Evaluate-Only Mode (`nIterations 0`)

Setting `nIterations 0` skips the calibration optimization loop entirely. The code uses the input parameters as-is to compute strain statistics and generate the lineout file. This is useful for:

- Quickly evaluating the quality of existing calibration parameters
- Generating a lineout from a known geometry without fitting
- Comparing parameter sets by inspecting their strain metrics

```text
nIterations 0
```

The output includes `MeanStrain`, `StdStrain`, per-panel microstrain, and a `.lineout.xy` file.

> [!TIP]
> A recommended starting configuration for multi-panel detectors:
> ```text
> nIterations 10
> NormalizeRingWeights 1
> PerPanelDistortion 1
> OutlierIterations 3
> DoubletSeparation 25
> L2Objective 1
> ```

---

## 8. Example Calibrant Data

A complete, ready-to-run CeO2 calibration example is included in the repository:

```
FF_HEDM/Example/Calibration/
├── parameters.txt          # Fully commented parameter file
├── CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif   # Calibrant image
├── dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif  # Dark field
└── mask.tif                # Detector gap/bad-pixel mask
```

The `parameters.txt` file is organized into clearly labeled sections with comments explaining every parameter:

- **Input Data** — file locations and data format
- **Detector Geometry** — pixel size, dimensions, Lsd, beam center, tilts
- **Beam / Sample** — wavelength, RhoD
- **Calibrant Material** — space group, lattice constants
- **Radial & Azimuthal Binning** — R range, eta range, bin sizes
- **Spatial Distortion Model** — p0–p5 distortion coefficients
- **Optimization Tolerances** — search ranges for all parameters
- **Iteration & Convergence** — nIterations, outlier rejection, MultFactor
- **Objective Function Weights** — ring normalization, radius weighting, SNR weighting, L2
- **Panel Geometry** — panel layout, gaps, anchoring, per-panel Lsd/distortion
- **Ring Selection** — excluded rings

This example uses a Pilatus 2M detector (6×8 panels, 1475×1679 pixels, 172 µm pixel size) with CeO2 calibrant at ~650 mm sample-to-detector distance.

---

## 9. Benchmark Testing

An automated benchmark script `tests/test_calibration_integration.py --calibration-only` validates the entire calibration pipeline:

### Usage

```bash
# Basic (single CPU)
python tests/test_calibration_integration.py --calibration-only

# Multi-threaded
python tests/test_calibration_integration.py --calibration-only -nCPUs 4

# Custom parameter file
python tests/test_calibration_integration.py --calibration-only -paramFN /path/to/params.txt

# Adjust pass/fail threshold (default: 50 microstrain)
python tests/test_calibration_integration.py --calibration-only -strainThreshold 40
```

### What it Does

1. Copies example data to a temporary directory
2. Generates `hkls.csv` via `GetHKLList`
3. Runs `CalibrantPanelShiftsOMP` with all features enabled (30 iterations, outlier rejection, per-panel Lsd, L2 objective, etc.)
4. Parses the output and reports strain statistics (mean, std, median, Q25, Q75, min, max)
5. Validates that mean strain ≤ threshold (default 50 µε)
6. Cleans up the temporary directory

### Expected Output

```
======================================================================
  Results
======================================================================
  MeanStrain:   ~35 µε
  StdStrain:    ~15 µε
  MedianStrain: ~30 µε
  ...
======================================================================
  ✅ PASS: MeanStrain ≤ 50 µε
======================================================================
```

---
 
## 10. See Also

- [FF_Analysis.md](FF_Analysis.md) — Standard FF-HEDM analysis using calibrated geometry
- [PF_Analysis.md](PF_Analysis.md) — Scanning/Point-Focus FF-HEDM analysis
- [FF_Dual_Datasets.md](FF_Dual_Datasets.md) — Dual-dataset FF-HEDM analysis
- [FF_Radial_Integration.md](FF_Radial_Integration.md) — Radial integration / caking using calibrated parameters
- [Forward_Simulation.md](Forward_Simulation.md) — Forward simulation for validation
- [GSAS-II_Integration.md](GSAS-II_Integration.md) — Importing caked output into GSAS-II
- [README.md](README.md) — High-level MIDAS overview and manual index

---

If you encounter any issues or have questions, please open an issue on this repository.