# AutoCalibrateZarr.py User Manual

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
| `-BCGuess` | An initial guess for the beam center `[y x]` (in pixels). If not provided, it is auto-detected. | `[0.0 0.0]` | `-BCGuess 1024.5 1021.0` |
| `-EtaBinSize` | The size of the azimuthal bins (in degrees) used by `CalibrantOMP` for fitting. | `5.0` | `-EtaBinSize 2.0` |
| `-Threshold` | A manual intensity threshold for segmenting the rings. If `0`, it's auto-calculated. | `0` | `-Threshold 500` |
 | `-NoMedian` | Set to `1` to skip the median filter calculation (faster, but less robust background subtraction). | `0` | `-NoMedian 1` |
 
 > [!TIP]
 > Using `-NoMedian 1` significantly speeds up processing, especially for large images, but requires a clean input or a pre-calculated background.
 
 > [!IMPORTANT]
 > If you provide either `-BadPxIntensity` or `-GapIntensity`, it is recommended to provide **both** if your data contains both types of artifacts.

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

## 7. Manual Calibration (Panel Shifts)
 
 While `AutoCalibrateZarr.py` provides a robust calibration for most standard setups, it treats the detector as a single continuous surface. For multi-panel detectors (e.g., arrays of detectors) where individual panels may have slight independent shifts (`dY`, `dZ`), a manual refinement step is required using the `CalibrantPanelShiftsOMP` binary.
 
 ### When to use this
 *   You are using a customized or multi-panel detector setup.
 *   You observe residuals or "kinks" in the Debye-Scherrer rings that correspond to panel boundaries.
 *   You need to refine the positions of individual panels to improve the global fit.
 
 ### Workflow
 
 1.  **Run Auto-Calibration First:**
     Run `AutoCalibrateZarr.py` as described above to obtain a good baseline geometry. This produces the `refined_MIDAS_params.txt` file.
 
 2.  **Prepare Parameter File:**
     Create a new parameter file (e.g., `manual_params.txt`) by copying `refined_MIDAS_params.txt`. You need to add/ensure the following keys are present to define the multi-panel geometry:
 
     ```text
     NPanelsY [Number of panels in Y]
     NPanelsZ [Number of panels in Z]
     PanelSizeY [Pixels per panel in Y]
     PanelSizeZ [Pixels per panel in Z]
     PanelGapsY [Gap sizes in Y pixels, space-separated]
     PanelGapsZ [Gap sizes in Z pixels, space-separated]
     PanelShiftsFile [Filename to save/load shifts, e.g., panel_shifts.txt]
     ```
 
     You can also tune the optimization parameters:
     ```text
     tolShifts 1.0     # Maximum allowed shift per iteration (in pixels)
     FixPanelID 0      # ID of the panel to keep fixed (anchor)
     ```
 
 3.  **Run `CalibrantPanelShiftsOMP`:**
     Execute the binary manually from the command line:
 
     ```bash
     /path/to/MIDAS/FF_HEDM/bin/CalibrantPanelShiftsOMP manual_params.txt [nCPUs]
     ```
     *   Replace `[nCPUs]` with the number of threads to use (e.g., 4 or 8).
 
 4.  **Review Output:**
     *   The program will print the "Indices per Panel" to visualize the coverage.
     *   It will display the refined geometry (`Lsd`, `BC`, tilts) and the refined panel shifts.
     *   The refined panel shifts will be saved to the file specified in `PanelShiftsFile`.
 
 5.  **Update Geometry:**
     Use the generated `panel_shifts.txt` and the refined global parameters for your subsequent analysis.
 
 ---
 
 ## 8. See Also

- [FF_Analysis.md](FF_Analysis.md) — Standard FF-HEDM analysis using calibrated geometry
- [PF_Analysis.md](PF_Analysis.md) — Scanning/pencil-beam FF-HEDM analysis
- [FF_dual_datasets.md](FF_dual_datasets.md) — Dual-dataset FF-HEDM analysis
- [ForwardSimulationManual.md](ForwardSimulationManual.md) — Forward simulation for validation

---

If you encounter any issues or have questions, please open an issue on this repository.