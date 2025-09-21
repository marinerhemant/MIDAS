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

1.  **File Input & Conversion:**
    *   The script takes a data file as input. If the format is TIFF, GE, or HDF5, it first calls `ffGenerateZipRefactor.py` to convert it into a standard `.MIDAS.zip` (Zarr) archive.
    *   It reads essential metadata (e.g., `SpaceGroup`, `Wavelength`, `PixelSize`) from the Zarr file or the provided parameter file.

2.  **Initial Image Processing:**
    *   It calculates an average 2D image from the input data (if it's a 3D stack).
    *   A heavy median filter (`diplib.MedianFilter`) is applied to create a robust model of the image background.
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