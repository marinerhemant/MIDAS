# Near-Field HEDM Calibration GUI

This is a graphical user interface (GUI) for visualizing and analyzing near-field High-Energy Diffraction Microscopy (HEDM) data. The primary purpose of this tool is to perform a near-field calibration to determine the precise beam position and detector distances using a known calibrant sample, such as single-crystal gold.

**DETAILED INSTRUCTIONS FOR CALIBRATION ARE AT:** [NF_CALIB](https://github.com/marinerhemant/MIDAS/manuals/NF_calibration.md)

## Features

*   **Image Visualization:** Load and display `.tif` diffraction images with adjustable contrast and logarithmic scaling.
*   **Profile Analysis:** Generate horizontal and vertical intensity line profiles to precisely locate diffraction features.
*   **Beam Center Calibration:** A guided workflow to determine the beam center at multiple detector distances.
*   **Detector Distance Calculation:** Tools to select corresponding diffraction spots across different distances and compute the sample-to-detector distance.
*   **Microstructure Correlation:** Load and visualize `.mic` or `.map` files, and correlate grain orientations with observed diffraction spots.
*   **Diffraction Simulation:** Simulate expected diffraction patterns for a given set of grain and experimental parameters.

## Getting Started

### Prerequisites

Ensure you have a working Python environment with the following libraries installed:
*   Tkinter
*   Pillow (PIL)
*   Matplotlib
*   NumPy

### Launching the Application

To run the GUI, execute the following command from your terminal:

```bash
python ~/opt/MIDAS/gui/nf.py &```

---

## User Guide

### 1. Loading Data

1.  **Launch the Application:** Run the command above to open the GUI.
2.  **Load Initial File:** Click the **FirstFile** button. This opens a file dialog.
3.  **Select Data:** Navigate to your data folder and select the first `.tif` image of your scan (e.g., `DetZBeamPosScan_000004.tif`).
4.  **Auto-Populate:** The application automatically fills in the `Folder`, `FNStem` (File Name Stem), and `StartFileNumberFirstLayer` fields.
5.  **Display Image:** Click the main **Load** button to display the image in the left panel.

### 2. Image Display and Navigation

The left panel shows the detector image.

*   **Toolbar:** Use the Matplotlib toolbar at the bottom of the window to **Zoom** to a region of interest or **Pan** across the image.
*   **Contrast Adjustment:**
    *   **MinThresh / MaxThresh:** Set the minimum and maximum intensity values for the color map to enhance contrast.
    *   **LogScale:** Check this box to apply a logarithmic scale, which is useful for seeing both faint and bright features simultaneously.
*   Click **Load** after changing any settings to refresh the image.

### 3. Part I: Determining the Beam Center

This procedure helps you find the `(x, y)` pixel coordinates of the beam center for each detector distance.

1.  **Generate Horizontal Profile:**
    *   Click the **BoxOutHor** button.
    *   On the image, click and drag to draw a selection box around a horizontal diffraction line.
    *   The right panel will now show an intensity plot summed along the box's vertical axis.
2.  **Generate Vertical Profile:**
    *   Click the **BoxOutVer** button.
    *   Draw a selection box around a vertical feature.
    *   The right panel will show an intensity plot summed along the box's horizontal axis.
3.  **Find the Center:**
    *   Hover your mouse over the intensity plot in the right panel. The `x` and `y` coordinates are shown in the bottom right of the window.
    *   Identify the `x` value corresponding to the center of the slope or peak for both the horizontal and vertical profiles.
    *   Repeat this process for the left and right edges of the beam and average the values to get the final beam center `(x, y)` for the current detector distance.
4.  **Enter Beam Center Values:**
    *   Click the **BeamCenter** button.
    *   A new window will appear. Input the calculated `(x, y)` beam center for each `Distance #`.
    *   Enter the known `Difference in distances` between your detectors (e.g., `3000` for 3mm).
    *   Click **Press this once done** to save.

### 4. Part II: Determining Detector Position

This procedure uses a single diffraction spot to precisely calculate the sample-to-detector distance.

1.  **Enable Background Correction (Recommended):**
    *   To improve signal quality, check the **SubtMedian** box.
    *   If a median background file has not been calculated, you can do so by clicking **CalcMedian**.
2.  **Select Spots:**
    *   Click the **Select Spots** button. A guide will appear. Read it and click **Ready!**.
    *   Using the `DistanceNr` and `FrameNumber` fields, navigate to a clear diffraction spot on the first detector.
    *   Click precisely on the center of the spot.
    *   Click the **Confirm Selection** button that appears.
    *   A dialog will ask for the next distance. Enter the number (e.g., `1`) and click **Load**.
    *   The GUI will load the corresponding image for the next detector distance. Find the *same* diffraction spot and repeat the selection process.
    *   Continue for all detector distances.
3.  **Compute Distances:**
    *   After selecting the spot for all distances, click the **Compute Distances** button.
    *   A popup will display the three calculated distances and three calculated Y-positions. Record these values.

### 5. Advanced Features: Microstructure Analysis

Correlate your diffraction data with an existing microstructure map.

1.  **Load Mic File:** Click **LoadMic** and select your `.mic` or `.map` file. The map will appear in the right panel.
2.  **Visualize Data:** Use the radio buttons (`Confidence`, `GrainID`, `Euler0`, etc.) to change the coloring of the map.
3.  **Select a Point:**
    *   Click the **SelectPoint** button.
    *   Click on a grain of interest within the microstructure map.
4.  **Simulate Spots:**
    *   The grain's orientation and position data will automatically populate a "Load Grain" window. Click **Confirm**.
    *   Click the **MakeSpots** button. The GUI will calculate the theoretical diffraction spots for that grain.
    *   A red circle will appear on the detector image, showing the predicted position of a diffraction spot. You can cycle through predicted spots using the `+` button and the `SpotNumber` field.

---
If you encounter any issues or have questions, please open an issue on this repository.