# Near-Field HEDM Calibration GUI

**Version:** 9.1  
**Contact:** hsharma@anl.gov

This is a graphical user interface (GUI) for visualizing and analyzing near-field High-Energy Diffraction Microscopy (HEDM) data. The primary purpose of this tool is to perform a near-field calibration to determine the precise beam position and detector distances using a known calibrant sample, such as single-crystal gold.

**DETAILED INSTRUCTIONS FOR CALIBRATION ARE AT:** [NF_Calibration.md](NF_Calibration.md)

## Features

*   **Image Visualization:** Load and display `.tif` diffraction images with adjustable contrast and logarithmic scaling.
*   **Profile Analysis:** Generate horizontal and vertical intensity line profiles to precisely locate diffraction features.
*   **Beam Center Calibration:** A guided workflow to determine the beam center at multiple detector distances.
*   **Detector Distance Calculation:** Tools to select corresponding diffraction spots across different distances and compute the sample-to-detector distance.
*   **Microstructure Correlation:** Load and visualize `.mic` or `.map` files, and correlate grain orientations with observed diffraction spots.
*   **Diffraction Simulation:** Simulate expected diffraction patterns for a given set of grain and experimental parameters.
 
 ```mermaid
 graph LR
     GUI[NF GUI Main]
     GUI --> Viz[Image Visualization]
     GUI --> Prof[Profile Analysis]
     GUI --> Calib[Calibration]
     GUI --> Recon[Reconstruction Viewer]
     GUI --> Sim[Simulation]
     
     Calib --> Beam[Beam Center]
     Calib --> Dist[Detector Distance]
     
     Recon --> MapFile[".map — binary grid (fast, preferred)"]
     Recon --> MicFile[".mic — text scatter (slower)"]
     
     Sim --> Grain[Load Grain / SelectPoint]
     Sim --> Spots[MakeSpots]
 ```
 
 ## Getting Started

### Prerequisites

Ensure you have a working Python environment with the following libraries installed:
*   Tkinter
*   Pillow (PIL)
*   Matplotlib
*   NumPy
*   tifffile (optional, recommended for faster TIFF loading)

### Launching the Application

To run the GUI, execute the following command from your terminal:

```bash
python ~/opt/MIDAS/gui/nf.py &
```

> [!TIP]
> Launch the GUI from your data directory (e.g. `cd Au_NF_5mm && python ~/opt/MIDAS/gui/nf.py &`). The GUI will auto-detect the folder, file stem, and starting frame number from the `.tif` files present.

---

## User Guide

### 1. Loading Data

1.  **Launch the Application:** Run the command above to open the GUI.
2.  **Auto-Detection:** When launched from a data directory, the GUI automatically sets `Folder`, `FNStem`, and `StartFileNumberFirstLayer` by scanning for `.tif` files. You can skip the `FirstFile` step in this case and go directly to **Load**.
3.  **Load Initial File (alternative):** Click the **FirstFile** button to open a file dialog and manually select your first `.tif` image.
4.  **Display Image:** Click the main **Load** button to display the image in the left panel.

#### BeamPos / DetZBeamPos Folder Mode

When the GUI is launched from a folder whose name contains **`BeamPos`** or **`DetZBeamPos`**, it activates a special navigation mode:

*   All `.tif` files in the folder are collected and sorted by their numeric suffix.
*   The `FrameNumber` field becomes an **index** (0, 1, 2, …) into this sorted list.
*   The `+`/`-` buttons step through files sequentially, regardless of file stem.
*   Median background subtraction is not available in this mode (no matching median files).
*   The console will print `BeamPos mode: found N files, navigating by index` at startup.

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

### 5. Advanced Features: Microstructure Analysis and Simulation

Load a reconstructed microstructure map to visualize results and simulate diffraction spots for individual grains.

#### 5.1 Loading Reconstruction Results

1.  **Load Mic File:** Click **LoadMic** and select your reconstruction output file. Two formats are supported:

    | Format | Extension | Rendering | Notes |
    |--------|-----------|-----------|-------|
    | **Binary map** | `.map` | `imshow` (fast) | Space-filling binary grid. Auto-generated by MIDAS reconstruction. **Preferred format.** |
    | **Text mic** | `.mic` | `scatter` (slower) | Sparse point-based format. Works but slower for large reconstructions. |

    > [!TIP]
    > Always use the `.map` file when available. It renders via `imshow` (pixel grid) rather than `scatter` (individual points), giving significantly faster display and interaction, especially for large reconstructions.

2.  **Visualize Data:** Use the radio buttons (`Confidence`, `GrainID`, `Euler0`, etc.) to change the coloring of the map.

#### 5.2 Diffraction Spot Simulation

1.  **Select a Point:**
    *   Click the **SelectPoint** button.
    *   Click on a grain of interest within the microstructure map.
2.  **Simulate Spots:**
    *   The grain's orientation and position data will automatically populate a "Load Grain" window. Click **Confirm**.
    *   Click the **MakeSpots** button. The GUI will calculate the theoretical diffraction spots for that grain.
    *   A red circle will appear on the detector image, showing the predicted position of a diffraction spot. You can cycle through predicted spots using the `+` button and the `SpotNumber` field.

---

## 6. Technical Implementation Details

### 6.1. Software Architecture
*   **Framework:** The GUI is built using **Tkinter** for the window management and control widgets (buttons, entries).
*   **Visualization:** **Matplotlib** figures are embedded directly into the Tkinter application using the `FigureCanvasTkAgg` backend. This allows for interactive features like zooming and panning within the GUI window.
*   **Image Processing:** Raw diffraction images (TIFF) are read using **tifffile** (preferred) or **Pillow** into **NumPy** arrays for efficient manipulation. Features like median background subtraction are implemented as vectorized array operations for real-time performance.
*   **Performance Optimizations (v9.1):**
    *   **Artist reuse:** When stepping through frames with `+`/`-`, the imshow artist is updated via `set_data()`/`set_clim()` instead of being recreated, avoiding expensive axis rebuild.
    *   **Deferred rendering:** `canvas.draw_idle()` defers render to the next idle event, preventing UI freezes.
    *   **Safe colorbar removal:** Colorbar cleanup is wrapped in exception handlers to prevent stale-axes crashes.
    *   **LoadMic default directory:** The file dialog defaults to the current working directory.

### 6.2. Simulation Backend
The "MakeSpots" simulation feature uses a hybrid approach:
*   **Frontend:** The GUI collects parameters (orientation, position, lattice constants) and writes them to a temporary parameter file in shared memory (`/dev/shm/ps.txt`).
*   **Backend:** It invokes the high-performance C binaries (`GetHKLList`, `SimulateDiffractionSpots`) via system calls.
*   **Data Exchange:** The heavy simulation results (spot positions) are written to `/dev/shm` by the C binaries and then read back by Python for plotting. This shared-memory architecture facilitates low-latency visualization of complex simulations.

---

## 7. See Also

- [NF_Analysis.md](NF_Analysis.md) — Single-resolution NF-HEDM reconstruction
- [NF_MultiResolution_Analysis.md](NF_MultiResolution_Analysis.md) — Multi-resolution iterative NF-HEDM reconstruction
- [NF_Calibration.md](NF_Calibration.md) — Detailed step-by-step calibration procedure
- [README.md](README.md) — High-level MIDAS overview and manual index

---

If you encounter any issues or have questions, please open an issue on this repository.