# Comprehensive User Manual for the Near-Field Calibration GUI

**Version:** 9.0  
**Contact:** hsharma@anl.gov

## Introduction: The Goal of Calibration

The purpose of this software and the procedures outlined below is to perform a near-field calibration. The goal is to determine the exact X-ray beam position on the detector and the precise sample-to-detector distance for multiple detector positions. This is accomplished using a known calibrant sample, such as single-crystal gold (Au).

This manual is divided into two primary sections, mirroring the calibration workflow:
*   **Part I: Determining the Beam Center:** Finding the central pixel of the direct beam at each detector distance.
*   **Part II: Determining Detector Positions:** Using diffraction spots from a gold standard to calculate the precise distance from the sample to each detector.

---

## Getting Started: Loading Data

> [!IMPORTANT]
> This procedure assumes 5mm as the first detector distance.


1.  **Launch the Application:**
    *   Open the GUI by executing the following command in your terminal:
        ```bash
        python ~/opt/MIDAS/gui/nf.py &
        ```

2.  **Load the Beam Position Scan File:**
    *   Click the **`FirstFile`** button.
    *   Navigate to your data folder. For Part I, select the Au 5mm beam position scan file. The filename stem will typically be **"DetZBeamPosScan"**.
    *   If no "DetZBeamPosScan" files exist, load the first image from your "Au_NF" folder for that distance.
    *   The GUI will auto-populate the `Folder`, `FNStem`, and `StartFileNumberFirstLayer` fields.

3.  **Display the Image:**
    *   Click the large **`Load`** button. The detector image will appear in the left-hand window.

---

## Part I: Determining the Beam Center at Each Detector Distance

In this part, we will find the average horizontal and vertical center of the beam for each detector position (e.g., 5mm, 7mm, 9mm). The detector image will show the beam as one or two white horizontal lines.
 
 ```mermaid
 graph TD
     subgraph "Beam Center Workflow"
         Start[Load DetZBeamPosScan] --> Identify{Identify Beam Lines}
         Identify -->|Two Lines| Lower[Use Lower Line]
         Identify -->|One Line| Single[Use Single Line]
         Lower --> EdgeL[Analyze Left Edge]
         Single --> EdgeL
         EdgeL --> BoxHorL[BoxOutHor: Get X_left]
         EdgeL --> BoxVerL[BoxOutVer: Get Y_left]
         BoxHorL --> EdgeR[Analyze Right Edge]
         BoxVerL --> EdgeR
         EdgeR --> BoxHorR[BoxOutHor: Get X_right]
         EdgeR --> BoxVerR[BoxOutVer: Get Y_right]
         BoxHorR --> Avg[Calculate Average (X, Y)]
         BoxVerR --> Avg
         Avg --> Next[Repeat for Next Distance]
     end
 ```

*   If a **`DetZBeamPosScan`** was used, you will see a **single horizontal diffraction line**.

> [!TIP]
> It is recommended to use the `DetZBeamPosScan` at 0 and 90 degrees to compute the rotation axis position.
*   If using standard **Au scan data**, you may see **two horizontal lines**. For this calibration, **use the lower of the two lines.**

We will determine the center of the left and right edges of this line and average the results.

#### **Step 1: Analyze the Left Edge of the Beam (5mm distance)**

**(a) Find the Horizontal Center of the Left Edge:**
1.  Click the **`BoxOutHor`** button. Your cursor is now active for selection on the image.
2.  On the detector image, click a point to the **top-left** of the beam's left edge.
3.  Click a second point to the **bottom-right** of the beam's left edge, ensuring your box fully encompasses the edge.
4.  An intensity profile will appear in the right-hand plot. This curve shows the integrated intensity vs. horizontal pixel position.
5.  Hover your mouse over this curve. The coordinates (`x`, `y`, `z`) are displayed in the bottom-left of the GUI window.
6.  Identify the **`x` value** that corresponds to the approximate center of the slope on the curve. This is the horizontal position of the left edge. **Record this value.**

**(b) Find the Vertical Center of the Left Edge:**
1.  Click the **`BoxOutVer`** button.
2.  On the detector image, draw a similar box around the left edge of the beam.
3.  A new intensity profile will appear on the right, showing intensity vs. vertical pixel position.
4.  Hover your mouse over the curve and identify the **`x` value** (which represents the vertical axis in this plot) at the center of the peak. **Record this value.**

#### **Step 2: Analyze the Right Edge of the Beam (5mm distance)**

1.  Repeat the steps from **(a)** and **(b)** for the **right edge** of the beam.
2.  Record the horizontal and vertical center values for the right edge.

#### **Step 3: Calculate the Beam Center at 5mm**

1.  **Average Horizontal Center:** (Left Edge Horizontal Value + Right Edge Horizontal Value) / 2
2.  **Average Vertical Center:** (Left Edge Vertical Value + Right Edge Vertical Value) / 2
3.  **Record these two final average values.** They are the beam center for the 5mm detector distance.

> [!NOTE]
> **If using AU scans**, use the `BoxOutHor` button to plot the intensity along the x-ray beam to determine an intensity **dip** across the Au-sample. Compute the middle of this dip and use that as the **Average Horizontal Center**.

#### **Step 4: Repeat for Other Detector Distances (7mm and 9mm)**

1.  In the **`DistanceNr`** box, change the value from `0` to `1` (for the second distance, e.g., 7mm).
2.  Click the **`Load`** button. The image for the next detector position will load.
3.  Repeat **Steps 1, 2, and 3** to find the average beam center for this distance.
4.  Change **`DistanceNr`** to `2` (for the third distance, e.g., 9mm), click **`Load`**, and repeat the process again.

You should now have three pairs of (horizontal, vertical) beam center coordinates, one for each detector distance.

---

## Part II: Determining the Detector Position at Each Distance

Now we will use the gold calibration scan data and the beam center values from Part I to calculate the precise detector distances.
 
 ```mermaid
 graph TD
     subgraph "Detector Position Workflow"
         Start[Load Gold Calibration Data] --> EnterBC[Enter Beam Centers]
         EnterBC --> InputBCs[Input: Horizontal/Vertical Centers, DistDiff]
         InputBCs --> ConfirmBC[Confirm & Close]
         ConfirmBC --> Select[Select Spots Mode]
         Select --> Pick1[Pick Spot on Det 1]
         Pick1 --> Pick2[Pick SAME Spot on Det 2]
         Pick2 --> Pick3[Pick SAME Spot on Det 3]
         Pick3 --> Compute[Click Compute Distances]
         Compute --> Result[Output: Precise Detector Distances]
     end
 ```
 
 #### **Step 5: Load the Gold Calibration Data**


1.  Click **`FirstFile`**.
2.  Select the **first gold calibration scan file**. The folder is typically "Au_NF", and you should choose the first image number for the 5mm detector distance.
3.  Ensure **`DistanceNr`** is set to `0` and click **`Load`**.

#### **Step 6: Enter All Calculated Center Values**


1.  Click the **`BeamCenter`** button. A new window titled "Enter beam center values (pixels)" will open.
2.  **Left Column:** Enter the three **horizontal center values of the Au sample** you just calculated in part I (one for each distance).
3.  **Right Column:** Enter the three **average vertical beam center values** you calculated in Part I (one for each distance).
4.  **Difference in distances:** Enter the approximate distance between your detectors in microns (e.g., `2000` for 2mm).
5.  Click **"Press this once done"**.

#### **Step 7: Select Diffraction Spots**

1.  Click the **`Select Spots`** button. A help window will appear. Read the instructions and click **`Ready!`**.
2.  Make sure you are viewing the last detector distance (`DistanceNr` = `2`, e.g., 9mm).
3.  Use the **`FrameNumber`** box or the **`+`** and **`-`** buttons to cycle through the scan images until you find a clear, strong diffraction spot..
4.  Adjust the **`MaxThresh`** value to ensure the spot is bright but not saturated.
5.  Click on the pixel with the maximum intensity in the diffraction spot.
6.  Click the **`Confirm Selection`** button that appears.

#### **Step 8: Select the Same Spot on Other Detectors**

1.  A dialog box will pop up. Enter `1` as the new distance and click **`Load`**. The GUI will now show the middle detector distance.
2.  The diffraction spot will have moved. Use the zoom tool (magnifying glass in the toolbar) if needed to find the same spot.
3.  Click on the pixel with the max intensity in this new location. Click **`Confirm Selection`**.
4.  The dialog box will pop up again. Enter `0` as the new distance and click **`Load`**.
5.  Find and click the same spot on this first detector. Click **`Confirm Selection`**.
6.  In the final pop-up, click **`Finished`**.

#### **Step 9: Compute Final Distances**

1.  A **`Compute Distances`** button will now be active. Click it.
2.  A final pop-up box will appear, listing the three precisely calculated detector distances and three calculated Y-positions.
3.  **Record these values.**

**The calibration is now complete.**

---

## Part III: Technical Implementation Details

### 1. Beam Center Determination
The beam center finding logic relies on **1D Integrated Profiling**. When you draw a box using `BoxOutHor` or `BoxOutVer`:
*   The software sums the pixel intensities along the orthogonal direction (e.g., summing rows for a horizontal profile).
*   This improves the Signal-to-Noise Ratio (SNR) significantly compared to looking at single pixel lines.
*   The "edges" of the beam are determined by the steep gradients in this integrated profile, allowing for sub-pixel visual estimation of the beam center (centroid of the rectangular beam profile).

### 2. Distance Calibration Algorithm
The `Compute Distances` function employs a **Ray-Triangulation** method based on the intercept theorem.
*   **Assumption:** Diffraction occurs at a point source (the sample) and expands as a cone.
*   **Input:** Spot radii ($R_1, R_2, ...$) measured from the beam center at known relative detector shifts ($\Delta d$).
*   **Calculation:** For every pair of detector positions ($i, j$), the Sample-to-Detector distance to the first position ($LSD_0$) is calculated as:
    $$ LSD_0 = \frac{R_i \cdot (j-i) \cdot \Delta d}{R_j - R_i} - (i \cdot \Delta d) $$
*   **Result:** The final reported distance is the arithmetic mean of the calculated $LSD$ values from all possible pairs of detector positions, providing a robust estimate that minimizes random measurement errors.

---

## Appendix: Additional GUI Features Quick Reference

| Button/Field        | Function                                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| **`Load`**              | Loads or reloads the image display based on the current settings (file, frame, thresholds, etc.).        |
| **`MinThresh`/`MaxThresh`** | Adjusts the minimum/maximum intensity values for the image display (contrast).                     |
| **`LogScale`**          | Toggles a logarithmic color scale for the image, useful for seeing faint features.                       |
| **`LineOutHor/Vert`**   | Plots a 1-pixel wide line profile (less common than `BoxOut`).                                           |
| **`LoadMic`**           | Loads a microstructure (`.mic` or `.map`) file for display in the right-hand panel.                      |
| **`SelectPoint`**       | Allows you to click on the loaded mic file to extract grain information for simulation.                  |
| **`LoadGrain`**         | Opens a window to manually input crystal orientation, position, and lattice parameters for simulation. |
| **`MakeSpots`**         | Simulates diffraction spots based on the currently loaded grain information.                             |
| **`SubtMedian`**        | Toggles the subtraction of a pre-calculated median background image.                                     |
| **`CalcMedian`**        | Calculates a median background image from a specified number of frames (`nFilesMedianCalc`).           |

---

## See Also

- [NF_Analysis.md](NF_Analysis.md) — Single-resolution NF-HEDM reconstruction
- [NF_MultiResolution_Analysis.md](NF_MultiResolution_Analysis.md) — Multi-resolution iterative NF-HEDM reconstruction
- [NF_gui.md](NF_gui.md) — NF-HEDM GUI reference
- [README.md](README.md) — High-level MIDAS overview and manual index

---

If you encounter any issues or have questions, please open an issue on this repository.