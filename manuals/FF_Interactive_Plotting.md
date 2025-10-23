**MIDAS FF-HEDM Interactive Viewer: User Manual**

**1\. Introduction**

The **MIDAS FF-HEDM Interactive Viewer** is a powerful web-based application designed for the comprehensive visualization and analysis of data from Far-Field High-Energy Diffraction Microscopy (ff-HEDM) experiments. Built with Python and the Dash framework, this tool allows researchers and material scientists to interactively explore complex, multi-dimensional datasets, bridging the gap between raw experimental output and insightful scientific discovery.

The viewer provides a suite of linked, interactive plots that display grain structures, individual diffraction spots, and reciprocal space data. Users can dynamically filter, color, and inspect grains and spots based on a wide range of properties, and even drill down to view the raw detector image data for a specific diffraction event.

**Key Features:**

- **3D Visualization:** View the spatial arrangement of all grains and diffraction spots.
- **Reciprocal Space Plotting:** Analyze G-vectors to understand crystal lattice orientations.
- **Dynamic Filtering:** Isolate grains of interest using sliders for properties like Eta, 2-Theta (tth), and Omega angles.
- **Interactive Selection:** Click on a grain to instantly view its associated diffraction spots in multiple projections.
- **Raw Data Inspection:** Click on a spot to see a 3D volume rendering of the raw detector data that produced it.
- **Detailed Data Table:** Inspect and sort the numerical data for all spots belonging to a selected grain.

**2\. Requirements**

Before running the application, ensure you have the necessary software libraries and that your data is structured correctly.

**2.1. Software Requirements**

The application is a Python script and requires several libraries to be installed. You can install them using pip:

codeCode

pip install dash dash-bootstrap-components pandas numpy plotly zarr

- **dash & dash-bootstrap-components:** The core framework for building the web application.
- **pandas & numpy:** Used for efficient data manipulation and numerical operations.
- **plotly:** The underlying library for generating the interactive charts and graphs.
- **zarr:** Required for reading the compressed Zarr data archives that contain the raw experimental parameters and detector images.

**2.2. Data Requirements (Deep Dive)**

The application requires a specific set of input files, passed via command-line arguments. The data must be organized into a results folder and a Zarr data file.

**A. Zarr Data File (-dataFileName)**

This must be a single, zip-compressed Zarr file (.zarr) containing essential experimental parameters and the raw detector data. The script expects the following internal Zarr group/dataset structure:

- /analysis/process/analysis_parameters/PixelSize: Pixel size in microns.
- /analysis/process/analysis_parameters/Wavelength: X-ray wavelength.
- /analysis/process/analysis_parameters/Lsd: Sample-to-detector distance.
- /analysis/process/analysis_parameters/RingThresh: Intensity threshold for spot detection.
- /analysis/process/analysis_parameters/ImTransOpt: Image transformation options.
- /measurement/process/scan_parameters/step: Omega rotation step size in degrees.
- /measurement/process/scan_parameters/start: Starting omega angle in degrees.
- /exchange/data: The primary 3D numpy array of raw detector images (Frames, Z, Y).
- /exchange/dark: The dark-field image data used for background correction.

**B. Results Folder (-resultFolder)**

This folder should contain the output files from the MIDAS reconstruction software. The viewer specifically requires the following CSV files:

**1. Grains.csv**

- **Description:** Contains information about each reconstructed grain.
- **Format:** A CSV file with a **9-line header** that the script skips.
- **Required Columns (by 0-based index):**
  - 0: Grain ID
  - 10: X-position of the grain centroid.
  - 11: Y-position of the grain centroid.
  - 12: Z-position of the grain centroid.
  - 19: Reconstruction error value.
  - 22: Grain size.
  - 23: Grain completeness/confidence score.
  - \-5 (5th from end): Strain error.
  - \-3, -2, -1 (last 3): Euler angles (0, 1, 2).

**2. SpotMatrix.csv**

- **Description:** Contains detailed information for every diffraction spot matched to a grain.
- **Format:** A CSV file with a **1-line header**.
- **Required Columns (by 0-based index):**
  - 0: Grain ID the spot belongs to.
  - 1: Unique Spot ID.
  - 2: Omega angle (fitted).
  - 3: Detector Y-position (pixels).
  - 4: Detector Z-position (pixels).
  - 5: Omega angle (raw, from frame number).
  - 6: Eta angle.
  - 7: Diffraction ring number.
  - 8: Y-position in microns.
  - 9: Z-position in microns.
  - 10: 2-Theta angle (tTheta). _Note: The script assumes this is Theta and multiplies by 2._
  - 11: Strain value for the spot.

**3. InputAll.csv**

- **Description:** Contains original information about all identified spots before indexing. This file is used to get the spot size. While the application will run without it, spot size information will be missing.
- **Format:** A CSV file with a **1-line header**.
- **Required Column (by 0-based index):**
  - 3: Spot size (e.g., number of pixels).

**3\. Getting Started**

To launch the interactive viewer, navigate to the directory containing the script in your terminal and run it with the appropriate command-line arguments.

**Command:**

codeCode

python &lt;script_name&gt;.py -resultFolder /path/to/your/results -dataFileName /path/to/your/data.zarr

**Arguments:**

- \-resultFolder: **(Required)** The full path to the folder containing Grains.csv, SpotMatrix.csv, etc.
- \-dataFileName: **(Required)** The full path to the Zarr file containing the raw data and parameters.
- \-HostName: (Optional) The IP address to host the application on. Defaults to 0.0.0.0, which makes it accessible on your local network.
- \-portNr: (Optional) The port number for the application. Defaults to 8050.

After running the command, you will see a message like Starting Dash server on <http://0.0.0.0:8050>. Open this URL in your web browser to access the viewer.

**4\. User Interface and Functionality Guide**

The application is organized into several linked sections. Interacting with one plot or control will often update others, allowing for a seamless analysis workflow.

**4.1. Global Views (Top Row)**

This section provides a high-level overview of all the spots in the dataset.

- **All Spots 3D View (Left):** A 3D scatter plot of every spot from SpotMatrix.csv, plotted in (Omega, Detector Y, Detector Z) coordinates. This shows the spatial distribution of diffraction events.
- **G-Vectors Reciprocal Space (Right):** A 3D scatter plot of the calculated G-vectors for each spot. This plot represents the data in reciprocal space, which is crucial for analyzing crystal orientations.
- **Controls:**
  - **Spot Color:** Choose a property to color the spots by (e.g., ringNr, grainIDColor, strain).
  - **Select Rings:** Use the checklists to show or hide spots belonging to specific diffraction rings.

**4.2. Grain Filtering and Controls (Middle Section)**

This section contains the primary controls for filtering the grains displayed in the plots below it.

- **Filter Grains by Spot Properties:**
  - **Eta, 2θ, and Omega Range Sliders:** Drag the sliders to define a range for these properties. The "Filtered Grains" plot will update to show only those grains that have at least one spot falling within _all_ selected ranges.
  - **Show Grain ID:** Enter a specific Grain ID to isolate and view only that grain. This filter is applied _in addition_ to the range slider filters.
- **Grain Color (3D Map):** Use the radio buttons to select which property is used to color the grains in the "Filtered Grains" plot (e.g., Confidence, GrainSize, IDColor).
- **Filtered Spot Color:** Use these radio buttons to control the coloring of spots in the grain-specific plots to the right ("Filtered Spots" 3D and 2D).

**4.3. Filtered and Selected Views (Main Plots)**

These plots display the data after the filters from the middle section have been applied. This is where you perform detailed investigation of specific grains.

- **Filtered Grains (Bottom-Left):**
  - **Functionality:** A 3D scatter plot showing the centroids of all grains that match the current filter criteria.
  - **Interaction:** **Click on a single grain** in this plot. This action is the primary trigger for updating all other grain-specific views.
- **Filtered Spots 3D (Bottom-Right of Grains plot):**
  - **Functionality:** Displays a 3D scatter plot (Omega, Y, Z) of all spots belonging to the single grain you clicked on in the "Filtered Grains" plot.
  - **Title:** The plot title provides the selected Grain ID and its mean/median strain.
- **Filtered Spots 2D (Below Grains plot):**
  - **Functionality:** A 2D scatter plot (Detector Y, Detector Z) of the spots for the selected grain. This represents the pattern of spots on the detector for that grain.
  - **Interaction:** **Click on a single spot** in this plot to load the raw data in the "Volume Plot".
- **Image Data / Volume Plot (Bottom-Right):**
  - **Functionality:** After clicking a spot in the 2D view, this plot loads and displays a 3D volume rendering of the raw detector data from the Zarr file. It shows a small window in frames (omega), detector Y, and detector Z centered on the selected spot.
  - **Purpose:** This allows you to visually inspect the raw diffraction signal and verify the quality of the spot.

**4.4. Spot Details Table (Bottom)**

- **Functionality:** A detailed table listing all spots for the currently selected grain. It populates after a grain is clicked in the "Filtered Grains" plot.
- **Features:**
  - **Sorting:** Click on column headers to sort the data.
  - **Filtering:** Type in the filter boxes below the headers to find specific values (e.g., show only spots with ringNr 5).
  - **Pagination:** Use the "Table Rows" dropdown to control how many spots are shown per page or to show all spots at once.

**5\. Example Workflow**

- **Initial Exploration:** Start by examining the **Global Views** at the top. Use the "Spot Color" and "Select Rings" options to get a feel for the overall dataset.
- **Filter for Grains of Interest:** Use the **Eta, 2θ, and Omega range sliders** to narrow down the grains shown in the "Filtered Grains" plot. For example, you might want to see only grains that contributed to a specific diffraction ring at low angles.
- **Select a Grain:** In the **"Filtered Grains"** plot, click on a grain that looks interesting (e.g., one with high Error or low Confidence).
- **Analyze the Grain's Spots:**
  - Observe the **"Filtered Spots 3D"** and **"2D"** plots that have now appeared. Use the "Filtered Spot Color" radio buttons to color the spots by strain to see the strain distribution within the grain.
  - Examine the **Spot Details Table** to see the exact numerical values for each spot. You can sort by strain to quickly find the most strained spots.
- **Inspect Raw Data:** In the **"Filtered Spots 2D"** plot, find a spot of interest (perhaps one with very high strain) and click on it.
- **Verify the Spot:** The **"Image Data"** plot will now show the 3D volume of the raw detector intensity around that spot. This helps confirm that it is a genuine, strong diffraction peak.
- **Repeat:** Select a new grain or adjust the filters to continue your exploration.
