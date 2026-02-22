# MIDAS FF-HEDM Desktop Image Viewer (`ff_asym.py`): User Manual

**Version:** 9.0  
**Contact:** hsharma@anl.gov

---

## 1. Introduction

The **MIDAS FF-HEDM Desktop Image Viewer** (`ff_asym.py`) is a lightweight Tkinter-based GUI application for inspecting raw detector images from FF-HEDM experiments. Unlike the interactive Dash-based plotter (see [FF_Interactive_Plotting.md](FF_Interactive_Plotting.md)), this tool focuses on **rapid, frame-by-frame inspection** of raw data at the beamline and during data reduction, supporting multiple file formats and real-time image processing.

**Key Capabilities:**

- View raw detector images from **binary (GE, custom)**, **HDF5**, **TIFF**, **bz2-compressed**, and **Zarr-ZIP (`.MIDAS.zip`)** files
- **Dark-field correction** with flexible HDF5 dataset path selection
- **Bad pixel masking** with an on/off toggle
- **Frame-by-frame navigation** through multi-frame files
- **Max and Sum projections** over arbitrary frame ranges
- **Threshold coloring** to highlight pixels below/above intensity thresholds
- **Logarithmic display** scaling
- **Ring overlay** for diffraction ring calibration verification
- **Image transformations**: horizontal flip, vertical flip, transpose

```mermaid
graph TD
    subgraph "Input Data"
        BIN["Binary Files (.ge, custom)"]
        H5["HDF5 Files (.h5, .hdf, .hdf5, .nxs)"]
        TIF["TIFF Files (.tif, .tiff)"]
        BZ2["Compressed (.bz2)"]
        ZIP["Zarr-ZIP (.MIDAS.zip)"]
        MASK["Mask File (int8 binary)"]
        DARK["Dark Image File"]
    end

    subgraph "ff_asym.py Core"
        GI["getImage()"]
        BIN --> GI
        H5 --> GI
        TIF --> GI
        BZ2 --> GI
        ZIP --> GI
        RM["readMask()"]
        MASK --> RM
        RM --> GI
        GD["getData() / getDataB()"]
        GI --> GD
        DARK --> GD
    end

    subgraph "Processing"
        GD --> MAX["getImageMax()"]
        GD --> SUM["getImageSum()"]
        GD --> SINGLE["Single Frame"]
    end

    subgraph "Display"
        MAX --> PLOT["Matplotlib Canvas"]
        SUM --> PLOT
        SINGLE --> PLOT
        PLOT --> THRESH["Threshold Coloring"]
        PLOT --> LOG["Log Scale"]
        PLOT --> RINGS["Ring Overlay"]
    end
```

---

## 2. Requirements

### 2.1. Software

| Package | Purpose | Required |
| :--- | :--- | :--- |
| Python 3.x | Runtime | **Yes** |
| numpy | Array operations | **Yes** |
| matplotlib | Plotting | **Yes** |
| tkinter | GUI framework | **Yes** (bundled with Python) |
| h5py | HDF5 file reading | For HDF5 files |
| tifffile | TIFF file reading | For TIFF files |
| zarr | Zarr-ZIP file reading | For `.MIDAS.zip` files |
| bz2, shutil | Compression handling | **Yes** (stdlib) |

### 2.2. Data Requirements

The viewer operates on **single detector files** containing one or more frames of 2D image data. Supported formats:

| Format | Extension(s) | Notes |
| :--- | :--- | :--- |
| Binary (GE) | `.ge1`–`.ge5`, custom | Fixed-size binary with header. Set `HeadSize` and `Bytes/Px`. |
| HDF5 | `.h5`, `.hdf`, `.hdf5`, `.nxs` | 2D `(Y, X)` or 3D `(frames, Y, X)` datasets. Dataset path configurable. |
| TIFF | `.tif`, `.tiff` | Single or multi-frame TIFF. Requires `tifffile`. |
| Zarr-ZIP | `.MIDAS.zip` | MIDAS archive produced by `ffGenerateZip.py`. Contains `exchange/data`, `exchange/dark`, and analysis parameters. |
| Compressed | `.bz2` | Any of the above, bz2-compressed. Decompressed transparently to a temp file. |

**Bad Pixel Mask:** A flat binary file of `int8` values with dimensions `NrPixelsVert × NrPixelsHor`. Values: `0` = good pixel, `1` = bad pixel.

---

## 3. Getting Started

### 3.1. Launching

```bash
cd /path/to/your/data_directory
python ~/opt/MIDAS/gui/ff_asym.py &
```

> [!TIP]
> Launch the GUI **from your data directory**. The GUI automatically scans the current working directory and pre-fills all file fields using the naming convention described below.

### 3.2. Automatic Filename Initialization

When launched from a data directory, `ff_asym.py` runs a background auto-detection thread that:

1. **Takes the directory name as the file stem.** If the CWD is `/path/to/ff_Holder3_50um/`, it looks for files starting with `ff_Holder3_50um_`.
2. **Finds data files** matching `<dir_name>_NNNNNN.<ext>` (e.g., `ff_Holder3_50um_000001.ge3`, `ff_Holder3_50um_000001.tif`).
3. **Extracts** the file stem, first file number, padding width, and file extension automatically.
4. **Finds dark files** by scanning for `dark_before_NNNNNN.<ext>` or `dark_after_NNNNNN.<ext>`. Prefers `dark_before` if both exist.
5. **Updates all GUI fields** (file stem, folder, first file number, extension, detector number, dark file settings) — no manual entry needed.

**The console will print the detection results at startup:**

```
Auto-detect: stem='ff_Holder3_50um', firstNr=1, padding=6, ext='ge3'
Auto-detect: dark='dark_before_000001', source=dark_before
Auto-detection complete. GUI updated.
```

**Expected directory layout:**

```
ff_Holder3_50um/                    ← directory name = file stem
├── ff_Holder3_50um_000001.ge3      ← data files (auto-detected)
├── ff_Holder3_50um_000002.ge3
├── ...
├── dark_before_000001.ge3          ← dark file (auto-detected, preferred)
└── dark_after_000001.ge3           ← fallback if dark_before not found
```

| Auto-detected field | Source |
|---|---|
| File stem | Directory name (e.g., `ff_Holder3_50um`) |
| First file number | Trailing number from first matching file (e.g., `1`) |
| Padding | Width of the number field (e.g., `6` for `000001`) |
| File extension | Extension of the first matching file (e.g., `ge3`, `tif`, `h5`) |
| Detector number | Extracted from GE extension (e.g., `3` from `.ge3`); `-1` for non-GE formats |
| Dark file | First `dark_before_*` file found; falls back to `dark_after_*` |
| Dark correction | Automatically enabled if dark file found |

> [!NOTE]
> If auto-detection fails (e.g., the directory name does not match the file naming pattern), the console prints a warning and all fields remain empty. You can then use the **FirstFile** button to select files manually.

### 3.3. Zarr-ZIP Auto-Detection and Loading

The viewer can open `.MIDAS.zip` archives produced by `ffGenerateZip.py`. These files contain the raw data, dark frames, and all analysis parameters in a single Zarr-ZIP archive.

**Auto-detection:** If a `*.MIDAS.zip` file exists in the current working directory, it takes priority over file-stem detection. The viewer automatically:

1. Opens the archive and reads `exchange/data` dimensions.
2. Reads `exchange/dark` and computes the mean dark frame (enables dark correction automatically).
3. Extracts detector and material parameters from `analysis/process/analysis_parameters/`.
4. Reads `ImTransOpt` and sets the HFlip/VFlip/Transpose checkbuttons accordingly (`1`=HFlip, `2`=VFlip, `3`=Transpose).
5. Runs `GetHKLListZarr` to generate `hkls.csv` if it doesn't already exist next to the zip file.
6. Pre-initializes ring overlays from the `RingThresh` entries stored in the archive.

**Console output on startup:**

```
Auto-detect: found Zarr-ZIP 'sample.MIDAS.zip'
  Data: 1440 frames, 2048x2048 pixels
  Dark: 10 frames, auto-enabled
  Lsd: 1000000.0
  BC: (1024.5, 1024.5)
  px: 200.0
  SpaceGroup: 225
  LatticeParameter: [5.411 5.411 5.411 90. 90. 90.]
  Wavelength: 0.172979
  ImTransOpt: [0] (HFlip=0, VFlip=0, Transpose=0)
  Running GetHKLListZarr to generate hkls.csv...
  hkls.csv generated successfully
  Rings pre-initialized: 8 rings from RingThresh
Zarr-ZIP loaded: /path/to/sample.MIDAS.zip
```

| Auto-loaded parameter | Zarr path |
|---|---|
| Detector distance (Lsd) | `analysis/process/analysis_parameters/Lsd` |
| Beam center Y | `analysis/process/analysis_parameters/YCen` |
| Beam center Z | `analysis/process/analysis_parameters/ZCen` |
| Pixel size | `analysis/process/analysis_parameters/PixelSize` |
| Space group | `analysis/process/analysis_parameters/SpaceGroup` |
| Lattice parameters | `analysis/process/analysis_parameters/LatticeParameter` |
| Wavelength | `analysis/process/analysis_parameters/Wavelength` |
| Image transforms | `analysis/process/analysis_parameters/ImTransOpt` |
| Ring thresholds | `analysis/process/analysis_parameters/RingThresh` |
| Tilts (tx, ty, tz) | `analysis/process/analysis_parameters/tx`, `ty`, `tz` |

**Manual loading:** You can also load a zip file at any time using the **Load ZIP** button in the File I/O panel. This opens a file dialog filtered to `*.zip` files.

> [!TIP]
> When loading a new zip file mid-session, any previously displayed ring overlays are automatically cleared before the new rings are shown.

### 3.4. Loading Your First Image (Manual)

If auto-detection populated the fields, you can click **Load Single** immediately. Otherwise:

1. Click **FirstFile** → select a data file (binary, HDF5, TIFF, or bz2).
2. For HDF5 files, set the **H5 Data** path (default: `/exchange/data`). Click **Browse** to browse the internal HDF5 structure.
3. Set **NrPixelsHor** and **NrPixelsVert** to match your detector dimensions.
4. For binary files, set **HeadSize** (e.g., `8192` for GE files) and **Bytes/Px** (`2` for uint16, `4` for int32).
5. Click **Load Single** to display the image.

```mermaid
flowchart LR
    A["Launch from data dir"] --> AUTO{"Auto-detect<br/>succeeded?"}
    AUTO --> |"Yes"| G["Click Load Single"]
    AUTO --> |"No"| B{"Select FirstFile"}
    B --> |"HDF5"| C["Set H5 Data path"]
    B --> |"Binary"| D["Set HeadSize, Bytes/Px"]
    B --> |"TIFF"| E["No extra config needed"]
    C --> F["Set NrPixels"]
    D --> F
    E --> F
    F --> G
    G --> H["Image Displayed"]
```

---

## 4. GUI Reference

The interface is organized into four `LabelFrame` control panels, each grouping related functionality.

### 4.1. File I/O Panel

| Control | Description |
| :--- | :--- |
| **FirstFile** | Opens a file dialog to select the primary data file. |
| **DarkFile** | Opens a file dialog to select the dark-field reference file. |
| **DarkCorr** | Checkbox to enable/disable dark-field subtraction. |
| **FirstFileNr** | First file number in a numbered file series. |
| **nFrames/File** | Number of frames per file (used for frame navigation). |
| **H5 Data** | HDF5 dataset path for the data images (e.g., `/exchange/data`). |
| **Browse** (Data) | Browse HDF5 file to select a dataset path interactively. |
| **H5 Dark** | HDF5 dataset path for the dark images (e.g., `/exchange/dark`). |
| **Browse** (Dark) | Browse HDF5 file to select a dark dataset path interactively. |
| **MaskFile** | Opens a file dialog to select a bad pixel mask file. |
| **ApplyMask** | Checkbox to enable/disable bad pixel masking. |
| **Load ZIP** | Opens a file dialog to load a `.MIDAS.zip` Zarr archive, auto-populating all parameters. |

#### 4.1.1. HDF5 Dark Correction: Special Cases

The viewer supports flexible dark correction for HDF5 files:

| Scenario | Configuration |
| :--- | :--- |
| **Same file, same dataset** | Select same H5 file for both FirstFile and DarkFile. Use same H5 path for both H5 Data and H5 Dark. |
| **Same file, different dataset** | Select same H5 file for both. Set **H5 Data** to `/exchange/data` and **H5 Dark** to `/exchange/dark` (for example). |
| **Different file, same dataset name** | Select different H5 files. Both H5 Data and H5 Dark can use the same path (e.g., `/exchange/data`). |
| **Different file, different dataset** | Select different H5 files. Set different paths for H5 Data and H5 Dark. |

### 4.2. Image Settings Panel

| Control | Description |
| :--- | :--- |
| **NrPixelsHor** | Horizontal detector size in pixels (default: 2048). |
| **NrPixelsVert** | Vertical detector size in pixels (default: 2048). |
| **HeadSize** | File header size in bytes (default: 8192, for GE files). |
| **Bytes/Px** | Bytes per pixel: `2` = uint16, `4` = int32 (default: 2). |
| **HFlip** | Flip image horizontally (left–right). |
| **VFlip** | Flip image vertically (top–bottom). |
| **Transp** | Transpose the image (swap rows and columns). |

### 4.3. Display Control Panel

| Control | Description |
| :--- | :--- |
| **FrameNr** | Current frame number to display (0-indexed). |
| **+** / **−** | Increment/decrement frame number and reload. |
| **MinThresh** | Minimum intensity threshold for display scaling. |
| **Color < Min** | When checked, pixels below MinThresh are colored **blue**. |
| **MaxThresh** | Maximum intensity threshold for display scaling. |
| **Color > Max** | When checked, pixels above MaxThresh are colored **red**. |
| **Update Plot** | Re-render the display with current threshold/color settings. |
| **LogScale** | Toggle logarithmic intensity scaling. |

### 4.4. Processing Panel

| Control | Description |
| :--- | :--- |
| **MaxOverFrames** | Compute pixel-wise maximum over a range of frames. |
| **SumOverFrames** | Compute pixel-wise sum over a range of frames. |
| **nFrames** | Number of frames to include in Max/Sum projection. |
| **StartFrame** | Starting frame number for Max/Sum projection. |
| **RingsMat** | Open dialog to specify ring material parameters (space group, wavelength, lattice constants). |
| **PlotRings** | Toggle diffraction ring overlay on the image. |
| **DetNum** | Detector number for multi-detector setups. |
| **Lsd** | Sample-to-detector distance (µm). |
| **BC** | Beam center coordinates (Y, Z) in pixels. |
| **Load Single** | Load and display a single frame from the current detector. |

---

## 5. Feature Details

### 5.1. Bad Pixel Masking

Bad pixel masks identify dead or hot detector pixels that should be excluded from analysis.

**Mask File Format:**
- Flat binary file of `int8` values
- Dimensions: `NrPixelsVert × NrPixelsHor` (row-major order)
- Values: `0` = good pixel, `1` = bad pixel

**Usage:**
1. Click **MaskFile** and select your mask file.
2. Check the **ApplyMask** checkbox.
3. Load or update the image — bad pixels will be set to `0`.

> [!NOTE]
> The mask is cached internally and only re-read when the mask file path, pixel dimensions, or image transformations (flip/transpose) change. This ensures efficient performance during frame-by-frame navigation and multi-frame projections.

### 5.2. Threshold Coloring

Threshold coloring provides visual highlighting of intensity outliers:

- **Color < Min (Blue):** All pixels with intensity below `MinThresh` are displayed in **blue**, making it easy to spot dead regions or background.
- **Color > Max (Red):** All pixels with intensity above `MaxThresh` are displayed in **red**, highlighting saturated or unusually bright pixels.

**Usage:**
1. Set `MinThresh` and `MaxThresh` values.
2. Check one or both of **Color < Min** and **Color > Max**.
3. Click **Update Plot** (or navigate to a new frame).

> [!TIP]
> Combine threshold coloring with **LogScale** to identify weak diffraction features against a noisy background. The blue/red coloring is applied via Matplotlib's `set_under()`/`set_over()` colormap methods.

### 5.3. Max/Sum Projections

These features allow you to compute pixel-wise aggregations over multiple frames, which is essential for identifying diffraction rings and checking data quality.

| Mode | Description |
| :--- | :--- |
| **MaxOverFrames** | For each pixel, take the maximum intensity value across all selected frames. Useful for seeing all diffraction spots in a single view. |
| **SumOverFrames** | For each pixel, sum the intensity values across all selected frames. Useful for enhancing weak features. |

Set **nFrames** and **StartFrame** to define the frame range, then check the desired mode and load the image.

### 5.4. HDF5 Dataset Path Browser

For HDF5 files, the internal dataset structure can be complex. The **Browse** buttons next to H5 Data and H5 Dark fields open an interactive tree browser that lists all groups and datasets within the selected HDF5 file, allowing you to pick the correct dataset path.

### 5.5. Ring Overlay

The ring overlay feature plots expected diffraction ring positions on top of the image, helping verify detector geometry calibration.

1. Click **RingsMat** and enter your material parameters (space group, wavelength, lattice constants, Lsd, pixel size, max ring radius).
2. Select which rings to display from the generated list.
3. Check **PlotRings** to toggle the overlay on/off.

---

## 6. Complete Workflow Flowchart

```mermaid
flowchart TD
    START["Launch ff_asym.py"] --> LOAD["Select FirstFile (Binary/HDF5/TIFF/bz2)"]
    
    LOAD --> FORMAT{"File Format?"}
    FORMAT --> |"HDF5"| H5CFG["Configure H5 Data path<br/>(click Browse)"]
    FORMAT --> |"Binary"| BINCFG["Set HeadSize, Bytes/Px"]
    FORMAT --> |"TIFF"| TIFCFG["No extra config"]
    FORMAT --> |"bz2"| BZ2["Auto-decompress → recurse"]
    
    H5CFG --> IMGCFG
    BINCFG --> IMGCFG
    TIFCFG --> IMGCFG
    BZ2 --> FORMAT
    
    IMGCFG["Set NrPixels, Flip/Transpose"] --> DARKQ{"Dark Correction?"}
    DARKQ --> |"Yes"| DARKSEL["Select DarkFile<br/>Set H5 Dark path<br/>Check DarkCorr"]
    DARKQ --> |"No"| MASKQ
    DARKSEL --> MASKQ
    
    MASKQ{"Apply Mask?"} --> |"Yes"| MASKCFG["Select MaskFile<br/>Check ApplyMask"]
    MASKQ --> |"No"| DISPLAY
    MASKCFG --> DISPLAY
    
    DISPLAY["Click Load Single<br/>or use +/- buttons"] --> PROC{"Processing Mode?"}
    PROC --> |"Single Frame"| VIEW["View Frame"]
    PROC --> |"MaxOverFrames"| MAXPROC["Compute Max Projection"]
    PROC --> |"SumOverFrames"| SUMPROC["Compute Sum Projection"]
    
    VIEW --> ENHANCE
    MAXPROC --> ENHANCE
    SUMPROC --> ENHANCE
    
    ENHANCE["Adjust Display"] --> THRESH["Set Min/Max Thresholds"]
    THRESH --> COLOR{"Threshold Coloring?"}
    COLOR --> |"Color < Min"| BLUE["Blue highlights"]
    COLOR --> |"Color > Max"| RED["Red highlights"]
    COLOR --> |"Both"| BOTH["Blue + Red"]
    COLOR --> |"Neither"| PLAIN["Standard colormap"]
    
    BLUE --> LOG
    RED --> LOG
    BOTH --> LOG
    PLAIN --> LOG
    
    LOG{"LogScale?"} --> |"Yes"| LOGVIEW["Logarithmic display"]
    LOG --> |"No"| LINVIEW["Linear display"]
    
    LOGVIEW --> RINGS
    LINVIEW --> RINGS
    
    RINGS{"Ring Overlay?"} --> |"Yes"| RINGCFG["Configure material<br/>Select rings<br/>Check PlotRings"]
    RINGS --> |"No"| DONE["Rendered Image"]
    RINGCFG --> DONE
```

---

## 7. Troubleshooting

| Problem | Solution |
| :--- | :--- |
| **Blank/white image** | Check `NrPixelsHor`/`NrPixelsVert` match your detector. For binary files, verify `HeadSize` and `Bytes/Px`. |
| **HDF5 dataset not found** | Use the **Browse** button to browse the internal structure. Ensure the path matches (e.g., `/exchange/data`). |
| **Image appears rotated** | Toggle **HFlip**, **VFlip**, or **Transp** to match your detector orientation. |
| **Mask not applying** | Ensure **ApplyMask** checkbox is checked and the mask file dimensions match `NrPixelsVert × NrPixelsHor`. |
| **Frame navigation not working** | Check that `nFrames/File` is set correctly. For HDF5, the frame count is determined by the dataset's first dimension. |
| **Blue/Red coloring not visible** | Verify **Color < Min** / **Color > Max** are checked and click **Update Plot**. |
| **Import error for tifffile** | Install with `pip install tifffile`. TIFF support is optional. |
| **Import error for h5py** | Install with `pip install h5py`. Required only for HDF5 files. |

---

## 8. See Also

- [FF_Interactive_Plotting.md](FF_Interactive_Plotting.md) — Dash-based interactive viewer for post-analysis visualization
- [FF_Analysis.md](FF_Analysis.md) — Standard FF-HEDM data reduction workflow
- [FF_calibration.md](FF_calibration.md) — Detector geometry calibration
- [README.md](README.md) — MIDAS manual index

---

If you encounter any issues or have questions, please open an issue on this repository.
