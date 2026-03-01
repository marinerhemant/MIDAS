# gui/ — Interactive Visualization GUIs

This directory contains interactive desktop applications for visualizing and analyzing MIDAS data. Two generations of viewers are available: modern PyQtGraph-based viewers (recommended) and legacy Tkinter-based viewers.

---

## Applications

### Modern PyQtGraph Viewers (Recommended)

#### FF-HEDM Viewer (`ff_asym_qt.py`)

```bash
cd <data_directory> && python ~/opt/MIDAS/gui/ff_asym_qt.py &
```

Fast PyQtGraph-based FF-HEDM viewer with navigation toolbar, P2–P98 auto-scaling, live ring overlays, dark subtraction, HDF5 browsing, log scale, and export PNG. See [GUIs_and_Visualization](../manuals/GUIs_and_Visualization.md) §1.

#### NF-HEDM Viewer (`nf_qt.py`)

```bash
cd <data_directory> && python ~/opt/MIDAS/gui/nf_qt.py &
```

NF-HEDM viewer with all FF features plus microstructure overlay (`.mic`/`.map`), spot simulation, BoxH/BoxV ROI tools, and beam center calibration. See [NF_GUI manual](../manuals/NF_GUI.md) and [GUIs_and_Visualization](../manuals/GUIs_and_Visualization.md) §2.

---

### Legacy Tkinter Viewers

#### FF-HEDM Viewer (`ff_asym.py`)

```bash
cd <data_directory> && python ~/opt/MIDAS/gui/ff_asym.py &
```

Tkinter + Matplotlib FF viewer. See [FF_Visualization manual](../manuals/FF_Visualization.md).

#### NF-HEDM Viewer (`nf.py`)

```bash
cd <data_directory> && python ~/opt/MIDAS/gui/nf.py &
```

Tkinter + Matplotlib NF viewer with calibration workflow. See [NF_GUI manual](../manuals/NF_GUI.md).

---

### Other Tools

#### Diffraction Tomography GUI (`dt.py`)

```bash
python ~/opt/MIDAS/gui/dt.py &
```

2D detector image viewer with ring overlays and 1D intensity profiles.

#### Image Manipulation (`imageManipulation.py`)

```bash
python ~/opt/MIDAS/gui/imageManipulation.py &
```

General-purpose image viewer: dark subtraction, flat-field, ROI, transforms, histograms.

---

## Shared Components

#### `gui_common.py`

Common library for PyQtGraph viewers:

| Component | Description |
|---|---|
| `MIDASImageView` | Image viewer with crosshair, navigation toolbar, axis origin control |
| `apply_theme()` | Dark/light palette for Qt + PyQtGraph |
| `AsyncWorker` | Background thread wrapper |
| `LogPanel` | Redirects `print()` to collapsible dock |
| `get_colormap()` | Colormap lookup with matplotlib fallback |

---

## Directory Structure

```
gui/
├── ff_asym_qt.py         # FF-HEDM PyQtGraph viewer (recommended)
├── nf_qt.py              # NF-HEDM PyQtGraph viewer (recommended)
├── gui_common.py         # Shared PyQtGraph components
├── ff_asym.py            # FF-HEDM Tkinter viewer (legacy)
├── nf.py                 # NF-HEDM Tkinter viewer (legacy)
├── dt.py                 # Diffraction tomography GUI
├── imageManipulation.py  # Image viewer and processing tools
└── GEBad/                # GE detector bad-pixel masks
```

---

## Requirements

### PyQtGraph Viewers
```
PyQt5
pyqtgraph
numpy
```

### Tkinter Viewers
```
tkinter (built-in)
matplotlib
numpy
Pillow
scipy
h5py
```

---

## See Also

- [GUIs_and_Visualization manual](../manuals/GUIs_and_Visualization.md) — Master GUI guide
- [NF_GUI manual](../manuals/NF_GUI.md) — NF-HEDM GUI user guide
- [FF_Visualization manual](../manuals/FF_Visualization.md) — FF-HEDM visualization
- [FF_Interactive_Plotting manual](../manuals/FF_Interactive_Plotting.md) — Browser-based FF exploration
