# gui/ — Interactive Visualization GUIs

This directory contains Tkinter-based interactive desktop applications for visualizing and analyzing MIDAS data. All GUIs are run with `python` and open a native window with matplotlib-embedded plots.

---

## Applications

### NF-HEDM GUI (`nf.py`)

**Interactive NF-HEDM calibration and visualization.**

```bash
python nf.py
```

Provides a desktop interface for:
- Viewing raw NF-HEDM detector images across all omega angles
- Adjusting detector geometry parameters (beam center, distance) interactively
- Visualizing reconstructed `.mic` orientation maps with confidence overlays
- Comparing reconstructions at different parameter settings
- Inspecting individual voxel orientations and confidence values

See [NF_gui manual](../manuals/NF_gui.md) for the full user guide.

---

### FF-HEDM GUI (`ff_asym.py`)

**Interactive FF-HEDM exploration.**

```bash
python ff_asym.py
```

Provides a desktop interface for:
- Browsing grain lists with filtering by orientation, position, confidence
- Viewing and comparing raw detector frames
- Inspecting individual spot assignments per grain
- Exploring the orientation space (pole figures, IPF)
- Loading HDF5 and Zarr-ZIP data directly

---

### Diffraction Tomography GUI (`dt.py`)

**Interactive diffraction tomography and caking visualization.**

```bash
python dt.py
```

Provides a desktop interface for:
- Viewing raw 2D detector images
- Overlaying calibrated ring positions
- Interactively adjusting integration parameters
- Viewing 1D intensity profiles vs. 2θ

---

### Image Manipulation (`imageManipulation.py`)

**Detector image viewing and processing toolkit.**

```bash
python imageManipulation.py
```

Provides utilities for:
- Loading and displaying detector images (GE, TIFF, HDF5)
- Dark subtraction and flat-field normalization
- Region-of-interest selection
- Image transformations (flip, transpose, rotate)
- Pixel statistics and histogram analysis

---

## Directory Structure

```
gui/
├── nf.py                 # NF-HEDM calibration and microstructure GUI
├── ff_asym.py            # FF-HEDM grain exploration GUI
├── dt.py                 # Diffraction tomography / caking GUI
├── imageManipulation.py  # Image viewer and processing tools
├── ff/                   # FF-specific GUI components
├── ff_dash_app/          # Dash-based FF visualization (alternative)
└── GEBad/                # GE detector bad-pixel masks
```

---

## Requirements

All GUIs use Python's built-in **Tkinter** for the window framework and **matplotlib** for plotting. These are included in standard Python/conda installations:

```
tkinter (built-in)
matplotlib
numpy
PIL / Pillow
scipy
h5py
```

---

## See Also

- [NF_gui manual](../manuals/NF_gui.md) — NF-HEDM GUI user guide
- [FF_Interactive_Plotting manual](../manuals/FF_Interactive_Plotting.md) — FF-HEDM interactive visualization
