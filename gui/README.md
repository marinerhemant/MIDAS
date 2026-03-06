# gui/ — Interactive Visualization & Viewers

Desktop applications and standalone viewers for MIDAS data visualization and analysis.

---

## Directory Structure

```
gui/
├── viewers/                 # Standalone analysis viewers (moved from utils/)
│   ├── plot_lineout_results.py
│   ├── plot_integrator_peaks.py
│   ├── plot_calibrant_results.py
│   ├── plot_phase_id_results.py
│   ├── plot_lineout_comparison.py
│   ├── live_viewer.py
│   ├── interactiveFFplotting.py
│   ├── pfIntensityViewer.py
│   ├── peak_sigma_statistics.py
│   ├── plotFFSpots3d.py
│   ├── plotFFSpots3dGrains.py
│   ├── plotGrains3d.py
│   ├── PlotFFNF.py
│   └── viz_caking.py
├── ff_asym_qt.py            # FF-HEDM PyQtGraph viewer (recommended)
├── nf_qt.py                 # NF-HEDM PyQtGraph viewer (recommended)
├── gui_common.py            # Shared PyQtGraph components
├── ff_asym.py               # FF-HEDM Tkinter viewer (legacy)
├── nf.py                    # NF-HEDM Tkinter viewer (legacy)
├── dt.py                    # Diffraction tomography GUI
├── imageManipulation.py     # Image viewer and processing tools
└── GEBad/                   # GE detector bad-pixel masks
```

---

## Detector Image Viewers

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

NF-HEDM viewer with all FF features plus microstructure overlay (`.mic`/`.map`), spot simulation, BoxH/BoxV ROI tools, and beam center calibration. See [NF_GUI manual](../manuals/NF_GUI.md).

### Legacy Tkinter Viewers

| Script | Description |
|--------|-------------|
| `ff_asym.py` | Tkinter + Matplotlib FF viewer. See [FF_Visualization](../manuals/FF_Visualization.md). |
| `nf.py` | Tkinter + Matplotlib NF viewer with calibration workflow. |

### Other Tools

| Script | Description |
|--------|-------------|
| `dt.py` | 2D detector image viewer with ring overlays and 1D intensity profiles. |
| `imageManipulation.py` | General-purpose image viewer: dark subtraction, flat-field, ROI, transforms, histograms. |

---

## Analysis Viewers (`viewers/`)

Standalone plotting and diagnostic viewers, moved from `utils/` as of 2026-03.

### Lineout & Peak Viewers

| Script | Description |
|--------|-------------|
| `plot_lineout_results.py` | **Lineout viewer.** PyQt6 viewer for `extract_lineouts.py` output. Shows corrected lineout, SNIP background, fitted profile, difference, and peak table with interactive row selection → peak highlighting. |
| `plot_lineout_comparison.py` | Overlay calibrant and integrator lineouts with ideal ring markers. |
| `plot_integrator_peaks.py` | Post-hoc peak analysis from `.caked.hdf.zarr.zip`. Ring-assigned scatter plots. |
| `peak_sigma_statistics.py` | Peak width (σ) statistics from FF-HEDM fitting results. |

### Calibration & Phase ID Viewers

| Script | Description |
|--------|-------------|
| `plot_calibrant_results.py` | **Calibrant QC.** PyQt6 viewer for `CalibrantPanelShiftsOMP` `_corr.csv` output. |
| `plot_phase_id_results.py` | Phase identification results viewer. |

### Real-Time & Interactive Viewers

| Script | Description |
|--------|-------------|
| `live_viewer.py` | **Real-time dashboard.** PyQtGraph live viewer for `lineout.bin` / `fit.bin` streams. See [FF_Radial_Integration](../manuals/FF_Radial_Integration.md) §6.3. |
| `interactiveFFplotting.py` | Dash-based interactive FF-HEDM browser. See [FF_Interactive_Plotting](../manuals/FF_Interactive_Plotting.md). |
| `pfIntensityViewer.py` | Point-focus / scanning HEDM intensity viewer. |

### 3D Visualization

| Script | Description |
|--------|-------------|
| `plotFFSpots3d.py` | 3D scatter plot of FF-HEDM diffraction spots. |
| `plotFFSpots3dGrains.py` | 3D scatter of FF spots, color-coded by grain. |
| `plotGrains3d.py` | 3D scatter of grain centroids with orientation coloring. |
| `PlotFFNF.py` | Overlay FF-HEDM grain centroids on NF-HEDM orientation maps. |
| `viz_caking.py` | Visualize radial integration (caking) results. |

---

## Shared Components

#### `gui_common.py`

| Component | Description |
|-----------|-------------|
| `MIDASImageView` | Image viewer with crosshair, navigation toolbar, axis origin control |
| `apply_theme()` | Dark/light palette for Qt + PyQtGraph |
| `AsyncWorker` | Background thread wrapper |
| `LogPanel` | Redirects `print()` to collapsible dock |
| `get_colormap()` | Colormap lookup with matplotlib fallback |

---

## Requirements

### PyQtGraph Viewers
```
PyQt5 / PyQt6
pyqtgraph
numpy
```

### Matplotlib-based Viewers (`viewers/`)
```
PyQt6 (for Qt viewers)
matplotlib
numpy
scipy
```

### Tkinter Viewers (legacy)
```
tkinter (built-in)
matplotlib
numpy
Pillow, scipy, h5py
```

---

## See Also

- [GUIs_and_Visualization](../manuals/GUIs_and_Visualization.md) — Master GUI guide
- [NF_GUI](../manuals/NF_GUI.md) — NF-HEDM GUI user guide
- [FF_Visualization](../manuals/FF_Visualization.md) — FF-HEDM visualization
- [FF_Interactive_Plotting](../manuals/FF_Interactive_Plotting.md) — Browser-based FF exploration
