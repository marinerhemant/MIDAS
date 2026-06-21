# gui/ — MIDAS Visualization & Analysis GUI

A unified PyQt5 desktop launcher (`midas_gui.py`) that hosts every existing
MIDAS viewer as a separate nav entry. Qt-native viewers are embedded in the
launcher window; Tkinter / Dash / Plotly viewers run as managed subprocesses.
No functionality is removed and every standalone script still runs directly.

**22 of 23 standalone viewers reachable from one launcher.**

---

## Quick start

```bash
source ~/miniconda3/bin/activate midas_env
python ~/opt/MIDAS/gui/midas_gui.py [data_directory]

# Theme override
python ~/opt/MIDAS/gui/midas_gui.py --theme dark
```

**File → Open Directory…** (Ctrl+O) forwards the path to every module that
supports it — FF/NF set their `folder`, Calibration finds the first
`*.corr.csv`, PF-HEDM Diagnostics pre-fills `SpotDiagnostics.bin`, 3D Grain
Explorer pre-fills every tab's Result folder, Digital Twin picks up
`Grains.csv`, etc.

---

## Module map

| Nav entry | Tabs | Backed by (legacy scripts) | How |
|---|---|---|---|
| **FF Viewer** | — | [gui/ff_asym_qt.py](gui/ff_asym_qt.py) `FFViewer` | embedded |
| **NF Viewer** | — | [gui/nf_qt.py](gui/nf_qt.py) `NFViewer` | embedded |
| **Live Integrator Monitor** | — | [gui/viewers/live_viewer.py](gui/viewers/live_viewer.py) `LiveViewer` | embedded, deferred construction |
| **Peak / Lineout Inspector** | Caked Peaks | [gui/viewers/plot_caked_peaks.py](gui/viewers/plot_caked_peaks.py) `CakedPeakViewer` | embedded, lazy |
| | Lineouts | [gui/viewers/plot_lineout_results.py](gui/viewers/plot_lineout_results.py) `LineoutViewer` | embedded, lazy |
| | Phase ID | [gui/viewers/plot_phase_id_results.py](gui/viewers/plot_phase_id_results.py) `PhaseIdViewer` | embedded, lazy |
| | Integrator Peaks | [gui/viewers/plot_integrator_peaks.py](gui/viewers/plot_integrator_peaks.py) | subprocess |
| | σ Statistics | [gui/viewers/peak_sigma_statistics.py](gui/viewers/peak_sigma_statistics.py) | subprocess |
| | Lineout Compare | [gui/viewers/plot_lineout_comparison.py](gui/viewers/plot_lineout_comparison.py) | subprocess |
| | Caking (Dash) | [gui/viewers/viz_caking.py](gui/viewers/viz_caking.py) | subprocess + browser |
| **Calibration** | — | [gui/viewers/plot_calibrant_results.py](gui/viewers/plot_calibrant_results.py) `CalibrantViewer` | embedded, lazy |
| **Image Tools** | — | [gui/imageManipulation.py](gui/imageManipulation.py) | subprocess (Tkinter) |
| **PF-HEDM Diagnostics** | Sinogram (Dash) | [gui/viewers/pfIntensityViewer.py](gui/viewers/pfIntensityViewer.py) | subprocess + browser |
| | Spot Diagnostics | [utils/spot_diagnostics.py](utils/spot_diagnostics.py) `SpotDiagPlotter` | embedded matplotlib (QtAgg) |
| **3D Grain Explorer** | Interactive FF (Dash) | [gui/viewers/interactiveFFplotting.py](gui/viewers/interactiveFFplotting.py) | subprocess + browser |
| | 3D Spots | [gui/viewers/plotFFSpots3d.py](gui/viewers/plotFFSpots3d.py) | subprocess |
| | 3D Spots / Grain | [gui/viewers/plotFFSpots3dGrains.py](gui/viewers/plotFFSpots3dGrains.py) | subprocess |
| | 3D Grains | [gui/viewers/plotGrains3d.py](gui/viewers/plotGrains3d.py) | subprocess |
| | FF↔NF | [gui/viewers/PlotFFNF.py](gui/viewers/PlotFFNF.py) | subprocess |
| **Digital Twin** | Microstructure simulator | [gui/dig_tw.py](gui/dig_tw.py) | subprocess + browser |
| | Reconstruction compare | [gui/dt.py](gui/dt.py) | subprocess (Tkinter) |

**Not yet absorbed:** [utils/AutoCalibrateZarr.py](utils/AutoCalibrateZarr.py)'s
embedded `CalibImageViewer` (tightly coupled to the calibration loop).

---

## Embed strategy

Three categories, chosen per viewer:

1. **Direct embed** — PyQt5 `QMainWindow` classes (FF, NF, Live, Spot
   Diagnostics' Qt port) are inserted into the launcher's `QStackedWidget`
   after `setWindowFlags(Qt.Widget)`. All menus, toolbars, status bars, and
   worker threads still work.
2. **Lazy embed** — PyQt6 viewers (Caked Peaks, Lineouts, Phase ID,
   Calibration) are imported only when the user opens that tab, so PyQt6
   does not collide with PyQt5 at launcher boot. The tab shows a folder
   picker until the user clicks Load.
3. **External process** — Tkinter (Image Tools, DT recon) and Dash
   (Interactive FF, PF Sinogram, Digital Twin, Caking viz) viewers run as
   managed subprocesses via the `ExternalLauncher` widget. The widget tracks
   PIDs, lets you launch multiple instances, and offers a single "Stop all"
   button plus optional "Open in browser" for Dash targets.

---

## Layout

```
gui/
├── midas_gui.py                # entry-point shim
├── midas_app/                  # launcher package (PyQt5)
│   ├── core/
│   │   ├── theme.py            # light/dark palette + colormaps
│   │   ├── async_worker.py     # QThread wrapper
│   │   ├── log_panel.py        # collapsible stdout dock
│   │   ├── io.py               # unified TIFF/HDF5/zarr/bz2/GE loader
│   │   ├── params.py           # ps.txt parser
│   │   └── results.py          # typed CSV/H5 loaders
│   ├── widgets/
│   │   ├── image_view.py       # PyQt5 MIDASImageView
│   │   ├── peak_table.py
│   │   ├── file_browser.py
│   │   ├── ring_overlay.py
│   │   └── external_launcher.py  # subprocess + PID tracker
│   ├── modules/
│   │   ├── ff_viewer.py        # FF Viewer
│   │   ├── nf_viewer.py        # NF Viewer
│   │   ├── live_monitor.py     # Live Integrator Monitor
│   │   ├── peak_inspector.py   # Peak / Lineout Inspector (7 sub-tabs)
│   │   ├── calibration.py      # Calibration
│   │   ├── image_tools.py      # Image Tools
│   │   ├── pf_diagnostics.py   # PF-HEDM Diagnostics (2 sub-tabs)
│   │   ├── grain_explorer.py   # 3D Grain Explorer (5 sub-tabs)
│   │   └── digital_twin.py     # Digital Twin (2 sub-tabs)
│   └── main.py                 # QMainWindow + nav rail + module registry
│
├── ff_asym_qt.py  nf_qt.py  gui_common.py        # standalone PyQt5 viewers
├── imageManipulation.py  dt.py                    # Tkinter
├── dig_tw.py                                      # Dash digital twin
├── viewers/                                       # standalone analysis viewers
└── archive/ GEBad/
```

---

## Standalone invocation still works

```bash
python gui/ff_asym_qt.py
python gui/nf_qt.py --dark
python gui/viewers/live_viewer.py --lineout lineout.bin --nRBins 500
python gui/viewers/plot_caked_peaks.py /path/to/work_dir
python gui/imageManipulation.py
python gui/dig_tw.py -mic Grains.csv
```

The launcher wrappers don't modify any of these scripts — they just import
or subprocess-invoke them.

---

## Requirements

```
PyQt5 + pyqtgraph              # launcher + FF/NF/Live/Spot-diag viewers
matplotlib                     # spot diagnostics embedded canvas
numpy, pandas, h5py, tifffile, zarr
# Optional — loaded only when those tabs open:
PyQt6                          # caked/lineout/phase_id/calibrant viewers
dash, plotly, dash-bootstrap    # Dash-based viewers (subprocess)
```

---

## Recent Updates (2026-06-09)

### `ff_asym_qt.py` — FF Viewer relabeling, median aggregation, inferno default

- **Default colormap** changed from `bone` to `inferno`.
- **Image Display panel relabeling:**
  - `Frame` → `Display Frame` with a `/ N` indicator showing the total number of frames in the currently-loaded file (updates automatically when a file is loaded or state is restored).
  - `Num Frames` → implicit in the consolidated aggregation row (see below).
  - `Max/Frames`, `Sum/Frames` → `Max`, `Sum`.
- **Aggregation row consolidated** — `# Frames:` spin box and the `Max`, `Sum`, `Median` checkboxes now appear on a single row in the Image Display panel so their visual relationship is unambiguous.
- **Median aggregation** — new `Median` checkbox buffers all frames into memory and applies `np.median(slab, axis=0)` at the end.  Selecting Median automatically unchecks Max and Sum.  State is saved and restored via the `'median_per_frames'` key.  Three execution paths support median:
  1. zarr slab (`np.median` on a numpy array slab)
  2. single-HDF5 slab (`np.median` on an HDF5-read slab)
  3. general `ThreadPoolExecutor` path (frames buffered into a list; `np.median(np.stack(frames), axis=0)` after all workers complete)

### `gui_common.py` — colorbar gradient sync

- `MIDASImageView.set_colormap` now also updates the pyqtgraph histogram LUT gradient strip (the coloured bar on the right side of the histogram) so the colorbar always matches the selected colormap.  The sync is done via `histogram.gradient.setColorMap(cmap)` with a safe `hasattr` guard for pyqtgraph version compatibility.

---

## See also

- [`manuals/GUIs_and_Visualization.md`](../manuals/GUIs_and_Visualization.md) — master GUI guide
- [`manuals/NF_GUI.md`](../manuals/NF_GUI.md) — NF-HEDM viewer reference
- [`manuals/FF_Visualization.md`](../manuals/FF_Visualization.md) — FF-HEDM visualization
- [Plan file](.claude/plans/do-a-thorough-audit-fluffy-jellyfish.md) — full audit + consolidation plan
