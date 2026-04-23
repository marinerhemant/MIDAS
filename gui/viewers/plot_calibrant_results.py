#!/usr/bin/env python3
"""
Interactive CalibrantPanelShiftsOMP results viewer (Qt edition).

Auto-detects *corr.csv files in the current directory.
Provides dropdown selectors for X, Y, and Color columns,
user-selectable color range, log scale, and colormap picker.

Usage:  python plot_calibrant_results.py [file.corr.csv]
"""

import glob
import os
import sys

import matplotlib
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

# ── CSV column map (from CalibrantPanelShiftsOMP header) ────────────────
# Legacy 16-column space-separated format
COLUMNS_LEGACY = [
    'Eta', 'Strain', 'RadFit', 'EtaCalc', 'DiffCalc', 'RadCalc',
    'Ideal2Theta', 'Outlier', 'YRawCorr', 'ZRawCorr', 'RingNr',
    'RadGlobal', 'IdealR', 'Fit2Theta', 'IdealA', 'FitA',
    'DeltaR', 'DeltaA',
]

# CalibrantIntegratorOMP per-bin CSV columns
COLUMNS_INTEGRATOR = ['RingNr', 'Eta', 'Diff', 'Rad', 'Y', 'Z', 'IsOutlier']

# Unified column list used by the viewer (superset)
COLUMNS = COLUMNS_LEGACY
COL = {name: idx for idx, name in enumerate(COLUMNS)}
PLOTTABLE = [c for c in COLUMNS if c != 'Outlier']


def _load_corr_csv(filename):
    """Auto-detect CalibrantIntegratorOMP vs CalibrantPanelShiftsOMP format.

    CalibrantIntegratorOMP: geometry header (comma-delimited, 2 lines),
      blank line, then '%Eta Strain ...' header + data.
    CalibrantPanelShiftsOMP: '%Eta Strain ...' header + data directly.
    compare_*.corr.csv: same 18-col base + extra strain-comparison columns.

    When the '%Eta ...' header carries more tokens than the 18-column legacy
    layout (e.g. the compare CSV), the extra tokens are used as column names
    so the viewer can expose them in the axis/color dropdowns.

    Returns (data_array, column_names).
    """
    with open(filename) as f:
        lines = f.readlines()

    # Find the per-bin data header line (starts with '%Eta' or '%' followed by column names)
    data_start = None
    header_tokens = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('%Eta') or stripped.startswith('% Eta'):
            data_start = i
            # Parse header tokens: strip leading '%' (with optional space) then split
            hdr = stripped.lstrip('%').strip()
            header_tokens = hdr.split()
            break

    if data_start is not None:
        data = np.genfromtxt(filename, skip_header=data_start + 1)
        ncols = data.shape[1] if data.ndim == 2 else 1
        # Prefer header-declared names when they cover every data column;
        # otherwise fall back to legacy to stay backward compatible with files
        # whose header abbreviates column names unexpectedly.
        if header_tokens and len(header_tokens) >= ncols:
            cols = header_tokens[:ncols]
        else:
            cols = COLUMNS_LEGACY[:ncols]
        return data, cols

    # Fallback: try legacy format (first line is the header)
    data = np.genfromtxt(filename, skip_header=1)
    ncols = data.shape[1] if data.ndim == 2 else 1
    cols = COLUMNS_LEGACY[:ncols]
    return data, cols


COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
    'coolwarm', 'RdYlBu', 'Spectral', 'bwr',
    'jet', 'turbo', 'rainbow',
    'Greys', 'Blues', 'Reds', 'YlOrRd',
]


# ── File discovery ──────────────────────────────────────────────────────
def find_corr_files():
    return sorted(glob.glob('*corr.csv'))


def _find_caked_file_for_corr(corr_path):
    """Locate a '*.caked.hdf' sibling of the given corr.csv.

    Tries (in order) exact stem matches for the three common corr.csv flavors
    ('<rawFN>.corr.csv', 'integrator_<stem>.corr.csv', 'compare_<stem>.corr.csv'),
    then falls back to the first '*.caked.hdf' in the same directory.
    """
    d = os.path.dirname(os.path.abspath(corr_path)) or '.'
    base = os.path.basename(corr_path)

    # Strip a leading 'integrator_' or 'compare_' prefix
    for prefix in ('integrator_', 'compare_'):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    # Strip trailing '.corr.csv'
    if base.endswith('.corr.csv'):
        base = base[:-len('.corr.csv')]

    # Try common patterns first
    for cand in (f'{base}.caked.hdf', f'{base}.h5.caked.hdf',
                 f'{base}.hdf.caked.hdf'):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            return p

    # Fallback: first .caked.hdf in the same directory
    matches = sorted(glob.glob(os.path.join(d, '*.caked.hdf')))
    return matches[0] if matches else None


def _load_caked_image(caked_path):
    """Load the OmegaSumFrame image and axis extents from a caked HDF file.

    Returns (image_2d, (eta_min, eta_max, r_min, r_max), dataset_path) or
    (None, None, None) on failure.
    """
    try:
        import h5py
    except ImportError:
        print('h5py not available — cannot load caked HDF', file=sys.stderr)
        return None, None, None

    try:
        with h5py.File(caked_path, 'r') as f:
            grp = f.get('OmegaSumFrame')
            if grp is None:
                # Fall back to IntegrationResult/FrameNr_0 if OmegaSumFrame absent
                grp = f.get('IntegrationResult')
                if grp is None:
                    return None, None, None
            ds_name = None
            for n in grp:
                if 'LastFrameNumber' in n or 'FrameNr' in n:
                    ds_name = n
                    break
            if ds_name is None:
                # Try any 2D dataset
                for n in grp:
                    if grp[n].ndim == 2:
                        ds_name = n
                        break
            if ds_name is None:
                return None, None, None
            img = grp[ds_name][()]  # (nR, nEta)

            reta = f.get('REtaMap')
            if reta is not None and reta.ndim == 3 and reta.shape[0] >= 3:
                r_plane = reta[0, :, :]
                eta_plane = reta[2, :, :]
                r_centers = np.nanmean(r_plane, axis=1)
                eta_centers = np.nanmean(eta_plane, axis=0)
            else:
                nR, nEta = img.shape
                r_centers = np.arange(nR, dtype=float)
                eta_centers = np.linspace(-180.0, 180.0, nEta,
                                          endpoint=False) + (360.0 / nEta) / 2.0
    except Exception as e:
        print(f'Failed to load caked HDF {caked_path}: {e}', file=sys.stderr)
        return None, None, None

    # Extent: imshow with origin='lower' → (xmin, xmax, ymin, ymax)
    eta_min = float(np.nanmin(eta_centers))
    eta_max = float(np.nanmax(eta_centers))
    r_min = float(np.nanmin(r_centers))
    r_max = float(np.nanmax(r_centers))
    return img, (eta_min, eta_max, r_min, r_max), f'{os.path.basename(caked_path)}:{ds_name}'


def choose_file(files):
    if not files:
        print('No *corr.csv files found in the current directory.', file=sys.stderr)
        sys.exit(1)
    if len(files) == 1:
        return files[0]
    print('Multiple corr.csv files found:')
    for i, f in enumerate(files):
        print(f'  [{i}] {f}')
    while True:
        try:
            choice = int(input('Select file number: '))
            if 0 <= choice < len(files):
                return files[choice]
        except (ValueError, EOFError):
            pass
        print(f'Please enter 0–{len(files)-1}')


# ── Qt Main Window ──────────────────────────────────────────────────────
class CalibrantViewer(QMainWindow):
    def __init__(self, filename=None):
        super().__init__()
        self.filename = filename or ''
        if filename and os.path.isfile(filename):
            self.raw, self._col_names = _load_corr_csv(filename)
            self._rebuild_col_map()
        else:
            self.raw = np.empty((0, len(COLUMNS_LEGACY)))
            self._col_names = COLUMNS_LEGACY
            self._rebuild_col_map()
        self.setWindowTitle(f'Calibrant Viewer — {self.filename}' if self.filename
                            else 'Calibrant Viewer')
        self.resize(1200, 750)

        # State
        self.x_col = 'RingNr'
        self.y_col = 'Strain'
        self.c_col = 'RingNr'
        self.exclude_outliers = True
        self.log_color = False
        self.cmap = 'tab10'
        self.c_min = None  # None = auto
        self.marker_size = 30
        self.c_max = None

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── Control bar (two rows) ──────────────────────────────────
        # Row 1: file + axis selectors (X, Y, Color, Cmap)
        # Row 2: display options (min/max, log, img-range, size, mode toggles)
        controls = QHBoxLayout()
        controls2 = QHBoxLayout()

        # File selector
        controls.addWidget(QLabel('File:'))
        self.combo_file = QComboBox()
        self.combo_file.setMinimumWidth(200)
        self._populate_file_list()
        self.combo_file.currentTextChanged.connect(self._on_file_changed)
        controls.addWidget(self.combo_file)

        btn_open = QPushButton('Open…')
        btn_open.clicked.connect(self._on_open_file)
        controls.addWidget(btn_open)

        controls.addWidget(self._vsep())

        # X axis
        controls.addWidget(QLabel('X:'))
        self.combo_x = QComboBox()
        self.combo_x.addItems(PLOTTABLE)
        self.combo_x.setCurrentText(self.x_col)
        self.combo_x.currentTextChanged.connect(self._on_x)
        controls.addWidget(self.combo_x)

        # Y axis
        controls.addWidget(QLabel('Y:'))
        self.combo_y = QComboBox()
        self.combo_y.addItems(PLOTTABLE)
        self.combo_y.setCurrentText(self.y_col)
        self.combo_y.currentTextChanged.connect(self._on_y)
        controls.addWidget(self.combo_y)

        # Separator
        controls.addWidget(self._vsep())

        # Color column
        controls.addWidget(QLabel('Color:'))
        self.combo_c = QComboBox()
        self.combo_c.addItems(PLOTTABLE)
        self.combo_c.setCurrentText(self.c_col)
        self.combo_c.currentTextChanged.connect(self._on_color)
        controls.addWidget(self.combo_c)

        # Colormap
        controls.addWidget(QLabel('Cmap:'))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(COLORMAPS)
        self.combo_cmap.setCurrentText(self.cmap)
        self.combo_cmap.currentTextChanged.connect(self._on_cmap)
        controls.addWidget(self.combo_cmap)

        controls.addStretch()

        # ── Row 2: display options ──────────────────────────────────
        # Color range
        controls2.addWidget(QLabel('Min:'))
        self.edit_cmin = QLineEdit()
        self.edit_cmin.setPlaceholderText('auto')
        self.edit_cmin.setFixedWidth(70)
        self.edit_cmin.editingFinished.connect(self._on_crange)
        controls2.addWidget(self.edit_cmin)

        controls2.addWidget(QLabel('Max:'))
        self.edit_cmax = QLineEdit()
        self.edit_cmax.setPlaceholderText('auto')
        self.edit_cmax.setFixedWidth(70)
        self.edit_cmax.editingFinished.connect(self._on_crange)
        controls2.addWidget(self.edit_cmax)

        # Log scale — applies to scatter color in normal mode, imshow intensity in caked mode
        self.chk_log = QCheckBox('Log')
        self.chk_log.setToolTip('Log-scale the scatter color range, or the caked imshow intensity')
        self.chk_log.toggled.connect(self._on_log)
        controls2.addWidget(self.chk_log)

        controls2.addWidget(self._vsep())

        # Caked-mode imshow intensity range (hidden unless Caked is active)
        self._lbl_imin = QLabel('Img Min:')
        controls2.addWidget(self._lbl_imin)
        self.edit_imin = QLineEdit()
        self.edit_imin.setPlaceholderText('auto')
        self.edit_imin.setFixedWidth(70)
        self.edit_imin.setToolTip('Caked imshow vmin (blank = 1% percentile)')
        self.edit_imin.editingFinished.connect(self._on_irange)
        controls2.addWidget(self.edit_imin)

        self._lbl_imax = QLabel('Img Max:')
        controls2.addWidget(self._lbl_imax)
        self.edit_imax = QLineEdit()
        self.edit_imax.setPlaceholderText('auto')
        self.edit_imax.setFixedWidth(70)
        self.edit_imax.setToolTip('Caked imshow vmax (blank = 99.5% percentile)')
        self.edit_imax.editingFinished.connect(self._on_irange)
        controls2.addWidget(self.edit_imax)

        self.chk_transpose = QCheckBox('Transpose')
        self.chk_transpose.setToolTip('Swap caked axes: R on X, η on Y')
        self.chk_transpose.toggled.connect(self._on_transpose)
        controls2.addWidget(self.chk_transpose)

        self._caked_widgets = (self._lbl_imin, self.edit_imin,
                               self._lbl_imax, self.edit_imax,
                               self.chk_transpose)
        for w in self._caked_widgets:
            w.setVisible(False)

        controls2.addWidget(self._vsep())

        # Outlier toggle
        self.chk_outlier = QCheckBox('Show outliers')
        self.chk_outlier.setChecked(not self.exclude_outliers)
        self.chk_outlier.toggled.connect(self._on_outlier)
        controls2.addWidget(self.chk_outlier)

        # Marker size (applies to scatter overlay in all modes)
        controls2.addWidget(QLabel('Size:'))
        self.spin_size = QSpinBox()
        self.spin_size.setRange(1, 300)
        self.spin_size.setSingleStep(2)
        self.spin_size.setValue(self.marker_size)
        self.spin_size.setToolTip('Scatter marker size (applies to all plot modes)')
        self.spin_size.valueChanged.connect(self._on_size)
        controls2.addWidget(self.spin_size)

        # Polar plot toggle
        self.chk_polar = QCheckBox('Polar')
        self.chk_polar.toggled.connect(self._on_polar)
        controls2.addWidget(self.chk_polar)

        # Caked-image overlay mode toggle
        self.chk_caked = QCheckBox('Caked')
        self.chk_caked.setToolTip(
            'imshow OmegaSumFrame from <rawFN>.caked.hdf and overlay '
            'fitted (Eta, RadFit) from the current corr.csv')
        self.chk_caked.toggled.connect(self._on_caked)
        controls2.addWidget(self.chk_caked)

        # Auto-range button
        self.btn_autorng = QPushButton('Auto range')
        self.btn_autorng.clicked.connect(self._on_autorange)
        controls2.addWidget(self.btn_autorng)

        controls2.addStretch()
        main_layout.addLayout(controls)
        main_layout.addLayout(controls2)

        # ── Matplotlib canvas ───────────────────────────────────────
        self.fig = Figure(figsize=(10, 6), dpi=100)
        # Fixed axes positions so colorbar doesn't steal space
        self.ax = self.fig.add_axes([0.08, 0.12, 0.78, 0.82])
        self.cax = self.fig.add_axes([0.88, 0.12, 0.02, 0.82])
        self.polar_mode = False
        self.caked_mode = False
        self.caked_transpose = False
        self.img_vmin = None  # caked imshow vmin (None = auto percentile)
        self.img_vmax = None
        self._caked_cache = {}  # corr_path -> (image, (xmin,xmax,ymin,ymax), caked_path)
        self.canvas = FigureCanvasQTAgg(self.fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.canvas)

        self._update()

    @staticmethod
    def _vsep():
        sep = QWidget()
        sep.setFixedWidth(2)
        sep.setStyleSheet('background-color: #999;')
        return sep

    # ── Data property ───────────────────────────────────────────────
    @property
    def data(self):
        d = self.raw
        if self.exclude_outliers:
            d = d[d[:, COL['Outlier']] == 0]
        return d

    # ── Callbacks ───────────────────────────────────────────────────
    def _on_x(self, text):
        self.x_col = text
        self._update()

    def _on_y(self, text):
        self.y_col = text
        self._update()

    def _on_color(self, text):
        self.c_col = text
        # Auto-select sensible colormap
        if text == 'RingNr':
            self.combo_cmap.setCurrentText('tab10')
        elif text.startswith('Strain'):
            self.combo_cmap.setCurrentText('coolwarm')
        else:
            if self.cmap in ('tab10', 'tab20', 'Set1', 'Set2', 'Set3'):
                self.combo_cmap.setCurrentText('viridis')
        self._clear_crange()
        self._update()

    def _on_cmap(self, text):
        self.cmap = text
        self._update()

    def _on_crange(self):
        try:
            self.c_min = float(self.edit_cmin.text()) if self.edit_cmin.text().strip() else None
        except ValueError:
            self.c_min = None
        try:
            self.c_max = float(self.edit_cmax.text()) if self.edit_cmax.text().strip() else None
        except ValueError:
            self.c_max = None
        self._update()

    def _on_log(self, checked):
        self.log_color = checked
        self._update()

    def _on_outlier(self, checked):
        self.exclude_outliers = not checked
        self._update()

    def _on_polar(self, checked):
        self.polar_mode = checked
        if checked and self.caked_mode:
            self.chk_caked.setChecked(False)  # triggers _on_caked(False)
        self._rebuild_axes()
        self._update()

    def _on_caked(self, checked):
        self.caked_mode = checked
        if checked and self.polar_mode:
            self.chk_polar.setChecked(False)
        for w in self._caked_widgets:
            w.setVisible(checked)
        self._rebuild_axes()
        self._update()

    def _on_size(self, value):
        self.marker_size = int(value)
        self._update()

    def _on_transpose(self, checked):
        self.caked_transpose = checked
        self._update()

    def _on_irange(self):
        try:
            self.img_vmin = (float(self.edit_imin.text())
                             if self.edit_imin.text().strip() else None)
        except ValueError:
            self.img_vmin = None
        try:
            self.img_vmax = (float(self.edit_imax.text())
                             if self.edit_imax.text().strip() else None)
        except ValueError:
            self.img_vmax = None
        self._update()

    def _on_autorange(self):
        self._clear_crange()
        self._update()

    def _populate_file_list(self):
        """Refresh the file combo with *corr.csv files in cwd."""
        self.combo_file.blockSignals(True)
        current = self.filename
        self.combo_file.clear()
        files = find_corr_files()
        self.combo_file.addItems(files)
        if current in files:
            self.combo_file.setCurrentText(current)
        elif files:
            self.combo_file.setCurrentText(files[0])
        self.combo_file.blockSignals(False)

    def _on_file_changed(self, text):
        if text and text != self.filename and os.path.isfile(text):
            self._load_file(text)

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open corr.csv', '', 'CSV files (*corr.csv);;All files (*)')
        if path:
            self._load_file(path)

    def _rebuild_col_map(self):
        """Rebuild COL dict and refresh combo boxes from current column names."""
        global COL, PLOTTABLE
        COL = {name: idx for idx, name in enumerate(self._col_names)}
        PLOTTABLE = [c for c in self._col_names if c != 'Outlier']
        # Refresh combo boxes if they exist
        for combo, current_attr in [
            (getattr(self, 'combo_x', None), 'x_col'),
            (getattr(self, 'combo_y', None), 'y_col'),
            (getattr(self, 'combo_c', None), 'c_col'),
        ]:
            if combo is not None:
                combo.blockSignals(True)
                current = getattr(self, current_attr)
                combo.clear()
                combo.addItems(PLOTTABLE)
                if current in PLOTTABLE:
                    combo.setCurrentText(current)
                combo.blockSignals(False)

    def _load_file(self, path):
        try:
            data, cols = _load_corr_csv(path)
            self.raw = data
            self._col_names = cols
            self._rebuild_col_map()
            self.filename = os.path.basename(path)
            self.setWindowTitle(f'Calibrant Viewer — {self.filename}')
            # Add to combo if not present
            if self.combo_file.findText(self.filename) < 0:
                self.combo_file.addItem(self.filename)
            self.combo_file.blockSignals(True)
            self.combo_file.setCurrentText(self.filename)
            self.combo_file.blockSignals(False)
            self._update()
        except Exception as e:
            print(f'Error loading {path}: {e}')

    def _clear_crange(self):
        self.c_min = None
        self.c_max = None
        self.edit_cmin.clear()
        self.edit_cmax.clear()

    # ── Axes management ──────────────────────────────────────────────
    def _rebuild_axes(self):
        """Recreate axes when switching between Cartesian and polar."""
        self.fig.clear()
        if self.polar_mode:
            self.ax = self.fig.add_axes([0.05, 0.05, 0.80, 0.88], polar=True)
            self.cax = self.fig.add_axes([0.90, 0.12, 0.02, 0.76])
        else:
            self.ax = self.fig.add_axes([0.08, 0.12, 0.78, 0.82])
            self.cax = self.fig.add_axes([0.88, 0.12, 0.02, 0.82])

    # ── Redraw ──────────────────────────────────────────────────────
    def _update(self):
        d = self.data
        self.ax.clear()

        if len(d) == 0:
            self.ax.text(0.5, 0.5, 'No data after filtering',
                         transform=self.ax.transAxes, ha='center', va='center')
            self.canvas.draw_idle()
            return

        xi, yi, ci = COL[self.x_col], COL[self.y_col], COL[self.c_col]
        c_data = d[:, ci].copy()
        c_label = self.c_col
        # Scale any fractional-strain column to µε for colorbar readability.
        # 'StrainDiffMicroStrain' is pre-scaled — leave it alone.
        if (self.c_col.startswith('Strain') and
                self.c_col != 'StrainDiffMicroStrain'):
            c_data = c_data * 1e6
            c_label = f'{self.c_col} (µε)'

        # Color normalization
        vmin = self.c_min if self.c_min is not None else np.nanmin(c_data)
        vmax = self.c_max if self.c_max is not None else np.nanmax(c_data)
        if self.log_color:
            # Clamp to positive for log scale
            safe_min = max(vmin, 1e-10)
            safe_max = max(vmax, safe_min * 1.1)
            norm = LogNorm(vmin=safe_min, vmax=safe_max)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        n_total = len(self.raw)
        n_shown = len(d)

        if self.polar_mode:
            # Polar plot: angular = Eta (degrees → radians), radial = Strain (µε)
            eta_rad = np.radians(d[:, COL['Eta']])
            strain_ue = d[:, COL['Strain']] * 1e6  # always use Strain as radial
            sc = self.ax.scatter(eta_rad, strain_ue, c=c_data,
                                 cmap=self.cmap, norm=norm,
                                 s=self.marker_size, alpha=0.7, edgecolors='none')
            self.ax.set_theta_zero_location('N')  # 0° at top
            self.ax.set_theta_direction(-1)        # clockwise
            self.ax.set_title(f'{self.filename}  ({n_shown}/{n_total} pts)\n'
                              f'Radial: Strain (µε), Angular: η',
                              fontsize=10, pad=20)
        elif self.caked_mode:
            # Caked imshow + (Eta, RadFit) overlay
            cache_key = self.filename
            cached = self._caked_cache.get(cache_key)
            if cached is None:
                caked_path = _find_caked_file_for_corr(self.filename)
                if caked_path is None:
                    self.ax.text(0.5, 0.5,
                                 'No *.caked.hdf found next to corr.csv',
                                 transform=self.ax.transAxes, ha='center', va='center')
                    self.canvas.draw_idle()
                    return
                img, extent, ds_label = _load_caked_image(caked_path)
                if img is None:
                    self.ax.text(0.5, 0.5,
                                 f'Could not load caked data from\n{os.path.basename(caked_path)}',
                                 transform=self.ax.transAxes, ha='center', va='center')
                    self.canvas.draw_idle()
                    return
                cached = (img, extent, ds_label)
                self._caked_cache[cache_key] = cached

            img, extent, ds_label = cached
            # Intensity scaling — log if the Log checkbox is on, else percentile
            finite = img[np.isfinite(img)]
            if finite.size == 0:
                self.ax.text(0.5, 0.5, 'Caked image is all NaN',
                             transform=self.ax.transAxes, ha='center', va='center')
                self.canvas.draw_idle()
                return
            if self.log_color:
                pos = finite[finite > 0]
                auto_lo = float(np.percentile(pos, 1)) if pos.size else 1e-3
                auto_hi = float(np.percentile(pos, 99.5)) if pos.size else 1.0
                lo = self.img_vmin if self.img_vmin is not None else auto_lo
                hi = self.img_vmax if self.img_vmax is not None else auto_hi
                # LogNorm requires strictly positive bounds
                lo = max(lo, 1e-6)
                if hi <= lo:
                    hi = lo * 1.1
                img_norm = LogNorm(vmin=lo, vmax=hi)
            else:
                auto_lo = float(np.percentile(finite, 1))
                auto_hi = float(np.percentile(finite, 99.5))
                lo = self.img_vmin if self.img_vmin is not None else auto_lo
                hi = self.img_vmax if self.img_vmax is not None else auto_hi
                if hi <= lo:
                    hi = lo + 1.0
                img_norm = Normalize(vmin=lo, vmax=hi)

            if self.caked_transpose:
                # Swap axes: R on X, η on Y
                img_display = img.T
                disp_extent = (extent[2], extent[3], extent[0], extent[1])
                sc_x = d[:, COL['RadFit']]
                sc_y = d[:, COL['Eta']]
                x_label = 'R (pixels)'
                y_label = 'η (degrees)'
            else:
                img_display = img
                disp_extent = extent
                sc_x = d[:, COL['Eta']]
                sc_y = d[:, COL['RadFit']]
                x_label = 'η (degrees)'
                y_label = 'R (pixels)'

            self.ax.imshow(img_display, extent=disp_extent, origin='lower',
                           aspect='auto', cmap='gray', norm=img_norm,
                           interpolation='nearest')

            sc = self.ax.scatter(sc_x, sc_y,
                                 c=c_data, cmap=self.cmap, norm=norm,
                                 s=self.marker_size, alpha=0.85,
                                 edgecolors='white', linewidths=0.3)
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label)
            self.ax.set_xlim(disp_extent[0], disp_extent[1])
            self.ax.set_ylim(disp_extent[2], disp_extent[3])
            self.ax.set_title(
                f'{self.filename}  ({n_shown}/{n_total} fits)\n'
                f'{ds_label}', fontsize=9)
        else:
            sc = self.ax.scatter(d[:, xi], d[:, yi], c=c_data,
                                 cmap=self.cmap, norm=norm,
                                 s=self.marker_size, alpha=0.7, edgecolors='none')

            # Reference line for lattice parameter plot
            if self.y_col == 'FitA':
                ideal_a = d[0, COL['IdealA']]
                self.ax.axhline(ideal_a, color='red', ls='-', alpha=0.5,
                                label=f'Ideal a = {ideal_a:.6f} Å')
                self.ax.legend(fontsize=8)

            self.ax.set_xlabel(self.x_col)
            self.ax.set_ylabel(self.y_col)
            self.ax.set_title(f'{self.filename}  ({n_shown}/{n_total} pts)', fontsize=10)

        # Colorbar — reuse fixed cax
        self.cax.clear()
        self._cbar = self.fig.colorbar(sc, cax=self.cax, label=c_label)
        self.canvas.draw_idle()


# ── Entry point ─────────────────────────────────────────────────────────
def main():
    app = QApplication.instance() or QApplication(sys.argv)

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        files = find_corr_files()
        filename = files[0] if files else None

    if filename:
        print(f'Loading {filename} …')
    else:
        print('No *corr.csv in cwd — use File > Open… in the viewer')

    viewer = CalibrantViewer(filename)
    viewer.show()
    sys.exit(app.exec())


# MIDAS version banner
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'utils'))
    from version import version_string as _vs
    print(_vs())
except Exception:
    pass

if __name__ == '__main__':
    main()
