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
    QLineEdit, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

# ── CSV column map (from CalibrantPanelShiftsOMP header) ────────────────
# Legacy 16-column space-separated format
COLUMNS_LEGACY = [
    'Eta', 'Strain', 'RadFit', 'EtaCalc', 'DiffCalc', 'RadCalc',
    'Ideal2Theta', 'Outlier', 'YRawCorr', 'ZRawCorr', 'RingNr',
    'RadGlobal', 'IdealR', 'Fit2Theta', 'IdealA', 'FitA',
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
      blank line, then '%Eta Strain ...' header + 16-col space-separated data.
    CalibrantPanelShiftsOMP: '%Eta Strain ...' header + 16-col data directly.

    Returns (data_array, column_names).
    """
    with open(filename) as f:
        lines = f.readlines()

    # Find the per-bin data header line (starts with '%Eta' or '%' followed by column names)
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('%Eta') or stripped.startswith('% Eta'):
            data_start = i
            break

    if data_start is not None:
        # Standard 16-column space-separated format (both executables)
        data = np.genfromtxt(filename, skip_header=data_start + 1)
        return data, COLUMNS_LEGACY

    # Fallback: try legacy format (first line is the header)
    data = np.genfromtxt(filename, skip_header=1)
    return data, COLUMNS_LEGACY


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
            self.raw, _ = _load_corr_csv(filename)
        else:
            self.raw = np.empty((0, 16))
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
        self.c_max = None

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── Control bar ─────────────────────────────────────────────
        controls = QHBoxLayout()

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

        # Color range
        controls.addWidget(QLabel('Min:'))
        self.edit_cmin = QLineEdit()
        self.edit_cmin.setPlaceholderText('auto')
        self.edit_cmin.setFixedWidth(70)
        self.edit_cmin.editingFinished.connect(self._on_crange)
        controls.addWidget(self.edit_cmin)

        controls.addWidget(QLabel('Max:'))
        self.edit_cmax = QLineEdit()
        self.edit_cmax.setPlaceholderText('auto')
        self.edit_cmax.setFixedWidth(70)
        self.edit_cmax.editingFinished.connect(self._on_crange)
        controls.addWidget(self.edit_cmax)

        # Log scale
        self.chk_log = QCheckBox('Log')
        self.chk_log.toggled.connect(self._on_log)
        controls.addWidget(self.chk_log)

        # Separator
        controls.addWidget(self._vsep())

        # Outlier toggle
        self.chk_outlier = QCheckBox('Show outliers')
        self.chk_outlier.setChecked(not self.exclude_outliers)
        self.chk_outlier.toggled.connect(self._on_outlier)
        controls.addWidget(self.chk_outlier)

        # Polar plot toggle
        self.chk_polar = QCheckBox('Polar')
        self.chk_polar.toggled.connect(self._on_polar)
        controls.addWidget(self.chk_polar)

        # Auto-range button
        self.btn_autorng = QPushButton('Auto range')
        self.btn_autorng.clicked.connect(self._on_autorange)
        controls.addWidget(self.btn_autorng)

        controls.addStretch()
        main_layout.addLayout(controls)

        # ── Matplotlib canvas ───────────────────────────────────────
        self.fig = Figure(figsize=(10, 6), dpi=100)
        # Fixed axes positions so colorbar doesn't steal space
        self.ax = self.fig.add_axes([0.08, 0.12, 0.78, 0.82])
        self.cax = self.fig.add_axes([0.88, 0.12, 0.02, 0.82])
        self.polar_mode = False
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
        elif text == 'Strain':
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
        self._rebuild_axes()
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

    def _load_file(self, path):
        try:
            data, _ = _load_corr_csv(path)
            self.raw = data
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
        if self.c_col == 'Strain':
            c_data = c_data * 1e6
            c_label = 'Strain (µε)'

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
                                 s=20, alpha=0.7, edgecolors='none')
            self.ax.set_theta_zero_location('N')  # 0° at top
            self.ax.set_theta_direction(-1)        # clockwise
            self.ax.set_title(f'{self.filename}  ({n_shown}/{n_total} pts)\n'
                              f'Radial: Strain (µε), Angular: η',
                              fontsize=10, pad=20)
        else:
            sc = self.ax.scatter(d[:, xi], d[:, yi], c=c_data,
                                 cmap=self.cmap, norm=norm,
                                 s=30, alpha=0.7, edgecolors='none')

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
