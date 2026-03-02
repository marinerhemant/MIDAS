#!/usr/bin/env python3
"""
Real-Time Visualization Dashboard for IntegratorFitPeaksGPUStream

Tails binary output files (lineout.bin, fit.bin) written by the GPU integrator
and displays live-updating plots using PyQtGraph.

Three panels:
  1. Heatmap of radial lineouts over time (scrolling)
  2. Current lineout (Intensity vs R)
  3. Peak parameter evolution (user-selectable peaks & params)

Usage:
  python live_viewer.py --lineout lineout.bin --fit fit.bin --nRBins 500 --nPeaks 5
  python live_viewer.py --lineout lineout.bin  # lineout-only mode (no peak panel)

Author: Hemant Sharma
"""

import argparse
import os
import sys
import time
import math
import json
from collections import deque

import numpy as np

# Ensure we can use headless fallback
os.environ.setdefault('QT_API', 'pyqt5')

from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

# ---------------- Constants ----------------
TIMER_INTERVAL_MS = 33  # ~30 fps
PARAM_NAMES = ['Imax', 'BG', 'Mixing (η)', 'Center', 'σ (width)', 'GoF', 'Area']
PARAM_SHORT = ['Imax', 'BG', 'η', 'Center', 'σ', 'GoF', 'Area']
COLORMAPS = ['viridis', 'inferno', 'plasma', 'magma', 'turbo', 'gray', 'gray_r', 'hot', 'cool']


# ============================================================
# Binary File Tailer
# ============================================================
class BinaryTailer:
    """Tails a binary file, reading complete records as they appear."""

    def __init__(self, filepath, record_doubles):
        """
        Parameters
        ----------
        filepath : str
            Path to binary file.
        record_doubles : int
            Number of doubles per record (frame).
        """
        self.filepath = filepath
        self.record_bytes = record_doubles * 8
        self.record_doubles = record_doubles
        self.offset = 0
        self._fh = None

    def open(self):
        """Open the file if it exists. Returns True if opened."""
        if self._fh is not None:
            return True
        if os.path.isfile(self.filepath):
            self._fh = open(self.filepath, 'rb')
            return True
        return False

    def seek_to_end(self):
        """Jump to current end of file (for reset)."""
        if self._fh:
            self._fh.seek(0, 2)
            self.offset = self._fh.tell()

    def read_new(self):
        """Read any new complete records. Returns list of numpy arrays."""
        if not self.open():
            return []

        records = []
        while True:
            data = self._fh.read(self.record_bytes)
            if data is None or len(data) < self.record_bytes:
                # Incomplete record — seek back
                if data:
                    self._fh.seek(-len(data), 1)
                break
            arr = np.frombuffer(data, dtype=np.float64)
            records.append(arr.copy())
            self.offset += self.record_bytes
        return records

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# ============================================================
# Scaled Axis Item (non-linear tick transform)
# ============================================================
class ScaledAxisItem(pg.AxisItem):
    """AxisItem whose tick labels are transformed through a function.

    Used to show 2θ or Q axes that correspond to the primary R axis.
    """

    def __init__(self, transform_fn, label, units='', *args, **kwargs):
        super().__init__(*args, orientation='bottom', **kwargs)
        self._transform = transform_fn
        self.setLabel(label, units=units)
        self.setStyle(tickLength=-5)

    def tickStrings(self, values, scale, spacing):
        """Convert R tick values through the transform function."""
        result = []
        for v in values:
            try:
                tv = self._transform(v * scale)
                result.append(f'{tv:.2f}')
            except (ValueError, ZeroDivisionError, OverflowError):
                result.append('')
        return result


# ============================================================
# Parameter file parser
# ============================================================
def parse_param_file(path):
    """Extract Lsd, px, Wavelength, RMin, RMax from a MIDAS parameter file.

    Returns dict with keys: lsd, px, wavelength (all in µm/Å as stored).
    Missing keys are omitted.
    """
    result = {}
    key_map = {
        'Lsd': 'lsd',
        'px': 'px',
        'Wavelength': 'wavelength',
        'RMin': 'rmin',
        'RMax': 'rmax',
        'RBinSize': 'rbinsize',
    }
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[0] in key_map:
                    result[key_map[parts[0]]] = float(parts[1])
    except Exception as e:
        print(f'Warning: could not parse param file {path}: {e}')
    return result


# ============================================================
# Main Viewer Window
# ============================================================
class LiveViewer(QtWidgets.QMainWindow):
    def __init__(self, lineout_path, fit_path=None, n_rbins=500, n_peaks=0,
                 max_history=0, theme='light', lsd=None, px=None,
                 wavelength=None, work_dir=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('MIDAS Live Viewer')
        self.resize(1400, 900)

        self.n_rbins = n_rbins
        self.n_peaks = n_peaks
        self.max_history = max_history  # 0 = unlimited
        self.current_theme = theme
        self.lsd = lsd  # sample-detector distance (µm)
        self.px = px    # pixel size (µm)
        self.wavelength = wavelength  # X-ray wavelength (Å)
        self.work_dir = work_dir or os.path.dirname(os.path.abspath(lineout_path))

        # Data storage
        self.lineout_history = deque(maxlen=max_history if max_history > 0 else None)
        self.r_values = None  # will be set from first record
        self.fit_history = deque(maxlen=max_history if max_history > 0 else None)
        self.frame_count = 0

        # Tailers
        self.lineout_tailer = BinaryTailer(lineout_path, n_rbins * 2)
        self.fit_tailer = BinaryTailer(fit_path, n_peaks * 7) if fit_path and n_peaks > 0 else None

        # State
        self.paused = False
        self.log_lineout = False
        self.log_heatmap = False
        self.selected_params = [3, 0]  # Center, Imax by default
        self.selected_peaks = list(range(n_peaks))  # All peaks selected initially
        self.decimation = 1  # show every Nth frame in heatmap
        self.font_size = 10  # base font size in pt

        # Peak pick state
        self._pick_mode = False
        self._picked_peaks = []  # list of R values
        self._pick_lines_lo = []  # InfiniteLine objects on lineout
        self._pick_lines_hm = []  # InfiniteLine objects on heatmap
        self._active_fit_lines_lo = []  # active fit overlays on lineout
        self._active_fit_lines_hm = []  # active fit overlays on heatmap

        # FPS tracking
        self._fps_time = time.monotonic()
        self._fps_frames = 0
        self._fps_value = 0.0

        self._apply_theme(theme)
        self._build_ui()
        self._setup_timer()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # ---- Toolbar ----
        toolbar = QtWidgets.QHBoxLayout()

        # Colormap
        toolbar.addWidget(QtWidgets.QLabel('Colormap:'))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        toolbar.addWidget(self.cmap_combo)

        toolbar.addSpacing(15)

        # Theme
        toolbar.addWidget(QtWidgets.QLabel('Theme:'))
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(['light', 'dark'])
        self.theme_combo.setCurrentText(self.current_theme)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        toolbar.addWidget(self.theme_combo)

        toolbar.addSpacing(15)

        # Log toggles
        self.chk_log_heatmap = QtWidgets.QCheckBox('Log Heatmap')
        self.chk_log_heatmap.toggled.connect(self._on_log_heatmap)
        toolbar.addWidget(self.chk_log_heatmap)

        self.chk_log_lineout = QtWidgets.QCheckBox('Log Lineout')
        self.chk_log_lineout.toggled.connect(self._on_log_lineout)
        toolbar.addWidget(self.chk_log_lineout)

        toolbar.addSpacing(15)

        # History depth
        toolbar.addWidget(QtWidgets.QLabel('History:'))
        self.spin_history = QtWidgets.QSpinBox()
        self.spin_history.setRange(0, 100000)
        self.spin_history.setValue(self.max_history)
        self.spin_history.setSpecialValueText('Unlimited')
        self.spin_history.setToolTip('Max frames in history (0 = unlimited)')
        self.spin_history.valueChanged.connect(self._on_history_changed)
        toolbar.addWidget(self.spin_history)

        toolbar.addSpacing(15)

        # Decimation
        toolbar.addWidget(QtWidgets.QLabel('Decimate:'))
        self.spin_decimate = QtWidgets.QSpinBox()
        self.spin_decimate.setRange(1, 100)
        self.spin_decimate.setValue(1)
        self.spin_decimate.setToolTip('Show every Nth frame in heatmap (1 = all)')
        self.spin_decimate.valueChanged.connect(self._on_decimate_changed)
        toolbar.addWidget(self.spin_decimate)

        # Font size
        toolbar.addWidget(QtWidgets.QLabel('Font:'))
        self.spin_font = QtWidgets.QSpinBox()
        self.spin_font.setRange(8, 24)
        self.spin_font.setValue(self.font_size)
        self.spin_font.setSuffix('pt')
        self.spin_font.setToolTip('UI font size')
        self.spin_font.valueChanged.connect(self._on_font_changed)
        toolbar.addWidget(self.spin_font)

        toolbar.addSpacing(15)

        # Pause / Reset
        self.btn_pause = QtWidgets.QPushButton('⏸ Pause')
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._on_pause)
        toolbar.addWidget(self.btn_pause)

        self.btn_reset = QtWidgets.QPushButton('🔄 Reset')
        self.btn_reset.clicked.connect(self._on_reset)
        toolbar.addWidget(self.btn_reset)

        toolbar.addStretch()

        # Frame counter + FPS
        self.lbl_frames = QtWidgets.QLabel('Frames: 0')
        self.lbl_frames.setStyleSheet('font-weight: bold;')
        toolbar.addWidget(self.lbl_frames)
        self.lbl_fps = QtWidgets.QLabel('  FPS: --')
        self.lbl_fps.setStyleSheet('font-weight: bold; color: #2a82da;')
        toolbar.addWidget(self.lbl_fps)

        main_layout.addLayout(toolbar)

        # ---- Peak Pick toolbar ----
        pick_bar = QtWidgets.QHBoxLayout()
        self.btn_pick = QtWidgets.QPushButton('\U0001F3AF Pick')
        self.btn_pick.setCheckable(True)
        self.btn_pick.setToolTip('Click on lineout/heatmap to select peak R values')
        self.btn_pick.toggled.connect(self._on_pick_toggled)
        pick_bar.addWidget(self.btn_pick)

        self.btn_clear_picks = QtWidgets.QPushButton('\u274C Clear')
        self.btn_clear_picks.setToolTip('Remove all picked peak markers')
        self.btn_clear_picks.clicked.connect(self._on_clear_picks)
        pick_bar.addWidget(self.btn_clear_picks)

        pick_bar.addSpacing(10)
        pick_bar.addWidget(QtWidgets.QLabel('Mode:'))
        self.send_mode_combo = QtWidgets.QComboBox()
        self.send_mode_combo.addItems(['Replace', 'Append'])
        self.send_mode_combo.setToolTip('Replace: fit only picked peaks.  Append: add to current set.')
        pick_bar.addWidget(self.send_mode_combo)

        self.btn_send = QtWidgets.QPushButton('\U0001F4E4 Send Peaks')
        self.btn_send.setToolTip('Write picked peaks to peak_update.txt for GPU process')
        self.btn_send.clicked.connect(self._on_send_peaks)
        pick_bar.addWidget(self.btn_send)

        pick_bar.addSpacing(20)
        self.lbl_picks = QtWidgets.QLabel('Picked: 0')
        pick_bar.addWidget(self.lbl_picks)
        pick_bar.addStretch()
        main_layout.addLayout(pick_bar)

        # ---- Peak param selector (if peaks) ----
        if self.n_peaks > 0:
            param_bar = QtWidgets.QHBoxLayout()
            param_bar.addWidget(QtWidgets.QLabel('Params:'))
            self.param_checks = []
            for i, name in enumerate(PARAM_SHORT):
                chk = QtWidgets.QCheckBox(name)
                chk.setChecked(i in self.selected_params)
                chk.toggled.connect(self._on_param_selection_changed)
                self.param_checks.append(chk)
                param_bar.addWidget(chk)

            param_bar.addSpacing(20)
            param_bar.addWidget(QtWidgets.QLabel('Peaks:'))

            btn_all = QtWidgets.QPushButton('All')
            btn_all.setFixedWidth(40)
            btn_all.clicked.connect(lambda: self._set_all_peaks(True))
            param_bar.addWidget(btn_all)
            btn_none = QtWidgets.QPushButton('None')
            btn_none.setFixedWidth(40)
            btn_none.clicked.connect(lambda: self._set_all_peaks(False))
            param_bar.addWidget(btn_none)

            self.peak_checks = []
            for p in range(self.n_peaks):
                chk = QtWidgets.QCheckBox(f'Pk {p}')
                chk.setChecked(True)
                chk.toggled.connect(self._on_peak_selection_changed)
                self.peak_checks.append(chk)
                param_bar.addWidget(chk)

            param_bar.addStretch()
            main_layout.addLayout(param_bar)

        # ---- Plot panels ----
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Top row: heatmap + lineout side by side
        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Panel 1: Heatmap
        self.heatmap_widget = pg.PlotWidget(title='Lineout Heatmap')
        self.heatmap_widget.setLabel('left', 'Frame #')
        self.heatmap_widget.setLabel('bottom', 'R (pixels)')
        self.heatmap_img = pg.ImageItem()
        self.heatmap_widget.addItem(self.heatmap_img)
        self._apply_colormap('viridis')
        # Click handler for peak pick on heatmap
        self.heatmap_widget.scene().sigMouseClicked.connect(self._on_heatmap_clicked)
        top_splitter.addWidget(self.heatmap_widget)

        # Panel 2: Current lineout
        self.lineout_widget = pg.PlotWidget(title='Current Lineout')
        self.lineout_widget.setLabel('left', 'Intensity')
        self.lineout_widget.setLabel('bottom', 'R (pixels)')
        self.lineout_widget.showGrid(x=True, y=True, alpha=0.3)
        self.lineout_curve = self.lineout_widget.plot(pen=pg.mkPen('c', width=1.5))
        # Click handler for peak pick on lineout
        self.lineout_widget.scene().sigMouseClicked.connect(self._on_lineout_clicked)
        top_splitter.addWidget(self.lineout_widget)

        # ---- Add secondary axes (2θ, Q) to both plots ----
        self._extra_axes = []  # keep references
        if self.lsd is not None and self.px is not None:
            def r_to_twotheta(r):
                return math.degrees(math.atan(r * self.px / self.lsd))

            for pw in [self.lineout_widget, self.heatmap_widget]:
                ax_2th = ScaledAxisItem(r_to_twotheta, '2θ', units='°')
                pi = pw.plotItem
                pi.layout.addItem(ax_2th, pi.layout.rowCount(), 1)
                ax_2th.linkToView(pi.vb)
                self._extra_axes.append(ax_2th)

            if self.wavelength is not None and self.wavelength > 0:
                def r_to_q(r):
                    theta = math.atan(r * self.px / self.lsd) / 2.0
                    return 4.0 * math.pi * math.sin(theta) / self.wavelength

                for pw in [self.lineout_widget, self.heatmap_widget]:
                    ax_q = ScaledAxisItem(r_to_q, 'Q', units='Å⁻¹')
                    pi = pw.plotItem
                    pi.layout.addItem(ax_q, pi.layout.rowCount(), 1)
                    ax_q.linkToView(pi.vb)
                    self._extra_axes.append(ax_q)

        top_splitter.setSizes([700, 700])
        splitter.addWidget(top_splitter)

        # Panel 3: Peak parameter evolution (only if peaks)
        if self.n_peaks > 0:
            self.peak_layout_widget = pg.GraphicsLayoutWidget(title='Peak Evolution')
            splitter.addWidget(self.peak_layout_widget)
            self.peak_plots = {}  # (peak_idx, param_idx) -> (plot, curve)
            self._rebuild_peak_grid()

        splitter.setSizes([500, 400])
        main_layout.addWidget(splitter)

    def _apply_theme(self, theme):
        """Apply dark or light theme to Qt palette and PyQtGraph."""
        app = QtWidgets.QApplication.instance()
        app.setStyle('Fusion')
        if theme == 'dark':
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
            palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(40, 40, 40))
            palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
            palette.setColor(QtGui.QPalette.Button, QtGui.QColor(50, 50, 50))
            palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
            app.setPalette(palette)
            pg.setConfigOptions(background='k', foreground='w')
        else:
            app.setPalette(app.style().standardPalette())
            pg.setConfigOptions(background='w', foreground='k')
        self.current_theme = theme

    def _apply_colormap(self, name):
        """Apply a colormap to the heatmap. Falls back to matplotlib if needed."""
        try:
            cmap = pg.colormap.get(name)
        except Exception:
            try:
                cmap = pg.colormap.getFromMatplotlib(name)
            except Exception:
                cmap = pg.colormap.get('viridis')
        lut = cmap.getLookupTable(nPts=256)
        self.heatmap_img.setLookupTable(lut)

    def _rebuild_peak_grid(self):
        """Rebuild the peak evolution plot grid based on selected params."""
        self.peak_layout_widget.clear()
        self.peak_plots = {}

        sel_params = [i for i, chk in enumerate(self.param_checks) if chk.isChecked()]
        self.selected_params = sel_params

        # Sort selected peaks by fitted Center (or by index if no data)
        sel_peaks = [i for i, chk in enumerate(self.peak_checks) if chk.isChecked()]
        self.selected_peaks = sel_peaks

        if not sel_params or not sel_peaks:
            return

        peak_order = list(sel_peaks)
        if self.fit_history:
            last_fit = self.fit_history[-1]
            centers = {p: last_fit[p * 7 + 3] for p in sel_peaks if p * 7 + 3 < len(last_fit)}
            peak_order = sorted(sel_peaks, key=lambda p: centers.get(p, p))

        colors = pg.intColor(0, hues=max(self.n_peaks, 1))
        for row, pk in enumerate(peak_order):
            for col, pi in enumerate(sel_params):
                p = self.peak_layout_widget.addPlot(row=row, col=col)
                fs = f'{self.font_size}pt'
                if row == 0:
                    p.setTitle(PARAM_SHORT[pi], size=fs)
                if col == 0:
                    p.setLabel('left', f'Pk {pk}', size=fs)
                p.showGrid(y=True, alpha=0.2)
                color = pg.intColor(pk, hues=max(self.n_peaks, 1))
                curve = p.plot(pen=pg.mkPen(color, width=1.5))
                self.peak_plots[(pk, pi)] = (p, curve)

    def _setup_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(TIMER_INTERVAL_MS)

    # ---- Callbacks ----
    def _on_theme_changed(self, theme):
        self._apply_theme(theme)
        # Force re-show to propagate palette changes
        self.hide()
        self.show()

    def _on_cmap_changed(self, name):
        self._apply_colormap(name)
        self._redraw_heatmap()

    def _on_decimate_changed(self, val):
        self.decimation = max(1, val)
        self._redraw_heatmap()

    def _on_font_changed(self, size):
        self.font_size = size
        font = QtGui.QFont()
        font.setPointSize(size)
        QtWidgets.QApplication.instance().setFont(font)
        # Also scale pyqtgraph axis labels and titles
        tick_font = QtGui.QFont()
        tick_font.setPointSize(max(size - 2, 6))
        for pw in [self.heatmap_widget, self.lineout_widget]:
            for axis_name in ['bottom', 'left']:
                ax = pw.getAxis(axis_name)
                ax.setTickFont(tick_font)
                ax.label.setFont(font)
        if self.n_peaks > 0:
            self._rebuild_peak_grid()

    def _on_log_heatmap(self, checked):
        self.log_heatmap = checked
        self._redraw_heatmap()

    def _on_log_lineout(self, checked):
        self.log_lineout = checked
        self.lineout_widget.setLogMode(x=False, y=checked)

    def _on_history_changed(self, val):
        self.max_history = val
        new_max = val if val > 0 else None
        # Rebuild deques with new maxlen
        self.lineout_history = deque(self.lineout_history, maxlen=new_max)
        self.fit_history = deque(self.fit_history, maxlen=new_max)

    def _on_pause(self, checked):
        self.paused = checked
        self.btn_pause.setText('▶ Resume' if checked else '⏸ Pause')

    def _on_reset(self):
        self.lineout_history.clear()
        self.fit_history.clear()
        self.frame_count = 0
        self._fps_time = time.monotonic()
        self._fps_frames = 0
        self._fps_value = 0.0
        self.lineout_tailer.seek_to_end()
        if self.fit_tailer:
            self.fit_tailer.seek_to_end()
        self._redraw_heatmap()
        self.lineout_curve.setData([], [])
        if self.n_peaks > 0:
            for (pk, pi), (plot, curve) in self.peak_plots.items():
                curve.setData([], [])
        self.lbl_frames.setText('Frames: 0')
        self.lbl_fps.setText('  FPS: --')

    def _on_param_selection_changed(self):
        self._rebuild_peak_grid()
        self._redraw_peak_evolution()

    def _on_peak_selection_changed(self):
        self._rebuild_peak_grid()
        self._redraw_peak_evolution()

    def _set_all_peaks(self, checked):
        for chk in self.peak_checks:
            chk.setChecked(checked)

    # ---- Peak Pick ----
    def _on_pick_toggled(self, checked):
        self._pick_mode = checked
        self.btn_pick.setText('\U0001F3AF Pick ✓' if checked else '\U0001F3AF Pick')

    def _add_pick_marker(self, r_val):
        """Add a yellow marker at the given R value on both plots."""
        self._picked_peaks.append(r_val)
        pen = pg.mkPen(color='y', width=1.5, style=QtCore.Qt.DashLine)
        line_lo = pg.InfiniteLine(pos=r_val, angle=90, pen=pen,
                                  label=f'{r_val:.1f}', labelOpts={'position': 0.9, 'color': 'y'})
        self.lineout_widget.addItem(line_lo)
        self._pick_lines_lo.append(line_lo)
        line_hm = pg.InfiniteLine(pos=r_val, angle=90, pen=pen)
        self.heatmap_widget.addItem(line_hm)
        self._pick_lines_hm.append(line_hm)
        self.lbl_picks.setText(f'Picked: {len(self._picked_peaks)}')

    def _on_lineout_clicked(self, ev):
        if not self._pick_mode:
            return
        vb = self.lineout_widget.plotItem.vb
        pos = vb.mapSceneToView(ev.scenePos())
        self._add_pick_marker(pos.x())

    def _on_heatmap_clicked(self, ev):
        if not self._pick_mode:
            return
        vb = self.heatmap_widget.plotItem.vb
        pos = vb.mapSceneToView(ev.scenePos())
        self._add_pick_marker(pos.x())

    def _on_clear_picks(self):
        for line in self._pick_lines_lo:
            self.lineout_widget.removeItem(line)
        for line in self._pick_lines_hm:
            self.heatmap_widget.removeItem(line)
        self._pick_lines_lo.clear()
        self._pick_lines_hm.clear()
        self._picked_peaks.clear()
        self.lbl_picks.setText('Picked: 0')

    def _on_send_peaks(self):
        if not self._picked_peaks:
            print('No peaks picked — nothing to send.')
            return
        mode = self.send_mode_combo.currentText().lower()
        peaks = sorted(self._picked_peaks)

        # Write atomically via temp + rename
        out_path = os.path.join(self.work_dir, 'peak_update.txt')
        tmp_path = out_path + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                f.write(f'mode {mode}\n')
                for p in peaks:
                    f.write(f'{p:.4f}\n')
            os.replace(tmp_path, out_path)
            print(f'Sent {len(peaks)} peaks ({mode}): {peaks}')
            # Flash markers green
            green_pen = pg.mkPen(color='g', width=2, style=QtCore.Qt.DashLine)
            for line in self._pick_lines_lo + self._pick_lines_hm:
                line.setPen(green_pen)
        except Exception as e:
            print(f'Failed to write peak_update.txt: {e}')

    def _update_active_overlays(self):
        """Read active_peaks.txt and draw red dashed lines on both plots."""
        ap_path = os.path.join(self.work_dir, 'active_peaks.txt')
        radii = []
        try:
            if os.path.exists(ap_path):
                with open(ap_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            radii.append(float(line))
        except Exception:
            pass

        # Also extract from fit_history if available and no file
        if not radii and self.fit_history and self.n_peaks > 0:
            last = self.fit_history[-1]
            for p in range(self.n_peaks):
                idx = p * 7 + 3  # Center
                if idx < len(last) and last[idx] > 0:
                    radii.append(last[idx])

        # Update overlay lines — reuse existing or create new
        red_pen = pg.mkPen(color='r', width=1.5, style=QtCore.Qt.DashLine)
        # Remove excess
        while len(self._active_fit_lines_lo) > len(radii):
            self.lineout_widget.removeItem(self._active_fit_lines_lo.pop())
            self.heatmap_widget.removeItem(self._active_fit_lines_hm.pop())
        # Add missing
        while len(self._active_fit_lines_lo) < len(radii):
            lo = pg.InfiniteLine(angle=90, pen=red_pen)
            hm = pg.InfiniteLine(angle=90, pen=red_pen)
            self.lineout_widget.addItem(lo)
            self.heatmap_widget.addItem(hm)
            self._active_fit_lines_lo.append(lo)
            self._active_fit_lines_hm.append(hm)
        # Update positions
        for i, r in enumerate(radii):
            self._active_fit_lines_lo[i].setPos(r)
            self._active_fit_lines_hm[i].setPos(r)

    # ---- Update loop ----
    def _update(self):
        if self.paused:
            return

        # Read new lineout records
        new_lineouts = self.lineout_tailer.read_new()
        new_fits = []
        if self.fit_tailer:
            new_fits = self.fit_tailer.read_new()

        if not new_lineouts:
            return

        n_new = len(new_lineouts)

        for rec in new_lineouts:
            # rec is nRBins*2 doubles: [R0, I0, R1, I1, ...]
            r_vals = rec[0::2]
            i_vals = rec[1::2]
            if self.r_values is None:
                self.r_values = r_vals.copy()
            self.lineout_history.append(i_vals.copy())
            self.frame_count += 1

        for rec in new_fits:
            self.fit_history.append(rec.copy())

        # Update frame label + FPS
        self._fps_frames += n_new
        now = time.monotonic()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._fps_value = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_time = now
            if self.frame_count % 100 < n_new:
                print(f'  [{self.frame_count} frames] {self._fps_value:.1f} fps')
        self.lbl_frames.setText(f'Frames: {self.frame_count}')
        fps_str = f'{self._fps_value:.1f}' if self._fps_value > 0 else '--'
        self.lbl_fps.setText(f'  FPS: {fps_str}')

        # Update lineout (last frame)
        if self.lineout_history:
            last_i = self.lineout_history[-1]
            r = self.r_values if self.r_values is not None else np.arange(len(last_i))
            self.lineout_curve.setData(r, last_i)

        # Update heatmap
        self._redraw_heatmap()

        # Update peak evolution
        if new_fits and self.n_peaks > 0:
            self._redraw_peak_evolution()

        # Update active fit overlays (every 10th frame to avoid disk thrashing)
        if self.frame_count % 10 == 0:
            self._update_active_overlays()

    def _redraw_heatmap(self):
        if not self.lineout_history:
            return
        img = np.array(self.lineout_history)
        if self.decimation > 1:
            img = img[::self.decimation]
        if self.log_heatmap:
            img = np.log10(np.clip(img, 1e-10, None))
        self.heatmap_img.setImage(img.T, autoLevels=True)

    def _redraw_peak_evolution(self):
        if not self.fit_history:
            return

        n_frames = len(self.fit_history)
        frame_nums = np.arange(self.frame_count - n_frames, self.frame_count)

        for (pk, pi), (plot, curve) in self.peak_plots.items():
            vals = np.array([self.fit_history[f][pk * 7 + pi]
                            for f in range(n_frames)
                            if pk * 7 + pi < len(self.fit_history[f])])
            if len(vals) > 0:
                x = frame_nums[:len(vals)]
                curve.setData(x, vals)

    # ---- Cleanup ----
    def closeEvent(self, event):
        self.timer.stop()
        self.lineout_tailer.close()
        if self.fit_tailer:
            self.fit_tailer.close()
        super().closeEvent(event)


# ============================================================
# Programmatic API
# ============================================================
def launch_viewer(lineout_path, fit_path=None, n_rbins=500, n_peaks=0,
                  max_history=0):
    """Launch the viewer in a subprocess (non-blocking).

    Parameters
    ----------
    lineout_path : str
        Path to lineout.bin
    fit_path : str, optional
        Path to fit.bin
    n_rbins : int
        Number of radial bins
    n_peaks : int
        Number of peaks per frame
    max_history : int
        Max frames to keep (0 = unlimited)
    """
    import subprocess
    cmd = [sys.executable, __file__,
           '--lineout', str(lineout_path),
           '--nRBins', str(n_rbins)]
    if fit_path:
        cmd.extend(['--fit', str(fit_path), '--nPeaks', str(n_peaks)])
    if max_history > 0:
        cmd.extend(['--history', str(max_history)])
    return subprocess.Popen(cmd)


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='MIDAS real-time visualization for IntegratorFitPeaksGPUStream')
    parser.add_argument('--lineout', required=True,
                        help='Path to lineout.bin (nRBins×2 doubles per frame)')
    parser.add_argument('--fit', default=None,
                        help='Path to fit.bin (nPeaks×7 doubles per frame)')
    parser.add_argument('--nRBins', type=int, required=True,
                        help='Number of radial bins per lineout')
    parser.add_argument('--nPeaks', type=int, default=0,
                        help='Number of peaks per frame (0 = no peak panel)')
    parser.add_argument('--history', type=int, default=0,
                        help='Max history frames (0 = unlimited)')
    parser.add_argument('--theme', choices=['dark', 'light'], default='light',
                        help='UI theme (default: light)')
    parser.add_argument('--params', default=None,
                        help='MIDAS parameter file (extracts Lsd, px, Wavelength)')
    parser.add_argument('--lsd', type=float, default=None,
                        help='Sample-detector distance (µm), overrides --params')
    parser.add_argument('--px', type=float, default=None,
                        help='Pixel size (µm), overrides --params')
    parser.add_argument('--wavelength', type=float, default=None,
                        help='X-ray wavelength (Å), overrides --params')
    args = parser.parse_args()

    # Resolve geometry from param file + explicit overrides
    lsd, px, wavelength = args.lsd, args.px, args.wavelength
    if args.params:
        pf = parse_param_file(args.params)
        if lsd is None:
            lsd = pf.get('lsd')
        if px is None:
            px = pf.get('px')
        if wavelength is None:
            wavelength = pf.get('wavelength')

    if lsd and px:
        print(f'Geometry: Lsd={lsd} µm, px={px} µm', end='')
        if wavelength:
            print(f', λ={wavelength} Å', end='')
        print('  →  2θ/Q axes enabled')
    else:
        print('Geometry not provided — 2θ/Q axes disabled')

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    pg.setConfigOptions(antialias=True)

    viewer = LiveViewer(
        lineout_path=args.lineout,
        fit_path=args.fit,
        n_rbins=args.nRBins,
        n_peaks=args.nPeaks,
        max_history=args.history,
        theme=args.theme,
        lsd=lsd,
        px=px,
        wavelength=wavelength,
    )
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
