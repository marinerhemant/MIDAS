#!/usr/bin/env python3
"""
NF-HEDM Viewer — PyQt5/PyQtGraph edition.

Dual-panel viewer with diffraction image display and lineout/mic visualization.
All image I/O and crystallography logic preserved from nf.py.
"""

import sys
import os
import math
import tempfile
import subprocess
import threading
import concurrent.futures
import glob
import json

import numpy as np
from math import sin, cos

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import PIL.Image
except ImportError:
    PIL = None

_this_dir = os.path.dirname(os.path.abspath(__file__))
_utils_dir = os.path.join(os.path.dirname(_this_dir), 'utils')
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _utils_dir not in sys.path:
    sys.path.append(_utils_dir)

from gui_common import (MIDASImageView, apply_theme, get_colormap,
                         AsyncWorker, LogPanel, add_shortcut, COLORMAPS)

try:
    import midas_config
    midas_config.run_startup_checks()
except ImportError:
    midas_config = None

deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823


# ═══════════════════════════════════════════════════════════════════════
#  Crystallography helpers (from nf.py)
# ═══════════════════════════════════════════════════════════════════════

def YZ4mREta(R, Eta):
    return -R * math.sin(Eta * deg2rad), R * math.cos(Eta * deg2rad)

def rotationTransforms(ts):
    txr, tyr, tzr = ts[0] * deg2rad, ts[1] * deg2rad, ts[2] * deg2rad
    Rx = np.array([[1, 0, 0], [0, cos(txr), -sin(txr)], [0, sin(txr), cos(txr)]])
    Ry = np.array([[cos(tyr), 0, sin(tyr)], [0, 1, 0], [-sin(tyr), 0, cos(tyr)]])
    Rz = np.array([[cos(tzr), -sin(tzr), 0], [sin(tzr), cos(tzr), 0], [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))

def DisplacementSpots(a, b, Lsd, yi, zi, omega):
    OmegaRad = deg2rad * omega
    sinOme, cosOme = math.sin(OmegaRad), math.cos(OmegaRad)
    xa = a * cosOme - b * sinOme
    ya = a * sinOme + b * cosOme
    t = 1 - (xa / Lsd)
    return [ya + (yi * t), t * zi]

def euler2orientmat(Euler):
    psi, phi, theta = Euler[0], Euler[1], Euler[2]
    cps, cph, cth = math.cos(psi), math.cos(phi), math.cos(theta)
    sps, sph, sth = math.sin(psi), math.sin(phi), math.sin(theta)
    m = np.zeros(9)
    m[0] = cth * cps - sth * cph * sps
    m[1] = -cth * cph * sps - sth * cps
    m[2] = sph * sps
    m[3] = cth * sps + sth * cph * cps
    m[4] = cth * cph * cps - sth * sps
    m[5] = -sph * cps
    m[6] = sth * sph
    m[7] = cth * sph
    m[8] = cph
    return m


# ═══════════════════════════════════════════════════════════════════════
#  Image I/O
# ═══════════════════════════════════════════════════════════════════════

def read_tiff(fn):
    """Read a TIFF file, return uint16 array."""
    if HAS_TIFFFILE:
        return tifffile.imread(fn).astype(np.uint16)
    elif PIL:
        return np.array(PIL.Image.open(fn), dtype=np.uint16)
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Auto-detect
# ═══════════════════════════════════════════════════════════════════════

def nf_auto_detect(folder, fnstem):
    """Auto-detect start frame number and BeamPos mode."""
    result = {}
    cwd = os.getcwd()
    basename = os.path.basename(cwd)

    if 'BeamPos' in basename or 'DetZBeamPos' in basename:
        tifs = sorted(glob.glob(os.path.join(cwd, '*.tif')),
                      key=lambda fp: int(os.path.splitext(os.path.basename(fp))[0].split('_')[-1])
                      if os.path.splitext(os.path.basename(fp))[0].split('_')[-1].isdigit()
                      else float('inf'))
        if tifs:
            result['beampos_mode'] = True
            result['beampos_files'] = tifs
            result['start_frame'] = 0
        return result

    stem_path = os.path.join(folder, fnstem + '_')
    matching = glob.glob(stem_path + '*.tif')
    if matching:
        nums = []
        for f in matching:
            base = os.path.splitext(os.path.basename(f))[0]
            try:
                nums.append(int(base.split('_')[-1]))
            except ValueError:
                continue
        if nums:
            result['start_frame'] = min(nums)
    return result


# ═══════════════════════════════════════════════════════════════════════
#  NF Viewer Main Window
# ═══════════════════════════════════════════════════════════════════════

class NFViewer(QtWidgets.QMainWindow):
    """NF-HEDM dual-panel viewer with image and lineout/mic display."""

    def __init__(self, theme='light'):
        super().__init__()
        self.setWindowTitle("NF Viewer (PyQtGraph) — MIDAS")
        self.resize(1700, 950)
        self._theme = theme
        self._init_state()
        self._build_ui()
        self._wire_signals()
        self._setup_shortcuts()
        self._start_auto_detect()

    def _init_state(self):
        self.ny = 2048
        self.nz = 2048
        self.n_files_per_dist = 720
        self.padding = 6
        self.n_distances = 6
        self.background = 0
        self.folder = os.path.dirname(os.getcwd())
        self.fnstem = os.path.basename(os.getcwd()) + '/' + os.path.basename(os.getcwd())
        self.frame_nr = 0
        self.start_frame_nr = 0
        self.dist = 0
        self.pixel_size = 1.48
        self.lsd = 0.0
        self.min_thresh = 0
        self.max_thresh = 100
        self.use_log = False
        self.use_median = False
        self.max_over_frames = False
        self.sum_over_frames = False

        # Beam centers
        self.bcs = np.zeros((self.n_distances, 2))
        self.spots = np.zeros((self.n_distances, 3))
        self.dist_diff = 0.0

        self._selecting_spots = False
        self._click_ix = 0.0
        self._click_iy = 0.0
        self._spot_crops = {}  # {dist: ndarray} cropped image around spot

        # Image data
        self.imarr2 = None

        # Mic file
        self.mic_file = None
        self.mic_data = None
        self.mic_data_cut = None
        self.mic_type = 1  # 1=text, 2=binary map
        self.mic_size_x = 0
        self.mic_size_y = 0
        self.mic_ref_x = 0
        self.mic_ref_y = 0
        self.col_mode = 10  # confidence default
        self.cut_confidence = 0.0
        self.max_conf = 1.0

        # Grain simulation
        self.om = np.zeros(9)
        self.pos = np.zeros(3)
        self.latC = np.zeros(6)
        self.wl = 0.0
        self.startome = 0.0
        self.omestep = 0.0
        self.sg = 0
        self.maxringrad = 0.0
        self.simulated_spots = []
        self.spot_nr = 1

        # Median
        self._median_dir = None

        # Beampos mode
        self._beampos_mode = False
        self._beampos_files = []

        # Line profile state


        # Consolidated H5 state
        self._h5_path = None
        self._h5_resolutions = []  # list of resolution labels from H5
        self._h5_current_resolution = None

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Menu bar
        file_menu = self.menuBar().addMenu('&File')
        save_act = file_menu.addAction('Save Session...')
        save_act.setShortcut('Ctrl+S')
        save_act.triggered.connect(self._save_session)
        load_act = file_menu.addAction('Load Session...')
        load_act.setShortcut('Ctrl+Shift+S')
        load_act.triggered.connect(self._load_session)

        # Toolbar
        tb = self._build_toolbar()
        main_layout.addLayout(tb)

        # Dual panel: image + lineout/mic
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left: diffraction image
        self.image_view = MIDASImageView(self, name='Diffraction', origin='br')
        self.image_view.set_colormap('bone')
        self.splitter.addWidget(self.image_view)

        # Right: lineout or mic map
        self.right_widget = QtWidgets.QStackedWidget()

        # Lineout plot
        self.lineout_plot = pg.PlotWidget(title="Line Profile")
        self.lineout_plot.setLabel('bottom', 'Position', 'pixels')
        self.lineout_plot.setLabel('left', 'Intensity')
        self.right_widget.addWidget(self.lineout_plot)

        # Mic scatter view
        self.mic_view = pg.PlotWidget(title="Mic File")
        self.mic_view.setAspectLocked(True)
        self.mic_scatter = pg.ScatterPlotItem(size=5, pen=None)
        self.mic_view.addItem(self.mic_scatter)
        self.right_widget.addWidget(self.mic_view)

        # Mic image view (for binary map type)
        self.mic_image_view = MIDASImageView(self, name='MicMap')
        self.right_widget.addWidget(self.mic_image_view)

        self.right_widget.setCurrentIndex(0)
        self.splitter.addWidget(self.right_widget)
        self.splitter.setSizes([800, 800])

        main_layout.addWidget(self.splitter, stretch=1)

        # Control panels — use a splitter for resizable proportions
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(self._build_file_panel(), stretch=2)
        ctrl.addWidget(self._build_image_panel(), stretch=2)
        ctrl.addWidget(self._build_processing_panel(), stretch=2)
        ctrl.addWidget(self._build_analysis_panel(), stretch=1)
        ctrl.addWidget(self._build_mic_panel(), stretch=2)
        main_layout.addLayout(ctrl)

        # Status bar
        self.status_label = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)
        self.stats_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.stats_label)
        self.frame_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.frame_label)

        # Log panel
        self.log_panel = LogPanel(self, "Log")
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_panel)
        self.log_panel.install_redirect()
        self.log_panel.hide()

    def _build_toolbar(self):
        tb = QtWidgets.QHBoxLayout()
        tb.addWidget(QtWidgets.QLabel("Cmap:"))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText('bone')
        tb.addWidget(self.cmap_combo)

        tb.addWidget(QtWidgets.QLabel("Theme:"))
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(['light', 'dark'])
        self.theme_combo.setCurrentText(self._theme)
        tb.addWidget(self.theme_combo)

        tb.addWidget(QtWidgets.QLabel("Font:"))
        self.font_spin = QtWidgets.QSpinBox()
        self.font_spin.setRange(8, 24)
        self.font_spin.setValue(10)
        self.font_spin.valueChanged.connect(self._on_font_changed)
        tb.addWidget(self.font_spin)

        self.log_check = QtWidgets.QCheckBox("Log")
        tb.addWidget(self.log_check)

        export_btn = QtWidgets.QPushButton("Export PNG")
        export_btn.clicked.connect(lambda: self.image_view.export_png())
        tb.addWidget(export_btn)

        log_btn = QtWidgets.QPushButton("Log Panel")
        log_btn.setCheckable(True)
        log_btn.toggled.connect(lambda c: self.log_panel.show() if c else self.log_panel.hide())
        tb.addWidget(log_btn)

        help_btn = QtWidgets.QPushButton("Help")
        help_btn.clicked.connect(self._show_help)
        tb.addWidget(help_btn)

        tb.addStretch()
        return tb

    def _build_file_panel(self):
        grp = QtWidgets.QGroupBox("File I/O")
        lay = QtWidgets.QGridLayout(grp)
        btn = QtWidgets.QPushButton("FirstFile")
        btn.clicked.connect(self._on_first_file)
        lay.addWidget(btn, 0, 0)

        lay.addWidget(QtWidgets.QLabel("Folder"), 0, 1)
        self.folder_edit = QtWidgets.QLineEdit(self.folder)
        self.folder_edit.setMinimumWidth(160)
        lay.addWidget(self.folder_edit, 0, 2)

        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.clicked.connect(self._on_browse_folder)
        lay.addWidget(btn_browse, 0, 3)

        lay.addWidget(QtWidgets.QLabel("FNStem"), 0, 4)
        self.fnstem_edit = QtWidgets.QLineEdit(self.fnstem)
        self.fnstem_edit.setMinimumWidth(120)
        lay.addWidget(self.fnstem_edit, 0, 5)

        lay.addWidget(QtWidgets.QLabel("NrPixY"), 1, 0)
        self.ny_edit = QtWidgets.QLineEdit(str(self.ny))
        self.ny_edit.setMinimumWidth(70)
        lay.addWidget(self.ny_edit, 1, 1)

        lay.addWidget(QtWidgets.QLabel("NrPixZ"), 1, 2)
        self.nz_edit = QtWidgets.QLineEdit(str(self.nz))
        self.nz_edit.setMinimumWidth(70)
        lay.addWidget(self.nz_edit, 1, 3)
        return grp

    def _build_image_panel(self):
        grp = QtWidgets.QGroupBox("Image")
        lay = QtWidgets.QGridLayout(grp)

        lay.addWidget(QtWidgets.QLabel("Frame"), 0, 0)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, 99999)
        lay.addWidget(self.frame_spin, 0, 1)

        lay.addWidget(QtWidgets.QLabel("Dist"), 0, 2)
        self.dist_spin = QtWidgets.QSpinBox()
        self.dist_spin.setRange(0, 20)
        lay.addWidget(self.dist_spin, 0, 3)

        lay.addWidget(QtWidgets.QLabel("nDist"), 0, 4)
        self.ndist_edit = QtWidgets.QLineEdit(str(self.n_distances))
        self.ndist_edit.setMinimumWidth(50)
        lay.addWidget(self.ndist_edit, 0, 5)

        lay.addWidget(QtWidgets.QLabel("nFl/D"), 0, 6)
        self.nfiles_edit = QtWidgets.QLineEdit(str(self.n_files_per_dist))
        self.nfiles_edit.setMinimumWidth(55)
        lay.addWidget(self.nfiles_edit, 0, 7)

        lay.addWidget(QtWidgets.QLabel("PxSz"), 1, 0)
        self.px_edit = QtWidgets.QLineEdit(str(self.pixel_size))
        self.px_edit.setMinimumWidth(60)
        lay.addWidget(self.px_edit, 1, 1)
        return grp

    def _build_processing_panel(self):
        grp = QtWidgets.QGroupBox("Processing")
        lay = QtWidgets.QGridLayout(grp)

        lay.addWidget(QtWidgets.QLabel("StartNr"), 0, 0)
        self.startframe_edit = QtWidgets.QLineEdit(str(self.start_frame_nr))
        self.startframe_edit.setMinimumWidth(60)
        lay.addWidget(self.startframe_edit, 0, 1)

        btn_median = QtWidgets.QPushButton("CalcMedian")
        btn_median.clicked.connect(self._on_calc_median)
        lay.addWidget(btn_median, 0, 2)

        self.median_check = QtWidgets.QCheckBox("SubtMedian")
        lay.addWidget(self.median_check, 0, 3)

        self.maxframes_check = QtWidgets.QCheckBox("MaxOverFr")
        lay.addWidget(self.maxframes_check, 0, 4)

        self.sumframes_check = QtWidgets.QCheckBox("SumOverFr")
        lay.addWidget(self.sumframes_check, 0, 5)

        lay.addWidget(QtWidgets.QLabel("Lsd(μm)"), 1, 0)
        self.lsd_edit = QtWidgets.QLineEdit(str(self.lsd))
        self.lsd_edit.setMinimumWidth(90)
        lay.addWidget(self.lsd_edit, 1, 1, 1, 2)

        lay.addWidget(QtWidgets.QLabel("MinI"), 2, 0)
        self.min_intensity_edit = QtWidgets.QLineEdit("0")
        self.min_intensity_edit.setMinimumWidth(70)
        lay.addWidget(self.min_intensity_edit, 2, 1)

        lay.addWidget(QtWidgets.QLabel("MaxI"), 2, 2)
        self.max_intensity_edit = QtWidgets.QLineEdit("1000")
        self.max_intensity_edit.setMinimumWidth(70)
        lay.addWidget(self.max_intensity_edit, 2, 3)

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_intensity_levels)
        lay.addWidget(apply_btn, 2, 4)

        return grp

    def _build_analysis_panel(self):
        grp = QtWidgets.QGroupBox("Analysis")
        lay = QtWidgets.QGridLayout(grp)

        btn_boxh = QtWidgets.QPushButton("BoxH")
        btn_boxh.clicked.connect(lambda: self._add_box_roi('h'))
        lay.addWidget(btn_boxh, 0, 0)

        btn_boxv = QtWidgets.QPushButton("BoxV")
        btn_boxv.clicked.connect(lambda: self._add_box_roi('v'))
        lay.addWidget(btn_boxv, 0, 1)

        btn_bc = QtWidgets.QPushButton("BeamCenter")
        btn_bc.clicked.connect(self._on_beam_center)
        lay.addWidget(btn_bc, 0, 2)

        btn_select = QtWidgets.QPushButton("SelectSpots")
        btn_select.clicked.connect(self._on_select_spots)
        lay.addWidget(btn_select, 0, 3)
        return grp

    def _build_mic_panel(self):
        grp = QtWidgets.QGroupBox("Mic File")
        lay = QtWidgets.QGridLayout(grp)

        btn_load = QtWidgets.QPushButton("LoadMic")
        btn_load.clicked.connect(self._on_load_mic)
        lay.addWidget(btn_load, 0, 0)

        btn_reload = QtWidgets.QPushButton("ReloadMic")
        btn_reload.clicked.connect(self._plot_mic)
        lay.addWidget(btn_reload, 0, 1)

        btn_load_h5 = QtWidgets.QPushButton("Load H5")
        btn_load_h5.clicked.connect(self._on_load_h5)
        if not HAS_H5PY:
            btn_load_h5.setEnabled(False)
            btn_load_h5.setToolTip("h5py not installed")
        lay.addWidget(btn_load_h5, 0, 2)

        # Color mode radio buttons
        self.col_group = QtWidgets.QButtonGroup(self)
        col_modes = [
            ("Conf", 10), ("GrainID", 0), ("Phase", 11),
            ("Eu0", 7), ("Eu1", 8), ("Eu2", 9),
            ("KAM", 12), ("GROD", 13), ("GrMap", 14)
        ]
        row, col_idx = 1, 0
        for label, val in col_modes:
            rb = QtWidgets.QRadioButton(label)
            if val == 10:
                rb.setChecked(True)
            self.col_group.addButton(rb, val)
            lay.addWidget(rb, row, col_idx)
            col_idx += 1
            if col_idx >= 3:
                col_idx = 0
                row += 1

        lay.addWidget(QtWidgets.QLabel("ConfCut"), row, 0)
        self.cut_conf_edit = QtWidgets.QLineEdit("0")
        self.cut_conf_edit.setMinimumWidth(55)
        lay.addWidget(self.cut_conf_edit, row, 1)
        self.max_conf_edit = QtWidgets.QLineEdit("1")
        self.max_conf_edit.setMinimumWidth(55)
        lay.addWidget(self.max_conf_edit, row, 2)

        # Resolution selector (hidden by default, shown when H5 is loaded)
        row += 1
        self._res_selector_label = QtWidgets.QLabel("Resolution:")
        self._res_selector_label.setVisible(False)
        lay.addWidget(self._res_selector_label, row, 0)

        self._res_combo = QtWidgets.QComboBox()
        self._res_combo.setVisible(False)
        self._res_combo.currentTextChanged.connect(self._on_resolution_changed)
        lay.addWidget(self._res_combo, row, 1, 1, 2)

        row += 1
        btn_grain = QtWidgets.QPushButton("LoadGrain")
        btn_grain.clicked.connect(self._on_load_grain)
        lay.addWidget(btn_grain, row + 1, 0)

        btn_spots = QtWidgets.QPushButton("MakeSpots")
        btn_spots.clicked.connect(self._on_make_spots)
        lay.addWidget(btn_spots, row + 1, 1)

        btn_selectpt = QtWidgets.QPushButton("SelectPoint")
        btn_selectpt.clicked.connect(self._on_select_point)
        lay.addWidget(btn_selectpt, row + 1, 2)
        return grp

    # ── Signals ────────────────────────────────────────────────────

    def _wire_signals(self):
        self.frame_spin.valueChanged.connect(self._load_and_display)
        self.dist_spin.valueChanged.connect(self._load_and_display)
        self.log_check.toggled.connect(self._on_log_toggled)
        self.median_check.toggled.connect(self._load_and_display)
        self.maxframes_check.toggled.connect(self._on_max_toggled)
        self.sumframes_check.toggled.connect(self._on_sum_toggled)
        self.cmap_combo.currentTextChanged.connect(
            lambda n: self.image_view.set_colormap(n))
        self.theme_combo.currentTextChanged.connect(
            lambda t: apply_theme(QtWidgets.QApplication.instance(), t))
        self.image_view.frameScrolled.connect(
            lambda d: self.frame_spin.setValue(self.frame_spin.value() + d))
        self.image_view.cursorMoved.connect(self._on_cursor_moved)
        self.image_view.dataStatsUpdated.connect(self._on_stats_updated)
        self.col_group.buttonClicked.connect(self._on_col_mode_changed)
        # Movie mode: advance frame by 1 (wraps at max)
        self.image_view.movieFrameAdvance.connect(self._movie_advance_frame)
        # Drag-and-drop: open dropped file
        self.image_view.fileDropped.connect(self._on_file_dropped)

    def _on_font_changed(self, size):
        QtWidgets.QApplication.instance().setStyleSheet(f'* {{ font-size: {size}pt; }}')

    def _show_help(self):
        QtWidgets.QMessageBox.information(self, 'NF Viewer — Controls',
            'Mouse Controls:\n'
            '  Scroll wheel — Zoom in/out\n'
            '  Right-click drag — Zoom rectangle\n'
            '  Left-click drag — Pan\n'
            '  Ctrl+Scroll wheel — Change frame\n'
            '  Right-click → View All — Reset zoom\n'
            '\n'
            'Keyboard Shortcuts:\n'
            '  ← / → — Previous / Next frame\n'
            '  L — Toggle log scale\n'
            '  Q — Quit\n'
            '\n'
            'Histogram (right side of image):\n'
            '  Drag top/bottom bars — Adjust thresholds\n'
            '  Right-click histogram — Change colormap\n'
            '\n'
            'Box Profile:\n'
            '  Click BoxH/V, then draw a rectangle on the image\n')

    def _setup_shortcuts(self):
        add_shortcut(self, 'Right', lambda: self.frame_spin.setValue(self.frame_spin.value() + 1))
        add_shortcut(self, 'Left', lambda: self.frame_spin.setValue(self.frame_spin.value() - 1))
        add_shortcut(self, 'L', lambda: self.log_check.toggle())
        add_shortcut(self, 'Q', self.close)

    # ── Session Save / Load ────────────────────────────────────────

    def _save_session(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Session', '', 'Session Files (*.session.json);;All (*)')
        if not fn:
            return
        if not fn.endswith('.session.json'):
            fn += '.session.json'
        state = {
            'viewer': 'nf',
            'folder': self.folder,
            'fnstem': self.fnstem,
            'padding': self.padding,
            'start_frame_nr': self.start_frame_nr,
            'n_files_per_dist': self.n_files_per_dist,
            'frame': self.frame_spin.value(),
            'distance': self.dist_spin.value(),
            'ny': self.ny, 'nz': self.nz,
            'colormap': self.cmap_combo.currentText(),
            'theme': self.theme_combo.currentText(),
            'log': self.log_check.isChecked(),
            'hflip': self.hflip_check.isChecked(),
            'vflip': self.vflip_check.isChecked(),
            'transpose': self.transpose_check.isChecked(),
            'median': self.median_check.isChecked(),
        }
        try:
            with open(fn, 'w') as f:
                json.dump(state, f, indent=2)
            print(f'Session saved: {fn}')
        except Exception as e:
            print(f'Session save failed: {e}')

    def _load_session(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Session', '', 'Session Files (*.session.json);;All (*)')
        if not fn:
            return
        try:
            with open(fn) as f:
                state = json.load(f)
        except Exception as e:
            print(f'Session load failed: {e}')
            return
        self.folder = state.get('folder', self.folder)
        self.fnstem = state.get('fnstem', self.fnstem)
        self.padding = state.get('padding', self.padding)
        self.start_frame_nr = state.get('start_frame_nr', self.start_frame_nr)
        self.n_files_per_dist = state.get('n_files_per_dist', self.n_files_per_dist)
        self.ny = state.get('ny', self.ny)
        self.nz = state.get('nz', self.nz)

        self.folder_edit.setText(self.folder)
        self.stem_edit.setText(self.fnstem)
        self.nypx_edit.setText(str(self.ny))
        self.nzpx_edit.setText(str(self.nz))

        self.cmap_combo.setCurrentText(state.get('colormap', 'bone'))
        self.theme_combo.setCurrentText(state.get('theme', 'light'))
        self.log_check.setChecked(state.get('log', False))
        self.hflip_check.setChecked(state.get('hflip', False))
        self.vflip_check.setChecked(state.get('vflip', False))
        self.transpose_check.setChecked(state.get('transpose', False))
        self.median_check.setChecked(state.get('median', False))

        self.dist_spin.setValue(state.get('distance', 0))
        self.frame_spin.setValue(state.get('frame', 0))
        print(f'Session loaded: {fn}')

    # ── Callbacks ──────────────────────────────────────────────────

    def _on_log_toggled(self, checked):
        self.use_log = checked
        self.image_view.set_log_mode(checked)

    def _on_max_toggled(self, checked):
        if checked:
            self.sumframes_check.blockSignals(True)
            self.sumframes_check.setChecked(False)
            self.sumframes_check.blockSignals(False)
        self._load_and_display()

    def _on_sum_toggled(self, checked):
        if checked:
            self.maxframes_check.blockSignals(True)
            self.maxframes_check.setChecked(False)
            self.maxframes_check.blockSignals(False)
        self._load_and_display()

    # ── SelectPoint (click mic → grain) ─────────────────────────────

    def _on_select_point(self):
        """Enable click-on-mic to auto-populate grain parameters."""
        if self.mic_data is None:
            self._on_load_mic()
        if self.mic_data is None:
            return
        if self.bcs[0][0] == 0 and self.bcs[0][1] == 0:
            self._on_beam_center()
        if float(self.lsd_edit.text()) == 0:
            val, ok = QtWidgets.QInputDialog.getDouble(self, "Lsd", "Enter Lsd (μm):", 5000, 0, 1e9, 1)
            if ok:
                self.lsd_edit.setText(str(val))
        self.status_label.setText("Click on a grain in the mic map...")
        # Connect click handler on the mic view
        if self.mic_type == 1:
            self.mic_view.scene().sigMouseClicked.connect(self._on_mic_clicked)
        elif self.mic_type == 2:
            self.mic_image_view.scene.sigMouseClicked.connect(self._on_mic_clicked)

    def _on_mic_clicked(self, event):
        """Handle click on mic map: find nearest grain, populate grain dialog."""
        # Disconnect to prevent repeated triggers
        try:
            self.mic_view.scene().sigMouseClicked.disconnect(self._on_mic_clicked)
        except Exception:
            pass
        try:
            self.mic_image_view.scene.sigMouseClicked.disconnect(self._on_mic_clicked)
        except Exception:
            pass

        if self.mic_data_cut is None or len(self.mic_data_cut) == 0:
            return

        # Get click position in data coords
        if self.mic_type == 1:
            pos = self.mic_view.plotItem.vb.mapSceneToView(event.scenePos())
            click_x, click_y = pos.x(), pos.y()
        else:
            pos = self.mic_image_view.getView().mapSceneToView(event.scenePos())
            click_x, click_y = pos.x(), pos.y()

        # Find nearest grain
        xs = self.mic_data_cut[:, 3]
        ys = self.mic_data_cut[:, 4]
        dists = (xs - click_x)**2 + (ys - click_y)**2
        best_idx = np.argmin(dists)
        row = self.mic_data_cut[best_idx]

        # Auto-populate grain params
        euler = row[7:10]
        self.om = euler2orientmat(euler)
        self.pos[0] = row[3]
        self.pos[1] = row[4]
        self.pos[2] = 0
        self.status_label.setText(
            f"Selected grain at ({row[3]:.1f}, {row[4]:.1f}), "
            f"Euler=({euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}), "
            f"Conf={row[10]:.3f}")

        # Open grain dialog pre-populated
        dlg = GrainDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._make_spots()

    # ── SelectSpots / ComputeDistances ──────────────────────────────

    def _on_select_spots(self):
        """Start interactive spot selection for calibration."""
        self._sync_params()
        if not np.any(self.bcs):
            reply = QtWidgets.QMessageBox.question(
                self, "Warning",
                "All beam centers are 0. Enter beam centers first?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self._on_beam_center()
                return

        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Spot Selection Guide")
        msg.setText(
            "1. Ensure correct beam centers and distance difference.\n"
            "2. It is recommended to have median correction enabled.\n"
            "3. Starting from the last distance, click on a diffraction spot.\n"
            "4. Click 'Confirm Selection' in the status bar.\n"
            "5. Repeat for each distance using the SAME spot.\n"
            "6. Click 'Compute Distances' when finished.")
        msg.addButton("Ready!", QtWidgets.QMessageBox.AcceptRole)
        msg.exec_()

        self._selecting_spots = True
        self._spot_confirm_btn = QtWidgets.QPushButton("Confirm Selection")
        self._spot_confirm_btn.clicked.connect(self._confirm_select_spot)
        self.statusBar().addWidget(self._spot_confirm_btn)
        self.status_label.setText("Click on a diffraction spot...")
        self.image_view.scene.sigMouseClicked.connect(self._on_spot_clicked)

    def _on_spot_clicked(self, event):
        """Record clicked position for spot selection."""
        if not self._selecting_spots:
            return
        pos = self.image_view.getView().mapSceneToView(event.scenePos())
        self._click_ix = pos.x()
        self._click_iy = pos.y()

        # Show crosshair marker at clicked position
        if hasattr(self, '_spot_marker') and self._spot_marker is not None:
            self.image_view.removeItem(self._spot_marker)
        marker = pg.ScatterPlotItem(
            [self._click_ix], [self._click_iy],
            symbol='+', size=20, pen=pg.mkPen('c', width=2), brush=None)
        marker.setZValue(1000)
        self.image_view.addItem(marker)
        self._spot_marker = marker

        self.status_label.setText(
            f"Spot at ({self._click_ix:.1f}, {self._click_iy:.1f}) — click 'Confirm Selection'")

    def _confirm_select_spot(self):
        """Confirm the selected spot and move to next distance."""
        self._sync_params()
        dist = self.dist
        xbc = self.bcs[dist][0]
        ybc = self.bcs[dist][1]
        self.spots[dist][0] = self._click_ix - xbc
        self.spots[dist][1] = self._click_iy - ybc
        self.spots[dist][2] = math.sqrt(self.spots[dist][0]**2 + self.spots[dist][1]**2)
        print(f"Spot confirmed for distance {dist}: "
              f"rel=({self.spots[dist][0]:.1f}, {self.spots[dist][1]:.1f}), "
              f"R={self.spots[dist][2]:.1f}")

        # Save crop around clicked spot for visualization
        if self.imarr2 is not None:
            half = 50
            cx, cy = int(round(self._click_ix)), int(round(self._click_iy))
            h, w = self.imarr2.shape
            y0, y1 = max(0, cy - half), min(h, cy + half)
            x0, x1 = max(0, cx - half), min(w, cx + half)
            if y1 > y0 and x1 > x0:
                self._spot_crops[dist] = self.imarr2[y0:y1, x0:x1].copy()

        # Ask: next distance or finished?
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Distance")
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addWidget(QtWidgets.QLabel(f"Previous distance was: {dist}. Which distance now?"))
        dist_edit = QtWidgets.QSpinBox()
        dist_edit.setRange(0, 20)
        dist_edit.setValue(max(0, dist - 1))
        layout.addWidget(dist_edit)
        btn_row = QtWidgets.QHBoxLayout()
        btn_load = QtWidgets.QPushButton("Load")
        btn_finish = QtWidgets.QPushButton("Finished")
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_finish)
        layout.addLayout(btn_row)

        def on_load():
            self.dist_spin.setValue(dist_edit.value())
            dlg.accept()

        def on_finish():
            self._selecting_spots = False
            try:
                self.image_view.scene.sigMouseClicked.disconnect(self._on_spot_clicked)
            except Exception:
                pass
            # Clean up confirm button and spot marker
            self.statusBar().removeWidget(self._spot_confirm_btn)
            self._spot_confirm_btn.deleteLater()
            self._spot_confirm_btn = None
            if hasattr(self, '_spot_marker') and self._spot_marker is not None:
                self.image_view.removeItem(self._spot_marker)
                self._spot_marker = None
            dlg.accept()
            # Auto-compute distances
            self._compute_distances()

        btn_load.clicked.connect(on_load)
        btn_finish.clicked.connect(on_finish)
        dlg.exec_()

    def _compute_distances(self, btn=None):
        """Compute sample-to-detector distances via ray triangulation."""
        self._sync_params()
        n = self.n_distances
        nsols = int(n * (n - 1) / 2)
        xs = np.zeros(nsols)
        ys = np.zeros(nsols)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                z1 = self.spots[i][1]
                z2 = self.spots[j][1]
                y1 = self.spots[i][0]
                y2 = self.spots[j][0]
                x = self.dist_diff * (j - i)
                if z2 != z1:
                    xs[idx] = x * z1 / (z2 - z1) - (self.dist_diff * i)
                    ys[idx] = y1 + (y2 - y1) * z1 / (z1 - z2)
                idx += 1

        self.lsd = float(np.mean(xs[:idx])) if idx > 0 else 0
        self.lsd_edit.setText(f"{self.lsd:.1f}")
        result_text = (f"Calculated distances: {xs[:idx]}\n"
                       f"Calculated Ys: {ys[:idx]}\n"
                       f"Mean Lsd: {self.lsd:.1f} μm")
        print(result_text)

        if btn is not None:
            self.statusBar().removeWidget(btn)
            btn.deleteLater()
        self.status_label.setText(f"Lsd = {self.lsd:.1f} μm")

        # ── Visual results dialog ──
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Distance Results — Lsd = {self.lsd:.1f} μm")
        dlg.resize(1000, 650)
        main_lay = QtWidgets.QVBoxLayout(dlg)

        # Summary text
        summary = QtWidgets.QLabel(result_text)
        summary.setWordWrap(True)
        main_lay.addWidget(summary)

        gw = pg.GraphicsLayoutWidget()
        main_lay.addWidget(gw, stretch=1)

        # Row 1: Spot crops for each distance
        crops = getattr(self, '_spot_crops', {})
        sorted_dists = sorted(crops.keys())
        n_crops = len(sorted_dists)

        if n_crops > 0:
            for d in sorted_dists:
                p = gw.addPlot(title=f"Dist {d}")
                img = pg.ImageItem(crops[d].T.astype(float))
                p.addItem(img)
                p.setAspectLocked(True)
                # Crosshair at center of crop
                ch, cw = crops[d].shape
                p.plot([cw / 2], [ch / 2], symbol='+', symbolSize=14,
                       symbolPen=pg.mkPen('c', width=2), pen=None)
                p.hideAxis('left')
                p.hideAxis('bottom')
            gw.nextRow()

        # Row 2: Triangulation ray diagram
        colspan = max(n_crops, 1)
        tri = gw.addPlot(title="Ray Triangulation", colspan=colspan)
        tri.setLabel('bottom', 'Distance from sample (μm)')
        tri.setLabel('left', 'Spot position (px from BC)')
        tri.addLegend(offset=(10, 10))

        # Draw rays: sample at origin, detectors at Lsd + d*dist_diff
        for d in range(n):
            spot_z = self.spots[d][1]
            if abs(spot_z) < 1e-6 and abs(self.spots[d][0]) < 1e-6:
                continue  # skip unset distances
            det_pos = self.lsd + d * self.dist_diff
            color = pg.intColor(d, n, maxValue=200)
            # Ray from sample (0, 0) through (det_pos, spot_z)
            tri.plot([0, det_pos], [0, spot_z],
                     pen=pg.mkPen(color, width=2), name=f"Dist {d}")
            tri.plot([det_pos], [spot_z],
                     symbol='o', symbolBrush=color, symbolSize=10, pen=None)

        # Mark sample position
        tri.plot([0], [0], symbol='x', symbolBrush='r', symbolSize=15,
                 symbolPen=pg.mkPen('r', width=2), pen=None, name='Sample')
        tri.addLine(x=0, pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.DashLine))

        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        main_lay.addWidget(close_btn)

        dlg.exec_()

    def _movie_advance_frame(self):
        """Advance frame by 1 for movie mode; wrap at max."""
        cur = self.frame_spin.value()
        mx = self.frame_spin.maximum()
        nxt = cur + 1 if cur < mx else 0
        self.frame_spin.setValue(nxt)

    def _on_file_dropped(self, path):
        """Handle file dropped onto the viewer."""
        if os.path.isdir(path):
            self.folder = path.rstrip('/') + '/'
            self.folder_edit.setText(path)
            self._load_and_display()
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext == '.tif' or ext == '.tiff':
                self.folder = os.path.dirname(path) + '/'
                self.folder_edit.setText(self.folder)
                self._load_and_display()
            elif ext in ('.mic', '.map'):
                self._load_mic_file(path)

    def _on_cursor_moved(self, x, y, val):
        self.status_label.setText(f"x={x:.1f}  y={y:.1f}  I={val:.0f}")

    def _on_stats_updated(self, dmin, dmax, p2, p98):
        self.stats_label.setText(f"Min={dmin:.0f}  Max={dmax:.0f}  [P2={p2:.0f}  P98={p98:.0f}]")
        # Only auto-populate MinI/MaxI on the very first image load
        if not getattr(self, '_levels_initialized', False):
            self._levels_initialized = True
            self.min_intensity_edit.setText(str(int(p2)))
            self.max_intensity_edit.setText(str(int(p98)))

    def _apply_intensity_levels(self):
        try:
            lo = float(self.min_intensity_edit.text())
            hi = float(self.max_intensity_edit.text())
            self.image_view.setLevels(lo, hi)
        except ValueError:
            pass

    def _on_col_mode_changed(self, btn):
        self.col_mode = self.col_group.id(btn)
        if self.mic_data is not None:
            self._plot_mic()

    def _sync_params(self):
        try:
            self.ny = int(self.ny_edit.text())
            self.nz = int(self.nz_edit.text())
            self.folder = self.folder_edit.text()
            self.fnstem = self.fnstem_edit.text()
            self.frame_nr = self.frame_spin.value()
            self.dist = self.dist_spin.value()
            self.n_distances = int(self.ndist_edit.text())
            self.n_files_per_dist = int(self.nfiles_edit.text())
            self.start_frame_nr = int(self.startframe_edit.text())
            self.pixel_size = float(self.px_edit.text())
            self.lsd = float(self.lsd_edit.text())
            self.use_median = self.median_check.isChecked()
            self.max_over_frames = self.maxframes_check.isChecked()
            self.sum_over_frames = self.sumframes_check.isChecked()
            self.cut_confidence = float(self.cut_conf_edit.text())
            self.max_conf = float(self.max_conf_edit.text())
        except ValueError:
            pass

    # ── File callbacks ─────────────────────────────────────────────

    def _on_first_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select First File")
        if not fn:
            return
        # Parse folder, stem, start nr
        self.frame_nr = 0
        self.frame_spin.setValue(0)
        idx = fn.rfind('/')
        self.folder = fn[:idx]
        self.folder_edit.setText(self.folder)
        fntot = fn[idx + 1:]
        parts = fntot.split('_')
        num_ext = parts[-1]
        num_part = num_ext.split('.')[0]
        self.padding = len(num_part)
        ext_len = len(num_ext.split('.')[1]) if '.' in num_ext else 0
        self.fnstem = fn[idx + 1:][:-(2 + self.padding + ext_len)]
        self.fnstem_edit.setText(self.fnstem)
        self.start_frame_nr = int(num_part)
        self.startframe_edit.setText(str(self.start_frame_nr))
        print(f"Loaded: stem={self.fnstem}, folder={self.folder}")
        self._load_and_display()

    def _on_browse_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            self.folder = d
            self.folder_edit.setText(d)

    # ── Load & Display ─────────────────────────────────────────────

    def _get_filenames(self):
        if self._beampos_mode:
            idx = self.frame_nr
            idx = max(0, min(idx, len(self._beampos_files) - 1))
            return self._beampos_files[idx], None

        median_fn = None
        if self._median_dir and os.path.isdir(self._median_dir):
            median_fn = os.path.join(self._median_dir,
                self.fnstem + "_Median_Background_Distance_" + str(self.dist) + ".bin")
            if not os.path.exists(median_fn):
                median_fn = None
        if median_fn is None:
            mf = os.path.join(self.folder, self.fnstem +
                "_Median_Background_Distance_" + str(self.dist) + ".bin")
            if os.path.exists(mf):
                median_fn = mf

        fnr = self.start_frame_nr + self.frame_nr + self.dist * self.n_files_per_dist
        file_fn = os.path.join(self.folder, self.fnstem + '_' +
                               str(fnr).zfill(self.padding) + '.tif')
        return file_fn, median_fn

    def _load_and_display(self):
        self._sync_params()

        if (self.max_over_frames or self.sum_over_frames) and not self._beampos_mode:
            if self.max_over_frames:
                tag = 'MaximumIntensity'
            else:
                tag = 'SumIntensity'
            if self.use_median:
                suffix = f'_{tag}MedianCorrected_Distance_' + str(self.dist) + '.bin'
            else:
                suffix = f'_{tag}_Distance_' + str(self.dist) + '.bin'
            # Check temp median dir first, then data folder
            fn = None
            if self._median_dir and os.path.isdir(self._median_dir):
                candidate = os.path.join(self._median_dir, self.fnstem + suffix)
                if os.path.exists(candidate):
                    fn = candidate
            if fn is None:
                fn = os.path.join(self.folder, self.fnstem + suffix)
            if os.path.exists(fn):
                with open(fn, 'rb') as f:
                    imarr = np.fromfile(f, dtype=np.uint16, count=self.ny * self.nz)
                self.imarr2 = imarr.reshape((self.nz, self.ny))
                self.imarr2[self.imarr2 < self.background] = 0
                self._last_loaded_fn = fn
            else:
                print(f"File not found: {fn}")
                return
        else:
            fns = self._get_filenames()
            if not os.path.exists(fns[0]):
                print(f"File not found: {fns[0]}")
                return
            imarr = read_tiff(fns[0])
            if imarr is None:
                return
            self._last_loaded_fn = fns[0]
            print(f"Read file {fns[0]}")

            if self.use_median and fns[1] is not None and os.path.exists(fns[1]):
                print(f"Subtracting median: {fns[1]}")
                with open(fns[1], 'rb') as f:
                    median = np.fromfile(f, dtype=np.uint16, count=self.ny * self.nz)
                median = median.reshape((self.nz, self.ny))
                self.imarr2 = np.subtract(imarr.astype(int), median.astype(int))
                self.imarr2[self.imarr2 < self.background] = 0
            else:
                if self.use_median:
                    print(f"SubtMedian ON but median file not found. _median_dir={self._median_dir}, median_fn={fns[1]}")
                self.imarr2 = imarr

        self.imarr2 = self.imarr2[::-1, ::-1].copy()
        # On first load, auto-levels; afterwards use user's MinI/MaxI
        if getattr(self, '_levels_initialized', False):
            try:
                lo = float(self.min_intensity_edit.text())
                hi = float(self.max_intensity_edit.text())
                self.image_view.set_image_data(self.imarr2.astype(float), auto_levels=False, levels=(lo, hi))
            except ValueError:
                self.image_view.set_image_data(self.imarr2.astype(float))
        else:
            self.image_view.set_image_data(self.imarr2.astype(float))
        # Show which file is loaded
        if hasattr(self, '_last_loaded_fn'):
            basename = os.path.basename(self._last_loaded_fn)
        else:
            basename = ''
        self.frame_label.setText(f"Frame {self.frame_nr}  Dist {self.dist}  |  {basename}")
        self.setWindowTitle(f"NF Viewer — {basename} [frame {self.frame_nr}, dist {self.dist}]")

        # Refresh box profile if an ROI is active
        if hasattr(self, '_box_roi') and self._box_roi is not None:
            self._update_box_profile()

    # ── Line Profile ───────────────────────────────────────────────



    # ── Box ROI (sum across rect) ──────────────────────────────────

    def _add_box_roi(self, direction):
        """Enter draw mode: user drags to define box ROI for line profile."""
        if self.imarr2 is None:
            return
        self.right_widget.setCurrentIndex(0)
        self._box_direction = direction
        self._box_drawing = True
        self._box_draw_start = None

        # Remove previous ROI and preview
        if hasattr(self, '_box_roi') and self._box_roi is not None:
            self.image_view.removeItem(self._box_roi)
            self._box_roi = None
        if hasattr(self, '_box_preview') and self._box_preview is not None:
            self.image_view.removeItem(self._box_preview)
            self._box_preview = None

        label = 'BoxH' if direction == 'h' else 'BoxV'
        self.status_label.setText(f"Draw {label}: click one corner, then click the opposite corner to drag-select the region of interest")
        self.setCursor(QtCore.Qt.CrossCursor)

        # Connect mouse events on the scene
        scene = self.image_view.scene
        scene.sigMouseClicked.connect(self._box_draw_click)

    def _box_draw_click(self, ev):
        """Handle mouse click for box draw mode."""
        if not getattr(self, '_box_drawing', False):
            return

        vb = self.image_view.getView().getViewBox()
        pos = vb.mapSceneToView(ev.scenePos())

        if self._box_draw_start is None:
            # First click: record start point, install move handler
            self._box_draw_start = (pos.x(), pos.y())
            self.status_label.setText("Now click the opposite corner...")
            # Add preview rectangle
            from PyQt5 import QtGui as _QtGui
            preview = QtWidgets.QGraphicsRectItem(pos.x(), pos.y(), 0, 0)
            preview.setPen(pg.mkPen('c', width=2, style=QtCore.Qt.DashLine))
            preview.setBrush(_QtGui.QBrush(_QtGui.QColor(0, 255, 255, 30)))
            preview.setZValue(1000)
            self.image_view.addItem(preview)
            self._box_preview = preview
            # Track mouse for preview
            self._box_move_proxy = pg.SignalProxy(
                self.image_view.scene.sigMouseMoved, rateLimit=30,
                slot=self._box_draw_move)
            ev.accept()
        else:
            # Second click: finalize ROI
            x0, y0 = self._box_draw_start
            x1, y1 = pos.x(), pos.y()
            self._box_finish_draw(x0, y0, x1, y1)
            ev.accept()

    def _box_draw_move(self, evt):
        """Update preview rectangle as mouse moves."""
        if not getattr(self, '_box_drawing', False) or self._box_draw_start is None:
            return
        pos = evt[0]
        vb = self.image_view.getView().getViewBox()
        if self.image_view.getView().sceneBoundingRect().contains(pos):
            mp = vb.mapSceneToView(pos)
            x0, y0 = self._box_draw_start
            x1, y1 = mp.x(), mp.y()
            rx, ry = min(x0, x1), min(y0, y1)
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            if hasattr(self, '_box_preview') and self._box_preview is not None:
                self._box_preview.setRect(rx, ry, rw, rh)

    def _box_finish_draw(self, x0, y0, x1, y1):
        """Create final RectROI from the two corner points."""
        # Cleanup draw mode
        self._box_drawing = False
        self._box_draw_start = None
        self.setCursor(QtCore.Qt.ArrowCursor)

        # Disconnect handlers
        try:
            self.image_view.scene.sigMouseClicked.disconnect(self._box_draw_click)
        except Exception:
            pass
        if hasattr(self, '_box_move_proxy'):
            self._box_move_proxy.disconnect()
            self._box_move_proxy = None

        # Remove preview
        if hasattr(self, '_box_preview') and self._box_preview is not None:
            self.image_view.removeItem(self._box_preview)
            self._box_preview = None

        # Create ROI at drawn bounds
        rx, ry = min(x0, x1), min(y0, y1)
        rw, rh = abs(x1 - x0), abs(y1 - y0)
        if rw < 2 or rh < 2:  # too small, ignore
            self.status_label.setText("Box too small, try again")
            return

        roi = pg.RectROI([rx, ry], [rw, rh], pen=pg.mkPen('c', width=2))
        self._box_roi = roi
        self.image_view.addItem(roi)
        roi.sigRegionChanged.connect(self._update_box_profile)
        self._update_box_profile()
        label = 'BoxH' if self._box_direction == 'h' else 'BoxV'
        self.status_label.setText(f"{label} ROI created — drag handles to adjust")

    def _update_box_profile(self):
        if not hasattr(self, '_box_roi') or self._box_roi is None or self.imarr2 is None:
            return
        # imageItem displays data.T (cols, rows), so pass transposed data
        # to match the imageItem's coordinate system
        data = self._box_roi.getArrayRegion(
            self.imarr2.T.astype(float), self.image_view.imageItem)
        if data is None or data.size == 0:
            return
        direction = getattr(self, '_box_direction', 'h')
        if direction == 'h':
            profile = np.mean(data, axis=1)
            title = 'BoxH'
        else:
            profile = np.mean(data, axis=0)
            title = 'BoxV'

        # Compute pixel origin so x-axis is in image pixel coordinates
        roi_pos = self._box_roi.pos()
        origin = roi_pos.x() if direction == 'h' else roi_pos.y()

        self.lineout_plot.clear()
        x = np.arange(len(profile)) + origin
        self.lineout_plot.plot(x, profile, pen=pg.mkPen('c', width=2.5))

        # ── Half-max edge detection ──
        pmin, pmax = float(np.min(profile)), float(np.max(profile))
        threshold = pmin + 0.5 * (pmax - pmin)
        above = profile >= threshold
        # Find crossings: transitions from below→above or above→below
        crossings = np.where(np.diff(above.astype(int)) != 0)[0]

        # Orange-red edge color — visible in both light and dark themes
        edge_pen = pg.mkPen(color=(255, 100, 0), width=2, style=QtCore.Qt.DashLine)
        threshold_pen = pg.mkPen(color=(255, 100, 0, 80), width=1, style=QtCore.Qt.DotLine)

        edge_info = ''
        if len(crossings) >= 2:
            # Interpolate for sub-pixel crossing positions
            edges = []
            for ci in crossings:
                dv = profile[ci + 1] - profile[ci]
                if abs(dv) > 1e-12:
                    frac = (threshold - profile[ci]) / dv
                else:
                    frac = 0.5
                edges.append(ci + frac + origin)
            # Use first and last crossing as the two edges
            e_left, e_right = edges[0], edges[-1]
            center = 0.5 * (e_left + e_right)
            width = abs(e_right - e_left)

            # Draw edges (orange-red dashed) and center (red solid)
            self.lineout_plot.addLine(x=e_left, pen=edge_pen)
            self.lineout_plot.addLine(x=e_right, pen=edge_pen)
            self.lineout_plot.addLine(x=center, pen=pg.mkPen('r', width=2.5))
            # Draw threshold line
            self.lineout_plot.plot([x[0], x[-1]], [threshold, threshold], pen=threshold_pen)
            edge_info = f'  Center={center:.1f}  Width={width:.1f}  Edges=[{e_left:.1f}, {e_right:.1f}]'
        elif len(crossings) == 1:
            # Single crossing — just mark it
            ci = crossings[0]
            dv = profile[ci + 1] - profile[ci]
            frac = (threshold - profile[ci]) / dv if abs(dv) > 1e-12 else 0.5
            edge_pos = ci + frac + origin
            self.lineout_plot.addLine(x=edge_pos, pen=edge_pen)
            edge_info = f'  Edge={edge_pos:.1f}'

        self.lineout_plot.setTitle(f"{title}  Mean={np.mean(data):.1f}  "
                                   f"Min={pmin:.0f}  Max={pmax:.0f}{edge_info}")

    # ── Mic File ───────────────────────────────────────────────────

    def _on_load_mic(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Mic File", os.getcwd(),
            "All Files (*);;Map Files (*.map)")
        if not fn:
            return
        self.mic_file = fn
        print(f"Loading mic: {fn}")

        if fn.endswith('.map'):
            self.mic_type = 2
            with open(fn, 'rb') as f:
                self.mic_size_x = int(np.fromfile(f, dtype=np.double, count=1)[0])
                self.mic_size_y = int(np.fromfile(f, dtype=np.double, count=1)[0])
                self.mic_ref_x = int(np.fromfile(f, dtype=np.double, count=1)[0])
                self.mic_ref_y = int(np.fromfile(f, dtype=np.double, count=1)[0])
                self.mic_data = np.fromfile(f, dtype=np.double)
            print(f"Map: {self.mic_size_x}x{self.mic_size_y}, {self.mic_data.size} values")
        else:
            self.mic_type = 1
            with open(fn, 'r') as f:
                self.mic_data = np.genfromtxt(f, skip_header=4)

        self._plot_mic()

    def _on_load_h5(self):
        """Load a consolidated NF-HEDM HDF5 file."""
        if not HAS_H5PY:
            QtWidgets.QMessageBox.warning(self, "Missing Dependency",
                                          "h5py is required to load consolidated H5 files.")
            return

        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Consolidated H5", os.getcwd(),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)")
        if not fn:
            return

        self._h5_path = fn
        print(f"Loading consolidated H5: {fn}")

        try:
            with h5py.File(fn, 'r') as h5:
                # Discover available resolutions
                resolutions = ["root"]
                has_maps = {"root": 'maps' in h5 and 'orientation' in h5.get('maps', {})}
                if 'multi_resolution' in h5:
                    for key in sorted(h5['multi_resolution'].keys()):
                        grp = h5[f'multi_resolution/{key}']
                        grid = grp.attrs.get('grid_size', '?')
                        ptype = grp.attrs.get('pass_type', '')
                        map_grp = f'multi_resolution/{key}/maps'
                        has_map = map_grp in h5 and 'orientation' in h5[map_grp]
                        warn = "" if has_map else " ⚠ slow"
                        label = f"{key} (grid={grid}, {ptype}){warn}"
                        resolutions.append(label)
                        has_maps[label] = has_map

                self._h5_resolutions = resolutions

                # Populate combo box
                self._res_combo.blockSignals(True)
                self._res_combo.clear()
                for r in resolutions:
                    self._res_combo.addItem(r)
                self._res_combo.blockSignals(False)

                # Show selector if multi-resolution
                has_multi = len(resolutions) > 1
                self._res_selector_label.setVisible(has_multi)
                self._res_combo.setVisible(has_multi)

                # Default to highest-resolution seeded loop with maps,
                # falling back to highest seeded, then root
                default_res = "root"
                for r in reversed(resolutions):
                    if '_seeded' in r and has_maps.get(r, False):
                        default_res = r
                        break
                else:
                    # No seeded with maps — try highest seeded anyway
                    for r in reversed(resolutions):
                        if '_seeded' in r:
                            default_res = r
                            break

                idx = resolutions.index(default_res)
                self._res_combo.setCurrentIndex(idx)
                self._load_h5_resolution(h5, default_res)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load H5: {e}")
            print(f"H5 load error: {e}")

    def _on_resolution_changed(self, text):
        """Handle resolution combo box change."""
        if not self._h5_path or not text:
            return
        try:
            with h5py.File(self._h5_path, 'r') as h5:
                self._load_h5_resolution(h5, text)
        except Exception as e:
            print(f"Resolution switch error: {e}")

    def _load_h5_resolution(self, h5, resolution_text):
        """Load voxel data from a specific resolution in the H5.

        Args:
            h5:              Open h5py.File object.
            resolution_text: "root" or "loop_N_type (grid=X, type)".
        """
        if resolution_text == "root":
            prefix = "voxels"
        else:
            # Extract key from display text: "loop_0_unseeded (grid=5.0, unseeded)"
            key = resolution_text.split(" (")[0]
            prefix = f"multi_resolution/{key}/voxels"

        if prefix not in h5:
            print(f"Resolution not found in H5: {prefix}")
            return

        # Read voxel data and assemble into mic_data format
        grp = h5[prefix]
        n = grp['position'].shape[0]

        # Build a mic_data array matching text mic format (15 cols)
        mic = np.zeros((n, 15))

        mic[:, 3:5] = grp['position'][()]
        mic[:, 7:10] = grp['euler_angles'][()]
        mic[:, 10] = grp['confidence'][()]

        if 'tri_edge_size' in grp:
            mic[:, 0] = grp['tri_edge_size'][()]
        if 'up_down' in grp:
            mic[:, 1] = grp['up_down'][()]
        if 'orientation_row_nr' in grp:
            mic[:, 2] = grp['orientation_row_nr'][()]
        if 'orientation_id' in grp:
            mic[:, 6] = grp['orientation_id'][()]
        if 'phase_nr' in grp:
            mic[:, 11] = grp['phase_nr'][()]
        if 'run_time' in grp:
            mic[:, 12] = grp['run_time'][()]

        self.mic_type = 1  # Treat as text mic for scatter plot
        self.mic_data = mic
        self.mic_file = self._h5_path
        self._h5_current_resolution = resolution_text

        # Check for binary map in H5
        map_prefix = prefix.replace('/voxels', '/maps')
        if map_prefix in h5 and 'orientation' in h5[map_prefix]:
            orient = h5[f"{map_prefix}/orientation"][()]
            h, w, _ = orient.shape
            self.mic_size_x = w
            self.mic_size_y = h
            extent = h5[f"{map_prefix}/extent"][()]
            self.mic_ref_x = int(extent[0])
            self.mic_ref_y = int(extent[2])

            # Build flat map data matching binary .map layout
            n_pix = w * h
            flat = np.zeros(n_pix * 7)
            for i in range(7):
                flat[i * n_pix:(i + 1) * n_pix] = orient[:, :, i].ravel()

            # Also load KAM, GROD, GrainID if present
            self._h5_extra_maps = {}
            for name, col in [('kam', 12), ('grod', 13), ('grain_id', 14)]:
                if f"{map_prefix}/{name}" in h5:
                    self._h5_extra_maps[col] = h5[f"{map_prefix}/{name}"][()]

            self.mic_type = 2  # Use image display
            self.mic_data = flat

        print(f"Loaded resolution: {resolution_text} ({n} voxels)")
        self._plot_mic()

    def _plot_mic(self):
        if self.mic_data is None:
            return
        self._sync_params()
        col = self.col_mode

        col_labels = {10: 'Confidence', 0: 'GrainID', 11: 'PhaseNr',
                      7: 'Euler0', 8: 'Euler1', 9: 'Euler2',
                      12: 'KAM', 13: 'GROD', 14: 'GrainMap'}

        if self.mic_type == 1:
            # Text mic file: scatter plot
            self.right_widget.setCurrentIndex(1)
            data = np.copy(self.mic_data)
            data = data[data[:, 10] > self.cut_confidence, :]
            self.mic_data_cut = data

            xs = data[:, 3]
            ys = data[:, 4]
            colors = data[:, col] if col < data.shape[1] else data[:, 10]

            # Normalize colors to jet colormap
            cmin = colors.min()
            cmax = colors.max()
            if col == 10:
                cmax = self.max_conf
            if cmax == cmin:
                cmax = cmin + 1
            norm = (colors - cmin) / (cmax - cmin)
            norm = np.clip(norm, 0, 1)

            cmap = pg.colormap.get('turbo')
            lut = cmap.getLookupTable(nPts=256)
            brush_indices = (norm * 255).astype(int)
            brushes = [pg.mkBrush(*lut[i]) for i in brush_indices]

            # Save current view range before updating data
            vb = self.mic_view.plotItem.vb
            had_data = self.mic_scatter.data is not None and len(self.mic_scatter.data) > 0
            saved_range = vb.viewRange() if had_data else None

            vb.enableAutoRange(enable=False)
            self.mic_scatter.setData(xs, ys, brush=brushes)
            self.mic_view.setTitle(f"Mic ({col_labels.get(col, 'Col ' + str(col))})")

            if saved_range is not None:
                vb.setRange(xRange=saved_range[0], yRange=saved_range[1], padding=0)

        elif self.mic_type == 2:
            # Binary map: image display
            self.right_widget.setCurrentIndex(2)
            sx, sy = self.mic_size_x, self.mic_size_y
            n = sx * sy
            bad = self.mic_data[:n] < self.cut_confidence

            if col == 10:  # Confidence
                arr = self.mic_data[:n].copy()
            elif col == 7:  # Euler0
                arr = self.mic_data[n:n * 2].copy()
            elif col == 8:  # Euler1
                arr = self.mic_data[n * 2:n * 3].copy()
            elif col == 9:  # Euler2
                arr = self.mic_data[n * 3:n * 4].copy()
            elif col == 0:  # OrientationID
                arr = self.mic_data[n * 4:n * 5].copy()
            elif col == 11:  # PhaseNr
                arr = self.mic_data[n * 5:n * 6].copy()
            elif col in [12, 13, 14]:
                # KAM/GROD/GrainMap from external files or H5
                if hasattr(self, '_h5_extra_maps') and col in self._h5_extra_maps:
                    arr = self._h5_extra_maps[col].ravel().copy()
                else:
                    ext_map = {12: '.kam', 13: '.grod', 14: '.grainId'}
                    ext_file = self.mic_file + ext_map[col]
                    if os.path.exists(ext_file):
                        with open(ext_file, 'rb') as f:
                            arr = np.fromfile(f, dtype=np.double)[4:]
                    else:
                        print(f"File not found: {ext_file}")
                        return
            else:
                arr = self.mic_data[:n].copy()

            arr[bad] = np.nan
            arr = arr.reshape((sy, sx))
            # Save current view range before updating image
            mic_vb = self.mic_image_view.getView()
            had_image = self.mic_image_view._raw_data is not None
            saved_range = mic_vb.viewRange() if had_image else None

            self.mic_image_view.set_image_data(arr)
            self.mic_image_view.set_colormap('turbo')

            if saved_range is not None:
                mic_vb.setRange(xRange=saved_range[0], yRange=saved_range[1], padding=0)

    # ── Beam Center ────────────────────────────────────────────────

    def _on_beam_center(self):
        dlg = BeamCenterDialog(self)
        dlg.exec_()

    # ── Median ─────────────────────────────────────────────────────

    def _on_calc_median(self):
        self._sync_params()
        import shutil
        if self._median_dir and os.path.isdir(self._median_dir):
            shutil.rmtree(self._median_dir, ignore_errors=True)
        self._median_dir = tempfile.mkdtemp(prefix='midas_median_')

        # Ensure subdirectory exists if fnstem contains path separators
        # (e.g. "subdir/stem" → MedianImageLibTiff writes to OutputDir/subdir/)
        fnstem_dir = os.path.dirname(self.fnstem)
        if fnstem_dir:
            os.makedirs(os.path.join(self._median_dir, fnstem_dir), exist_ok=True)

        if midas_config and midas_config.MIDAS_NF_BIN_DIR:
            cmd = os.path.join(midas_config.MIDAS_NF_BIN_DIR, 'MedianImageLibTiff')
        else:
            cmd = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/MedianImageLibTiff')

        total_cpus = os.cpu_count() or 4
        cpus_per = max(1, total_cpus // self.n_distances)

        def run_median(d):
            pf = os.path.join(self._median_dir, f'ps_dist{d}.txt')
            with open(pf, 'w') as f:
                f.write('extReduced bin\n')
                f.write('extOrig tif\n')
                f.write('WFImages 0\n')
                f.write('OrigFileName ' + self.fnstem + '\n')
                f.write('NrFilesPerDistance ' + str(self.n_files_per_dist) + '\n')
                f.write('NrPixels ' + str(self.ny) + '\n')
                f.write('DataDirectory ' + self.folder + '\n')
                f.write('OutputDirectory ' + self._median_dir + '\n')
                tempnr = self.start_frame_nr
                f.write('RawStartNr ' + str(tempnr) + '\n')
                f.write('ReducedFileName ' + self.fnstem + '\n')
            try:
                subprocess.run([cmd, pf, str(d + 1), str(cpus_per)], check=True)
            except Exception as e:
                print(f"Median error dist {d}: {e}")

        worker = AsyncWorker(
            target=lambda: list(concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_distances).map(run_median, range(self.n_distances))))
        def _on_median_done(_):
            print(f"Median computed for {self.n_distances} distances")
            self.median_check.setChecked(True)
            self._load_and_display()
        worker.finished_signal.connect(_on_median_done)
        worker.start()
        self._median_worker = worker

    # ── Grain Simulation (placeholders) ────────────────────────────

    def _on_load_grain(self):
        """Open grain parameter dialog."""
        dlg = GrainDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._make_spots()

    def _on_make_spots(self):
        """Generate and display simulated spots."""
        if not self.simulated_spots:
            self._on_load_grain()
        elif self.simulated_spots:
            self._plot_spot(self.spot_nr)

    def _make_spots(self):
        """Run GetHKLList + GenSeedOrientations + SimulateDiffractionSpots."""
        with tempfile.TemporaryDirectory() as tmp:
            pf = os.path.join(tmp, 'ps.txt')
            with open(pf, 'w') as f:
                f.write('SpaceGroup ' + str(self.sg) + '\n')
                f.write('Wavelength ' + str(self.wl) + '\n')
                lsd_val = float(self.lsd_edit.text())
                f.write('Lsd ' + str(lsd_val) + '\n')
                f.write('MaxRingRad ' + str(self.maxringrad) + '\n')
                f.write('LatticeConstant ' + ' '.join(str(v) for v in self.latC) + '\n')

            if midas_config and midas_config.MIDAS_NF_BIN_DIR:
                hkl_bin = os.path.join(midas_config.MIDAS_NF_BIN_DIR, 'GetHKLList')
                gen_bin = os.path.join(midas_config.MIDAS_NF_BIN_DIR, 'GenSeedOrientationsFF2NFHEDM')
                sim_bin = os.path.join(midas_config.MIDAS_NF_BIN_DIR, 'SimulateDiffractionSpots')
            else:
                hkl_bin = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/GetHKLList')
                gen_bin = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/GenSeedOrientationsFF2NFHEDM')
                sim_bin = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/SimulateDiffractionSpots')

            try:
                subprocess.run([hkl_bin, pf], check=True, cwd=tmp)
            except Exception as e:
                print(f"GetHKLList failed: {e}"); return

            orinfn = os.path.join(tmp, 'orin.txt')
            oroutfn = os.path.join(tmp, 'orout.txt')
            instr = "120 " + " ".join(str(self.om[i]) for i in range(9))
            instr += " " + " ".join(str(self.pos[i]) for i in range(3))
            instr += " " + " ".join(str(self.latC[i]) for i in range(6)) + "\n"
            with open(orinfn, 'w') as f:
                f.write(instr)

            try:
                subprocess.run([gen_bin, orinfn, oroutfn], check=True, cwd=tmp)
                subprocess.run([sim_bin, str(lsd_val), oroutfn], check=True, cwd=tmp)
            except Exception as e:
                print(f"Spot simulation failed: {e}"); return

            spots_fn = os.path.join(tmp, 'SimulatedDiffractionSpots.txt')
            if os.path.exists(spots_fn):
                with open(spots_fn, 'r') as f:
                    self.simulated_spots = f.readlines()
                print(f"Generated {len(self.simulated_spots)} simulated spots")
                self.spot_nr = 1
                self._plot_spot(1)
            else:
                print("SimulatedDiffractionSpots.txt not found")

    def _plot_spot(self, spot_nr):
        """Display a single simulated spot on the image."""
        if not self.simulated_spots or spot_nr < 1 or spot_nr > len(self.simulated_spots):
            return
        self._sync_params()
        dist = self.dist
        lsd_val = float(self.lsd_edit.text())
        this_lsd = lsd_val + dist * self.dist_diff
        sim_lsd = lsd_val

        line = self.simulated_spots[spot_nr - 1].split()
        rad = float(line[0]) * this_lsd / sim_lsd
        eta = float(line[1])
        thisome = float(line[2])

        frame_to_read = int((thisome - self.startome) / self.omestep) if self.omestep != 0 else 0
        self.frame_spin.setValue(frame_to_read)

        ys, zs = YZ4mREta(rad, eta)
        ya = self.pos[0] * math.sin(thisome * deg2rad) + self.pos[1] * math.cos(thisome * deg2rad)
        xa = -self.pos[1] * math.sin(thisome * deg2rad) + self.pos[0] * math.cos(thisome * deg2rad)
        px = self.pixel_size
        yn = (ya + ys * (1 - xa / this_lsd)) / px + self.bcs[dist][0]
        zn = (zs * (1 - xa / this_lsd)) / px + self.bcs[dist][1]

        # Clear old overlays and add spot markers
        self.image_view.clear_overlays()
        # Red: spot position
        spot_item = pg.ScatterPlotItem([yn], [zn], size=12, pen=pg.mkPen('r', width=2), brush=None, symbol='o')
        self.image_view.add_overlay(spot_item)
        # Blue: beam center
        bc_item = pg.ScatterPlotItem([self.bcs[dist][0]], [self.bcs[dist][1]], size=10,
                                      pen=pg.mkPen('b', width=2), brush=None, symbol='star')
        self.image_view.add_overlay(bc_item)
        print(f"Spot {spot_nr}: frame={frame_to_read} yn={yn:.1f} zn={zn:.1f} ω={thisome:.1f}°")

    # ── Auto-detect ────────────────────────────────────────────────

    def _start_auto_detect(self):
        worker = AsyncWorker(
            target=nf_auto_detect, args=(self.folder, self.fnstem))
        worker.finished_signal.connect(self._apply_auto_detect)
        worker.start()
        self._detect_worker = worker

    def _apply_auto_detect(self, result):
        if not result:
            return
        if result.get('beampos_mode'):
            self._beampos_mode = True
            self._beampos_files = result['beampos_files']
            print(f"BeamPos mode: {len(self._beampos_files)} files")
        if 'start_frame' in result:
            self.start_frame_nr = result['start_frame']
            self.startframe_edit.setText(str(self.start_frame_nr))
            self.frame_spin.setValue(0)
            self.setWindowTitle(self.windowTitle() + " [files detected]")
            self._load_and_display()


# ═══════════════════════════════════════════════════════════════════════
#  Grain Parameter Dialog
# ═══════════════════════════════════════════════════════════════════════

class GrainDialog(QtWidgets.QDialog):
    """Enter/edit grain orientation, position, and material parameters."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("Grain Parameters")
        self.resize(700, 350)
        self._build_ui()

    def _build_ui(self):
        lay = QtWidgets.QFormLayout(self)

        # Orientation matrix (3x3 = 9 entries)
        om_row = QtWidgets.QHBoxLayout()
        self.om_edits = []
        for i in range(9):
            e = QtWidgets.QLineEdit(str(self.viewer.om[i]))
            e.setMinimumWidth(75)
            om_row.addWidget(e)
            self.om_edits.append(e)
        lay.addRow("Orient. Matrix:", om_row)

        # Position (3 entries)
        pos_row = QtWidgets.QHBoxLayout()
        self.pos_edits = []
        for i in range(3):
            e = QtWidgets.QLineEdit(str(self.viewer.pos[i]))
            e.setMinimumWidth(90)
            pos_row.addWidget(e)
            self.pos_edits.append(e)
        lay.addRow("Position (μm):", pos_row)

        # Lattice constants (6 entries)
        lc_row = QtWidgets.QHBoxLayout()
        self.lc_edits = []
        for i in range(6):
            e = QtWidgets.QLineEdit(str(self.viewer.latC[i]))
            e.setMinimumWidth(75)
            lc_row.addWidget(e)
            self.lc_edits.append(e)
        lay.addRow("Lattice Const:", lc_row)

        self.wl_edit = QtWidgets.QLineEdit(str(self.viewer.wl))
        lay.addRow("Wavelength (Å):", self.wl_edit)
        self.startome_edit = QtWidgets.QLineEdit(str(self.viewer.startome))
        lay.addRow("StartOmega (°):", self.startome_edit)
        self.omestep_edit = QtWidgets.QLineEdit(str(self.viewer.omestep))
        lay.addRow("OmegaStep (°):", self.omestep_edit)
        self.sg_edit = QtWidgets.QLineEdit(str(self.viewer.sg))
        lay.addRow("SpaceGroup:", self.sg_edit)
        self.maxrad_edit = QtWidgets.QLineEdit(str(self.viewer.maxringrad))
        lay.addRow("MaxRingRad (μm):", self.maxrad_edit)

        btn = QtWidgets.QPushButton("Confirm")
        btn.clicked.connect(self._accept)
        lay.addRow(btn)

    def _accept(self):
        v = self.viewer
        for i in range(9):
            v.om[i] = float(self.om_edits[i].text())
        for i in range(3):
            v.pos[i] = float(self.pos_edits[i].text())
        for i in range(6):
            v.latC[i] = float(self.lc_edits[i].text())
        v.wl = float(self.wl_edit.text())
        v.startome = float(self.startome_edit.text())
        v.omestep = float(self.omestep_edit.text())
        v.sg = int(self.sg_edit.text())
        v.maxringrad = float(self.maxrad_edit.text())
        self.accept()


# ═══════════════════════════════════════════════════════════════════════
#  Beam Center Dialog
# ═══════════════════════════════════════════════════════════════════════

class BeamCenterDialog(QtWidgets.QDialog):
    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.setWindowTitle("Beam Centers (pixels)")
        self.resize(400, 300)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Enter beam center for each distance:"))

        self.entries = []
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Distance"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("Y"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Z"), 0, 2)
        for d in range(viewer.n_distances):
            grid.addWidget(QtWidgets.QLabel(str(d)), d + 1, 0)
            ey = QtWidgets.QLineEdit(str(viewer.bcs[d][0]))
            ez = QtWidgets.QLineEdit(str(viewer.bcs[d][1]))
            grid.addWidget(ey, d + 1, 1)
            grid.addWidget(ez, d + 1, 2)
            self.entries.append((ey, ez))
        lay.addLayout(grid)

        lay.addWidget(QtWidgets.QLabel("Distance difference (μm):"))
        self.dist_diff_edit = QtWidgets.QLineEdit(str(viewer.dist_diff))
        lay.addWidget(self.dist_diff_edit)

        btn = QtWidgets.QPushButton("Confirm")
        btn.clicked.connect(self._accept)
        lay.addWidget(btn)

    def _accept(self):
        for d, (ey, ez) in enumerate(self.entries):
            self.viewer.bcs[d][0] = float(ey.text())
            self.viewer.bcs[d][1] = float(ez.text())
        self.viewer.dist_diff = float(self.dist_diff_edit.text())
        self.accept()


# ═══════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    app = QtWidgets.QApplication(sys.argv)
    theme = 'light'
    if '--dark' in sys.argv:
        theme = 'dark'
    apply_theme(app, theme)
    viewer = NFViewer(theme=theme)
    viewer.show()
    sys.exit(app.exec_())


# MIDAS version banner
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'utils'))
    from version import version_string as _vs
    print(_vs())
except Exception:
    pass

if __name__ == '__main__':
    main()
