#!/usr/bin/env python3
"""
FF-HEDM Viewer — PyQt5/PyQtGraph edition.

Replaces ff_asym.py (Tkinter+Matplotlib) with a modern, reactive GUI.
All image I/O and crystallography logic preserved from ff_asym.py.
"""

import sys
import os
import math
import tempfile
import subprocess
import threading
import shutil
import bz2
import glob
import itertools
import json
import re

import numpy as np
from numpy import linalg as LA
from math import sin, cos, sqrt


from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

try:
    import h5py
except ImportError:
    h5py = None
try:
    import tifffile
except ImportError:
    tifffile = None
try:
    import zarr
except ImportError:
    zarr = None

# Import shared utilities
_this_dir = os.path.dirname(os.path.abspath(__file__))
_utils_dir = os.path.join(os.path.dirname(_this_dir), 'utils')
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _utils_dir not in sys.path:
    sys.path.append(_utils_dir)

from gui_common import (MIDASImageView, apply_theme, get_colormap,
                         AsyncWorker, LogPanel, add_shortcut, COLORMAPS,
                         draw_lab_frame_axes)
import multidet as _md

try:
    import midas_config
    midas_config.run_startup_checks()
except ImportError:
    midas_config = None

# ── Constants ──
deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823
_color_cycle_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                       '#911eb4', '#42d4f4', '#f032e6', '#bfef45']


# ═══════════════════════════════════════════════════════════════════════
#  Crystallography helpers (unchanged from ff_asym.py)
# ═══════════════════════════════════════════════════════════════════════

def CalcEtaAngle(XYZ):
    alpha = rad2deg * np.arccos(np.divide(XYZ[2, :], LA.norm(XYZ[1:, :], axis=0)))
    alpha[XYZ[1, :] > 0] = -alpha[XYZ[1, :] > 0]
    return alpha

def CalcEtaAngleRad(y, z):
    Rad = sqrt(y * y + z * z)
    alpha = rad2deg * math.acos(z / Rad) if Rad > 0 else 0
    if y > 0:
        alpha = -alpha
    return [alpha, Rad]

def YZ4mREta(R, Eta):
    return [-R * sin(Eta * deg2rad), R * cos(Eta * deg2rad)]

def det_transforms(txv, tyv, tzv):
    txr, tyr, tzr = txv * deg2rad, tyv * deg2rad, tzv * deg2rad
    Rx = np.array([[1, 0, 0], [0, cos(txr), -sin(txr)], [0, sin(txr), cos(txr)]])
    Ry = np.array([[cos(tyr), 0, sin(tyr)], [0, 1, 0], [-sin(tyr), 0, cos(tyr)]])
    Rz = np.array([[cos(tzr), -sin(tzr), 0], [sin(tzr), cos(tzr), 0], [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))


# ═══════════════════════════════════════════════════════════════════════
#  Image I/O (preserved from ff_asym.py)
# ═══════════════════════════════════════════════════════════════════════

def get_bz2_data(fn):
    inner_name = fn[:-4] if fn.lower().endswith('.bz2') else fn
    suffix = os.path.splitext(inner_name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        with bz2.BZ2File(fn, 'rb') as source:
            shutil.copyfileobj(source, tmp)
        return tmp.name


def build_filename(folder, fstem, fnum, padding, det_nr, ext, sep_folder=False):
    fldr = folder
    if sep_folder and det_nr != -1:
        fldr = os.path.join(folder, 'ge' + str(det_nr))
    if det_nr != -1:
        fn = os.path.join(fldr, fstem + '_' + str(fnum).zfill(padding) + '.ge' + str(det_nr))
    else:
        fn = os.path.join(fldr, fstem + '_' + str(fnum).zfill(padding) + '.' + ext)
    if not os.path.exists(fn) and os.path.exists(fn + '.bz2'):
        fn = fn + '.bz2'
    return fn


def read_image(fn, header, bytes_per_pixel, ny, nz, frame_idx=0,
               do_transpose=False, do_hflip=False, do_vflip=False,
               mask=None, zarr_store=None, zarr_dark_mean=None,
               is_dark=False, hdf5_data_path='', hdf5_dark_path=''):
    """Read a single image frame. Returns 2D float array."""
    # Zarr-ZIP mode
    if zarr_store is not None:
        try:
            if is_dark:
                data = zarr_dark_mean.copy() if zarr_dark_mean is not None else np.zeros((ny, nz))
            else:
                dset = zarr_store['exchange/data']
                data = dset[frame_idx, :, :].astype(float) if frame_idx < dset.shape[0] else np.zeros((ny, nz))
            data = data.astype(float)[::-1, ::-1].copy()
            if do_transpose: data = np.transpose(data)
            if do_hflip and do_vflip: data = data[::-1, ::-1].copy()
            elif do_hflip: data = data[::-1, :].copy()
            elif do_vflip: data = data[:, ::-1].copy()
            if mask is not None and mask.shape == data.shape:
                data[mask == 1] = 0
            return data
        except Exception as e:
            print(f"Error reading Zarr: {e}")
            return np.zeros((ny, nz))

    # Compression
    if fn.endswith('.bz2'):
        temp_fn = get_bz2_data(fn)
        try:
            return read_image(temp_fn, header, bytes_per_pixel, ny, nz, frame_idx,
                              do_transpose, do_hflip, do_vflip, mask,
                              hdf5_data_path=hdf5_data_path, hdf5_dark_path=hdf5_dark_path)
        finally:
            if os.path.exists(temp_fn): os.remove(temp_fn)

    ext = os.path.splitext(fn)[1].lower()

    if ext in ['.h5', '.hdf', '.hdf5', '.nxs']:
        with h5py.File(fn, 'r') as f:
            dset_path = hdf5_dark_path if is_dark else hdf5_data_path
            if dset_path in f:
                dset = f[dset_path]
                data = dset[frame_idx, :, :] if dset.ndim == 3 else dset[:]
            else:
                return np.zeros((ny, nz))
    elif ext in ['.tif', '.tiff']:
        if tifffile:
            data = tifffile.imread(fn, key=frame_idx)
        else:
            return np.zeros((ny, nz))
    elif ext == '.cbf':
        from read_cbf import read_cbf as _read_cbf
        _, data = _read_cbf(fn, check_md5=False)
    else:
        with open(fn, 'rb') as f:
            skip = header + frame_idx * (bytes_per_pixel * ny * nz)
            f.seek(skip, os.SEEK_SET)
            dtype = np.uint16 if bytes_per_pixel == 2 else np.int32
            data = np.fromfile(f, dtype=dtype, count=ny * nz)
        data = data.reshape((ny, nz))

    data = data.astype(float)
    if do_transpose: data = np.transpose(data)
    if do_hflip and do_vflip: data = data[::-1, ::-1].copy()
    elif do_hflip: data = data[::-1, :].copy()
    elif do_vflip: data = data[:, ::-1].copy()
    if mask is not None and mask.shape == data.shape:
        data[mask == 1] = 0
    return data


def read_image_max(fn, header, bytes_per_pixel, ny, nz, n_frames, start_frame=0,
                   do_transpose=False, do_hflip=False, do_vflip=False,
                   mask=None, hdf5_data_path=''):
    """Compute pixel-wise max over frames."""
    ext = os.path.splitext(fn)[1].lower()
    data_max = None
    for i in range(start_frame, start_frame + n_frames):
        frame = read_image(fn, header, bytes_per_pixel, ny, nz, i,
                           do_transpose, do_hflip, do_vflip, mask,
                           hdf5_data_path=hdf5_data_path)
        data_max = frame if data_max is None else np.maximum(data_max, frame)
    return data_max


def read_mask(fn, ny, nz, do_transpose=False, do_hflip=False, do_vflip=False):
    """Read uint8 TIFF mask file. 1 = masked, 0 = good pixel."""
    if not fn or not os.path.exists(fn):
        return None
    try:
        if tifffile is None:
            print("tifffile not installed — cannot read mask")
            return None
        mask = tifffile.imread(fn).astype(np.uint8)
        if mask.shape != (ny, nz):
            print(f"Mask shape {mask.shape} does not match image ({ny}, {nz})")
            return None
        if do_transpose: mask = np.transpose(mask)
        if do_hflip and do_vflip: mask = mask[::-1, ::-1].copy()
        elif do_hflip: mask = mask[::-1, :].copy()
        elif do_vflip: mask = mask[:, ::-1].copy()
        return mask
    except Exception as e:
        print(f"Error reading mask: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Auto-detection (from ff_asym.py, adapted)
# ═══════════════════════════════════════════════════════════════════════

def _parse_numbered_filename(basename):
    """Parse a numbered filename like 'stem_00001.ext' into components.

    Finds the last ``_DIGITS.`` pattern in *basename* and splits there.
    This handles filenames with multiple dots and special characters,
    e.g. ``frame_%I.cbf_00001_Varex_1_00001.cbf``.

    Returns (stem, file_nr, padding, ext) or None if no pattern found.
    """
    matches = list(re.finditer(r'_(\d+)\.', basename))
    if not matches:
        return None
    last = matches[-1]
    digits = last.group(1)
    stem = basename[:last.start()]          # everything before _DIGITS
    ext  = basename[last.end():]            # everything after the dot  (no leading dot)
    return stem, int(digits), len(digits), ext


def auto_detect_files(cwd):
    """Auto-detect data files, dark files, and zarr-zip from directory.
    Returns dict with keys: file_stem, folder, padding, first_nr, ext,
    dark_stem, dark_num, n_frames, zarr_zip."""
    result = {}
    basename = os.path.basename(cwd)
    if not basename:
        return result

    # Check for zarr-zip
    zips = [f for f in os.listdir(cwd) if f.endswith('.MIDAS.zip')]
    if zips:
        result['zarr_zip'] = os.path.join(cwd, sorted(zips)[0])
        return result

    all_files = sorted(os.listdir(cwd))
    data_files = []
    for f in all_files:
        parsed = _parse_numbered_filename(f)
        if parsed and parsed[0].startswith(basename + '_'):
            # a bit redundant but keeps dark_* away
            data_files.append(f)

    if not data_files:
        return result

    first = data_files[0]
    parsed = _parse_numbered_filename(first)
    if parsed is None:
        return result

    stem, first_nr, padding, ext = parsed
    result['first_nr'] = first_nr
    result['padding'] = padding
    result['file_stem'] = stem
    result['folder'] = cwd + '/'
    result['ext'] = ext
    result['n_frames'] = 1

    # Dark files
    for f in all_files:
        dp = _parse_numbered_filename(f)
        if dp:
            prefix_lower = dp[0].lower()
            if prefix_lower.endswith('dark_before') or prefix_lower.endswith('dark_after'):
                result['dark_stem'] = dp[0]
                result['dark_num'] = dp[1]
                break

    # MIDAS-style param file (ps*.txt, Parameters*.txt, params*.txt)
    candidates = []
    for f in all_files:
        fl = f.lower()
        if fl.endswith('.txt') and (fl.startswith('ps') or fl.startswith('parameter')
                                     or fl.startswith('params')):
            candidates.append(os.path.join(cwd, f))
    if len(candidates) == 1:
        result['param_file'] = candidates[0]
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Ring computation
# ═══════════════════════════════════════════════════════════════════════

def compute_ring_points(ring_rad, lsd_local, lsd_orig, bc, px):
    """Compute Y,Z arrays for a ring overlay."""
    etas = np.linspace(-180, 180, 360)
    Y, Z = [], []
    rad2 = ring_rad * (lsd_local / lsd_orig) if lsd_orig > 0 else ring_rad
    for eta in etas:
        tmp = YZ4mREta(rad2, eta)
        Y.append(tmp[0] / px + bc[0])
        Z.append(tmp[1] / px + bc[1])
    return np.array(Y), np.array(Z)


# ═══════════════════════════════════════════════════════════════════════
#  FFViewer Main Window
# ═══════════════════════════════════════════════════════════════════════

class FFViewer(QtWidgets.QMainWindow):
    """FF-HEDM image viewer with reactive controls and ring overlays."""

    def __init__(self, theme='light'):
        super().__init__()
        self.setWindowTitle("FF Viewer (PyQtGraph) — MIDAS")
        self.resize(1500, 950)
        self._theme = theme

        self._init_state()
        self._build_ui()
        self._wire_signals()
        self._setup_shortcuts()
        self._start_auto_detect()

    # ── State ──────────────────────────────────────────────────────

    def _init_state(self):
        # Image params
        self.ny = 2048
        self.nz = 2048
        self.header_size = 8192
        self.bytes_per_pixel = 2
        self.pixel_size = 200.0
        self.file_stem = ''
        self.folder = os.getcwd() + '/'
        self.padding = 6
        self.ext = 'tif'
        self.first_file_nr = 1
        self.n_frames_per_file = 1
        self.frame_nr = 0
        self.dark_stem = ''
        self.dark_folder = ''
        self.dark_num = 0
        self.det_nr = -1
        self.sep_folder = False

        # Display
        self.bdata = None
        self.threshold = 0.0
        self.max_threshold = 2000.0
        self.use_log = False
        self.use_dark = False
        self.apply_mask = False
        self.mask_fn = ''
        self.mask_data = None
        self.hflip = False
        self.vflip = False
        self.do_transpose = False
        self.colormap_name = 'bone'

        # Detector
        self.lsd = [0, 0, 0, 0]
        self.lsd_local = 1000000.0
        self.lsd_orig = 1000000.0
        self.bc_local = [1024.0, 1024.0]
        self.bcs = None
        self.tx = [0]; self.ty = [0]; self.tz = [0]
        self.tx_local = 0.0
        self._tx_shift = (0, 0)        # (r_min, c_min) of expanded rotated canvas
        self.n_detectors = 1
        self.start_det_nr = 1; self.end_det_nr = 1
        self.dark_cache = {}

        # Rings
        self.ring_rads = None
        self.ring_nrs = []
        self.hkls = []
        self.rings_to_show = []
        self.show_rings = False
        self._ring_items = []

        # Lab-frame axes overlay
        self.show_axes = False

        # Multi-detector mode
        self.multi_mode = False
        self._det_states = [_md.DetectorState() for _ in range(4)]
        self.composite_op = 'max'                # 'max' or 'sum'
        self.big_det_size = 4096
        self._big_det_auto = True                # autopick on first param load
        self._composite_frame_cache = (None, -1) # (frame_idx, n_dets) → array

        # Crystallography
        self.sg = 225
        self.wl = 0.172979
        self.lattice_const = [5.41116, 5.41116, 5.41116, 90, 90, 90]
        self.temp_lsd = 1000000.0
        self.temp_max_ring_rad = 2000000.0
        self.wedge = 0.0

        # Zarr
        self.zarr_zip_path = None
        self.zarr_store = None
        self.zarr_dark_mean = None

        # HDF5
        self.hdf5_data_path = '/exchange/data'
        self.hdf5_dark_path = '/exchange/dark'

    # ── UI Construction ────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Menu bar ──
        file_menu = self.menuBar().addMenu('&File')
        save_act = file_menu.addAction('Save Session...')
        save_act.setShortcut('Ctrl+S')
        save_act.triggered.connect(self._save_session)
        load_act = file_menu.addAction('Load Session...')
        load_act.setShortcut('Ctrl+Shift+S')
        load_act.triggered.connect(self._load_session)
        file_menu.addSeparator()
        param_act = file_menu.addAction('Load Param File...')
        param_act.setShortcut('Ctrl+P')
        param_act.triggered.connect(self._on_load_param_file)

        # ── Toolbar ──
        tb = self._build_toolbar()
        main_layout.addLayout(tb)

        # ── Image View ──
        self.image_view = MIDASImageView(self)
        self.image_view.set_colormap(self.colormap_name)
        self.font_spin = self.image_view._font_spin
        self.image_view.fontSizeChanged.connect(self._on_font_changed)
        main_layout.addWidget(self.image_view, stretch=1)

        # ── Control Panels ──
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(4)
        # Stack the single-detector and multi-detector data-source panels;
        # the toolbar Multi-Det checkbox swaps which one is visible.
        self._file_stack = QtWidgets.QStackedWidget()
        self._file_stack.addWidget(self._build_file_panel())     # 0: single
        self._file_stack.addWidget(self._build_multi_panel())    # 1: multi
        ctrl.addWidget(self._file_stack, stretch=3)
        ctrl.addWidget(self._build_image_display_panel(), stretch=3)
        ctrl.addWidget(self._build_processing_panel(), stretch=2)
        main_layout.addLayout(ctrl)

        # ── Status Bar ──
        self.status_label = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)
        self.stats_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.stats_label)
        self.frame_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.frame_label)

        # ── Log Panel ──
        self.log_panel = LogPanel(self, "Log")
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_panel)
        self.log_panel.install_redirect()
        self.log_panel.hide()

        # ── Intensity vs Frame Dock ──
        self._ivf_dock = QtWidgets.QDockWidget("Intensity vs Frame", self)
        self._ivf_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetFloatable)
        ivf_widget = QtWidgets.QWidget()
        ivf_lay = QtWidgets.QVBoxLayout(ivf_widget)
        ivf_lay.setContentsMargins(2, 2, 2, 2)

        self._ivf_plot = pg.PlotWidget(title="Intensity vs Frame")
        self._ivf_plot.setLabel('bottom', 'Frame')
        self._ivf_plot.setLabel('left', 'Intensity')
        self._ivf_mean_curve = self._ivf_plot.plot(pen='c', name='Mean')
        self._ivf_max_curve = self._ivf_plot.plot(pen='r', name='Max')
        self._ivf_marker = self._ivf_plot.plot(pen=None, symbol='o',
                                                symbolSize=8, symbolBrush='y')
        self._ivf_plot.addLegend()
        self._ivf_plot.scene().sigMouseClicked.connect(self._on_ivf_clicked)
        ivf_lay.addWidget(self._ivf_plot)

        ivf_btn_lay = QtWidgets.QHBoxLayout()
        self._ivf_compute_btn = QtWidgets.QPushButton("Compute")
        self._ivf_compute_btn.setToolTip("Sweep all frames and compute mean/max intensity")
        self._ivf_compute_btn.clicked.connect(self._compute_ivf)
        ivf_btn_lay.addWidget(self._ivf_compute_btn)
        ivf_btn_lay.addStretch()
        ivf_lay.addLayout(ivf_btn_lay)

        self._ivf_dock.setWidget(ivf_widget)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._ivf_dock)
        self._ivf_dock.hide()

        # IvF data
        self._ivf_means = []
        self._ivf_maxs = []
        self._ivf_frames = []

        # View menu to toggle docks
        view_menu = self.menuBar().addMenu('&View')
        view_menu.addAction(self.log_panel.toggleViewAction())
        view_menu.addAction(self._ivf_dock.toggleViewAction())

        # Apply initial font so the viewer opens at a readable size.
        self._on_font_changed(self.font_spin.value())

    def _build_toolbar(self):
        tb = QtWidgets.QHBoxLayout()

        # Colormap
        tb.addWidget(QtWidgets.QLabel("Cmap:"))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText('bone')
        tb.addWidget(self.cmap_combo)

        # Theme
        tb.addWidget(QtWidgets.QLabel("Theme:"))
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(['light', 'dark'])
        self.theme_combo.setCurrentText(self._theme)
        tb.addWidget(self.theme_combo)

        # Log
        self.log_check = QtWidgets.QCheckBox("Log")
        tb.addWidget(self.log_check)

        # Rings
        self.rings_check = QtWidgets.QCheckBox("Rings")
        tb.addWidget(self.rings_check)

        # Lab-frame axes
        self.axes_check = QtWidgets.QCheckBox("Lab Axes")
        self.axes_check.setToolTip(
            "Overlay MIDAS lab-frame axes from beam center.\n"
            "+Y arrow → left, +Z arrow → up, ⊗ at BC = +X (beam, into page).\n"
            "Use this to verify ImTransOpt — features should be in the\n"
            "physically expected lab-frame quadrant.")
        tb.addWidget(self.axes_check)

        # Detector mode selector (single vs multi-panel configurations)
        tb.addWidget(QtWidgets.QLabel("Mode:"))
        self.detector_mode_combo = QtWidgets.QComboBox()
        self.detector_mode_combo.addItem("Single Panel")
        self.detector_mode_combo.addItem("1-ID-E HYDRA (GE1-GE4)")
        self.detector_mode_combo.setToolTip(
            "Select detector configuration.\n"
            "Single Panel: one detector, uses the Data Source panel.\n"
            "1-ID-E HYDRA: composite of up to 4 GE detectors; each reads\n"
            "its own ps.txt for BC, tx, ImTransOpt, dataLoc, GapIntensity.")
        tb.addWidget(self.detector_mode_combo)

        tb.addWidget(QtWidgets.QLabel("Composite:"))
        self.composite_combo = QtWidgets.QComboBox()
        self.composite_combo.addItems(['max', 'sum'])
        self.composite_combo.setCurrentText(self.composite_op)
        self.composite_combo.setEnabled(False)   # only meaningful in multi-mode
        tb.addWidget(self.composite_combo)

        # Intensity controls (moved from Display panel)
        tb.addWidget(QtWidgets.QLabel("Min I:"))
        self.min_intensity_edit = QtWidgets.QLineEdit("0")
        self.min_intensity_edit.setFixedWidth(60)
        tb.addWidget(self.min_intensity_edit)
        tb.addWidget(QtWidgets.QLabel("Max I:"))
        self.max_intensity_edit = QtWidgets.QLineEdit("1000")
        self.max_intensity_edit.setFixedWidth(60)
        tb.addWidget(self.max_intensity_edit)
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_intensity_levels)
        self.min_intensity_edit.returnPressed.connect(self._apply_intensity_levels)
        self.max_intensity_edit.returnPressed.connect(self._apply_intensity_levels)
        tb.addWidget(apply_btn)

        # Export
        export_btn = QtWidgets.QPushButton("Export PNG")
        export_btn.clicked.connect(lambda: self.image_view.export_png())
        tb.addWidget(export_btn)

        # Toggle log panel
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
        grp = QtWidgets.QGroupBox("Data Source")
        lay = QtWidgets.QGridLayout(grp)
        lay.setColumnStretch(4, 1)

        btn_first = QtWidgets.QPushButton("First File")
        btn_first.clicked.connect(self._on_first_file)
        btn_first.setFixedWidth(90)
        lay.addWidget(btn_first, 0, 0)

        btn_dark = QtWidgets.QPushButton("Dark File")
        btn_dark.clicked.connect(self._on_dark_file)
        btn_dark.setFixedWidth(90)
        lay.addWidget(btn_dark, 0, 1)

        self.dark_check = QtWidgets.QCheckBox("Dark")
        lay.addWidget(self.dark_check, 0, 2)

        btn_zip = QtWidgets.QPushButton("Load ZIP")
        btn_zip.clicked.connect(self._on_load_zip)
        lay.addWidget(btn_zip, 0, 3)

        lay.addWidget(QtWidgets.QLabel("File Nr"), 1, 0)
        self.file_nr_edit = QtWidgets.QLineEdit(str(self.first_file_nr))
        self.file_nr_edit.setMinimumWidth(70)
        lay.addWidget(self.file_nr_edit, 1, 1)

        lay.addWidget(QtWidgets.QLabel("Frames/File"), 1, 2)
        self.nframes_edit = QtWidgets.QLineEdit("1")
        self.nframes_edit.setMinimumWidth(70)
        lay.addWidget(self.nframes_edit, 1, 3)

        lay.addWidget(QtWidgets.QLabel("H5 Data"), 2, 0)
        self.h5data_edit = QtWidgets.QLineEdit(self.hdf5_data_path)
        lay.addWidget(self.h5data_edit, 2, 1, 1, 3)

        lay.addWidget(QtWidgets.QLabel("Mask"), 3, 0)
        self.mask_edit = QtWidgets.QLineEdit("")
        lay.addWidget(self.mask_edit, 3, 1, 1, 1)
        btn_mask = QtWidgets.QPushButton("Browse")
        btn_mask.clicked.connect(self._on_browse_mask)
        lay.addWidget(btn_mask, 3, 2)
        self.mask_check = QtWidgets.QCheckBox("Apply")
        lay.addWidget(self.mask_check, 3, 3)

        lay.addWidget(QtWidgets.QLabel("H5 Dark"), 4, 0)
        self.h5dark_edit = QtWidgets.QLineEdit(self.hdf5_dark_path)
        lay.addWidget(self.h5dark_edit, 4, 1, 1, 3)

        btn_param = QtWidgets.QPushButton("Load Params")
        btn_param.setToolTip(
            "Load a MIDAS-style parameter file (ps.txt / Parameters.txt).\n"
            "Populates detector geometry, transforms, crystallography, "
            "and file layout fields, then redraws.")
        btn_param.clicked.connect(self._on_load_param_file)
        lay.addWidget(btn_param, 5, 0)
        self.instr_only_check = QtWidgets.QCheckBox("Instr. only")
        self.instr_only_check.setToolTip(
            "When checked, loading a param file applies only instrument/geometry\n"
            "parameters (LSD, BC, pixel size, wavelength, space group, etc.).\n"
            "File layout fields (Folder, FileStem, StartNr, Ext) are ignored.")
        lay.addWidget(self.instr_only_check, 5, 1)
        self.param_label = QtWidgets.QLabel("")
        self.param_label.setStyleSheet("color: gray;")
        lay.addWidget(self.param_label, 5, 2, 1, 2)

        return grp

    def _build_image_display_panel(self):
        """Merged Image Settings + Display panel."""
        grp = QtWidgets.QGroupBox("Image & Display")
        lay = QtWidgets.QGridLayout(grp)

        # Row 0: pixel dimensions + frame
        lay.addWidget(QtWidgets.QLabel("Pixels H"), 0, 0)
        self.nz_edit = QtWidgets.QLineEdit(str(self.nz))
        self.nz_edit.setFixedWidth(55)
        lay.addWidget(self.nz_edit, 0, 1)

        lay.addWidget(QtWidgets.QLabel("Pixels V"), 0, 2)
        self.ny_edit = QtWidgets.QLineEdit(str(self.ny))
        self.ny_edit.setFixedWidth(55)
        lay.addWidget(self.ny_edit, 0, 3)

        lay.addWidget(QtWidgets.QLabel("Frame"), 0, 4)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, 99999)
        self.frame_spin.setValue(0)
        lay.addWidget(self.frame_spin, 0, 5)

        # Row 1: header, bytes/pixel, num frames
        lay.addWidget(QtWidgets.QLabel("Header"), 1, 0)
        self.header_edit = QtWidgets.QLineEdit(str(self.header_size))
        self.header_edit.setFixedWidth(55)
        lay.addWidget(self.header_edit, 1, 1)

        lay.addWidget(QtWidgets.QLabel("Bytes/Pixel"), 1, 2)
        self.bpp_edit = QtWidgets.QLineEdit(str(self.bytes_per_pixel))
        self.bpp_edit.setFixedWidth(35)
        lay.addWidget(self.bpp_edit, 1, 3)

        lay.addWidget(QtWidgets.QLabel("Num Frames"), 1, 4)
        self.max_frames_spin = QtWidgets.QSpinBox()
        self.max_frames_spin.setRange(1, 99999)
        self.max_frames_spin.setValue(240)
        lay.addWidget(self.max_frames_spin, 1, 5)

        # Row 2: pixel size + transforms
        lay.addWidget(QtWidgets.QLabel("Pixel Size (μm)"), 2, 0, 1, 2)
        self.px_edit = QtWidgets.QLineEdit(str(self.pixel_size))
        self.px_edit.setFixedWidth(55)
        lay.addWidget(self.px_edit, 2, 2)

        self.hflip_check = QtWidgets.QCheckBox("HFlip")
        lay.addWidget(self.hflip_check, 2, 3)
        self.vflip_check = QtWidgets.QCheckBox("VFlip")
        lay.addWidget(self.vflip_check, 2, 4)
        self.transpose_check = QtWidgets.QCheckBox("Transpose")
        lay.addWidget(self.transpose_check, 2, 5)

        # Row 3: Max/Sum toggles
        self.max_check = QtWidgets.QCheckBox("Max/Frames")
        lay.addWidget(self.max_check, 3, 0, 1, 3)
        self.sum_check = QtWidgets.QCheckBox("Sum/Frames")
        lay.addWidget(self.sum_check, 3, 3, 1, 3)

        return grp

    def _build_processing_panel(self):
        grp = QtWidgets.QGroupBox("Detector & Rings")
        lay = QtWidgets.QGridLayout(grp)

        btn_rings = QtWidgets.QPushButton("Rings Material")
        btn_rings.clicked.connect(self._on_ring_selection)
        lay.addWidget(btn_rings, 0, 0, 1, 2)

        lay.addWidget(QtWidgets.QLabel("Lsd (μm)"), 1, 0)
        self.lsd_edit = QtWidgets.QLineEdit(str(self.lsd_local))
        self.lsd_edit.setMinimumWidth(100)
        lay.addWidget(self.lsd_edit, 1, 1)

        lay.addWidget(QtWidgets.QLabel("Beam Ctr Y"), 2, 0)
        self.bcy_edit = QtWidgets.QLineEdit(str(self.bc_local[0]))
        self.bcy_edit.setMinimumWidth(90)
        lay.addWidget(self.bcy_edit, 2, 1)

        lay.addWidget(QtWidgets.QLabel("Beam Ctr Z"), 3, 0)
        self.bcz_edit = QtWidgets.QLineEdit(str(self.bc_local[1]))
        self.bcz_edit.setMinimumWidth(90)
        lay.addWidget(self.bcz_edit, 3, 1)

        lay.addWidget(QtWidgets.QLabel("Tx (deg)"), 4, 0)
        self.tx_edit = QtWidgets.QLineEdit(str(self.tx_local))
        self.tx_edit.setMinimumWidth(90)
        self.tx_edit.setToolTip(
            "Detector tilt about beam axis (deg).\n"
            "Image is rotated around (Beam Ctr Y, Beam Ctr Z) by -Tx to undo the tilt.\n"
            "Cursor R/η are reported in the corrected frame.")
        lay.addWidget(self.tx_edit, 4, 1)

        return grp

    # ── Multi-detector data-source panel ───────────────────────────

    def _build_multi_panel(self):
        """4 detector rows in a single shared QGridLayout so all ... buttons
        align. Above the rows: autoload controls and shared HDF5 path fields.
        """
        grp = QtWidgets.QGroupBox(
            "Multi-Detector Data Source — load one ps.txt per detector")
        outer = QtWidgets.QVBoxLayout(grp)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(3)

        # ── Autoload / auto-fill row ──────────────────────────────────
        autoload_row = QtWidgets.QHBoxLayout()
        autoload_btn = QtWidgets.QPushButton("Autoload from one param file…")
        autoload_btn.setToolTip(
            "Pick any one detector's param file. The viewer derives the other\n"
            "three by substituting ge1/ge2/ge3/ge4 in the path, parses each\n"
            "found param file for Folder/FileStem/StartNr/Padding/Ext, and\n"
            "populates the data + dark file for every detector that resolves\n"
            "to an existing file. Slots whose files don't exist stay empty.")
        autoload_btn.clicked.connect(self._on_autoload_multi)
        autoload_row.addWidget(autoload_btn)
        self._autofill_check = QtWidgets.QCheckBox("Auto-fill siblings")
        self._autofill_check.setChecked(True)
        self._autofill_check.setToolTip(
            "When picking a data, dark, or param file for one GE detector,\n"
            "automatically derive and load the equivalent files for the other\n"
            "GE detectors by substituting the detector tag (e.g. ge1→ge2).")
        autoload_row.addWidget(self._autofill_check)
        autoload_row.addStretch()
        self._autoload_status = QtWidgets.QLabel("")
        self._autoload_status.setStyleSheet("color: #888888;")
        autoload_row.addWidget(self._autoload_status)
        outer.addLayout(autoload_row)

        # ── Shared HDF5 dataset paths ─────────────────────────────────
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(QtWidgets.QLabel("Data path:"))
        self._multi_data_path_edit = QtWidgets.QLineEdit(self.hdf5_data_path)
        self._multi_data_path_edit.setToolTip(
            "HDF5 dataset path for frame data in each detector file\n"
            "(e.g. /exchange/data). Applied to all detectors.")
        self._multi_data_path_edit.setFixedWidth(150)
        self._multi_data_path_edit.editingFinished.connect(self._on_multi_paths_changed)
        path_row.addWidget(self._multi_data_path_edit)
        path_row.addSpacing(12)
        path_row.addWidget(QtWidgets.QLabel("Dark path:"))
        self._multi_dark_path_edit = QtWidgets.QLineEdit('/exchange/data_dark')
        self._multi_dark_path_edit.setToolTip(
            "HDF5 dataset path for the dark frame in each detector file\n"
            "(e.g. /exchange/data_dark). Applied to all detectors.")
        self._multi_dark_path_edit.setFixedWidth(150)
        self._multi_dark_path_edit.editingFinished.connect(self._on_multi_paths_changed)
        path_row.addWidget(self._multi_dark_path_edit)
        path_row.addStretch()
        outer.addLayout(path_row)

        # ── Single shared grid for all 4 GE detector rows ─────────────
        # Columns:  0=enable  1=lbl  2=file_lbl  3=btn  4=lbl  5=file_lbl  6=btn  7=clr
        #           (GE chk)  Data   <filename>   ...    Dark   <filename>   ...    ✕
        # Row 2*i+1 (param row):
        #           (span)    Param  <filename>   ...    <status, span 4>
        card_grid = QtWidgets.QGridLayout()
        card_grid.setSpacing(2)
        card_grid.setColumnStretch(2, 2)   # data/param filename column
        card_grid.setColumnStretch(5, 1)   # dark filename column

        self._det_widgets = []
        for i in range(4):
            r0 = i * 2      # data+dark row
            r1 = i * 2 + 1  # param+status row

            # Add a thin separator line above GE2-GE4
            if i > 0:
                sep = QtWidgets.QFrame()
                sep.setFrameShape(QtWidgets.QFrame.HLine)
                sep.setStyleSheet("color: #cccccc;")
                card_grid.addWidget(sep, r0, 0, 1, 8)
                r0 += 1
                r1 += 1
                # Shift subsequent rows down by 1 for the separator
                # (recalculate based on actual inserted rows)

        # Rebuild without the separator complexity — use a cleaner row scheme
        card_grid = QtWidgets.QGridLayout()
        card_grid.setSpacing(2)
        card_grid.setColumnStretch(2, 2)   # data/param filename column
        card_grid.setColumnStretch(5, 1)   # dark filename column

        self._det_widgets = []
        grid_row = 0
        for i in range(4):
            if i > 0:
                sep = QtWidgets.QFrame()
                sep.setFrameShape(QtWidgets.QFrame.HLine)
                sep.setFrameShadow(QtWidgets.QFrame.Sunken)
                card_grid.addWidget(sep, grid_row, 0, 1, 8)
                grid_row += 1

            r0 = grid_row      # data + dark row
            r1 = grid_row + 1  # param + status row

            en = QtWidgets.QCheckBox(f"GE{i+1}")
            en.setChecked(True)
            en.setToolTip(f"Include detector GE{i+1} in the composite")
            en.toggled.connect(lambda c, idx=i: self._on_det_enabled(idx, c))
            card_grid.addWidget(en, r0, 0, 2, 1)

            # Data row
            card_grid.addWidget(QtWidgets.QLabel("Data"), r0, 1)
            data_lbl = QtWidgets.QLabel("(none)")
            data_lbl.setStyleSheet("color: gray;")
            card_grid.addWidget(data_lbl, r0, 2)
            data_btn = QtWidgets.QPushButton("…")
            data_btn.setFixedWidth(28)
            data_btn.clicked.connect(lambda _, idx=i: self._on_pick_det_data(idx))
            card_grid.addWidget(data_btn, r0, 3)

            card_grid.addWidget(QtWidgets.QLabel("Dark"), r0, 4)
            dark_lbl = QtWidgets.QLabel("(same)")
            dark_lbl.setStyleSheet("color: gray;")
            card_grid.addWidget(dark_lbl, r0, 5)
            dark_btn = QtWidgets.QPushButton("…")
            dark_btn.setFixedWidth(28)
            dark_btn.clicked.connect(lambda _, idx=i: self._on_pick_det_dark(idx))
            card_grid.addWidget(dark_btn, r0, 6)
            dark_clr = QtWidgets.QPushButton("✕")
            dark_clr.setFixedWidth(24)
            dark_clr.setToolTip("Clear external dark (use path in data file)")
            dark_clr.clicked.connect(lambda _, idx=i: self._on_clear_det_dark(idx))
            card_grid.addWidget(dark_clr, r0, 7)

            # Param row
            card_grid.addWidget(QtWidgets.QLabel("Param"), r1, 1)
            param_lbl = QtWidgets.QLabel("(none)")
            param_lbl.setStyleSheet("color: gray;")
            card_grid.addWidget(param_lbl, r1, 2)
            param_btn = QtWidgets.QPushButton("…")
            param_btn.setFixedWidth(28)
            param_btn.clicked.connect(lambda _, idx=i: self._on_pick_det_param(idx))
            card_grid.addWidget(param_btn, r1, 3)

            status_lbl = QtWidgets.QLabel("")
            status_lbl.setStyleSheet("color: #888888; font-size: 9pt;")
            card_grid.addWidget(status_lbl, r1, 4, 1, 4)

            grid_row += 2
            self._det_widgets.append(dict(
                enable=en, data_lbl=data_lbl, dark_lbl=dark_lbl,
                param_lbl=param_lbl, status_lbl=status_lbl))

        outer.addLayout(card_grid)

        # ── BigDetSize control + Save composite ───────────────────────
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(QtWidgets.QLabel("BigDetSize:"))
        self.bigdet_spin = QtWidgets.QSpinBox()
        self.bigdet_spin.setRange(256, 16384)
        self.bigdet_spin.setSingleStep(256)
        self.bigdet_spin.setValue(self.big_det_size)
        self.bigdet_spin.setToolTip(
            "Composite canvas size (pixels per side). Auto-pick = 2×max(NrPixels)\n"
            "until you change this manually.")
        self.bigdet_spin.valueChanged.connect(self._on_bigdet_changed)
        bottom.addWidget(self.bigdet_spin)

        bottom.addStretch()
        save_btn = QtWidgets.QPushButton("Save Composite Frame…")
        save_btn.setToolTip("Save the current composited frame as a TIFF")
        save_btn.clicked.connect(self._on_save_composite)
        bottom.addWidget(save_btn)
        outer.addLayout(bottom)
        return grp

    # ── Signal Wiring (reactive) ───────────────────────────────────

    def _wire_signals(self):
        # Frame change → load + display
        self.frame_spin.valueChanged.connect(self._load_and_display)
        # Toggles → replot
        self.log_check.toggled.connect(self._on_log_toggled)
        self.dark_check.toggled.connect(self._load_and_display)
        self.rings_check.toggled.connect(self._on_rings_toggled)
        self.axes_check.toggled.connect(self._on_axes_toggled)
        self.detector_mode_combo.currentIndexChanged.connect(self._on_detector_mode_changed)
        self.composite_combo.currentTextChanged.connect(self._on_composite_op_changed)
        self.mask_check.toggled.connect(self._load_and_display)
        self.hflip_check.toggled.connect(self._load_and_display)
        self.vflip_check.toggled.connect(self._load_and_display)
        self.transpose_check.toggled.connect(self._load_and_display)
        self.max_check.toggled.connect(self._on_max_toggled)
        self.sum_check.toggled.connect(self._on_sum_toggled)
        # Colormap
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        # Theme
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        # Mouse wheel on image → frame scroll
        self.image_view.frameScrolled.connect(self._on_frame_scroll)
        # Cursor tracking
        self.image_view.cursorMoved.connect(self._on_cursor_moved)
        # Live ring redraw on BC/Lsd change
        self.bcy_edit.editingFinished.connect(self._redraw_if_rings)
        self.bcz_edit.editingFinished.connect(self._redraw_if_rings)
        self.lsd_edit.editingFinished.connect(self._redraw_if_rings)
        self.tx_edit.editingFinished.connect(self._load_and_display)
        self.h5dark_edit.editingFinished.connect(self._load_and_display)
        self.image_view.dataStatsUpdated.connect(self._on_stats_updated)
        # Movie mode: advance frame by 1 (wraps at max)
        self.image_view.movieFrameAdvance.connect(self._movie_advance_frame)
        # Drag-and-drop: open dropped file
        self.image_view.fileDropped.connect(self._on_file_dropped)

    def _show_help(self):
        QtWidgets.QMessageBox.information(self, 'FF Viewer — Controls',
            'Mouse Controls:\n'
            '  Ctrl+Scroll wheel — Change frame\n'
            '  Right-click drag — Zoom rectangle (in Zoom mode)\n'
            '  Left-click drag — Pan (in Pan mode)\n'
            '  Double-click — Reset to full view\n'
            '\n'
            'Keyboard Shortcuts:\n'
            '  ← / → — Previous / Next frame\n'
            '  L — Toggle log scale\n'
            '  R — Toggle ring overlay\n'
            '  A — Toggle MIDAS lab-frame axes\n'
            '  Q — Quit\n'
            '\n'
            'MIDAS lab frame (axes overlay):\n'
            '  +Y → display LEFT, +Z → display UP\n'
            '  +X → into the page (beam direction, ⊗ at BC)\n'
            '  η = 0 at +Z, +90° at −Y, ±180° at −Z, −90° at +Y\n'
            '  View is "looking from source toward detector"\n'
            '\n'
            'Multi-Detector mode (toolbar Multi-Det checkbox):\n'
            '  Loads up to 4 GE detector HDF5 files. Each detector reads its\n'
            '  own ps.txt for BC, tx, ImTransOpt, dataLoc, BadPxIntensity,\n'
            '  GapIntensity. Frames are inverse-warped into a common\n'
            '  BigDetSize×BigDetSize composite (rotated by per-detector tx\n'
            '  about the composite center) and combined by max or sum.\n'
            '\n'
            'Histogram (right side of image):\n'
            '  Drag top/bottom bars — Adjust thresholds\n'
            '  Right-click histogram — Change colormap\n')

    def _setup_shortcuts(self):
        add_shortcut(self, 'Right', lambda: self.frame_spin.setValue(self.frame_spin.value() + 1))
        add_shortcut(self, 'Left', lambda: self.frame_spin.setValue(self.frame_spin.value() - 1))
        add_shortcut(self, 'L', lambda: self.log_check.toggle())
        add_shortcut(self, 'R', lambda: self.rings_check.toggle())
        add_shortcut(self, 'A', lambda: self.axes_check.toggle())
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
            'viewer': 'ff',
            'folder': self.folder,
            'file_stem': self.file_stem,
            'first_file_nr': self.first_file_nr,
            'padding': self.padding,
            'ext': self.ext,
            'frame': self.frame_spin.value(),
            'ny': self.ny, 'nz': self.nz,
            'header_size': self.header_size,
            'bytes_per_pixel': self.bytes_per_pixel,
            'n_frames_per_file': self.n_frames_per_file,
            'lsd': float(self.lsd_edit.text()),
            'bcy': float(self.bcy_edit.text()),
            'bcz': float(self.bcz_edit.text()),
            'tx': float(self.tx_edit.text()),
            'px': float(self.px_edit.text()),
            'colormap': self.cmap_combo.currentText(),
            'theme': self.theme_combo.currentText(),
            'log': self.log_check.isChecked(),
            'hflip': self.hflip_check.isChecked(),
            'vflip': self.vflip_check.isChecked(),
            'transpose': self.transpose_check.isChecked(),
            'show_rings': self.rings_check.isChecked(),
            'show_axes': self.axes_check.isChecked(),
            'use_dark': self.dark_check.isChecked(),
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
        self.file_stem = state.get('file_stem', self.file_stem)
        self.first_file_nr = state.get('first_file_nr', self.first_file_nr)
        self.padding = state.get('padding', self.padding)
        self.ext = state.get('ext', self.ext)
        self.ny = state.get('ny', self.ny)
        self.nz = state.get('nz', self.nz)
        self.header_size = state.get('header_size', self.header_size)
        self.bytes_per_pixel = state.get('bytes_per_pixel', self.bytes_per_pixel)
        self.n_frames_per_file = state.get('n_frames_per_file', self.n_frames_per_file)

        self.file_nr_edit.setText(str(self.first_file_nr))
        self.nypx_edit.setText(str(self.ny))
        self.nzpx_edit.setText(str(self.nz))
        self.header_edit.setText(str(self.header_size))
        self.bpp_edit.setText(str(self.bytes_per_pixel))
        self.nframes_edit.setText(str(self.n_frames_per_file))
        self.lsd_edit.setText(str(state.get('lsd', 1000000.0)))
        self.bcy_edit.setText(str(state.get('bcy', 1024.0)))
        self.bcz_edit.setText(str(state.get('bcz', 1024.0)))
        self.tx_edit.setText(str(state.get('tx', 0.0)))
        self.px_edit.setText(str(state.get('px', 200.0)))

        self.cmap_combo.setCurrentText(state.get('colormap', 'bone'))
        self.theme_combo.setCurrentText(state.get('theme', 'light'))
        self.log_check.setChecked(state.get('log', False))
        self.hflip_check.setChecked(state.get('hflip', False))
        self.vflip_check.setChecked(state.get('vflip', False))
        self.transpose_check.setChecked(state.get('transpose', False))
        self.rings_check.setChecked(state.get('show_rings', False))
        self.axes_check.setChecked(state.get('show_axes', False))
        self.dark_check.setChecked(state.get('use_dark', False))

        self.frame_spin.setValue(state.get('frame', 0))
        print(f'Session loaded: {fn}')

    # ── Parameter File ─────────────────────────────────────────────

    @staticmethod
    def _parse_param_file(fn):
        """Parse a MIDAS-style param file (ps.txt / Parameters.txt).

        Format: one entry per line, ``key value [value...]``. ``#`` starts a
        comment, optional trailing semicolons are stripped. Keys that repeat
        across lines (e.g. ``ImTransOpt``, ``RingThresh``) are accumulated
        into a list of value-token-lists.
        Returns: dict mapping key -> list of value-token-lists.
        """
        params = {}
        with open(fn, 'r') as f:
            for raw in f:
                line = raw.split('#', 1)[0].strip().rstrip(';').strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                key = tokens[0]
                params.setdefault(key, []).append(tokens[1:])
        return params

    def _on_load_param_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select MIDAS Parameter File", os.getcwd(),
            "Param Files (*.txt);;All (*)")
        if fn:
            self._apply_param_file(fn)

    def _apply_param_file(self, fn):
        """Read fn, populate GUI fields from MIDAS params, then reload."""
        try:
            params = self._parse_param_file(fn)
        except Exception as e:
            print(f"Param file read failed: {e}")
            return

        # Helpers — return None if missing/unparseable so we can leave fields alone.
        def first_tokens(*keys):
            for k in keys:
                if k in params:
                    return params[k][0]
            return None

        def get_str(*keys):
            v = first_tokens(*keys)
            return v[0] if v else None

        def get_float(*keys):
            v = first_tokens(*keys)
            try:
                return float(v[0]) if v else None
            except (ValueError, IndexError):
                return None

        def get_int(*keys):
            v = first_tokens(*keys)
            try:
                return int(float(v[0])) if v else None
            except (ValueError, IndexError):
                return None

        def get_floats(n, *keys):
            v = first_tokens(*keys)
            if not v or len(v) < n:
                return None
            try:
                return [float(x) for x in v[:n]]
            except ValueError:
                return None

        applied = []  # for log
        instr_only = getattr(self, 'instr_only_check', None) and self.instr_only_check.isChecked()

        # ── Files / layout ──
        if not instr_only:
            folder = get_str('RawFolder', 'Folder')
            if folder:
                self.folder = folder.rstrip('/').rstrip('\\') + '/'
                applied.append(f"folder={folder}")
            fs = get_str('FileStem')
            if fs:
                self.file_stem = fs
                applied.append(f"FileStem={fs}")
            sn = get_int('StartNr', 'StartFileNrFirstLayer')
            if sn is not None:
                self.first_file_nr = sn
                self.file_nr_edit.setText(str(sn))
                applied.append(f"StartNr={sn}")
            pad = get_int('Padding')
            if pad is not None:
                self.padding = pad
            ext = get_str('Ext')
            if ext:
                self.ext = ext.lstrip('.')
                applied.append(f"Ext={self.ext}")
                # Detect ge<digit> as multi-detector marker
                if self.ext.startswith('ge') and len(self.ext) == 3 and self.ext[-1].isdigit():
                    self.det_nr = int(self.ext[-1])

        # ── Detector pixels ──
        npx = get_int('NrPixels')
        if npx is not None:
            self.ny = npx
            self.nz = npx
        npy = get_int('NrPixelsY')
        if npy is not None:
            self.ny = npy
        npz = get_int('NrPixelsZ')
        if npz is not None:
            self.nz = npz
        if any(v is not None for v in (npx, npy, npz)):
            self.ny_edit.setText(str(self.ny))
            self.nz_edit.setText(str(self.nz))
            applied.append(f"NrPixels={self.ny}x{self.nz}")

        hs = get_int('HeadSize', 'HeaderSize')
        if hs is not None:
            self.header_size = hs
            self.header_edit.setText(str(hs))
            applied.append(f"HeadSize={hs}")
        bpp = get_int('BytesPerPixel')
        if bpp is not None:
            self.bytes_per_pixel = bpp
            self.bpp_edit.setText(str(bpp))
            applied.append(f"BytesPerPixel={bpp}")

        px = get_float('px', 'PixelSize')
        if px is not None:
            self.pixel_size = px
            self.px_edit.setText(str(px))
            applied.append(f"px={px}")

        # ── Geometry ──
        lsd = get_float('Lsd')
        if lsd is not None:
            self.lsd_local = lsd
            self.lsd_orig = lsd
            self.lsd_edit.setText(str(lsd))
            applied.append(f"Lsd={lsd}")
        bc = get_floats(2, 'BC')
        if bc is None:
            ycen = get_float('YCen')
            zcen = get_float('ZCen')
            if ycen is not None and zcen is not None:
                bc = [ycen, zcen]
        if bc:
            self.bc_local = bc
            self.bcy_edit.setText(str(bc[0]))
            self.bcz_edit.setText(str(bc[1]))
            applied.append(f"BC={bc[0]},{bc[1]}")

        tx = get_float('tx')
        if tx is not None:
            self.tx_local = tx
            self.tx_edit.setText(str(tx))
            applied.append(f"tx={tx}")

        wd = get_float('Wedge')
        if wd is not None:
            self.wedge = wd

        # ── Crystallography (rings) ──
        wl = get_float('Wavelength')
        if wl is not None:
            self.wl = (12.398 / wl) if wl > 1.0 else wl
            applied.append(f"Wavelength={self.wl:.5f}Å")
        sg = get_int('SpaceGroup', 'SpaceGroupNumber')
        if sg is not None:
            self.sg = sg
            applied.append(f"SpaceGroup={sg}")
        lc = get_floats(6, 'LatticeConstant', 'LatticeParameter')
        if lc:
            self.lattice_const = lc
        mr = get_float('MaxRingRad', 'RhoD')
        if mr is not None:
            self.temp_max_ring_rad = mr

        # ── Image transforms (ImTransOpt: 1=HFlip 2=VFlip 3=Transpose, repeatable) ──
        if 'ImTransOpt' in params:
            opts = []
            for line_vals in params['ImTransOpt']:
                for v in line_vals:
                    try:
                        opts.append(int(v))
                    except ValueError:
                        pass
            self.hflip_check.setChecked(1 in opts)
            self.vflip_check.setChecked(2 in opts)
            self.transpose_check.setChecked(3 in opts)
            applied.append(f"ImTransOpt={opts}")

        # ── Frames per file ──
        nfs = get_int('NrFilesPerSweep', 'nFramesPerFile', 'NFramesPerFile')
        if nfs is not None:
            self.n_frames_per_file = nfs
            self.nframes_edit.setText(str(nfs))

        # ── HDF5 dataset paths (dataLoc / darkLoc) ──
        # APS GE files commonly store dark at /exchange/data_dark, not the
        # default /exchange/dark — and the param file may explicitly point
        # the dark to a different dataset (e.g. /exchange/data inside an
        # external dark-only file). Strip a trailing '/' since h5py is
        # forgiving but the comparison `dpath in f` relies on exact match.
        dl = get_str('dataLoc')
        if dl:
            self.hdf5_data_path = dl.rstrip('/') or '/'
            if hasattr(self, 'h5data_edit'):
                self.h5data_edit.setText(self.hdf5_data_path)
            applied.append(f"dataLoc={self.hdf5_data_path}")
        dl = get_str('darkLoc')
        if dl:
            self.hdf5_dark_path = dl.rstrip('/') or '/'
            if hasattr(self, 'h5dark_edit'):
                self.h5dark_edit.setText(self.hdf5_dark_path)
            applied.append(f"darkLoc={self.hdf5_dark_path}")

        # ── Dark file ──
        ds = get_str('DarkStem', 'Dark')
        if ds:
            if os.sep in ds or '/' in ds:
                self.dark_folder = os.path.dirname(ds) + '/'
                basename = os.path.basename(ds)
                parsed = _parse_numbered_filename(basename)
                if parsed:
                    self.dark_stem = parsed[0]
                    self.dark_num = parsed[1]
            else:
                self.dark_stem = ds
            self.dark_check.setChecked(True)
            applied.append(f"Dark={ds}")

        self.param_label.setText(os.path.basename(fn))
        print(f"Param file: {fn}")
        for entry in applied:
            print(f"  {entry}")

        self._load_and_display()
        if self.show_rings and self.ring_rads:
            self._draw_rings()

    # ── Callbacks ──────────────────────────────────────────────────

    def _on_log_toggled(self, checked):
        self.use_log = checked
        self.image_view.set_log_mode(checked)

    def _on_max_toggled(self, checked):
        if checked:
            self.sum_check.setChecked(False)
        self._load_and_display()

    def _on_sum_toggled(self, checked):
        if checked:
            self.max_check.setChecked(False)
        self._load_and_display()

    def _on_cmap_changed(self, name):
        self.colormap_name = name
        self.image_view.set_colormap(name)

    def _on_theme_changed(self, theme):
        self._theme = theme
        apply_theme(QtWidgets.QApplication.instance(), theme)

    def _on_font_changed(self, size):
        QtWidgets.QApplication.instance().setStyleSheet(f'* {{ font-size: {size}pt; }}')
        # pyqtgraph axes ignore Qt stylesheets — update tick fonts explicitly.
        font = QtGui.QFont('', int(size))
        if hasattr(self, 'image_view'):
            pg_iv = getattr(self.image_view, '_iv', None)
            view = getattr(pg_iv, 'view', None) if pg_iv is not None else None
            if view is not None and hasattr(view, 'getAxis'):
                for ax_name in ('left', 'bottom', 'right', 'top'):
                    try:
                        view.getAxis(ax_name).setTickFont(font)
                    except Exception:
                        pass
        # pyqtgraph TextItems also don't pick up the stylesheet — redraw axes.
        if self.show_axes:
            self._draw_axes()

    def _on_frame_scroll(self, delta):
        self.frame_spin.setValue(self.frame_spin.value() + delta)

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
            self._load_and_display()
        elif path.endswith('.zip'):
            self._load_zarr_zip(path)
        elif os.path.isfile(path):
            # Treat like FirstFile selection
            self.folder = os.path.dirname(path) + '/'
            basename = os.path.basename(path)
            parsed = _parse_numbered_filename(basename)
            if parsed:
                self.file_stem, self.first_file_nr, self.padding, self.ext = parsed
                self.file_nr_edit.setText(str(self.first_file_nr))
            self._load_and_display()

    def _on_cursor_moved(self, x, y, val):
        px = float(self.px_edit.text() or 200)
        bcy = float(self.bcy_edit.text() or 0)
        bcz = float(self.bcz_edit.text() or 0)
        try:
            eta, rr = CalcEtaAngleRad(-x + bcy, y - bcz)
            status = (f"x={x:.1f}  y={y:.1f}  I={val:.0f}  "
                      f"R={rr:.1f}px  η={eta:.1f}°")
            # Show nearest ring info
            if self.show_rings and self.ring_rads:
                r_um = rr * px  # convert R from pixels to μm
                best_i, best_diff = None, float('inf')
                for i, rad in enumerate(self.ring_rads):
                    diff = abs(r_um - rad)
                    if diff < best_diff:
                        best_diff = diff
                        best_i = i
                if best_i is not None:
                    nr = self.ring_nrs[best_i]
                    hkl = self.hkls[best_i]
                    status += f"  | Ring {nr} ({hkl[0]},{hkl[1]},{hkl[2]})"
            self.status_label.setText(status)
        except Exception:
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

    # ── Intensity vs Frame ─────────────────────────────────────────

    def _compute_ivf(self):
        """Sweep all frames and compute mean/max intensity."""
        self._ivf_compute_btn.setEnabled(False)
        self._ivf_compute_btn.setText("Computing...")
        self._ivf_dock.show()

        def _worker():
            means, maxs, frames = [], [], []
            self._sync_params()
            n = self.frame_spin.maximum() + 1
            for i in range(n):
                file_nr = self.first_file_nr + i // max(1, self.n_frames_per_file)
                frame_in = i % max(1, self.n_frames_per_file)
                fn = build_filename(self.folder, self.file_stem, file_nr,
                                    self.padding, self.det_nr, self.ext, self.sep_folder)
                try:
                    data = read_image(fn, self.header_size, self.bytes_per_pixel,
                                      self.ny, self.nz, frame_in,
                                      False, False, False,
                                      None, self.zarr_store, self.zarr_dark_mean,
                                      hdf5_data_path=self.hdf5_data_path,
                                      hdf5_dark_path=self.hdf5_dark_path)
                    means.append(float(np.mean(data)))
                    maxs.append(float(np.max(data)))
                    frames.append(i)
                except Exception:
                    break
            return frames, means, maxs

        worker = AsyncWorker(_worker)

        def _done(result):
            frames, means, maxs = result
            self._ivf_frames = frames
            self._ivf_means = means
            self._ivf_maxs = maxs
            self._ivf_mean_curve.setData(frames, means)
            self._ivf_max_curve.setData(frames, maxs)
            self._update_ivf_marker()
            self._ivf_compute_btn.setEnabled(True)
            self._ivf_compute_btn.setText("Compute")
            print(f"IvF: computed {len(frames)} frames")

        worker.finished.connect(_done)
        worker.start()
        # Keep reference to prevent GC
        self._ivf_worker = worker

    def _on_ivf_clicked(self, ev):
        """Click on intensity-vs-frame plot to jump to that frame."""
        if not self._ivf_frames:
            return
        vb = self._ivf_plot.plotItem.vb
        pos = vb.mapSceneToView(ev.scenePos())
        frame = int(round(pos.x()))
        frame = max(0, min(frame, self.frame_spin.maximum()))
        self.frame_spin.setValue(frame)
        self._update_ivf_marker()

    def _update_ivf_marker(self):
        """Update yellow marker on IvF plot to current frame."""
        if not self._ivf_frames:
            return
        f = self.frame_spin.value()
        if f < len(self._ivf_means):
            self._ivf_marker.setData([f], [self._ivf_means[f]])

    def _on_rings_toggled(self, checked):
        self.show_rings = checked
        if checked:
            if self.ring_rads is None:
                self._on_ring_selection()
            else:
                self._draw_rings()
        else:
            self.image_view.clear_overlays('rings')

    def _on_axes_toggled(self, checked):
        self.show_axes = checked
        if checked:
            self._draw_axes()
        else:
            self.image_view.clear_overlays('axes')

    # ── Multi-detector callbacks ───────────────────────────────────

    def _on_detector_mode_changed(self, index):
        """Switch between single-detector and multi-detector data sources."""
        checked = index > 0
        if checked and not self.multi_mode:
            # Save single-mode state so we can restore on toggle-off.
            self._single_mode_state = dict(
                bcy=self.bcy_edit.text(),
                bcz=self.bcz_edit.text(),
                ny=self.ny_edit.text(),
                nz=self.nz_edit.text(),
                tx=self.tx_edit.text(),
            )
        elif not checked and self.multi_mode:
            saved = getattr(self, '_single_mode_state', None)
            if saved:
                self.bcy_edit.setText(saved['bcy'])
                self.bcz_edit.setText(saved['bcz'])
                self.ny_edit.setText(saved['ny'])
                self.nz_edit.setText(saved['nz'])
                self.tx_edit.setText(saved['tx'])

        self.multi_mode = checked
        self._file_stack.setCurrentIndex(1 if checked else 0)
        self.composite_combo.setEnabled(checked)
        if checked:
            # Composite center = (BigDetSize/2, BigDetSize/2). Push that into
            # the BC fields so rings, lab axes, cursor R/η work in the lab
            # frame. Tx is identity (composite is already in lab frame).
            self.tx_edit.setText("0")
            self._update_bc_for_multi()
        # Either direction: trigger a fresh display.
        self._load_and_display()

    def _on_multi_paths_changed(self):
        """Propagate the shared data/dark path fields to all DetectorStates."""
        data_path = self._multi_data_path_edit.text().strip() or '/exchange/data'
        dark_path = self._multi_dark_path_edit.text().strip() or '/exchange/data_dark'
        for s in self._det_states:
            s.data_loc = data_path
            s.dark_loc = dark_path
            s._dark_image = None
        if self.multi_mode:
            self._load_and_display()

    def _on_composite_op_changed(self, text):
        self.composite_op = text
        if self.multi_mode:
            self._load_and_display()

    def _on_bigdet_changed(self, value):
        self.big_det_size = int(value)
        self._big_det_auto = False
        # Geometry changed → invalidate every detector's cached coord map.
        for s in self._det_states:
            s._inv_coords = None
            s._inv_cache_key = ()
        self._update_bc_for_multi()
        if self.multi_mode:
            self._load_and_display()

    def _update_bc_for_multi(self):
        """When in multi-mode, BC of the displayed composite is its center."""
        half = self.big_det_size / 2.0
        self.bcy_edit.setText(str(half))
        self.bcz_edit.setText(str(half))
        self.ny_edit.setText(str(self.big_det_size))
        self.nz_edit.setText(str(self.big_det_size))
        self.ny = self.nz = self.big_det_size
        self.bc_local = [half, half]
        self.lsd_orig = self.lsd_local

    def _on_det_enabled(self, idx, checked):
        self._det_states[idx].enabled = bool(checked)
        if self.multi_mode:
            self._load_and_display()

    def _autofill_siblings(self, src_idx, seed_path, setter):
        """If Auto-fill siblings is on, derive sibling GE paths from seed_path
        and call setter(tgt_idx, path) for each sibling that exists on disk.

        Uses smart tag detection first; falls back to direct ge<N> substitution
        if the smart detection can't find >=2 siblings.
        """
        if not getattr(self, '_autofill_check', None) or not self._autofill_check.isChecked():
            return

        tag = self._find_detector_tag(seed_path)
        if tag is not None:
            prefix, src_digit = tag
        else:
            # Fallback: try common GE prefixes with the source detector number.
            src_digit = str(src_idx + 1)
            prefix = None
            for pfx in ('ge', 'GE', 'det', 'Det', 'panel'):
                if (pfx + src_digit) in seed_path or (pfx + src_digit).upper() in seed_path.upper():
                    prefix = pfx
                    break
            if prefix is None:
                self._autoload_status.setText("Auto-fill: no detector tag found in path")
                return

        src_label = prefix + src_digit
        filled, missing = [], []
        for d in '1234':
            tgt_idx = int(d) - 1
            if tgt_idx == src_idx:
                continue
            tgt_label = prefix + d
            sib = self._derive_ge_path(seed_path, src_label, tgt_label)
            if sib and os.path.exists(sib):
                setter(tgt_idx, sib)
                self._refresh_det_widget(tgt_idx)
                filled.append(f"GE{tgt_idx+1}")
            else:
                missing.append(f"GE{tgt_idx+1}")

        parts = []
        if filled:
            parts.append(f"filled: {', '.join(filled)}")
        if missing:
            parts.append(f"not found: {', '.join(missing)}")
        self._autoload_status.setText("Auto-fill — " + "; ".join(parts) if parts else "Auto-fill: no siblings found")

    def _on_pick_det_data(self, idx):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Select GE{idx+1} data HDF5",
            os.path.dirname(self._det_states[idx].data_file) or os.getcwd(),
            "HDF5 (*.h5 *.hdf5 *.hdf *.nxs);;All (*)")
        if not fn:
            return
        self._det_states[idx].data_file = fn
        self._det_states[idx]._dark_image = None
        self._refresh_det_widget(idx)
        def _set_data(i, path):
            self._det_states[i].data_file = path
            self._det_states[i]._dark_image = None
        self._autofill_siblings(idx, fn, _set_data)
        if self.multi_mode:
            self._load_and_display()

    def _on_pick_det_dark(self, idx):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Select GE{idx+1} dark HDF5",
            os.path.dirname(self._det_states[idx].dark_file
                             or self._det_states[idx].data_file) or os.getcwd(),
            "HDF5 (*.h5 *.hdf5 *.hdf *.nxs);;All (*)")
        if not fn:
            return
        self._det_states[idx].dark_file = fn
        self._det_states[idx]._dark_image = None
        self._refresh_det_widget(idx)
        def _set_dark(i, path):
            self._det_states[i].dark_file = path
            self._det_states[i]._dark_image = None
        self._autofill_siblings(idx, fn, _set_dark)
        if self.multi_mode:
            self._load_and_display()

    def _on_clear_det_dark(self, idx):
        self._det_states[idx].dark_file = ''
        self._det_states[idx]._dark_image = None
        self._refresh_det_widget(idx)
        if self.multi_mode:
            self._load_and_display()

    def _on_pick_det_param(self, idx):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Select GE{idx+1} parameter file",
            os.path.dirname(self._det_states[idx].param_file) or os.getcwd(),
            "Param Files (*.txt);;All (*)")
        if not fn:
            return
        try:
            params = self._det_states[idx].load_param_file(fn)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Param read failed", f"GE{idx+1}: {e}")
            return
        # First detector loaded → import shared crystallography params.
        self._absorb_shared_params(params)
        def _set_param(i, path):
            try:
                self._det_states[i].load_param_file(path)
            except Exception:
                pass
        self._autofill_siblings(idx, fn, _set_param)
        # Auto-pick BigDetSize on first param file load.
        if self._big_det_auto:
            new_size = _md.autopick_big_det_size(self._det_states)
            if new_size != self.big_det_size:
                self.big_det_size = new_size
                # block the spin's signal so we don't re-trigger _on_bigdet_changed
                self.bigdet_spin.blockSignals(True)
                self.bigdet_spin.setValue(new_size)
                self.bigdet_spin.blockSignals(False)
        self._update_bc_for_multi()
        self._refresh_det_widget(idx)
        if self.multi_mode:
            self._load_and_display()

    # ── Autoload: one param file → all 4 detector slots ─────────────

    @staticmethod
    def _derive_ge_path(path, src_label, tgt_label):
        """Replace `src_label` (e.g. 'ge1', 'det2') with `tgt_label` in every
        component of `path`. Returns the derived path, or None when no
        substitution actually happened."""
        from pathlib import Path as _Path
        p = _Path(path)
        new_parts = [
            re.sub(re.escape(src_label), tgt_label, part, flags=re.IGNORECASE)
            for part in p.parts
        ]
        candidate = str(_Path(*new_parts))
        return None if candidate == str(p) else candidate

    @staticmethod
    def _find_detector_tag(seed_path):
        """Detect the detector-index tag in `seed_path`.

        A tag is a substring like 'ge1', 'det1', 'panel1' — an alphabetic
        prefix followed by a single digit 1-4 (not adjacent to other digits).
        Identified by trying each candidate tag's substitutions with
        digits 1/2/3/4 and picking the one with the most existing siblings.

        Returns (prefix, source_digit) — e.g. ('ge', '1'), ('det', '2') —
        or None if no robust tag with ≥2 sibling matches is found.
        """
        # Candidate prefixes: scan for `<alpha>+<1-4>` patterns where the
        # digit isn't part of a longer numeric run. For each match, also try
        # shorter prefixes (e.g. for 'panel1' also try 'el1', 'l1') so we
        # don't get fooled by accidentally-long alphabetic runs.
        candidates = set()
        for m in re.finditer(r'([A-Za-z]+)([1-4])(?![0-9])', seed_path):
            prefix = m.group(1)
            digit = m.group(2)
            # Take 1..min(len, 6) trailing chars as candidate prefixes.
            for n in range(1, min(len(prefix), 6) + 1):
                candidates.add((prefix[-n:], digit))

        best = None
        best_count = 0
        for prefix, digit in candidates:
            src_tag = prefix + digit
            count = 0
            for d in '1234':
                tgt_tag = prefix + d
                if d == digit:
                    if os.path.exists(seed_path):
                        count += 1
                    continue
                sib = FFViewer._derive_ge_path(seed_path, src_tag, tgt_tag)
                if sib and os.path.exists(sib):
                    count += 1
            # Prefer higher sibling count; on tie prefer the longer prefix
            # (more specific, less likely to be a false positive substring).
            score = (count, len(prefix))
            best_score = (best_count, len(best[0])) if best else (0, 0)
            if score > best_score:
                best_count = count
                best = (prefix, digit)
        return best if best_count >= 2 else None

    def _on_autoload_multi(self):
        """Pick one param file; auto-fill all 4 detector slots from it.

        Auto-detects the detector tag (e.g. ``ge1``, ``det1``, ``panel1``) by
        finding which substring of the seed path, when its trailing digit
        is substituted with 1-4, yields the most existing sibling files.
        Then for each detector, parses its param file for
        Folder/FileStem/StartNr/Padding/Ext, builds the expected data file
        path, and populates that slot if the file exists. Slots whose param
        or data files don't exist stay empty.
        """
        seed_fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select any one detector's param file (autoload all 4)",
            os.getcwd(), "Param Files (*.txt);;All (*)")
        if not seed_fn:
            return

        tag = self._find_detector_tag(seed_fn)
        if tag is None:
            QtWidgets.QMessageBox.warning(
                self, "Autoload",
                "Could not detect a detector tag (e.g. ge1, det1, panel1) in\n"
                "the selected param file's path that has ≥2 sibling files.\n\n"
                "Use the per-slot pickers in the cards below instead, or\n"
                "rename your files so the detector index is in the path.")
            return
        prefix, src_digit = tag
        src_idx = int(src_digit) - 1
        ge_labels = tuple(prefix + str(i) for i in range(1, 5))
        print(f"Autoload: detected detector tag '{prefix}<N>' (seed has '{prefix}{src_digit}')")

        found, missing = [], []
        first_params = None
        src_label = ge_labels[src_idx]
        for tgt_idx, tgt_label in enumerate(ge_labels):
            if tgt_idx == src_idx:
                pf = seed_fn
            else:
                pf = self._derive_ge_path(seed_fn, src_label, tgt_label)

            # Reset slot to a clean state before populating.
            self._det_states[tgt_idx] = _md.DetectorState()
            if not pf or not os.path.exists(pf):
                missing.append((tgt_label, "param not found"))
                self._refresh_det_widget(tgt_idx)
                continue

            s = self._det_states[tgt_idx]
            try:
                params = s.load_param_file(pf)
            except Exception as e:
                missing.append((tgt_label, f"param parse failed: {e}"))
                self._refresh_det_widget(tgt_idx)
                continue
            if first_params is None:
                first_params = params

            # Build the data file path from the param file's contents.
            raw = self._parse_param_file(pf)
            def _get_str(k):
                v = raw.get(k)
                return v[0][0] if v and v[0] else None
            def _get_int(k):
                v = _get_str(k)
                try: return int(float(v)) if v is not None else None
                except (ValueError, TypeError): return None

            folder = _get_str('Folder') or _get_str('RawFolder')
            stem = _get_str('FileStem')
            start_nr = _get_int('StartNr') or _get_int('StartFileNrFirstLayer')
            padding = _get_int('Padding') or 6
            ext = _get_str('Ext') or '.h5'
            if not (folder and stem and start_nr is not None):
                missing.append((tgt_label, "param missing Folder/FileStem/StartNr"))
                self._refresh_det_widget(tgt_idx)
                continue
            if not ext.startswith('.'):
                ext = '.' + ext
            data_fn = os.path.join(folder, f"{stem}_{str(start_nr).zfill(padding)}{ext}")
            if not os.path.exists(data_fn):
                missing.append((tgt_label, f"data file not found: {os.path.basename(data_fn)}"))
                self._refresh_det_widget(tgt_idx)
                continue

            s.data_file = data_fn
            self._refresh_det_widget(tgt_idx)
            found.append(tgt_label)

        # Pull shared crystallography params from whichever param loaded first.
        if first_params is not None:
            self._absorb_shared_params(first_params)

        # Auto-pick BigDetSize from the loaded detectors.
        if self._big_det_auto:
            new_size = _md.autopick_big_det_size(self._det_states)
            if new_size != self.big_det_size:
                self.big_det_size = new_size
                self.bigdet_spin.blockSignals(True)
                self.bigdet_spin.setValue(new_size)
                self.bigdet_spin.blockSignals(False)
        self._update_bc_for_multi()

        # Status line + console summary.
        status_bits = []
        if found:
            status_bits.append(f"loaded: {', '.join(found)}")
        if missing:
            status_bits.append(f"skipped: {', '.join(lbl for lbl, _ in missing)}")
        self._autoload_status.setText("  ·  ".join(status_bits))
        print(f"Autoload from {os.path.basename(seed_fn)}:")
        for lbl in found:
            print(f"  {lbl}: ✓")
        for lbl, why in missing:
            print(f"  {lbl}: skipped — {why}")

        if not found:
            QtWidgets.QMessageBox.warning(
                self, "Autoload",
                "No detector files were resolved. Check that the param-file\n"
                "path contains a ge1/ge2/ge3/ge4 tag and that the per-detector\n"
                "param files actually exist alongside the seed.")
            return

        # Auto-engage Multi-Det mode and trigger a composite.
        if not self.multi_mode:
            self.detector_mode_combo.setCurrentIndex(1)
        else:
            self._load_and_display()

    def _absorb_shared_params(self, params):
        """Pull crystallography + pixel params from a per-detector file
        into the global GUI state (used for ring computation)."""
        # Sync shared HDF5 path fields from the first param file loaded.
        if hasattr(self, '_multi_data_path_edit') and params.get('data_loc'):
            self._multi_data_path_edit.setText(params['data_loc'])
        if hasattr(self, '_multi_dark_path_edit') and params.get('dark_loc'):
            self._multi_dark_path_edit.setText(params['dark_loc'])
        if params.get('px') is not None:
            self.pixel_size = params['px']
            self.px_edit.setText(str(params['px']))
        if params.get('lsd') is not None:
            self.lsd_local = params['lsd']
            self.lsd_orig  = params['lsd']
            self.lsd_edit.setText(str(params['lsd']))
        if params.get('wavelength') is not None:
            self.wl = params['wavelength']
        if params.get('space_group') is not None:
            self.sg = params['space_group']
        if params.get('lattice_constant') is not None:
            self.lattice_const = params['lattice_constant']
        if params.get('max_ring_rad') is not None:
            self.temp_max_ring_rad = params['max_ring_rad']

    def _refresh_det_widget(self, idx):
        s = self._det_states[idx]
        w = self._det_widgets[idx]
        w['data_lbl'].setText(os.path.basename(s.data_file) if s.data_file else "(none)")
        w['dark_lbl'].setText(os.path.basename(s.dark_file) if s.dark_file else "(same)")
        w['param_lbl'].setText(os.path.basename(s.param_file) if s.param_file else "(none)")
        bits = []
        if s.param_file:
            bits.append(f"BC=({s.bc_y:.1f},{s.bc_z:.1f}) tx={s.tx:g}°")
            if s.im_trans_opts:
                bits.append(f"ImTransOpt={s.im_trans_opts}")
        if s.data_file:
            nf = s.n_frames()
            if nf:
                bits.append(f"{nf} frames @ {s.data_loc}")
        w['status_lbl'].setText("  ".join(bits))

    def _on_save_composite(self):
        """Save the most recent composite array as a TIFF."""
        if self.bdata is None:
            QtWidgets.QMessageBox.information(
                self, "Nothing to save", "Load detectors and a frame first.")
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save composite frame",
            f"composite_frame{self.frame_nr}.tif",
            "TIFF (*.tif *.tiff)")
        if not fn:
            return
        if not fn.lower().endswith(('.tif', '.tiff')):
            fn += '.tif'
        try:
            if tifffile is None:
                raise ImportError("tifffile not installed")
            tifffile.imwrite(fn, self.bdata.astype(np.float32))
            print(f"Composite saved: {fn}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))

    def _on_first_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select First File")
        if not fn:
            return
        check_fn = fn[:-4] if fn.endswith('.bz2') else fn
        basename = os.path.basename(check_fn)
        parsed = _parse_numbered_filename(basename)
        if parsed is None:
            print(f"Warning: could not parse numbered filename: {basename}")
            return
        self.file_stem, self.first_file_nr, self.padding, self.ext = parsed
        self.folder = os.path.dirname(fn) + '/'
        self.file_nr_edit.setText(str(self.first_file_nr))
        self.det_nr = -1
        if self.ext.startswith('ge') and len(self.ext) == 3 and self.ext[-1].isdigit():
            self.det_nr = int(self.ext[-1])
        # Try HDF5 dimension detection
        ext_lower = os.path.splitext(check_fn)[1].lower()
        if ext_lower in ['.h5', '.hdf', '.hdf5', '.nxs'] and h5py:
            self._detect_hdf5_dims(fn)
        print(f"Loaded: stem={self.file_stem}, folder={self.folder}, ext={self.ext}")
        self._load_and_display()

    def _on_dark_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Dark File")
        if not fn:
            return
        self.dark_folder = os.path.dirname(fn) + '/'
        basename = os.path.basename(fn)
        parsed = _parse_numbered_filename(basename)
        if parsed:
            self.dark_stem = parsed[0]
            self.dark_num = parsed[1]
        self.dark_check.setChecked(True)
        print(f"Dark: {fn}")

    def _on_load_zip(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select MIDAS Zarr-ZIP", "", "ZIP Files (*.zip);;All (*)")
        if fn:
            self._load_zarr_zip(fn)

    def _detect_hdf5_dims(self, fn):
        try:
            with h5py.File(fn, 'r') as f:
                dpath = self.hdf5_data_path
                if dpath in f:
                    shape = f[dpath].shape
                    if len(shape) == 3:
                        self.n_frames_per_file = shape[0]
                        self.ny = shape[1]; self.nz = shape[2]
                    elif len(shape) == 2:
                        self.ny = shape[0]; self.nz = shape[1]
                    self.ny_edit.setText(str(self.ny))
                    self.nz_edit.setText(str(self.nz))
                    self.nframes_edit.setText(str(self.n_frames_per_file))
                    self.frame_spin.setMaximum(self.n_frames_per_file - 1)
        except Exception as e:
            print(f"HDF5 detect error: {e}")

    def _on_browse_mask(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Mask File (TIFF)", os.getcwd(),
            "TIFF Files (*.tif *.tiff);;All Files (*)")
        if fn:
            self.mask_edit.setText(fn)
            self.mask_check.setChecked(True)

    def _load_zarr_zip(self, zip_path):
        if zarr is None:
            print("zarr not installed")
            return
        try:
            zr = zarr.open(zip_path, 'r')
        except Exception as e:
            print(f"Error: {e}"); return
        self.zarr_zip_path = zip_path
        self.zarr_store = zr
        if 'exchange/data' in zr:
            s = zr['exchange/data'].shape
            self.n_frames_per_file = s[0]; self.ny = s[1]; self.nz = s[2]
            self.ny_edit.setText(str(self.ny)); self.nz_edit.setText(str(self.nz))
            self.nframes_edit.setText(str(self.n_frames_per_file))
            self.frame_spin.setMaximum(self.n_frames_per_file - 1)
            self.max_frames_spin.setValue(self.n_frames_per_file)
        if 'exchange/dark' in zr:
            dk = zr['exchange/dark'][:]
            if dk.ndim == 3 and dk.shape[0] > 0 and np.any(dk):
                self.zarr_dark_mean = np.mean(dk, axis=0).astype(float)
                self.dark_check.setChecked(True)
        # Read params
        pp = 'analysis/process/analysis_parameters'
        if pp in zr:
            p = zr[pp]
            if 'Lsd' in p:
                v = float(p['Lsd'][0])
                self.lsd_local = v; self.lsd_orig = v
                self.lsd_edit.setText(str(v))
            if 'YCen' in p and 'ZCen' in p:
                self.bc_local = [float(p['YCen'][0]), float(p['ZCen'][0])]
                self.bcy_edit.setText(str(self.bc_local[0]))
                self.bcz_edit.setText(str(self.bc_local[1]))
            if 'tx' in p:
                self.tx_local = float(p['tx'][0])
                self.tx_edit.setText(str(self.tx_local))
            if 'PixelSize' in p:
                self.pixel_size = float(p['PixelSize'][0])
                self.px_edit.setText(str(self.pixel_size))
            if 'Wavelength' in p:
                self.wl = float(p['Wavelength'][0])
                print(f"  Wavelength: {self.wl:.6f} Å")
            if 'SpaceGroup' in p:
                self.sg = int(p['SpaceGroup'][0])
                print(f"  SpaceGroup: {self.sg}")
            lc_key = ('LatticeParameter' if 'LatticeParameter' in p
                       else 'LatticeConstant' if 'LatticeConstant' in p
                       else None)
            if lc_key:
                lc = p[lc_key][:]
                if len(lc) >= 6:
                    self.lattice_const = [float(x) for x in lc[:6]]
                    print(f"  LatticeConstant: {self.lattice_const}")
            if 'ImTransOpt' in p:
                opts = [int(x) for x in p['ImTransOpt'][:]]
                self.hflip_check.setChecked(1 in opts)
                self.vflip_check.setChecked(2 in opts)
                self.transpose_check.setChecked(3 in opts)
                if opts:
                    print(f"  ImTransOpt: {opts}")

        print(f"Loaded ZIP: {zip_path}")
        self._load_and_display()

    # ── Load & Display ─────────────────────────────────────────────

    def _sync_params(self):
        """Read current values from UI widgets."""
        try:
            self.ny = int(self.ny_edit.text())
            self.nz = int(self.nz_edit.text())
            self.header_size = int(self.header_edit.text())
            self.bytes_per_pixel = int(self.bpp_edit.text())
            self.pixel_size = float(self.px_edit.text())
            self.hflip = self.hflip_check.isChecked()
            self.vflip = self.vflip_check.isChecked()
            self.do_transpose = self.transpose_check.isChecked()
            self.use_dark = self.dark_check.isChecked()
            self.apply_mask = self.mask_check.isChecked()
            self.frame_nr = self.frame_spin.value()
            self.first_file_nr = int(self.file_nr_edit.text())
            self.n_frames_per_file = int(self.nframes_edit.text())
            self.lsd_local = float(self.lsd_edit.text())
            self.bc_local = [float(self.bcy_edit.text()), float(self.bcz_edit.text())]
            try:
                self.tx_local = float(self.tx_edit.text())
            except (ValueError, AttributeError):
                self.tx_local = 0.0
            self.hdf5_data_path = self.h5data_edit.text()
            self.hdf5_dark_path = self.h5dark_edit.text()
        except ValueError:
            pass

    @staticmethod
    def _rotate_image(data, tx_deg, bc):
        """Rotate `data` around `bc` by -tx_deg with an expanded output canvas.

        Returns ``(rotated, (r_min, c_min))``. The output is sized to the tight
        bounding box of the rotated input so no pixels are clipped, even when
        BC sits outside the original detector area. The ``(r_min, c_min)``
        shift tells callers where to position the image item in scene coords:
        place the image at scene rect ``(c_min, r_min, new_W, new_H)`` and
        all other overlays (rings, lab axes, cursor) remain in the original
        scene coordinate system.

        ``bc = [col, row]`` (MIDAS Y, Z convention).

        No-op (returns ``data`` unchanged with shift=(0,0)) when |tx_deg| is
        ~0 or when scipy is unavailable.
        """
        if data is None or abs(float(tx_deg)) < 1e-9:
            return data, (0, 0)
        try:
            from scipy.ndimage import affine_transform
        except ImportError:
            print("scipy not installed — cannot apply Tx rotation")
            return data, (0, 0)
        H, W = data.shape[:2]
        theta = -float(tx_deg) * deg2rad
        c, s = math.cos(theta), math.sin(theta)
        bc_i = float(bc[1])   # row (Z)
        bc_j = float(bc[0])   # col (Y)
        M = np.array([[c, -s], [s, c]])
        center = np.array([bc_i, bc_j])

        # Forward-map the 4 input corners → tight rotated bounding box.
        # Forward map (input → output) is M.T @ (input − center) + center.
        corners = np.array([[0, 0], [0, W - 1],
                            [H - 1, 0], [H - 1, W - 1]], dtype=float)
        corners_out = (M.T @ (corners - center).T).T + center
        r_min = int(math.floor(corners_out[:, 0].min()))
        c_min = int(math.floor(corners_out[:, 1].min()))
        r_max = int(math.ceil(corners_out[:, 0].max()))
        c_max = int(math.ceil(corners_out[:, 1].max()))
        new_H = r_max - r_min + 1
        new_W = c_max - c_min + 1

        # Shift the inverse-map offset so the bounding box aligns at output(0,0).
        shift = np.array([r_min, c_min], dtype=float)
        offset = center - M @ (center - shift)
        rotated = affine_transform(data, M, offset=offset,
                                    output_shape=(new_H, new_W),
                                    order=1, mode='constant', cval=0.0)
        return rotated, (r_min, c_min)

    def _apply_tx_rotation(self, data):
        """Rotate ``data`` using the viewer's Tx field and BC, and remember
        the canvas shift so the display path can position the image item
        at the correct scene rectangle (so rings/axes/cursor stay aligned)."""
        rotated, self._tx_shift = self._rotate_image(
            data, self.tx_local, self.bc_local)
        return rotated

    def _apply_tx_image_rect(self, data):
        """Position the displayed image at the correct scene rectangle so
        the expanded rotated canvas lines up with the original-coords
        BC, rings overlay, lab-axes overlay, and cursor R/η.

        For Tx=0 (no rotation) this resets the rect to (0, 0, W, H), the
        default image-item position.
        """
        if data is None:
            return
        H, W = data.shape[:2]
        r_min, c_min = self._tx_shift
        # data here is already the rotated/expanded array; its shape is the
        # bounding-box size (new_H, new_W). The scene rectangle is the
        # bounding-box in original-scene coords: (c_min, r_min) to
        # (c_min + W, r_min + H).
        try:
            self.image_view.set_image_rect(c_min, r_min, W, H)
        except Exception as e:
            print(f"set_image_rect failed: {e}")

    def _load_and_display_multi(self):
        """Composite the 4 detector frames into one BigDet array and display."""
        states = self._det_states
        loaded = [s for s in states if s.enabled and s.data_file and s.param_file]
        if not loaded:
            self.frame_label.setText("Multi-Det: no detectors loaded")
            return

        # Use the smallest n_frames across loaded detectors as the upper bound;
        # don't crash if the user is mid-load with mismatched datasets.
        nf_per = [s.n_frames() for s in loaded]
        nf = min(n for n in nf_per if n > 0) if any(n > 0 for n in nf_per) else 0
        if nf > 0 and self.frame_spin.maximum() != nf - 1:
            self.frame_spin.setMaximum(max(nf - 1, 0))

        frame_idx = self.frame_nr
        bds = int(self.big_det_size)
        px = float(self.pixel_size)
        op = self.composite_op

        self.frame_label.setText(
            f"Frame {frame_idx}  |  multi-det compositing ({len(loaded)})…")

        def _worker():
            import time as _time
            t0 = _time.monotonic()
            data = _md.composite_frame(states, frame_idx, bds, px,
                                        op=op, subtract_dark=True,
                                        parallel=True)
            return data, _time.monotonic() - t0

        worker = AsyncWorker(target=_worker)

        def _done(result):
            data, elapsed = result
            data = self._apply_tx_rotation(data)   # respects single Tx field
            self.bdata = data
            if getattr(self, '_levels_initialized', False):
                try:
                    lo = float(self.min_intensity_edit.text())
                    hi = float(self.max_intensity_edit.text())
                    self.image_view.set_image_data(data, auto_levels=False,
                                                    levels=(lo, hi))
                except ValueError:
                    self.image_view.set_image_data(data)
            else:
                self.image_view.set_image_data(data)
            self._apply_tx_image_rect(data)
            self.frame_label.setText(
                f"Frame {frame_idx}  |  composite (op={op}, "
                f"{len(loaded)} det, {bds}², {1000*elapsed:.0f} ms)")
            self.setWindowTitle(
                f"FF Viewer — Multi-Det composite [frame {frame_idx}]")
            if self.show_rings and self.ring_rads:
                self._draw_rings()
            if self.show_axes:
                self._draw_axes()

        def _err(msg):
            self.frame_label.setText(f"Multi-Det error: {msg}")
            print(f"Multi-Det error: {msg}")

        worker.finished_signal.connect(_done)
        worker.error_signal.connect(_err)
        worker.start()
        self._multi_worker = worker  # prevent GC

    def _load_and_display(self):
        """Load current frame and display."""
        self._sync_params()

        # Multi-detector mode: composite the 4 detector frames into one image
        # in a worker thread, then go through the standard display path.
        if self.multi_mode:
            self._load_and_display_multi()
            return

        # Mask
        mask = None
        if self.apply_mask and self.mask_edit.text():
            mask = read_mask(self.mask_edit.text(), self.ny, self.nz,
                             self.do_transpose, self.hflip, self.vflip)

        # Dark subtraction
        dark_data = None
        if self.use_dark and not self.zarr_store:
            dark_folder = self.dark_folder if self.dark_folder else self.folder
            dark_fn = build_filename(dark_folder, self.dark_stem, self.dark_num,
                                     self.padding, self.det_nr, self.ext)
            if os.path.exists(dark_fn):
                ext_lower = os.path.splitext(dark_fn)[1].lower()
                if ext_lower in ['.h5', '.hdf', '.hdf5', '.nxs'] and h5py:
                    # HDF5 dark: average all frames in the dark dataset
                    try:
                        with h5py.File(dark_fn, 'r') as f:
                            dpath = self.hdf5_dark_path
                            if dpath in f:
                                dset = f[dpath]
                                if dset.ndim == 3:
                                    dark_data = np.mean(dset[:], axis=0).astype(float)
                                else:
                                    dark_data = dset[:].astype(float)
                                if self.do_transpose:
                                    dark_data = np.transpose(dark_data)
                                if self.hflip and self.vflip:
                                    dark_data = dark_data[::-1, ::-1].copy()
                                elif self.hflip:
                                    dark_data = dark_data[::-1, :].copy()
                                elif self.vflip:
                                    dark_data = dark_data[:, ::-1].copy()
                    except Exception as e:
                        print(f"Error reading HDF5 dark: {e}")
                else:
                    dark_data = read_image(dark_fn, self.header_size, self.bytes_per_pixel,
                                           self.ny, self.nz, 0,
                                           self.do_transpose, self.hflip, self.vflip)

        # MaxOverFrames / SumOverFrames — parallel computation
        if self.max_check.isChecked() or self.sum_check.isChecked():
            n_accum = self.max_frames_spin.value()
            use_sum = self.sum_check.isChecked()
            start_frame = self.frame_nr
            mode_str = "Sum" if use_sum else "Max"

            # Disable controls while computing
            self.max_check.setEnabled(False)
            self.sum_check.setEnabled(False)
            self.frame_label.setText(f"Computing {mode_str} over {n_accum} frames...")

            # Capture all parameters for the worker thread
            params = dict(
                n_accum=n_accum, use_sum=use_sum, start_frame=start_frame,
                folder=self.folder, file_stem=self.file_stem,
                first_file_nr=self.first_file_nr, padding=self.padding,
                det_nr=self.det_nr, ext=self.ext, sep_folder=self.sep_folder,
                header_size=self.header_size, bytes_per_pixel=self.bytes_per_pixel,
                ny=self.ny, nz=self.nz,
                do_transpose=self.do_transpose, hflip=self.hflip, vflip=self.vflip,
                mask=mask, dark_data=dark_data,
                zarr_store=self.zarr_store, zarr_dark_mean=self.zarr_dark_mean,
                hdf5_data_path=self.hdf5_data_path,
                hdf5_dark_path=self.hdf5_dark_path,
                n_frames_per_file=self.n_frames_per_file,
            )

            def _parallel_accum():
                import concurrent.futures, time as _time
                t0 = _time.monotonic()
                p = params

                # ── Fast path: Zarr slab read ──
                if p['zarr_store'] is not None and 'exchange/data' in p['zarr_store']:
                    dset = p['zarr_store']['exchange/data']
                    end_frame = min(p['start_frame'] + p['n_accum'], dset.shape[0])
                    slab = dset[p['start_frame']:end_frame, :, :]  # (N, ny, nz)
                    if p['use_sum']:
                        result = np.sum(slab.astype(np.float64), axis=0)
                    else:
                        result = np.max(slab.astype(np.float64), axis=0)
                    # Apply same transforms as read_image for zarr
                    result = result[::-1, ::-1].copy()
                    if p['do_transpose']: result = np.transpose(result)
                    if p['hflip'] and p['vflip']: result = result[::-1, ::-1].copy()
                    elif p['hflip']: result = result[::-1, :].copy()
                    elif p['vflip']: result = result[:, ::-1].copy()
                    if p['mask'] is not None and p['mask'].shape == result.shape:
                        result[p['mask'] == 1] = 0
                    if p['dark_data'] is not None:
                        result = np.maximum(result - p['dark_data'], 0)
                    elapsed = _time.monotonic() - t0
                    return result, end_frame - p['start_frame'], elapsed

                # ── Fast path: single HDF5 file with slab read ──
                n_fpf = max(1, p['n_frames_per_file'])
                first_file = p['first_file_nr'] + p['start_frame'] // n_fpf
                last_file = p['first_file_nr'] + (p['start_frame'] + p['n_accum'] - 1) // n_fpf
                fn0 = build_filename(p['folder'], p['file_stem'], first_file,
                                     p['padding'], p['det_nr'], p['ext'], p['sep_folder'])
                ext_lower = os.path.splitext(fn0)[1].lower()
                if (first_file == last_file and ext_lower in ['.h5', '.hdf', '.hdf5', '.nxs']
                        and h5py and os.path.exists(fn0)):
                    try:
                        with h5py.File(fn0, 'r') as f:
                            dp = p['hdf5_data_path']
                            if dp in f and f[dp].ndim == 3:
                                dset = f[dp]
                                f_start = p['start_frame'] % n_fpf
                                f_end = min(f_start + p['n_accum'], dset.shape[0])
                                slab = dset[f_start:f_end, :, :].astype(np.float64)
                                if p['use_sum']:
                                    result = np.sum(slab, axis=0)
                                else:
                                    result = np.max(slab, axis=0)
                                if p['do_transpose']: result = np.transpose(result)
                                if p['hflip'] and p['vflip']: result = result[::-1, ::-1].copy()
                                elif p['hflip']: result = result[::-1, :].copy()
                                elif p['vflip']: result = result[:, ::-1].copy()
                                if p['mask'] is not None and p['mask'].shape == result.shape:
                                    result[p['mask'] == 1] = 0
                                if p['dark_data'] is not None:
                                    result = np.maximum(result - p['dark_data'], 0)
                                elapsed = _time.monotonic() - t0
                                return result, f_end - f_start, elapsed
                    except Exception as e:
                        print(f"HDF5 slab read failed, falling back to parallel: {e}")

                # ── General path: ThreadPoolExecutor for raw/TIFF/multi-file ──
                def _read_one(frame_idx):
                    fr = p['start_frame'] + frame_idx
                    f_nr = p['first_file_nr'] + fr // n_fpf
                    f_in = fr % n_fpf
                    fn = build_filename(p['folder'], p['file_stem'], f_nr,
                                        p['padding'], p['det_nr'], p['ext'],
                                        p['sep_folder'])
                    return read_image(
                        fn, p['header_size'], p['bytes_per_pixel'],
                        p['ny'], p['nz'], f_in,
                        p['do_transpose'], p['hflip'], p['vflip'],
                        p['mask'], p['zarr_store'], p['zarr_dark_mean'],
                        hdf5_data_path=p['hdf5_data_path'],
                        hdf5_dark_path=p['hdf5_dark_path'])

                n_workers = min(p['n_accum'], os.cpu_count() or 4, 8)
                data_accum = None
                count = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_read_one, i): i for i in range(p['n_accum'])}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            frame = future.result()
                        except Exception:
                            continue
                        if p['dark_data'] is not None:
                            frame = np.maximum(frame - p['dark_data'], 0)
                        if data_accum is None:
                            data_accum = frame.astype(np.float64)
                        else:
                            if p['use_sum']:
                                data_accum += frame
                            else:
                                np.maximum(data_accum, frame, out=data_accum)
                        count += 1

                if data_accum is None:
                    data_accum = np.zeros((p['ny'], p['nz']))
                elapsed = _time.monotonic() - t0
                return data_accum, count, elapsed

            worker = AsyncWorker(target=_parallel_accum)

            def _on_accum_done(result):
                data_out, n_done, elapsed = result
                data_out = self._apply_tx_rotation(data_out)
                self.bdata = data_out
                if getattr(self, '_levels_initialized', False):
                    try:
                        lo = float(self.min_intensity_edit.text())
                        hi = float(self.max_intensity_edit.text())
                        self.image_view.set_image_data(data_out, auto_levels=False, levels=(lo, hi))
                    except ValueError:
                        self.image_view.set_image_data(data_out)
                else:
                    self.image_view.set_image_data(data_out)
                self._apply_tx_image_rect(data_out)
                self.max_check.setEnabled(True)
                self.sum_check.setEnabled(True)
                fn_display = (os.path.basename(self.zarr_zip_path or '')
                              if self.zarr_store else os.path.basename(
                                  build_filename(self.folder, self.file_stem,
                                                 self.first_file_nr, self.padding,
                                                 self.det_nr, self.ext, self.sep_folder)))
                self.frame_label.setText(f"Frame {self.frame_nr}  |  {fn_display}")
                self.setWindowTitle(f"FF Viewer — {fn_display} [frame {self.frame_nr}]")
                if self.show_rings and self.ring_rads:
                    self._draw_rings()
                if self.show_axes:
                    self._draw_axes()
                print(f"{mode_str}OverFrames: {n_done} frames from frame {start_frame} in {elapsed:.2f}s")

            def _on_accum_error(msg):
                print(f"Error in {mode_str}OverFrames: {msg}")
                self.max_check.setEnabled(True)
                self.sum_check.setEnabled(True)
                self.frame_label.setText(f"Frame {self.frame_nr}  |  Error")

            worker.finished_signal.connect(_on_accum_done)
            worker.error_signal.connect(_on_accum_error)
            worker.start()
            self._accum_worker = worker  # prevent GC
            return  # display handled by callback
        else:
            # Single frame
            file_nr = self.first_file_nr + self.frame_nr // max(1, self.n_frames_per_file)
            frame_in_file = self.frame_nr % max(1, self.n_frames_per_file)
            fn = build_filename(self.folder, self.file_stem, file_nr,
                                self.padding, self.det_nr, self.ext, self.sep_folder)

            data = read_image(fn, self.header_size, self.bytes_per_pixel,
                              self.ny, self.nz, frame_in_file,
                              self.do_transpose, self.hflip, self.vflip,
                              mask, self.zarr_store, self.zarr_dark_mean,
                              hdf5_data_path=self.hdf5_data_path,
                              hdf5_dark_path=self.hdf5_dark_path)

            if dark_data is not None:
                data = np.maximum(data - dark_data, 0)

        data = self._apply_tx_rotation(data)
        self.bdata = data
        # On first load, auto-levels; afterwards use user's MinI/MaxI
        if getattr(self, '_levels_initialized', False):
            try:
                lo = float(self.min_intensity_edit.text())
                hi = float(self.max_intensity_edit.text())
                self.image_view.set_image_data(data, auto_levels=False, levels=(lo, hi))
            except ValueError:
                self.image_view.set_image_data(data)
        else:
            self.image_view.set_image_data(data)
        self._apply_tx_image_rect(data)
        basename = os.path.basename(fn) if not self.zarr_store else os.path.basename(self.zarr_zip_path or '')
        self.frame_label.setText(f"Frame {self.frame_nr}  |  {basename}")
        self.setWindowTitle(f"FF Viewer — {basename} [frame {self.frame_nr}]")
        if self.show_rings and self.ring_rads:
            self._draw_rings()
        if self.show_axes:
            self._draw_axes()

    # ── Rings ──────────────────────────────────────────────────────

    def _redraw_if_rings(self):
        if self.show_rings and self.ring_rads:
            self._draw_rings()
        if self.show_axes:
            self._draw_axes()

    def _draw_rings(self):
        self.image_view.clear_overlays('rings')
        if not self.ring_rads:
            return
        px = self.pixel_size
        # Always read current values from text fields
        try:
            bc_y = float(self.bcy_edit.text())
            bc_z = float(self.bcz_edit.text())
            lsd = float(self.lsd_edit.text())
        except ValueError:
            return
        self.bc_local = [bc_y, bc_z]
        self.lsd_local = lsd
        colors = _color_cycle_colors
        for idx, rad in enumerate(self.ring_rads):
            Y, Z = compute_ring_points(rad, lsd, self.lsd_orig,
                                        [bc_y, bc_z], px)
            color = colors[idx % len(colors)]
            curve = pg.PlotDataItem(Y, Z, pen=pg.mkPen(color, width=1.5))
            self.image_view.add_overlay(curve, 'rings')

    def _draw_axes(self):
        """Overlay MIDAS lab-frame axes anchored at the beam center.

        Convention (looking from source toward detector):
          +Y → display LEFT, +Z → display UP, +X → INTO page (⊗ at BC),
          η = 0 toward +Z; an arc from 0°→+45° shows η-sweep direction.
        Used to verify ImTransOpt: features should be in the expected
        lab-frame quadrant relative to BC.
        """
        try:
            bc_y = float(self.bcy_edit.text())
            bc_z = float(self.bcz_edit.text())
        except ValueError:
            self.image_view.clear_overlays('axes')
            return
        # Scale label font to GUI font setting; minimum 12pt for readability.
        gui_pt = self.font_spin.value() if hasattr(self, 'font_spin') else 10
        font_size = max(12, int(round(gui_pt * 1.4)))
        draw_lab_frame_axes(self.image_view, bc_y, bc_z, self.ny, self.nz,
                            font_size=font_size)

    def _on_ring_selection(self):
        """Open ring material selection dialog."""
        # Sync current GUI values so dialog picks them up
        try:
            self.pixel_size = float(self.px_edit.text())
        except ValueError:
            pass
        try:
            self.lsd_local = float(self.lsd_edit.text())
        except ValueError:
            pass
        dlg = RingSelectionDialog(self, auto_generate=self.zarr_store is not None)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.ring_rads = dlg.ring_rads
            self.ring_nrs = dlg.ring_nrs
            self.hkls = dlg.hkls
            self.rings_to_show = dlg.rings_to_show
            self.rings_check.setChecked(True)
            self._draw_rings()


    # ── Auto-detect ────────────────────────────────────────────────

    def _start_auto_detect(self):
        self._detect_worker = AsyncWorker(
            target=auto_detect_files, args=(os.getcwd(),))
        self._detect_worker.finished_signal.connect(self._apply_auto_detect)
        self._detect_worker.start()

    def _apply_auto_detect(self, result):
        if not result:
            return
        if 'zarr_zip' in result:
            self._load_zarr_zip(result['zarr_zip'])
            return
        if 'file_stem' in result:
            self.file_stem = result['file_stem']
            self.folder = result.get('folder', self.folder)
            self.padding = result.get('padding', self.padding)
            self.first_file_nr = result.get('first_nr', self.first_file_nr)
            self.ext = result.get('ext', self.ext)
            self.file_nr_edit.setText(str(self.first_file_nr))
            if 'dark_stem' in result:
                self.dark_stem = result['dark_stem']
                self.dark_num = result['dark_num']
                self.dark_check.setChecked(True)
            # HDF5 detection
            ext_l = (self.ext or '').lower()
            if any(ext_l.endswith(e) for e in ['h5', 'hdf', 'hdf5', 'nxs']):
                fn = build_filename(self.folder, self.file_stem, self.first_file_nr,
                                     self.padding, self.det_nr, self.ext)
                if os.path.exists(fn):
                    self._detect_hdf5_dims(fn)
            print(f"Auto-detected: {self.file_stem} in {self.folder}")
            self.setWindowTitle(self.windowTitle() + " [files detected]")
            self._load_and_display()
        if 'param_file' in result:
            self._apply_param_file(result['param_file'])




# ═══════════════════════════════════════════════════════════════════════
#  Ring Selection Dialog
# ═══════════════════════════════════════════════════════════════════════

class RingSelectionDialog(QtWidgets.QDialog):
    """Two-step dialog: enter material params → select rings from list."""

    def __init__(self, viewer, parent=None, auto_generate=False):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.ring_rads = []
        self.ring_nrs = []
        self.hkls = []
        self.rings_to_show = []
        self.setWindowTitle("Ring Material Selection")
        self.resize(500, 400)
        if auto_generate:
            self._build_material_page()
            self._generate_and_select()
        else:
            self._build_material_page()

    def _build_material_page(self):
        lay = QtWidgets.QFormLayout(self)

        # ── Material preset buttons ──
        preset_row = QtWidgets.QHBoxLayout()
        btn_ceo2 = QtWidgets.QPushButton("CeO2")
        btn_lab6 = QtWidgets.QPushButton("LaB6")
        btn_ceo2.clicked.connect(lambda: self._apply_preset(225, [5.4116, 5.4116, 5.4116, 90.0, 90.0, 90.0]))
        btn_lab6.clicked.connect(lambda: self._apply_preset(221, [4.1569, 4.1569, 4.1569, 90.0, 90.0, 90.0]))
        preset_row.addWidget(btn_ceo2)
        preset_row.addWidget(btn_lab6)
        preset_row.addStretch()
        lay.addRow("Material:", preset_row)

        self.sg_edit = QtWidgets.QLineEdit(str(self.viewer.sg))
        self.wl_edit = QtWidgets.QLineEdit(str(self.viewer.wl))
        self.px_edit = QtWidgets.QLineEdit(str(self.viewer.pixel_size))
        self.lsd_edit = QtWidgets.QLineEdit(str(self.viewer.lsd_local))
        self.maxrad_edit = QtWidgets.QLineEdit(str(self.viewer.temp_max_ring_rad))
        self.lc_edits = []
        for i in range(6):
            e = QtWidgets.QLineEdit(str(self.viewer.lattice_const[i]))
            self.lc_edits.append(e)

        lay.addRow("SpaceGroup:", self.sg_edit)
        lay.addRow("Wavelength (Å) or Energy (keV):", self.wl_edit)
        lc_row = QtWidgets.QHBoxLayout()
        for e in self.lc_edits:
            e.setMinimumWidth(70)
            lc_row.addWidget(e)
        lay.addRow("Lattice Const (Å):", lc_row)
        lay.addRow("Lsd (μm):", self.lsd_edit)
        lay.addRow("MaxRingRad (μm):", self.maxrad_edit)
        lay.addRow("Pixel Size (μm):", self.px_edit)

        btn = QtWidgets.QPushButton("Generate Rings")
        btn.clicked.connect(self._generate_and_select)
        lay.addRow(btn)

    def _apply_preset(self, sg, lattice):
        """Auto-populate SpaceGroup and LatticeParameters from a material preset."""
        self.sg_edit.setText(str(sg))
        for i, val in enumerate(lattice):
            self.lc_edits[i].setText(str(val))

    def _generate_and_select(self):
        # Write temp param file and run GetHKLList
        wl = float(self.wl_edit.text())
        if wl > 1:
            wl = 12.398 / wl
        self.viewer.sg = int(self.sg_edit.text())
        self.viewer.wl = wl
        self.viewer.pixel_size = float(self.px_edit.text())
        self.viewer.lsd_local = float(self.lsd_edit.text())
        self.viewer.lsd_orig = self.viewer.lsd_local
        self.viewer.temp_max_ring_rad = float(self.maxrad_edit.text())
        for i in range(6):
            self.viewer.lattice_const[i] = float(self.lc_edits[i].text())

        if midas_config and midas_config.MIDAS_BIN_DIR:
            hkl_bin = os.path.join(midas_config.MIDAS_BIN_DIR, 'GetHKLList')
        else:
            hkl_bin = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')

        lp = self.viewer.lattice_const
        cmd = [
            hkl_bin,
            '--sg', str(self.viewer.sg),
            '--lp', str(lp[0]), str(lp[1]), str(lp[2]),
                    str(lp[3]), str(lp[4]), str(lp[5]),
            '--wl', str(wl),
            '--lsd', str(self.viewer.lsd_local),
            '--maxR', str(self.viewer.temp_max_ring_rad),
            '--stdout',
        ]
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True)
        except Exception as e:
            print(f"GetHKLList failed: {e}")
            return

        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            print("GetHKLList produced no output")
            return
        lines = lines[1:]  # skip header

        # Parse rings
        all_rings = []
        for rn in range(1, 101):
            for line in lines:
                parts = line.split()
                if len(parts) >= 11 and parts[4].isdigit() and int(parts[4]) == rn:
                    all_rings.append({
                        'nr': rn,
                        'hkl': [int(parts[0]), int(parts[1]), int(parts[2])],
                        'rad': float(parts[-1].strip()),
                        'display': line.strip()
                    })
                    break

        if not all_rings:
            print("No rings found")
            return

        # Show list for selection
        self._show_ring_list(all_rings)

    def _show_ring_list(self, all_rings):
        # Clear existing layout properly
        old = self.layout()
        if old is not None:
            while old.count():
                item = old.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            QtWidgets.QWidget().setLayout(old)  # reparent to delete

        lay = QtWidgets.QVBoxLayout(self)
        self.layout().addWidget(QtWidgets.QLabel("Select rings (Ctrl+click for multi):"))
        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        for r in all_rings:
            self._list.addItem(f"Ring {r['nr']}: HKL={r['hkl']}  Rad={r['rad']:.1f}μm")
        self._list.selectAll()
        self.layout().addWidget(self._list)

        btn = QtWidgets.QPushButton("Done")
        btn.clicked.connect(lambda: self._accept_rings(all_rings))
        self.layout().addWidget(btn)
        self._all_rings = all_rings

    def _accept_rings(self, all_rings):
        sel = self._list.selectedIndexes()
        self.ring_rads = [all_rings[i.row()]['rad'] for i in sel]
        self.ring_nrs = [all_rings[i.row()]['nr'] for i in sel]
        self.hkls = [all_rings[i.row()]['hkl'] for i in sel]
        self.rings_to_show = self.ring_nrs[:]
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
    viewer = FFViewer(theme=theme)
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
