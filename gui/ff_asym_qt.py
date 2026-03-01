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

import numpy as np
from numpy import linalg as LA
from math import sin, cos, sqrt
from multiprocessing.dummy import Pool

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
                         AsyncWorker, LogPanel, add_shortcut, COLORMAPS)

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
    """Read binary int8 mask file."""
    if not fn or not os.path.exists(fn):
        return None
    try:
        mask = np.fromfile(fn, dtype=np.int8, count=ny * nz).reshape((ny, nz))
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
        dot = f.find('.')
        name = f[:dot] if dot > 0 else f
        if name.startswith(basename + '_'):
            parts = name.split('_')
            if parts[-1].isdigit():
                data_files.append(f)

    if not data_files:
        return result

    first = data_files[0]
    dot = first.find('.')
    name_part = first[:dot]
    ext_part = first[dot + 1:]
    parts = name_part.split('_')
    if not parts[-1].isdigit():
        return result

    result['first_nr'] = int(parts[-1])
    result['padding'] = len(parts[-1])
    result['file_stem'] = '_'.join(parts[:-1])
    result['folder'] = cwd + '/'
    result['ext'] = ext_part
    result['n_frames'] = 1

    # Dark files
    for f in all_files:
        dot = f.find('.')
        name = f[:dot] if dot > 0 else f
        fparts = name.split('_')
        if len(fparts) >= 2 and fparts[-1].isdigit():
            prefix = '_'.join(fparts[:-1]).lower()
            if prefix.endswith('dark_before') or prefix.endswith('dark_after'):
                result['dark_stem'] = '_'.join(fparts[:-1])
                result['dark_num'] = int(fparts[-1])
                break
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
        self.hdf5_datasets = []

    # ── UI Construction ────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Toolbar ──
        tb = self._build_toolbar()
        main_layout.addLayout(tb)

        # ── Image View ──
        self.image_view = MIDASImageView(self)
        self.image_view.set_colormap(self.colormap_name)
        main_layout.addWidget(self.image_view, stretch=1)

        # ── Control Panels ──
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(self._build_file_panel())
        ctrl.addWidget(self._build_image_panel())
        ctrl.addWidget(self._build_display_panel())
        ctrl.addWidget(self._build_processing_panel())
        main_layout.addLayout(ctrl)

        # ── Status Bar ──
        self.status_label = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)
        self.frame_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.frame_label)

        # ── Log Panel ──
        self.log_panel = LogPanel(self, "Log")
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_panel)
        self.log_panel.install_redirect()
        self.log_panel.hide()

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

        # Export
        export_btn = QtWidgets.QPushButton("Export PNG")
        export_btn.clicked.connect(self.image_view.export_png)
        tb.addWidget(export_btn)

        # Toggle log panel
        log_btn = QtWidgets.QPushButton("Log Panel")
        log_btn.setCheckable(True)
        log_btn.toggled.connect(lambda c: self.log_panel.show() if c else self.log_panel.hide())
        tb.addWidget(log_btn)

        tb.addStretch()
        return tb

    def _build_file_panel(self):
        grp = QtWidgets.QGroupBox("File I/O")
        lay = QtWidgets.QGridLayout(grp)

        btn_first = QtWidgets.QPushButton("FirstFile")
        btn_first.clicked.connect(self._on_first_file)
        lay.addWidget(btn_first, 0, 0)

        btn_dark = QtWidgets.QPushButton("DarkFile")
        btn_dark.clicked.connect(self._on_dark_file)
        lay.addWidget(btn_dark, 0, 1)

        self.dark_check = QtWidgets.QCheckBox("Dark")
        lay.addWidget(self.dark_check, 0, 2)

        btn_zip = QtWidgets.QPushButton("Load ZIP")
        btn_zip.clicked.connect(self._on_load_zip)
        lay.addWidget(btn_zip, 0, 3)

        lay.addWidget(QtWidgets.QLabel("FileNr"), 1, 0)
        self.file_nr_edit = QtWidgets.QLineEdit(str(self.first_file_nr))
        self.file_nr_edit.setMaximumWidth(60)
        lay.addWidget(self.file_nr_edit, 1, 1)

        lay.addWidget(QtWidgets.QLabel("nFr/File"), 1, 2)
        self.nframes_edit = QtWidgets.QLineEdit("1")
        self.nframes_edit.setMaximumWidth(60)
        lay.addWidget(self.nframes_edit, 1, 3)

        lay.addWidget(QtWidgets.QLabel("H5 Data"), 2, 0)
        self.h5data_edit = QtWidgets.QLineEdit(self.hdf5_data_path)
        lay.addWidget(self.h5data_edit, 2, 1, 1, 2)
        btn_h5 = QtWidgets.QPushButton("Browse")
        btn_h5.clicked.connect(lambda: self._browse_h5(False))
        lay.addWidget(btn_h5, 2, 3)

        lay.addWidget(QtWidgets.QLabel("Mask"), 3, 0)
        self.mask_edit = QtWidgets.QLineEdit("")
        lay.addWidget(self.mask_edit, 3, 1, 1, 2)
        self.mask_check = QtWidgets.QCheckBox("Apply")
        lay.addWidget(self.mask_check, 3, 3)

        return grp

    def _build_image_panel(self):
        grp = QtWidgets.QGroupBox("Image Settings")
        lay = QtWidgets.QGridLayout(grp)

        lay.addWidget(QtWidgets.QLabel("NrPixH"), 0, 0)
        self.nz_edit = QtWidgets.QLineEdit(str(self.nz))
        self.nz_edit.setMaximumWidth(60)
        lay.addWidget(self.nz_edit, 0, 1)

        lay.addWidget(QtWidgets.QLabel("NrPixV"), 1, 0)
        self.ny_edit = QtWidgets.QLineEdit(str(self.ny))
        self.ny_edit.setMaximumWidth(60)
        lay.addWidget(self.ny_edit, 1, 1)

        lay.addWidget(QtWidgets.QLabel("Header"), 2, 0)
        self.header_edit = QtWidgets.QLineEdit(str(self.header_size))
        self.header_edit.setMaximumWidth(60)
        lay.addWidget(self.header_edit, 2, 1)

        lay.addWidget(QtWidgets.QLabel("Byt/Px"), 3, 0)
        self.bpp_edit = QtWidgets.QLineEdit(str(self.bytes_per_pixel))
        self.bpp_edit.setMaximumWidth(60)
        lay.addWidget(self.bpp_edit, 3, 1)

        self.hflip_check = QtWidgets.QCheckBox("HFlip")
        lay.addWidget(self.hflip_check, 4, 0)
        self.vflip_check = QtWidgets.QCheckBox("VFlip")
        lay.addWidget(self.vflip_check, 4, 1)
        self.transpose_check = QtWidgets.QCheckBox("Transp")
        lay.addWidget(self.transpose_check, 5, 0, 1, 2)

        lay.addWidget(QtWidgets.QLabel("PixSz(μm)"), 6, 0)
        self.px_edit = QtWidgets.QLineEdit(str(self.pixel_size))
        self.px_edit.setMaximumWidth(60)
        lay.addWidget(self.px_edit, 6, 1)

        return grp

    def _build_display_panel(self):
        grp = QtWidgets.QGroupBox("Display")
        lay = QtWidgets.QGridLayout(grp)

        lay.addWidget(QtWidgets.QLabel("Frame"), 0, 0)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, 99999)
        self.frame_spin.setValue(0)
        lay.addWidget(self.frame_spin, 0, 1, 1, 2)

        self.max_check = QtWidgets.QCheckBox("MaxOverFrames")
        lay.addWidget(self.max_check, 1, 0, 1, 2)
        self.sum_check = QtWidgets.QCheckBox("SumOverFrames")
        lay.addWidget(self.sum_check, 2, 0, 1, 2)

        lay.addWidget(QtWidgets.QLabel("nFrames"), 3, 0)
        self.max_frames_spin = QtWidgets.QSpinBox()
        self.max_frames_spin.setRange(1, 99999)
        self.max_frames_spin.setValue(240)
        lay.addWidget(self.max_frames_spin, 3, 1, 1, 2)

        return grp

    def _build_processing_panel(self):
        grp = QtWidgets.QGroupBox("Processing")
        lay = QtWidgets.QGridLayout(grp)

        btn_rings = QtWidgets.QPushButton("RingsMat")
        btn_rings.clicked.connect(self._on_ring_selection)
        lay.addWidget(btn_rings, 0, 0)

        btn_calib = QtWidgets.QPushButton("Calibrate")
        btn_calib.clicked.connect(self._on_calibrate)
        lay.addWidget(btn_calib, 0, 1)

        lay.addWidget(QtWidgets.QLabel("Lsd"), 1, 0)
        self.lsd_edit = QtWidgets.QLineEdit(str(self.lsd_local))
        self.lsd_edit.setMaximumWidth(100)
        lay.addWidget(self.lsd_edit, 1, 1)

        lay.addWidget(QtWidgets.QLabel("BC_Y"), 2, 0)
        self.bcy_edit = QtWidgets.QLineEdit(str(self.bc_local[0]))
        self.bcy_edit.setMaximumWidth(80)
        lay.addWidget(self.bcy_edit, 2, 1)

        lay.addWidget(QtWidgets.QLabel("BC_Z"), 3, 0)
        self.bcz_edit = QtWidgets.QLineEdit(str(self.bc_local[1]))
        self.bcz_edit.setMaximumWidth(80)
        lay.addWidget(self.bcz_edit, 3, 1)

        return grp

    # ── Signal Wiring (reactive) ───────────────────────────────────

    def _wire_signals(self):
        # Frame change → load + display
        self.frame_spin.valueChanged.connect(self._load_and_display)
        # Toggles → replot
        self.log_check.toggled.connect(self._on_log_toggled)
        self.dark_check.toggled.connect(self._load_and_display)
        self.rings_check.toggled.connect(self._on_rings_toggled)
        self.mask_check.toggled.connect(self._load_and_display)
        self.hflip_check.toggled.connect(self._load_and_display)
        self.vflip_check.toggled.connect(self._load_and_display)
        self.transpose_check.toggled.connect(self._load_and_display)
        # Colormap
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        # Theme
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        # Mouse wheel on image → frame scroll
        self.image_view.frameScrolled.connect(self._on_frame_scroll)
        # Cursor tracking
        self.image_view.cursorMoved.connect(self._on_cursor_moved)

    def _setup_shortcuts(self):
        add_shortcut(self, 'Right', lambda: self.frame_spin.setValue(self.frame_spin.value() + 1))
        add_shortcut(self, 'Left', lambda: self.frame_spin.setValue(self.frame_spin.value() - 1))
        add_shortcut(self, 'L', lambda: self.log_check.toggle())
        add_shortcut(self, 'R', lambda: self.rings_check.toggle())

    # ── Callbacks ──────────────────────────────────────────────────

    def _on_log_toggled(self, checked):
        self.use_log = checked
        self.image_view.set_log_mode(checked)

    def _on_cmap_changed(self, name):
        self.colormap_name = name
        self.image_view.set_colormap(name)

    def _on_theme_changed(self, theme):
        self._theme = theme
        apply_theme(QtWidgets.QApplication.instance(), theme)

    def _on_frame_scroll(self, delta):
        self.frame_spin.setValue(self.frame_spin.value() + delta)

    def _on_cursor_moved(self, x, y, val):
        px = float(self.px_edit.text() or 200)
        bcy = float(self.bcy_edit.text() or 0)
        bcz = float(self.bcz_edit.text() or 0)
        try:
            eta, rr = CalcEtaAngleRad(-x + bcy, y - bcz)
            self.status_label.setText(
                f"x={x:.1f}  y={y:.1f}  I={val:.0f}  "
                f"R={rr:.1f}px  η={eta:.1f}°"
            )
        except Exception:
            self.status_label.setText(f"x={x:.1f}  y={y:.1f}  I={val:.0f}")

    def _on_rings_toggled(self, checked):
        self.show_rings = checked
        if checked:
            if self.ring_rads is None:
                self._on_ring_selection()
            else:
                self._draw_rings()
        else:
            self.image_view.clear_overlays()

    def _on_first_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select First File")
        if not fn:
            return
        check_fn = fn[:-4] if fn.endswith('.bz2') else fn
        basename = os.path.basename(check_fn)
        dot = basename.find('.')
        if dot == -1:
            return
        name_part = basename[:dot]
        ext_part = basename[dot + 1:]
        parts = name_part.split('_')
        if parts[-1].isdigit():
            self.first_file_nr = int(parts[-1])
            self.padding = len(parts[-1])
            self.file_stem = '_'.join(parts[:-1])
        self.folder = os.path.dirname(fn) + '/'
        self.ext = ext_part
        self.file_nr_edit.setText(str(self.first_file_nr))
        self.det_nr = -1
        if ext_part.startswith('ge') and len(ext_part) == 3 and ext_part[-1].isdigit():
            self.det_nr = int(ext_part[-1])
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
        basename = os.path.basename(fn)
        dot = basename.find('.')
        name_part = basename[:dot] if dot > 0 else basename
        parts = name_part.split('_')
        if parts[-1].isdigit():
            self.dark_num = int(parts[-1])
            self.dark_stem = '_'.join(parts[:-1])
        self.dark_check.setChecked(True)
        print(f"Dark: {self.dark_stem}_{self.dark_num}")

    def _on_load_zip(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select MIDAS Zarr-ZIP", "", "ZIP Files (*.zip);;All (*)")
        if fn:
            self._load_zarr_zip(fn)

    def _detect_hdf5_dims(self, fn):
        try:
            with h5py.File(fn, 'r') as f:
                self.hdf5_datasets = []
                def visit(name, node):
                    if isinstance(node, h5py.Dataset):
                        self.hdf5_datasets.append('/' + name)
                f.visititems(visit)
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

    def _browse_h5(self, is_dark):
        if not self.hdf5_datasets:
            print("No HDF5 datasets cached.")
            return
        item, ok = QtWidgets.QInputDialog.getItem(
            self, "Select HDF5 Path", "Dataset:", self.hdf5_datasets, 0, False)
        if ok:
            if is_dark:
                self.hdf5_dark_path = item
            else:
                self.hdf5_data_path = item
                self.h5data_edit.setText(item)

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
            if 'PixelSize' in p:
                self.pixel_size = float(p['PixelSize'][0])
                self.px_edit.setText(str(self.pixel_size))
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
            self.hdf5_data_path = self.h5data_edit.text()
        except ValueError:
            pass

    def _load_and_display(self):
        """Load current frame and display."""
        self._sync_params()

        # Mask
        mask = None
        if self.apply_mask and self.mask_edit.text():
            mask = read_mask(self.mask_edit.text(), self.ny, self.nz,
                             self.do_transpose, self.hflip, self.vflip)

        # Dark subtraction
        dark_data = None
        if self.use_dark and not self.zarr_store:
            dark_fn = build_filename(self.folder, self.dark_stem, self.dark_num,
                                     self.padding, self.det_nr, self.ext)
            if os.path.exists(dark_fn):
                dark_data = read_image(dark_fn, self.header_size, self.bytes_per_pixel,
                                       self.ny, self.nz, 0,
                                       self.do_transpose, self.hflip, self.vflip)

        # Main image
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

        self.bdata = data
        self.image_view.set_image_data(data)
        self.frame_label.setText(f"Frame {self.frame_nr}")
        if self.show_rings and self.ring_rads:
            self._draw_rings()

    # ── Rings ──────────────────────────────────────────────────────

    def _draw_rings(self):
        self.image_view.clear_overlays()
        if not self.ring_rads:
            return
        px = self.pixel_size
        colors = _color_cycle_colors
        for idx, rad in enumerate(self.ring_rads):
            Y, Z = compute_ring_points(rad, self.lsd_local, self.lsd_orig,
                                        self.bc_local, px)
            color = colors[idx % len(colors)]
            curve = pg.PlotDataItem(Y, Z, pen=pg.mkPen(color, width=1.5))
            self.image_view.add_overlay(curve)

    def _on_ring_selection(self):
        """Open ring material selection dialog."""
        dlg = RingSelectionDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.ring_rads = dlg.ring_rads
            self.ring_nrs = dlg.ring_nrs
            self.hkls = dlg.hkls
            self.rings_to_show = dlg.rings_to_show
            self.rings_check.setChecked(True)
            self._draw_rings()

    def _on_calibrate(self):
        print("Calibration not yet ported — use ff_asym.py for now")

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


# ═══════════════════════════════════════════════════════════════════════
#  Ring Selection Dialog
# ═══════════════════════════════════════════════════════════════════════

class RingSelectionDialog(QtWidgets.QDialog):
    """Two-step dialog: enter material params → select rings from list."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer)
        self.viewer = viewer
        self.ring_rads = []
        self.ring_nrs = []
        self.hkls = []
        self.rings_to_show = []
        self.setWindowTitle("Ring Material Selection")
        self.resize(500, 400)
        self._build_material_page()

    def _build_material_page(self):
        lay = QtWidgets.QFormLayout(self)
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
            e.setMaximumWidth(60)
            lc_row.addWidget(e)
        lay.addRow("Lattice Const (Å):", lc_row)
        lay.addRow("Lsd (μm):", self.lsd_edit)
        lay.addRow("MaxRingRad (μm):", self.maxrad_edit)
        lay.addRow("Pixel Size (μm):", self.px_edit)

        btn = QtWidgets.QPushButton("Generate Rings")
        btn.clicked.connect(self._generate_and_select)
        lay.addRow(btn)

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

        with tempfile.TemporaryDirectory() as tmp:
            pf = os.path.join(tmp, 'ps.txt')
            with open(pf, 'w') as f:
                f.write('Wavelength ' + str(wl) + '\n')
                f.write('SpaceGroup ' + str(self.viewer.sg) + '\n')
                f.write('Lsd ' + str(self.viewer.lsd_local) + '\n')
                f.write('MaxRingRad ' + str(self.viewer.temp_max_ring_rad) + '\n')
                f.write('LatticeConstant ' + ' '.join(str(v) for v in self.viewer.lattice_const) + '\n')

            if midas_config and midas_config.MIDAS_BIN_DIR:
                hkl_bin = os.path.join(midas_config.MIDAS_BIN_DIR, 'GetHKLList')
            else:
                hkl_bin = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')

            try:
                subprocess.run([hkl_bin, pf], check=True, cwd=tmp)
            except Exception as e:
                print(f"GetHKLList failed: {e}")
                return

            hkl_file = os.path.join(tmp, 'hkls.csv')
            if not os.path.exists(hkl_file):
                print("hkls.csv not generated")
                return
            with open(hkl_file) as f:
                header = f.readline()
                lines = f.readlines()

        # Parse rings
        all_rings = []
        for rn in range(1, 101):
            for line in lines:
                parts = line.split()
                if len(parts) >= 11 and int(parts[4]) == rn:
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
        # Clear layout and show list
        while self.layout().count():
            item = self.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                pass  # sub-layouts cleaned up by parent

        self.setLayout(QtWidgets.QVBoxLayout())
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


if __name__ == '__main__':
    main()
