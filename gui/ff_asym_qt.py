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

# Register bundled HDF5 compression filters (Blosc, LZ4, Bitshuffle, Zstd)
# with libhdf5 before any h5py.File() call. Without this, opening detector
# files written with these filters fails with "can't open directory
# (/usr/local/lib/plugin)" because libhdf5 falls back to its built-in
# plugin search path.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


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
                         draw_lab_frame_axes, draw_caking_overlay)
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


def build_filename(folder, fstem, fnum, padding, det_nr, ext, sep_folder=False, sep='_'):
    fldr = folder
    if sep_folder and det_nr != -1:
        fldr = os.path.join(folder, 'ge' + str(det_nr))
    # padding == 0 → single-file mode: filename has no numeric suffix
    # (used when the user opens an image that doesn't follow stem_NNN.ext naming).
    num_part = '' if padding == 0 else sep + str(fnum).zfill(padding)
    if det_nr != -1:
        fn = os.path.join(fldr, fstem + num_part + '.ge' + str(det_nr))
    else:
        fn = os.path.join(fldr, fstem + num_part + '.' + ext)
    if not os.path.exists(fn) and os.path.exists(fn + '.bz2'):
        fn = fn + '.bz2'
    return fn


def apply_image_transforms(arr, do_transpose=False, do_hflip=False, do_vflip=False):
    """Apply MIDAS-convention orientation transforms to a 2D array.

    Single source of truth for the convention; every reader (data, dark,
    mask, slab fast-paths) and the display path go through it. Keeping it
    in one place prevents the "data flipped on cols, dark flipped on rows"
    trap that previously caused dark-residual artifacts when HFlip/VFlip
    were applied with subtly different axes in different code paths.

      HFlip      → reverse columns (axis 1)  — matches MIDAS C ImTransOpt 1
      VFlip      → reverse rows    (axis 0)  — matches MIDAS C ImTransOpt 2
      Transpose  → swap axes                 — matches MIDAS C ImTransOpt 3

    Order: transpose first, then HFlip/VFlip combined.
    """
    if arr is None:
        return None
    if do_transpose:
        arr = np.transpose(arr)
    if do_hflip and do_vflip:
        arr = arr[::-1, ::-1].copy()
    elif do_hflip:
        arr = arr[:, ::-1].copy()
    elif do_vflip:
        arr = arr[::-1, :].copy()
    return arr


def read_image(fn, header, bytes_per_pixel, ny, nz, frame_idx=0,
               mask=None, zarr_store=None, zarr_dark_mean=None,
               is_dark=False, hdf5_data_path='', hdf5_dark_path=''):
    """Read a single image frame. Returns 2D float array in RAW orientation.

    Orientation transforms (HFlip / VFlip / Transpose) are NOT applied here
    — apply them with ``apply_image_transforms`` after dark subtraction so
    data and dark stay aligned in raw coordinates.

    The Zarr branch keeps a hard-coded ``[::-1, ::-1]`` un-rotation because
    MIDAS-format zarr zips store frames in the opposite chirality on disk;
    that is part of the file format, not a user-selectable orientation, so
    it stays inside the reader.

    Mask values are zeroed in raw orientation. The mask must be raw too.
    """
    # Zarr-ZIP mode
    if zarr_store is not None:
        try:
            if is_dark:
                data = zarr_dark_mean.copy() if zarr_dark_mean is not None else np.zeros((ny, nz))
            else:
                dset = zarr_store['exchange/data']
                data = dset[frame_idx, :, :].astype(float) if frame_idx < dset.shape[0] else np.zeros((ny, nz))
            # Zarr file-format un-rotation (NOT a user transform).
            data = data.astype(float)[::-1, ::-1].copy()
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
                              mask=mask,
                              hdf5_data_path=hdf5_data_path,
                              hdf5_dark_path=hdf5_dark_path)
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
    if mask is not None and mask.shape == data.shape:
        data[mask == 1] = 0
    return data


def read_image_max(fn, header, bytes_per_pixel, ny, nz, n_frames, start_frame=0,
                   mask=None, hdf5_data_path=''):
    """Compute pixel-wise max over frames in RAW orientation."""
    data_max = None
    for i in range(start_frame, start_frame + n_frames):
        frame = read_image(fn, header, bytes_per_pixel, ny, nz, i,
                           mask=mask, hdf5_data_path=hdf5_data_path)
        data_max = frame if data_max is None else np.maximum(data_max, frame)
    return data_max


def read_mask(fn, ny, nz):
    """Read uint8 TIFF mask file in RAW orientation. 1 = masked, 0 = good.

    Mask file is assumed to be in the same on-disk orientation as the data
    (which is the natural case — both come from the same detector run).
    Apply user-selected HFlip/VFlip together with the data after dark
    subtraction via ``apply_image_transforms``.
    """
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
        return mask
    except Exception as e:
        print(f"Error reading mask: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Auto-detection (from ff_asym.py, adapted)
# ═══════════════════════════════════════════════════════════════════════

def _parse_numbered_filename(basename):
    """Parse a numbered filename like 'stem_00001.ext' or 'stem-00001.ext'.

    Finds the last ``[_-]DIGITS.`` pattern in *basename* and splits there.
    This handles filenames with multiple dots and special characters,
    e.g. ``frame_%I.cbf_00001_Varex_1_00001.cbf`` as well as hyphen-
    separated names like ``CeO2_D1600_BS600-00001.tif``.

    Returns (stem, file_nr, padding, ext, sep) or None if no pattern found.
    """
    matches = list(re.finditer(r'([_-])(\d+)\.', basename))
    if not matches:
        return None
    last = matches[-1]
    sep = last.group(1)
    digits = last.group(2)
    stem = basename[:last.start()]          # everything before sep+DIGITS
    ext  = basename[last.end():]            # everything after the dot  (no leading dot)
    return stem, int(digits), len(digits), ext, sep


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
        if parsed and (parsed[0].startswith(basename + '_')
                       or parsed[0].startswith(basename + '-')):
            # a bit redundant but keeps dark_* away
            data_files.append(f)

    if not data_files:
        return result

    first = data_files[0]
    parsed = _parse_numbered_filename(first)
    if parsed is None:
        return result

    stem, first_nr, padding, ext, sep = parsed
    result['first_nr'] = first_nr
    result['padding'] = padding
    result['file_stem'] = stem
    result['folder'] = cwd + '/'
    result['ext'] = ext
    result['sep'] = sep
    result['n_frames'] = 1

    # Dark files
    for f in all_files:
        dp = _parse_numbered_filename(f)
        if dp:
            prefix_lower = dp[0].lower()
            if prefix_lower.endswith('dark_before') or prefix_lower.endswith('dark_after'):
                result['dark_stem'] = dp[0]
                result['dark_num'] = dp[1]
                result['dark_sep'] = dp[4]
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

    def __init__(self, theme='light', auto_detect=True):
        super().__init__()
        self.setWindowTitle("FF Viewer (PyQtGraph) — MIDAS")
        self.resize(1500, 950)
        self._theme = theme

        self._init_state()
        self._build_ui()
        self._wire_signals()
        self._setup_shortcuts()
        # auto_detect=False lets embedders (e.g. the caking launcher) suppress
        # the CWD scan that would otherwise pull in an unrelated .MIDAS.zip
        # or numbered data stem on a fresh launch.
        if auto_detect:
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
        self.file_sep = '_'        # separator between stem and frame number ('_' or '-')
        self.first_file_nr = 1
        self.n_frames_per_file = 1
        self.frame_nr = 0
        self.dark_stem = ''
        self.dark_folder = ''
        self.dark_num = 0
        self.dark_sep = '_'
        self.dark_fn = ''          # full path set by "Dark File" dialog; preferred over reconstruction
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
        self.colormap_name = 'inferno'

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

        # Caking overlay
        self.show_caking = False
        self.cake_params_per_det = {}  # {det_nr: {'R_MIN':…,'ETA_MIN':…, …}}
        self.cake_params_file = ''

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

        # Lock flags: set True when the user explicitly sets the path via a button
        # or by editing the field; prevents param-file loading from overwriting them.
        self._folder_locked = False   # set by _on_first_file
        self._dark_locked   = False   # set by _on_dark_file
        self._h5data_locked = False   # set by editing h5data_edit
        self._h5dark_locked = False   # set by editing h5dark_edit
        # Multi-det (HYDRA) equivalents — set when the user edits the shared
        # data/dark path fields; prevents _absorb_shared_params from reverting
        # the user's choice on the next param-file load. External dark files
        # often store dark frames at /exchange/data while data files keep
        # them at /exchange/data_dark, so the user's path needs to stick.
        self._multi_data_loc_locked = False
        self._multi_dark_loc_locked = False

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
        file_menu.addSeparator()
        quit_act = file_menu.addAction('Quit')
        quit_act.setShortcut('Ctrl+Q')
        quit_act.triggered.connect(self.close)

        # ── Toolbar ──
        tb = self._build_toolbar()
        main_layout.addLayout(tb)

        # ── Image View ──
        # Start in 'bl' for single-panel mode: data coming out of MIDAS with
        # a correct ImTransOpt is already in physical chirality, so no extra
        # display flip is needed. _on_detector_mode_changed swaps to 'br'
        # when Multi-Det is enabled, because the HYDRA composite stitch
        # introduces an X flip that has to be cancelled at display time.
        self.image_view = MIDASImageView(self, origin='bl')
        self.image_view.set_colormap(self.colormap_name)
        self.font_spin = self.image_view._font_spin
        self.image_view.fontSizeChanged.connect(self._on_font_changed)
        self.image_view.levelsChanged.connect(self._on_hist_levels_dragged)

        # ── Control Panels (built first so they can go in the splitter) ──
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(4)
        # Stack the single-detector and multi-detector data-source panels;
        # the toolbar Multi-Det checkbox swaps which one is visible.
        self._file_stack = QtWidgets.QStackedWidget()
        self._file_stack.addWidget(self._build_file_panel())     # 0: single
        self._file_stack.addWidget(self._build_multi_panel())    # 1: multi
        ctrl.addWidget(self._file_stack, stretch=3)
        ctrl.addWidget(self._build_image_display_panel(), stretch=3)
        ctrl.addWidget(self._build_processing_panel(), stretch=2)
        ctrl_widget = QtWidgets.QWidget()
        ctrl_widget.setLayout(ctrl)

        # Wrap the controls in a scroll area so:
        #   - the bottom's *effective* minimum stays small (~one row),
        #     letting the user shrink the window without hitting a tall
        #     floor from the multi-detector panel's full content height,
        #   - swapping the QStackedWidget page (single ↔ multi) or growing
        #     content (cake editor, status labels) scrolls inside the
        #     viewport instead of stealing height from the image area.
        ctrl_scroll = QtWidgets.QScrollArea()
        ctrl_scroll.setWidget(ctrl_widget)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        ctrl_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        ctrl_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # Let the user drag the splitter down to a small slice; the scroll
        # area's vertical scrollbar takes over once the inner widget can't
        # fit. Without this the splitter clamps to the controls' natural
        # minimumSizeHint and the divider feels stuck.
        ctrl_scroll.setMinimumHeight(80)

        # Vertical splitter so the image dominates at startup but the user
        # can drag the divider for more controls space. Stretch factors
        # 1:0 send any extra vertical space to the image, never the
        # bottom. Initial bottom size is the controls' sizeHint capped at
        # BOTTOM_INIT_CAP so the image gets a generous starting share even
        # when the controls' natural height is large (multi-det panel,
        # tall fonts); the user can drag the divider either way.
        BOTTOM_INIT_CAP = 320
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.image_view)
        splitter.addWidget(ctrl_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        ctrl_widget.adjustSize()
        sb_w = self.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
        bottom_h = min(ctrl_widget.sizeHint().height() + sb_w + 4,
                       BOTTOM_INIT_CAP)
        splitter.setSizes([900, bottom_h])
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter, stretch=1)
        self._main_splitter = splitter

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
        self.cmap_combo.setCurrentText('inferno')
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

        # Caking overlay is driven by the Plot/Clear button in the cake editor
        # panel (HYDRA mode), not from a top-bar checkbox. C shortcut still toggles.

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
        self.min_intensity_edit.setMinimumWidth(110)
        tb.addWidget(self.min_intensity_edit)
        tb.addWidget(QtWidgets.QLabel("Max I:"))
        self.max_intensity_edit = QtWidgets.QLineEdit("1000")
        self.max_intensity_edit.setMinimumWidth(110)
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

        tb.addWidget(QtWidgets.QLabel("Font:"))
        self.font_size_combo = QtWidgets.QComboBox()
        self.font_size_combo.addItems(["8", "9", "10", "11", "12", "14"])
        app = QtWidgets.QApplication.instance()
        self.font_size_combo.setCurrentText(str(app.font().pointSize()))
        self.font_size_combo.setFixedWidth(50)
        self.font_size_combo.currentTextChanged.connect(self._on_font_size_changed)
        tb.addWidget(self.font_size_combo)

        return tb

    def _on_font_size_changed(self, size_str):
        try:
            size = int(size_str)
        except ValueError:
            return
        self._on_font_changed(size)

    def _build_file_panel(self):
        grp = QtWidgets.QGroupBox("Data Source")
        lay = QtWidgets.QGridLayout(grp)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setVerticalSpacing(2)
        lay.setHorizontalSpacing(4)
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

        self.dark_label = QtWidgets.QLabel("")
        self.dark_label.setStyleSheet("color: gray; font-size: 9pt;")
        lay.addWidget(self.dark_label, 0, 4)

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
            "All file/path fields are ignored: Folder, FileStem, StartNr, Ext,\n"
            "dataLoc/darkLoc (HDF5 paths), and DarkStem/Dark.")
        lay.addWidget(self.instr_only_check, 5, 1)
        self.param_label = QtWidgets.QLabel("")
        self.param_label.setStyleSheet("color: gray;")
        lay.addWidget(self.param_label, 5, 2, 1, 2)

        # Single-mode cake editor (one row for the active detector). The
        # multi-detector panel gets the 4-row variant via _build_cake_widget().
        lay.addWidget(self._build_single_cake_widget(), 6, 0, 1, 5)

        # See _build_image_display_panel: stretch row pushes everything up
        # so the panel's natural height is minimal — the whole control strip
        # shrinks because the HBoxLayout no longer needs to match a tall
        # over-spread sibling.
        lay.setRowStretch(7, 1)

        return grp

    # In single-panel mode the cake params are stored under this index in
    # cake_params_per_det. Keeping it at 1 means the single-mode editor and
    # any HYDRA-mode GE1 editor share the same backing dict entry, which
    # the integrator workflow expects.
    _SINGLE_CAKE_DET = 1

    def _active_cake_det(self):
        """Dict key for the current active cake-params entry.

        Single mode: always 1 (``_SINGLE_CAKE_DET``) so the editor, the
        param-file fallback, and the overlay all read/write the same slot.
        Multi mode: the running ``det_nr`` (1–4 for GE1–GE4).

        Without this, ``self.det_nr`` may be -1 (non-HYDRA single panel),
        which silently splits the cake state across two dict keys and the
        overlay ends up drawing stale values.
        """
        return self.det_nr if self.multi_mode else self._SINGLE_CAKE_DET

    def _build_single_cake_widget(self):
        """Compact cake editor for single-panel mode (one row, generic).

        Same Load/Save/Plot buttons as the HYDRA editor and the same 9
        editable columns, but no GE1–GE4 labels — single mode can be any
        single panel (1-ID-E GE5, 20-ID-E/D varex, 1-ID-C varex, etc.).
        """
        w = QtWidgets.QWidget()
        cl = QtWidgets.QGridLayout(w)
        cl.setContentsMargins(0, 4, 0, 0)
        cl.setHorizontalSpacing(2)
        cl.setVerticalSpacing(1)

        btn_load = QtWidgets.QPushButton("Load Cake")
        btn_load.setToolTip("Load a cake_parameters CSV.")
        btn_load.clicked.connect(self._on_pick_cake_file)
        cl.addWidget(btn_load, 0, 0)

        btn_save = QtWidgets.QPushButton("Save Cake")
        btn_save.setToolTip("Write the current cake parameters back to CSV.")
        btn_save.clicked.connect(self._on_save_cake_file)
        cl.addWidget(btn_save, 0, 1)

        self._single_cake_plot_btn = QtWidgets.QPushButton("Plot")
        self._single_cake_plot_btn.setToolTip(
            "Draw / clear the caking sector overlay.\nKeyboard shortcut: C")
        self._single_cake_plot_btn.clicked.connect(self._toggle_cake_overlay)
        cl.addWidget(self._single_cake_plot_btn, 0, 2)

        fm = btn_load.fontMetrics()
        btn_w = max(fm.horizontalAdvance(b.text())
                    for b in (btn_load, btn_save, self._single_cake_plot_btn)) + 60
        for b in (btn_load, btn_save, self._single_cake_plot_btn):
            b.setFixedWidth(btn_w)

        self.single_cake_label = QtWidgets.QLabel("")
        self.single_cake_label.setStyleSheet("color: gray;")
        cl.addWidget(self.single_cake_label, 0, 3, 1, 7)

        _hs = "color: gray; font-size: 9pt;"
        column_labels = ["R min", "R max", "R step",
                         "η min", "η max", "η step",
                         "ω sum", "ω start", "ω step"]
        cl.addWidget(QtWidgets.QLabel(""), 1, 0)
        for col, txt in enumerate(column_labels):
            lbl = QtWidgets.QLabel(txt)
            lbl.setStyleSheet(_hs)
            # Right-align so the column header sits directly above the
            # right-aligned numeric value in the edit below it.
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
            cl.addWidget(lbl, 1, col + 1)

        cake_lbl = QtWidgets.QLabel("Cake")
        cake_lbl.setStyleSheet("font-weight: bold;")
        cl.addWidget(cake_lbl, 2, 0)

        self._single_cake_edits = {}
        det = self._SINGLE_CAKE_DET
        for col, key in enumerate(self.CAKE_KEYS):
            e = QtWidgets.QLineEdit("")
            e.setMinimumWidth(80)
            e.setAlignment(QtCore.Qt.AlignRight)
            e.setPlaceholderText("—")
            cl.addWidget(e, 2, col + 1)
            e.textEdited.connect(
                lambda _txt, d=det, k=key, ed=e:
                    self._on_cake_param_edited(d, k, ed))
            e.editingFinished.connect(
                lambda d=det, k=key, ed=e:
                    self._on_cake_param_edited(d, k, ed))
            self._single_cake_edits[key] = e

        return w

    def _build_cake_widget(self):
        """Build the cake-parameter editor (Load/Save + per-GE edit rows).

        Returned widget is owned by whatever layout adds it; caller is
        responsible for placement. Only meaningful in HYDRA/multi-det mode.
        """
        cake_widget = QtWidgets.QWidget()
        self._cake_widget = cake_widget
        cl = QtWidgets.QGridLayout(cake_widget)
        cl.setContentsMargins(0, 2, 0, 0)
        cl.setSpacing(3)

        btn_cake = QtWidgets.QPushButton("Load Cake")
        btn_cake.setToolTip(
            "Load cake_parameters.1ide.geN.csv; GE siblings auto-detected.")
        btn_cake.clicked.connect(self._on_pick_cake_file)
        cl.addWidget(btn_cake, 0, 0)

        btn_save_cake = QtWidgets.QPushButton("Save Cake")
        btn_save_cake.setToolTip(
            "Write current cake parameters back to cake_parameters.1ide.geN.csv files.")
        btn_save_cake.clicked.connect(self._on_save_cake_file)
        cl.addWidget(btn_save_cake, 0, 1)

        self._cake_plot_btn = QtWidgets.QPushButton("Plot")
        self._cake_plot_btn.setToolTip(
            "Draw / clear the caking sector overlay on the composite.\n"
            "Keyboard shortcut: C")
        self._cake_plot_btn.clicked.connect(self._toggle_cake_overlay)
        cl.addWidget(self._cake_plot_btn, 0, 2)

        # Pin all three buttons to identical width, sized to the longest label
        # plus generous padding for the QPushButton frame (varies by style).
        # We use setFixedWidth so the grid layout cannot stretch one column.
        fm = btn_cake.fontMetrics()
        btn_w = max(fm.horizontalAdvance(b.text())
                    for b in (btn_cake, btn_save_cake, self._cake_plot_btn)) + 60
        btn_cake.setFixedWidth(btn_w)
        btn_save_cake.setFixedWidth(btn_w)
        self._cake_plot_btn.setFixedWidth(btn_w)

        self.cake_label = QtWidgets.QLabel("")
        self.cake_label.setStyleSheet("color: gray;")
        cl.addWidget(self.cake_label, 0, 3, 1, 4)

        _hs = "color: gray; font-size: 9pt;"
        column_labels = ["R min", "R max", "R step",
                         "η min", "η max", "η step",
                         "ω sum", "ω start", "ω step"]
        cl.addWidget(QtWidgets.QLabel(""), 1, 0)
        for col, txt in enumerate(column_labels):
            lbl = QtWidgets.QLabel(txt)
            lbl.setStyleSheet(_hs)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            cl.addWidget(lbl, 1, col + 1)

        self._cake_edits = {}
        for i, det in enumerate([1, 2, 3, 4]):
            row = 2 + i
            ge_lbl = QtWidgets.QLabel(f"GE{det}")
            ge_lbl.setStyleSheet(
                f"color: {_color_cycle_colors[i]}; font-weight: bold;")
            cl.addWidget(ge_lbl, row, 0)
            edits = {}
            for col, key in enumerate(self.CAKE_KEYS):
                e = QtWidgets.QLineEdit("")
                e.setMinimumWidth(80)
                e.setAlignment(QtCore.Qt.AlignRight)
                e.setPlaceholderText("—")
                cl.addWidget(e, row, col + 1)
                e.textEdited.connect(
                    lambda _txt, d=det, k=key, ed=e: self._on_cake_param_edited(d, k, ed))
                e.editingFinished.connect(
                    lambda d=det, k=key, ed=e: self._on_cake_param_edited(d, k, ed))
                edits[key] = e
            self._cake_edits[det] = edits

        return cake_widget

    def _build_image_display_panel(self):
        """Merged Image Settings + Display panel."""
        grp = QtWidgets.QGroupBox("Image & Display")
        lay = QtWidgets.QGridLayout(grp)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setVerticalSpacing(2)
        lay.setHorizontalSpacing(4)

        # Row 0: pixel dimensions + frame
        lay.addWidget(QtWidgets.QLabel("Pixels H"), 0, 0)
        self.nz_edit = QtWidgets.QLineEdit(str(self.nz))
        self.nz_edit.setFixedWidth(55)
        lay.addWidget(self.nz_edit, 0, 1)

        lay.addWidget(QtWidgets.QLabel("Pixels V"), 0, 2)
        self.ny_edit = QtWidgets.QLineEdit(str(self.ny))
        self.ny_edit.setFixedWidth(55)
        lay.addWidget(self.ny_edit, 0, 3)

        lay.addWidget(QtWidgets.QLabel("Display Frame"), 0, 4)
        # Frame spin + "/ N" max indicator side-by-side in the same grid cell.
        frame_row = QtWidgets.QHBoxLayout()
        frame_row.setContentsMargins(0, 0, 0, 0)
        frame_row.setSpacing(4)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, 99999)
        self.frame_spin.setValue(0)
        frame_row.addWidget(self.frame_spin)
        self.frame_max_label = QtWidgets.QLabel("/ —")
        self.frame_max_label.setToolTip(
            "Total frames in the currently-loaded file (n_frames_per_file).")
        frame_row.addWidget(self.frame_max_label)
        frame_w = QtWidgets.QWidget()
        frame_w.setLayout(frame_row)
        lay.addWidget(frame_w, 0, 5)

        # Row 1: header, bytes/pixel   (aggregation controls moved to row 3)
        lay.addWidget(QtWidgets.QLabel("Header"), 1, 0)
        self.header_edit = QtWidgets.QLineEdit(str(self.header_size))
        self.header_edit.setFixedWidth(55)
        lay.addWidget(self.header_edit, 1, 1)

        lay.addWidget(QtWidgets.QLabel("Bytes/Pixel"), 1, 2)
        self.bpp_edit = QtWidgets.QLineEdit(str(self.bytes_per_pixel))
        self.bpp_edit.setFixedWidth(35)
        lay.addWidget(self.bpp_edit, 1, 3)

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

        # Row 3: aggregation controls — frame count + mutually-exclusive mode
        # all on one row so their relationship is visually obvious.
        agg_label = QtWidgets.QLabel("# Frames:")
        agg_label.setToolTip(
            "Number of frames to aggregate (Max/Sum/Median), starting at "
            "Display Frame. May span multiple sibling files.")
        lay.addWidget(agg_label, 3, 0)
        self.max_frames_spin = QtWidgets.QSpinBox()
        self.max_frames_spin.setRange(1, 99999)
        self.max_frames_spin.setValue(240)
        self.max_frames_spin.setToolTip(agg_label.toolTip())
        lay.addWidget(self.max_frames_spin, 3, 1)
        self.max_check = QtWidgets.QCheckBox("Max")
        lay.addWidget(self.max_check, 3, 2)
        self.sum_check = QtWidgets.QCheckBox("Sum")
        lay.addWidget(self.sum_check, 3, 3)
        self.median_check = QtWidgets.QCheckBox("Median")
        self.median_check.setToolTip(
            "Per-pixel median across # Frames. Loads all frames into memory "
            "before reducing (slower than Max/Sum on large slabs).")
        lay.addWidget(self.median_check, 3, 4, 1, 2)

        # Row 4: aggregation progress bar — hidden unless a Max/Sum/Median
        # job is running.
        self.agg_progress = QtWidgets.QProgressBar()
        self.agg_progress.setRange(0, 1)
        self.agg_progress.setValue(0)
        self.agg_progress.setTextVisible(True)
        self.agg_progress.setFormat("")
        self.agg_progress.setVisible(False)
        lay.addWidget(self.agg_progress, 4, 0, 1, 6)

        # Stretch row: absorbs the extra height the QHBoxLayout gives this
        # panel (it's stretched to match the tallest sibling, Data Source),
        # so the actual rows pack tightly at the top instead of spreading out.
        lay.setRowStretch(5, 1)

        return grp

    def _build_processing_panel(self):
        grp = QtWidgets.QGroupBox("Detector & Rings")
        lay = QtWidgets.QGridLayout(grp)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setVerticalSpacing(2)
        lay.setHorizontalSpacing(4)

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

        # See _build_image_display_panel: same trick to keep rows packed at top.
        lay.setRowStretch(5, 1)

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
        self._multi_data_path_edit.setMinimumWidth(260)
        self._multi_data_path_edit.editingFinished.connect(self._on_multi_paths_changed)
        path_row.addWidget(self._multi_data_path_edit)
        path_row.addSpacing(12)
        path_row.addWidget(QtWidgets.QLabel("Dark path:"))
        self._multi_dark_path_edit = QtWidgets.QLineEdit('/exchange/data_dark')
        self._multi_dark_path_edit.setToolTip(
            "HDF5 dataset path for the dark frame in each detector file\n"
            "(e.g. /exchange/data_dark). Applied to all detectors.")
        self._multi_dark_path_edit.setMinimumWidth(260)
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
            r2 = grid_row + 2  # cake row

            en = QtWidgets.QCheckBox(f"GE{i+1}")
            en.setChecked(True)
            en.setToolTip(f"Include detector GE{i+1} in the composite")
            en.setStyleSheet(f"color: {_color_cycle_colors[i]}; font-weight: bold;")
            en.toggled.connect(lambda c, idx=i: self._on_det_enabled(idx, c))
            card_grid.addWidget(en, r0, 0, 3, 1)

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
            status_lbl.setStyleSheet("color: #888888;")
            card_grid.addWidget(status_lbl, r1, 4, 1, 4)

            # Cake row
            card_grid.addWidget(QtWidgets.QLabel("Cake"), r2, 1)
            cake_lbl = QtWidgets.QLabel("(none)")
            cake_lbl.setStyleSheet("color: gray;")
            card_grid.addWidget(cake_lbl, r2, 2)
            cake_btn = QtWidgets.QPushButton("…")
            cake_btn.setFixedWidth(28)
            cake_btn.clicked.connect(lambda _, idx=i: self._on_pick_det_cake(idx))
            card_grid.addWidget(cake_btn, r2, 3)
            cake_status_lbl = QtWidgets.QLabel("")
            cake_status_lbl.setStyleSheet("color: #888888;")
            card_grid.addWidget(cake_status_lbl, r2, 4, 1, 4)

            grid_row += 3
            self._det_widgets.append(dict(
                enable=en, data_lbl=data_lbl, dark_lbl=dark_lbl,
                param_lbl=param_lbl, status_lbl=status_lbl,
                cake_lbl=cake_lbl, cake_status_lbl=cake_status_lbl))

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

        # Cake parameter editor (Load/Save + per-GE fields). Only meaningful
        # in HYDRA mode, and the multi panel itself is hidden in single mode.
        outer.addWidget(self._build_cake_widget())
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
        self.median_check.toggled.connect(self._on_median_toggled)
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
        self.h5dark_edit.editingFinished.connect(lambda: setattr(self, '_h5dark_locked', True))
        self.h5data_edit.editingFinished.connect(lambda: setattr(self, '_h5data_locked', True))
        # Live red-highlight when the dataset path doesn't exist in the file.
        self.h5data_edit.textChanged.connect(self._validate_h5_paths)
        self.h5dark_edit.textChanged.connect(self._validate_h5_paths)
        if hasattr(self, '_multi_data_path_edit'):
            self._multi_data_path_edit.textChanged.connect(self._validate_h5_paths)
        if hasattr(self, '_multi_dark_path_edit'):
            self._multi_dark_path_edit.textChanged.connect(self._validate_h5_paths)
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
        add_shortcut(self, 'C', self._toggle_cake_overlay)
        add_shortcut(self, 'Q', self.close)

    # ── Session Save / Load ────────────────────────────────────────

    def _save_session(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Session', '', 'Session Files (*.session.json);;All (*)')
        if not fn:
            return
        if not fn.endswith('.session.json'):
            fn += '.session.json'

        def _try_float(w):
            try:
                return float(w.text())
            except (ValueError, AttributeError):
                return None

        state = {
            'viewer': 'ff',
            # File / data
            'folder': self.folder,
            'file_stem': self.file_stem,
            'first_file_nr': self.first_file_nr,
            'padding': self.padding,
            'ext': self.ext,
            'file_sep': self.file_sep,
            'det_nr': self.det_nr,
            'sep_folder': self.sep_folder,
            'frame': self.frame_spin.value(),
            'n_frames_per_file': self.n_frames_per_file,
            'hdf5_data_path': self.h5data_edit.text(),
            'hdf5_dark_path': self.h5dark_edit.text(),
            # Dark
            'use_dark': self.dark_check.isChecked(),
            'dark_fn': self.dark_fn,
            'dark_folder': self.dark_folder,
            'dark_stem': self.dark_stem,
            'dark_num': self.dark_num,
            # Mask
            'mask_fn': self.mask_fn,
            'mask_path_in_field': self.mask_edit.text(),
            'apply_mask': self.mask_check.isChecked(),
            # Image geometry
            'ny': self.ny, 'nz': self.nz,
            'header_size': self.header_size,
            'bytes_per_pixel': self.bytes_per_pixel,
            'lsd': _try_float(self.lsd_edit),
            'bcy': _try_float(self.bcy_edit),
            'bcz': _try_float(self.bcz_edit),
            'tx': _try_float(self.tx_edit),
            'px': _try_float(self.px_edit),
            # Display
            'min_intensity': _try_float(self.min_intensity_edit),
            'max_intensity': _try_float(self.max_intensity_edit),
            'composite_mode': self.composite_combo.currentText(),
            'detector_mode': self.detector_mode_combo.currentText(),
            'max_frames_spin': self.max_frames_spin.value(),
            'max_per_frames': self.max_check.isChecked(),
            'sum_per_frames': self.sum_check.isChecked(),
            'median_per_frames': self.median_check.isChecked(),
            'colormap': self.cmap_combo.currentText(),
            'theme': self.theme_combo.currentText(),
            'log': self.log_check.isChecked(),
            'hflip': self.hflip_check.isChecked(),
            'vflip': self.vflip_check.isChecked(),
            'transpose': self.transpose_check.isChecked(),
            'show_rings': self.rings_check.isChecked(),
            'show_axes': self.axes_check.isChecked(),
            # Crystallography
            'sg': self.sg,
            'wl': self.wl,
            # Caking / param-file
            'show_caking': self.show_caking,
            'cake_params_file': self.cake_params_file,
            # Full per-detector cake table — preserves GUI edits that aren't
            # in the seed CSV or param files. Keys stringified for JSON.
            'cake_params_per_det': {
                str(k): dict(v) for k, v in self.cake_params_per_det.items()
            },
            'instr_only': self.instr_only_check.isChecked(),
        }

        # ── HYDRA / multi-detector state ─────────────────────────────
        # Per-detector slots, shared HDF5 dataset paths, BigDetSize and the
        # auto-fill toggle. Per-detector geometry is rebuilt by re-loading
        # the param file at load time, so only the file paths + enabled flag
        # need to round-trip.
        try:
            state['big_det_size'] = int(self.big_det_size)
        except (TypeError, ValueError):
            pass
        if hasattr(self, '_multi_data_path_edit'):
            state['multi_data_loc'] = self._multi_data_path_edit.text()
        if hasattr(self, '_multi_dark_path_edit'):
            state['multi_dark_loc'] = self._multi_dark_path_edit.text()
        if hasattr(self, '_autofill_check'):
            state['multi_autofill_siblings'] = self._autofill_check.isChecked()
        state['det_states'] = [
            {
                'enabled': bool(s.enabled),
                'data_file': s.data_file,
                'dark_file': s.dark_file,
                'param_file': s.param_file,
            }
            for s in getattr(self, '_det_states', [])
        ]
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

        # ── Model attributes ─────────────────────────────────────────
        self.folder = state.get('folder', self.folder)
        self.file_stem = state.get('file_stem', self.file_stem)
        self.first_file_nr = state.get('first_file_nr', self.first_file_nr)
        self.padding = state.get('padding', self.padding)
        self.ext = state.get('ext', self.ext)
        self.file_sep = state.get('file_sep', self.file_sep)
        self.det_nr = state.get('det_nr', self.det_nr)
        self.sep_folder = state.get('sep_folder', self.sep_folder)
        self.ny = state.get('ny', self.ny)
        self.nz = state.get('nz', self.nz)
        self.header_size = state.get('header_size', self.header_size)
        self.bytes_per_pixel = state.get('bytes_per_pixel', self.bytes_per_pixel)
        self.n_frames_per_file = state.get('n_frames_per_file', self.n_frames_per_file)
        self.hdf5_data_path = state.get('hdf5_data_path', self.hdf5_data_path)
        self.hdf5_dark_path = state.get('hdf5_dark_path', self.hdf5_dark_path)
        self.dark_fn = state.get('dark_fn', self.dark_fn)
        self.dark_folder = state.get('dark_folder', self.dark_folder)
        self.dark_stem = state.get('dark_stem', self.dark_stem)
        self.dark_num = state.get('dark_num', self.dark_num)
        self.mask_fn = state.get('mask_fn', self.mask_fn)
        self.sg = state.get('sg', self.sg)
        self.wl = state.get('wl', self.wl)

        # ── Widgets that mirror model attributes ─────────────────────
        self.file_nr_edit.setText(str(self.first_file_nr))
        self.ny_edit.setText(str(self.ny))
        self.nz_edit.setText(str(self.nz))
        self.header_edit.setText(str(self.header_size))
        self.bpp_edit.setText(str(self.bytes_per_pixel))
        self.nframes_edit.setText(str(self.n_frames_per_file))
        self._update_frame_max_label()
        self.h5data_edit.setText(self.hdf5_data_path)
        self.h5dark_edit.setText(self.hdf5_dark_path)
        self.mask_edit.setText(state.get('mask_path_in_field', self.mask_fn or ''))

        # Geometry edits
        self.lsd_edit.setText(str(state.get('lsd', 1000000.0)))
        self.bcy_edit.setText(str(state.get('bcy', 1024.0)))
        self.bcz_edit.setText(str(state.get('bcz', 1024.0)))
        self.tx_edit.setText(str(state.get('tx', 0.0)))
        self.px_edit.setText(str(state.get('px', 200.0)))

        # Intensity range (set before Apply so it sticks). Mark levels as
        # initialized so the next _on_stats_updated does NOT auto-populate
        # MinI/MaxI from P2/P98 and overwrite what we just loaded.
        min_i = state.get('min_intensity')
        if min_i is not None:
            self.min_intensity_edit.setText(str(min_i))
        max_i = state.get('max_intensity')
        if max_i is not None:
            self.max_intensity_edit.setText(str(max_i))
        if min_i is not None or max_i is not None:
            self._levels_initialized = True

        # Display state
        self.composite_combo.setCurrentText(state.get('composite_mode',
                                                      self.composite_combo.currentText()))
        self.detector_mode_combo.setCurrentText(state.get('detector_mode',
                                                          self.detector_mode_combo.currentText()))
        self.max_frames_spin.setValue(int(state.get('max_frames_spin',
                                                    self.max_frames_spin.value())))
        self.cmap_combo.setCurrentText(state.get('colormap', 'inferno'))
        self.theme_combo.setCurrentText(state.get('theme', 'light'))
        self.log_check.setChecked(state.get('log', False))
        self.hflip_check.setChecked(state.get('hflip', False))
        self.vflip_check.setChecked(state.get('vflip', False))
        self.transpose_check.setChecked(state.get('transpose', False))
        self.rings_check.setChecked(state.get('show_rings', False))
        self.axes_check.setChecked(state.get('show_axes', False))
        self.max_check.setChecked(state.get('max_per_frames', False))
        self.sum_check.setChecked(state.get('sum_per_frames', False))
        self.median_check.setChecked(state.get('median_per_frames', False))
        self.mask_check.setChecked(state.get('apply_mask', False))
        self.instr_only_check.setChecked(state.get('instr_only', False))

        # Caking overlay
        cake_f = state.get('cake_params_file', '')
        if cake_f and os.path.exists(cake_f):
            self._load_cake_file(cake_f)
        if state.get('show_caking', False):
            self._show_cake_overlay()

        # ── HYDRA / multi-detector state ─────────────────────────────
        # Restore shared HDF5 paths and BigDetSize first so any auto-pick
        # we'd otherwise do gets pinned to the saved value.
        if 'big_det_size' in state:
            try:
                self.big_det_size = int(state['big_det_size'])
                if hasattr(self, 'bigdet_spin'):
                    self.bigdet_spin.blockSignals(True)
                    self.bigdet_spin.setValue(self.big_det_size)
                    self.bigdet_spin.blockSignals(False)
                # User pinned BigDetSize via the saved session; turn off auto
                # so a later param-file load doesn't bump it.
                self._big_det_auto = False
            except (TypeError, ValueError):
                pass
        if hasattr(self, '_multi_data_path_edit') and 'multi_data_loc' in state:
            self._multi_data_path_edit.setText(state['multi_data_loc'] or '')
            self._multi_data_loc_locked = True
        if hasattr(self, '_multi_dark_path_edit') and 'multi_dark_loc' in state:
            self._multi_dark_path_edit.setText(state['multi_dark_loc'] or '')
            self._multi_dark_loc_locked = True
        if hasattr(self, '_autofill_check') and 'multi_autofill_siblings' in state:
            self._autofill_check.setChecked(bool(state['multi_autofill_siblings']))

        det_payload = state.get('det_states') or []
        if det_payload and hasattr(self, '_det_states'):
            first_params = None
            for idx, ds in enumerate(det_payload[:len(self._det_states)]):
                self._det_states[idx] = _md.DetectorState()
                s = self._det_states[idx]
                s.enabled = bool(ds.get('enabled', True))
                pf = ds.get('param_file') or ''
                df = ds.get('data_file') or ''
                dk = ds.get('dark_file') or ''
                if pf and os.path.isfile(pf):
                    try:
                        params = s.load_param_file(pf)
                        if first_params is None:
                            first_params = params
                        self._extract_cake_keys_for_det(pf, idx + 1)
                    except Exception as e:
                        print(f'Session load: GE{idx + 1} param parse failed: {e}')
                else:
                    s.param_file = pf  # remember the path even if missing
                # load_param_file may overwrite dark_file from a `Dark <path>`
                # line; restore the user's explicit picks afterward.
                s.data_file = df
                s.dark_file = dk
                # Per-detector caches depend on file paths — invalidate.
                s._dark_image = None
                s._dark_cache_key = ()
                # Mirror the enabled flag into the per-card checkbox so the
                # UI matches the model.
                if hasattr(self, '_det_widgets'):
                    try:
                        en = self._det_widgets[idx].get('enable')
                        if en is not None:
                            en.blockSignals(True)
                            en.setChecked(s.enabled)
                            en.blockSignals(False)
                    except Exception:
                        pass
                if hasattr(self, '_refresh_det_widget'):
                    try:
                        self._refresh_det_widget(idx)
                    except Exception:
                        pass
            if first_params is not None:
                try:
                    self._absorb_shared_params(first_params)
                except Exception:
                    pass
            if hasattr(self, '_populate_cake_edits'):
                try:
                    self._populate_cake_edits()
                except Exception:
                    pass
            # Refresh path-validity tinting now that file paths are in.
            try:
                self._validate_h5_paths()
            except Exception:
                pass

        # Restore the full cake-params table AFTER param/CSV loads above so
        # any user-edited values in the session override the reconstructed
        # defaults. JSON keys come back as strings — coerce to int per det.
        cake_payload = state.get('cake_params_per_det') or {}
        if cake_payload:
            for k, v in cake_payload.items():
                try:
                    det_key = int(k)
                except (TypeError, ValueError):
                    det_key = k
                self.cake_params_per_det[det_key] = dict(v)
            if hasattr(self, '_populate_cake_edits'):
                try:
                    self._populate_cake_edits()
                except Exception:
                    pass

        # Dark / frame index last (these trigger reloads — but only when the
        # state actually changes; setChecked(False) on an already-unchecked
        # box is a no-op, so we force the reload explicitly below).
        self.dark_check.setChecked(state.get('use_dark', False))
        self.frame_spin.setValue(state.get('frame', 0))

        # Push intensity levels into the image view to match the loaded fields
        try:
            self._apply_intensity_levels()
        except Exception:
            pass

        # ── Force a reload so the restored state actually takes effect ───
        # setText() doesn't fire editingFinished, so the HDF5 dataset-path
        # fields (h5dark_edit, h5data_edit, _multi_dark_path_edit, …) never
        # ran their handlers; the DetectorState data_loc/dark_loc still hold
        # their pre-load defaults, and the figure renders without the dark.
        # Mirror _on_multi_paths_changed for HYDRA, then kick a redraw.
        if (hasattr(self, '_multi_data_path_edit') and
                hasattr(self, '_multi_dark_path_edit') and
                hasattr(self, '_det_states')):
            data_path = (self._multi_data_path_edit.text().strip()
                         or '/exchange/data')
            dark_path = (self._multi_dark_path_edit.text().strip()
                         or '/exchange/data_dark')
            for s in self._det_states:
                s.data_loc = data_path
                s.dark_loc = dark_path
                s._dark_image = None
                s._dark_cache_key = ()
            self._multi_data_loc_locked = True
            self._multi_dark_loc_locked = True
        try:
            self._load_and_display()
        except Exception as e:
            print(f'Session reload failed: {e}')

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

    def _warn_multi_imtransopt(self, opts, source=''):
        """Pop a warning when more than one non-zero ImTransOpt is specified.

        The C analysis stream applies ImTransOpt entries in the literal order
        given. The GUI display always applies transpose first, then HFlip and
        VFlip combined. For non-commuting combinations (e.g. transpose+flip on
        a non-square image, or specific orderings the user explicitly chose),
        the displayed image may not exactly match what DetectorMapper /
        IntegratorZarrOMP / PeaksFitting see. Single non-zero ops always agree.
        """
        active = [o for o in opts if o in (1, 2, 3)]
        if len(active) <= 1:
            return
        names = {1: 'HFlip', 2: 'VFlip', 3: 'Transpose'}
        seq = ' → '.join(f'{n}={names[n]}' for n in active)
        QtWidgets.QMessageBox.warning(
            self, "ImTransOpt: multiple operations",
            f"{source + ': ' if source else ''}ImTransOpt has {len(active)} "
            f"operations: {seq}.\n\n"
            "MIDAS analysis applies these in the order shown. The viewer "
            "applies transpose first, then HFlip/VFlip combined. For "
            "combinations where order matters (e.g. transpose ↔ flip on a "
            "non-square image), the displayed image may differ from what "
            "the analysis stream sees.")

    @staticmethod
    def _h5_dataset_exists(file_path, dataset_path):
        """Return True if dataset_path is in the H5 file at file_path.

        Returns False if the file exists but the dataset does not.
        Returns None when the answer is unknown (h5py missing, file missing,
        path is empty, file open failed, etc.) so callers can leave the
        UI styling alone instead of falsely flagging it red.
        """
        if not file_path or not dataset_path or h5py is None:
            return None
        if not os.path.exists(file_path):
            return None
        try:
            with h5py.File(file_path, 'r') as f:
                return dataset_path in f
        except Exception:
            return None

    def _set_h5_invalid_style(self, edit, invalid):
        """Light-red background when the dataset path is missing in the file."""
        edit.setStyleSheet("background-color: #ff8a8a; color: black;"
                           if invalid else "")

    def _validate_h5_paths(self):
        """Validate H5 dataset paths against current data/dark files.

        Sets a red background on the H5 Data / H5 Dark line edits when their
        dataset path doesn't exist in the corresponding loaded file. No-ops
        when the file isn't loaded yet (path may still be valid once loaded).
        """
        # Single-mode fields
        if hasattr(self, 'h5data_edit') and hasattr(self, 'h5dark_edit'):
            data_fn = None
            try:
                data_fn = build_filename(
                    self.folder, self.file_stem, self.first_file_nr,
                    self.padding, self.det_nr, self.ext, sep=self.file_sep)
            except Exception:
                pass
            res = self._h5_dataset_exists(data_fn, self.h5data_edit.text().strip())
            self._set_h5_invalid_style(self.h5data_edit, res is False)

            dark_fn = self.dark_fn or None
            if not dark_fn:
                try:
                    cand = build_filename(
                        self.dark_folder or self.folder, self.dark_stem,
                        self.dark_num, self.padding, self.det_nr, self.ext,
                        sep=self.dark_sep)
                    if os.path.exists(cand):
                        dark_fn = cand
                except Exception:
                    pass
            # If no dedicated dark file, the integrator looks inside the data
            # file at the dark path, so validate against data_fn in that case.
            check_against = dark_fn or data_fn
            res = self._h5_dataset_exists(check_against,
                                           self.h5dark_edit.text().strip())
            self._set_h5_invalid_style(self.h5dark_edit, res is False)

        # Multi-mode shared paths
        if hasattr(self, '_multi_data_path_edit'):
            data_path = self._multi_data_path_edit.text().strip()
            invalid = False
            for s in getattr(self, '_det_states', []):
                if s.enabled and s.data_file and os.path.exists(s.data_file):
                    if self._h5_dataset_exists(s.data_file, data_path) is False:
                        invalid = True
                        break
            self._set_h5_invalid_style(self._multi_data_path_edit, invalid)
        if hasattr(self, '_multi_dark_path_edit'):
            dark_path = self._multi_dark_path_edit.text().strip()
            invalid = False
            for s in getattr(self, '_det_states', []):
                if not s.enabled:
                    continue
                fn = s.dark_file or s.data_file
                if fn and os.path.exists(fn):
                    if self._h5_dataset_exists(fn, dark_path) is False:
                        invalid = True
                        break
            self._set_h5_invalid_style(self._multi_dark_path_edit, invalid)

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
        # Skip data-path fields if the user already chose a file via "First File"
        # (self._folder_locked) — their choice takes priority over the param file.
        if not instr_only and not self._folder_locked:
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

        # ── Caking params (fallback when no CSV is present) ──
        # Use the same dict key the editor and overlay agree on, so a
        # non-HYDRA single panel (det_nr = -1) doesn't end up writing to
        # a different slot than _draw_caking will later read.
        cp = self.cake_params_per_det.setdefault(self._active_cake_det(), {})
        for src_key, dst_key in [('RMin', 'R_MIN'), ('RMax', 'R_MAX'),
                                  ('RBinSize', 'R_STEP'), ('EtaMin', 'ETA_MIN'),
                                  ('EtaMax', 'ETA_MAX'), ('EtaBinSize', 'ETA_STEP')]:
            v = get_float(src_key)
            if v is not None:
                cp[dst_key] = v
        # Auto-detect sibling cake_parameters CSV files next to the param file
        self._autofind_cake_file(fn)
        if hasattr(self, '_cake_edits'):
            self._populate_cake_edits()

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
            self._warn_multi_imtransopt(opts, source=os.path.basename(fn))

        # ── Frames per file ──
        nfs = get_int('NrFilesPerSweep', 'nFramesPerFile', 'NFramesPerFile')
        if nfs is not None:
            self.n_frames_per_file = nfs
            self.nframes_edit.setText(str(nfs))
            self._update_frame_max_label()

        # ── HDF5 dataset paths (dataLoc / darkLoc) ──
        # Skipped when the user has manually edited the corresponding field
        # (self._h5data_locked / self._h5dark_locked), so their choice takes
        # priority. APS GE files commonly store dark at /exchange/data_dark.
        if not instr_only and not self._h5data_locked:
            dl = get_str('dataLoc')
            if dl:
                self.hdf5_data_path = dl.rstrip('/') or '/'
                if hasattr(self, 'h5data_edit'):
                    self.h5data_edit.setText(self.hdf5_data_path)
                applied.append(f"dataLoc={self.hdf5_data_path}")
        if not instr_only and not self._h5dark_locked:
            dl = get_str('darkLoc')
            if dl:
                self.hdf5_dark_path = dl.rstrip('/') or '/'
                if hasattr(self, 'h5dark_edit'):
                    self.h5dark_edit.setText(self.hdf5_dark_path)
                applied.append(f"darkLoc={self.hdf5_dark_path}")

        # ── Dark file ──
        # Skip if the user already chose a dark file via "Dark File" button.
        if not instr_only and not self._dark_locked:
            ds = get_str('DarkStem', 'Dark')
            if ds:
                if os.sep in ds or '/' in ds:
                    self.dark_folder = os.path.dirname(ds) + '/'
                    basename = os.path.basename(ds)
                    parsed = _parse_numbered_filename(basename)
                    if parsed:
                        self.dark_stem = parsed[0]
                        self.dark_num = parsed[1]
                        self.dark_sep = parsed[4]
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
        if self.show_caking and self.cake_params_per_det:
            self._draw_caking()

    # ── Callbacks ──────────────────────────────────────────────────

    def _update_frame_max_label(self):
        """Sync the '/ N' indicator next to the Display Frame spin with the
        currently-known frames-per-file count."""
        if not hasattr(self, 'frame_max_label'):
            return
        n = int(getattr(self, 'n_frames_per_file', 0) or 0)
        self.frame_max_label.setText(f"/ {n}" if n > 0 else "/ —")

    def _on_log_toggled(self, checked):
        self.use_log = checked
        self.image_view.set_log_mode(checked)

    def _on_max_toggled(self, checked):
        if checked:
            self.sum_check.setChecked(False)
            if hasattr(self, 'median_check'):
                self.median_check.setChecked(False)
        self._load_and_display()

    def _on_sum_toggled(self, checked):
        if checked:
            self.max_check.setChecked(False)
            if hasattr(self, 'median_check'):
                self.median_check.setChecked(False)
        self._load_and_display()

    def _on_median_toggled(self, checked):
        """Handle the Median checkbox toggle.

        Median aggregation requires all frames to be buffered in memory before
        reduction (unlike Max/Sum which stream frame-by-frame), so it is
        mutually exclusive with the other modes.  When enabled, Max and Sum are
        unchecked automatically.  The display is refreshed immediately.
        """
        if checked:
            self.max_check.setChecked(False)
            self.sum_check.setChecked(False)
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
            # The histogram (intensity colorbar) is its own LUT widget with
            # its own axis — pyqtgraph doesn't propagate the main view's
            # tick font there, so set it directly. Different pyqtgraph
            # versions expose the axis as either hist.axis (proxied) or
            # hist.item.axis (the underlying HistogramLUTItem).
            hist = getattr(getattr(pg_iv, 'ui', None), 'histogram', None)
            if hist is not None:
                hist_axis = (getattr(hist, 'axis', None)
                             or getattr(getattr(hist, 'item', None), 'axis', None))
                if hist_axis is not None:
                    try:
                        hist_axis.setTickFont(font)
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
            # Treat like FirstFile selection (uses fallback to single-file mode
            # when the dropped name has no _NNN / -NNN suffix).
            self._open_single_file(path)

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

    def _on_hist_levels_dragged(self, lo: float, hi: float):
        """Mirror histogram region drags into the MinI/MaxI text fields.

        Uses blockSignals so updating the text doesn't re-trigger
        _apply_intensity_levels (which would push back into the histogram
        and create a feedback loop). Levels arrive in linear units —
        MIDASImageView handles the log↔linear conversion.
        """
        for w, val in ((self.min_intensity_edit, lo),
                       (self.max_intensity_edit, hi)):
            blocked = w.blockSignals(True)
            try:
                w.setText(f'{int(round(val))}')
            finally:
                w.blockSignals(blocked)

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
                                    self.padding, self.det_nr, self.ext,
                                    self.sep_folder, sep=self.file_sep)
                try:
                    # Mean/max are orientation-invariant, so raw data is fine.
                    data = read_image(fn, self.header_size, self.bytes_per_pixel,
                                      self.ny, self.nz, frame_in,
                                      mask=None,
                                      zarr_store=self.zarr_store,
                                      zarr_dark_mean=self.zarr_dark_mean,
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
        # Multi-det composite needs the X-flip ('br') so its stitched output
        # reads in physical chirality; single-panel data is already correct
        # under 'bl'. Switch before the redraw so overlays pick up the new
        # origin in the same pass.
        self.image_view.set_origin('br' if checked else 'bl')
        if checked:
            # Composite center = (BigDetSize/2, BigDetSize/2). Push that into
            # the BC fields so rings, lab axes, cursor R/η work in the lab
            # frame. Tx is identity (composite is already in lab frame).
            self.tx_edit.setText("0")
            self._update_bc_for_multi()
        # Either direction: trigger a fresh display.
        self._load_and_display()

    def _on_multi_paths_changed(self):
        """Propagate the shared data/dark path fields to all DetectorStates.

        editingFinished fires on every focus-loss, not just real edits — so
        pressing Plot or switching colormap (which transfers focus away from
        these fields) used to silently overwrite each state's data_loc /
        dark_loc back to whatever string the *field* held. If the field had
        a stale default like /exchange/data_dark while s.dark_loc had been
        corrected to /exchange/data, the overwrite resurrected the bug each
        time the user clicked elsewhere. Guard with a change check so a
        no-op focus-loss is truly a no-op.
        """
        data_path = self._multi_data_path_edit.text().strip() or '/exchange/data'
        dark_path = self._multi_dark_path_edit.text().strip() or '/exchange/data_dark'
        changed = any(
            s.data_loc != data_path or s.dark_loc != dark_path
            for s in self._det_states)
        if not changed:
            return
        self._multi_data_loc_locked = True
        self._multi_dark_loc_locked = True
        try:
            _md.reset_warn_once()
        except AttributeError:
            pass
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

    def _extract_cake_keys_for_det(self, fn, det_key):
        # parse_detector_param_file in multidet drops the cake-range keys, so
        # the HYDRA per-detector path never seeds cake_params_per_det. Single
        # mode does this inline in _apply_param_file (RMin/RMax/EtaMin/EtaMax/
        # RBinSize/EtaBinSize → R_*/ETA_*). Mirror it here so the overlay can
        # draw without a separate cake CSV being loaded.
        try:
            raw = self._parse_param_file(fn)
        except Exception:
            return
        def _gf(key):
            v = raw.get(key)
            try:
                return float(v[0][0]) if v and v[0] else None
            except (ValueError, IndexError):
                return None
        cp = self.cake_params_per_det.setdefault(det_key, {})
        for src_key, dst_key in [('RMin', 'R_MIN'), ('RMax', 'R_MAX'),
                                  ('RBinSize', 'R_STEP'),
                                  ('EtaMin', 'ETA_MIN'), ('EtaMax', 'ETA_MAX'),
                                  ('EtaBinSize', 'ETA_STEP')]:
            v = _gf(src_key)
            if v is not None:
                cp[dst_key] = v

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
        self._extract_cake_keys_for_det(fn, idx + 1)
        def _set_param(i, path):
            try:
                self._det_states[i].load_param_file(path)
                self._extract_cake_keys_for_det(path, i + 1)
            except Exception:
                pass
        self._autofill_siblings(idx, fn, _set_param)
        if hasattr(self, '_populate_cake_edits'):
            self._populate_cake_edits()
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
            self._extract_cake_keys_for_det(pf, tgt_idx + 1)

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
        if hasattr(self, '_populate_cake_edits'):
            self._populate_cake_edits()

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
        # Skip when the user has manually edited the field — external dark
        # files don't follow the data file's dark_loc convention.
        if (hasattr(self, '_multi_data_path_edit') and params.get('data_loc')
                and not self._multi_data_loc_locked):
            self._multi_data_path_edit.setText(params['data_loc'])
        if (hasattr(self, '_multi_dark_path_edit') and params.get('dark_loc')
                and not self._multi_dark_loc_locked):
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
        cp = self.cake_params_per_det.get(idx + 1)
        if cp:
            cake_bits = []
            if 'R_MIN' in cp and 'R_MAX' in cp:
                cake_bits.append(f"R=[{cp['R_MIN']:g},{cp['R_MAX']:g}]")
            if 'ETA_MIN' in cp and 'ETA_MAX' in cp:
                cake_bits.append(f"η=[{cp['ETA_MIN']:g},{cp['ETA_MAX']:g}]°")
            if 'ETA_STEP' in cp:
                cake_bits.append(f"step={cp['ETA_STEP']:g}°")
            w['cake_status_lbl'].setText("  ".join(cake_bits))
        else:
            w['cake_status_lbl'].setText("")

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
        self._open_single_file(fn)

    def _open_single_file(self, fn):
        """Configure viewer state to display *fn*.

        Tries to parse a ``stem_NNN.ext`` / ``stem-NNN.ext`` pattern so the user
        can step through a numbered series. If no numeric suffix is present,
        falls back to single-file mode (padding=0) so the picked image still
        loads on its own.
        """
        check_fn = fn[:-4] if fn.endswith('.bz2') else fn
        basename = os.path.basename(check_fn)
        parsed = _parse_numbered_filename(basename)
        if parsed is None:
            stem, ext = os.path.splitext(basename)
            self.file_stem = stem
            self.ext = ext.lstrip('.') or 'bin'
            self.file_sep = ''
            self.padding = 0           # signals build_filename to drop the numeric suffix
            self.first_file_nr = 0
            print(f"Single-file mode: {basename}")
        else:
            (self.file_stem, self.first_file_nr, self.padding,
             self.ext, self.file_sep) = parsed
        self.folder = os.path.dirname(fn) + '/'
        self._folder_locked = True   # user explicitly chose data path; param file won't override
        self.file_nr_edit.setText(str(self.first_file_nr))
        self.det_nr = -1
        if self.ext.startswith('ge') and len(self.ext) == 3 and self.ext[-1].isdigit():
            self.det_nr = int(self.ext[-1])
        ext_lower = os.path.splitext(check_fn)[1].lower()
        if ext_lower in ['.h5', '.hdf', '.hdf5', '.nxs'] and h5py:
            self._detect_hdf5_dims(fn)
        elif ext_lower in ['.tif', '.tiff'] and tifffile is not None:
            self._detect_tiff_dims(fn)
        print(f"Loaded: stem={self.file_stem}, folder={self.folder}, ext={self.ext}")
        self._load_and_display()

    def _detect_tiff_dims(self, fn):
        """Read TIFF page shape and count; update ny/nz/n_frames edits."""
        try:
            with tifffile.TiffFile(fn) as tf:
                n_pages = len(tf.pages)
                if n_pages == 0:
                    return
                shape = tf.pages[0].shape
                if len(shape) >= 2:
                    self.ny, self.nz = int(shape[0]), int(shape[1])
                self.n_frames_per_file = n_pages
                if hasattr(self, 'ny_edit'):
                    self.ny_edit.setText(str(self.ny))
                if hasattr(self, 'nz_edit'):
                    self.nz_edit.setText(str(self.nz))
                if hasattr(self, 'nframes_edit'):
                    self.nframes_edit.setText(str(n_pages))
                self._update_frame_max_label()
        except Exception as e:
            print(f"TIFF dimension detection failed: {e}")

    def _on_dark_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Dark File")
        if not fn:
            return
        self.dark_fn = fn                           # store full path directly
        self._dark_locked = True   # user explicitly chose dark; param file won't override
        self.dark_folder = os.path.dirname(fn) + '/'
        basename = os.path.basename(fn)
        parsed = _parse_numbered_filename(basename)
        if parsed:
            self.dark_stem = parsed[0]
            self.dark_num = parsed[1]
            self.dark_sep = parsed[4]
        self.dark_check.setChecked(True)
        if hasattr(self, 'dark_label'):
            self.dark_label.setText(basename)
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
                    self._update_frame_max_label()
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
            self._update_frame_max_label()
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
                self._warn_multi_imtransopt(opts, source=os.path.basename(zip_path))

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
            self._update_frame_max_label()
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
        if nf > 0:
            if self.frame_spin.maximum() != nf - 1:
                self.frame_spin.setMaximum(max(nf - 1, 0))
            # Nothing else in the HYDRA path updates the model's per-file
            # frame count, leaving it stuck at the init default (1). Push
            # the actual value here so downstream consumers
            # (n_frames_per_file-aware file_nr / frame_in derivations, the
            # "/ N" indicator that _update_frame_max_label drives) see the truth.
            self.n_frames_per_file = nf
            self._update_frame_max_label()

        frame_idx = self.frame_nr
        bds = int(self.big_det_size)
        px = float(self.pixel_size)
        op = self.composite_op

        # Per-frame aggregation across multiple frames (Max / Sum / Median).
        # Each iteration calls _md.composite_frame (which already parallelizes
        # across detectors), then reduces across frames. Median buffers every
        # frame in memory; Max/Sum stream a running accumulator.
        agg_mode = None
        if self.sum_check.isChecked():
            agg_mode = 'sum'
        elif self.median_check.isChecked():
            agg_mode = 'median'
        elif self.max_check.isChecked():
            agg_mode = 'max'
        n_accum = 1
        if agg_mode is not None and nf > 0:
            n_accum = min(self.max_frames_spin.value(), nf - frame_idx)
            if n_accum < 1:
                n_accum = 1

        if agg_mode is None:
            self.frame_label.setText(
                f"Frame {frame_idx}  |  multi-det compositing ({len(loaded)})…")
        else:
            self.frame_label.setText(
                f"Frame {frame_idx}  |  multi-det {agg_mode.capitalize()} "
                f"over {n_accum} frames…")

        # Set up the progress bar for aggregation jobs (>1 frame). Single
        # frames are fast enough that the bar would just flicker.
        show_progress = agg_mode is not None and n_accum > 1
        if show_progress:
            self.agg_progress.setRange(0, n_accum)
            self.agg_progress.setValue(0)
            self.agg_progress.setFormat(
                f"{agg_mode.capitalize()} 0 / {n_accum}")
            self.agg_progress.setVisible(True)
        else:
            self.agg_progress.setVisible(False)

        def _worker(emit_progress):
            import time as _time
            t0 = _time.monotonic()
            if agg_mode is None or n_accum <= 1:
                data = _md.composite_frame(states, frame_idx, bds, px,
                                            op=op, subtract_dark=True,
                                            parallel=True)
                return data, _time.monotonic() - t0, 1

            accum = None
            frames_buf = [] if agg_mode == 'median' else None
            for i in range(n_accum):
                f = _md.composite_frame(states, frame_idx + i, bds, px,
                                         op=op, subtract_dark=True,
                                         parallel=True)
                if agg_mode == 'median':
                    frames_buf.append(f.astype(np.float32, copy=False))
                elif agg_mode == 'sum':
                    f64 = f.astype(np.float64, copy=False)
                    accum = f64 if accum is None else accum + f64
                else:  # max
                    accum = (f.astype(np.float32, copy=False) if accum is None
                             else np.maximum(accum, f))
                emit_progress(i + 1, n_accum)
            if agg_mode == 'median':
                data = np.median(np.stack(frames_buf, axis=0), axis=0)
            elif agg_mode == 'sum':
                data = accum.astype(np.float32)
            else:
                data = accum
            return data, _time.monotonic() - t0, n_accum

        worker = AsyncWorker(target=_worker)
        # Bind progress emitter to the worker we just constructed. The signal
        # crosses the worker→GUI thread boundary via Qt's queued connection.
        worker._args = (worker.progress_signal.emit,)
        if show_progress:
            def _on_prog(cur, tot):
                self.agg_progress.setValue(cur)
                self.agg_progress.setFormat(
                    f"{agg_mode.capitalize()} {cur} / {tot}")
            worker.progress_signal.connect(_on_prog)

        def _done(result):
            if show_progress:
                self.agg_progress.setVisible(False)
            data, elapsed, n_used = result
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
            if agg_mode is None:
                self.frame_label.setText(
                    f"Frame {frame_idx}  |  composite (op={op}, "
                    f"{len(loaded)} det, {bds}², {1000*elapsed:.0f} ms)")
                self.setWindowTitle(
                    f"FF Viewer — Multi-Det composite [frame {frame_idx}]")
            else:
                self.frame_label.setText(
                    f"Frames {frame_idx}–{frame_idx + n_used - 1}  |  "
                    f"{agg_mode.capitalize()} composite (op={op}, "
                    f"{len(loaded)} det, {bds}², {1000*elapsed:.0f} ms)")
                self.setWindowTitle(
                    f"FF Viewer — Multi-Det {agg_mode.capitalize()} "
                    f"[frames {frame_idx}–{frame_idx + n_used - 1}]")
            if self.show_rings and self.ring_rads:
                self._draw_rings()
            if self.show_axes:
                self._draw_axes()
            if self.show_caking and self.cake_params_per_det:
                self._draw_caking()

        def _err(msg):
            if show_progress:
                self.agg_progress.setVisible(False)
            self.frame_label.setText(f"Multi-Det error: {msg}")
            print(f"Multi-Det error: {msg}")

        worker.finished_signal.connect(_done)
        worker.error_signal.connect(_err)
        worker.start()
        self._multi_worker = worker  # prevent GC

    def _load_and_display(self):
        """Load current frame and display."""
        self._sync_params()
        # Re-check H5 dataset path validity now that any new file/path is in.
        self._validate_h5_paths()

        # Multi-detector mode: composite the 4 detector frames into one image
        # in a worker thread, then go through the standard display path.
        if self.multi_mode:
            self._load_and_display_multi()
            return

        # Mask (raw orientation — flipped along with data after subtraction)
        mask = None
        if self.apply_mask and self.mask_edit.text():
            mask = read_mask(self.mask_edit.text(), self.ny, self.nz)

        # Dark subtraction — prefer the directly stored path; fall back to
        # build_filename reconstruction for backwards-compat with older sessions.
        # Final fallback: read dark from the current data file when hdf5_dark_path
        # is set but no separate dark file has been designated.
        # Dark is loaded in RAW orientation; the user-selected
        # transpose/HFlip/VFlip are applied to (data − dark) at the end.
        dark_data = None
        if self.use_dark and not self.zarr_store:
            dark_fn = None
            if self.dark_fn and os.path.exists(self.dark_fn):
                dark_fn = self.dark_fn
            else:
                dark_folder = self.dark_folder if self.dark_folder else self.folder
                candidate = build_filename(dark_folder, self.dark_stem, self.dark_num,
                                           self.padding, self.det_nr, self.ext,
                                           sep=self.dark_sep)
                if os.path.exists(candidate):
                    dark_fn = candidate
                elif self.hdf5_dark_path:
                    # No separate dark file — read dark dataset from the data file itself
                    file_nr = self.first_file_nr + self.frame_nr // max(1, self.n_frames_per_file)
                    current_fn = build_filename(self.folder, self.file_stem, file_nr,
                                               self.padding, self.det_nr, self.ext,
                                               self.sep_folder, sep=self.file_sep)
                    if os.path.exists(current_fn):
                        dark_fn = current_fn
            if dark_fn and os.path.exists(dark_fn):
                ext_lower = os.path.splitext(dark_fn)[1].lower()
                if ext_lower in ['.h5', '.hdf', '.hdf5', '.nxs'] and h5py:
                    # HDF5 dark: average all frames in the dark dataset.
                    # No flips here — done in raw, applied later.
                    try:
                        with h5py.File(dark_fn, 'r') as f:
                            dpath = self.hdf5_dark_path
                            if dpath in f:
                                dset = f[dpath]
                                if dset.ndim == 3:
                                    dark_data = np.mean(dset[:], axis=0).astype(float)
                                else:
                                    dark_data = dset[:].astype(float)
                    except Exception as e:
                        print(f"Error reading HDF5 dark: {e}")
                else:
                    dark_data = read_image(dark_fn, self.header_size, self.bytes_per_pixel,
                                           self.ny, self.nz, 0)

        # Max / Sum / Median aggregation — parallel computation
        if (self.max_check.isChecked() or self.sum_check.isChecked()
                or self.median_check.isChecked()):
            n_accum = self.max_frames_spin.value()
            if self.sum_check.isChecked():
                mode = 'sum'
            elif self.median_check.isChecked():
                mode = 'median'
            else:
                mode = 'max'
            start_frame = self.frame_nr
            mode_str = mode.capitalize()

            # Disable controls while computing
            self.max_check.setEnabled(False)
            self.sum_check.setEnabled(False)
            self.median_check.setEnabled(False)
            self.frame_label.setText(f"Computing {mode_str} over {n_accum} frames...")

            # Capture all parameters for the worker thread
            params = dict(
                n_accum=n_accum, mode=mode, start_frame=start_frame,
                folder=self.folder, file_stem=self.file_stem,
                first_file_nr=self.first_file_nr, padding=self.padding,
                det_nr=self.det_nr, ext=self.ext, sep_folder=self.sep_folder,
                file_sep=self.file_sep,
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
                """Accumulate ``n_accum`` frames starting at ``start_frame``.

                Three execution paths in priority order:

                1. **Zarr slab** — if a zarr store is open, reads the entire
                   sub-slab ``[start:end]`` in one call and reduces with
                   ``np.sum``, ``np.max``, or ``np.median`` along axis-0.
                   Fastest for local zarr archives.
                2. **Single HDF5 slab** — if all requested frames live in one
                   HDF5 file, reads the slab from ``hdf5_data_path`` and
                   reduces the same way.  Falls back to path 3 on any error.
                3. **ThreadPoolExecutor** — general path for raw detector
                   files, TIFF stacks, and multi-file spans.  Each worker
                   reads one frame; results are streamed into a running
                   max/sum accumulator, or buffered into a list for median
                   (median requires all frames simultaneously).

                Dark subtraction happens in raw orientation before transforms
                so each path applies the same per-pixel correction regardless
                of the aggregation mode.  User transforms (flip / transpose)
                are applied once at the end via ``apply_image_transforms``.

                Returns ``(result_array, n_frames_used, elapsed_seconds)``.
                """
                import concurrent.futures, time as _time
                t0 = _time.monotonic()
                p = params

                # All fast paths below accumulate in RAW orientation, subtract
                # the (raw) dark, then apply the user transforms ONCE at the
                # very end via apply_image_transforms — no per-path flip
                # convention to keep in sync.

                # ── Fast path: Zarr slab read ──
                if p['zarr_store'] is not None and 'exchange/data' in p['zarr_store']:
                    dset = p['zarr_store']['exchange/data']
                    end_frame = min(p['start_frame'] + p['n_accum'], dset.shape[0])
                    slab = dset[p['start_frame']:end_frame, :, :]  # (N, ny, nz)
                    slab64 = slab.astype(np.float64)
                    if p['mode'] == 'sum':
                        result = np.sum(slab64, axis=0)
                    elif p['mode'] == 'median':
                        result = np.median(slab64, axis=0)
                    else:
                        result = np.max(slab64, axis=0)
                    # Zarr file-format un-rotation (NOT a user transform).
                    result = result[::-1, ::-1].copy()
                    if p['mask'] is not None and p['mask'].shape == result.shape:
                        result[p['mask'] == 1] = 0
                    if p['dark_data'] is not None:
                        result = result - p['dark_data']
                    result = apply_image_transforms(result,
                        p['do_transpose'], p['hflip'], p['vflip'])
                    elapsed = _time.monotonic() - t0
                    return result, end_frame - p['start_frame'], elapsed

                # ── Fast path: single HDF5 file with slab read ──
                n_fpf = max(1, p['n_frames_per_file'])
                first_file = p['first_file_nr'] + p['start_frame'] // n_fpf
                last_file = p['first_file_nr'] + (p['start_frame'] + p['n_accum'] - 1) // n_fpf
                fn0 = build_filename(p['folder'], p['file_stem'], first_file,
                                     p['padding'], p['det_nr'], p['ext'], p['sep_folder'],
                                     sep=p['file_sep'])
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
                                if p['mode'] == 'sum':
                                    result = np.sum(slab, axis=0)
                                elif p['mode'] == 'median':
                                    result = np.median(slab, axis=0)
                                else:
                                    result = np.max(slab, axis=0)
                                if p['mask'] is not None and p['mask'].shape == result.shape:
                                    result[p['mask'] == 1] = 0
                                if p['dark_data'] is not None:
                                    result = result - p['dark_data']
                                result = apply_image_transforms(result,
                                    p['do_transpose'], p['hflip'], p['vflip'])
                                elapsed = _time.monotonic() - t0
                                return result, f_end - f_start, elapsed
                    except Exception as e:
                        print(f"HDF5 slab read failed, falling back to parallel: {e}")

                # ── General path: ThreadPoolExecutor for raw/TIFF/multi-file ──
                # Each worker returns a RAW frame; dark subtraction and the
                # final user transform happen once after accumulation below.
                def _read_one(frame_idx):
                    fr = p['start_frame'] + frame_idx
                    f_nr = p['first_file_nr'] + fr // n_fpf
                    f_in = fr % n_fpf
                    fn = build_filename(p['folder'], p['file_stem'], f_nr,
                                        p['padding'], p['det_nr'], p['ext'],
                                        p['sep_folder'], sep=p['file_sep'])
                    return read_image(
                        fn, p['header_size'], p['bytes_per_pixel'],
                        p['ny'], p['nz'], f_in,
                        mask=p['mask'],
                        zarr_store=p['zarr_store'],
                        zarr_dark_mean=p['zarr_dark_mean'],
                        hdf5_data_path=p['hdf5_data_path'],
                        hdf5_dark_path=p['hdf5_dark_path'])

                n_workers = min(p['n_accum'], os.cpu_count() or 4, 8)
                data_accum = None
                # Median needs every frame held simultaneously; max/sum stream.
                buffered_frames: list[np.ndarray] | None = (
                    [] if p['mode'] == 'median' else None)
                count = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_read_one, i): i for i in range(p['n_accum'])}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            frame = future.result()
                        except Exception:
                            continue
                        # Dark in raw orientation; subtract per-frame so
                        # max/sum/median behaves correctly per pixel.
                        if p['dark_data'] is not None:
                            frame = frame - p['dark_data']
                        if buffered_frames is not None:
                            buffered_frames.append(frame.astype(np.float64))
                        elif data_accum is None:
                            data_accum = frame.astype(np.float64)
                        else:
                            if p['mode'] == 'sum':
                                data_accum += frame
                            else:  # 'max'
                                np.maximum(data_accum, frame, out=data_accum)
                        count += 1

                if buffered_frames is not None:
                    if buffered_frames:
                        data_accum = np.median(np.stack(buffered_frames, axis=0),
                                               axis=0)
                    else:
                        data_accum = np.zeros((p['ny'], p['nz']))
                if data_accum is None:
                    data_accum = np.zeros((p['ny'], p['nz']))
                # Apply user transforms once on the final accumulated raw result.
                data_accum = apply_image_transforms(data_accum,
                    p['do_transpose'], p['hflip'], p['vflip'])
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
                self.median_check.setEnabled(True)
                fn_display = (os.path.basename(self.zarr_zip_path or '')
                              if self.zarr_store else os.path.basename(
                                  build_filename(self.folder, self.file_stem,
                                                 self.first_file_nr, self.padding,
                                                 self.det_nr, self.ext, self.sep_folder,
                                                 sep=self.file_sep)))
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
                self.median_check.setEnabled(True)
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
                                self.padding, self.det_nr, self.ext, self.sep_folder,
                                sep=self.file_sep)

            data = read_image(fn, self.header_size, self.bytes_per_pixel,
                              self.ny, self.nz, frame_in_file,
                              mask=mask,
                              zarr_store=self.zarr_store,
                              zarr_dark_mean=self.zarr_dark_mean,
                              hdf5_data_path=self.hdf5_data_path,
                              hdf5_dark_path=self.hdf5_dark_path)

            # data and dark are both in RAW orientation here; subtract first,
            # then apply the user transforms once. This is the change that
            # makes HFlip/VFlip purely a display flip and rules out the
            # axis-swap class of bugs (data flipped on cols, dark on rows).
            if dark_data is not None:
                data = data - dark_data
            data = apply_image_transforms(data,
                self.do_transpose, self.hflip, self.vflip)

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
        if self.show_caking and self.cake_params_per_det:
            self._draw_caking()

    # ── Rings ──────────────────────────────────────────────────────

    def _redraw_if_rings(self):
        if self.show_rings and self.ring_rads:
            self._draw_rings()
        if self.show_axes:
            self._draw_axes()
        if self.show_caking and self.cake_params_per_det:
            self._draw_caking()

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
        # Scale label font to GUI font setting; slightly smaller than before
        # so larger arrows don't make labels feel oversized.
        gui_pt = self.font_spin.value() if hasattr(self, 'font_spin') else 10
        font_size = max(10, int(round(gui_pt * 1.0)))
        draw_lab_frame_axes(self.image_view, bc_y, bc_z, self.ny, self.nz,
                            font_size=font_size)

    # ── Caking overlay ─────────────────────────────────────────────

    # Cake CSV column order. R/ETA params are editable + drive the overlay;
    # OME params are passed through verbatim so save round-trips don't drop them.
    CAKE_KEYS = ('R_MIN', 'R_MAX', 'R_STEP',
                 'ETA_MIN', 'ETA_MAX', 'ETA_STEP',
                 'OME_SUM', 'OME_START', 'OME_STEP')

    @staticmethod
    def _parse_cake_params_csv(fn):
        """Parse header+data-row CSV.  Returns dict of {key: float} or {} on error.

        Expected format:
          R_MIN,R_MAX,R_STEP,ETA_MIN,ETA_MAX,ETA_STEP,OME_SUM,OME_START,OME_STEP
          440,2200,0.5,125,200,5,20,0,0.25
        Any header in CAKE_KEYS with a parseable float value is kept; unknown
        columns are ignored.
        """
        try:
            with open(fn) as f:
                rows = [r.strip() for r in f
                        if r.strip() and not r.strip().startswith('#')]
            if len(rows) < 2:
                return {}
            headers = [h.strip() for h in rows[0].split(',')]
            values  = [v.strip() for v in rows[1].split(',')]
            result = {}
            for h, v in zip(headers, values):
                if h in FFViewer.CAKE_KEYS:
                    try:
                        result[h] = float(v)
                    except ValueError:
                        pass
            return result
        except Exception:
            return {}

    def _autofind_cake_file(self, param_fn):
        """Scan the directory of param_fn for any cake_parameters*.csv files
        and load the first one found (with sibling expansion).
        Respects the Auto-fill siblings checkbox."""
        if not getattr(self, '_autofill_check', None) or not self._autofill_check.isChecked():
            return
        param_dir = os.path.dirname(param_fn) or '.'
        try:
            all_files = os.listdir(param_dir)
        except Exception:
            return
        candidates = sorted(
            f for f in all_files
            if f.lower().startswith('cake_parameters') and f.lower().endswith('.csv'))
        if not candidates:
            return
        seed = os.path.join(param_dir, candidates[0])
        self._load_cake_file(seed, autofound=True)

    def _load_cake_file(self, fn, autofound=False):
        """Parse fn; auto-derive and load geN siblings; populate cake_params_per_det."""
        tag = self._find_detector_tag(fn)
        src_digit = int(tag[1]) if tag else None

        parsed = self._parse_cake_params_csv(fn)
        if not parsed:
            if not autofound:
                QtWidgets.QMessageBox.warning(
                    self, "Cake params",
                    f"No recognised cake parameters found in:\n{fn}")
            return

        self.cake_params_file = fn
        key = src_digit if src_digit else 1
        self.cake_params_per_det[key] = parsed
        print(f"Cake params ge{key} from {os.path.basename(fn)}: {parsed}")

        # Auto-load siblings (ge1…ge4) in the same directory — obeys Auto-fill checkbox
        loaded_keys = {key: fn}
        auto_sib = getattr(self, '_autofill_check', None) and self._autofill_check.isChecked()
        if tag and auto_sib:
            prefix, src_d = tag
            for d in '1234':
                if d == src_d:
                    continue
                sib = self._derive_ge_path(fn, prefix + src_d, prefix + d)
                if sib and os.path.exists(sib):
                    sib_parsed = self._parse_cake_params_csv(sib)
                    if sib_parsed:
                        self.cake_params_per_det[int(d)] = sib_parsed
                        loaded_keys[int(d)] = sib
                        print(f"Cake params ge{d} from {os.path.basename(sib)}: {sib_parsed}")

        # Refresh per-detector widgets for every key that was loaded
        if self.multi_mode:
            for det_key, cake_fn in loaded_keys.items():
                widget_idx = det_key - 1
                if 0 <= widget_idx < len(self._det_widgets):
                    self._det_widgets[widget_idx]['cake_lbl'].setText(
                        os.path.basename(cake_fn))
                    self._refresh_det_widget(widget_idx)

        # Update Data Source panel label and editable fields
        if hasattr(self, 'cake_label'):
            self.cake_label.setText(os.path.basename(fn))
        if hasattr(self, 'single_cake_label'):
            self.single_cake_label.setText(os.path.basename(fn))
        if hasattr(self, '_cake_edits'):
            self._populate_cake_edits()

        if self.show_caking:
            self._draw_caking()

    def _populate_cake_edits(self):
        """Fill the cake parameter edit boxes from cake_params_per_det.

        Populates both the HYDRA 4-row editor and the single-mode 1-row
        editor so the two stay in sync after Load Cake / programmatic update.
        """
        for det in [1, 2, 3, 4]:
            p = self.cake_params_per_det.get(det, {})
            for key, edit in self._cake_edits[det].items():
                val = p.get(key, '')
                edit.setText(str(val) if val != '' else '')
        # Single-mode editor mirrors det = _SINGLE_CAKE_DET.
        if hasattr(self, '_single_cake_edits'):
            p = self.cake_params_per_det.get(self._SINGLE_CAKE_DET, {})
            for key, edit in self._single_cake_edits.items():
                val = p.get(key, '')
                edit.setText(str(val) if val != '' else '')

    def _on_cake_param_edited(self, det, key, edit):
        """User edited a cake parameter field — update state and redraw.

        Mirrors the new value into the *other* editor's matching field so
        the HYDRA 4-row and single-mode 1-row editors stay in sync.
        """
        try:
            val = float(edit.text().strip())
        except ValueError:
            return
        prev = self.cake_params_per_det.setdefault(det, {}).get(key)
        if prev == val:
            return  # nothing changed; skip redundant redraw
        self.cake_params_per_det[det][key] = val
        # Sync the sibling editor's matching field (without triggering its
        # textEdited signal again, which would loop).
        siblings = []
        if hasattr(self, '_cake_edits') and det in self._cake_edits:
            siblings.append(self._cake_edits[det].get(key))
        if (hasattr(self, '_single_cake_edits')
                and det == self._SINGLE_CAKE_DET):
            siblings.append(self._single_cake_edits.get(key))
        for e2 in siblings:
            if e2 is not None and e2 is not edit:
                e2.blockSignals(True)
                e2.setText(str(val))
                e2.blockSignals(False)
        # Keep the per-GE inline status label (R=[..,..] η=[..,..] step=..) in sync
        if hasattr(self, '_det_widgets') and 1 <= det <= len(self._det_widgets):
            self._refresh_det_widget(det - 1)
        # If overlay is on, refresh it. If it's off, leave it off — the user
        # will hit Plot when they want to see the result.
        if self.show_caking:
            self._draw_caking()

    def _confirm_cake_overwrite(self, targets: list[tuple[int, str]]) -> bool:
        """If any path in ``targets`` already exists, show a confirmation
        dialog listing the existing files. Returns True if the user wants to
        proceed (or no files would be overwritten), False if cancelled.

        ``_write_cake_csv`` appends '.csv' when missing; mirror that here so
        the existence check matches the actual write target.
        """
        existing = []
        for _det, fn in targets:
            check = fn if fn.lower().endswith('.csv') \
                else os.path.splitext(fn)[0] + '.csv'
            if os.path.isfile(check):
                existing.append(check)
        if not existing:
            return True
        msg = ("The following cake parameter file"
               f"{'s' if len(existing) > 1 else ''} already exist and will "
               "be overwritten:\n\n  "
               + "\n  ".join(existing)
               + "\n\nProceed?")
        reply = QtWidgets.QMessageBox.question(
            self, "Overwrite cake parameters?", msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        return reply == QtWidgets.QMessageBox.Yes

    def _on_save_cake_file(self):
        """Save all loaded GE detector cake params to their CSV files."""
        if not self.cake_params_per_det:
            QtWidgets.QMessageBox.information(
                self, "Save Cake", "No cake parameters to save.")
            return

        if self.cake_params_file:
            tag = self._find_detector_tag(self.cake_params_file)
            if tag:
                prefix, src_d = tag
                # Collect every target path FIRST so we can check overwrites
                # in one prompt before writing anything.
                targets: list[tuple[int, str]] = []
                for det in [1, 2, 3, 4]:
                    if det not in self.cake_params_per_det:
                        continue
                    if str(det) == src_d:
                        sib = self.cake_params_file
                    else:
                        sib = self._derive_ge_path(
                            self.cake_params_file,
                            prefix + src_d, prefix + str(det))
                    if sib:
                        targets.append((det, sib))
                if targets:
                    if not self._confirm_cake_overwrite(targets):
                        return
                    saved = []
                    for det, sib in targets:
                        written = self._write_cake_csv(sib, det)
                        if written:
                            saved.append(os.path.basename(written))
                    if saved:
                        print("Cake params saved:\n  " + "\n  ".join(saved))
                        return

        # No seed cake file (or no sibling derivation worked) — prompt for one.
        # In multi-det mode with multiple detectors loaded, derive sibling
        # paths from the chosen filename by substituting the geN tag (or
        # appending _geN if no tag is present) so every detector gets a file.
        default_dir = (os.path.dirname(self.cake_params_file)
                       or (os.path.dirname(self._det_states[0].param_file)
                           if getattr(self, '_det_states', None) and
                              self._det_states[0].param_file else '')
                       or os.getcwd())
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Cake Parameters", default_dir,
            "CSV Files (*.csv);;All (*)")
        if not fn:
            return
        det_keys = sorted(self.cake_params_per_det.keys())
        if len(det_keys) <= 1:
            # Single-det: QFileDialog has already confirmed overwrite of `fn`,
            # but check derived-extension form (.csv-appended) too just in case.
            det = det_keys[0]
            if not self._confirm_cake_overwrite([(det, fn)]):
                return
            written = self._write_cake_csv(fn, det)
            print(f"Cake params saved:\n  {written or fn}")
            return

        # Multi-det save: build the full target list first, then prompt once
        # with all would-be-overwritten siblings.
        base = os.path.basename(fn)
        m = re.search(r'([A-Za-z]+)([1-4])(?![0-9])', base)
        targets = []
        if m:
            prefix, src_d = m.group(1), m.group(2)
            for det in det_keys:
                if str(det) == src_d:
                    sib = fn
                else:
                    sib = self._derive_ge_path(
                        fn, prefix + src_d, prefix + str(det)) or fn
                targets.append((det, sib))
        else:
            stem, ext = os.path.splitext(fn)
            for det in det_keys:
                targets.append((det, f"{stem}_ge{det}{ext}"))
        if not self._confirm_cake_overwrite(targets):
            return
        saved = []
        for det, sib in targets:
            written = self._write_cake_csv(sib, det)
            if written:
                saved.append(os.path.basename(written))
        print("Cake params saved:\n  " + "\n  ".join(saved))

    def _write_cake_csv(self, fn, det_nr):
        """Write cake_params_per_det[det_nr] to fn in header+data row format.

        Writes all known columns (R/ETA/OME) so a load → edit → save round-trip
        preserves OME values even when they're not edited in the GUI.
        Ensures the output file ends in ``.csv``. Returns the actual path
        written (with extension), or None on failure.
        """
        if not fn.lower().endswith('.csv'):
            fn = os.path.splitext(fn)[0] + '.csv'
        p = self.cake_params_per_det.get(det_nr, {})
        header = ','.join(self.CAKE_KEYS)
        values = ','.join(str(p.get(k, '')) for k in self.CAKE_KEYS)
        try:
            with open(fn, 'w') as f:
                f.write(header + '\n')
                f.write(values + '\n')
            return fn
        except Exception as e:
            print(f"Cake save failed ({os.path.basename(fn)}): {e}")
            return None

    def _on_pick_cake_file(self):
        """Open file dialog to select a cake_parameters CSV."""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Cake Parameters CSV",
            os.path.dirname(self.cake_params_file) or os.getcwd(),
            "CSV Files (*.csv);;All (*)")
        if fn:
            self._load_cake_file(fn)

    def _on_pick_det_cake(self, idx):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Select GE{idx+1} cake parameters CSV",
            os.path.dirname(self.cake_params_file) or
            os.path.dirname(self._det_states[idx].param_file) or os.getcwd(),
            "CSV Files (*.csv);;All (*)")
        if fn:
            self._load_cake_file(fn)

    def _show_cake_overlay(self):
        """Turn the caking overlay on (or refresh it if already on)."""
        if not self.cake_params_per_det:
            QtWidgets.QMessageBox.information(
                self, "Caking Overlay",
                "No caking parameters loaded.\n"
                "Load a params file with RMin/RMax/EtaMin/… keys,\n"
                "or pick a cake_parameters CSV via 'Cake File…'.")
            return
        self.show_caking = True
        self._draw_caking()
        self._update_cake_plot_button()

    def _hide_cake_overlay(self):
        self.show_caking = False
        self.image_view.clear_overlays('caking')
        self._update_cake_plot_button()

    def _toggle_cake_overlay(self):
        """C-shortcut and Plot/Clear button handler — flip the overlay state."""
        if self.show_caking:
            self._hide_cake_overlay()
        else:
            self._show_cake_overlay()

    def _update_cake_plot_button(self):
        label = "Clear" if self.show_caking else "Plot"
        for attr in ('_cake_plot_btn', '_single_cake_plot_btn'):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setText(label)

    def _draw_caking(self):
        """Draw caking sector overlays from cake_params_per_det."""
        self.image_view.clear_overlays('caking')
        if not self.cake_params_per_det:
            return
        try:
            bc_y = float(self.bcy_edit.text())
            bc_z = float(self.bcz_edit.text())
        except ValueError:
            return

        colors = _color_cycle_colors
        if self.multi_mode:
            det_list = [d for d in sorted(self.cake_params_per_det.keys())
                        if 1 <= d <= 4 and self._det_states[d - 1].enabled]
        else:
            # Single mode: always use the slot the editor and param-file
            # fallback agree on (_SINGLE_CAKE_DET via _active_cake_det()).
            # Previously this read self.det_nr, which on a non-HYDRA panel
            # is -1 and pointed at a stale dict entry.
            key = self._active_cake_det()
            det_list = [key] if key in self.cake_params_per_det else []

        detectors = []
        for i, det in enumerate(det_list):
            p = self.cake_params_per_det[det]
            try:
                r_min  = float(p['R_MIN'])
                r_max  = float(p['R_MAX'])
                e_step = float(p['ETA_STEP'])
                if self.multi_mode:
                    # Cake CSV ETA values are in MIDAS lab frame (eta=atan2(-Y,Z)).
                    # With the image_view's 'br' origin, +Y_lab is on display-LEFT,
                    # so MIDAS eta=+90° (= -Y_lab) lands on display-RIGHT — the
                    # same place draw_caking_overlay puts eta_arg=+90°. So pass
                    # cake_eta straight through. Per-detector tx is already
                    # baked into the data placement via compute_inv_coords.
                    eta_min = float(p['ETA_MIN'])
                    eta_max = float(p['ETA_MAX'])
                    color = colors[(det - 1) % len(colors)]
                else:
                    try:
                        tx_offset = float(self.tx_edit.text())
                    except ValueError:
                        tx_offset = 0.0
                    eta_min = float(p['ETA_MIN']) + tx_offset
                    eta_max = float(p['ETA_MAX']) + tx_offset
                    color = colors[i % len(colors)]
                detectors.append((color, r_min, r_max,
                                   eta_min, eta_max, e_step))
            except KeyError:
                pass

        draw_caking_overlay(self.image_view, bc_y, bc_z, detectors)

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
            self.file_sep = result.get('sep', self.file_sep)
            self.file_nr_edit.setText(str(self.first_file_nr))
            if 'dark_stem' in result:
                self.dark_stem = result['dark_stem']
                self.dark_num = result['dark_num']
                self.dark_sep = result.get('dark_sep', self.file_sep)
                self.dark_check.setChecked(True)
            # HDF5 detection
            ext_l = (self.ext or '').lower()
            if any(ext_l.endswith(e) for e in ['h5', 'hdf', 'hdf5', 'nxs']):
                fn = build_filename(self.folder, self.file_stem, self.first_file_nr,
                                     self.padding, self.det_nr, self.ext,
                                     sep=self.file_sep)
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

    # Material presets. Each entry is (name, sg, lattice, dspacings).
    #   Crystalline:  sg + 6-element lattice; dspacings=None → uses GetHKLList.
    #   d-spacing:    dspacings list (Å); sg=lattice=None → bypasses GetHKLList,
    #                 computes ring radii directly via Bragg's law.
    # First entry is a no-op placeholder so opening the combo doesn't clobber
    # current values.
    _MATERIAL_PRESETS = [
        ("(custom)",  None,  None,                                              None),
        ("CeO2",      225,  [5.4116,  5.4116,  5.4116,  90.0, 90.0, 90.0],      None),
        ("LaB6",      221,  [4.1569,  4.1569,  4.1569,  90.0, 90.0, 90.0],      None),
        ("Si",        227,  [5.43102, 5.43102, 5.43102, 90.0, 90.0, 90.0],      None),
        ("Al",        225,  [4.0495,  4.0495,  4.0495,  90.0, 90.0, 90.0],      None),
        ("Au",        225,  [4.0786,  4.0786,  4.0786,  90.0, 90.0, 90.0],      None),
        ("Cu",        225,  [3.6149,  3.6149,  3.6149,  90.0, 90.0, 90.0],      None),
        ("Ni",        225,  [3.5238,  3.5238,  3.5238,  90.0, 90.0, 90.0],      None),
        ("Fe (bcc)",  229,  [2.8665,  2.8665,  2.8665,  90.0, 90.0, 90.0],      None),
        ("Fe (fcc)",  225,  [3.6467,  3.6467,  3.6467,  90.0, 90.0, 90.0],      None),
        ("W",         229,  [3.1652,  3.1652,  3.1652,  90.0, 90.0, 90.0],      None),
        ("Ti (hcp)",  194,  [2.9508,  2.9508,  4.6855,  90.0, 90.0, 120.0],     None),
        ("Mg (hcp)",  194,  [3.2094,  3.2094,  5.2107,  90.0, 90.0, 120.0],     None),
        # SAXS calibrant: silver behenate lamellar d001 = 58.380 Å, higher
        # orders are d001/n. Listing the first ~12 orders covers typical
        # SAXS-to-low-WAXS detector ranges.
        ("AgBe (SAXS)", None, None,
            [58.380/n for n in range(1, 13)]),
    ]

    def _build_material_page(self):
        lay = QtWidgets.QFormLayout(self)

        # ── Material preset dropdown ──
        self.material_combo = QtWidgets.QComboBox()
        for name, _sg, _lat, _ds in self._MATERIAL_PRESETS:
            self.material_combo.addItem(name)
        self.material_combo.setToolTip(
            "Pick a calibrant/common material to auto-fill SpaceGroup and\n"
            "Lattice Constants (or d-spacings for SAXS calibrants like AgBe).\n"
            "Select '(custom)' to edit by hand.")
        self.material_combo.currentIndexChanged.connect(self._on_material_changed)
        lay.addRow("Material:", self.material_combo)

        self.sg_edit = QtWidgets.QLineEdit(str(self.viewer.sg))
        self.wl_edit = QtWidgets.QLineEdit(str(self.viewer.wl))
        self.px_edit = QtWidgets.QLineEdit(str(self.viewer.pixel_size))
        self.lsd_edit = QtWidgets.QLineEdit(str(self.viewer.lsd_local))
        self.maxrad_edit = QtWidgets.QLineEdit(str(self.viewer.temp_max_ring_rad))
        self.lc_edits = []
        for i in range(6):
            e = QtWidgets.QLineEdit(str(self.viewer.lattice_const[i]))
            self.lc_edits.append(e)
        # d-spacings field: when non-empty, ring radii are computed directly
        # from these via Bragg's law and SpaceGroup/Lattice are ignored.
        self.dspacings_edit = QtWidgets.QLineEdit("")
        self.dspacings_edit.setPlaceholderText(
            "e.g. 58.380, 29.190, 19.460  (overrides SpaceGroup/Lattice)")
        self.dspacings_edit.setToolTip(
            "Optional: comma- or space-separated d-spacings in Å. When set,\n"
            "ring radii are computed directly from Bragg's law and the\n"
            "SpaceGroup/Lattice fields above are ignored. Useful for SAXS\n"
            "calibrants like silver behenate (AgBe).")

        lay.addRow("SpaceGroup:", self.sg_edit)
        lay.addRow("Wavelength (Å) or Energy (keV):", self.wl_edit)
        lc_row = QtWidgets.QHBoxLayout()
        for e in self.lc_edits:
            e.setMinimumWidth(70)
            lc_row.addWidget(e)
        lay.addRow("Lattice Const (Å):", lc_row)
        lay.addRow("d-spacings (Å):", self.dspacings_edit)
        lay.addRow("Lsd (μm):", self.lsd_edit)
        lay.addRow("MaxRingRad (μm):", self.maxrad_edit)
        lay.addRow("Pixel Size (μm):", self.px_edit)

        btn = QtWidgets.QPushButton("Generate Rings")
        btn.clicked.connect(self._generate_and_select)
        lay.addRow(btn)

    def _on_material_changed(self, idx):
        """Fill SpaceGroup/Lattice or d-spacings field from the preset."""
        if idx <= 0:  # '(custom)' — leave fields alone
            return
        _name, sg, lattice, dspacings = self._MATERIAL_PRESETS[idx]
        if dspacings is not None:
            # d-spacing-only preset (e.g. AgBe): fill d-spacings, clear the
            # SpaceGroup/Lattice fields so it's visually clear they're unused.
            self.dspacings_edit.setText(", ".join(f"{d:.4f}" for d in dspacings))
        else:
            self.dspacings_edit.setText("")
            self._apply_preset(sg, lattice)

    def _apply_preset(self, sg, lattice):
        """Auto-populate SpaceGroup and LatticeParameters from a material preset."""
        self.sg_edit.setText(str(sg))
        for i, val in enumerate(lattice):
            self.lc_edits[i].setText(str(val))

    @staticmethod
    def _parse_dspacings(text):
        """Parse a comma/space-separated d-spacings list. Returns [] on empty."""
        text = (text or '').strip()
        if not text:
            return []
        tokens = re.split(r'[,\s]+', text)
        out = []
        for t in tokens:
            if not t:
                continue
            try:
                v = float(t)
            except ValueError:
                continue
            if v > 0:
                out.append(v)
        return out

    def _generate_and_select(self):
        # Write temp param file and run GetHKLList
        wl = float(self.wl_edit.text())
        if wl > 1:
            wl = 12.398 / wl
        self.viewer.wl = wl
        self.viewer.pixel_size = float(self.px_edit.text())
        self.viewer.lsd_local = float(self.lsd_edit.text())
        self.viewer.lsd_orig = self.viewer.lsd_local
        self.viewer.temp_max_ring_rad = float(self.maxrad_edit.text())

        # d-spacings branch: bypass GetHKLList, compute ring radii from Bragg.
        dspacings = self._parse_dspacings(self.dspacings_edit.text())
        if dspacings:
            self._generate_from_dspacings(dspacings, wl)
            return

        self.viewer.sg = int(self.sg_edit.text())
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

    def _generate_from_dspacings(self, dspacings, wl):
        """Compute ring radii directly from d-spacings (Bragg) and pick rings.

        2θ = 2·arcsin(λ/(2d)); r_um = lsd_um · tan(2θ). Used for materials
        like silver behenate where the SAXS rings are characterised by their
        lamellar d-spacing rather than a crystallographic (sg, lattice).
        Synthesises HKL labels as (0,0,n) for the nth listed d-spacing — the
        viewer overlay only uses these for display.
        """
        lsd = self.viewer.lsd_local
        max_r = self.viewer.temp_max_ring_rad
        all_rings = []
        skipped = []
        for n, d in enumerate(dspacings, start=1):
            ratio = wl / (2.0 * d)
            if abs(ratio) >= 1.0:
                skipped.append((n, d, "wavelength too large for this d"))
                continue
            two_theta = 2.0 * math.asin(ratio)
            r = lsd * math.tan(two_theta)
            if r <= 0:
                continue
            if r > max_r:
                skipped.append((n, d, f"r={r:.1f}μm > MaxRingRad"))
                continue
            all_rings.append({
                'nr': n,
                'hkl': [0, 0, n],
                'rad': r,
                'display': f"d={d:.4f}Å  2θ={math.degrees(two_theta):.4f}°  r={r:.2f}μm",
            })
        for n, d, why in skipped:
            print(f"  d-spacing #{n} (d={d:.4f}Å) skipped: {why}")
        if not all_rings:
            print("No rings found from d-spacings (all out of range)")
            return
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
