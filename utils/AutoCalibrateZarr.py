#!/usr/bin/env python
"""
AutoCalibrateZarr — Automated detector geometry calibration using WAXS calibrant rings.

Accepts Zarr .zip, HDF5, GE binary, or TIFF images.  File format is auto-detected
from the extension.  Calls CalibrantPanelShiftsOMP (C/OpenMP) for the heavy lifting,
including multi-iteration refinement, outlier ring rejection, and doublet detection.

Usage examples
--------------
  # Minimal (auto-detect everything)
  python AutoCalibrateZarr.py --data CeO2_00001.zip

  # Non-Zarr input (auto-converts, needs param file)
  python AutoCalibrateZarr.py --data CeO2.h5 --params setup.txt

  # With hints for faster convergence
  python AutoCalibrateZarr.py --data CeO2.tif --params ps.txt \\
      --lsd-guess 1200000 --bc-guess 1024 1024

Legacy syntax with -dataFN, -paramFN, etc. is still accepted.
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # macOS: prevent dual-libomp abort

# diplib MUST be imported before numpy/numba/scipy/skimage so that diplib's
# libomp.dylib loads first.  KMP_DUPLICATE_LIB_OK then lets numba's libomp
# coexist without aborting.  Reversing this order causes segfaults on macOS.
try:
    import diplib as dip
    _HAS_DIPLIB = True
except ImportError:
    _HAS_DIPLIB = False

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import zarr
import subprocess
from skimage import measure
import argparse
import sys
import midas_config
midas_config.run_startup_checks()
import pandas as pd
from PIL import Image
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
import traceback
import h5py
import io
import re
import numba
from numba import jit
from scipy import ndimage
import multiprocessing as mp
from functools import partial


def _safe_median_filter(data, kernel_size=101, n_iters=5):
    """Run diplib MedianFilter; fall back to scipy if diplib unavailable."""
    if not _HAS_DIPLIB:
        logger.warning("diplib not available, using scipy median_filter (slower)")
        for _ in range(n_iters):
            data = ndimage.median_filter(data, size=kernel_size)
        return data

    try:
        dip_img = dip.Image(data)
        ks = [kernel_size, kernel_size]
        for _ in range(n_iters):
            dip_img = dip.MedianFilter(dip_img, ks)
        return np.asarray(dip_img).astype(np.float64)
    except Exception as e:
        logger.error(f"diplib MedianFilter failed: {e}. Falling back to scipy.")
        for _ in range(n_iters):
            data = ndimage.median_filter(data, size=kernel_size)
        return data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("autocal.log")
    ]
)
logger = logging.getLogger(__name__)

# Get Python executable path
pytpath = sys.executable

# Determine the installation path from the script's location
def get_install_path():
    """Determines the MIDAS installation path based on script location"""
    script_dir = Path(__file__).resolve().parent
    install_dir = script_dir.parent
    return str(install_dir)

INSTALL_PATH = get_install_path()
env = dict(os.environ)

# ---- Calibration state ----
@dataclass
class CalibState:
    """All calibration state in one place — no globals needed."""
    # Geometry (refined iteratively by C code)
    lsd: float = 1000000.0
    ybc: float = 1024.0
    zbc: float = 1024.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    p0: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    p3: float = 0.0
    p4: float = 0.0
    mean_strain: float = 1.0
    std_strain: float = 0.0
    rhod: float = 0.0

    # Ring info
    n_planes: int = 0
    rings_to_exclude: list = field(default_factory=list)

    # Image / file
    nr_pixels_y: int = 0
    nr_pixels_z: int = 0
    skip_frame: int = 0
    space_group: int = 225
    px: float = 200.0
    latc: np.ndarray = field(default_factory=lambda: np.zeros(6))
    wavelength: float = 0.0
    midas_dtype: int = 1

    # File paths
    folder: str = ''
    fstem: str = ''
    ext: str = ''
    dark_name: str = ''
    fnumber: int = 1
    pad: int = 6
    data_loc: str = ''
    dark_loc: str = ''

    # Panel
    panel_params: list = field(default_factory=list)
    panel_shifts_file: str = ''
    panel_grid: object = None          # PanelGrid from auto-detection
    tol_shifts: float = 3.0
    tol_rotation: float = 1.0
    per_panel_lsd: int = 1
    per_panel_distortion: int = 1
    fix_panel_id: int = -1             # -1 = auto (closest to BC)

    # Bad pixel / gap (B2 fix: declared fields instead of ad-hoc attrs)
    bad_px_intensity: float = field(default_factory=lambda: float('nan'))
    gap_intensity: float = field(default_factory=lambda: float('nan'))
    bad_gap_arr: list = field(default_factory=list)
    mask_file: str = ''

    # HDF5 output
    h5_file: object = None

    # Image transforms
    im_trans_opt: list = field(default_factory=lambda: [0])

    @property
    def bc(self) -> str:
        """Backward-compat: 'ybc zbc' string for param file writing."""
        return f'{self.ybc} {self.zbc}'


@dataclass
class PanelGrid:
    """Auto-detected panel layout from a mask file."""
    n_panels_y: int = 0   # number of panels along Y (columns)
    n_panels_z: int = 0   # number of panels along Z (rows)
    panel_size_y: int = 0  # panel width in pixels (along Y/cols)
    panel_size_z: int = 0  # panel height in pixels (along Z/rows)
    gaps_y: list = field(default_factory=list)  # column gaps (n_panels_y - 1)
    gaps_z: list = field(default_factory=list)  # row gaps (n_panels_z - 1)
    fix_panel_id: int = 0  # panel to fix (closest to center)
    n_panels_total: int = 0
    row_starts: list = field(default_factory=list)
    col_starts: list = field(default_factory=list)


def detect_panels_from_mask(mask_file, min_panel_pixels=1000, bc_guess=None):
    """Auto-detect panel layout from a mask TIFF.

    Convention: 0 = good pixel, non-zero = bad pixel.
    Returns PanelGrid or None if < 2 panels found.
    """
    mask_img = np.array(Image.open(mask_file))
    good = (mask_img == 0).astype(np.int32)
    labeled, n_regions = ndimage.label(good)

    if n_regions < 2:
        logger.info(f"Mask has {n_regions} regions — no multi-panel layout detected")
        return None

    slices = ndimage.find_objects(labeled)

    # Collect bounding boxes of substantial panels
    panels = []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        region = (labeled[sl] == (i + 1))
        n_px = np.sum(region)
        if n_px < min_panel_pixels:
            continue
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        h, w = r1 - r0, c1 - c0
        fill = n_px / (h * w)
        if fill < 0.9:  # skip non-rectangular regions
            continue
        panels.append((r0, r1, c0, c1, h, w))

    if len(panels) < 2:
        logger.info(f"Only {len(panels)} valid panels found — skipping panel detection")
        return None

    # Cluster row starts and col starts (tolerance 2px for alignment jitter)
    def cluster_values(vals, tol=2):
        vals = sorted(set(vals))
        clusters = []
        for v in vals:
            if not clusters or v - clusters[-1] > tol:
                clusters.append(v)
        return clusters

    row_starts = cluster_values([p[0] for p in panels])
    col_starts = cluster_values([p[2] for p in panels])
    n_rows = len(row_starts)
    n_cols = len(col_starts)

    if n_rows * n_cols < len(panels) * 0.8:
        logger.warning(f"Grid detection mismatch: {n_rows}×{n_cols} = {n_rows*n_cols} "
                       f"vs {len(panels)} panels. Skipping.")
        return None

    # Panel dimensions: median height and width
    panel_h = int(np.median([p[4] for p in panels]))
    panel_w = int(np.median([p[5] for p in panels]))

    # Compute gaps
    gaps_z = [row_starts[i+1] - (row_starts[i] + panel_h) for i in range(n_rows - 1)]
    gaps_y = [col_starts[i+1] - (col_starts[i] + panel_w) for i in range(n_cols - 1)]

    # Ensure gaps are non-negative (clamp to 0)
    gaps_z = [max(0, g) for g in gaps_z]
    gaps_y = [max(0, g) for g in gaps_y]

    # Find panel closest to center (or BC guess) for fixPanel
    cy = mask_img.shape[1] / 2.0  # center col = Y
    cz = mask_img.shape[0] / 2.0  # center row = Z
    if bc_guess is not None and bc_guess[0] > 0 and bc_guess[1] > 0:
        cy, cz = bc_guess[0], bc_guess[1]  # ybc, zbc

    best_panel_id = 0
    best_dist = float('inf')
    panel_id = 0
    for ri, rs in enumerate(row_starts):
        for ci, cs in enumerate(col_starts):
            pc_z = rs + panel_h / 2.0
            pc_y = cs + panel_w / 2.0
            d = (pc_y - cy)**2 + (pc_z - cz)**2
            if d < best_dist:
                best_dist = d
                best_panel_id = panel_id
            panel_id += 1

    pg = PanelGrid(
        n_panels_y=n_cols,
        n_panels_z=n_rows,
        panel_size_y=panel_w,
        panel_size_z=panel_h,
        gaps_y=gaps_y,
        gaps_z=gaps_z,
        fix_panel_id=best_panel_id,
        n_panels_total=n_rows * n_cols,
        row_starts=row_starts,
        col_starts=col_starts,
    )

    logger.info(f"Detected panel grid: {n_rows}×{n_cols} = {n_rows*n_cols} panels, "
                f"size={panel_h}×{panel_w}px, "
                f"gapsZ={gaps_z}, gapsY={gaps_y}, "
                f"fixPanel={best_panel_id}")
    return pg

# ---- Blocking 2×2 image viewer (PyQt6 + pyqtgraph) ----
COLORMAPS = ['viridis', 'inferno', 'plasma', 'magma', 'cividis',
             'gray', 'hot', 'cool', 'bone', 'copper']

_qapp = None  # lazy singleton

def _ensure_qapp():
    """Create QApplication if needed."""
    global _qapp
    from PyQt6.QtWidgets import QApplication
    if _qapp is None:
        _qapp = QApplication.instance() or QApplication(sys.argv)
    return _qapp


class CalibImageViewer:
    """Blocking PyQt6/pyqtgraph 2×2 image viewer with shared zoom.

    Panels: Raw | Background | Corrected | Corrected + Rings
    Controls: colormap dropdown, clim min/max, Continue button.
    show_and_wait() blocks until 'Continue' is clicked.
    """

    def __init__(self, title='AutoCalibrateZarr'):
        _ensure_qapp()
        import pyqtgraph as pg
        from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                                     QHBoxLayout, QComboBox, QLabel,
                                     QLineEdit, QPushButton)

        self._pg = pg
        self.win = QMainWindow()
        self.win.setWindowTitle(title)
        self.win.resize(1400, 900)
        self._continued = False

        central = QWidget()
        self.win.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Control bar
        controls = QHBoxLayout()

        controls.addWidget(QLabel('Colormap:'))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText('viridis')
        self.cmap_combo.currentTextChanged.connect(self._on_cmap)
        controls.addWidget(self.cmap_combo)

        controls.addWidget(QLabel('  Clim Min:'))
        self.edit_cmin = QLineEdit()
        self.edit_cmin.setPlaceholderText('auto')
        self.edit_cmin.setFixedWidth(80)
        self.edit_cmin.editingFinished.connect(self._on_clim)
        controls.addWidget(self.edit_cmin)

        controls.addWidget(QLabel('  Max:'))
        self.edit_cmax = QLineEdit()
        self.edit_cmax.setPlaceholderText('auto')
        self.edit_cmax.setFixedWidth(80)
        self.edit_cmax.editingFinished.connect(self._on_clim)
        controls.addWidget(self.edit_cmax)

        auto_btn = QPushButton('Auto Range')
        auto_btn.clicked.connect(self._on_auto)
        controls.addWidget(auto_btn)

        controls.addStretch()

        self.continue_btn = QPushButton('▶ Continue')
        self.continue_btn.setStyleSheet(
            'QPushButton { background-color: #2ecc71; color: white; '
            'font-weight: bold; padding: 6px 20px; border-radius: 4px; }')
        self.continue_btn.clicked.connect(self._on_continue)
        controls.addWidget(self.continue_btn)

        layout.addLayout(controls)

        # 2×2 pyqtgraph grid with shared zoom
        gw = pg.GraphicsLayoutWidget()
        layout.addWidget(gw)

        self.panels = {}
        self.items = {}
        labels = [('Raw', 0, 0), ('Background', 0, 1),
                  ('Corrected', 1, 0), ('Corrected + Rings', 1, 1)]

        first_vb = None
        for name, row, col in labels:
            p = gw.addPlot(row=row, col=col, title=name)
            p.setAspectLocked(True)
            p.invertY(False)
            img = pg.ImageItem()
            p.addItem(img)
            self.panels[name] = p
            self.items[name] = img
            if first_vb is None:
                first_vb = p.getViewBox()
            else:
                p.getViewBox().setXLink(first_vb)
                p.getViewBox().setYLink(first_vb)

        self._ring_items = []
        self._lut = None
        self._on_cmap('viridis')  # set initial colormap LUT
        self.win.show()

    def _set_image(self, name, img):
        """Set image on a panel and auto-range the view."""
        item = self.items[name]
        # pyqtgraph ImageItem expects (x, y) = (cols, rows), so transpose
        item.setImage(img.T, autoLevels=False)
        if self._lut is not None:
            item.setLookupTable(self._lut)
        # Apply current clim if set, otherwise auto
        self._apply_clim_to(item, img)
        # Auto-range the first panel's view (linked panels follow)
        if name == 'Raw':
            self.panels[name].getViewBox().autoRange()

    def _apply_clim_to(self, item, img=None):
        """Apply current clim values to an ImageItem."""
        try:
            vmin = float(self.edit_cmin.text()) if self.edit_cmin.text().strip() else None
        except ValueError:
            vmin = None
        try:
            vmax = float(self.edit_cmax.text()) if self.edit_cmax.text().strip() else None
        except ValueError:
            vmax = None
        if vmin is not None and vmax is not None:
            item.setLevels([vmin, vmax])
        elif img is not None:
            item.setLevels([np.nanmin(img), np.nanmax(img)])

    def set_raw(self, img):
        self._raw_img = img
        self._set_image('Raw', img)
        # Auto clim from raw
        pos = img[img > 0] if np.any(img > 0) else img.ravel()
        m, s = np.median(pos), np.std(pos)
        self.edit_cmin.setText(f'{m:.1f}')
        self.edit_cmax.setText(f'{m + 2*s:.1f}')
        self._on_clim()

    def set_bg(self, img):
        self._set_image('Background', img)

    def set_corr(self, img):
        self._set_image('Corrected', img)

    def set_rings(self, img, ring_radii, bc, rings_to_exclude=None):
        self._set_image('Corrected + Rings', img)
        pg = self._pg
        p = self.panels['Corrected + Rings']
        # Remove old ring overlays
        for item in self._ring_items:
            p.removeItem(item)
        self._ring_items = []
        # Draw new rings as ellipses (CircleROI)
        for i, rad in enumerate(ring_radii):
            ring_nr = i + 1
            if rings_to_exclude and ring_nr in rings_to_exclude:
                continue
            # pyqtgraph coords: x=col, y=row → bc[1]=col, bc[0]=row
            r_px = rad / self._px if hasattr(self, '_px') else rad
            circle = pg.CircleROI(
                [bc[1] - r_px, bc[0] - r_px], [2 * r_px, 2 * r_px],
                pen=pg.mkPen('c', width=1), movable=False)
            # Remove resize/rotate handles
            for h in circle.getHandles():
                circle.removeHandle(h)
            p.addItem(circle)
            self._ring_items.append(circle)

    def show_and_wait(self):
        """Block until user clicks Continue."""
        self.continue_btn.setEnabled(True)
        _qapp.processEvents()
        _qapp.exec()

    def _on_continue(self):
        self._continued = True
        _qapp.quit()

    def _on_cmap(self, name):
        pg = self._pg
        try:
            cmap = pg.colormap.get(name, source='matplotlib')
        except Exception:
            cmap = pg.colormap.get('viridis', source='matplotlib')
        self._lut = cmap.getLookupTable(nPts=256)
        for item in self.items.values():
            item.setLookupTable(self._lut)

    def _on_clim(self):
        try:
            vmin = float(self.edit_cmin.text()) if self.edit_cmin.text().strip() else None
        except ValueError:
            vmin = None
        try:
            vmax = float(self.edit_cmax.text()) if self.edit_cmax.text().strip() else None
        except ValueError:
            vmax = None
        if vmin is not None and vmax is not None:
            for item in self.items.values():
                item.setLevels([vmin, vmax])

    def _on_auto(self):
        if hasattr(self, '_raw_img'):
            pos = self._raw_img[self._raw_img > 0] if np.any(self._raw_img > 0) else self._raw_img.ravel()
            m, s = np.median(pos), np.std(pos)
            self.edit_cmin.setText(f'{m:.1f}')
            self.edit_cmax.setText(f'{m + 2*s:.1f}')
            self._on_clim()

    def process_events(self):
        """Process Qt events (non-blocking)."""
        if _qapp:
            _qapp.processEvents()


class MyParser(argparse.ArgumentParser):
    """Custom argument parser with improved error handling"""
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)


# ---- Known calibrant materials ----
CALIBRANTS = {
    'ceo2': {
        'name': 'CeO2',
        'space_group': 225,
        'lattice': np.array([5.4116, 5.4116, 5.4116, 90.0, 90.0, 90.0]),
    },
    'lab6': {
        'name': 'LaB6',
        'space_group': 221,
        'lattice': np.array([4.1569, 4.1569, 4.1569, 90.0, 90.0, 90.0]),
    },
}

# Filename patterns → calibrant key  (checked case-insensitively)
_CALIBRANT_PATTERNS = [
    # LaB6 variants (check before CeO2 since 'la' is less common)
    ('lab6',  'lab6'),
    ('lab_6', 'lab6'),
    ('lab-6', 'lab6'),
    ('lanthanumhexaboride', 'lab6'),
    # CeO2 variants
    ('ceo2',  'ceo2'),
    ('ceo_2', 'ceo2'),
    ('ceo-2', 'ceo2'),
    ('ceriumoxide', 'ceo2'),
    ('ceria', 'ceo2'),
]


def detect_calibrant(filename):
    """Detect calibrant material from filename.

    Returns calibrant dict or None.
    """
    base = Path(filename).stem.lower()
    for pattern, key in _CALIBRANT_PATTERNS:
        if pattern in base:
            cal = CALIBRANTS[key]
            logger.info(f"Auto-detected calibrant from filename: {cal['name']} "
                        f"(SpaceGroup {cal['space_group']})")
            return cal
    return None


# hc in keV·Å
_HC_KEV_ANGSTROM = 12.3984198


def parse_filename_hints(filename):
    """Extract energy (keV) and distance (mm) from filename tokens.

    Recognizes patterns like:
      71p676keV  or  71.676keV  →  energy 71.676 keV  →  wavelength 0.17301 Å
      657mm                     →  distance 657 mm    →  Lsd 657000 µm

    Returns dict with keys 'wavelength' (Å) and/or 'lsd' (µm), only for
    values that were found.
    """
    base = Path(filename).stem
    hints = {}

    # Energy:  match e.g. '71p676keV', '71.676keV', '30keV'
    # Uses 'p' or '.' as decimal separator
    energy_match = re.search(
        r'(?:^|[_\-])([\d]+(?:[p.][\d]+)?)keV(?:[_\-.]|$)',
        base, re.IGNORECASE
    )
    if energy_match:
        energy_str = energy_match.group(1).replace('p', '.')
        energy_kev = float(energy_str)
        if energy_kev > 0:
            wavelength = _HC_KEV_ANGSTROM / energy_kev
            hints['wavelength'] = wavelength
            logger.info(f"Auto-detected energy from filename: {energy_kev} keV "
                        f"→ wavelength {wavelength:.5f} Å")

    # Distance:  match e.g. '657mm', '210mm'
    dist_match = re.search(
        r'(?:^|[_\-])([\d]+(?:[p.][\d]+)?)mm(?:[_\-.]|$)',
        base, re.IGNORECASE
    )
    if dist_match:
        dist_str = dist_match.group(1).replace('p', '.')
        dist_mm = float(dist_str)
        if dist_mm > 0:
            lsd_um = dist_mm * 1000.0
            hints['lsd'] = lsd_um
            logger.info(f"Auto-detected distance from filename: {dist_mm} mm "
                        f"→ Lsd {lsd_um:.0f} µm")

    return hints


# ---- File format detection ----
def detect_format(path):
    """Auto-detect file format from extension.

    Returns 0=Zarr, 1=HDF5, 2=GE binary, 3=TIFF
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == '.zip':
        return 0
    elif ext in ('.h5', '.hdf5', '.hdf', '.nxs', '.nx'):
        return 1
    elif ext.startswith('.ge') or ext in ('.raw', '.bin'):
        return 2
    elif ext in ('.tif', '.tiff'):
        return 3
    else:
        logger.warning(f"Cannot auto-detect format for '{ext}', assuming Zarr zip")
        return 0


# ---- HDF5 save helpers ----
def save_data_to_hdf5(data, h5file, dataset_name, metadata=None):
    """Save data arrays to an HDF5 file."""
    try:
        group = h5file.create_group(dataset_name)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression='gzip')
            else:
                group.attrs[key] = value
        if metadata:
            for key, value in metadata.items():
                group.attrs[key] = value
    except Exception as e:
        logger.error(f"Error saving data to HDF5: {e}")

def save_raw_image_data(raw, h5file):
    save_data_to_hdf5({'image': raw}, h5file, 'raw_image')

def save_background_data(background, h5file):
    save_data_to_hdf5({'image': background}, h5file, 'background')

def save_threshold_data(thresh, h5file):
    save_data_to_hdf5({'image': thresh}, h5file, 'threshold')

def save_ring_data(thresh, bc_new, sim_rads, h5file):
    save_data_to_hdf5({
        'image': thresh,
        'beam_center_y': bc_new[0],
        'beam_center_z': bc_new[1],
        'ring_radii': sim_rads
    }, h5file, 'ring_overlay')

def save_results_dataframe(df, h5file):
    try:
        group = h5file.create_group('results')
        for col in df.columns:
            group.create_dataset(col, data=df[col].values, compression='gzip')
    except Exception as e:
        logger.error(f"Error saving results dataframe: {e}")


# ---- File readers ----
def fileReader(f, dset, skip_frame=0):
    """Read data from Zarr file with handling for skipFrames."""
    data = f[dset][:]
    data = data[skip_frame:, :, :]
    _, nz, ny = data.shape
    data[data < 1] = 1
    return np.mean(data, axis=0), ny, nz


def detect_data_type(data_fn):
    """Auto-detect CalibrantPanelShiftsOMP DataType from file.

    Returns int: 1=uint16-raw, 6=tiff-uint32, 7=tiff-uint8,
                 8=HDF5, 9=tiff-uint16
    """
    ext = Path(data_fn).suffix.lower()
    if ext in ('.h5', '.hdf5', '.hdf', '.nxs'):
        return 8
    if ext in ('.tif', '.tiff'):
        try:
            import tifffile
            with tifffile.TiffFile(data_fn) as tif:
                dtype = tif.pages[0].dtype
                if dtype == np.uint8:
                    return 7
                if dtype == np.uint16:
                    return 9
                if dtype == np.uint32:
                    return 6
        except ImportError:
            # Fallback: use PIL
            img = Image.open(data_fn)
            mode = img.mode
            if mode == 'I':         # 32-bit
                return 6
            if mode == 'L':         # 8-bit
                return 7
        return 6  # default for TIFF
    if ext.startswith('.ge'):
        return 1  # uint16 raw binary
    return 1  # default


def read_image_for_estimation(data_fn, dark_fn, data_loc, dark_loc,
                              skip_frame=0, data_type=None):
    """Read image data into numpy for beam center / ring detection.

    Supports HDF5, TIFF, GE binary, and Zarr.
    Returns: raw, dark, ny, nz
    """
    ext = Path(data_fn).suffix.lower()

    if ext in ('.h5', '.hdf5', '.hdf', '.nxs'):
        # Direct HDF5 reading
        dl = data_loc or 'exchange/data'
        dkl = dark_loc or 'exchange/dark'
        with h5py.File(data_fn, 'r') as f:
            data = f[dl][:]
            data = data[skip_frame:]
            if data.ndim == 3:
                raw = np.mean(data, axis=0).astype(np.float64)
            else:
                raw = data.astype(np.float64)
            nz, ny = raw.shape

            if dkl in f:
                dark_data = f[dkl][:]
                if dark_data.ndim == 3:
                    dark = np.mean(dark_data, axis=0).astype(np.float64)
                else:
                    dark = dark_data.astype(np.float64)
            else:
                dark = np.zeros_like(raw)

        # Separate dark file
        if dark_fn and os.path.exists(dark_fn):
            with h5py.File(dark_fn, 'r') as f:
                dkl2 = dark_loc or 'exchange/dark'
                dark_data = f[dkl2][:]
                if dark_data.ndim == 3:
                    dark = np.mean(dark_data, axis=0).astype(np.float64)
                else:
                    dark = dark_data.astype(np.float64)

        raw[raw < 1] = 1
        return raw, dark, ny, nz

    elif ext in ('.tif', '.tiff'):
        img = np.array(Image.open(data_fn)).astype(np.float64)
        nz, ny = img.shape
        img[img < 1] = 1
        dark = np.zeros_like(img)
        if dark_fn and os.path.exists(dark_fn):
            dark = np.array(Image.open(dark_fn)).astype(np.float64)
        return img, dark, ny, nz

    elif ext == '.zip' or ext == '.zarr':
        # Zarr path (backward compat)
        import zarr
        f = zarr.open(data_fn, mode='r')
        raw, ny, nz = fileReader(f, '/exchange/data', skip_frame)
        dark, _, _ = fileReader(f, '/exchange/dark', skip_frame)
        return raw, dark, ny, nz

    else:
        # GE binary or other raw format
        dt = data_type or 1
        dtype_map = {1: np.uint16, 2: np.float64, 3: np.float32,
                     4: np.uint32, 5: np.int32}
        np_dtype = dtype_map.get(dt, np.uint16)
        raw = np.fromfile(data_fn, dtype=np_dtype)
        # Assume square detector
        side = int(np.sqrt(len(raw)))
        if side * side != len(raw):
            # Try with 8192-byte header
            raw = np.fromfile(data_fn, dtype=np_dtype, offset=8192)
            side = int(np.sqrt(len(raw)))
        raw = raw.reshape(side, side).astype(np.float64)
        nz, ny = raw.shape
        raw[raw < 1] = 1
        dark = np.zeros_like(raw)
        if dark_fn and os.path.exists(dark_fn):
            dark = np.fromfile(dark_fn, dtype=np_dtype)
            if len(dark) > side * side:
                dark = np.fromfile(dark_fn, dtype=np_dtype, offset=8192)
            dark = dark[:side*side].reshape(side, side).astype(np.float64)
        return raw, dark, ny, nz


def run_get_hkl_list_cli(sg, latc, wavelength, lsd, max_ring_rad):
    """Run GetHKLList in CLI mode (no temp param file needed).

    Returns numpy array with columns:
        h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius
    """
    hkl_bin = os.path.join(INSTALL_PATH, 'FF_HEDM/bin/GetHKLList')
    cmd = [
        hkl_bin,
        '--sg', str(int(sg)),
        '--lp', *[f'{x:.6f}' for x in latc],
        '--wl', f'{wavelength:.6f}',
        '--lsd', f'{lsd:.1f}',
        '--maxR', f'{max_ring_rad:.1f}',
        '--stdout',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=env, check=True)
        lines = result.stdout.strip().split('\n')
        # Filter to only numeric data lines (skip info/header lines)
        data_lines = []
        for l in lines:
            l = l.strip()
            if not l:
                continue
            try:
                float(l.split()[0])
                data_lines.append(l)
            except (ValueError, IndexError):
                continue  # skip non-numeric lines (headers, info)
        if not data_lines:
            logger.error(f"GetHKLList returned no numeric data. "
                         f"Full output:\n{result.stdout}")
            raise RuntimeError("GetHKLList returned no data")
        return np.array([[float(x) for x in l.split()] for l in data_lines])
    except subprocess.CalledProcessError as e:
        logger.error(f"GetHKLList failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error running GetHKLList CLI: {e}")
        raise


def generateZip(resFol, pfn, dfn='', darkfn='', dloc='', darkloc='',
                nchunks=-1, preproc=-1, outf='ZipOut.txt', errf='ZipErr.txt',
                NrPixelsY=0, NrPixelsZ=0):
    """Generate a Zarr zip file from other file formats."""
    cmd = [
        pytpath,
        os.path.join(INSTALL_PATH, 'utils/ffGenerateZipRefactor.py'),
        '-resultFolder', resFol,
        '-paramFN', pfn
    ]
    if dfn:
        cmd.extend(['-dataFN', dfn])
    if darkfn:
        cmd.extend(['-darkFN', darkfn])
    if dloc:
        cmd.extend(['-dataLoc', dloc])
    if darkloc:
        cmd.extend(['-darkLoc', darkloc])
    if nchunks != -1:
        cmd.extend(['-numFrameChunks', str(nchunks)])
    if preproc != -1:
        cmd.extend(['-preProcThresh', str(preproc)])
    if NrPixelsY != 0:
        cmd.extend(['-numPxY', str(NrPixelsY)])
    if NrPixelsZ != 0:
        cmd.extend(['-numPxZ', str(NrPixelsZ)])

    cmd_str = ' '.join(cmd)
    outfile_path = os.path.join(resFol, outf)
    errfile_path = os.path.join(resFol, errf)

    try:
        with open(outfile_path, 'w') as outfile, open(errfile_path, 'w') as errfile:
            return_code = subprocess.call(cmd_str, shell=True, stdout=outfile, stderr=errfile)

        if return_code != 0:
            with open(errfile_path, 'r') as errfile:
                error_content = errfile.read()
            logger.error(f"Error executing generateZip - Return code: {return_code}")
            logger.error(f"Error content: {error_content}")
            sys.exit(1)

        with open(outfile_path, 'r') as outfile:
            lines = outfile.readlines()
            for line in lines:
                if "Error" in line or "error" in line or "ERROR" in line:
                    logger.error(f"Error detected in generateZip output: {line.strip()}")
                    sys.exit(1)
            for line in lines:
                if line.startswith('OutputZipName'):
                    return line.split()[1]

        logger.error("Could not find output zip name in generateZip output")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Exception in generateZip: {e}")
        sys.exit(1)


def process_calibrant_output(output_file, state):
    """Parse CalibrantPanelShiftsOMP stdout and update state with refined params."""
    try:
        with open(output_file) as f:
            output = f.readlines()

        useful = 0
        for line in output:
            if 'Number of planes being considered' in line and state.n_planes == 0:
                state.n_planes = int(line.rstrip().split()[-1][:-1])
            if useful == 1:
                if 'Copy to par' in line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                if 'Lsd ' in line:
                    state.lsd = float(parts[1])
                if 'BC ' in line:
                    state.ybc = float(parts[1])
                    state.zbc = float(parts[2])
                if 'ty ' in line:
                    state.ty = float(parts[1])
                if 'tz ' in line:
                    state.tz = float(parts[1])
                if 'p0 ' in line:
                    state.p0 = float(parts[1])
                if 'p1 ' in line:
                    state.p1 = float(parts[1])
                if 'p2 ' in line:
                    state.p2 = float(parts[1])
                if 'p3 ' in line:
                    state.p3 = float(parts[1])
                if 'p4 ' in line:
                    state.p4 = float(parts[1])
                if 'RhoD ' in line:
                    state.rhod = float(parts[1])
                if 'MeanStrain ' in line:
                    state.mean_strain = float(parts[1]) / 1e6
                if 'StdStrain ' in line:
                    state.std_strain = float(parts[1]) / 1e6
            if 'Mean Values' in line:
                useful = 1
    except Exception as e:
        logger.error(f"Error processing calibrant output: {e}")


# ---- Beam center and ring detection (JIT-accelerated) ----

@jit(nopython=True)
def _compute_distances(coords, point):
    distances = np.zeros(len(coords))
    for i in range(len(coords)):
        diff = coords[i] - point
        distances[i] = np.sqrt(diff[0]**2 + diff[1]**2)
    return distances


@jit(nopython=True)
def _find_center_point(coords, bbox):
    edge_indices = np.where(coords[:, 0] == bbox[0])[0]
    if len(edge_indices) == 0:
        return np.array([-1.0, -1.0])
    edgecoorda = coords[edge_indices[len(edge_indices)//2]]
    distances = _compute_distances(coords, edgecoorda)
    furthest_idx = np.argmax(distances)
    edgecoordb = coords[furthest_idx]
    max_distance = distances[furthest_idx]
    arcLen = max_distance / 2
    candidate_indices = np.where(np.abs(distances - arcLen) < 2)[0]
    if len(candidate_indices) == 0:
        return np.array([-1.0, -1.0])
    candidatea = coords[candidate_indices[len(candidate_indices)//2]]
    candidateb = candidatea
    midpointa = (edgecoorda + candidatea) / 2
    midpointb = (edgecoordb + candidateb) / 2
    x1, y1 = edgecoorda
    x2, y2 = candidatea
    x3, y3 = candidateb
    x4, y4 = edgecoordb
    x5, y5 = midpointa
    x6, y6 = midpointb
    if y4 == y3 or y2 == y1:
        return np.array([-1.0, -1.0])
    m1 = (x1 - x2) / (y2 - y1)
    m2 = (x3 - x4) / (y4 - y3)
    if abs(m1 - m2) < 1e-10:
        return np.array([-1.0, -1.0])
    x = (y6 - y5 + m1 * x5 - m2 * x6) / (m1 - m2)
    y = m1 * (x - x5) + y5
    return np.array([x, y])


def _process_single_label(label_info):
    label, mask, minArea = label_info
    if np.sum(mask) < minArea:
        return None
    coords = np.array(np.where(mask)).T
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    bbox = (rows.min(), cols.min(), rows.max(), cols.max())
    center = _find_center_point(coords, bbox)
    if center[0] < 0 or center[1] < 0:
        return None
    return center


def detect_beam_center_optimized(thresh, minArea, num_processes=None):
    """Detect beam center from thresholded image."""
    try:
        if not hasattr(detect_beam_center_optimized, "_initialized"):
            dummy_coords = np.array([[0, 0], [1, 1], [2, 2]])
            dummy_bbox = (0, 0, 2, 2)
            _ = _find_center_point(dummy_coords, dummy_bbox)
            detect_beam_center_optimized._initialized = True

        labels, nlabels = ndimage.label(thresh)
        if nlabels == 0:
            logger.warning("No beam centers detected!")
            return np.array([0.0, 0.0])

        label_data = []
        for label in range(1, nlabels + 1):
            mask = (labels == label)
            label_data.append((label, mask, minArea))

        if num_processes is None:
            num_processes = min(mp.cpu_count(), len(label_data))

        if len(label_data) > 1 and num_processes > 1:
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(_process_single_label, label_data)
            all_centers = [r for r in results if r is not None]
        else:
            all_centers = []
            for data in label_data:
                result = _process_single_label(data)
                if result is not None:
                    all_centers.append(result)

        if not all_centers:
            logger.warning("No valid beam centers calculated!")
            return np.array([0.0, 0.0])

        centers_array = np.array(all_centers)
        return np.array([np.median(centers_array[:, 0]),
                         np.median(centers_array[:, 1])])
    except Exception as e:
        logger.error(f"Error detecting beam center: {e}")
        return np.array([0.0, 0.0])


def detect_ring_radii(labels, props, bc_computed, minArea):
    """Detect ring radii from labeled image and beam center."""
    try:
        rads = []
        nlabels = len(props) + 1
        for label in range(1, nlabels):
            if np.sum(labels == label) > minArea:
                coords = props[label-1].coords
                rad = np.mean(np.linalg.norm(
                    np.transpose(coords) - bc_computed[:, None], axis=0))
                toAdd = True
                for existing_rad in rads:
                    if np.abs(existing_rad - rad) < 20:
                        toAdd = False
                        break
                if toAdd:
                    rads.append(rad)
        if not rads:
            logger.warning("No rings detected!")
            return np.array([])
        return np.sort(np.array(rads))
    except Exception as e:
        logger.error(f"Error detecting ring radii: {e}")
        return np.array([])


def estimate_lsd(rads, sim_rads, sim_rad_ratios, firstRing, initialLsd):
    """Estimate sample-to-detector distance from ring radii."""
    if len(rads) == 0:
        return initialLsd
    try:
        radRatios = rads / rads[0]
        lsds = []
        for i in range(len(rads)):
            bestMatch = 10000
            bestRowNr = -1
            for j in range(firstRing - 1, len(sim_rads)):
                if np.abs(1 - (radRatios[i] / sim_rad_ratios[j])) < 0.02:
                    match_quality = np.abs(1 - (radRatios[i] / sim_rad_ratios[j]))
                    if match_quality < bestMatch:
                        bestMatch = match_quality
                        bestRowNr = j
            if bestRowNr != -1:
                lsds.append(initialLsd * rads[i] / sim_rads[bestRowNr])
        if not lsds:
            logger.warning("Could not estimate Lsd from rings - using initial guess")
            return initialLsd
        return np.median(lsds)
    except Exception as e:
        logger.error(f"Error estimating Lsd: {e}")
        return initialLsd


def create_param_file(output_file, params):
    """Create a parameter file with the given parameters."""
    try:
        with open(output_file, 'w') as pf:
            for key, value in params.items():
                if isinstance(value, (list, np.ndarray)):
                    pf.write(f"{key} {' '.join(map(str, value))}\n")
                else:
                    pf.write(f"{key} {value}\n")
    except Exception as e:
        logger.error(f"Error creating parameter file {output_file}: {e}")


def process_tiff_input(dataFN, badPxIntensity, gapIntensity, darkFN=''):
    """Process TIFF input file and convert to GE format."""
    bad_gap_arr = []
    NrPixelsY = 0
    NrPixelsZ = 0
    try:
        img = Image.open(dataFN)
        logger.info("Data was a tiff image. Will convert to a ge file.")
        img = np.array(img)

        if not np.isnan(badPxIntensity) and not np.isnan(gapIntensity):
            bad_gap_arr = img == badPxIntensity
            bad_gap_arr = np.logical_or(bad_gap_arr, img == gapIntensity)

        if img.shape[1] != 2048:
            NrPixelsY = img.shape[1]
        if img.shape[0] != 2048:
            NrPixelsZ = img.shape[0]

        ge_file = f"{dataFN}.ge"
        with open(ge_file, 'wb') as f:
            f.write(b'\x00' * 8192)
            img.tofile(f)

        ge_dark_file = ''
        if darkFN:
            try:
                dark_img = Image.open(darkFN)
                dark_img = np.array(dark_img)
                ge_dark_file = f"{darkFN}.ge"
                with open(ge_dark_file, 'wb') as f:
                    f.write(b'\x00' * 8192)
                    dark_img.tofile(f)
                logger.info(f"Converted dark TIFF to GE: {ge_dark_file}")
            except Exception as e:
                logger.error(f"Error processing dark TIFF {darkFN}: {e}")
                ge_dark_file = ''

        return ge_file, ge_dark_file, NrPixelsY, NrPixelsZ, bad_gap_arr
    except Exception as e:
        logger.error(f"Error processing TIFF input: {e}")
        sys.exit(1)


def run_get_hkl_list(param_file):
    """Run GetHKLList with proper error handling."""
    try:
        cmd = f"{os.path.join(INSTALL_PATH, 'FF_HEDM/bin/GetHKLList')} {param_file}"
        with open('hkls_screen_out.csv', 'w') as f:
            subprocess.call(cmd, shell=True, env=env, stdout=f)
        return np.genfromtxt('hkls.csv', skip_header=1)
    except Exception as e:
        logger.error(f"Error running GetHKLList: {e}")
        raise


def runMIDAS(rawFN, state, n_iterations=40, mult_factor=2.5,
             doublet_separation=25, outlier_iterations=3,
             eta_bin_size=5.0, max_width=1000, n_cpus=None,
             stage=0, stage_label=''):
    """Run CalibrantPanelShiftsOMP.

    stage=0: full optimization (legacy behavior)
    stage=1: geometry only — Lsd, BC, tilts (tolP=0, no panels)
    stage=2: full — distortion + panels using refined geometry from stage 1
    """
    if n_cpus is None:
        n_cpus = os.cpu_count() or 8

    ps_file = f"{rawFN}ps.txt"

    try:
        with open(ps_file, 'w') as pf:
            # Ring exclusions
            for ringNr in state.rings_to_exclude:
                pf.write(f'RingsToExclude {ringNr}\n')

            # File info
            pf.write(f'Folder {state.folder}\n')
            pf.write(f'FileStem {state.fstem}\n')
            pf.write(f'Ext {state.ext}\n')
            for transOpt in state.im_trans_opt:
                pf.write(f'ImTransOpt {transOpt}\n')
            pf.write(f'Width {max_width}\n')

            # Tolerances — stage 1: lock distortion, stage 2: full
            pf.write('tolTilts 3\n')
            pf.write('tolBC 20\n')
            pf.write('tolLsd 25000\n')
            if stage == 1:
                pf.write('tolP 0\n')  # lock distortion at current values
            else:
                pf.write('tolP 2E-3\n')

            # Current geometry
            pf.write(f'tx {state.tx}\n')
            pf.write(f'ty {state.ty}\n')
            pf.write(f'tz {state.tz}\n')
            pf.write('Wedge 0\n')
            pf.write(f'p0 {state.p0}\n')
            pf.write(f'p1 {state.p1}\n')
            pf.write(f'p2 {state.p2}\n')
            pf.write(f'p3 {state.p3}\n')
            if state.p4 != 0.0:
                pf.write(f'p4 {state.p4}\n')
            pf.write(f'EtaBinSize {eta_bin_size}\n')
            pf.write('HeadSize 0\n')

            # Bad pixel / gap
            if not math.isnan(state.bad_px_intensity):
                pf.write(f'BadPxIntensity {state.bad_px_intensity}\n')
            if not math.isnan(state.gap_intensity):
                pf.write(f'GapIntensity {state.gap_intensity}\n')

            # Data specifics
            pf.write(f'Dark {state.dark_name}\n')
            pf.write(f'StartNr {state.fnumber}\n')
            pf.write(f'EndNr {state.fnumber}\n')
            pf.write(f'Padding {state.pad}\n')
            pf.write(f'NrPixelsY {state.nr_pixels_y}\n')
            pf.write(f'NrPixelsZ {state.nr_pixels_z}\n')
            pf.write(f'px {state.px}\n')
            pf.write(f'Wavelength {state.wavelength}\n')
            pf.write(f'SpaceGroup {state.space_group}\n')
            pf.write(f'DataType {state.midas_dtype}\n')
            pf.write(f'Lsd {state.lsd}\n')
            pf.write(f'RhoD {state.rhod}\n')
            pf.write(f'MultFactor {mult_factor}\n')
            pf.write(f'BC {state.bc}\n')
            pf.write(f'LatticeConstant {" ".join(map(str, state.latc))}\n')

            # Direct file reading params (skip GE conversion)
            if state.data_loc:
                pf.write(f'dataLoc {state.data_loc}\n')
            if state.dark_loc:
                pf.write(f'darkLoc {state.dark_loc}\n')

            # C-side iteration and advanced features
            pf.write(f'nIterations {n_iterations}\n')
            pf.write(f'DoubletSeparation {doublet_separation}\n')
            pf.write(f'OutlierIterations {outlier_iterations}\n')
            pf.write('NormalizeRingWeights 1\n')
            pf.write('MinIndicesForFit 5\n')
            pf.write('WeightByRadius 1\n')
            pf.write('WeightByFitSNR 1\n')
            pf.write('L2Objective 1\n')

            # Panel parameters — only in stage 2 or stage 0
            if stage != 1:
                if state.panel_grid is not None:
                    pg = state.panel_grid
                    pf.write(f'NPanelsY {pg.n_panels_y}\n')
                    pf.write(f'NPanelsZ {pg.n_panels_z}\n')
                    pf.write(f'PanelSizeY {pg.panel_size_y}\n')
                    pf.write(f'PanelSizeZ {pg.panel_size_z}\n')
                    pf.write(f'PanelGapsY {" ".join(str(g) for g in pg.gaps_y)}\n')
                    pf.write(f'PanelGapsZ {" ".join(str(g) for g in pg.gaps_z)}\n')
                    fix_id = state.fix_panel_id if state.fix_panel_id >= 0 else pg.fix_panel_id
                    pf.write(f'FixPanelID {fix_id}\n')
                    pf.write(f'tolShifts {state.tol_shifts}\n')
                    pf.write(f'tolRotation {state.tol_rotation}\n')
                    pf.write(f'PerPanelLsd {state.per_panel_lsd}\n')
                    pf.write(f'PerPanelDistortion {state.per_panel_distortion}\n')
                    if state.panel_shifts_file:
                        pf.write(f'PanelShiftsFile {state.panel_shifts_file}\n')
                elif state.panel_params:
                    for pp in state.panel_params:
                        pf.write(f'{pp}\n')
                    if state.panel_shifts_file:
                        pf.write(f'PanelShiftsFile {state.panel_shifts_file}\n')
                    pf.write(f'tolShifts {state.tol_shifts}\n')

            # Mask file
            if state.mask_file:
                pf.write(f'MaskFile {state.mask_file}\n')

        # Run CalibrantPanelShiftsOMP with all available CPUs
        calibrant_exe = os.path.join(INSTALL_PATH, 'FF_HEDM/bin/CalibrantPanelShiftsOMP')
        calibrant_cmd = f"{calibrant_exe} {ps_file} {n_cpus}"

        logger.info(f"{stage_label}Running CalibrantPanelShiftsOMP with {n_cpus} CPUs, "
                     f"{n_iterations} iterations, DoubletSep={doublet_separation}px")

        # Run and capture output in real-time while also saving to file
        out_file = 'calibrant_screen_out.csv'
        with open(out_file, 'w') as f:
            proc = subprocess.Popen(
                calibrant_cmd, shell=True, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )
            for line in proc.stdout:
                f.write(line)
                # Print iteration progress to terminal
                if 'MeanStrain' in line or 'Iteration' in line or 'Restoring best' in line:
                    print(f"  {line.rstrip()}")
                elif 'Doublet detected' in line:
                    print(f"  {line.rstrip()}")
            proc.wait()

        # Parse output for refined parameters
        process_calibrant_output(out_file, state)

    except Exception as e:
        logger.error(f"Error running MIDAS: {traceback.format_exc()}")


def _make_temp_param_file(args, calibrant, filename_hints):
    """Generate a temporary parameter file from CLI args and auto-detected values.

    This allows running without --params when --px and calibrant/energy
    info are available (from filename or defaults).
    """
    px = args.px if args.px > 0 else 200.0
    wavelength = filename_hints.get('wavelength', 0.0)
    latc = calibrant['lattice']
    sg = calibrant['space_group']

    if wavelength <= 0:
        logger.error("Cannot create temp param file: wavelength not available "
                     "(set energy in filename or provide --params)")
        print("ERROR: Wavelength not available. Either include energy in the "
              "filename (e.g. 71p676keV) or provide --params")
        sys.exit(1)

    temp_fn = '_autocal_temp_params.txt'
    with open(temp_fn, 'w') as f:
        f.write(f"SpaceGroup {sg}\n")
        f.write(f"LatticeParameter {' '.join(str(v) for v in latc)}\n")
        f.write(f"Wavelength {wavelength}\n")
        f.write(f"px {px}\n")
        f.write(f"SkipFrame 0\n")
        f.write(f"tx {args.tx}\n")
    logger.info(f"Auto-generated temp param file: {temp_fn} "
                f"(SG={sg}, px={px}, λ={wavelength:.5f}Å, {calibrant['name']})")
    return temp_fn


def main():
    """Main function to run the automated calibration."""
    try:
        parser = MyParser(
            description='Automated Calibration for WAXS using continuous rings-like signal. '
                        'Accepts Zarr .zip, HDF5, GE binary, or TIFF images. '
                        'File format is auto-detected from the extension.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Primary arguments (new --flag style with backward-compat aliases)
        parser.add_argument('--data', '-dataFN', '-d', type=str, required=True,
                            help='Data file: .zip (Zarr), .h5 (HDF5), .ge* (GE binary), or .tif/.tiff')
        parser.add_argument('--dark', '-darkFN', type=str, default='',
                            help='Separate dark field image file')
        parser.add_argument('--params', '-paramFN', '-p', type=str, default='',
                            help='MIDAS parameter file (required for non-Zarr inputs)')
        parser.add_argument('--data-loc', '-dataLoc', type=str, default='',
                            help='HDF5 dataset path (default: /exchange/data)')
        parser.add_argument('--dark-loc', '-darkLoc', type=str, default='',
                            help='HDF5 dark field dataset path (default: exchange/dark)')

        # Format control
        parser.add_argument('--convert', '-ConvertFile', type=int, default=-1,
                            help='Force format conversion: 0=Zarr, 1=HDF5, 2=GE, 3=TIFF. '
                                 'Default: auto-detect from extension.')

        # Calibration control
        parser.add_argument('--n-iterations', type=int, default=40,
                            help='Number of C-side calibration iterations')
        parser.add_argument('--mult-factor', '-MultFactor', type=float, default=2.5,
                            help='Outlier ring rejection factor (× median strain)')
        parser.add_argument('--doublet-separation', type=float, default=25.0,
                            help='Doublet detection threshold (pixels)')
        parser.add_argument('--outlier-iterations', type=int, default=3,
                            help='Per-ring outlier removal iterations')
        parser.add_argument('--first-ring', '-FirstRingNr', type=int, default=1,
                            help='First ring number to use')
        parser.add_argument('--eta-bin-size', '-EtaBinSize', type=float, default=5.0,
                            help='Azimuthal bin size (degrees)')

        # Geometry guesses
        parser.add_argument('--lsd-guess', '-LsdGuess', type=float, default=1000000,
                            help='Initial guess for detector distance (µm)')
        parser.add_argument('--bc-guess', '-BCGuess', type=float, default=[0.0, 0.0], nargs=2,
                            help='Initial guess for beam center [Y Z] (pixels)')
        parser.add_argument('--px', type=float, default=0,
                            help='Pixel size in µm (e.g. 200, 172). If set, no param file needed for non-Zarr inputs.')
        parser.add_argument('--tx', type=float, default=0.0,
                            help='Detector tilt tx (radians, not fitted but passed to CalibrantPanelShiftsOMP)')
        parser.add_argument('--mask', '-MaskFile', type=str, default='',
                            help='Mask TIFF file for bad/gap pixels (passed as MaskFile to CalibrantPanelShiftsOMP)')

        # Panel optimization (auto-detected from mask)
        parser.add_argument('--tol-shifts', type=float, default=3.0,
                            help='Panel shift tolerance in pixels (default: 3.0)')
        parser.add_argument('--tol-rotation', type=float, default=1.0,
                            help='Panel rotation tolerance in degrees (default: 1.0)')
        parser.add_argument('--per-panel-lsd', type=int, default=1,
                            help='Enable per-panel Lsd optimization (0=off, 1=on)')
        parser.add_argument('--fix-panel', type=int, default=-1,
                            help='Panel to fix during optimization (-1=auto: closest to BC)')

        # Image handling
        parser.add_argument('--im-trans', '-ImTransOpt', type=int, default=[0], nargs='*',
                            help='Image transformations: 0=none, 1=flipLR, 2=flipUD, 3=transpose')
        parser.add_argument('--bad-px', '-BadPxIntensity', type=float, default=np.nan,
                            help='Bad pixel intensity value')
        parser.add_argument('--gap-px', '-GapIntensity', type=float, default=np.nan,
                            help='Gap pixel intensity value')
        parser.add_argument('--threshold', '-Threshold', type=float, default=0,
                            help='Manual threshold for ring detection (0=auto)')

        # Output
        parser.add_argument('--plots', '-MakePlots', '-P', type=int, default=0,
                            help='Make plots: 0=no, 1=yes')
        parser.add_argument('--save-hdf', '-SavePlotsHDF', type=str, default='',
                            help='Save all data arrays to this HDF5 file')
        parser.add_argument('--no-median', '-NoMedian', type=int, default=0,
                            help='Skip median filter: 0=use median, 1=skip')

        # CPUs
        parser.add_argument('--cpus', type=int, default=0,
                            help='Number of CPUs for CalibrantPanelShiftsOMP (0=all)')

        # Data type override
        parser.add_argument('--data-type', type=int, default=-1,
                            help='Override DataType for CalibrantPanelShiftsOMP: '
                                 '1=uint16-raw, 6=tiff-uint32, 7=tiff-uint8, '
                                 '8=HDF5, 9=tiff-uint16. Default: auto-detect.')

        # Output
        parser.add_argument('--output', '-o', type=str, default='',
                            help='Output parameter filename (default: refined_MIDAS_params_<stem>.txt)')

        args, unparsed = parser.parse_known_args()

        # ---- Initialize state ----
        state = CalibState()
        state.bad_px_intensity = args.bad_px
        state.gap_intensity = args.gap_px
        state.im_trans_opt = args.im_trans
        state.data_loc = args.data_loc
        state.dark_loc = args.dark_loc
        state.tol_shifts = args.tol_shifts
        state.tol_rotation = args.tol_rotation
        state.per_panel_lsd = args.per_panel_lsd
        state.fix_panel_id = args.fix_panel

        dataFN = args.data
        darkFN = args.dark
        DrawPlots = bool(args.plots)
        firstRing = int(args.first_ring)
        multFactor = float(args.mult_factor)
        threshold = args.threshold
        noMedian = args.no_median
        n_cpus = args.cpus if args.cpus > 0 else os.cpu_count() or 8

        # HDF5 output
        if args.save_hdf:
            logger.info(f"Will save data arrays to HDF5 file: {args.save_hdf}")
            state.h5_file = h5py.File(args.save_hdf, 'w')
            meta_group = state.h5_file.create_group('metadata')
            meta_group.attrs['file_name'] = os.path.basename(dataFN)
            meta_group.attrs['date_created'] = pd.Timestamp.now().isoformat()

        # Constants
        mrr = 2000000  # maximum radius to simulate rings
        initialLsd = args.lsd_guess
        minArea = 300
        maxW = 1000

        # Parse energy/distance hints from filename
        filename_hints = parse_filename_hints(args.data)
        if 'lsd' in filename_hints and args.lsd_guess == 1000000:
            initialLsd = filename_hints['lsd']
            logger.info(f"Using Lsd from filename: {initialLsd:.0f} µm")

        logger.info(f"Starting automated calibration for: {dataFN}")

        # ---- Auto-detect file format and DataType ----
        convertFile = args.convert
        if convertFile < 0:
            convertFile = detect_format(dataFN)
            logger.info(f"Auto-detected format: {['Zarr','HDF5','GE','TIFF'][convertFile]}")

        if args.data_type >= 0:
            state.midas_dtype = args.data_type
            logger.info(f"DataType override from --data-type: {state.midas_dtype}")
        else:
            state.midas_dtype = detect_data_type(dataFN)
            logger.info(f"Auto-detected DataType: {state.midas_dtype}")

        # Detect calibrant from filename
        calibrant = detect_calibrant(args.data)
        if calibrant is None:
            calibrant = CALIBRANTS['ceo2']
            logger.info(f"No calibrant detected from filename, defaulting to {calibrant['name']}")

        # ---- Read image data for geometry estimation ----
        # For --convert 0 (force Zarr), use old pipeline
        if convertFile == 0 and not dataFN.endswith('.zip'):
            # User forced Zarr conversion from non-Zarr input
            psFN = args.params
            if not psFN:
                psFN = _make_temp_param_file(args, calibrant, filename_hints)
            logger.info("Generating Zarr file (--convert 0)")
            dataFN = generateZip('.', psFN, dfn=dataFN, nchunks=100,
                                 preproc=0, darkfn=darkFN,
                                 dloc=args.data_loc, darkloc=args.dark_loc)

        logger.info(f"Reading image for geometry estimation: {dataFN}")
        raw, dark, ny, nz = read_image_for_estimation(
            dataFN, darkFN, args.data_loc, args.dark_loc,
            skip_frame=state.skip_frame, data_type=state.midas_dtype)
        state.nr_pixels_y = ny
        state.nr_pixels_z = nz

        # Extract params from Zarr metadata if Zarr input
        if convertFile == 0 or dataFN.endswith('.zip'):
            import zarr
            dataF = zarr.open(dataFN, mode='r')
            ap = '/analysis/process/analysis_parameters'
            if f'{ap}/SpaceGroup' in dataF:
                state.space_group = dataF[f'{ap}/SpaceGroup'][0].item()
            else:
                state.space_group = calibrant['space_group']
            if f'{ap}/SkipFrame' in dataF:
                state.skip_frame = dataF[f'{ap}/SkipFrame'][0].item()
            if f'{ap}/PixelSize' in dataF:
                state.px = dataF[f'{ap}/PixelSize'][0].item()
            elif args.px > 0:
                state.px = args.px
            if f'{ap}/LatticeParameter' in dataF:
                state.latc = dataF[f'{ap}/LatticeParameter'][:]
            else:
                state.latc = calibrant['lattice']
            if f'{ap}/Wavelength' in dataF:
                state.wavelength = dataF[f'{ap}/Wavelength'][:].item()
            elif 'wavelength' in filename_hints:
                state.wavelength = filename_hints['wavelength']
            if f'{ap}/tx' in dataF:
                state.tx = dataF[f'{ap}/tx'][:].item()
            if f'{ap}/MaskFile' in dataF:
                mf = dataF[f'{ap}/MaskFile'][0]
                if isinstance(mf, bytes): mf = mf.decode()
                state.mask_file = str(mf)
        else:
            # Non-Zarr: use calibrant defaults and CLI args
            state.space_group = calibrant['space_group']
            state.latc = calibrant['lattice']
            if args.px > 0:
                state.px = args.px
            elif state.px == 0:
                state.px = 200.0
            if 'wavelength' in filename_hints:
                state.wavelength = filename_hints['wavelength']

            # Read additional params from param file
            if args.params:
                try:
                    with open(args.params) as pf:
                        for line in pf:
                            parts = line.split()
                            if len(parts) < 2:
                                continue
                            key = parts[0]
                            if key == 'Wavelength':
                                state.wavelength = float(parts[1])
                            elif key == 'SpaceGroup':
                                state.space_group = int(parts[1])
                            elif key in ('LatticeConstant', 'LatticeParameter'):
                                state.latc = np.array([float(x) for x in parts[1:7]])
                            elif key == 'px':
                                state.px = float(parts[1])
                            elif key == 'tx':
                                state.tx = float(parts[1])
                            elif key in ('NrPixels', 'NrPixelsY'):
                                state.nr_pixels_y = int(parts[1])
                            elif key == 'NrPixelsZ':
                                state.nr_pixels_z = int(parts[1])
                except Exception as e:
                    logger.warning(f"Could not fully parse param file: {e}")

        # CLI overrides
        if args.tx != 0.0:
            state.tx = args.tx
        # Save pre-mask copy for viewer display
        raw_for_display = raw.copy()

        if args.mask:
            state.mask_file = str(Path(args.mask).absolute())
            logger.info(f"MaskFile from --mask: {state.mask_file}")
        elif not state.mask_file:
            # Auto-detect: if image has -1 or -2 intensity pixels, generate mask
            mask_intensities = []
            n_neg1 = int(np.sum(raw == -1))
            n_neg2 = int(np.sum(raw == -2))
            if n_neg1 > 0:
                mask_intensities.append(-1)
            if n_neg2 > 0:
                mask_intensities.append(-2)
            if mask_intensities:
                logger.info(f"Auto-detected mask pixels: "
                            f"{n_neg1} at -1, {n_neg2} at -2 — generating mask")
                from generate_mask import generate_mask as _gen_mask
                # Avoid overwriting: use unique name in current directory
                auto_mask_fn = os.path.join(
                    os.path.dirname(os.path.abspath(dataFN)),
                    f"{Path(dataFN).stem}_autocal_mask.tif")
                if os.path.exists(auto_mask_fn):
                    logger.info(f"Auto-mask already exists: {auto_mask_fn}")
                else:
                    auto_mask_fn = _gen_mask(
                        os.path.abspath(dataFN), mask_intensities,
                        output_path=auto_mask_fn)
                    logger.info(f"Generated auto-mask: {auto_mask_fn}")
                state.mask_file = str(Path(auto_mask_fn).absolute())

        # Apply mask file: zero out masked pixels before any processing
        bad_gap_arr = []
        if state.mask_file:
            try:
                mask_img = np.array(Image.open(state.mask_file))
                mask_bool = (mask_img != 0)  # True where pixel is BAD (0=good, nonzero=bad)
                if mask_bool.shape != raw.shape:
                    logger.warning(f"Mask shape {mask_bool.shape} != image shape {raw.shape}, skipping mask")
                else:
                    raw[mask_bool] = 0
                    dark[mask_bool] = 0
                    bad_gap_arr = mask_bool
                    n_good = np.sum(~mask_bool)
                    logger.info(f"Applied mask from {state.mask_file}: "
                                f"{np.sum(mask_bool)} pixels masked, "
                                f"{n_good} active ({100*n_good/mask_bool.size:.1f}%)")
            except Exception as e:
                logger.error(f"Failed to read mask file {state.mask_file}: {e}")

            # Auto-detect panel layout from mask
            bc_guess = None
            if state.ybc > 0 and state.zbc > 0:
                bc_guess = (state.ybc, state.zbc)
            state.panel_grid = detect_panels_from_mask(state.mask_file, bc_guess=bc_guess)

        # Parse raw filename for CalibrantPanelShiftsOMP file naming
        rawFN = os.path.abspath(dataFN)

        # Apply bad_gap_arr mask if created from mask file
        if len(bad_gap_arr) != 0:
            raw = np.ma.masked_array(raw, mask=bad_gap_arr)

        # Set dark_name for CalibrantPanelShiftsOMP
        if darkFN:
            state.dark_name = os.path.abspath(darkFN)
        else:
            state.dark_name = os.path.abspath(dataFN)  # dark from same file
        if not state.folder:
            state.folder = os.path.dirname(rawFN) or os.getcwd()

        # Apply image transformations (for Python-side geometry estimation)
        for transOpt in state.im_trans_opt:
            if transOpt == 1:
                raw = np.fliplr(raw)
                dark = np.fliplr(dark)
                raw_for_display = np.fliplr(raw_for_display)
            elif transOpt == 2:
                raw = np.flipud(raw)
                dark = np.flipud(dark)
                raw_for_display = np.flipud(raw_for_display)
            elif transOpt == 3:
                raw = np.transpose(raw)
                dark = np.transpose(dark)
                raw_for_display = np.transpose(raw_for_display)

        if state.h5_file:
            save_raw_image_data(raw, state.h5_file)

        # Create viewer (initially shows raw pre-mask image)
        viewer = None
        if DrawPlots:
            try:
                os.environ.setdefault('QT_MAC_WANTS_LAYER', '1')  # macOS Cocoa compat
                viewer = CalibImageViewer(title=f'AutoCalibrateZarr — {os.path.basename(rawFN)}')
                viewer.set_raw(np.log(raw_for_display.astype(np.float64) + 1))
                viewer.process_events()
            except Exception as e:
                logger.warning(f"Could not create viewer (no display?): {e}")
                logger.warning("Continuing without plots")
                viewer = None

        # ---- Ring simulation (B4 fix: compute rhod first) ----
        logger.info("Running initial ring simulation")

        # Validate required params for GetHKLList
        if state.wavelength <= 0:
            logger.error("Wavelength is required but not set. Provide it via --params or filename (e.g. 71p676keV).")
            sys.exit(1)
        if np.all(state.latc == 0):
            logger.error("LatticeParameter is required but not set. Provide it via --params.")
            sys.exit(1)

        # B4 fix: compute rhod before first GetHKLList call
        NrPixelsY = state.nr_pixels_y if state.nr_pixels_y > 0 else 2048
        NrPixelsZ = state.nr_pixels_z if state.nr_pixels_z > 0 else 2048
        state.rhod = max(NrPixelsY, NrPixelsZ) * state.px

        # Use GetHKLList CLI mode (no temp param files needed)
        hkls = run_get_hkl_list_cli(
            state.space_group, state.latc, state.wavelength,
            initialLsd, state.rhod)
        sim_rads = np.unique(hkls[:, -1]) / state.px
        sim_rad_ratios = sim_rads / sim_rads[0]

        # ---- Background subtraction and thresholding ----
        # np.asarray strips masked-array wrapper (diplib segfaults on masked arrays)
        data = np.asarray(raw).astype(np.float64)
        # DataType already detected by detect_data_type() earlier

        if noMedian == 0:
            logger.info("Applying median filter for background estimation")
            # Ensure fully contiguous C-ordered copy
            data = np.ascontiguousarray(data.copy())
            logger.info(f"  data shape={data.shape}, dtype={data.dtype}, "
                        f"min={np.min(data):.1f}, max={np.max(data):.1f}")
            data2 = _safe_median_filter(data, kernel_size=101, n_iters=5)
        else:
            logger.info("Skipping median filter, using dark subtraction only")
            data2 = dark.astype(np.float64)

        logger.info('Finished with median, now processing data.')
        data = data.astype(float)

        if state.h5_file:
            save_background_data(data2, state.h5_file)

        if DrawPlots and viewer and noMedian == 0:
            viewer.set_bg(np.log(np.asarray(data2) + 1))
            viewer.process_events()

        # Background subtraction and thresholding
        data_corr = data - data2
        if noMedian == 1:
            threshold = 0
        elif threshold == 0:
            threshold = 100 * (1 + np.std(data_corr) // 100)
        data_corr[data_corr < threshold] = 0
        thresh = data_corr.copy()
        thresh[thresh > 0] = 255

        if state.h5_file:
            save_threshold_data(thresh, state.h5_file)

        if DrawPlots and viewer and noMedian == 0:
            viewer.set_corr(thresh)
            viewer.process_events()

        # ---- Beam center and ring detection ----
        bcg = args.bc_guess
        if bcg[0] == 0:
            logger.info("Auto-detecting beam center")
            labels, nlabels = measure.label(thresh, return_num=True)
            props = measure.regionprops(labels)

            for label in range(1, nlabels):
                if np.sum(labels == label) < minArea:
                    thresh[labels == label] = 0

            bc_computed = detect_beam_center_optimized(thresh, minArea, 6)
            rads = detect_ring_radii(labels, props, bc_computed, minArea)

            if args.lsd_guess == 1000000:
                initialLsd = estimate_lsd(rads, sim_rads, sim_rad_ratios,
                                          firstRing, initialLsd)
        else:
            bc_computed = np.flip(np.array(bcg))

        bc_new = bc_computed
        logger.info(f"FN: {rawFN}, Beam Center guess: {np.flip(bc_new)}, Lsd guess: {initialLsd}")

        # Re-run ring simulation with updated Lsd (CLI mode)
        hkls = run_get_hkl_list_cli(
            state.space_group, state.latc, state.wavelength,
            initialLsd, state.rhod)
        sim_rads = np.unique(hkls[:, -1]) / state.px
        sim_rad_ratios = sim_rads / sim_rads[0]

        if state.h5_file:
            save_ring_data(thresh, bc_new, sim_rads, state.h5_file)

        # Read RingsToExclude and panel params from param file
        state.rings_to_exclude = []
        state.panel_params = []
        if args.params:
            try:
                with open(args.params, 'r') as pf:
                    for line in pf:
                        if line.startswith('RingsToExclude'):
                            parts = line.split()
                            if len(parts) > 1:
                                state.rings_to_exclude.append(int(parts[1]))
                        elif any(line.startswith(pk) for pk in [
                            'NPanelsY', 'NPanelsZ', 'PanelSizeY', 'PanelSizeZ',
                            'PanelGapsY', 'PanelGapsZ', 'FixPanelID',
                            'PerPanelLsd', 'PerPanelDistortion',
                            'DistortionOrder', 'tolP4', 'tolLsdPanel', 'tolP2Panel',
                            'OutlierIterations', 'WeightByRadius',
                            'WeightByFitSNR', 'L2Objective',
                            'NormalizeRingWeights', 'nIterations',
                            'DoubletSeparation', 'tolRotation',
                            'MinIndicesForFit',
                        ]):
                            state.panel_params.append(line.strip())
                logger.info(f"Loaded manual exclusions: {state.rings_to_exclude}")
            except Exception as e:
                logger.warning(f"Could not read params from {args.params}: {e}")

        # Display rings overlay
        NrPixelsY = state.nr_pixels_y if state.nr_pixels_y > 0 else 2048
        NrPixelsZ = state.nr_pixels_z if state.nr_pixels_z > 0 else 2048
        if DrawPlots and viewer:
            if noMedian == 0:
                viewer.set_rings(thresh, sim_rads, bc_new, state.rings_to_exclude)
            else:
                log_img = np.log(data_corr + 1)
                viewer.set_rings(log_img, sim_rads, bc_new, state.rings_to_exclude)
            logger.info("Showing image viewer — click 'Continue' to proceed")
            viewer.show_and_wait()

        # ---- Prepare for MIDAS calibration ----
        # Parse filename components for CalibrantPanelShiftsOMP
        base = os.path.basename(rawFN)
        try:
            state.fnumber = int(base.split('_')[-1].split('.')[0])
            state.pad = len(base.split('_')[-1].split('.')[0])
            state.fstem = '_'.join(base.split('_')[:-1])
            state.ext = '.' + '.'.join(base.split('_')[-1].split('.')[1:])
        except (ValueError, IndexError):
            # Filename doesn't follow STEM_NNNN.ext pattern
            stem_ext = os.path.splitext(base)
            state.fstem = stem_ext[0]
            state.ext = stem_ext[1]
            state.fnumber = 1
            state.pad = 6

        state.lsd = initialLsd
        state.ybc = bc_new[1]
        state.zbc = bc_new[0]
        state.panel_shifts_file = f"{state.fstem}_panel_shifts.txt"
        state.nr_pixels_y = NrPixelsY
        state.nr_pixels_z = NrPixelsZ

        # Refine RhoD — maximum radius to edge from beam center
        edges = np.array([[0, 0], [NrPixelsY, 0], [NrPixelsY, NrPixelsZ], [0, NrPixelsZ]])
        state.rhod = np.max(np.linalg.norm(
            np.transpose(edges) - bc_new[:, None], axis=0)) * state.px

        # ---- Multi-stage CalibrantPanelShiftsOMP ----
        # Stage 1: geometry only (Lsd, BC, tilts) — lock distortion, no panels
        # Stage 2: full (distortion + panels) using refined geometry from stage 1
        stage1_iters = min(10, args.n_iterations)

        logger.info(f"Running MIDAS calibration: 2-stage, {n_cpus} CPUs, "
                     f"DoubletSep={args.doublet_separation}px")

        print(f"\n{'='*60}")
        print(f"  Stage 1: Geometry (Lsd, BC, tilts) — {stage1_iters} iterations")
        print(f"  DoubletSeparation={args.doublet_separation}px, "
              f"MultFactor={multFactor}")
        print(f"{'='*60}")

        runMIDAS(rawFN, state,
                 n_iterations=stage1_iters,
                 mult_factor=multFactor,
                 doublet_separation=args.doublet_separation,
                 outlier_iterations=args.outlier_iterations,
                 eta_bin_size=args.eta_bin_size,
                 max_width=maxW,
                 n_cpus=n_cpus,
                 stage=1,
                 stage_label='[Stage 1/2] ')

        logger.info(f"Stage 1 result: Lsd={state.lsd:.1f}, "
                     f"BC=({state.ybc:.2f}, {state.zbc:.2f}), "
                     f"ty={state.ty:.6f}, tz={state.tz:.6f}")

        print(f"\n{'='*60}")
        print(f"  Stage 2: Full (distortion + panels) — "
              f"{args.n_iterations} iterations")
        print(f"{'='*60}")

        runMIDAS(rawFN, state,
                 n_iterations=args.n_iterations,
                 mult_factor=multFactor,
                 doublet_separation=args.doublet_separation,
                 outlier_iterations=args.outlier_iterations,
                 eta_bin_size=args.eta_bin_size,
                 max_width=maxW,
                 n_cpus=n_cpus,
                 stage=2,
                 stage_label='[Stage 2/2] ')

        # ---- Generate final results ----
        logger.info("Generating final results data")
        corr_file = f"{rawFN}.corr.csv"
        if os.path.exists(corr_file):
            df = pd.read_csv(corr_file, delimiter=' ')

            if DrawPlots:
                # Launch interactive calibrant viewer (plot_calibrant_results.py)
                viewer_script = os.path.join(
                    INSTALL_PATH, 'gui/viewers/plot_calibrant_results.py')
                if os.path.exists(viewer_script):
                    subprocess.Popen(
                        [sys.executable, viewer_script, corr_file],
                        env=env)
                    logger.info(f"Launched plot_calibrant_results.py on {corr_file}")
                else:
                    logger.warning(f"Viewer not found: {viewer_script}")

            if state.h5_file:
                save_results_dataframe(df, state.h5_file)

        # Save parameters to HDF5
        if state.h5_file:
            params_group = state.h5_file.create_group('parameters')
            for key, value in {
                'lsd': state.lsd, 'beam_center': state.bc,
                'ty': state.ty, 'tz': state.tz,
                'p0': state.p0, 'p1': state.p1, 'p2': state.p2, 'p3': state.p3,
                'mean_strain': state.mean_strain, 'std_strain': state.std_strain,
                'wavelength': state.wavelength, 'pixel_size': state.px,
                'space_group': state.space_group,
                'rings_excluded': ','.join(map(str, state.rings_to_exclude))
            }.items():
                params_group.attrs[key] = value
            state.h5_file.close()
            logger.info(f"All data arrays saved to HDF5 file: {args.save_hdf}")

        # ---- Print final results ----
        print(f"\n{'='*60}")
        print("  Converged — Best Parameters:")
        print(f"{'='*60}")
        print(f"  Lsd            {state.lsd}")
        print(f"  BC             {state.bc}")
        print(f"  tx             {state.tx}")
        print(f"  ty             {state.ty}")
        print(f"  tz             {state.tz}")
        print(f"  p0             {state.p0}")
        print(f"  p1             {state.p1}")
        print(f"  p2             {state.p2}")
        print(f"  p3             {state.p3}")
        print(f"  RhoD           {state.rhod}")
        print(f"  Mean Strain    {state.mean_strain}")
        print(f"  Std Strain     {state.std_strain}")
        print(f"{'='*60}")

        logger.info(f'Lsd {state.lsd}')
        logger.info(f'BC {state.bc}')
        logger.info(f'Mean Strain: {state.mean_strain}')

        # ---- Write final parameter file (B5 fix: complete output) ----
        if args.output:
            psName = args.output
        else:
            psName = f'refined_MIDAS_params_{state.fstem}.txt'
        logger.info(f"Writing final parameters to {psName}")

        final_params = {
            # Geometry
            'Lsd': state.lsd, 'BC': state.bc,
            'tx': state.tx, 'ty': state.ty, 'tz': state.tz,
            'p0': state.p0, 'p1': state.p1, 'p2': state.p2, 'p3': state.p3,
            'RhoD': state.rhod, 'Wavelength': state.wavelength, 'px': state.px,
            # B5 fix: added missing fields
            'SpaceGroup': state.space_group,
            'LatticeConstant': ' '.join(f'{x:.6f}' for x in state.latc),
            'NrPixelsY': NrPixelsY, 'NrPixelsZ': NrPixelsZ,
            'DataType': state.midas_dtype,
            'Folder': state.folder,
            'FileStem': state.fstem,
            'Ext': state.ext,
            'Dark': state.dark_name,
            'Padding': state.pad,
            'StartNr': state.fnumber,
            'EndNr': state.fnumber,
            'skipFrame': state.skip_frame,
            # D7: Integration params (user requested)
            'RMin': 10, 'RMax': 1000, 'RBinSize': 1,
            'EtaMin': -180, 'EtaMax': 180,
            'EtaBinSize': 5, 'DoSmoothing': 1, 'DoPeakFit': 1,
            'MultiplePeaks': 1,
        }

        if state.p4 != 0.0:
            final_params['p4'] = state.p4

        if state.mask_file:
            final_params['MaskFile'] = state.mask_file

        if state.data_loc:
            final_params['dataLoc'] = state.data_loc
        if state.dark_loc:
            final_params['darkLoc'] = state.dark_loc

        with open(psName, 'w') as pf:
            for key, value in final_params.items():
                pf.write(f"{key} {value}\n")

            if state.panel_grid is not None:
                pg = state.panel_grid
                pf.write(f'NPanelsY {pg.n_panels_y}\n')
                pf.write(f'NPanelsZ {pg.n_panels_z}\n')
                pf.write(f'PanelSizeY {pg.panel_size_y}\n')
                pf.write(f'PanelSizeZ {pg.panel_size_z}\n')
                pf.write(f'PanelGapsY {" ".join(str(g) for g in pg.gaps_y)}\n')
                pf.write(f'PanelGapsZ {" ".join(str(g) for g in pg.gaps_z)}\n')
                fix_id = state.fix_panel_id if state.fix_panel_id >= 0 else pg.fix_panel_id
                pf.write(f'FixPanelID {fix_id}\n')
                pf.write(f'tolShifts {state.tol_shifts}\n')
                pf.write(f'tolRotation {state.tol_rotation}\n')
                if state.panel_shifts_file:
                    pf.write(f'PanelShiftsFile {state.panel_shifts_file}\n')
            elif state.panel_params:
                for pp in state.panel_params:
                    pf.write(f'{pp}\n')
                if state.panel_shifts_file:
                    pf.write(f'PanelShiftsFile {state.panel_shifts_file}\n')
                pf.write(f'tolShifts {state.tol_shifts}\n')

            for transOpt in state.im_trans_opt:
                pf.write(f"ImTransOpt {transOpt}\n")

            if not math.isnan(state.bad_px_intensity):
                pf.write(f"BadPxIntensity {state.bad_px_intensity}\n")
            if not math.isnan(state.gap_intensity):
                pf.write(f"GapIntensity {state.gap_intensity}\n")

        logger.info("Calibration completed successfully")
        print(f"\n  Output written to: {psName}")

        # ---- Cleanup intermediate files ----
        cleanup_files = [
            f"{rawFN}ps.txt",           # CalibrantPanelShiftsOMP param file
            'calibrant_screen_out.csv',  # CalibrantPanelShiftsOMP stdout log
            'hkls.csv',                  # HKL list (regenerated each run)
            'hkls_screen_out.csv',       # GetHKLList stdout
            f"{rawFN}.corr.csv",         # per-point correction data
            f"{rawFN}.lineout.xy",       # lineout data
        ]
        for f in cleanup_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    logger.debug(f"Cleaned up: {f}")
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Error in main function: {traceback.format_exc()}")
        if hasattr(state, 'h5_file') and state.h5_file:
            state.h5_file.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
