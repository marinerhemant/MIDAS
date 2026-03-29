#!/usr/bin/env python
"""
AutoCalibrateZarr — Automated detector geometry calibration using WAXS calibrant rings.

Accepts Zarr .zip, HDF5, GE binary, or TIFF images.  File format is auto-detected
from the extension.  Calls CalibrantPanelShiftsOMP or CalibrantIntegratorOMP (C/OpenMP) for the heavy lifting,
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


def apply_imtrans(img, im_trans_opts):
    """Apply MIDAS image transformations (same as C DoImageTransformations).

    Parameters
    ----------
    img : ndarray (nz, ny)
        Image in row=Z, col=Y layout.
    im_trans_opts : list of int
        Transformation codes: 0 = no-op, 1 = flip Y (cols),
        2 = flip Z (rows), 3 = transpose.

    Returns
    -------
    img : ndarray
        Transformed image.
    """
    for opt in im_trans_opts:
        if opt == 1:
            img = img[:, ::-1]   # flip Y (columns)
        elif opt == 2:
            img = img[::-1, :]   # flip Z (rows)
        elif opt == 3:
            img = img.T          # transpose
    return np.ascontiguousarray(img)


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
    p5: float = 0.0
    p6: float = 0.0
    p7: float = 0.0
    p8: float = 0.0
    p9: float = 0.0
    p10: float = 0.0
    mean_strain: float = 1.0
    std_strain: float = 0.0
    rhod: float = 0.0

    # Ring info
    n_planes: int = 0
    rings_to_exclude: list = field(default_factory=list)
    max_ring_number: int = 0  # 0 = no limit, >0 = exclude rings > this

    # Parallax
    fit_parallax: int = 0
    parallax_in: float = 0.0
    tol_parallax: float = 200.0

    # Peak fitting mode (0=pV default, 1=TCH GSAS-II)
    peak_fit_mode: int = 0

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

    # Extra params from user's param file (pass-through to CalibrantPanelShiftsOMP)
    extra_params: dict = field(default_factory=dict)

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

# ---- Blocking 2×2 image viewer (matplotlib) ----


class CalibImageViewer:
    """Matplotlib 2×2 image viewer for calibration.

    Panels: Raw | Background | Corrected | Corrected + Rings
    show_and_wait() blocks until the figure is closed.
    """

    def __init__(self, title='AutoCalibrateZarr'):
        self._title = title
        self._cmap = 'viridis'
        self._raw_img = None
        self._bg_img = None
        self._corr_img = None
        self._rings_img = None
        self._ring_radii = None
        self._ring_bc = None
        self._rings_to_exclude = None

    def set_raw(self, img):
        self._raw_img = img

    def set_bg(self, img):
        self._bg_img = img

    def set_corr(self, img):
        self._corr_img = img

    def set_rings(self, img, ring_radii, bc, rings_to_exclude=None):
        self._rings_img = img
        self._ring_radii = ring_radii
        self._ring_bc = bc
        self._rings_to_exclude = rings_to_exclude

    def show_and_wait(self):
        """Build the 2x2 figure and block until it is closed."""
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig.canvas.manager.set_window_title(self._title)

        # Compute clim from raw image
        vmin, vmax = None, None
        if self._raw_img is not None:
            pos = self._raw_img[self._raw_img > 0] if np.any(self._raw_img > 0) else self._raw_img.ravel()
            m, s = np.median(pos), np.std(pos)
            vmin, vmax = m, m + 2 * s

        # Raw
        ax = axes[0, 0]
        ax.set_title('Raw')
        if self._raw_img is not None:
            ax.imshow(self._raw_img, cmap=self._cmap, vmin=vmin, vmax=vmax,
                      origin='lower', aspect='equal')

        # Background
        ax = axes[0, 1]
        ax.set_title('Background')
        if self._bg_img is not None:
            ax.imshow(self._bg_img, cmap=self._cmap, vmin=vmin, vmax=vmax,
                      origin='lower', aspect='equal')

        # Corrected
        ax = axes[1, 0]
        ax.set_title('Corrected')
        if self._corr_img is not None:
            ax.imshow(self._corr_img, cmap=self._cmap, origin='lower', aspect='equal')

        # Corrected + Rings
        ax = axes[1, 1]
        ax.set_title('Corrected + Rings')
        if self._rings_img is not None:
            ax.imshow(self._rings_img, cmap=self._cmap, origin='lower', aspect='equal')
        if self._ring_radii is not None and self._ring_bc is not None:
            for i, rad in enumerate(self._ring_radii):
                ring_nr = i + 1
                if self._rings_to_exclude and ring_nr in self._rings_to_exclude:
                    continue
                circle = Circle((self._ring_bc[1], self._ring_bc[0]), rad,
                                fill=False, edgecolor='cyan', linewidth=0.8)
                ax.add_patch(circle)

        fig.tight_layout()
        # Leave space at bottom for Continue/Cancel buttons
        fig.subplots_adjust(bottom=0.07)
        from matplotlib.widgets import Button
        ax_cont = fig.add_axes([0.36, 0.01, 0.14, 0.04])
        btn_cont = Button(ax_cont, '▶ Continue', color='#2ecc71', hovercolor='#27ae60')
        btn_cont.label.set_color('white')
        btn_cont.label.set_fontweight('bold')
        btn_cont.on_clicked(lambda event: plt.close(fig))

        ax_cancel = fig.add_axes([0.52, 0.01, 0.12, 0.04])
        btn_cancel = Button(ax_cancel, '✕ Cancel', color='#e74c3c', hovercolor='#c0392b')
        btn_cancel.label.set_color('white')
        btn_cancel.label.set_fontweight('bold')
        def _cancel(event):
            plt.close(fig)
            logger.info("Calibration cancelled by user.")
            sys.exit(0)
        btn_cancel.on_clicked(_cancel)
        plt.show(block=True)

    def process_events(self):
        """No-op for matplotlib backend."""
        pass


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
    elif ext == '.cbf':
        return 4
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
    if ext == '.cbf':
        return 10  # native CBF (ReadCBFFrame in FileReader.c)
    return 1  # default


def _detect_mask_sentinels(arr):
    """Detect sentinel intensity values (-1, -2) used for gap/bad pixels.

    Must be called BEFORE clamping (arr[arr < 1] = 1).
    Returns a list of sentinel values found (e.g. [-1], [-2], [-1, -2], or []).
    """
    sentinels = []
    if int(np.sum(arr == -1)) > 0:
        sentinels.append(-1)
    if int(np.sum(arr == -2)) > 0:
        sentinels.append(-2)
    return sentinels


def read_image_for_estimation(data_fn, dark_fn, data_loc, dark_loc,
                              skip_frame=0, data_type=None):
    """Read image data into numpy for beam center / ring detection.

    Supports HDF5, TIFF, GE binary, and Zarr.
    Returns: raw, dark, ny, nz, mask_sentinels
        mask_sentinels: list of sentinel intensity values found before clamping
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

        mask_sentinels = _detect_mask_sentinels(raw)
        raw[raw < 1] = 1
        return raw, dark, ny, nz, mask_sentinels

    elif ext in ('.tif', '.tiff'):
        img = np.array(Image.open(data_fn)).astype(np.float64)
        nz, ny = img.shape
        mask_sentinels = _detect_mask_sentinels(img)
        img[img < 1] = 1
        dark = np.zeros_like(img)
        if dark_fn and os.path.exists(dark_fn):
            dark = np.array(Image.open(dark_fn)).astype(np.float64)
        return img, dark, ny, nz, mask_sentinels

    elif ext == '.cbf':
        from read_cbf import read_cbf as _read_cbf
        _, raw = _read_cbf(data_fn, check_md5=False)
        raw = raw.astype(np.float64)
        nz, ny = raw.shape
        mask_sentinels = _detect_mask_sentinels(raw)
        raw[raw < 1] = 1
        dark = np.zeros_like(raw)
        if dark_fn and os.path.exists(dark_fn):
            dark_ext = Path(dark_fn).suffix.lower()
            if dark_ext == '.cbf':
                _, dark = _read_cbf(dark_fn, check_md5=False)
                dark = dark.astype(np.float64)
            else:
                dark = np.array(Image.open(dark_fn)).astype(np.float64)
        return raw, dark, ny, nz, mask_sentinels

    elif ext == '.zip' or ext == '.zarr':
        # Zarr path (backward compat)
        import zarr
        f = zarr.open(data_fn, mode='r')
        raw, ny, nz = fileReader(f, '/exchange/data', skip_frame)
        dark, _, _ = fileReader(f, '/exchange/dark', skip_frame)
        return raw, dark, ny, nz, []

    else:
        # GE binary or other raw format
        dt = data_type or 1
        dtype_map = {1: np.uint16, 2: np.float64, 3: np.float32,
                     4: np.uint32, 5: np.int32}
        np_dtype = dtype_map.get(dt, np.uint16)
        header_size = 8192  # standard GE header

        def _read_ge(fn, offset):
            """Read a GE file, return averaged 2D frame."""
            arr = np.fromfile(fn, dtype=np_dtype, offset=offset)
            total = len(arr)
            # Try to find a square side that divides evenly
            for candidate in [2048, 4096, 1024, 512]:
                frame_px = candidate * candidate
                if total >= frame_px and total % frame_px == 0:
                    nframes = total // frame_px
                    arr = arr.reshape(nframes, candidate, candidate)
                    return np.mean(arr, axis=0).astype(np.float64)
            # Fallback: single square frame
            side = int(np.sqrt(total))
            if side * side == total:
                return arr.reshape(side, side).astype(np.float64)
            raise ValueError(
                f"Cannot reshape {total} pixels into square frames "
                f"(tried 2048², 4096², 1024², 512², √{total}={side})")

        # Try with header first, then without
        try:
            raw = _read_ge(data_fn, header_size)
        except (ValueError, Exception):
            raw = _read_ge(data_fn, 0)

        nz, ny = raw.shape
        mask_sentinels = _detect_mask_sentinels(raw)
        raw[raw < 1] = 1
        dark = np.zeros_like(raw)
        if dark_fn and os.path.exists(dark_fn):
            try:
                dark = _read_ge(dark_fn, header_size)
            except (ValueError, Exception):
                dark = _read_ge(dark_fn, 0)
        return raw, dark, ny, nz, mask_sentinels


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
                if 'p5 ' in line:
                    state.p5 = float(parts[1])
                if 'p6 ' in line:
                    state.p6 = float(parts[1])
                if 'p7 ' in line:
                    state.p7 = float(parts[1])
                if 'p8 ' in line:
                    state.p8 = float(parts[1])
                if 'p9 ' in line:
                    state.p9 = float(parts[1])
                if 'p10 ' in line:
                    state.p10 = float(parts[1])
                if 'parallax ' in line:
                    state.parallax_in = float(parts[1])
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


def detect_beam_center_optimized(thresh, minArea, num_processes=None,
                                  labels=None, props=None):
    """Detect beam center from thresholded image.

    Uses 4× downsampling for fast labeling, then runs the chord-bisector
    on each surviving region.  Accepts pre-computed *labels*/*props* but
    will recompute on the downsampled image for speed.
    """
    try:
        # Warm up numba JIT on first call
        if not hasattr(detect_beam_center_optimized, "_initialized"):
            dummy_coords = np.array([[0, 0], [1, 1], [2, 2]])
            dummy_bbox = (0, 0, 2, 2)
            _ = _find_center_point(dummy_coords, dummy_bbox)
            detect_beam_center_optimized._initialized = True

        # Downsample for speed (4× reduction each axis → 16× fewer pixels)
        scale = 4
        h, w = thresh.shape
        sh, sw = h // scale, w // scale
        # Use block-max so thin arcs survive downsampling
        ds = thresh[:sh*scale, :sw*scale].reshape(sh, scale, sw, scale).max(axis=(1, 3))

        ds_labels, ds_nlabels = ndimage.label(ds)
        if ds_nlabels == 0:
            logger.warning("No beam centers detected!")
            return np.array([0.0, 0.0])

        from skimage import measure as _measure
        ds_props = _measure.regionprops(ds_labels)

        ds_minArea = max(1, minArea // (scale * scale))

        # Single-pass: iterate regionprops (coords & bbox already computed)
        all_centers = []
        for rp in ds_props:
            if rp.area < ds_minArea:
                continue
            coords = rp.coords  # (N, 2) array of (row, col)
            bbox = (rp.bbox[0], rp.bbox[1], rp.bbox[2], rp.bbox[3])
            center = _find_center_point(coords, bbox)
            if center[0] >= 0 and center[1] >= 0:
                all_centers.append(center)

        if not all_centers:
            logger.warning("No valid beam centers calculated!")
            return np.array([0.0, 0.0])

        centers_array = np.array(all_centers) * scale  # scale back to full res
        return np.array([np.median(centers_array[:, 0]),
                         np.median(centers_array[:, 1])])
    except Exception as e:
        logger.error(f"Error detecting beam center: {e}")
        return np.array([0.0, 0.0])


def detect_ring_radii(labels, props, bc_computed, minArea):
    """Detect ring radii from labeled image and beam center."""
    try:
        rads = []
        for rp in props:
            if rp.area < minArea:
                continue
            coords = rp.coords
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


def estimate_lsd(rads, sim_rads, sim_rad_ratios, firstRing, initialLsd, max_ring=0):
    """Estimate sample-to-detector distance from ring radii.
    
    Try assigning detected ring k to simulated ring j (for multiple k),
    compute a trial Lsd, then count how many other detected rings match.
    Pick the assignment with the most consistent matches.
    
    Tests det[0], det[1], det[2] as starting points to handle spurious
    detections at small radii (e.g. panel artifacts on Pilatus).
    """
    if len(rads) == 0:
        return initialLsd
    try:
        n_sim = len(sim_rads) if max_ring <= 0 else min(max_ring, len(sim_rads))
        logger.info(f"  estimate_lsd: {len(rads)} detected rings, "
                    f"{n_sim} simulated rings (firstRing={firstRing}, max_ring={max_ring})")
        logger.info(f"  estimate_lsd: initialLsd={initialLsd:.1f}, "
                    f"detected radii (px): {np.round(rads, 1)}")
        logger.info(f"  estimate_lsd: sim radii (px, first {min(n_sim,10)}): "
                    f"{np.round(sim_rads[:min(n_sim,10)], 1)}")

        best_lsd = initialLsd
        best_score = (0, 1e9)  # (match_count, -lsd_std) — max count, then min std
        best_hypothesis = (-1, -1)

        # Try multiple detected rings as the starting point
        max_det_start = min(3, len(rads))
        for det_start in range(max_det_start):
            for hyp in range(firstRing - 1, n_sim):
                # Trial Lsd: detected ring 'det_start' = sim ring 'hyp'
                trial_lsd = initialLsd * rads[det_start] / sim_rads[hyp]

                # At this trial Lsd, what would the sim radii be in pixels?
                scale = trial_lsd / initialLsd
                trial_sim_px = sim_rads * scale

                # Count how many detected rings match a simulated ring (within 5%)
                matches = 0
                lsds_this = []
                for det_rad in rads:
                    diffs = np.abs(trial_sim_px[:n_sim] - det_rad)
                    best_j = np.argmin(diffs)
                    rel_err = diffs[best_j] / det_rad
                    if rel_err < 0.05:
                        matches += 1
                        lsds_this.append(initialLsd * det_rad / sim_rads[best_j])

                lsd_std = np.std(lsds_this) if len(lsds_this) >= 2 else 1e9
                lsd_median = np.median(lsds_this) if lsds_this else trial_lsd

                # Score: primary = most matches, secondary = lowest std
                is_better = False
                if matches > best_score[0]:
                    is_better = True
                elif matches == best_score[0] and lsd_std < best_score[1]:
                    is_better = True

                if is_better:
                    best_score = (matches, lsd_std)
                    best_hypothesis = (det_start, hyp)
                    best_lsd = lsd_median

                logger.info(f"    hyp det[{det_start}]→sim[{hyp}], "
                            f"trial_lsd={trial_lsd:.0f}, "
                            f"matches={matches}/{len(rads)}, lsd_std={lsd_std:.1f}, "
                            f"lsd_median={lsd_median:.1f}"
                            f"{' ← BEST' if is_better else ''}")

        logger.info(f"  estimate_lsd: best hypothesis: det ring {best_hypothesis[0]} "
                    f"→ sim ring {best_hypothesis[1]}, "
                    f"{best_score[0]}/{len(rads)} matches, "
                    f"lsd_std={best_score[1]:.1f}, Lsd={best_lsd:.1f}")

        return best_lsd
    except Exception as e:
        logger.error(f"Error estimating Lsd: {e}")
        return initialLsd


# ---- Auto-detect max ring number ----

def auto_detect_max_ring(sim_rads_px, npy, npz, bc_y, bc_z,
                         data=None, min_separation_px=5.0,
                         max_overlap_run=3, snr_threshold=3.0):
    """Auto-detect the maximum usable ring number.

    Criteria (applied in order):
    1. Ring must fit on the detector (R < 95% of max BC-to-corner distance).
    2. Adjacent rings must be separated by >= min_separation_px.
       Stop after max_overlap_run consecutive too-close pairs.
    3. If data is provided, compute radial intensity profile and check
       SNR of each ring.  Drop rings with SNR < snr_threshold.

    Parameters
    ----------
    sim_rads_px : array
        Simulated ring radii in pixels, sorted ascending.
    npy, npz : int
        Detector dimensions.
    bc_y, bc_z : float
        Beam center in pixels.
    data : ndarray or None
        Background-subtracted image (for SNR estimation).
        Shape can be (nz, ny) or (ny, nz).
    min_separation_px : float
        Minimum required separation between adjacent rings (pixels).
    max_overlap_run : int
        Stop after this many consecutive too-close pairs.
    snr_threshold : float
        Minimum ring SNR to include (peak / background_std).

    Returns
    -------
    max_ring : int
        Number of usable rings (0 = use all).
    """
    if len(sim_rads_px) <= 1:
        return 0  # no limit

    # --- Criterion 1: Detector extent ---
    corners = np.array([
        [0, 0], [0, npz - 1], [npy - 1, 0], [npy - 1, npz - 1]
    ], dtype=float)
    max_R = max(np.sqrt((c[0] - bc_y)**2 + (c[1] - bc_z)**2)
                for c in corners)
    extent_limit = 0.95 * max_R

    # --- Criterion 2: Ring separation ---
    # Find where consecutive close pairs start
    close_run = 0
    separation_limit = len(sim_rads_px)
    for i in range(len(sim_rads_px)):
        if sim_rads_px[i] > extent_limit:
            separation_limit = i
            logger.info(f"  auto_detect_max_ring: ring {i+1} "
                        f"(R={sim_rads_px[i]:.1f}px) exceeds detector "
                        f"extent ({extent_limit:.1f}px)")
            break
        if i < len(sim_rads_px) - 1:
            sep = sim_rads_px[i + 1] - sim_rads_px[i]
            if sep < min_separation_px:
                close_run += 1
                if close_run >= max_overlap_run:
                    separation_limit = i + 2 - max_overlap_run
                    logger.info(
                        f"  auto_detect_max_ring: {max_overlap_run} consecutive "
                        f"close ring pairs at ring {separation_limit+1} "
                        f"(sep < {min_separation_px}px)")
                    break
            else:
                close_run = 0

    max_ring = separation_limit

    # --- Criterion 3: SNR from radial profile ---
    if data is not None and max_ring > 0:
        try:
            # Compute radial distance for each pixel
            nz_img, ny_img = data.shape
            yy, zz = np.meshgrid(np.arange(ny_img), np.arange(nz_img))
            # BC is (y_px, z_px); image is (z_row, y_col)
            R_img = np.sqrt((yy - bc_y)**2 + (zz - bc_z)**2)

            # Compute azimuthally-averaged radial profile (1px bins)
            max_r_int = int(np.ceil(sim_rads_px[min(max_ring, len(sim_rads_px)) - 1])) + 30
            max_r_int = min(max_r_int, int(np.max(R_img)))
            r_bins = np.arange(0, max_r_int + 1)
            r_idx = np.clip(R_img.astype(int), 0, max_r_int)
            radial_sum = np.bincount(r_idx.ravel(), weights=data.ravel(),
                                     minlength=max_r_int + 1)
            radial_count = np.bincount(r_idx.ravel(), minlength=max_r_int + 1)
            radial_count[radial_count == 0] = 1  # avoid div by zero
            radial_profile = radial_sum / radial_count

            # Estimate per-ring SNR
            snr_limit = max_ring
            low_snr_run = 0
            max_low_snr_run = 3  # consecutive weak rings before cutoff
            for i in range(min(max_ring, len(sim_rads_px))):
                rc = int(round(sim_rads_px[i]))
                if rc >= len(radial_profile):
                    snr_limit = i
                    break

                # Adaptive window: ±max(8, 3% of rc) to handle Lsd errors
                pk_hw = max(8, int(0.03 * rc))  # peak half-width
                bg_hw = 3 * pk_hw               # background half-width

                # Peak: max in window around ring
                lo_pk = max(0, rc - pk_hw)
                hi_pk = min(len(radial_profile), rc + pk_hw + 1)
                peak_val = np.max(radial_profile[lo_pk:hi_pk])

                # Background: wider annulus excluding ring core
                lo_bg = max(0, rc - bg_hw)
                hi_bg = min(len(radial_profile), rc + bg_hw + 1)
                bg_mask = np.ones(hi_bg - lo_bg, dtype=bool)
                # Exclude peak zone from background
                core_lo = max(0, rc - pk_hw - lo_bg)
                core_hi = min(hi_bg - lo_bg, rc + pk_hw + 1 - lo_bg)
                bg_mask[core_lo:core_hi] = False
                bg_vals = radial_profile[lo_bg:hi_bg][bg_mask]

                if len(bg_vals) > 2:
                    bg_mean = np.mean(bg_vals)
                    bg_std = max(np.std(bg_vals), 1e-10)
                    snr = (peak_val - bg_mean) / bg_std
                else:
                    snr = 0.0

                logger.debug(f"    ring {i+1} R={sim_rads_px[i]:.1f}px SNR={snr:.1f}")

                if snr < snr_threshold:
                    low_snr_run += 1
                    if low_snr_run >= max_low_snr_run:
                        snr_limit = i + 1 - max_low_snr_run
                        logger.info(
                            f"  auto_detect_max_ring: {max_low_snr_run} "
                            f"consecutive low-SNR rings starting at ring "
                            f"{snr_limit+1} (SNR < {snr_threshold})")
                        break
                else:
                    low_snr_run = 0

            max_ring = min(max_ring, snr_limit)
        except Exception as e:
            logger.warning(f"  auto_detect_max_ring: SNR estimation failed: {e}")

    if max_ring >= len(sim_rads_px) or max_ring <= 0:
        # If SNR gave 0 (all rings failed SNR from the start),
        # the SNR calculation is unreliable — fall back to extent limit
        if max_ring <= 0 and separation_limit > 0 and separation_limit < len(sim_rads_px):
            # SNR unreliable, use extent-based limit, cap at 25
            max_ring = min(separation_limit, 25)
            logger.info(f"  auto_detect_max_ring: SNR unreliable, "
                        f"using extent-based limit: {max_ring} of "
                        f"{len(sim_rads_px)} rings")
            return max_ring
        logger.info(f"  auto_detect_max_ring: all {len(sim_rads_px)} rings usable")
        return 0  # 0 means no limit
    else:
        logger.info(f"  auto_detect_max_ring: using {max_ring} of "
                    f"{len(sim_rads_px)} rings")
        return max_ring


# ---- Tilted-detector auto-guess (direct geometry fit) ----

def _pixel_to_R(y_px, z_px, bc_y, bc_z, lsd, ty_deg, tz_deg, px):
    """Compute radial distance R (in pixels) for given pixel coords and geometry.

    Uses the exact MIDAS convention from DetectorGeometry.c:
      Yc = (-Y + Ycen) * px    (Y is flipped)
      Zc = (Z - Zcen) * px
      ABC = [0, Yc, Zc]
      ABCPr = TRs @ ABC        where TRs = Rx @ (Ry @ Rz)
      XYZ = [Lsd + ABCPr[0], ABCPr[1], ABCPr[2]]
      R_um = (Lsd / XYZ[0]) * sqrt(XYZ[1]^2 + XYZ[2]^2)
      R_px = R_um / px

    Parameters
    ----------
    y_px, z_px : array-like
        Pixel coordinates (row, col in MIDAS convention).
    bc_y, bc_z : float
        Beam center in pixels.
    lsd : float
        Sample-to-detector distance in µm.
    ty_deg, tz_deg : float
        Detector tilts in degrees.
    px : float
        Pixel size in µm.

    Returns
    -------
    R_px : ndarray
        Radial distance in pixels for each input pixel.
    """
    tyr = np.deg2rad(ty_deg)
    tzr = np.deg2rad(tz_deg)
    # TRs = Rx @ (Ry @ Rz), with tx=0 → Rx = I
    cy, sy = np.cos(tyr), np.sin(tyr)
    cz, sz = np.cos(tzr), np.sin(tzr)
    # Ry @ Rz (tx=0, so Rx=I → TRs = Ry @ Rz)
    TRs = np.array([[ cy*cz, -cy*sz,  sy],
                     [    sz,     cz,   0],
                     [-sy*cz,  sy*sz,  cy]])

    Yc = (-np.asarray(y_px, dtype=np.float64) + bc_y) * px
    Zc = (np.asarray(z_px, dtype=np.float64) - bc_z) * px

    # ABCPr = TRs @ [0, Yc, Zc]  →  only columns 1,2 of TRs matter
    ABCPr_0 = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    ABCPr_1 = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    ABCPr_2 = TRs[2, 1] * Yc + TRs[2, 2] * Zc

    XYZ_0 = lsd + ABCPr_0
    XYZ_1 = ABCPr_1
    XYZ_2 = ABCPr_2

    R_um = (lsd / XYZ_0) * np.sqrt(XYZ_1**2 + XYZ_2**2)
    return R_um / px


def auto_guess_tilted(arc_coords, sim_rads_px, bc_init, lsd_init, px,
                      max_tilt=5.0):
    """Refine (BC_Y, BC_Z, Lsd, ty, tz) by direct geometry fit on arc pixels.

    Instead of fitting ellipses, this uses the MIDAS forward model to compute
    the expected radial distance R for each arc pixel given trial geometry
    parameters. The objective minimizes the sum of squared residuals between
    computed R and the nearest known ring radius.

    Parameters
    ----------
    arc_coords : ndarray, shape (N, 2)
        Detected arc pixel coordinates (y_px, z_px).
    sim_rads_px : ndarray
        Known ring radii in pixels (from GetHKLList).
        Should be limited to the usable rings (max_ring_number).
    bc_init : tuple (bc_y, bc_z)
        Initial beam center guess in pixels.
    lsd_init : float
        Initial Lsd guess in µm.
    px : float
        Pixel size in µm.
    max_tilt : float
        Maximum tilt angle bound in degrees.

    Returns
    -------
    result : dict with keys 'bc_y', 'bc_z', 'lsd', 'ty', 'tz', 'residual'
    """
    from scipy.optimize import minimize

    y_arc = arc_coords[:, 0]
    z_arc = arc_coords[:, 1]

    # Subsample for speed if too many pixels
    max_pts = 10000
    if len(y_arc) > max_pts:
        idx = np.random.default_rng(42).choice(len(y_arc), max_pts, replace=False)
        y_arc = y_arc[idx]
        z_arc = z_arc[idx]

    def objective(params):
        bc_y, bc_z, lsd, ty, tz = params
        R_computed = _pixel_to_R(y_arc, z_arc, bc_y, bc_z, lsd, ty, tz, px)
        # For each pixel, find distance to nearest known ring
        diffs = np.abs(R_computed[:, None] - sim_rads_px[None, :])
        min_diffs = np.min(diffs, axis=1)
        return np.sum(min_diffs**2)

    x0 = [bc_init[0], bc_init[1], lsd_init, 0.0, 0.0]
    # Bounds: BC ± 200 px, Lsd ± 15%, tilts ± max_tilt
    bounds = [
        (bc_init[0] - 200, bc_init[0] + 200),
        (bc_init[1] - 200, bc_init[1] + 200),
        (lsd_init * 0.85, lsd_init * 1.15),
        (-max_tilt, max_tilt),
        (-max_tilt, max_tilt),
    ]

    logger.info(f"  auto_guess_tilted: fitting {len(y_arc)} arc pixels "
                f"against {len(sim_rads_px)} rings")
    logger.info(f"  initial: BC=({bc_init[0]:.1f}, {bc_init[1]:.1f}), "
                f"Lsd={lsd_init:.0f}, ty=0, tz=0")

    # Evaluate objective at initial point (no tilts)
    initial_obj = objective(x0)
    initial_residual = initial_obj / len(y_arc)
    logger.info(f"  initial residual: {initial_residual:.4f} px²/pixel")

    # If initial residual is already very good, skip optimization
    # (tilts would be near-zero and optimizer may create spurious minima)
    if initial_residual < 5.0:
        logger.info(f"  Skipping tilt optimization — initial geometry already good "
                    f"(residual {initial_residual:.2f} < 5.0 px²/pixel)")
        return {
            'bc_y': bc_init[0], 'bc_z': bc_init[1],
            'lsd': lsd_init, 'ty': 0.0, 'tz': 0.0,
            'residual': initial_residual
        }

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-12})

    bc_y, bc_z, lsd, ty, tz = result.x
    residual = result.fun / len(y_arc)

    logger.info(f"  auto_guess_tilted result: BC=({bc_y:.1f}, {bc_z:.1f}), "
                f"Lsd={lsd:.0f}, ty={ty:.3f}°, tz={tz:.3f}°, "
                f"residual={residual:.4f} px²")

    # Reject if optimizer hit bounds (sign of runaway) or got worse
    at_bound = (abs(ty) >= max_tilt - 0.01 or abs(tz) >= max_tilt - 0.01 or
                lsd <= lsd_init * 0.86 or lsd >= lsd_init * 1.14)
    if at_bound or residual > initial_residual:
        logger.warning(f"  Rejecting tilt optimization — "
                       f"{'at bounds' if at_bound else 'residual increased'}, "
                       f"using initial estimates")
        return {
            'bc_y': bc_init[0], 'bc_z': bc_init[1],
            'lsd': lsd_init, 'ty': 0.0, 'tz': 0.0,
            'residual': initial_residual
        }

    return {
        'bc_y': bc_y,
        'bc_z': bc_z,
        'lsd': lsd,
        'ty': ty,
        'tz': tz,
        'residual': residual,
    }


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


def run_integrator_validation(refined_params_file, data_file, dark_file,
                              state, n_cpus=4):
    """Run DetectorMapper + IntegratorZarrOMP to independently validate calibration.

    Uses the refined geometry to integrate the calibrant image, fits peaks at
    the known ring positions, and writes an integrator_<fn>.corr.csv in the
    same 16-column format as CalibrantPanelShiftsOMP's corr.csv.

    Parameters
    ----------
    refined_params_file : str
        Path to the refined_MIDAS_params_<stem>.txt file.
    data_file : str
        Path to the calibrant data file (TIFF, HDF5, etc.).
    dark_file : str
        Path to the dark file (or '' if none).
    state : CalibState
        The calibration state with refined geometry.
    n_cpus : int
        Number of CPUs for the integrator.
    """
    import csv
    import tempfile

    mapper_bin = os.path.join(INSTALL_PATH, 'FF_HEDM/bin/DetectorMapper')
    integrator_bin = os.path.join(INSTALL_PATH, 'FF_HEDM/bin/IntegratorZarrOMP')
    hkl_bin = os.path.join(INSTALL_PATH, 'FF_HEDM/bin/GetHKLList')

    for b in [mapper_bin, integrator_bin, hkl_bin]:
        if not os.path.exists(b):
            logger.warning(f"Validation skipped: {b} not found")
            return

    work_dir = tempfile.mkdtemp(prefix='midas_intval_')
    logger.info(f"Integrator validation work dir: {work_dir}")

    try:
        # --- 1. Get ring radii from GetHKLList ---
        hkl_result = subprocess.run(
            [hkl_bin,
             '--sg', str(int(state.space_group)),
             '--lp', *[f'{x:.6f}' for x in state.latc],
             '--wl', f'{state.wavelength:.6f}',
             '--lsd', f'{state.lsd:.1f}',
             '--maxR', f'{state.rhod:.1f}',
             '--stdout'],
            capture_output=True, text=True, env=env)

        # Parse ring radii (columns: h k l D-spacing RingNr ... Radius)
        ring_radii = {}  # RingNr -> (Radius_um, DSpacing, Ideal2Theta)
        for line in hkl_result.stdout.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 11:
                try:
                    ring_nr = int(float(parts[4]))
                    d_spacing = float(parts[3])
                    ideal_2theta = float(parts[9])
                    radius_um = float(parts[10])
                    if ring_nr not in ring_radii:
                        ring_radii[ring_nr] = (radius_um, d_spacing, ideal_2theta)
                except (ValueError, IndexError):
                    continue

        if not ring_radii:
            logger.warning("Validation skipped: no ring radii from GetHKLList")
            return

        sorted_rings = sorted(ring_radii.items())  # [(ring_nr, (R_um, d, 2th)), ...]
        if state.max_ring_number > 0:
            sorted_rings = [(rn, v) for rn, v in sorted_rings if rn <= state.max_ring_number]
        logger.info(f"Integrator validation: {len(sorted_rings)} rings (max_ring={state.max_ring_number})")

        # --- 2. Write integrator parameter file ---
        max_r_px = max(r_um / state.px for _, (r_um, _, _) in sorted_rings)
        integ_params = os.path.join(work_dir, 'integrator_params.txt')
        logger.info(f"Integrator params: Lsd={state.lsd:.3f} BC=({state.ybc:.4f},{state.zbc:.4f}) "
                    f"ty={state.ty:.8f} tz={state.tz:.8f} "
                    f"p0={state.p0:.6e} p1={state.p1:.6e} p2={state.p2:.6e} p3={state.p3:.6e} "
                    f"p4={state.p4:.6e} p5={state.p5:.6e} p6={state.p6:.6e} p7={state.p7:.6e} "
                    f"p8={state.p8:.6e} p9={state.p9:.6e} p10={state.p10:.6e}")
        with open(integ_params, 'w') as f:
            f.write(f"Lsd {state.lsd}\n")
            f.write(f"BC {state.ybc} {state.zbc}\n")
            f.write(f"tx {state.tx}\n")
            f.write(f"ty {state.ty}\n")
            f.write(f"tz {state.tz}\n")
            f.write(f"p0 {state.p0}\n")
            f.write(f"p1 {state.p1}\n")
            f.write(f"p2 {state.p2}\n")
            f.write(f"p3 {state.p3}\n")
            if state.p4 != 0.0:
                f.write(f"p4 {state.p4}\n")
            if state.p5 != 0.0:
                f.write(f"p5 {state.p5}\n")
            if state.p6 != 0.0:
                f.write(f"p6 {state.p6}\n")
            if state.p7 != 0.0:
                f.write(f"p7 {state.p7}\n")
            if state.p8 != 0.0:
                f.write(f"p8 {state.p8}\n")
            if state.p9 != 0.0:
                f.write(f"p9 {state.p9}\n")
            if state.p10 != 0.0:
                f.write(f"p10 {state.p10}\n")
            f.write(f"RhoD {state.rhod}\n")
            f.write(f"Wavelength {state.wavelength}\n")
            f.write(f"px {state.px}\n")
            f.write(f"NrPixelsY {state.nr_pixels_y}\n")
            f.write(f"NrPixelsZ {state.nr_pixels_z}\n")
            f.write(f"RMin 10\n")
            f.write(f"RMax {max_r_px + 50:.0f}\n")
            f.write(f"RBinSize 0.25\n")
            f.write(f"EtaMin -180\n")
            f.write(f"EtaMax 180\n")
            f.write(f"EtaBinSize 1\n")
            f.write(f"Normalize 1\n")
            f.write(f"Folder {work_dir}\n")
            for transOpt in state.im_trans_opt:
                f.write(f"ImTransOpt {transOpt}\n")
            if state.mask_file:
                f.write(f"MaskFile {state.mask_file}\n")
            if state.parallax_in != 0.0:
                f.write(f"Parallax {state.parallax_in}\n")

        # --- 3. Run DetectorMapper ---
        logger.info("Running DetectorMapper...")
        dm_result = subprocess.run(
            [mapper_bin, integ_params, '-nCPUs', str(n_cpus)],
            capture_output=True, text=True, cwd=work_dir, env=env)
        if dm_result.returncode != 0:
            logger.error(f"DetectorMapper failed: {dm_result.stderr[-500:]}")
            return

        map_bin_path = os.path.join(work_dir, 'Map.bin')
        if not os.path.exists(map_bin_path):
            logger.error("DetectorMapper did not produce Map.bin")
            return

        # --- 4. Write peak params file ---
        peak_params_fn = os.path.join(work_dir, 'peak_params.txt')
        max_peaks = min(len(sorted_rings), 25)
        with open(peak_params_fn, 'w') as f:
            f.write("DoPeakFit 1\n")
            f.write("FitROIPadding 30\n")
            for i, (ring_nr, (r_um, _, _)) in enumerate(sorted_rings):
                if i >= max_peaks:
                    break
                r_px = r_um / state.px
                f.write(f"PeakLocation {r_px:.6f}\n")

        # --- 5. Run IntegratorZarrOMP ---
        logger.info("Running IntegratorZarrOMP...")
        integ_cmd = [
            integrator_bin,
            '-paramFN', integ_params,
            '-dataFN', os.path.abspath(data_file),
            '-nCPUs', str(n_cpus),
            '-PeakParamsFN', peak_params_fn,
        ]
        if dark_file and os.path.exists(dark_file):
            integ_cmd.extend(['-darkFN', os.path.abspath(dark_file)])

        integ_result = subprocess.run(
            integ_cmd, capture_output=True, text=True, cwd=work_dir, env=env)
        if integ_result.returncode != 0:
            logger.error(f"IntegratorZarrOMP failed: {integ_result.stderr[-500:]}")
            return

        # --- 6. Parse _fit_per_eta.csv ---
        # Find the output file
        fit_csvs = [f for f in os.listdir(work_dir) if f.endswith('_fit_per_eta.csv')]
        if not fit_csvs:
            logger.warning("No _fit_per_eta.csv found — integrator may not have produced peak fits")
            return

        fit_csv_path = os.path.join(work_dir, fit_csvs[0])
        per_eta_rows = []
        with open(fit_csv_path) as csvf:
            reader = csv.DictReader(csvf)
            for row in reader:
                try:
                    per_eta_rows.append({
                        'eta_deg': float(row['EtaCen']),
                        'peak_nr': int(row['PeakIdx']),
                        'center_px': float(row['R_px']),
                        'tth_deg': float(row['TwoTheta_deg']),
                    })
                except (KeyError, ValueError):
                    continue

        if not per_eta_rows:
            logger.warning("No peak fit data in _fit_per_eta.csv")
            return

        logger.info(f"Parsed {len(per_eta_rows)} per-eta peak fits")

        # --- 7. Write integrator_fn.corr.csv ---
        # Note: CI's M-step (CalibrationCore.c:386) computes
        #   RIdeal = Lsd * tan(2θ) / px  (no distortion polynomial)
        # and compares against R from dg_pixel_to_REta (which includes
        # distortion).  After calibration the two match closely (~10 µε).
        # IZOMP's peak fitter (pf_fit_single_peak) produces slightly
        # different centres than CI's (calib_fit_peak_shape), so the
        # validation strain will be somewhat larger than CI's.
        # Compute lattice parameter 'a' from fitted 2theta using Bragg's law
        # For cubic: a = d * sqrt(h^2+k^2+l^2), but since we don't have hkl per row,
        # we use: d = wavelength / (2 * sin(theta)), and IdealA from ideal 2theta.
        data_basename = os.path.basename(data_file)
        data_stem = os.path.splitext(data_basename)[0]
        corr_csv_name = f"integrator_{data_stem}.corr.csv"
        # Write to cwd (data dir may be read-only)
        corr_csv_path = os.path.join(os.getcwd(), corr_csv_name)

        n_written = 0
        with open(corr_csv_path, 'w') as out:
            out.write("%Eta Strain RadFit EtaCalc DiffCalc RadCalc "
                      "Ideal2Theta Outlier YRawCorr ZRawCorr RingNr "
                      "RadGlobal IdealR Fit2Theta IdealA FitA\n")

            for row in per_eta_rows:
                fitted_r_px = row['center_px']
                eta_deg = row['eta_deg']
                fitted_2theta = row['tth_deg']

                # Skip zero/invalid fits
                if fitted_r_px <= 0 or fitted_2theta <= 0:
                    continue

                # Dynamically match fitted radius to the closest valid ideal ring
                best_match = None
                best_err = 0.05  # 5% relative error tracking ceiling
                for idx, (rng_nr, (i_r_um, d_sp, i_2th)) in enumerate(sorted_rings):
                    i_r_px = i_r_um / state.px  # Lsd*tan(2θ)/px (undistorted, CI convention)
                    err = abs(fitted_r_px - i_r_px) / (i_r_px if i_r_px > 0 else 1.0)
                    if err < best_err:
                        best_err = err
                        best_match = (rng_nr, i_r_px, i_2th, d_sp)

                if best_match is None:
                    continue  # Noise or completely mismatched peak

                ring_nr, ideal_r_px, ideal_2theta, d_spacing = best_match

                # Strain matching CI convention (CalibrationCore.c:388):
                #   strain = 1 - R_px / RIdeal_px
                # Both R values are now in the same distortion-corrected
                # pixel space (IZOMP fitted R vs distorted ideal R).
                strain = 1.0 - fitted_r_px / ideal_r_px if ideal_r_px > 0 else 0.0
                diff_calc = abs(strain)

                # Y/Z from R,Eta (approximate: ignoring distortion)
                eta_rad = math.radians(eta_deg)
                # MIDAS convention (DetectorGeometry.c): Yc = (-Y + Ycen)*px
                # → Y_pixel = Ycen + R*sin(η)  (CI uses same: line 298)
                y_raw = state.ybc + fitted_r_px * math.sin(eta_rad)
                z_raw = state.zbc + fitted_r_px * math.cos(eta_rad)

                # Note: no detector-extent filter here — the integrator
                # already constrains output to physical detector pixels.
                # Filtering by y_raw/z_raw vs NrPixels is incorrect for
                # offset detectors where BC lies outside the sensor.

                # Lattice parameter from Bragg's law
                # d_fit = wavelength / (2 * sin(fitted_2theta/2 * deg2rad))
                # For cubic: a = d * sqrt(h²+k²+l²)
                # We compute ratio: a_fit / a_ideal = d_fit / d_ideal
                ideal_theta_rad = math.radians(ideal_2theta / 2.0)
                fit_theta_rad = math.radians(fitted_2theta / 2.0)

                d_ideal = state.wavelength / (2.0 * math.sin(ideal_theta_rad)) if ideal_theta_rad > 0 else 0
                d_fit = state.wavelength / (2.0 * math.sin(fit_theta_rad)) if fit_theta_rad > 0 else 0

                # Ideal lattice param (assume cubic, from first lattice const)
                ideal_a = state.latc[0] if state.latc[0] > 0 else d_ideal
                fit_a = ideal_a * (d_fit / d_ideal) if d_ideal > 0 else ideal_a

                out.write(f"{eta_deg:.6f} {strain:.10e} {fitted_r_px:.6f} "
                          f"{eta_deg:.6f} {diff_calc:.10e} {fitted_r_px:.6f} "
                          f"{ideal_2theta:.6f} 0 {y_raw:.4f} {z_raw:.4f} "
                          f"{ring_nr} {fitted_r_px:.6f} {ideal_r_px:.6f} "
                          f"{fitted_2theta:.6f} {ideal_a:.6f} {fit_a:.6f}\n")
                n_written += 1

        logger.info(f"Wrote {n_written} entries to {corr_csv_path}")
        print(f"\n  Integrator validation: {corr_csv_path}")
        print(f"  ({n_written} peak-fit entries across {len(sorted_rings)} rings)")

    except Exception as e:
        logger.error(f"Integrator validation failed: {traceback.format_exc()}")
    finally:
        # Clean up work directory (disabled for debugging)
        # import shutil
        # try:
        #     shutil.rmtree(work_dir, ignore_errors=True)
        # except Exception:
        #     pass
        logger.info(f"Keeping work dir for debugging: {work_dir}")


def runMIDAS(rawFN, state, n_iterations=40, mult_factor=5,
             doublet_separation=25, outlier_iterations=3,
             eta_bin_size=1.0, max_width=1000, n_cpus=None,
             stage=0, stage_label='',
             trimmed_mean_fraction=0.75,
             remove_outliers_between_iters=1,
             iter_offset=0):
    """Run CalibrantIntegratorOMP.

    stage=0: full optimization (legacy behavior)
    stage=1: geometry only — Lsd, BC, tilts (tolP=0, no panels)
    stage=2: full — distortion + panels using refined geometry from stage 1
    """
    if n_cpus is None:
        n_cpus = os.cpu_count() or 8

    ps_file = os.path.join(os.getcwd(), f"{os.path.basename(rawFN)}ps.txt")

    try:
        with open(ps_file, 'w') as pf:
            # Ring exclusions
            for ringNr in state.rings_to_exclude:
                pf.write(f'RingsToExclude {ringNr}\n')
            if state.max_ring_number > 0:
                pf.write(f'MaxRingNumber {state.max_ring_number}\n')
            if state.fit_parallax > 0:
                pf.write(f'FitParallax {state.fit_parallax}\n')
                pf.write(f'Parallax {state.parallax_in}\n')
                pf.write(f'tolParallax {state.tol_parallax}\n')
            if state.peak_fit_mode > 0:
                pf.write(f'PeakFitMode {state.peak_fit_mode}\n')

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
                pf.write('tolP6 90\n')  # phase: cos(2η) has period 180°, need ±90°
                pf.write('tolP7 1E-3\n')
                pf.write('tolP8 180\n')
                pf.write('tolP9 1E-3\n')
                pf.write('tolP10 180\n')

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
            if state.p5 != 0.0:
                pf.write(f'p5 {state.p5}\n')
            if state.p6 != 0.0:
                pf.write(f'p6 {state.p6}\n')
            # Break polar coordinate gradient singularities before passing to NLopt
            _p7_seed = state.p7 if state.p7 != 0.0 else (1e-4 if stage != 1 else 0.0)
            _p8_seed = state.p8 if state.p8 != 0.0 else (45.0 if stage != 1 else 0.0)
            _p9_seed = state.p9 if state.p9 != 0.0 else (1e-4 if stage != 1 else 0.0)
            _p10_seed = state.p10 if state.p10 != 0.0 else (45.0 if stage != 1 else 0.0)

            if _p7_seed != 0.0: pf.write(f'p7 {_p7_seed}\n')
            if _p8_seed != 0.0: pf.write(f'p8 {_p8_seed}\n')
            if _p9_seed != 0.0: pf.write(f'p9 {_p9_seed}\n')
            if _p10_seed != 0.0: pf.write(f'p10 {_p10_seed}\n')
            pf.write(f'EtaBinSize {eta_bin_size}\n')
            pf.write(f'HeadSize {8192 if state.midas_dtype == 1 else 0}\n')

            # Bad pixel / gap
            if not math.isnan(state.bad_px_intensity):
                pf.write(f'BadPxIntensity {state.bad_px_intensity}\n')
            if not math.isnan(state.gap_intensity):
                pf.write(f'GapIntensity {state.gap_intensity}\n')

            # Data specifics
            if state.dark_name:
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
            pf.write(f'IterOffset {iter_offset}\n')
            pf.write(f'DoubletSeparation {doublet_separation}\n')
            pf.write(f'MultFactor {mult_factor}\n')
            pf.write(f'OutlierIterations {outlier_iterations}\n')
            pf.write('NormalizeRingWeights 1\n')
            pf.write('MinIndicesForFit 5\n')
            pf.write('WeightByRadius 1\n')
            pf.write('WeightByFitSNR 1\n')
            pf.write('L2Objective 1\n')
            pf.write(f'TrimmedMeanFraction {trimmed_mean_fraction}\n')
            pf.write(f'RemoveOutliersBetweenIters {remove_outliers_between_iters}\n')

            # CalibrantIntegratorOMP parameters
            pf.write('RBinWidth 4\n')

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

            # Pass-through extra params from user's param file (last-wins
            # semantics: appending at the end overrides any earlier defaults)
            if state.extra_params:
                pf.write('\n# Extra parameters from user param file\n')
                for k, v in state.extra_params.items():
                    pf.write(f'{k} {v}\n')

        # Run calibrant executable (CalibrantIntegratorOMP)
        calibrant_exe = os.path.join(INSTALL_PATH, 'FF_HEDM/bin/CalibrantIntegratorOMP')
        exe_name = 'CalibrantIntegratorOMP'
        calibrant_cmd = f"{calibrant_exe} {ps_file} {n_cpus}"

        logger.info(f"{stage_label}Running {exe_name} with {n_cpus} CPUs, "
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
                # Print key output to terminal
                stripped = line.rstrip()
                if any(kw in line for kw in ['MeanStrain', 'Iteration', 'Restoring best',
                                              'Doublet detected', 'nIterations',
                                              'per-ring', 'Ring ', '---',
                                              'Best result', 'Post-loop']):
                    print(f"  {stripped}")
                # Per-ring table rows: start with spaces then digits (e.g. "     4  70256.63 ...")
                elif stripped and stripped.lstrip().replace('.','',1).replace('-','',1).replace('+','',1)[:4].replace(' ','').isdigit():
                    print(f"  {stripped}")
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
    px = args.px if args.px is not None else 200.0
    wavelength = args.wavelength if args.wavelength is not None else filename_hints.get('wavelength', 0.0)
    latc = calibrant['lattice']
    sg = calibrant['space_group']

    if wavelength <= 0:
        logger.error("Cannot create temp param file: wavelength not available "
                     "(use --wavelength, set energy in filename, or provide --params)")
        print("ERROR: Wavelength not available. Use --wavelength, include energy in the "
              "filename (e.g. 71p676keV), or provide --params")
        sys.exit(1)

    temp_fn = '_autocal_temp_params.txt'
    with open(temp_fn, 'w') as f:
        f.write(f"SpaceGroup {sg}\n")
        f.write(f"LatticeParameter {' '.join(str(v) for v in latc)}\n")
        f.write(f"Wavelength {wavelength}\n")
        f.write(f"px {px}\n")
        f.write(f"SkipFrame 0\n")
        f.write(f"tx {args.tx if args.tx is not None else 0.0}\n")
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
        parser.add_argument('--data', '-dataFN', type=str, required=True,
                            help='Data file: .zip (Zarr), .h5 (HDF5), .ge* (GE binary), .tif/.tiff, or .cbf')
        parser.add_argument('--dark', '-darkFN', type=str, default='',
                            help='Separate dark field image file')
        parser.add_argument('--params', '-paramFN', type=str, default='',
                            help='MIDAS parameter file (required for non-Zarr inputs)')
        parser.add_argument('--data-loc', '-dataLoc', type=str, default='',
                            help='HDF5 dataset path (default: /exchange/data)')
        parser.add_argument('--dark-loc', '-darkLoc', type=str, default='',
                            help='HDF5 dark field dataset path (default: exchange/dark)')

        # Format control
        parser.add_argument('--convert', '-ConvertFile', type=int, default=-1,
                            help='Force format conversion: 0=Zarr, 1=HDF5, 2=GE, 3=TIFF, 4=CBF. '
                                 'Default: auto-detect from extension.')

        # Calibration control
        parser.add_argument('--n-iterations', type=int, default=40,
                            help='Number of C-side calibration iterations')
        parser.add_argument('--mult-factor', '-MultFactor', type=float, default=5,
                            help='Outlier ring rejection factor (× median strain)')
        parser.add_argument('--doublet-separation', type=float, default=25.0,
                            help='Doublet detection threshold (pixels)')
        parser.add_argument('--outlier-iterations', type=int, default=3,
                            help='Per-ring outlier removal iterations')
        parser.add_argument('--first-ring', '-FirstRingNr', type=int, default=1,
                            help='First ring number to use')
        parser.add_argument('--max-ring', '-MaxRingNumber', type=int, default=None,
                            help='Maximum ring number to use (0 = no limit, default: auto)')
        parser.add_argument('--fit-parallax', type=int, default=None,
                            help='Fit parallax correction (0=no, 1=yes)')
        parser.add_argument('--parallax-guess', type=float, default=None,
                            help='Initial guess for parallax (µm)')
        parser.add_argument('--tol-parallax', type=float, default=None,
                            help='Tolerance for parallax bounds (µm)')
        parser.add_argument('--peak-fit-mode', type=int, default=None,
                            help='Peak fitting mode: 0=pV (default), 1=TCH (GSAS-II)')
        parser.add_argument('--trimmed-mean-fraction', type=float, default=0.75,
                            help='Fraction of points to keep in optimizer objective '
                                 '(e.g. 0.75 = trim worst 25%%). 1.0 = off.')
        parser.add_argument('--remove-outliers-between-iters', type=int, default=1,
                            help='Remove outlier points between calibration iterations '
                                 '(0=off, 1=on). Default: 1')
        parser.add_argument('--eta-bin-size', '-EtaBinSize', type=float, default=1.0,
                            help='Azimuthal bin size (degrees)')

        # Geometry guesses
        parser.add_argument('--lsd-guess', '-LsdGuess', type=float, default=None,
                            help='Initial guess for detector distance (µm, default: 1000000)')
        parser.add_argument('--bc-guess', '-BCGuess', type=float, default=None, nargs=2,
                            help='Initial guess for beam center [Y Z] (pixels)')
        parser.add_argument('--px', type=float, default=None,
                            help='Pixel size in µm (e.g. 200, 172). If set, no param file needed for non-Zarr inputs.')
        parser.add_argument('--tx', type=float, default=None,
                            help='Detector tilt tx (radians, not fitted but passed to CalibrantPanelShiftsOMP)')
        parser.add_argument('--wavelength', '-wl', type=float, default=None,
                            help='X-ray wavelength in Angstroms (e.g. 0.2066). Overrides filename and param file.')
        parser.add_argument('--material', '-mat', type=str, default=None,
                            choices=[None, 'ceo2', 'lab6'],
                            help='Calibrant material: ceo2 or lab6. Overrides filename detection.')
        parser.add_argument('--mask', '-MaskFile', type=str, default='',
                            help='Mask TIFF file for bad/gap pixels (passed as MaskFile to CalibrantPanelShiftsOMP)')

        parser.add_argument('--skip-panels', action='store_true', default=False,
                            help='Use mask for pixel masking but skip panel auto-detection and fitting')

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
        parser.add_argument('--plots', '-MakePlots', type=int, default=0,
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

        # (--use-integrator removed: CalibrantIntegratorOMP is now the sole exe)

        # Output
        parser.add_argument('--output', '-o', type=str, default='',
                            help='Output parameter filename (default: refined_MIDAS_params_<stem>.txt)')
        parser.add_argument('--no-validate', action='store_true',
                            help='Skip post-calibration integrator validation')

        args = parser.parse_args()

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
        # Initialize from CLI (None = not provided; will be filled from param file or defaults later)
        if args.max_ring is not None:
            state.max_ring_number = args.max_ring
        if args.fit_parallax is not None:
            state.fit_parallax = args.fit_parallax
        if args.parallax_guess is not None:
            state.parallax_in = args.parallax_guess
        if args.tol_parallax is not None:
            state.tol_parallax = args.tol_parallax
        if args.peak_fit_mode is not None:
            state.peak_fit_mode = args.peak_fit_mode

        # HDF5 output
        if args.save_hdf:
            logger.info(f"Will save data arrays to HDF5 file: {args.save_hdf}")
            state.h5_file = h5py.File(args.save_hdf, 'w')
            meta_group = state.h5_file.create_group('metadata')
            meta_group.attrs['file_name'] = os.path.basename(dataFN)
            meta_group.attrs['date_created'] = pd.Timestamp.now().isoformat()

        # Constants
        mrr = 2000000  # maximum radius to simulate rings
        initialLsd = args.lsd_guess if args.lsd_guess is not None else 1000000.0
        _lsd_from_cli = args.lsd_guess is not None
        minArea = 300
        maxW = 1000

        # Track which params came from the param file (vs auto-detection)
        param_file_keys = set()

        # Parse energy/distance hints from filename
        filename_hints = parse_filename_hints(args.data)

        logger.info(f"Starting automated calibration for: {dataFN}")

        # ---- Auto-detect file format and DataType ----
        convertFile = args.convert
        if convertFile < 0:
            convertFile = detect_format(dataFN)
            logger.info(f"Auto-detected format: {['Zarr','HDF5','GE','TIFF','CBF'][convertFile]}")

        if args.data_type >= 0:
            state.midas_dtype = args.data_type
            logger.info(f"DataType override from --data-type: {state.midas_dtype}")
        else:
            state.midas_dtype = detect_data_type(dataFN)
            logger.info(f"Auto-detected DataType: {state.midas_dtype}")

        # Detect calibrant: CLI --material > filename detection > default CeO2
        if args.material:
            calibrant = CALIBRANTS[args.material]
            logger.info(f"Using calibrant from --material: {calibrant['name']} "
                        f"(SpaceGroup {calibrant['space_group']})")
        else:
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
        raw, dark, ny, nz, mask_sentinels = read_image_for_estimation(
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
            elif args.px is not None:
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
            # Non-Zarr: read param file FIRST, then auto-detect only
            # for values not explicitly set by the param file.
            state.space_group = calibrant['space_group']
            state.latc = calibrant['lattice'].copy()

            # Keys we explicitly parse and apply to state/initialLsd
            _KNOWN_PARAM_KEYS = {
                'Wavelength', 'SpaceGroup', 'LatticeConstant', 'LatticeParameter',
                'px', 'lsd', 'Lsd', 'BC', 'bc', 'tx', 'NrPixels', 'NrPixelsY',
                'NrPixelsZ', 'RhoD', 'rhod',
                # Second-pass keys also parsed here now:
                'RingsToExclude', 'MaxRingNumber', 'FitParallax', 'Parallax',
                'tolParallax', 'MaskFile', 'MaskFN',
                'ImTransOpt', 'BadPxIntensity', 'GapIntensity',
                'Dark', 'dataLoc', 'darkLoc', 'skipFrame', 'SkipFrame',
                'DataType', 'PeakFitMode',
            }

            # --- Single-pass param file read ---
            if args.params:
                try:
                    with open(args.params) as pf:
                        for line in pf:
                            line_stripped = line.strip()
                            if not line_stripped or line_stripped.startswith('#'):
                                continue
                            parts = line_stripped.split()
                            if len(parts) < 2:
                                continue
                            key = parts[0]

                            # --- Explicitly parsed keys ---
                            if key == 'Wavelength':
                                state.wavelength = float(parts[1])
                                param_file_keys.add('Wavelength')
                            elif key == 'SpaceGroup':
                                state.space_group = int(parts[1])
                                param_file_keys.add('SpaceGroup')
                            elif key in ('LatticeConstant', 'LatticeParameter'):
                                state.latc = np.array([float(x) for x in parts[1:7]])
                                param_file_keys.add('LatticeParameter')
                            elif key == 'px':
                                if args.px is None:  # CLI not provided → use param file
                                    state.px = float(parts[1])
                                param_file_keys.add('px')
                            elif key in ('lsd', 'Lsd'):
                                if not _lsd_from_cli:  # CLI not provided → use param file
                                    initialLsd = float(parts[1])
                                param_file_keys.add('lsd')
                            elif key in ('BC', 'bc'):
                                if len(parts) >= 3 and args.bc_guess is None:
                                    args.bc_guess = [float(parts[1]), float(parts[2])]
                                param_file_keys.add('BC')
                            elif key == 'tx':
                                if args.tx is None:
                                    state.tx = float(parts[1])
                                param_file_keys.add('tx')
                            elif key in ('NrPixels', 'NrPixelsY'):
                                state.nr_pixels_y = int(parts[1])
                                param_file_keys.add('NrPixelsY')
                            elif key == 'NrPixelsZ':
                                state.nr_pixels_z = int(parts[1])
                                param_file_keys.add('NrPixelsZ')
                            elif key in ('RhoD', 'rhod'):
                                state.rhod = float(parts[1])
                                param_file_keys.add('RhoD')
                            elif key == 'RingsToExclude':
                                state.rings_to_exclude.append(int(parts[1]))
                                param_file_keys.add('RingsToExclude')
                            elif key == 'MaxRingNumber':
                                if args.max_ring is None:  # CLI not provided
                                    state.max_ring_number = int(parts[1])
                                param_file_keys.add('MaxRingNumber')
                            elif key == 'FitParallax':
                                if args.fit_parallax is None:
                                    state.fit_parallax = int(parts[1])
                                param_file_keys.add('FitParallax')
                            elif key == 'Parallax':
                                if args.parallax_guess is None:
                                    state.parallax_in = float(parts[1])
                                param_file_keys.add('Parallax')
                            elif key == 'tolParallax':
                                if args.tol_parallax is None:
                                    state.tol_parallax = float(parts[1])
                                param_file_keys.add('tolParallax')
                            elif key == 'PeakFitMode':
                                state.peak_fit_mode = int(parts[1])
                                param_file_keys.add('PeakFitMode')
                            elif key in ('MaskFile', 'MaskFN'):
                                if not args.mask:
                                    state.mask_file = parts[1]
                                param_file_keys.add('MaskFile')
                            elif key == 'ImTransOpt':
                                # Collect all ImTransOpt lines
                                if 'ImTransOpt' not in param_file_keys:
                                    state.im_trans_opt = []
                                state.im_trans_opt.append(int(parts[1]))
                                param_file_keys.add('ImTransOpt')
                            elif key == 'BadPxIntensity':
                                state.bad_px_intensity = float(parts[1])
                                param_file_keys.add('BadPxIntensity')
                            elif key == 'GapIntensity':
                                state.gap_intensity = float(parts[1])
                                param_file_keys.add('GapIntensity')
                            elif key == 'Dark':
                                if not darkFN:
                                    darkFN = parts[1]
                                param_file_keys.add('Dark')
                            elif key == 'dataLoc':
                                if not state.data_loc:
                                    state.data_loc = parts[1]
                                param_file_keys.add('dataLoc')
                            elif key == 'darkLoc':
                                if not state.dark_loc:
                                    state.dark_loc = parts[1]
                                param_file_keys.add('darkLoc')
                            elif key in ('skipFrame', 'SkipFrame'):
                                state.skip_frame = int(parts[1])
                                param_file_keys.add('SkipFrame')
                            elif key == 'DataType':
                                if args.data_type < 0:  # CLI not provided
                                    state.midas_dtype = int(parts[1])
                                param_file_keys.add('DataType')
                            else:
                                # Unrecognized key → store for pass-through
                                # to CalibrantPanelShiftsOMP
                                if key not in state.extra_params:
                                    state.extra_params[key] = ' '.join(parts[1:])
                                    param_file_keys.add(key)

                    if param_file_keys:
                        logger.info(f"Loaded from parameter file: {sorted(param_file_keys)}")
                    if state.extra_params:
                        logger.info(f"Pass-through params for CalibrantPanelShiftsOMP: "
                                    f"{sorted(state.extra_params.keys())}")
                except Exception as e:
                    logger.warning(f"Could not fully parse param file: {e}")

            # --- px: CLI > param file > auto-detect ---
            if args.px is not None:
                state.px = args.px
                logger.info(f"Using px from --px: {state.px:.1f} µm")
            elif 'px' not in param_file_keys:
                if convertFile == 4:
                    from read_cbf import read_cbf_metadata
                    cbf_hdr = read_cbf_metadata(dataFN)
                    pilatus_hdr = cbf_hdr.get('pilatus', {})
                    if not isinstance(pilatus_hdr, dict):
                        pilatus_hdr = {}
                    px_vals = pilatus_hdr.get('Pixel_size', None)
                    if px_vals and isinstance(px_vals, (list, tuple)) and len(px_vals) > 0:
                        cbf_px_m = float(px_vals[0])
                        if cbf_px_m > 0:
                            state.px = cbf_px_m * 1e6
                            logger.info(f"Auto-sensed pixel size from CBF header: {state.px:.1f} µm")
                        else:
                            state.px = 200.0
                    else:
                        state.px = 200.0
                elif state.px == 200.0:  # still at default
                    # Auto-detect pixel size from detector shape
                    _px_detected = False
                    try:
                        fn_ext = Path(dataFN).suffix.lower()
                        if fn_ext in ('.tif', '.tiff'):
                            _img = Image.open(dataFN)
                            _shape = (_img.size[1], _img.size[0])
                        elif fn_ext in ('.h5', '.hdf5', '.hdf', '.nxs'):
                            with h5py.File(dataFN, 'r') as _f:
                                _dloc = args.data_loc or 'exchange/data'
                                _shape = _f[_dloc].shape[-2:]
                        else:
                            _shape = None

                        if _shape is not None:
                            _sorted = sorted(_shape)
                            if _sorted == [1475, 1679]:
                                state.px = 172.0
                                _px_detected = True
                                logger.info(f"Auto-detected Pilatus detector "
                                            f"({_shape[0]}×{_shape[1]}) → pixel size 172 µm")
                            elif _sorted == [2048, 2048]:
                                state.px = 200.0
                                _px_detected = True
                                logger.info(f"Auto-detected GE detector "
                                            f"({_shape[0]}×{_shape[1]}) → pixel size 200 µm")
                            elif _sorted == [2880, 2880]:
                                state.px = 150.0
                                _px_detected = True
                                logger.info(f"Auto-detected Varex detector "
                                            f"({_shape[0]}×{_shape[1]}) → pixel size 150 µm")
                    except Exception:
                        pass

                    if not _px_detected:
                        # Fallback: check filename keywords
                        base_lower = Path(dataFN).name.lower()
                        if '.vrx' in base_lower or 'varex' in base_lower:
                            state.px = 150.0
                            logger.info("Auto-detected Varex detector from filename → pixel size 150 µm")
                        else:
                            state.px = 200.0
            else:
                logger.info(f"Using px from parameter file: {state.px:.1f} µm")

            # --- wavelength: CLI > param file > filename hint > 0 ---
            if args.wavelength is not None:
                state.wavelength = args.wavelength
                logger.info(f"Using wavelength from --wavelength: {state.wavelength:.6f} Å")
            elif 'Wavelength' not in param_file_keys and 'wavelength' in filename_hints:
                state.wavelength = filename_hints['wavelength']
                logger.info(f"Using wavelength from filename: {state.wavelength:.6f} Å")
            elif 'Wavelength' in param_file_keys:
                logger.info(f"Using wavelength from parameter file: {state.wavelength:.6f} Å")

            # --- lsd: CLI > param file > filename hint > default ---
            if _lsd_from_cli:
                logger.info(f"Using Lsd from --lsd-guess: {initialLsd:.0f} µm")
            elif 'lsd' in param_file_keys:
                logger.info(f"Using Lsd from parameter file: {initialLsd:.0f} µm")
            elif 'lsd' in filename_hints:
                initialLsd = filename_hints['lsd']
                logger.info(f"Using Lsd from filename: {initialLsd:.0f} µm")

            # --- tx: CLI > param file > default ---
            if args.tx is not None:
                state.tx = args.tx
                logger.info(f"Using tx from --tx: {state.tx}")

            # --- material: CLI > param file > calibrant auto-detect ---
            if args.material:
                state.space_group = calibrant['space_group']
                state.latc = calibrant['lattice'].copy()
                logger.info(f"Material from --material: {calibrant['name']} "
                            f"(SG={state.space_group}, a={state.latc[0]:.4f})")

            logger.info(f"Using pixel size: {state.px:.1f} µm")

        # Save pre-mask copy for viewer display
        raw_for_display = raw.copy()

        if args.mask:
            state.mask_file = str(Path(args.mask).absolute())
            logger.info(f"MaskFile from --mask: {state.mask_file}")
        elif not state.mask_file:
            # Auto-detect: use sentinel values detected before clamping
            if mask_sentinels:
                logger.info(f"Auto-detected mask sentinel values: "
                            f"{mask_sentinels} — generating mask")
                mask_intensities = mask_sentinels
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

            # Auto-detect panel layout from mask (unless --skip-panels)
            if not args.skip_panels:
                bc_guess = None
                if state.ybc > 0 and state.zbc > 0:
                    bc_guess = (state.ybc, state.zbc)
                state.panel_grid = detect_panels_from_mask(state.mask_file, bc_guess=bc_guess)
            else:
                logger.info("--skip-panels: mask used for pixel masking only, no panel fitting")

        # Parse raw filename for CalibrantPanelShiftsOMP file naming
        rawFN = os.path.abspath(dataFN)

        # Apply bad_gap_arr mask if created from mask file
        if len(bad_gap_arr) != 0:
            raw = np.ma.masked_array(raw, mask=bad_gap_arr)

        # Set dark_name for CalibrantPanelShiftsOMP
        if darkFN:
            state.dark_name = os.path.abspath(darkFN)
        elif state.ext.lower() in ('.h5', '.hdf5', '.hdf', '.nxs'):
            # HDF5 files can embed dark in the same file
            state.dark_name = os.path.abspath(dataFN)
        else:
            # All other formats: no dark subtraction without explicit dark
            state.dark_name = ''
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
        # NOTE: BC detection runs on the UN-TRANSFORMED image, then we apply
        #       ImTransOpt as a coordinate transform afterwards.
        bcg = args.bc_guess
        if bcg is None or bcg[0] == 0:
            import time as _time
            logger.info("Auto-detecting beam center")
            t0 = _time.perf_counter()
            labels, nlabels = measure.label(thresh, return_num=True)
            props = measure.regionprops(labels)
            t1 = _time.perf_counter()
            logger.info(f"  Labeling + regionprops: {t1-t0:.3f}s  ({nlabels} labels)")

            # Bulk-filter small regions (single pass via lookup table)
            keep = np.zeros(nlabels + 1, dtype=bool)
            for rp in props:
                if rp.area >= minArea:
                    keep[rp.label] = True
            thresh[~keep[labels]] = 0
            n_kept = int(np.sum(keep))
            t2 = _time.perf_counter()
            logger.info(f"  Bulk filter: {t2-t1:.3f}s  ({n_kept} regions kept, {nlabels-n_kept} removed)")

            # Beam center uses internal 4× downsampling for speed
            # Returns [row, col] — image is already in MIDAS convention
            # (flipped at load time, lines 2248-2261), so row=BC_Z, col=BC_Y
            bc_raw = detect_beam_center_optimized(thresh, minArea)
            t3 = _time.perf_counter()
            logger.info(f"  Beam center detection: {t3-t2:.3f}s  "
                        f"BC_Y(col)={bc_raw[1]:.1f}, BC_Z(row)={bc_raw[0]:.1f}")

            # bc_computed = [row, col] = [BC_Z, BC_Y] (MIDAS convention)
            bc_computed = bc_raw

            rads = detect_ring_radii(labels, props, bc_computed, minArea)
            t4 = _time.perf_counter()
            logger.info(f"  Ring radii detection: {t4-t3:.3f}s  (total BC pipeline: {t4-t0:.3f}s)")

            # Auto-detect max ring number if not already set
            # bc_computed = [row, col] = [BC_Z, BC_Y]
            # auto_detect_max_ring wants (bc_y, bc_z) = (col, row)
            if state.max_ring_number <= 0:
                NrPixelsY_det = state.nr_pixels_y if state.nr_pixels_y > 0 else data.shape[1]
                NrPixelsZ_det = state.nr_pixels_z if state.nr_pixels_z > 0 else data.shape[0]
                state.max_ring_number = auto_detect_max_ring(
                    sim_rads, NrPixelsY_det, NrPixelsZ_det,
                    bc_computed[1], bc_computed[0],  # (bc_y=col, bc_z=row)
                    data=data)

            if not _lsd_from_cli and 'lsd' not in param_file_keys:
                # Use estimate_lsd regardless of filename hint;
                # initialLsd (from filename or default) serves as seed
                initialLsd = estimate_lsd(rads, sim_rads, sim_rad_ratios,
                                          firstRing, initialLsd,
                                          max_ring=state.max_ring_number)

            # --- Tilted-detector geometry refinement ---
            # Disabled: auto_guess_tilted has not been validated on real
            # images.  CalibrantPanelShiftsOMP will find tilts iteratively.
            state.ty = 0.0
            state.tz = 0.0
            logger.info("  Tilt estimation disabled — tilts set to 0 "
                        "(CalibrantPanelShiftsOMP will refine)")
        else:
            bc_computed = np.flip(np.array(bcg))

        bc_new = bc_computed
        logger.info(f"FN: {rawFN}, Beam Center guess: {np.flip(bc_new)}, Lsd guess: {initialLsd}, "
                     f"ty guess: {state.ty:.3f}°, tz guess: {state.tz:.3f}°")

        # Re-run ring simulation with updated Lsd (CLI mode)
        hkls = run_get_hkl_list_cli(
            state.space_group, state.latc, state.wavelength,
            initialLsd, state.rhod)
        sim_rads = np.unique(hkls[:, -1]) / state.px
        sim_rad_ratios = sim_rads / sim_rads[0]
        # Filter to max_ring for display
        if state.max_ring_number > 0 and state.max_ring_number < len(sim_rads):
            sim_rads_display = sim_rads[:state.max_ring_number]
        else:
            sim_rads_display = sim_rads

        if state.h5_file:
            save_ring_data(thresh, bc_new, sim_rads, state.h5_file)

        # Rings/panel params already loaded in the single-pass param-file
        # read above (non-Zarr branch), or from Zarr metadata.
        # For Zarr inputs, read rings/panels from the param file if provided.
        if (convertFile == 0 or dataFN.endswith('.zip')) and args.params:
            try:
                with open(args.params, 'r') as pf:
                    for line in pf:
                        line_stripped = line.strip()
                        if not line_stripped or line_stripped.startswith('#'):
                            continue
                        parts = line_stripped.split()
                        if len(parts) < 2:
                            continue
                        key = parts[0]
                        if key == 'RingsToExclude':
                            state.rings_to_exclude.append(int(parts[1]))
                        elif key == 'MaxRingNumber':
                            if args.max_ring is None:
                                state.max_ring_number = int(parts[1])
                        elif key == 'FitParallax':
                            if args.fit_parallax is None:
                                state.fit_parallax = int(parts[1])
                        elif key == 'Parallax':
                            if args.parallax_guess is None:
                                state.parallax_in = float(parts[1])
                        elif key == 'tolParallax':
                            if args.tol_parallax is None:
                                state.tol_parallax = float(parts[1])
                        elif key not in param_file_keys:
                            # Extra pass-through for Zarr inputs
                            state.extra_params[key] = ' '.join(parts[1:])
                logger.info(f"Loaded manual exclusions: {state.rings_to_exclude}")
            except Exception as e:
                logger.warning(f"Could not read params from {args.params}: {e}")
        elif state.rings_to_exclude:
            logger.info(f"Loaded manual exclusions: {state.rings_to_exclude}")

        # Display rings overlay
        NrPixelsY = state.nr_pixels_y if state.nr_pixels_y > 0 else 2048
        NrPixelsZ = state.nr_pixels_z if state.nr_pixels_z > 0 else 2048
        if DrawPlots and viewer:
            if noMedian == 0:
                viewer.set_rings(thresh, sim_rads_display, bc_new, state.rings_to_exclude)
            else:
                log_img = np.log(data_corr + 1)
                viewer.set_rings(log_img, sim_rads_display, bc_new, state.rings_to_exclude)
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
                 stage_label='[Stage 1/2] ',
                 trimmed_mean_fraction=args.trimmed_mean_fraction,
                 remove_outliers_between_iters=args.remove_outliers_between_iters)

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
                 stage_label='[Stage 2/2] ',
                 trimmed_mean_fraction=args.trimmed_mean_fraction,
                 remove_outliers_between_iters=args.remove_outliers_between_iters,
                 iter_offset=stage1_iters)

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
                'p4': state.p4, 'p5': state.p5, 'p6': state.p6,
                'p7': state.p7, 'p8': state.p8, 'p9': state.p9, 'p10': state.p10,
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
        if state.p4 != 0.0:
            print(f"  p4             {state.p4}")
        if state.p5 != 0.0:
            print(f"  p5             {state.p5}")
        if state.p6 != 0.0:
            print(f"  p6             {state.p6}")
        if state.p7 != 0.0:
            print(f"  p7             {state.p7}")
        if state.p8 != 0.0:
            print(f"  p8             {state.p8}")
        if state.p9 != 0.0:
            print(f"  p9             {state.p9}")
        if state.p10 != 0.0:
            print(f"  p10            {state.p10}")
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
            'EtaBinSize': 1, 'DoSmoothing': 1, 'DoPeakFit': 1,
            'MultiplePeaks': 1,
            'PeakFitMode': state.peak_fit_mode,
        }

        if state.p4 != 0.0:
            final_params['p4'] = state.p4
        if state.p5 != 0.0:
            final_params['p5'] = state.p5
        if state.p6 != 0.0:
            final_params['p6'] = state.p6
        if state.p7 != 0.0:
            final_params['p7'] = state.p7
        if state.p8 != 0.0:
            final_params['p8'] = state.p8
        if state.p9 != 0.0:
            final_params['p9'] = state.p9
        if state.p10 != 0.0:
            final_params['p10'] = state.p10
        if state.parallax_in != 0.0:
            final_params['Parallax'] = state.parallax_in

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

        # ---- Integrator validation ----
        if not args.no_validate:
            logger.info("Running integrator validation...")
            run_integrator_validation(
                refined_params_file=psName,
                data_file=dataFN,
                dark_file=darkFN,
                state=state,
                n_cpus=n_cpus,
            )

    except Exception as e:
        logger.error(f"Error in main function: {traceback.format_exc()}")
        if hasattr(state, 'h5_file') and state.h5_file:
            state.h5_file.close()
        sys.exit(1)


# MIDAS version banner
try:
    from version import version_string as _vs
    print(_vs())
except Exception:
    pass

if __name__ == "__main__":
    main()
