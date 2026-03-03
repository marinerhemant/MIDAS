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

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import os
import zarr
import subprocess
from skimage import measure
import matplotlib.patches as mpatches
plt.rcParams['figure.figsize'] = [10, 10]
import argparse
import sys
import midas_config
midas_config.run_startup_checks()
import plotly.graph_objects as go
import pandas as pd
import diplib as dip
from plotly.subplots import make_subplots
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
    lsd: str = '1000000'
    bc: str = '1024 1024'
    tx: float = 0.0
    ty: str = '0'
    tz: str = '0'
    p0: str = '0'
    p1: str = '0'
    p2: str = '0'
    p3: str = '0'
    p4: str = '0'
    mean_strain: str = '1.0'
    std_strain: str = '0.0'
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

    # Panel
    panel_params: list = field(default_factory=list)
    panel_shifts_file: str = ''

    # Bad pixel mask
    bad_gap_arr: list = field(default_factory=list)
    mask_file: str = ''

    # HDF5 output
    h5_file: object = None

    # Image transforms
    im_trans_opt: list = field(default_factory=lambda: [0])


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


def generateZip(resFol, pfn, dfn='', darkfn='', dloc='', nchunks=-1,
                preproc=-1, outf='ZipOut.txt', errf='ZipErr.txt',
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
                if 'Lsd ' in line:
                    state.lsd = line.split()[1]
                if 'BC ' in line:
                    state.bc = line.split()[1] + ' ' + line.split()[2]
                if 'ty ' in line:
                    state.ty = line.split()[1]
                if 'tz ' in line:
                    state.tz = line.split()[1]
                if 'p0 ' in line:
                    state.p0 = line.split()[1]
                if 'p1 ' in line:
                    state.p1 = line.split()[1]
                if 'p2 ' in line:
                    state.p2 = line.split()[1]
                if 'p3 ' in line:
                    state.p3 = line.split()[1]
                if 'p4 ' in line:
                    state.p4 = line.split()[1]
                if 'MeanStrain ' in line:
                    state.mean_strain = str(float(line.split()[1]) / 1e6)
                if 'StdStrain ' in line:
                    state.std_strain = str(float(line.split()[1]) / 1e6)
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
             eta_bin_size=5.0, max_width=1000, n_cpus=None):
    """Run CalibrantPanelShiftsOMP with all features enabled.

    A single call handles multi-iteration refinement, ring outlier rejection,
    doublet detection, SNR weighting, and ring normalization internally.
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

            # Tolerances
            pf.write('tolTilts 3\n')
            pf.write('tolBC 20\n')
            pf.write('tolLsd 15000\n')
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

            # C-side iteration and advanced features
            pf.write(f'nIterations {n_iterations}\n')
            pf.write(f'DoubletSeparation {doublet_separation}\n')
            pf.write(f'OutlierIterations {outlier_iterations}\n')
            pf.write('NormalizeRingWeights 1\n')
            pf.write('MinIndicesForFit 5\n')
            pf.write('WeightByFitSNR 1\n')

            # Panel parameters
            if state.panel_params:
                for pp in state.panel_params:
                    pf.write(f'{pp}\n')
                pf.write(f'PanelShiftsFile {state.panel_shifts_file}\n')
                pf.write('tolShifts 1\n')

            # Mask file
            if state.mask_file:
                pf.write(f'MaskFile {state.mask_file}\n')

        # Run CalibrantPanelShiftsOMP with all available CPUs
        calibrant_exe = os.path.join(INSTALL_PATH, 'FF_HEDM/bin/CalibrantPanelShiftsOMP')
        calibrant_cmd = f"{calibrant_exe} {ps_file} {n_cpus}"

        logger.info(f"Running CalibrantPanelShiftsOMP with {n_cpus} CPUs, "
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

        args, unparsed = parser.parse_known_args()

        # ---- Initialize state ----
        state = CalibState()
        state.bad_px_intensity = args.bad_px
        state.gap_intensity = args.gap_px
        state.im_trans_opt = args.im_trans

        dataFN = args.data
        darkFN = args.dark
        dataLoc = args.data_loc
        DrawPlots = int(args.plots)
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

        # ---- Auto-detect file format ----
        convertFile = args.convert
        if convertFile < 0:
            convertFile = detect_format(dataFN)
            logger.info(f"Auto-detected format: {['Zarr','HDF5','GE','TIFF'][convertFile]}")

        # ---- File format conversion ----
        bad_gap_arr = []

        # Detect calibrant + filename hints early (needed for temp param file)
        calibrant = detect_calibrant(args.data)
        if calibrant is None:
            calibrant = CALIBRANTS['ceo2']
            logger.info(f"No calibrant detected from filename, defaulting to {calibrant['name']}")

        if convertFile == 3:
            logger.info("Processing TIFF input")
            dataFN, darkGeFN, ny, nz, bad_gap_arr = process_tiff_input(
                dataFN, args.bad_px, args.gap_px, darkFN)
            if ny > 0:
                state.nr_pixels_y = ny
            if nz > 0:
                state.nr_pixels_z = nz
            logger.info(f"Converted TIFF to GE: {dataFN}, NrPixelsY={state.nr_pixels_y}, "
                        f"NrPixelsZ={state.nr_pixels_z}")
            psFN = args.params
            if not psFN:
                psFN = _make_temp_param_file(args, calibrant, filename_hints)
            dataFN = generateZip('.', psFN, dfn=dataFN, darkfn=darkGeFN,
                                 nchunks=100, preproc=0,
                                 NrPixelsY=state.nr_pixels_y,
                                 NrPixelsZ=state.nr_pixels_z)

        if convertFile == 1 or convertFile == 2:
            psFN = args.params
            if not psFN:
                psFN = _make_temp_param_file(args, calibrant, filename_hints)
            logger.info("Generating zip file")
            dataFN = generateZip('.', psFN, dfn=dataFN, nchunks=100,
                                 preproc=0, darkfn=darkFN, dloc=dataLoc)

        # ---- Read Zarr file ----
        logger.info(f"Reading Zarr file: {dataFN}")
        dataF = zarr.open(dataFN, mode='r')
        dataFN = os.path.basename(dataFN)

        # calibrant already detected above

        # Extract parameters from Zarr (with calibrant defaults as fallback)
        state.skip_frame = 0
        ap = '/analysis/process/analysis_parameters'
        if f'{ap}/SpaceGroup' in dataF:
            state.space_group = dataF[f'{ap}/SpaceGroup'][0].item()
        else:
            state.space_group = calibrant['space_group']
            logger.info(f"SpaceGroup not in Zarr, using {calibrant['name']} default: {state.space_group}")

        if f'{ap}/SkipFrame' in dataF:
            state.skip_frame = dataF[f'{ap}/SkipFrame'][0].item()

        if f'{ap}/PixelSize' in dataF:
            state.px = dataF[f'{ap}/PixelSize'][0].item()
        elif args.px > 0:
            state.px = args.px
            logger.info(f"PixelSize from --px: {state.px} µm")
        else:
            state.px = 200.0
            logger.info(f"PixelSize not in Zarr, using default: {state.px} µm")

        if f'{ap}/LatticeParameter' in dataF:
            state.latc = dataF[f'{ap}/LatticeParameter'][:]
        else:
            state.latc = calibrant['lattice']
            logger.info(f"LatticeParameter not in Zarr, using {calibrant['name']} default: {state.latc}")

        if f'{ap}/Wavelength' in dataF:
            state.wavelength = dataF[f'{ap}/Wavelength'][:].item()
        elif 'wavelength' in filename_hints:
            state.wavelength = filename_hints['wavelength']
            logger.info(f"Wavelength from filename: {state.wavelength:.5f} Å")
        else:
            state.wavelength = 0.0
            logger.warning("Wavelength not found in Zarr or filename — must be set in param file")

        if f'{ap}/tx' in dataF:
            state.tx = dataF[f'{ap}/tx'][:].item()
        else:
            state.tx = 0.0
        # CLI --tx overrides Zarr/default
        if args.tx != 0.0:
            state.tx = args.tx
            logger.info(f"tx from --tx: {state.tx}")

        if f'{ap}/MaskFile' in dataF:
            mf = dataF[f'{ap}/MaskFile'][0]
            if isinstance(mf, bytes): mf = mf.decode()
            state.mask_file = str(mf)
        # CLI --mask overrides Zarr/default
        if args.mask:
            state.mask_file = str(Path(args.mask).absolute())
            logger.info(f"MaskFile from --mask: {state.mask_file}")

        # Read data and dark
        raw, ny, nz = fileReader(dataF, '/exchange/data', state.skip_frame)
        dark, _, _ = fileReader(dataF, '/exchange/dark', state.skip_frame)
        state.nr_pixels_y = ny
        state.nr_pixels_z = nz

        # Apply mask file: zero out masked pixels before any processing
        if state.mask_file:
            try:
                mask_img = np.array(Image.open(state.mask_file))
                # Convention: mask==0 means bad/masked pixel, mask!=0 means good pixel
                mask_bool = (mask_img == 0)  # True where pixel is BAD
                if mask_bool.shape != raw.shape:
                    logger.warning(f"Mask shape {mask_bool.shape} != image shape {raw.shape}, skipping mask")
                else:
                    raw[mask_bool] = 0
                    dark[mask_bool] = 0
                    # Merge with existing bad_gap_arr
                    if len(bad_gap_arr) != 0:
                        bad_gap_arr = np.logical_or(bad_gap_arr, mask_bool)
                    else:
                        bad_gap_arr = mask_bool
                    logger.info(f"Applied mask from {state.mask_file}: "
                                f"{np.sum(mask_bool)} pixels masked")
            except Exception as e:
                logger.error(f"Failed to read mask file {state.mask_file}: {e}")

        # Save as GE files (needed by CalibrantPanelShiftsOMP)
        rawFN = dataFN.split('.zip')[0] + '.ge5'
        darkFN_ge = 'dark_' + rawFN
        raw.tofile(rawFN)
        if len(bad_gap_arr) != 0:
            raw = np.ma.masked_array(raw, mask=bad_gap_arr)
        dark.tofile(darkFN_ge)
        state.dark_name = darkFN_ge

        # Apply image transformations
        for transOpt in state.im_trans_opt:
            if transOpt == 1:
                raw = np.fliplr(raw)
                dark = np.fliplr(dark)
            elif transOpt == 2:
                raw = np.flipud(raw)
                dark = np.flipud(dark)
            elif transOpt == 3:
                raw = np.transpose(raw)
                dark = np.transpose(dark)

        if state.h5_file:
            save_raw_image_data(raw, state.h5_file)

        if DrawPlots == 1:
            fig = plt.figure()
            plt.imshow(np.log(raw),
                       clim=[np.median(np.log(raw)),
                             np.median(np.log(raw)) + np.std(np.log(raw))],
                       origin='lower')
            plt.colorbar()
            plt.title('Raw image')
            plt.show()

        # ---- Ring simulation ----
        logger.info("Running initial ring simulation")
        sim_params = {
            'Wavelength': state.wavelength,
            'SpaceGroup': state.space_group,
            'Lsd': initialLsd,
            'MaxRingRad': mrr,
            'LatticeConstant': state.latc
        }
        create_param_file('ps_init_sim.txt', sim_params)
        hkls = run_get_hkl_list('ps_init_sim.txt')
        sim_rads = np.unique(hkls[:, -1]) / state.px
        sim_rad_ratios = sim_rads / sim_rads[0]

        # ---- Background subtraction and thresholding ----
        data = raw.astype(np.float64)

        # Determine DataType for CalibrantPanelShiftsOMP
        if raw.dtype == np.uint32:
            state.midas_dtype = 4
        elif raw.dtype == np.int32:
            state.midas_dtype = 5
        elif raw.dtype == np.float32:
            state.midas_dtype = 3
        elif raw.dtype == np.float64:
            state.midas_dtype = 2
        else:
            state.midas_dtype = 1  # default Uint16

        if noMedian == 0:
            logger.info("Applying median filter for background estimation")
            data2 = dip.MedianFilter(data, 101)
            for _ in range(4):
                data2 = dip.MedianFilter(data2, 101)
        else:
            logger.info("Skipping median filter, using dark subtraction only")
            data2 = dark.astype(np.float64)

        logger.info('Finished with median, now processing data.')
        data = data.astype(float)

        if state.h5_file:
            save_background_data(data2, state.h5_file)

        if DrawPlots == 1 and noMedian == 0:
            fig = plt.figure()
            plt.imshow(np.log(data2), origin='lower')
            plt.colorbar()
            plt.title('Computed background')
            plt.show()

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

        if DrawPlots == 1 and noMedian == 0:
            fig = plt.figure()
            plt.imshow(thresh, origin='lower')
            plt.colorbar()
            plt.title('Cleaned image')
            plt.show()

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

        # Re-run ring simulation with updated Lsd
        sim_params['Lsd'] = initialLsd
        create_param_file('ps.txt', sim_params)
        hkls = run_get_hkl_list('ps.txt')
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
        if DrawPlots == 1:
            fig, ax = plt.subplots()
            if noMedian == 0:
                plt.imshow(thresh, origin='lower')
            else:
                log_img = np.log(data_corr + 1)
                vmin = np.median(log_img)
                vmax = vmin + np.std(log_img)
                plt.imshow(log_img, clim=[vmin, vmax], origin='lower')
                plt.colorbar()
            for ringNr, rad in enumerate(sim_rads, start=1):
                if ringNr in state.rings_to_exclude:
                    continue
                e1 = mpatches.Arc((bc_new[1], bc_new[0]), 2*rad, 2*rad,
                                  angle=0, theta1=-180, theta2=180, color='blue')
                ax.add_patch(e1)
            ax.axis([0, NrPixelsY, 0, NrPixelsZ])
            ax.set_aspect('equal')
            plt.title('Overlaid rings')
            plt.show()

        # ---- Prepare for MIDAS calibration ----
        state.fnumber = int(rawFN.split('_')[-1].split('.')[0])
        state.pad = len(rawFN.split('_')[-1].split('.')[0])
        state.fstem = os.path.basename('_'.join(rawFN.split('_')[:-1]))
        state.ext = '.' + '.'.join(rawFN.split('_')[-1].split('.')[1:])
        state.folder = os.path.dirname(rawFN)
        if not state.folder:
            state.folder = os.getcwd()

        state.lsd = str(initialLsd)
        state.bc = f"{bc_new[1]} {bc_new[0]}"
        state.panel_shifts_file = f"{state.fstem}_panel_shifts.txt"
        state.nr_pixels_y = NrPixelsY
        state.nr_pixels_z = NrPixelsZ

        # Calculate RhoD — maximum radius to edge from beam center
        edges = np.array([[0, 0], [NrPixelsY, 0], [NrPixelsY, NrPixelsZ], [0, NrPixelsZ]])
        state.rhod = np.max(np.linalg.norm(
            np.transpose(edges) - bc_new[:, None], axis=0)) * state.px

        # ---- Single call to CalibrantPanelShiftsOMP ----
        logger.info(f"Running MIDAS calibration: {args.n_iterations} iterations, "
                     f"{n_cpus} CPUs, DoubletSep={args.doublet_separation}px")
        print(f"\n{'='*60}")
        print(f"  CalibrantPanelShiftsOMP: {args.n_iterations} iterations, "
              f"{n_cpus} CPUs")
        print(f"  DoubletSeparation={args.doublet_separation}px, "
              f"MultFactor={multFactor}")
        print(f"{'='*60}")

        runMIDAS(rawFN, state,
                 n_iterations=args.n_iterations,
                 mult_factor=multFactor,
                 doublet_separation=args.doublet_separation,
                 outlier_iterations=args.outlier_iterations,
                 eta_bin_size=args.eta_bin_size,
                 max_width=maxW,
                 n_cpus=n_cpus)

        # ---- Generate final results ----
        logger.info("Generating final results data")
        corr_file = f"{rawFN}.corr.csv"
        if os.path.exists(corr_file):
            df = pd.read_csv(corr_file, delimiter=' ')

            if DrawPlots == 1:
                fig = make_subplots(rows=1, cols=2,
                                   specs=[[{"type": "scatter"}, {"type": "scatterpolar"}]])
                fig.add_trace(
                    go.Scatter(mode='markers', x=df['RadFit'], y=df['Strain'],
                               marker=dict(color=df['Ideal2Theta']), showlegend=True),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatterpolar(r=df['Strain'], theta=df['EtaCalc'],
                                    mode='markers',
                                    marker=dict(color=df['Ideal2Theta']),
                                    showlegend=True),
                    row=1, col=2
                )
                html_file = f"{rawFN}.html"
                fig.write_html(html_file)
                logger.info(f"Interactive plots written to: {html_file}")

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

        # ---- Write final parameter file ----
        psName = 'refined_MIDAS_params.txt'
        logger.info(f"Writing final parameters to {psName}")

        final_params = {
            'Lsd': state.lsd, 'BC': state.bc,
            'tx': state.tx, 'ty': state.ty, 'tz': state.tz,
            'p0': state.p0, 'p1': state.p1, 'p2': state.p2, 'p3': state.p3,
            'RhoD': state.rhod, 'Wavelength': state.wavelength, 'px': state.px,
            'RMin': 10, 'RMax': 1000, 'RBinSize': 1,
            'EtaMin': -180, 'EtaMax': 180,
            'NrPixelsY': NrPixelsY, 'NrPixelsZ': NrPixelsZ,
            'EtaBinSize': 5, 'DoSmoothing': 1, 'DoPeakFit': 1,
            'skipFrame': state.skip_frame,
            'MultiplePeaks': 1, 'DataType': state.midas_dtype
        }

        if state.p4 != '0':
            final_params['p4'] = state.p4

        if state.mask_file:
            final_params['MaskFile'] = state.mask_file

        with open(psName, 'w') as pf:
            for key, value in final_params.items():
                pf.write(f"{key} {value}\n")

            if state.panel_params:
                for pp in state.panel_params:
                    pf.write(f'{pp}\n')
                pf.write(f'PanelShiftsFile {state.panel_shifts_file}\n')
                pf.write('tolShifts 1\n')

            for transOpt in state.im_trans_opt:
                pf.write(f"ImTransOpt {transOpt}\n")

            if not math.isnan(state.bad_px_intensity):
                pf.write(f"BadPxIntensity {state.bad_px_intensity}\n")
            if not math.isnan(state.gap_intensity):
                pf.write(f"GapIntensity {state.gap_intensity}\n")

        logger.info("Calibration completed successfully")
        print(f"\n  Output written to: {psName}")

    except Exception as e:
        logger.error(f"Error in main function: {traceback.format_exc()}")
        if hasattr(state, 'h5_file') and state.h5_file:
            state.h5_file.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
