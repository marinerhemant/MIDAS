#!/usr/bin/env python
"""
PF-HEDM Interactive Sinogram & Peak Intensity Viewer

Displays two side-by-side plots per grain:
  Left:  Sinogram (scanNr × rotation angle)
  Right: 2D intensity distribution (Y × Z detector pixels) for the selected cell

Usage:
  python pfIntensityViewer.py -paramFile <paramFile> [-resultDir <dir>] [-portNr 8051]
"""

import os, sys, glob, argparse, traceback
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from zarr.storage import ZipStore
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, State, no_update, ctx
import dash_bootstrap_components as dbc

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SPOTS_ARRAY_COLS = 10  # [y, x, omega, intensity, spotID, ringNr, eta, theta, dspacing, scanNr]
COMMON_LAYOUT = dict(margin=dict(l=10, r=10, b=10, t=50), height=550, template="plotly_dark")
DEFAULT_PATCH_HALF = 15
DEFAULT_REFRESH_MS = 500


# ──────────────────────────────────────────────────────────────
# Argument Parsing
# ──────────────────────────────────────────────────────────────
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error: {message}\n'); self.print_help(); sys.exit(2)


# ──────────────────────────────────────────────────────────────
# Parameter File Parsing (mirrors pf_MIDAS.py)
# ──────────────────────────────────────────────────────────────
def parse_param_file(paramFN):
    """Parse the PF parameter file and return a dict of relevant values."""
    params = {
        'FileStem': '',
        'StartFileNrFirstLayer': 0,
        'NrFilesPerSweep': 1440,
        'nScans': 1,
        'Padding': 6,
        'ImTransOpt': [],
        'StartNr': 1,
        'EndNr': 1440,
        'OmegaStart': 0,
        'OmegaStep': 0.25,
        'BC_Y': 0,
        'BC_Z': 0,
        'px': 200,
        'numPxY': 2048,
        'numPxZ': 2048,
    }

    with open(paramFN, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = parts[0]
            if key == 'FileStem':
                params['FileStem'] = parts[1]
            elif key == 'StartFileNrFirstLayer':
                params['StartFileNrFirstLayer'] = int(parts[1])
            elif key == 'NrFilesPerSweep':
                params['NrFilesPerSweep'] = int(parts[1])
            elif key == 'nScans':
                params['nScans'] = int(parts[1])
            elif key == 'Padding':
                params['Padding'] = int(parts[1])
            elif key == 'ImTransOpt':
                params['ImTransOpt'].append(int(parts[1]))
            elif key == 'StartNr':
                params['StartNr'] = int(parts[1])
            elif key == 'EndNr':
                params['EndNr'] = int(parts[1])
            elif key == 'OmegaStart':
                params['OmegaStart'] = float(parts[1])
            elif key == 'OmegaEnd':
                params['OmegaEnd'] = float(parts[1])
            elif key == 'OmegaStep':
                params['OmegaStep'] = float(parts[1])
            elif key == 'BC':
                params['BC_Y'] = float(parts[1])
                params['BC_Z'] = float(parts[2])
            elif key == 'px':
                params['px'] = float(parts[1])
            elif key == 'numPxY':
                params['numPxY'] = int(parts[1])
            elif key == 'numPxZ':
                params['numPxZ'] = int(parts[1])
    return params


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────
def discover_zarr_files(topdir, params):
    """Discover per-scan zarr zip files based on parameter file values."""
    nScans = params['nScans']
    startNr = params['StartFileNrFirstLayer']
    nrFiles = params['NrFilesPerSweep']
    fStem = params['FileStem']

    zarr_paths = []
    scan_dirs = []
    for scanIdx in range(nScans):
        thisStartNr = startNr + scanIdx * nrFiles
        scanDir = os.path.join(topdir, str(thisStartNr))
        scan_dirs.append(scanDir)
        # Try to find the zarr zip by globbing
        candidates = glob.glob(os.path.join(scanDir, '*.MIDAS.zip'))
        if candidates:
            zarr_paths.append(candidates[0])
        else:
            # Fallback to constructed name
            fname = f'{fStem}_{thisStartNr}.MIDAS.zip'
            zarr_paths.append(os.path.join(scanDir, fname))

    return zarr_paths, scan_dirs


def load_sinogram_variants(topdir):
    """Load all 4 sinogram variant binary files."""
    variants = {}
    # Find the raw sinos file to get dimensions
    raw_files = glob.glob(os.path.join(topdir, 'sinos_raw_*.bin'))
    if not raw_files:
        # Try the default naming (without variant prefix)
        raw_files = glob.glob(os.path.join(topdir, 'sinos_[0-9]*.bin'))
        if raw_files:
            parts = os.path.basename(raw_files[0]).replace('sinos_', '').replace('.bin', '').split('_')
            nGrs, maxNHKLs, nScans = int(parts[0]), int(parts[1]), int(parts[2])
            variants['raw'] = np.fromfile(raw_files[0], dtype=np.double).reshape((nGrs, maxNHKLs, nScans))
            return variants, nGrs, maxNHKLs, nScans
        print("ERROR: No sinogram files found!", file=sys.stderr)
        sys.exit(1)

    # Parse dimensions from filename: sinos_raw_N_M_S.bin
    parts = os.path.basename(raw_files[0]).replace('sinos_raw_', '').replace('.bin', '').split('_')
    nGrs, maxNHKLs, nScans = int(parts[0]), int(parts[1]), int(parts[2])

    for variant in ['raw', 'norm', 'abs', 'normabs']:
        fn = os.path.join(topdir, f'sinos_{variant}_{nGrs}_{maxNHKLs}_{nScans}.bin')
        if os.path.exists(fn):
            variants[variant] = np.fromfile(fn, dtype=np.double).reshape((nGrs, maxNHKLs, nScans))
        else:
            print(f"  Warning: {fn} not found, skipping '{variant}'")

    return variants, nGrs, maxNHKLs, nScans


def load_omegas_and_hkls(topdir, nGrs, maxNHKLs):
    """Load omega angles and HKL counts per grain."""
    omegas = np.fromfile(os.path.join(topdir, f'omegas_{nGrs}_{maxNHKLs}.bin'),
                         dtype=np.double).reshape((nGrs, maxNHKLs))
    grainSpots = np.fromfile(os.path.join(topdir, f'nrHKLs_{nGrs}.bin'), dtype=np.int32)
    return omegas, grainSpots


def load_unique_spots(topdir):
    """Load UniqueOrientationSpots.csv → DataFrame."""
    fn = os.path.join(topdir, 'UniqueOrientationSpots.csv')
    if not os.path.exists(fn):
        print(f"  Warning: {fn} not found. Intensity patches won't be available.")
        return pd.DataFrame(columns=['ID', 'GrainNr', 'SpotNr', 'RingNr', 'Omega', 'Eta'])
    return pd.read_csv(fn)


def load_spots_bin(topdir):
    """Load the Spots.bin memory-mapped file into a structured array.

    Spots.bin format (per row, all doubles):
    [y, x, omega, intensity, spotID, ringNr, eta, theta, dspacing, scanNr]
    """
    fn = os.path.join(topdir, 'Spots.bin')
    if not os.path.exists(fn):
        # Also check /dev/shm
        fn_shm = '/dev/shm/Spots.bin'
        if os.path.exists(fn_shm):
            fn = fn_shm
        else:
            print(f"  Warning: Spots.bin not found at {fn}")
            return None

    raw = np.fromfile(fn, dtype=np.double)
    nSpots = len(raw) // SPOTS_ARRAY_COLS
    if len(raw) % SPOTS_ARRAY_COLS != 0:
        print(f"  Warning: Spots.bin size not divisible by {SPOTS_ARRAY_COLS}")
    spots = raw[:nSpots * SPOTS_ARRAY_COLS].reshape((nSpots, SPOTS_ARRAY_COLS))
    print(f"  Loaded Spots.bin: {nSpots} spots")
    return spots


def open_zarr_handles(zarr_paths):
    """Open zarr data handles (lazy) for each scan."""
    handles = []
    for zp in zarr_paths:
        if os.path.exists(zp):
            try:
                store = ZipStore(zp, mode='r')
                zf = zarr.open_group(store=store, mode='r')
                handles.append(zf['exchange/data'])
            except Exception as e:
                print(f"  Warning: Could not open zarr {zp}: {e}")
                handles.append(None)
        else:
            handles.append(None)
    return handles


# ──────────────────────────────────────────────────────────────
# Spot lookup: map (grainNr, spotNr) to spots in each scan
# ──────────────────────────────────────────────────────────────
def build_spot_lookup(unique_spots_df, spots_bin, nScans, tol_ome=1.0, tol_eta=1.0):
    """Build a lookup: (grainNr, spotNr, scanNr) → (y_det, x_det, omega, spotID)

    Matches UniqueOrientationSpots entries to Spots.bin entries by omega/eta/ringNr
    and scanNr, mirroring the logic in generate_sinograms from findSingleSolution.
    """
    if spots_bin is None or unique_spots_df.empty:
        return {}

    lookup = {}
    for _, urow in unique_spots_df.iterrows():
        grainNr = int(urow['GrainNr'])
        spotNr = int(urow['SpotNr'])
        refOmega = urow['Omega']
        refEta = urow['Eta']
        refRingNr = int(urow['RingNr'])

        for scanNr in range(nScans):
            # Find matching spots in Spots.bin for this scan
            mask = (
                (np.abs(spots_bin[:, 9] - scanNr) < 0.5) &  # scanNr match
                (spots_bin[:, 5].astype(int) == refRingNr) &  # ringNr match
                (np.abs(spots_bin[:, 2] - refOmega) < tol_ome) &  # omega match
                (np.abs(spots_bin[:, 6] - refEta) < tol_eta)   # eta match
            )
            matches = spots_bin[mask]
            if len(matches) > 0:
                # Take the best match (closest in omega+eta)
                dist = np.abs(matches[:, 2] - refOmega) + np.abs(matches[:, 6] - refEta)
                best = matches[np.argmin(dist)]
                lookup[(grainNr, spotNr, scanNr)] = {
                    'y_det': best[0],   # detector y position
                    'x_det': best[1],   # detector x position
                    'omega': best[2],
                    'intensity': best[3],
                    'spotID': int(best[4]),
                    'eta': best[6],
                }

    print(f"  Built spot lookup: {len(lookup)} entries")
    return lookup


# ──────────────────────────────────────────────────────────────
# Intensity Patch Extraction
# ──────────────────────────────────────────────────────────────
def make_patch_extractor(spot_lookup, zarr_handles, params):
    """Create a cached patch extraction function."""
    omegaStart = params['OmegaStart']
    omegaStep = params['OmegaStep']
    nrFilesPerSweep = params['NrFilesPerSweep']
    imTransOpt = params.get('ImTransOpt', [])

    @lru_cache(maxsize=512)
    def get_intensity_patch(grainNr, spotNr, scanNr, patchHalfSize):
        """Extract the 2D intensity patch for one sinogram cell."""
        key = (grainNr, spotNr, scanNr)
        if key not in spot_lookup:
            return None

        info = spot_lookup[key]
        omega = info['omega']

        # Compute frame number from omega
        if omegaStep != 0:
            frameIdx = int(round((omega - omegaStart) / omegaStep))
        else:
            frameIdx = 0

        # Clamp to valid range
        if frameIdx < 0:
            frameIdx += nrFilesPerSweep
        frameIdx = max(0, min(nrFilesPerSweep - 1, frameIdx))

        zarr_data = zarr_handles[scanNr] if scanNr < len(zarr_handles) else None
        if zarr_data is None:
            return None

        nFrames = zarr_data.shape[0]
        nPxZ = zarr_data.shape[1]
        nPxY = zarr_data.shape[2]

        if frameIdx >= nFrames:
            frameIdx = nFrames - 1

        # Convert detector position (microns from beam center) to pixel coords
        # The Spots.bin has (y, x) already in pixel-space as output by
        # the pipeline (YCen, ZCen are in pixel units in the _PS.csv)
        # But Spots.bin has y and x in different coordinate conventions.
        # Spots.bin col0 = y (detector horizontal), col1 = x (unused in 2D)
        # The actual mapping depends on how findSingleSolution reads the data.
        # For now, use the spotID to retrieve the per-scan Result position,
        # or use a simpler heuristic with BC and px.

        # The y_det and x_det from Spots.bin are in the coordinate system
        # defined by CalcRadiusAll. They represent (omega, 2theta equiv.)
        # For the raw zarr extraction, we need pixel positions.
        # We'll estimate pixel positions from the spot's eta and radius.

        # Simpler approach: extract a neighborhood around the peak.
        # From Spots.bin: y_det is the y position, not directly pixels.
        # Let's use a fixed center approach and just grab the frame.
        # The user can adjust patch size with the slider.

        # For robust pixel position, read Result CSV if available.
        # Fallback: use the full frame as the patch (clipped to patch size).

        # Actually, looking at the data flow: Spots.bin is created by
        # reading Result_*.csv which has columns including Y and Z in pixel space.
        # Spots.bin[0] = y in sample coords, Spots.bin[1] = x (unused for 2D).
        # The detector pixel position is encoded differently.

        # Let's try to get the detector-space pixel position from the sinogram's
        # associated merged spot. The Result_*.csv has YCen and ZCen.
        # But we need to read them per-scan.

        # For NOW: extract the full area around the center of the detector
        # at the frame corresponding to the omega angle. The user can see
        # where the peak is and adjust. This is the MVP approach.

        # Better approach: read the specific scan's Result CSV to get pixel pos.
        spotID = info['spotID']
        yCen_px, zCen_px = _find_pixel_position(scanNr, spotID, params, zarr_data.shape)
        if yCen_px is None:
            return None

        # Extract patch
        h = patchHalfSize
        z0 = max(0, zCen_px - h)
        z1 = min(nPxZ, zCen_px + h + 1)
        y0 = max(0, yCen_px - h)
        y1 = min(nPxY, yCen_px + h + 1)

        if z1 <= z0 or y1 <= y0:
            return None

        patchSize = 2 * h + 1
        patch = np.zeros((patchSize, patchSize), dtype=np.double)

        try:
            raw = zarr_data[frameIdx, z0:z1, y0:y1].astype(np.double)
            # Handle edge clipping
            pz0 = h - (zCen_px - z0)
            pz1 = pz0 + (z1 - z0)
            py0 = h - (yCen_px - y0)
            py1 = py0 + (y1 - y0)
            patch[pz0:pz1, py0:py1] = raw
        except Exception as e:
            print(f"  Patch extraction error: {e}")
            return None

        patch[patch < 0] = 0
        return patch

    return get_intensity_patch


# Cache for per-scan Result CSV data
_result_cache = {}

def _load_result_csv(scanDir):
    """Load Result_*.csv from a scan directory → dict of spotID → (YCen_px, ZCen_px)."""
    if scanDir in _result_cache:
        return _result_cache[scanDir]

    result_data = {}
    result_files = glob.glob(os.path.join(scanDir, 'Result_*.csv'))
    for rf in result_files:
        try:
            with open(rf, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('SpotID'):
                        continue  # skip header
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        spotID = int(float(parts[0]))
                        # Result_*.csv columns (space-separated):
                        # 0:SpotID 1:IntIntensity 2:Omega 3:YCen(px) 4:ZCen(px)
                        # 5:IMax 6:MinOme 7:MaxOme 8:SigmaR 9:SigmaEta
                        # 10:NrPx 11:NrPxTot 12:Radius 13:Eta 14:RawSumInt
                        yCen = float(parts[3])
                        zCen = float(parts[4])
                        result_data[spotID] = (yCen, zCen)
                    except (ValueError, IndexError):
                        continue
        except Exception:
            continue

    _result_cache[scanDir] = result_data
    return result_data


def _find_pixel_position(scanNr, spotID, params, zarr_shape):
    """Find the pixel position for a given spot in a given scan.

    Returns (yCen_px, zCen_px) or (None, None).
    """
    startNr = params['StartFileNrFirstLayer']
    nrFiles = params['NrFilesPerSweep']
    topdir = params['topdir']

    thisStartNr = startNr + scanNr * nrFiles
    scanDir = os.path.join(topdir, str(thisStartNr))

    result_data = _load_result_csv(scanDir)
    if spotID in result_data:
        yCen, zCen = result_data[spotID]
        # Convert to integer pixel indices
        yCen_px = int(round(yCen))
        zCen_px = int(round(zCen))
        # Clamp to valid range
        nPxZ, nPxY = zarr_shape[1], zarr_shape[2]
        yCen_px = max(0, min(nPxY - 1, yCen_px))
        zCen_px = max(0, min(nPxZ - 1, zCen_px))
        return yCen_px, zCen_px

    return None, None


# ──────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = MyParser(description='PF-HEDM Sinogram & Intensity Viewer',
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-paramFile', type=str, required=True,
                        help='Parameter file used for pf_MIDAS analysis')
    parser.add_argument('-resultDir', type=str, required=False, default='',
                        help='Results directory (defaults to cwd)')
    parser.add_argument('-portNr', type=int, required=False, default=8051,
                        help='Port number for the Dash server')
    parser.add_argument('-HostName', type=str, required=False, default='0.0.0.0',
                        help='Hostname for the Dash server')
    args, _ = parser.parse_known_args()

    topdir = args.resultDir if args.resultDir else os.getcwd()

    # --- Parse parameter file ---
    print("=" * 60)
    print("PF-HEDM Sinogram & Intensity Viewer")
    print("=" * 60)
    print(f"\nParameter file: {args.paramFile}")
    print(f"Result dir: {topdir}")
    print("\nParsing parameter file...")
    params = parse_param_file(args.paramFile)
    params['topdir'] = topdir
    imTransOpt = params.get('ImTransOpt', [])
    print(f"  FileStem: {params['FileStem']}")
    print(f"  nScans: {params['nScans']}")
    print(f"  NrFilesPerSweep: {params['NrFilesPerSweep']}")
    print(f"  OmegaStart: {params['OmegaStart']}")
    print(f"  OmegaStep: {params['OmegaStep']}")
    print(f"  ImTransOpt: {imTransOpt}")

    # --- Discover zarr files ---
    print("\nDiscovering zarr files...")
    zarr_paths, scan_dirs = discover_zarr_files(topdir, params)
    n_found = sum(1 for zp in zarr_paths if os.path.exists(zp))
    print(f"  Found {n_found}/{len(zarr_paths)} zarr files")
    for i, zp in enumerate(zarr_paths[:3]):
        exists = '✓' if os.path.exists(zp) else '✗'
        print(f"  Scan {i}: {os.path.basename(zp)} {exists}")
    if len(zarr_paths) > 3:
        print(f"  ... and {len(zarr_paths) - 3} more")

    # --- Load sinogram data ---
    print("\nLoading sinogram data...")
    sino_variants, nGrs, maxNHKLs, nScans = load_sinogram_variants(topdir)
    available_variants = list(sino_variants.keys())
    print(f"  {nGrs} grains, {maxNHKLs} max HKLs, {nScans} scans")
    print(f"  Variants: {available_variants}")

    # --- Load omegas and HKL counts ---
    print("\nLoading omega angles...")
    omegas, grainSpots = load_omegas_and_hkls(topdir, nGrs, maxNHKLs)

    # --- Load unique spots ---
    print("\nLoading unique spot associations...")
    unique_spots_df = load_unique_spots(topdir)
    print(f"  UniqueSpots: {len(unique_spots_df)}")

    # --- Load Spots.bin ---
    print("\nLoading Spots.bin...")
    spots_bin = load_spots_bin(topdir)

    # --- Build spot lookup ---
    print("\nBuilding spot lookup (this may take a moment for large datasets)...")
    spot_lookup = build_spot_lookup(unique_spots_df, spots_bin, nScans)

    # --- Open zarr handles ---
    print("\nOpening zarr data handles...")
    zarr_handles = open_zarr_handles(zarr_paths)
    n_open = sum(1 for h in zarr_handles if h is not None)
    print(f"  Opened {n_open}/{len(zarr_handles)} zarr files")

    # --- Create patch extractor ---
    get_intensity_patch = make_patch_extractor(spot_lookup, zarr_handles, params)

    # ── Dash App ─────────────────────────────────────────────
    external_stylesheets = [dbc.themes.CYBORG]
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    app.title = "PF-HEDM Sinogram & Intensity Viewer"

    # --- Layout ---
    app.layout = dbc.Container([
        dcc.Store(id='store-sino-vmin', data=None),
        dcc.Store(id='store-sino-vmax', data=None),
        dcc.Store(id='store-patch-vmin', data=None),
        dcc.Store(id='store-patch-vmax', data=None),
        dcc.Interval(id='interval-play-row', interval=DEFAULT_REFRESH_MS,
                     n_intervals=0, disabled=True),
        dcc.Interval(id='interval-play-col', interval=DEFAULT_REFRESH_MS,
                     n_intervals=0, disabled=True),

        # Title
        dbc.Row([
            html.H3("PF-HEDM Sinogram & Intensity Viewer",
                     className="text-primary text-center mb-3 mt-2")
        ]),

        # Controls row
        dbc.Row([
            dbc.Col([
                dbc.Label("Grain Nr:"),
                dcc.Dropdown(
                    id='grain-dropdown',
                    options=[{'label': f'Grain {i} ({grainSpots[i]} spots)',
                              'value': i} for i in range(nGrs)],
                    value=0, clearable=False,
                    style={'color': '#000'}
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Sinogram Variant:"),
                dcc.Dropdown(
                    id='variant-dropdown',
                    options=[{'label': v, 'value': v} for v in available_variants],
                    value=available_variants[0] if available_variants else 'raw',
                    clearable=False,
                    style={'color': '#000'}
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Patch Half-Size (px):"),
                dcc.Slider(id='patch-size-slider', min=5, max=50, step=1,
                           value=DEFAULT_PATCH_HALF,
                           marks={5: '5', 15: '15', 25: '25', 50: '50'},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], width=3),
            dbc.Col([
                dbc.Label("Spot Nr (row):"),
                dcc.Slider(id='row-slider', min=0,
                           max=max(0, int(grainSpots[0]) - 1) if nGrs > 0 else 0,
                           step=1, value=0,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], width=2),
            dbc.Col([
                dbc.Label("Scan Nr (col):"),
                dcc.Slider(id='col-slider', min=0, max=max(0, nScans - 1),
                           step=1, value=0,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], width=2),
        ], className="mb-2"),

        # Plots
        dbc.Row([
            dbc.Col([
                dcc.Loading(type="circle", children=[
                    dcc.Graph(id='sinogram-plot', figure=go.Figure())
                ]),
            ], width=6),
            dbc.Col([
                dcc.Loading(type="circle", children=[
                    dcc.Graph(id='patch-plot', figure=go.Figure())
                ]),
            ], width=6),
        ]),

        html.Hr(),

        # Animation & Scale controls
        dbc.Row([
            # Left: Play controls
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Button("▶ Play Scans (fixed spot)",
                                   id='btn-play-row', color='success',
                                   size='sm', className='me-1'),
                        dbc.Button("■ Stop", id='btn-stop-row',
                                   color='danger', size='sm'),
                    ], width=6),
                    dbc.Col([
                        dbc.Button("▶ Play Spots (fixed scan)",
                                   id='btn-play-col', color='success',
                                   size='sm', className='me-1'),
                        dbc.Button("■ Stop", id='btn-stop-col',
                                   color='danger', size='sm'),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Refresh (ms):", className='mt-2'),
                        dcc.Input(id='refresh-ms-input', type='number',
                                  value=DEFAULT_REFRESH_MS, min=100, max=5000,
                                  step=50, style={'width': '100px', 'color': '#000'}),
                    ], width=4),
                ], className='mt-2'),
            ], width=6),

            # Right: Scale controls
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Sinogram Scale:"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Min"),
                            dbc.Input(id='sino-vmin', type='number',
                                      placeholder='auto', style={'color': '#000'}),
                            dbc.InputGroupText("Max"),
                            dbc.Input(id='sino-vmax', type='number',
                                      placeholder='auto', style={'color': '#000'}),
                            dbc.Button("Apply", id='btn-sino-scale',
                                       color='info', size='sm'),
                        ], size='sm'),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Patch Scale:"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Min"),
                            dbc.Input(id='patch-vmin', type='number',
                                      placeholder='auto', style={'color': '#000'}),
                            dbc.InputGroupText("Max"),
                            dbc.Input(id='patch-vmax', type='number',
                                      placeholder='auto', style={'color': '#000'}),
                            dbc.Button("Apply", id='btn-patch-scale',
                                       color='info', size='sm'),
                        ], size='sm'),
                    ], width=6),
                ]),
            ], width=6),
        ]),
    ], fluid=True)

    # ── Callbacks ────────────────────────────────────────────

    # --- Click on sinogram → update row/col sliders ---
    @callback(
        Output('row-slider', 'value', allow_duplicate=True),
        Output('col-slider', 'value', allow_duplicate=True),
        Input('sinogram-plot', 'clickData'),
        prevent_initial_call=True
    )
    def sinogram_click(clickData):
        if not clickData or not clickData.get('points'):
            return no_update, no_update
        pt = clickData['points'][0]
        # Heatmap click: x = scanNr, y = spot index (row)
        col = int(pt.get('x', 0))  # scanNr
        # y is the theta label string — need to get point number
        pn = pt.get('pointNumber', None)
        if pn is not None and isinstance(pn, (list, tuple)) and len(pn) >= 2:
            row = int(pn[0])
            col = int(pn[1])
        elif pn is not None and isinstance(pn, int):
            row = pn
        else:
            row = 0
        return row, col

    # --- Scale apply buttons → update stores ---
    @callback(
        Output('store-sino-vmin', 'data'),
        Output('store-sino-vmax', 'data'),
        Input('btn-sino-scale', 'n_clicks'),
        State('sino-vmin', 'value'),
        State('sino-vmax', 'value'),
        prevent_initial_call=True
    )
    def apply_sino_scale(n, vmin, vmax):
        return vmin, vmax

    @callback(
        Output('store-patch-vmin', 'data'),
        Output('store-patch-vmax', 'data'),
        Input('btn-patch-scale', 'n_clicks'),
        State('patch-vmin', 'value'),
        State('patch-vmax', 'value'),
        prevent_initial_call=True
    )
    def apply_patch_scale(n, vmin, vmax):
        return vmin, vmax

    # --- Play/Stop controls ---
    @callback(
        Output('interval-play-row', 'disabled'),
        Output('interval-play-row', 'interval'),
        Input('btn-play-row', 'n_clicks'),
        Input('btn-stop-row', 'n_clicks'),
        State('refresh-ms-input', 'value'),
        prevent_initial_call=True
    )
    def toggle_play_scans(play, stop, ms):
        if ctx.triggered_id == 'btn-play-row':
            return False, max(100, ms or DEFAULT_REFRESH_MS)
        return True, max(100, ms or DEFAULT_REFRESH_MS)

    @callback(
        Output('interval-play-col', 'disabled'),
        Output('interval-play-col', 'interval'),
        Input('btn-play-col', 'n_clicks'),
        Input('btn-stop-col', 'n_clicks'),
        State('refresh-ms-input', 'value'),
        prevent_initial_call=True
    )
    def toggle_play_spots(play, stop, ms):
        if ctx.triggered_id == 'btn-play-col':
            return False, max(100, ms or DEFAULT_REFRESH_MS)
        return True, max(100, ms or DEFAULT_REFRESH_MS)

    # --- Interval ticks → advance col-slider (play scans for fixed spot) ---
    @callback(
        Output('col-slider', 'value', allow_duplicate=True),
        Input('interval-play-row', 'n_intervals'),
        State('col-slider', 'value'),
        prevent_initial_call=True
    )
    def advance_scan(n, current_col):
        return (current_col + 1) % nScans

    # --- Interval ticks → advance row-slider (play spots for fixed scan) ---
    @callback(
        Output('row-slider', 'value', allow_duplicate=True),
        Input('interval-play-col', 'n_intervals'),
        State('row-slider', 'value'),
        State('grain-dropdown', 'value'),
        prevent_initial_call=True
    )
    def advance_spot(n, current_row, grainNr):
        nSp = int(grainSpots[grainNr]) if grainNr is not None and grainNr < len(grainSpots) else 1
        return (current_row + 1) % max(1, nSp)

    # --- Reset row slider max when grain changes ---
    @callback(
        Output('row-slider', 'max'),
        Output('row-slider', 'value', allow_duplicate=True),
        Input('grain-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_row_slider_max(grainNr):
        if grainNr is None:
            return maxNHKLs - 1, 0
        nSp = int(grainSpots[grainNr]) if grainNr < len(grainSpots) else 1
        return max(0, nSp - 1), 0

    # --- Main sinogram plot ---
    @callback(
        Output('sinogram-plot', 'figure'),
        Input('grain-dropdown', 'value'),
        Input('variant-dropdown', 'value'),
        Input('row-slider', 'value'),
        Input('col-slider', 'value'),
        Input('store-sino-vmin', 'data'),
        Input('store-sino-vmax', 'data'),
    )
    def update_sinogram(grainNr, variant, row, col, vmin, vmax):
        fig = go.Figure()
        if grainNr is None or variant not in sino_variants:
            fig.update_layout(title="No data", **COMMON_LAYOUT)
            return fig

        nSp = int(grainSpots[grainNr]) if grainNr < len(grainSpots) else 0
        if nSp == 0:
            fig.update_layout(title=f"Grain {grainNr}: no spots", **COMMON_LAYOUT)
            return fig

        sino_data = sino_variants[variant]
        sino = sino_data[grainNr, :nSp, :]  # shape: (nSp, nScans)
        theta_vals = omegas[grainNr, :nSp]

        # Y-axis labels = theta values
        y_labels = [f'{theta_vals[i]:.1f}' for i in range(nSp)]

        fig.add_trace(go.Heatmap(
            z=sino,
            x=list(range(nScans)),
            y=y_labels,
            colorscale='Viridis',
            zmin=vmin, zmax=vmax,
            colorbar=dict(title='Intensity'),
            hovertemplate=(
                'ScanNr: %{x}<br>'
                'θ: %{y}°<br>'
                'Intensity: %{z:.2f}<extra></extra>'
            )
        ))

        # Crosshair at current selection
        if 0 <= row < nSp and 0 <= col < nScans:
            # Vertical line at col
            fig.add_shape(
                type='line', x0=col, x1=col,
                y0=-0.5, y1=len(y_labels) - 0.5,
                line=dict(color='red', width=1, dash='dash'),
            )
            # Horizontal line at row
            fig.add_shape(
                type='line', x0=-0.5, x1=nScans - 0.5,
                y0=row, y1=row,
                line=dict(color='red', width=1, dash='dash'),
            )
            # Marker at intersection
            fig.add_trace(go.Scatter(
                x=[col], y=[y_labels[row]],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                showlegend=False,
                hoverinfo='skip',
            ))

        fig.update_layout(
            title=f'Sinogram: Grain {grainNr} ({variant}) — {nSp} spots',
            xaxis_title='Scan Nr',
            yaxis_title='Rotation Angle (°)',
            clickmode='event',
            **COMMON_LAYOUT,
        )
        return fig

    # --- Main intensity patch plot ---
    @callback(
        Output('patch-plot', 'figure'),
        Input('grain-dropdown', 'value'),
        Input('row-slider', 'value'),
        Input('col-slider', 'value'),
        Input('patch-size-slider', 'value'),
        Input('store-patch-vmin', 'data'),
        Input('store-patch-vmax', 'data'),
    )
    def update_patch(grainNr, row, col, patchHalf, vmin, vmax):
        fig = go.Figure()
        if grainNr is None:
            fig.update_layout(title="Select a grain", **COMMON_LAYOUT)
            return fig

        nSp = int(grainSpots[grainNr]) if grainNr < len(grainSpots) else 0
        if nSp == 0 or row >= nSp:
            fig.update_layout(
                title=f"Spot {row} out of range (grain has {nSp} spots)",
                **COMMON_LAYOUT)
            return fig

        theta_val = omegas[grainNr, row]

        patch = get_intensity_patch(grainNr, row, col, patchHalf)
        if patch is None:
            fig.update_layout(
                title=f'No data: Grain {grainNr}, Spot {row}, Scan {col}',
                **COMMON_LAYOUT)
            return fig

        fig.add_trace(go.Heatmap(
            z=patch,
            colorscale='Viridis',
            zmin=vmin, zmax=vmax,
            colorbar=dict(title='Intensity'),
            hovertemplate=(
                'Y: %{x}<br>'
                'Z: %{y}<br>'
                'Intensity: %{z:.1f}<extra></extra>'
            )
        ))

        fig.update_layout(
            title=(f'Intensity: Grain {grainNr}, Spot {row} '
                   f'(θ={theta_val:.1f}°), Scan {col}'),
            xaxis_title='Y (pixels)',
            yaxis_title='Z (pixels)',
            yaxis=dict(scaleanchor='x'),
            **COMMON_LAYOUT,
        )
        return fig

    # ── Run ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Starting server: http://{args.HostName}:{args.portNr}")
    print(f"{'=' * 60}\n")
    app.run(port=args.portNr, host=args.HostName, debug=False)
