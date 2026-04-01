#!/usr/bin/env python
"""Forward Simulation -> Sinogram -> Tomo Reconstruction

Given an orientation matrix (or multiple orientations) and a PF-HEDM dataset,
this script:
  1. Runs ForwardSimulationCompressed to predict diffraction spots
  2. Extracts intensity patches from detector data at predicted spot locations
  3. Builds an omega-sorted sinogram
  4. Runs tomographic reconstruction via midas_tomo_python

Orientation sources (mutually exclusive):
  --orient "O11 O12 ... O33"   : single orientation (9 values)
  --grainsFile Grains.csv       : multi-grain file from ProcessGrains
  --h5File microstructure_pf.h5 : PF-HEDM consolidated HDF5

For multi-grain inputs (--grainsFile / --h5File), the forward simulation runs
once with all grains.  Sinograms and reconstructions are produced per grain,
saved under  <outDir>/grain_NNNN/ .

Data sources (auto-detected unless --useRaw or --useInputAll is set):
  - Zarr ZIP archives ({FileStem}_{LayerNr}.MIDAS.zip)
  - Raw detector files: HDF5 (.h5, .hdf5, .vrx.h5), GE binary (.ge, .ge5),
    TIFF (.tif, .tiff), CBF (.cbf) — all with optional .bz2 compression
  - InputAll CSVs (GrainRadius matching via --useInputAll)

Usage:
  # Single orientation:
  python forward_sim_sinogram.py --paramFile ps_sto_pf.txt \
      --orient "O11 O12 ... O33" --nCPUs 4

  # Multi-grain from Grains.csv:
  python forward_sim_sinogram.py --paramFile ps_sto_pf.txt \
      --grainsFile Grains.csv --nCPUs 4

  # Multi-grain from PF-HEDM h5 file:
  python forward_sim_sinogram.py --paramFile ps_sto_pf.txt \
      --h5File microstructure_pf.h5 --nCPUs 4

  # Force raw file mode:
  python forward_sim_sinogram.py --paramFile ps_sto_pf.txt \
      --orient "O11 O12 ... O33" --useRaw --nCPUs 4
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from numba import jit

# Disable HDF5 file locking — must be set before h5py is imported.
# Read-only access doesn't need locks, and POSIX locks over NFS cause
# severe contention when many workers open files concurrently.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

# ---------------------------------------------------------------------------
# Portable copies of BZ2Context and apply_correction from ffGenerateZipRefactor.
# Copied here to avoid pulling in zarr/numcodecs/midas_config at import time.
# ---------------------------------------------------------------------------

class BZ2Context:
    """Context manager for transparent .bz2 decompression.

    If the file ends in .bz2, decompresses it (keeping original),
    yields the uncompressed path, and deletes the temp file on exit.
    If the file is not .bz2, yields the original path unchanged.
    """
    def __init__(self, filepath):
        self.filepath = str(filepath)
        self.temp_path = None
        self.is_bz2 = self.filepath.endswith('.bz2')

    def __enter__(self):
        if not self.is_bz2:
            return self.filepath
        self.temp_path = self.filepath[:-4]
        subprocess.run(['bzip2', '-d', '-k', '-f', self.filepath], check=True)
        return self.temp_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_bz2 and self.temp_path and os.path.exists(self.temp_path):
            os.remove(self.temp_path)


@jit(nopython=True)
def apply_correction(img, dark_mean, pre_proc_thresh_val):
    """Applies dark correction with threshold. Numba JIT, type-agnostic."""
    result = np.empty(img.shape, dtype=img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i, j, k] < pre_proc_thresh_val[j, k]:
                    result[i, j, k] = 0
                else:
                    result[i, j, k] = max(0, int(img[i, j, k]) - int(dark_mean[j, k]))
    return result


# ---------------------------------------------------------------------------
# Helpers: parse the MIDAS parameter file
# ---------------------------------------------------------------------------

def parse_param_file(fn):
    """Read a MIDAS ps_*.txt parameter file and return a dict of key values."""
    params = {}
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            key, val = parts
            # accumulate list-valued keys
            if key in ('RingThresh', 'ImTransOpt', 'OmegaRange', 'BoxSize',
                        'RingNumbers'):
                params.setdefault(key, []).append(val)
            else:
                params[key] = val
    return params


def get_scan_dirs(base_dir, start_file_nr, n_scans, scan_step):
    """Return list of (scanIdx, dirPath) tuples for all scan directories."""
    dirs = []
    for i in range(n_scans):
        layer_nr = start_file_nr + i * scan_step
        d = os.path.join(base_dir, str(layer_nr))
        dirs.append((i, layer_nr, d))
    return dirs


# ---------------------------------------------------------------------------
# Quaternion / orientation helpers
# ---------------------------------------------------------------------------

def quat2orient_mat(q):
    """Convert quaternion (w, x, y, z) to a flat 9-element orientation matrix.

    Returns row-major: [O11 O12 O13 O21 O22 O23 O31 O32 O33].
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    om = np.empty(9, dtype=np.float64)
    om[0] = 1 - 2*(y*y + z*z)
    om[1] = 2*(x*y - w*z)
    om[2] = 2*(x*z + w*y)
    om[3] = 2*(x*y + w*z)
    om[4] = 1 - 2*(x*x + z*z)
    om[5] = 2*(y*z - w*x)
    om[6] = 2*(x*z - w*y)
    om[7] = 2*(y*z + w*x)
    om[8] = 1 - 2*(x*x + y*y)
    return om


# ---------------------------------------------------------------------------
# Read multi-grain orientations from H5 or Grains.csv
# ---------------------------------------------------------------------------

def read_h5_orientations(h5_path):
    """Read grain orientations from a PF-HEDM microstructure_pf.h5 file.

    Returns a list of dicts, one per voxel/grain:
      {'grainID': int, 'orient': 9-elem array, 'pos': (x,y,z),
       'lattice': (a,b,c,alpha,beta,gamma)}

    The H5 file is expected to contain (written by pf_MIDAS.py):
      voxels/orientation_matrix  (N, 3, 3)
      voxels/quaternion          (N, 4)  – used if orient_matrix absent
      voxels/position            (N, 3)  – optional, defaults to (0,0,0)
      voxels/lattice_params      (N, 6)  – optional
    """
    import h5py
    grains = []
    with h5py.File(h5_path, 'r') as hf:
        # Orientation matrices
        if 'voxels/orientation_matrix' in hf:
            om_all = np.array(hf['voxels/orientation_matrix'])  # (N,3,3)
            om_flat = om_all.reshape(-1, 9)
        elif 'voxels/quaternion' in hf:
            quats = np.array(hf['voxels/quaternion'])  # (N,4)
            om_flat = np.array([quat2orient_mat(q) for q in quats])
        else:
            raise ValueError(f'{h5_path} has neither '
                             f'voxels/orientation_matrix nor voxels/quaternion')

        n_grains = om_flat.shape[0]

        # Positions (optional)
        if 'voxels/position' in hf:
            pos_all = np.array(hf['voxels/position'])  # (N,3)
        else:
            pos_all = np.zeros((n_grains, 3))

        # Lattice parameters (optional)
        if 'voxels/lattice_params' in hf:
            lp_all = np.array(hf['voxels/lattice_params'])  # (N,6)
        else:
            lp_all = None

        for i in range(n_grains):
            g = {
                'grainID': i + 1,
                'orient': om_flat[i],
                'pos': tuple(pos_all[i]),
            }
            if lp_all is not None:
                g['lattice'] = tuple(lp_all[i])
            else:
                g['lattice'] = None  # will use param-file default
            grains.append(g)

    print(f'  Read {len(grains)} orientations from {h5_path}')
    return grains


def parse_grains_csv_file(csv_path):
    """Parse a MIDAS Grains.csv file (written by ProcessGrains).

    Returns a list of dicts with the same structure as read_h5_orientations.
    """
    grains = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            vals = line.split('\t')
            if len(vals) < 19:
                vals = line.split()  # try space-separated
            if len(vals) < 19:
                continue
            fvals = [float(v) for v in vals]
            g = {
                'grainID': int(fvals[0]),
                'orient': np.array(fvals[1:10]),
                'pos': (fvals[10], fvals[11], fvals[12]),
                'lattice': tuple(fvals[13:19]),
            }
            grains.append(g)
    print(f'  Parsed {len(grains)} grains from {csv_path}')
    return grains


# ---------------------------------------------------------------------------
# Step 1: Create temporary Grains.csv and run ForwardSimulationCompressed
# ---------------------------------------------------------------------------

def create_grains_csv(orient_mat, lattice_str, out_path):
    """Write a single-grain Grains.csv for ForwardSimulationCompressed."""
    om = orient_mat  # 9-element flat array
    lat_parts = lattice_str.split()
    a, b, c = float(lat_parts[0]), float(lat_parts[1]), float(lat_parts[2])
    alpha, beta, gamma = float(lat_parts[3]), float(lat_parts[4]), float(lat_parts[5])

    with open(out_path, 'w') as f:
        f.write('%NumGrains 1\n')
        f.write('%BeamCenter 0.000000\n')
        f.write('%BeamThickness 1000.000000\n')
        f.write('%GlobalPosition 0.000000\n')
        f.write('%NumPhases 1\n')
        f.write('%PhaseInfo\n')
        f.write(f'%\tSpaceGroup:221\n')
        f.write(f'%\tLattice Parameter: {a:.6f} {b:.6f} {c:.6f} '
                f'{alpha:.6f} {beta:.6f} {gamma:.6f}\n')
        f.write('%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t'
                'X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\n')
        # GrainID=1, orient, pos=(0,0,0), lattice params, zeros for strain
        f.write(f'1\t{om[0]:.6f}\t{om[1]:.6f}\t{om[2]:.6f}\t'
                f'{om[3]:.6f}\t{om[4]:.6f}\t{om[5]:.6f}\t'
                f'{om[6]:.6f}\t{om[7]:.6f}\t{om[8]:.6f}\t'
                f'0.0\t0.0\t0.0\t'
                f'{a:.6f}\t{b:.6f}\t{c:.6f}\t'
                f'{alpha:.6f}\t{beta:.6f}\t{gamma:.6f}\n')
    print(f'  Created {out_path}')


def create_multi_grains_csv(grains_list, lattice_str, out_path):
    """Write a multi-grain Grains.csv for ForwardSimulationCompressed.

    Parameters
    ----------
    grains_list : list of dict
        Each dict has 'grainID', 'orient' (9-elem), 'pos' (3-tuple),
        and optionally 'lattice' (6-tuple).  If lattice is None, the
        default *lattice_str* from the param file is used.
    lattice_str : str
        Default lattice constants from the parameter file.
    out_path : str
        Output Grains.csv path.
    """
    lat_parts = lattice_str.split()
    a0, b0, c0 = float(lat_parts[0]), float(lat_parts[1]), float(lat_parts[2])
    al0, be0, ga0 = float(lat_parts[3]), float(lat_parts[4]), float(lat_parts[5])

    n = len(grains_list)
    with open(out_path, 'w') as f:
        f.write(f'%NumGrains {n}\n')
        f.write('%BeamCenter 0.000000\n')
        f.write('%BeamThickness 1000.000000\n')
        f.write('%GlobalPosition 0.000000\n')
        f.write('%NumPhases 1\n')
        f.write('%PhaseInfo\n')
        f.write(f'%\tSpaceGroup:221\n')
        f.write(f'%\tLattice Parameter: {a0:.6f} {b0:.6f} {c0:.6f} '
                f'{al0:.6f} {be0:.6f} {ga0:.6f}\n')
        f.write('%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t'
                'X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\n')
        for g in grains_list:
            om = g['orient']
            x, y, z = g['pos']
            if g.get('lattice') is not None:
                a, b, c, al, be, ga = g['lattice']
            else:
                a, b, c, al, be, ga = a0, b0, c0, al0, be0, ga0
            f.write(f'{g["grainID"]}\t'
                    f'{om[0]:.6f}\t{om[1]:.6f}\t{om[2]:.6f}\t'
                    f'{om[3]:.6f}\t{om[4]:.6f}\t{om[5]:.6f}\t'
                    f'{om[6]:.6f}\t{om[7]:.6f}\t{om[8]:.6f}\t'
                    f'{x:.6f}\t{y:.6f}\t{z:.6f}\t'
                    f'{a:.6f}\t{b:.6f}\t{c:.6f}\t'
                    f'{al:.6f}\t{be:.6f}\t{ga:.6f}\n')
    print(f'  Created {out_path} with {n} grains')


def create_fwd_param_file(orig_param_file, grains_csv, out_param_file):
    """Create a modified param file for ForwardSimulationCompressed.

    - Points InputFile to the Grains.csv
    - Sets WriteSpots 1, WriteImage 0
    - Keeps nScans as 1 (we only need spot positions, not per-scan sims)
    """
    with open(orig_param_file) as f:
        lines = f.readlines()

    # Parse OmegaEnd from the original file if not present
    omega_start = None
    omega_step = None
    start_nr = None
    end_nr = None
    has_omega_end = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        parts = stripped.split()
        if parts[0] == 'OmegaStart':
            omega_start = float(parts[1])
        elif parts[0] == 'OmegaStep':
            omega_step = float(parts[1])
        elif parts[0] == 'StartNr':
            start_nr = int(parts[1])
        elif parts[0] == 'EndNr':
            end_nr = int(parts[1])
        elif parts[0] == 'OmegaEnd':
            has_omega_end = True

    with open(out_param_file, 'w') as f:
        wrote_input = False
        wrote_spots = False
        wrote_image = False
        wrote_nscans = False
        for line in lines:
            stripped = line.strip()
            key = stripped.split()[0] if stripped and not stripped.startswith('#') else ''
            if key == 'nScans':
                f.write('nScans 1\n')
                wrote_nscans = True
            elif key == 'WriteSpots':
                f.write('WriteSpots 1\n')
                wrote_spots = True
            elif key == 'WriteImage':
                f.write('WriteImage 0\n')
                wrote_image = True
            elif key in ('InFileName', 'RefinementFileName', 'InputFile'):
                f.write(f'InFileName {grains_csv}\n')
                wrote_input = True
            else:
                f.write(line)
        # Add any missing keys
        if not wrote_input:
            f.write(f'InFileName {grains_csv}\n')
        if not wrote_spots:
            f.write('WriteSpots 1\n')
        if not wrote_image:
            f.write('WriteImage 0\n')
        if not wrote_nscans:
            f.write('nScans 1\n')
        # Add OmegaEnd if not in original file
        if not has_omega_end and omega_start is not None and omega_step is not None:
            if end_nr is not None and start_nr is not None:
                omega_end = omega_start + (end_nr - start_nr + 1) * omega_step
                f.write(f'OmegaEnd {omega_end}\n')
                print(f'  Added OmegaEnd {omega_end} '
                      f'(from OmegaStart={omega_start}, '
                      f'EndNr={end_nr}, OmegaStep={omega_step})')
    print(f'  Created {out_param_file}')


def run_forward_simulation(param_file, n_cpus, work_dir):
    """Run ForwardSimulationCompressed and return path to SpotMatrixGen.csv."""
    # Try to find the binary
    fwd_sim = shutil.which('ForwardSimulationCompressed')
    if fwd_sim is None:
        # Try MIDAS standard locations
        midas_root = os.environ.get('MIDAS_ROOT',
                                     os.path.expanduser('~/opt/MIDAS'))
        fwd_sim = os.path.join(midas_root, 'FF_HEDM', 'bin',
                               'ForwardSimulationCompressed')
        if not os.path.isfile(fwd_sim):
            raise FileNotFoundError(
                f'Cannot find ForwardSimulationCompressed binary. '
                f'Tried PATH and {fwd_sim}')

    cmd = [fwd_sim, os.path.basename(param_file), str(n_cpus)]
    print(f'  Running: {" ".join(cmd)} in {work_dir}')
    t0 = time.time()
    result = subprocess.run(cmd, cwd=work_dir,
                            capture_output=True, text=True)
    print(f'  ForwardSimulationCompressed finished in {time.time()-t0:.1f}s')
    if result.returncode != 0:
        print('STDOUT:', result.stdout[-2000:] if len(result.stdout) > 2000
                                                else result.stdout)
        print('STDERR:', result.stderr[-2000:] if len(result.stderr) > 2000
                                                else result.stderr)
        raise RuntimeError(f'ForwardSimulationCompressed failed with rc={result.returncode}')

    spot_file = os.path.join(work_dir, 'SpotMatrixGen.csv')
    if not os.path.isfile(spot_file):
        raise FileNotFoundError(f'SpotMatrixGen.csv not produced in {work_dir}')
    return spot_file


def parse_spot_matrix_gen(fn):
    """Parse SpotMatrixGen.csv into a structured array.

    Columns: GrainID SpotID Omega DetectorHor DetectorVert OmeRaw Eta
             RingNr YLab ZLab Theta StrainError ScanNr RingRad omeBin
    """
    data = np.genfromtxt(fn, skip_header=1, delimiter='\t')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    print(f'  Parsed {len(data)} spots from SpotMatrixGen.csv')

    # Build a structured list sorted by omega
    spots = []
    for row in data:
        spots.append({
            'grainID':    int(row[0]),
            'spotID':     int(row[1]),
            'omega':      row[2],       # corrected omega
            'detHor':     row[3],       # y pixel
            'detVert':    row[4],       # z pixel
            'omeRaw':     row[5],
            'eta':        row[6],
            'ringNr':     int(row[7]),
            'yLab':       row[8],
            'zLab':       row[9],
            'theta':      row[10],
            'scanNr':     int(row[12]) if len(row) > 12 else 0,
            'ringRad':    row[13] if len(row) > 13 else 0,
            'omeBin':     int(row[14]) if len(row) > 14 else 0,
        })

    # Sort by omega
    spots.sort(key=lambda s: s['omega'])
    print(f'  Spots sorted by omega: [{spots[0]["omega"]:.2f}° .. '
          f'{spots[-1]["omega"]:.2f}°]')
    return spots


def _match_spots_single_scan(args):
    """Worker: match predicted spots against InputAllExtraInfoFittingAll in one scan dir.

    Pre-filters observed spots within ±lab_tol (microns) in YLab/ZLab and
    ±ome_tol (degrees) in Omega, within the same ring.  Then picks the
    closest by Euclidean distance.

    Returns (scan_idx, intensities_1d) or (scan_idx, None) on failure.
    """
    (scan_idx, layer_nr, scan_dir, pred_ylab, pred_zlab, pred_omega,
     pred_ring_nr, lab_tol, ome_tol) = args

    # Find the InputAllExtraInfoFittingAll file
    input_all_fn = os.path.join(scan_dir, 'InputAllExtraInfoFittingAll.csv')
    if not os.path.isfile(input_all_fn):
        # Try numbered variants
        for suffix in ['0', '1']:
            alt = os.path.join(scan_dir,
                               f'InputAllExtraInfoFittingAll{suffix}.csv')
            if os.path.isfile(alt):
                input_all_fn = alt
                break
        else:
            return (scan_idx, None)

    try:
        obs = np.loadtxt(input_all_fn, skiprows=1)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
    except Exception:
        return (scan_idx, None)

    if len(obs) == 0:
        return (scan_idx, None)

    # Columns: 0:YLab 1:ZLab 2:Omega 3:GrainRadius 5:RingNumber
    obs_ylab = obs[:, 0]
    obs_zlab = obs[:, 1]
    obs_omega = obs[:, 2]
    obs_grain_radius = obs[:, 3]
    obs_ring = obs[:, 5].astype(int)

    n_pred = len(pred_ylab)
    intensities = np.zeros(n_pred, dtype=np.float64)

    for i in range(n_pred):
        # Pre-filter: same ring, within ±lab_tol in Y/Z, ±ome_tol in omega
        mask = ((obs_ring == pred_ring_nr[i]) &
                (np.abs(obs_ylab - pred_ylab[i]) <= lab_tol) &
                (np.abs(obs_zlab - pred_zlab[i]) <= lab_tol) &
                (np.abs(obs_omega - pred_omega[i]) <= ome_tol))

        if not np.any(mask):
            continue

        # Pick closest within the tolerance box
        dy = obs_ylab[mask] - pred_ylab[i]
        dz = obs_zlab[mask] - pred_zlab[i]
        dist = dy**2 + dz**2
        best_idx = np.argmin(dist)
        intensities[i] = obs_grain_radius[mask][best_idx]

    return (scan_idx, intensities)


def build_sinogram_from_inputall(spots, scan_dirs, px_size, omega_step,
                                  n_workers=4, pxtol=4, ome_frame_tol=2):
    """Build sinogram using GrainRadius from InputAllExtraInfoFittingAll files.

    For each scan dir, matches predicted spots (by YLab/ZLab/Omega/RingNr)
    against observed spots and uses GrainRadius as intensity.

    pxtol:         pixel tolerance (default 4 pixels, converted to microns)
    ome_frame_tol: omega tolerance in frames (default 2, converted to degrees)
    """
    from multiprocessing import Pool

    n_hkls = len(spots)
    n_scans = len(scan_dirs)
    sino = np.zeros((n_hkls, n_scans), dtype=np.float64)
    omegas = np.array([s['omega'] for s in spots])

    # Convert tolerances to physical units
    lab_tol = pxtol * px_size       # pixels → microns
    ome_tol = ome_frame_tol * omega_step  # frames → degrees

    pred_ylab = [s['yLab'] for s in spots]
    pred_zlab = [s['zLab'] for s in spots]
    pred_omega = [s['omega'] for s in spots]
    pred_ring_nr = [s['ringNr'] for s in spots]

    print(f'\n  Building sinogram from InputAll: {n_hkls} HKLs × {n_scans} scans')
    print(f'  Tolerances: ±{pxtol} px (±{lab_tol:.1f} µm), '
          f'±{ome_frame_tol} frames (±{ome_tol:.3f}°)')
    print(f'  Using {n_workers} parallel workers')

    worker_args = []
    for scan_idx, layer_nr, scan_dir in scan_dirs:
        worker_args.append((
            scan_idx, layer_nr, scan_dir,
            pred_ylab, pred_zlab, pred_omega, pred_ring_nr,
            lab_tol, ome_tol
        ))

    t0 = time.time()
    completed = 0
    with Pool(processes=n_workers) as pool:
        for scan_idx, intensities in pool.imap_unordered(
                _match_spots_single_scan, worker_args):
            completed += 1
            if intensities is not None:
                sino[:, scan_idx] = intensities
            else:
                layer_nr = scan_dirs[scan_idx][1]
                print(f'    WARNING: scan {scan_idx} (layer {layer_nr}) '
                      f'InputAll not found or empty')
            if completed % 10 == 0 or completed == n_scans:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f'    Progress: {completed}/{n_scans} scans '
                      f'[{elapsed:.1f}s, {rate:.1f} scans/s]')

    return sino, omegas


# ---------------------------------------------------------------------------
# Step 2: Extract intensity from zip files to build sinogram
# ---------------------------------------------------------------------------

def inverse_transform_coords(y, z, trans_opts, nr_pixels_y, nr_pixels_z):
    """Map predicted detector coords (in transformed space) back to raw zarr coords.

    The MIDAS pipeline applies ImTransOpt to the raw image before calibration
    and forward simulation.  ForwardSim reports positions in the *transformed*
    image.  To index into the *raw* zarr data we must invert the transform.

    TransOpt meanings (applied to image):
      0 = no-op
      1 = FlipLR  (mirror Y axis)       inverse is the same: FlipLR
      2 = FlipUD  (mirror Z axis)       inverse is the same: FlipUD
      3 = Transpose (swap Y <-> Z)      inverse is the same: Transpose

    Inverse is applied in **reverse** order of the forward list.
    """
    yf, zf = float(y), float(z)
    for opt in reversed(trans_opts):
        if opt == 1:    # FlipLR
            yf = nr_pixels_y - 1.0 - yf
        elif opt == 2:  # FlipUD
            zf = nr_pixels_z - 1.0 - zf
        elif opt == 3:  # Transpose
            yf, zf = zf, yf
    return yf, zf


def _pred_to_raw_coords(y_pred, z_pred, trans_opts, ny, nz):
    """Map ForwardSim predicted coords to raw zarr/HDF5 (row, col).

    Pipeline: raw → ImTransOpt → transpose → ForwardSim coords.
    Inverse:  ForwardSim (Y,Z) → undo ImTransOpt → (y_raw, z_raw)
              → read at zarr[z_raw, y_raw]  (Z=row, Y=col).
    """
    if trans_opts:
        y_raw, z_raw = inverse_transform_coords(
            y_pred, z_pred, trans_opts, ny, nz)
    else:
        y_raw, z_raw = float(y_pred), float(z_pred)
    # zarr layout: (NrPixelsZ, NrPixelsY) → row=Z, col=Y
    return int(round(z_raw)), int(round(y_raw))


def _detect_format(ext):
    """Detect file format from extension, stripping .bz2 and leading dots.

    Returns one of: 'hdf5', 'tiff', 'cbf', 'ge'.
    """
    clean = ext.lower()
    if clean.endswith('.bz2'):
        clean = clean[:-4]
    # Strip everything up to and including the last dot to get the final suffix
    # e.g., '.vrx.h5' -> 'h5', '.ge5' -> 'ge5', '.tif' -> 'tif'
    last_dot = clean.rfind('.')
    if last_dot >= 0:
        suffix = clean[last_dot + 1:]
    else:
        suffix = clean
    # Strip trailing digits for GE variants: ge2, ge3, ge5 -> ge
    suffix_base = suffix.rstrip('0123456789')

    if suffix in ('h5', 'hdf5', 'hdf', 'nxs'):
        return 'hdf5'
    elif suffix_base in ('tif', 'tiff'):
        return 'tiff'
    elif suffix == 'cbf':
        return 'cbf'
    else:
        return 'ge'


def _build_raw_filename(raw_folder, file_stem, layer_nr, padding, ext):
    """Construct raw data filename.

    Ext from the ps file already includes the leading dot (e.g., '.vrx.h5').
    Pattern: {RawFolder}/{FileStem}_{paddedLayerNr}{Ext}
    Also checks for .bz2 variant if the plain file doesn't exist.
    """
    fn = os.path.join(raw_folder,
                      f'{file_stem}_{layer_nr:0{padding}d}{ext}')
    if os.path.isfile(fn):
        return fn
    fn_bz2 = fn + '.bz2'
    if os.path.isfile(fn_bz2):
        return fn_bz2
    return None


class _RawFileReader:
    """Lazy frame-at-a-time reader for raw detector files.

    Opens the file once, reads individual frames on demand via __getitem__.
    Keeps the file handle open until close() or context-manager exit.
    Supports HDF5, TIFF, GE binary, and CBF formats with .bz2.
    """

    def __init__(self, raw_folder, file_stem, layer_nr, padding, ext,
                 header_size, bytes_per_pixel, nr_pixels_y, nr_pixels_z,
                 data_loc='exchange/data'):
        self.fn = _build_raw_filename(raw_folder, file_stem, layer_nr,
                                       padding, ext)
        self.fmt = _detect_format(ext)
        self.header_size = header_size
        self.bytes_per_pixel = bytes_per_pixel
        self.nr_pixels_y = nr_pixels_y
        self.nr_pixels_z = nr_pixels_z
        self.data_loc = data_loc
        self.n_frames = 0
        self._bz2_ctx = None
        self._hf = None       # h5py File handle
        self._dataset = None   # h5py dataset reference
        self._tiff_data = None # TIFF: full array (typically small)
        self._ge_path = None   # GE: path to uncompressed file

    def open(self):
        """Open the file and determine frame count. Returns self or None."""
        if self.fn is None:
            return None

        self._bz2_ctx = BZ2Context(self.fn)
        uncompressed = self._bz2_ctx.__enter__()

        try:
            if self.fmt == 'hdf5':
                import h5py
                self._hf = h5py.File(uncompressed, 'r')
                if self.data_loc in self._hf:
                    self._dataset = self._hf[self.data_loc]
                else:
                    ds_name = os.path.basename(self.data_loc)
                    if ds_name in self._hf:
                        self._dataset = self._hf[ds_name]
                    else:
                        self.close()
                        return None
                self.n_frames = self._dataset.shape[0]

            elif self.fmt == 'tiff':
                import tifffile
                self._tiff_data = tifffile.imread(uncompressed)
                if self._tiff_data.ndim == 2:
                    self._tiff_data = self._tiff_data.reshape(
                        1, *self._tiff_data.shape)
                self.n_frames = self._tiff_data.shape[0]

            elif self.fmt == 'cbf':
                from read_cbf import read_cbf as _read_cbf
                _, cbf_data = _read_cbf(uncompressed, check_md5=False)
                if cbf_data.ndim == 2:
                    cbf_data = cbf_data.reshape(1, *cbf_data.shape)
                self._tiff_data = cbf_data  # reuse same slot
                self.n_frames = cbf_data.shape[0]

            else:  # GE binary
                self._ge_path = uncompressed
                file_size = os.path.getsize(uncompressed)
                n_pixels = self.nr_pixels_y * self.nr_pixels_z
                self.n_frames = ((file_size - self.header_size) //
                                 (self.bytes_per_pixel * n_pixels))

        except Exception as e:
            print(f'  WARNING: Failed to open {self.fn}: {e}')
            self.close()
            return None

        return self

    def read_frame(self, fi):
        """Read a single frame by index. Returns 2D ndarray (nZ, nY) as float64."""
        if self.fmt == 'hdf5':
            return self._dataset[fi].astype(np.float64)

        elif self.fmt in ('tiff', 'cbf'):
            return self._tiff_data[fi].astype(np.float64)

        else:  # GE binary
            dtype = (np.uint32 if self.bytes_per_pixel == 4
                     else np.uint16)
            n_pixels = self.nr_pixels_y * self.nr_pixels_z
            offset = (self.header_size +
                      fi * self.bytes_per_pixel * n_pixels)
            data = np.fromfile(self._ge_path, dtype=dtype,
                               count=n_pixels, offset=offset)
            return data.reshape(
                self.nr_pixels_z, self.nr_pixels_y).astype(np.float64)

    @property
    def supports_partial_read(self):
        """True if the format supports reading subregions without loading
        the full frame (HDF5 only — the dataset supports hyperslab selection)."""
        return self.fmt == 'hdf5'

    def read_patch(self, fi, z0, z1, y0, y1):
        """Read a subregion of one frame directly from the dataset.

        Only supported for HDF5 (supports_partial_read == True).
        Returns 2D ndarray of shape (z1-z0, y1-y0) as float64.
        """
        return self._dataset[fi, z0:z1, y0:y1].astype(np.float64)

    def read_frames_batch(self, frame_indices):
        """Read multiple full frames in one I/O call.

        Fallback for non-HDF5 formats where partial reads aren't possible.

        Parameters
        ----------
        frame_indices : list of int
            Must be sorted ascending.

        Returns
        -------
        dict mapping frame_index -> 2D ndarray (nZ, nY) as float64
        """
        if not frame_indices:
            return {}

        if self.fmt == 'hdf5':
            # Single HDF5 hyperslab read for all frames at once
            batch = self._dataset[frame_indices].astype(np.float64)
            return {fi: batch[i] for i, fi in enumerate(frame_indices)}

        elif self.fmt in ('tiff', 'cbf'):
            # Already in memory, just index
            return {fi: self._tiff_data[fi].astype(np.float64)
                    for fi in frame_indices}

        else:  # GE binary — read contiguous slab min..max
            dtype = (np.uint32 if self.bytes_per_pixel == 4
                     else np.uint16)
            n_pixels = self.nr_pixels_y * self.nr_pixels_z
            fi_min, fi_max = frame_indices[0], frame_indices[-1]
            slab_count = fi_max - fi_min + 1
            offset = (self.header_size +
                      fi_min * self.bytes_per_pixel * n_pixels)
            slab = np.fromfile(
                self._ge_path, dtype=dtype,
                count=slab_count * n_pixels, offset=offset
            ).reshape(slab_count, self.nr_pixels_z, self.nr_pixels_y
                      ).astype(np.float64)
            return {fi: slab[fi - fi_min] for fi in frame_indices}

    def close(self):
        """Release file handles."""
        if self._hf is not None:
            self._hf.close()
            self._hf = None
            self._dataset = None
        self._tiff_data = None
        self._ge_path = None
        if self._bz2_ctx is not None:
            self._bz2_ctx.__exit__(None, None, None)
            self._bz2_ctx = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _load_dark_mean(dark_fn, dark_loc, ext, header_size, bytes_per_pixel,
                    nr_pixels_y, nr_pixels_z, skip_frame):
    """Load dark file and compute mean dark frame.

    Parameters
    ----------
    dark_fn : str
        Path to the dark file.
    dark_loc : str
        HDF5 dataset path for dark data.
    ext : str
        File extension (used for format detection if dark file extension differs).
    skip_frame : int
        Number of initial frames to skip when computing mean.

    Returns
    -------
    dark_mean : ndarray of shape (nZ, nY) as float64, or None on failure.
    """
    if not dark_fn or not os.path.isfile(dark_fn):
        return None

    # Detect format from the dark file's own extension
    dark_ext = ''
    dark_lower = dark_fn.lower()
    if dark_lower.endswith('.bz2'):
        dark_ext = dark_fn[:-4]  # strip .bz2 to get real ext
    else:
        dark_ext = dark_fn
    # Get suffix for format detection
    from pathlib import Path as _Path
    dark_path = _Path(dark_ext)
    dark_suffix = ''.join(dark_path.suffixes)  # e.g., '.vrx.h5'
    if not dark_suffix:
        dark_suffix = ext  # fall back to data extension

    fmt = _detect_format(dark_suffix)

    try:
        with BZ2Context(dark_fn) as uncompressed:
            if fmt == 'hdf5':
                import h5py
                with h5py.File(uncompressed, 'r') as hf:
                    dark_frames = None
                    if dark_loc in hf:
                        dark_frames = hf[dark_loc][:]
                    else:
                        # Try just the dataset name
                        ds_name = os.path.basename(dark_loc)
                        if ds_name in hf:
                            dark_frames = hf[ds_name][:]
                    if dark_frames is None:
                        print(f'  WARNING: dark dataset {dark_loc} not found in {dark_fn}')
                        return None
                    if dark_frames.ndim == 2:
                        dark_frames = dark_frames.reshape(1, *dark_frames.shape)

            elif fmt == 'tiff':
                import tifffile
                dark_frames = tifffile.imread(uncompressed)
                if dark_frames.ndim == 2:
                    dark_frames = dark_frames.reshape(1, *dark_frames.shape)

            elif fmt == 'cbf':
                try:
                    from read_cbf import read_cbf as _read_cbf
                    _, dark_data = _read_cbf(uncompressed, check_md5=False)
                    dark_frames = dark_data.reshape(1, *dark_data.shape)
                except ImportError:
                    print('  WARNING: read_cbf not available for dark file')
                    return None

            else:  # GE binary
                dtype = np.uint32 if bytes_per_pixel == 4 else np.uint16
                file_size = os.path.getsize(uncompressed)
                n_pixels = nr_pixels_y * nr_pixels_z
                n_frames = (file_size - header_size) // (bytes_per_pixel * n_pixels)
                raw = np.fromfile(uncompressed, dtype=dtype, offset=header_size)
                dark_frames = raw[:n_frames * n_pixels].reshape(
                    (n_frames, nr_pixels_z, nr_pixels_y))

        # Compute mean, skipping initial frames
        dark_frames = dark_frames.astype(np.float64)
        if dark_frames.shape[0] > skip_frame:
            dark_mean = np.mean(dark_frames[skip_frame:], axis=0)
        else:
            dark_mean = dark_frames[0]
        return dark_mean

    except Exception as e:
        print(f'  WARNING: Failed to load dark file {dark_fn}: {e}')
        return None


def _process_single_scan_raw(args):
    """Worker for sinogram building from raw detector files.

    HDF5: reads small patches directly via partial read.
    Other formats: batch-reads full frames, dark-corrects, extracts patches.
    Returns (scan_idx, intensities_1d) or (scan_idx, None) on failure.
    """
    (scan_idx, layer_nr, scan_dir, file_stem, padding, ext,
     header_size, bytes_per_pixel,
     spot_det_hor, spot_det_vert, spot_frame_lists,
     patch_half, trans_opts, nr_pixels_y, nr_pixels_z,
     raw_folder, data_loc, dark_mean, pre_proc_thresh) = args

    reader = _RawFileReader(
        raw_folder, file_stem, layer_nr, padding, ext,
        header_size, bytes_per_pixel, nr_pixels_y, nr_pixels_z,
        data_loc=data_loc)

    if reader.open() is None:
        return (scan_idx, None)

    try:
        n_total = reader.n_frames
        nz, ny = nr_pixels_z, nr_pixels_y
        n_hkls = len(spot_det_hor)
        intensities = np.zeros(n_hkls, dtype=np.float64)

        if reader.supports_partial_read:
            # ── HDF5 fast path: read small patches directly ──
            for hkl_idx in range(n_hkls):
                rc, cc = _pred_to_raw_coords(
                    spot_det_hor[hkl_idx], spot_det_vert[hkl_idx],
                    trans_opts, ny, nz)
                r0 = max(0, rc - patch_half)
                r1 = min(nz, rc + patch_half + 1)
                c0 = max(0, cc - patch_half)
                c1 = min(ny, cc + patch_half + 1)
                if r0 >= r1 or c0 >= c1:
                    continue

                if dark_mean is not None:
                    dp = dark_mean[r0:r1, c0:c1]
                    if pre_proc_thresh >= 0:
                        tp = dp + pre_proc_thresh

                for fi in spot_frame_lists[hkl_idx]:
                    if 0 <= fi < n_total:
                        patch = reader.read_patch(fi, r0, r1, c0, c1)
                        if dark_mean is not None:
                            if pre_proc_thresh >= 0:
                                patch = np.where(
                                    patch < tp, 0.0,
                                    np.maximum(0.0, patch - dp))
                            else:
                                patch = np.maximum(0.0, patch - dp)
                        intensities[hkl_idx] += float(np.sum(patch))

        else:
            # ── Fallback: batch-read full frames, extract patches ──
            frame_to_spots = {}
            for hkl_idx in range(n_hkls):
                rc, cc = _pred_to_raw_coords(
                    spot_det_hor[hkl_idx], spot_det_vert[hkl_idx],
                    trans_opts, ny, nz)
                for fi in spot_frame_lists[hkl_idx]:
                    if 0 <= fi < n_total:
                        frame_to_spots.setdefault(fi, []).append(
                            (hkl_idx, rc, cc))

            needed_frames = sorted(frame_to_spots.keys())
            frames_dict = reader.read_frames_batch(needed_frames)

            for fi in needed_frames:
                frame = frames_dict[fi]
                if dark_mean is not None:
                    if pre_proc_thresh >= 0:
                        thresh_arr = dark_mean + pre_proc_thresh
                        frame = np.where(frame < thresh_arr, 0.0,
                                         np.maximum(0.0, frame - dark_mean))
                    else:
                        frame = np.maximum(0.0, frame - dark_mean)

                for hkl_idx, rc, cc in frame_to_spots[fi]:
                    r0 = max(0, rc - patch_half)
                    r1 = min(nz, rc + patch_half + 1)
                    c0 = max(0, cc - patch_half)
                    c1 = min(ny, cc + patch_half + 1)
                    if r0 < r1 and c0 < c1:
                        intensities[hkl_idx] += float(
                            np.sum(frame[r0:r1, c0:c1]))

        return (scan_idx, intensities)
    finally:
        reader.close()


def build_sinogram_raw(spots, scan_dirs, file_stem, padding, ext,
                       header_size, bytes_per_pixel, raw_folder,
                       patch_half=10, ome_half=2, skip_frame=1,
                       n_workers=4, trans_opts=None,
                       nr_pixels_y=2880, nr_pixels_z=2880,
                       data_loc='exchange/data',
                       dark_fn=None, dark_loc='exchange/dark',
                       pre_proc_thresh=-1):
    """Build sinogram from raw detector files (HDF5, GE, TIFF, CBF).

    Same interface as build_sinogram but reads raw files from raw_folder
    instead of zarr zip archives. Applies dark correction if dark_fn is
    provided.

    Parameters
    ----------
    data_loc : str
        HDF5 dataset path for image data.
    dark_fn : str or None
        Path to dark file. If None, no dark correction.
    dark_loc : str
        HDF5 dataset path for dark data.
    pre_proc_thresh : int
        Pre-processing threshold above dark mean. -1 = simple subtraction.
    """
    from multiprocessing import Pool

    n_hkls = len(spots)
    n_scans = len(scan_dirs)
    sino = np.zeros((n_hkls, n_scans), dtype=np.float64)
    omegas = np.array([s['omega'] for s in spots])

    # Load dark frames once (shared across all workers)
    dark_mean = None
    if dark_fn:
        print(f'  Loading dark file: {dark_fn}')
        print(f'  Dark dataset: {dark_loc}')
        dark_mean = _load_dark_mean(
            dark_fn, dark_loc, ext, header_size, bytes_per_pixel,
            nr_pixels_y, nr_pixels_z, skip_frame)
        if dark_mean is not None:
            print(f'  Dark mean loaded: shape={dark_mean.shape}, '
                  f'mean={np.mean(dark_mean):.1f}')
            if pre_proc_thresh >= 0:
                print(f'  Pre-proc threshold: dark_mean + {pre_proc_thresh}')
            else:
                print(f'  Simple dark subtraction (no threshold)')
        else:
            print(f'  WARNING: Could not load dark file')

    spot_frames = []
    for s in spots:
        ob = s['omeBin']
        actual_bin = ob + skip_frame
        frames_needed = list(range(actual_bin - ome_half,
                                    actual_bin + ome_half + 1))
        spot_frames.append(frames_needed)

    spot_det_hor = [s['detHor'] for s in spots]
    spot_det_vert = [s['detVert'] for s in spots]

    if trans_opts is None:
        trans_opts = []

    print(f'\n  Building sinogram from raw: {n_hkls} HKLs × {n_scans} scans')
    print(f'  Patch size: {2*patch_half+1}x{2*patch_half+1}x{2*ome_half+1}')
    if trans_opts:
        print(f'  ImTransOpt: {trans_opts}')
    print(f'  Using {n_workers} parallel workers')

    worker_args = []
    for scan_idx, layer_nr, scan_dir in scan_dirs:
        worker_args.append((
            scan_idx, layer_nr, scan_dir, file_stem, padding, ext,
            header_size, bytes_per_pixel,
            spot_det_hor, spot_det_vert, spot_frames,
            patch_half, trans_opts, nr_pixels_y, nr_pixels_z,
            raw_folder, data_loc, dark_mean, pre_proc_thresh
        ))

    t0 = time.time()
    completed = 0
    with Pool(processes=n_workers) as pool:
        for scan_idx, intensities in pool.imap_unordered(
                _process_single_scan_raw, worker_args):
            completed += 1
            if intensities is not None:
                sino[:, scan_idx] = intensities
            else:
                layer_nr = scan_dirs[scan_idx][1]
                print(f'    WARNING: scan {scan_idx} (layer {layer_nr}) '
                      f'raw file not found or failed')
            if completed % 10 == 0 or completed == n_scans:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f'    Progress: {completed}/{n_scans} scans '
                      f'[{elapsed:.1f}s, {rate:.1f} scans/s]')

    return sino, omegas


def _process_single_scan(args):
    """Worker for sinogram building from zarr zip files.

    Reads small patches at the correct (row, col) in the raw frame.
    Returns (scan_idx, intensities_1d) or (scan_idx, None) on failure.
    """
    import zarr

    (scan_idx, layer_nr, scan_dir, file_stem, padding,
     spot_det_hor, spot_det_vert, spot_frame_lists,
     patch_half, trans_opts, nr_pixels_y, nr_pixels_z) = args

    zip_name = f'{file_stem}_{layer_nr:0{padding}d}.MIDAS.zip'
    zip_path = os.path.join(scan_dir, zip_name)
    if not os.path.isfile(zip_path):
        return (scan_idx, None)

    try:
        store = zarr.storage.ZipStore(zip_path, mode='r')
        zg = zarr.open_group(store, mode='r')
        data = zg['exchange/data']
        n_total, nz, ny = data.shape
    except Exception:
        return (scan_idx, None)

    n_hkls = len(spot_det_hor)
    intensities = np.zeros(n_hkls, dtype=np.float64)

    # Build reverse map: frame -> [(hkl_idx, row_center, col_center), ...]
    frame_to_spots = {}
    for hkl_idx in range(n_hkls):
        rc, cc = _pred_to_raw_coords(
            spot_det_hor[hkl_idx], spot_det_vert[hkl_idx],
            trans_opts, ny, nz)
        for fi in spot_frame_lists[hkl_idx]:
            if 0 <= fi < n_total:
                frame_to_spots.setdefault(fi, []).append(
                    (hkl_idx, rc, cc))

    # Read only small patches — never full frames
    for fi in sorted(frame_to_spots.keys()):
        for hkl_idx, rc, cc in frame_to_spots[fi]:
            r0 = max(0, rc - patch_half)
            r1 = min(nz, rc + patch_half + 1)
            c0 = max(0, cc - patch_half)
            c1 = min(ny, cc + patch_half + 1)
            if r0 < r1 and c0 < c1:
                patch = data[fi, r0:r1, c0:c1]
                intensities[hkl_idx] += float(np.sum(patch))

    store.close()
    return (scan_idx, intensities)


def build_sinogram(spots, scan_dirs, file_stem, padding,
                   patch_half=10, ome_half=2, skip_frame=1,
                   n_workers=4, trans_opts=None,
                   nr_pixels_y=2880, nr_pixels_z=2880):
    """Build sinogram from zarr zip files.

    Returns (sino, omegas) where sino has shape (nHKLs, nScans).
    """
    from multiprocessing import Pool

    n_hkls = len(spots)
    n_scans = len(scan_dirs)
    sino = np.zeros((n_hkls, n_scans), dtype=np.float64)
    omegas = np.array([s['omega'] for s in spots])

    spot_frames = []
    for s in spots:
        ob = s['omeBin']
        actual_bin = ob + skip_frame
        frames_needed = list(range(actual_bin - ome_half,
                                    actual_bin + ome_half + 1))
        spot_frames.append(frames_needed)

    spot_det_hor = [s['detHor'] for s in spots]
    spot_det_vert = [s['detVert'] for s in spots]

    if trans_opts is None:
        trans_opts = []

    print(f'\n  Building sinogram: {n_hkls} HKLs × {n_scans} scans')
    print(f'  Patch size: {2*patch_half+1}×{2*patch_half+1}×{2*ome_half+1}')
    if trans_opts:
        print(f'  ImTransOpt: {trans_opts}')
    print(f'  Using {n_workers} parallel workers')

    worker_args = []
    for scan_idx, layer_nr, scan_dir in scan_dirs:
        worker_args.append((
            scan_idx, layer_nr, scan_dir, file_stem, padding,
            spot_det_hor, spot_det_vert, spot_frames,
            patch_half, trans_opts, nr_pixels_y, nr_pixels_z
        ))

    t0 = time.time()
    completed = 0
    with Pool(processes=n_workers) as pool:
        for scan_idx, intensities in pool.imap_unordered(
                _process_single_scan, worker_args):
            completed += 1
            if intensities is not None:
                sino[:, scan_idx] = intensities
            else:
                layer_nr = scan_dirs[scan_idx][1]
                print(f'    WARNING: scan {scan_idx} (layer {layer_nr}) '
                      f'failed or missing')
            if completed % 10 == 0 or completed == n_scans:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f'    Progress: {completed}/{n_scans} scans '
                      f'[{elapsed:.1f}s, {rate:.1f} scans/s]')

    return sino, omegas


# ---------------------------------------------------------------------------
# Step 3: Tomographic reconstruction
# ---------------------------------------------------------------------------

def run_tomo_recon(sino, omegas, work_dir, n_cpus=4):
    """Run tomographic reconstruction on the sinogram.

    sino: shape (nHKLs, nScans) — each row is one HKL, each col is one scan
    omegas: 1D array of omega angles per HKL

    For tomo recon, we need sinograms shaped (nSlices, nThetas, detXdim).
    Here: nSlices=1, nThetas=nHKLs, detXdim=nScans.
    The 'thetas' are the omega angles of each HKL.
    """
    # Add MIDAS tomo module path
    midas_root = os.environ.get('MIDAS_ROOT',
                                 os.path.expanduser('~/opt/MIDAS'))
    tomo_dir = os.path.join(midas_root, 'TOMO')
    if tomo_dir not in sys.path:
        sys.path.insert(0, tomo_dir)

    from midas_tomo_python import run_tomo_from_sinos

    # sino is (nHKLs, nScans), tomo expects (nSlices, nThetas, detXdim)
    # = (1, nHKLs, nScans)
    sino_3d = sino[np.newaxis, :, :]  # (1, nHKLs, nScans)

    print(f'\n  Running tomo reconstruction...')
    print(f'    Sinogram shape for tomo: {sino_3d.shape}')
    print(f'    Theta range: [{omegas[0]:.2f}, {omegas[-1]:.2f}] degrees')

    os.makedirs(work_dir, exist_ok=True)

    recon = run_tomo_from_sinos(
        sino_3d,
        work_dir,
        omegas,
        shifts=0.0,
        filterNr=2,       # Hann filter
        doLog=0,           # intensities, not transmission
        extraPad=0,
        autoCentering=1,
        numCPUs=n_cpus,
        doCleanup=1,
    )
    # recon shape: (nrShifts=1, nSlices=1, reconDim, reconDim)
    print(f'    Reconstruction shape: {recon.shape}')
    return recon[0, 0, :, :]  # 2D (reconDim, reconDim)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def _save_sinogram_tif(sino, path):
    """Save a 2D sinogram as a 16-bit TIF, normalized to [0, 65535]."""
    from PIL import Image
    vis = sino.copy()
    if vis.max() > 0:
        vis = vis / vis.max() * 65535
    Image.fromarray(vis.astype(np.uint16)).save(path)


def save_outputs(sino, omegas, recon, spots, out_dir):
    """Save sinogram, reconstruction, and metadata."""
    os.makedirs(out_dir, exist_ok=True)

    sino_npy = os.path.join(out_dir, 'sinogram.npy')
    np.save(sino_npy, sino)
    print(f'  Saved sinogram: {sino_npy} shape={sino.shape}')

    try:
        sino_tif = os.path.join(out_dir, 'sinogram.tif')
        _save_sinogram_tif(sino, sino_tif)
        print(f'  Saved sinogram TIF: {sino_tif}')
    except ImportError:
        print('  (Pillow not available, TIF output skipped)')

    ome_file = os.path.join(out_dir, 'omegas.txt')
    np.savetxt(ome_file, omegas, fmt='%.6f')
    print(f'  Saved omegas: {ome_file}')

    recon_npy = os.path.join(out_dir, 'reconstruction.npy')
    np.save(recon_npy, recon)
    print(f'  Saved reconstruction: {recon_npy} shape={recon.shape}')

    try:
        from PIL import Image
        recon_vis = recon.copy()
        if recon_vis.max() > 0:
            recon_vis = recon_vis / recon_vis.max() * 65535
        recon_tif = os.path.join(out_dir, 'reconstruction.tif')
        Image.fromarray(recon_vis.astype(np.uint16)).save(recon_tif)
        print(f'  Saved reconstruction TIF: {recon_tif}')
    except ImportError:
        pass

    spot_csv = os.path.join(out_dir, 'spot_list.csv')
    with open(spot_csv, 'w') as f:
        f.write('HKL_idx,Omega,Eta,RingNr,Theta,DetHor,DetVert,OmeBin\n')
        for i, s in enumerate(spots):
            f.write(f'{i},{s["omega"]:.4f},{s["eta"]:.4f},{s["ringNr"]},'
                    f'{s["theta"]:.4f},{s["detHor"]:.2f},{s["detVert"]:.2f},'
                    f'{s["omeBin"]}\n')
    print(f'  Saved spot list: {spot_csv}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_sinogram_for_spots(spots, scan_dirs, args, params_dict):
    """Build a sinogram for the given list of spots.

    Encapsulates the data-source detection and sinogram construction logic
    so it can be reused for single-grain or per-grain multi-grain runs.

    Parameters
    ----------
    spots : list of dict
        Parsed spots (from parse_spot_matrix_gen).
    scan_dirs : list of tuple
        From get_scan_dirs.
    args : argparse.Namespace
        CLI arguments.
    params_dict : dict
        Derived parameters (file_stem, padding, ext, etc.).

    Returns
    -------
    (sino, omegas) : tuple of ndarray
    """
    p = params_dict  # shorthand

    if args.useInputAll:
        sino, omegas = build_sinogram_from_inputall(
            spots, scan_dirs,
            px_size=p['px_size'], omega_step=p['omega_step'],
            n_workers=args.nCPUs, pxtol=args.pxTol,
            ome_frame_tol=args.omeTol,
        )
    else:
        use_raw = p.get('use_raw', args.useRaw)
        if use_raw:
            sino, omegas = build_sinogram_raw(
                spots, scan_dirs, p['file_stem'], p['padding'], p['ext'],
                p['header_size'], p['bytes_per_pixel'], p['raw_folder'],
                patch_half=p['patch_half'], ome_half=p['ome_half'],
                skip_frame=p['skip_frame'], n_workers=args.nCPUs,
                trans_opts=p['trans_opts'],
                nr_pixels_y=p['nr_pixels_y'], nr_pixels_z=p['nr_pixels_z'],
                data_loc=p['data_loc'],
                dark_fn=p['dark_fn'], dark_loc=p['dark_loc'],
                pre_proc_thresh=p['pre_proc_thresh'],
            )
        else:
            sino, omegas = build_sinogram(
                spots, scan_dirs, p['file_stem'], p['padding'],
                patch_half=p['patch_half'], ome_half=p['ome_half'],
                skip_frame=p['skip_frame'], n_workers=args.nCPUs,
                trans_opts=p['trans_opts'],
                nr_pixels_y=p['nr_pixels_y'], nr_pixels_z=p['nr_pixels_z'],
            )
    return sino, omegas


def _process_single_grain(grain_spots, grain_id, out_dir, scan_dirs,
                          args, params_dict):
    """Build sinogram + tomo reconstruction for one grain.

    Saves results to  out_dir/grain_{grainID:04d}/  and returns the
    sub-directory path.
    """
    grain_dir = os.path.join(out_dir, f'grain_{grain_id:04d}')
    os.makedirs(grain_dir, exist_ok=True)

    print(f'\n  ── Grain {grain_id}: {len(grain_spots)} spots ──')
    sino, omegas = _build_sinogram_for_spots(
        grain_spots, scan_dirs, args, params_dict)

    print(f'    Sinogram shape: {sino.shape}, '
          f'non-zero: {np.count_nonzero(sino)}/{sino.size}, '
          f'max: {sino.max():.1f}')

    tomo_work = os.path.join(grain_dir, 'tomo_work')
    recon = run_tomo_recon(sino, omegas, tomo_work, n_cpus=args.nCPUs)

    save_outputs(sino, omegas, recon, grain_spots, grain_dir)
    return grain_dir


def main():
    parser = argparse.ArgumentParser(
        description='Forward simulation -> sinogram -> tomo reconstruction')
    parser.add_argument('--paramFile', required=True,
                        help='MIDAS parameter file (e.g. ps_sto_pf.txt)')
    parser.add_argument('--orient', required=False, default=None,
                        help='9 orientation matrix elements, space-separated '
                             '(single-grain mode)')
    parser.add_argument('--grainsFile', required=False, default=None,
                        help='Path to a Grains.csv file (multi-grain mode). '
                             'Written by ProcessGrains.')
    parser.add_argument('--h5File', required=False, default=None,
                        help='Path to a microstructure_pf.h5 file '
                             '(multi-grain mode). Written by pf_MIDAS.')
    parser.add_argument('--nCPUs', type=int, default=4,
                        help='Number of CPUs for ForwardSim and tomo')
    parser.add_argument('--patchSize', type=int, default=21,
                        help='Detector patch size (pixels, odd number)')
    parser.add_argument('--omePatch', type=int, default=5,
                        help='Number of omega frames to integrate')
    parser.add_argument('--outDir', default='fwd_sino_output',
                        help='Output directory')
    parser.add_argument('--skipFwdSim', action='store_true',
                        help='Skip forward simulation (use existing '
                             'SpotMatrixGen.csv)')
    parser.add_argument('--useInputAll', action='store_true',
                        help='Use GrainRadius from InputAllExtraInfoFittingAll.csv '
                             'in each scan dir instead of zarr patch extraction. '
                             'Matches predicted spots by (YLab, ZLab, Omega, RingNr).')
    parser.add_argument('--useRaw', action='store_true',
                        help='Force reading raw detector files instead of '
                             'zarr zip archives.')
    parser.add_argument('--pxTol', type=int, default=4,
                        help='Pixel tolerance for InputAll matching (default: 4)')
    parser.add_argument('--omeTol', type=int, default=2,
                        help='Omega tolerance in frames for InputAll matching '
                             '(default: 2)')
    parser.add_argument('--dataLoc', default=None,
                        help='HDF5 dataset path for image data '
                             '(default: exchange/data)')
    parser.add_argument('--darkLoc', default=None,
                        help='HDF5 dataset path for dark data '
                             '(default: exchange/dark)')
    parser.add_argument('--preProcThresh', type=int, default=None,
                        help='Pre-processing threshold above dark mean. '
                             '-1 = simple dark subtraction')
    parser.add_argument('--noDark', action='store_true',
                        help='Disable dark correction entirely.')
    args = parser.parse_args()

    # ── Validate orientation source ─────────────────────────────────────
    n_orient_src = sum([
        args.orient is not None,
        args.grainsFile is not None,
        args.h5File is not None,
    ])
    multi_grain = args.grainsFile is not None or args.h5File is not None

    if n_orient_src > 1:
        sys.exit('ERROR: --orient, --grainsFile, and --h5File are mutually '
                 'exclusive.  Provide exactly one.')
    if n_orient_src == 0 and not args.skipFwdSim:
        sys.exit('ERROR: provide one of --orient, --grainsFile, or --h5File '
                 '(or use --skipFwdSim).')

    base_dir = os.getcwd()
    param_file = os.path.join(base_dir, args.paramFile)
    if not os.path.isfile(param_file):
        sys.exit(f'ERROR: parameter file not found: {param_file}')

    # ── Load orientation(s) ─────────────────────────────────────────────
    grains_list = None  # populated for multi-grain modes
    orient_vals = None  # populated for single-orient mode

    if args.orient:
        orient_vals = [float(x) for x in args.orient.split()]
        if len(orient_vals) != 9:
            sys.exit(f'ERROR: expected 9 orientation values, '
                     f'got {len(orient_vals)}')
        print(f'\nOrientation matrix:')
        for i in range(3):
            print(f'  [{orient_vals[3*i]:.6f}  {orient_vals[3*i+1]:.6f}  '
                  f'{orient_vals[3*i+2]:.6f}]')

    elif args.h5File:
        h5_path = args.h5File
        if not os.path.isabs(h5_path):
            h5_path = os.path.join(base_dir, h5_path)
        if not os.path.isfile(h5_path):
            sys.exit(f'ERROR: H5 file not found: {h5_path}')
        print(f'\nReading orientations from H5: {h5_path}')
        grains_list = read_h5_orientations(h5_path)
        if not grains_list:
            sys.exit('ERROR: no orientations found in H5 file')

    elif args.grainsFile:
        grains_path = args.grainsFile
        if not os.path.isabs(grains_path):
            grains_path = os.path.join(base_dir, grains_path)
        if not os.path.isfile(grains_path):
            sys.exit(f'ERROR: Grains.csv not found: {grains_path}')
        print(f'\nReading orientations from Grains.csv: {grains_path}')
        grains_list = parse_grains_csv_file(grains_path)
        if not grains_list:
            sys.exit('ERROR: no grains found in Grains.csv')

    # Parse param file
    params = parse_param_file(param_file)
    file_stem = params.get('FileStem', 'data')
    start_file_nr = int(params.get('StartFileNrFirstLayer', '1'))
    n_scans = int(params.get('nScans', '1'))
    scan_step = int(params.get('ScanStep',
                               params.get('NrFilesPerSweep', '1')))
    padding = int(params.get('Padding', '6'))
    skip_frame = int(params.get('SkipFrame', '0'))
    omega_start = float(params.get('OmegaStart', '-180'))
    omega_step = float(params.get('OmegaStep', '0.25'))
    lattice_str = params.get('LatticeConstant',
                              '3.9091 3.9091 3.9091 90 90 90')

    nr_pixels_y = int(params.get('NrPixelsY',
                                  params.get('NrPixels', '2880')))
    nr_pixels_z = int(params.get('NrPixelsZ',
                                  params.get('NrPixels', '2880')))
    px_size = float(params.get('px', '150'))
    im_trans_opt_strs = params.get('ImTransOpt', [])
    trans_opts = [int(x) for x in im_trans_opt_strs]

    raw_folder = params.get('RawFolder', '')
    ext = params.get('Ext', '')
    header_size = int(params.get('HeadSize', '8192'))
    pixel_value = int(params.get('PixelValue', '2'))
    bytes_per_pixel = 4 if pixel_value == 4 else 2

    dark_fn = '' if args.noDark else params.get('Dark', '')
    data_loc = args.dataLoc if args.dataLoc else params.get('dataLoc',
                                                             'exchange/data')
    dark_loc = args.darkLoc if args.darkLoc else params.get('darkLoc',
                                                             'exchange/dark')
    pre_proc_thresh = (args.preProcThresh if args.preProcThresh is not None
                       else int(params.get('preProcThresh', '-1')))

    patch_half = args.patchSize // 2
    ome_half = args.omePatch // 2

    print(f'\nDataset parameters:')
    print(f'  FileStem: {file_stem}')
    print(f'  StartFileNr: {start_file_nr}, nScans: {n_scans}, '
          f'ScanStep: {scan_step}')
    print(f'  OmegaStart: {omega_start}, OmegaStep: {omega_step}')
    print(f'  SkipFrame: {skip_frame}, Padding: {padding}')
    print(f'  NrPixels: {nr_pixels_y}x{nr_pixels_z}')
    if trans_opts:
        print(f'  ImTransOpt: {trans_opts}')
    if raw_folder:
        print(f'  RawFolder: {raw_folder}')
    if dark_fn:
        print(f'  Dark: {dark_fn}')

    scan_dirs = get_scan_dirs(base_dir, start_file_nr, n_scans, scan_step)

    # ── Detect data source (zip vs raw) once ────────────────────────────
    use_raw = args.useRaw
    if not use_raw and not args.useInputAll:
        first_layer_nr = scan_dirs[0][1]
        first_scan_dir = scan_dirs[0][2]
        zip_name = f'{file_stem}_{first_layer_nr:0{padding}d}.MIDAS.zip'
        zip_path = os.path.join(first_scan_dir, zip_name)
        if not os.path.isfile(zip_path):
            use_raw = True
            print(f'  Auto-detect: no zip file, switching to raw mode')
    if use_raw and not raw_folder and not args.useInputAll:
        sys.exit('ERROR: RawFolder not set and no zarr zip files.')

    # Collect derived params into a dict for helper functions
    params_dict = dict(
        file_stem=file_stem, padding=padding, ext=ext,
        header_size=header_size, bytes_per_pixel=bytes_per_pixel,
        raw_folder=raw_folder, patch_half=patch_half, ome_half=ome_half,
        skip_frame=skip_frame, trans_opts=trans_opts,
        nr_pixels_y=nr_pixels_y, nr_pixels_z=nr_pixels_z,
        data_loc=data_loc, dark_fn=dark_fn, dark_loc=dark_loc,
        pre_proc_thresh=pre_proc_thresh, px_size=px_size,
        omega_step=omega_step, use_raw=use_raw,
    )

    out_dir = os.path.join(base_dir, args.outDir)
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Forward simulation ──────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 1: Forward Simulation')
    print('='*60)

    if not args.skipFwdSim:
        fwd_dir = os.path.join(out_dir, 'fwd_sim')
        os.makedirs(fwd_dir, exist_ok=True)

        # Copy hkls.csv
        hkls_src = os.path.join(base_dir, 'hkls.csv')
        if not os.path.isfile(hkls_src):
            hkls_src = os.path.join(scan_dirs[0][2], 'hkls.csv')
        if os.path.isfile(hkls_src):
            shutil.copy2(hkls_src, os.path.join(fwd_dir, 'hkls.csv'))
        else:
            print('  WARNING: hkls.csv not found')

        with open(os.path.join(fwd_dir, 'positions.csv'), 'w') as f:
            f.write('0.0\n')

        grains_csv = os.path.join(fwd_dir, 'Grains.csv')

        if multi_grain:
            # Multi-grain: write all grains into one Grains.csv
            create_multi_grains_csv(grains_list, lattice_str, grains_csv)
        else:
            # Single grain
            create_grains_csv(orient_vals, lattice_str, grains_csv)

        fwd_param = os.path.join(fwd_dir, 'ps_fwd.txt')
        create_fwd_param_file(param_file, 'Grains.csv', fwd_param)

        spot_file = run_forward_simulation('ps_fwd.txt', args.nCPUs, fwd_dir)
        shutil.copy2(spot_file, os.path.join(out_dir, 'SpotMatrixGen.csv'))
    else:
        spot_file = os.path.join(out_dir, 'SpotMatrixGen.csv')
        if not os.path.isfile(spot_file):
            spot_file = os.path.join(base_dir, 'SpotMatrixGen.csv')
        if not os.path.isfile(spot_file):
            sys.exit(f'ERROR: SpotMatrixGen.csv not found')
        print(f'  Using existing SpotMatrixGen.csv: {spot_file}')

    all_spots = parse_spot_matrix_gen(spot_file)

    # ── Determine grain IDs present ─────────────────────────────────────
    grain_ids = sorted(set(s['grainID'] for s in all_spots))
    n_grain_ids = len(grain_ids)
    print(f'  Found spots for {n_grain_ids} grain(s): {grain_ids}')

    # ── Step 2 + 3: Build sinograms + Tomo (per grain) ──────────────────
    if n_grain_ids == 1 and not multi_grain:
        # ── Single-grain path (original behaviour) ──────────────────────
        print('\n' + '='*60)
        print('STEP 2: Build Sinogram')
        print('='*60)

        sino, omegas = _build_sinogram_for_spots(
            all_spots, scan_dirs, args, params_dict)

        print(f'\n  Sinogram shape: {sino.shape}')
        print(f'  Non-zero cells: {np.count_nonzero(sino)} / {sino.size} '
              f'({100*np.count_nonzero(sino)/sino.size:.1f}%)')
        print(f'  Max intensity: {sino.max():.1f}')

        print('\n' + '='*60)
        print('STEP 3: Tomographic Reconstruction')
        print('='*60)
        tomo_work = os.path.join(out_dir, 'tomo_work')
        recon = run_tomo_recon(sino, omegas, tomo_work, n_cpus=args.nCPUs)

        print('\n' + '='*60)
        print('SAVING OUTPUTS')
        print('='*60)
        save_outputs(sino, omegas, recon, all_spots, out_dir)
        print(f'\nAll done! Results in: {out_dir}')

    else:
        # ── Multi-grain path ────────────────────────────────────────────
        print('\n' + '='*60)
        print(f'MULTI-GRAIN MODE: processing {n_grain_ids} grain(s)')
        print('='*60)

        grain_dirs = []
        for gid in grain_ids:
            grain_spots = [s for s in all_spots if s['grainID'] == gid]
            if not grain_spots:
                print(f'  WARNING: grain {gid} has 0 spots, skipping')
                continue
            gdir = _process_single_grain(
                grain_spots, gid, out_dir, scan_dirs, args, params_dict)
            grain_dirs.append((gid, gdir))

        # ── Summary ────────────────────────────────────────────────────
        print('\n' + '='*60)
        print('MULTI-GRAIN SUMMARY')
        print('='*60)
        for gid, gdir in grain_dirs:
            print(f'  grain_{gid:04d}/  →  {gdir}')
        print(f'\nAll done! {len(grain_dirs)} grain(s) processed.')
        print(f'Results root: {out_dir}')


if __name__ == '__main__':
    main()

