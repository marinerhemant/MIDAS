#!/usr/bin/env python
"""
Forward Simulation → Sinogram → Tomo Reconstruction

Given an orientation matrix and a PF-HEDM dataset, this script:
  1. Runs ForwardSimulationCompressed to predict diffraction spots
  2. Searches per-scan MIDAS zip files and extracts intensity patches
  3. Builds an omega-sorted sinogram
  4. Runs tomographic reconstruction via midas_tomo_python

Usage:
  cd <doIndexing0 directory>
  python forward_sim_sinogram.py \
      --paramFile ps_sto_pf.txt \
      --orient "0.695618 0.661920 0.279243 -0.479059 0.717044 -0.506310 -0.535367 0.218425 0.815888" \
      --nCPUs 4 \
      --outDir fwd_sino_output

Author: Auto-generated for hruszkewycz_mar26 analysis
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


def create_fwd_param_file(orig_param_file, grains_csv, out_param_file):
    """Create a modified param file for ForwardSimulationCompressed.

    - Points InputFile to the Grains.csv
    - Sets WriteSpots 1, WriteImage 0
    - Keeps nScans as 1 (we only need spot positions, not per-scan sims)
    """
    with open(orig_param_file) as f:
        lines = f.readlines()

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
            elif key == 'InputFile':
                f.write(f'InputFile {grains_csv}\n')
                wrote_input = True
            else:
                f.write(line)
        # Add any missing keys
        if not wrote_input:
            f.write(f'InputFile {grains_csv}\n')
        if not wrote_spots:
            f.write('WriteSpots 1\n')
        if not wrote_image:
            f.write('WriteImage 0\n')
        if not wrote_nscans:
            f.write('nScans 1\n')
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


# ---------------------------------------------------------------------------
# Step 2: Extract intensity from zip files to build sinogram
# ---------------------------------------------------------------------------

def read_frame_from_zarr(zarr_path, frame_idx, n_frames):
    """Open a MIDAS zarr zip and read a single frame as numpy array."""
    import zarr
    store = zarr.storage.ZipStore(zarr_path, mode='r')
    zg = zarr.open_group(store, mode='r')
    data = zg['exchange/data']
    # data shape: (nFrames, nPixelsZ, nPixelsY)
    if frame_idx < 0 or frame_idx >= data.shape[0]:
        return None
    frame = np.array(data[frame_idx, :, :], dtype=np.float64)
    store.close()
    return frame


def read_frames_from_zarr(zarr_path, frame_indices):
    """Read multiple frames from a MIDAS zarr zip efficiently.

    Returns dict: frame_idx -> 2D numpy array
    """
    import zarr
    store = zarr.storage.ZipStore(zarr_path, mode='r')
    zg = zarr.open_group(store, mode='r')
    data = zg['exchange/data']
    n_total = data.shape[0]

    result = {}
    for fi in frame_indices:
        if 0 <= fi < n_total:
            result[fi] = np.array(data[fi, :, :], dtype=np.float64)
    store.close()
    return result


def extract_patch_intensity(frames_dict, center_y, center_z,
                             frame_indices, patch_half=10):
    """Sum intensity in a (2*patch_half+1) × (2*patch_half+1) × nFrames patch.

    center_y, center_z are detector pixel coordinates (float).
    """
    cy = int(round(center_y))
    cz = int(round(center_z))
    total = 0.0
    for fi in frame_indices:
        if fi not in frames_dict:
            continue
        frame = frames_dict[fi]
        nz, ny = frame.shape
        y0 = max(0, cy - patch_half)
        y1 = min(ny, cy + patch_half + 1)
        z0 = max(0, cz - patch_half)
        z1 = min(nz, cz + patch_half + 1)
        if y0 < y1 and z0 < z1:
            total += np.sum(frame[z0:z1, y0:y1])
    return total


def build_sinogram(spots, scan_dirs, file_stem, padding,
                   patch_half=10, ome_half=2, skip_frame=1):
    """Build sinogram: shape (nHKLs, nScans).

    For each scan's zip file, read frames around each spot's omeBin
    and extract a patch of intensity.

    spots: list of spot dicts, already sorted by omega
    scan_dirs: list of (scanIdx, layerNr, dirPath)
    """
    n_hkls = len(spots)
    n_scans = len(scan_dirs)
    sino = np.zeros((n_hkls, n_scans), dtype=np.float64)
    omegas = np.array([s['omega'] for s in spots])

    # Pre-compute which frames we need per spot
    spot_frames = []
    for s in spots:
        ob = s['omeBin']
        # Account for skipFrame: zarr data includes the skip frame
        # The actual data frame in zarr = omeBin + skipFrame
        actual_bin = ob + skip_frame
        frames_needed = list(range(actual_bin - ome_half,
                                    actual_bin + ome_half + 1))
        spot_frames.append(frames_needed)

    # All unique frames needed across all spots
    all_frames = set()
    for ff in spot_frames:
        all_frames.update(ff)
    all_frames = sorted(all_frames)

    print(f'\n  Building sinogram: {n_hkls} HKLs × {n_scans} scans')
    print(f'  Patch size: {2*patch_half+1}×{2*patch_half+1}×{2*ome_half+1}')
    print(f'  Total unique frames per scan: {len(all_frames)}')

    t0 = time.time()
    for scan_idx, layer_nr, scan_dir in scan_dirs:
        # Find the zip file
        zip_name = f'{file_stem}_{layer_nr:0{padding}d}.MIDAS.zip'
        zip_path = os.path.join(scan_dir, zip_name)
        if not os.path.isfile(zip_path):
            print(f'    WARNING: {zip_path} not found, skipping scan {scan_idx}')
            continue

        # Read all needed frames at once
        frames_dict = read_frames_from_zarr(zip_path, all_frames)
        if not frames_dict:
            print(f'    WARNING: No frames read from {zip_path}')
            continue

        # Extract intensity for each spot
        for hkl_idx, s in enumerate(spots):
            intensity = extract_patch_intensity(
                frames_dict,
                center_y=s['detHor'],
                center_z=s['detVert'],
                frame_indices=spot_frames[hkl_idx],
                patch_half=patch_half
            )
            sino[hkl_idx, scan_idx] = intensity

        elapsed = time.time() - t0
        rate = (scan_idx + 1) / elapsed if elapsed > 0 else 0
        print(f'    Scan {scan_idx+1}/{n_scans} (layer {layer_nr}) done '
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

def save_outputs(sino, omegas, recon, spots, out_dir):
    """Save sinogram, reconstruction, and metadata."""
    os.makedirs(out_dir, exist_ok=True)

    # Save sinogram as npy
    sino_npy = os.path.join(out_dir, 'sinogram.npy')
    np.save(sino_npy, sino)
    print(f'  Saved sinogram: {sino_npy} shape={sino.shape}')

    # Save sinogram as TIF
    try:
        from PIL import Image
        # Normalize for visualization
        sino_vis = sino.copy()
        if sino_vis.max() > 0:
            sino_vis = sino_vis / sino_vis.max() * 65535
        sino_tif = os.path.join(out_dir, 'sinogram.tif')
        Image.fromarray(sino_vis.astype(np.uint16)).save(sino_tif)
        print(f'  Saved sinogram TIF: {sino_tif}')
    except ImportError:
        print('  (Pillow not available, TIF output skipped)')

    # Save omegas
    ome_file = os.path.join(out_dir, 'omegas.txt')
    np.savetxt(ome_file, omegas, fmt='%.6f')
    print(f'  Saved omegas: {ome_file}')

    # Save reconstruction
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
        print('  (Pillow not available, TIF output skipped)')

    # Save spot list CSV
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

def main():
    parser = argparse.ArgumentParser(
        description='Forward simulation → sinogram → tomo reconstruction')
    parser.add_argument('--paramFile', required=True,
                        help='MIDAS parameter file (e.g. ps_sto_pf.txt)')
    parser.add_argument('--orient', required=True,
                        help='9 orientation matrix elements, space-separated')
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
    args = parser.parse_args()

    base_dir = os.getcwd()
    param_file = os.path.join(base_dir, args.paramFile)
    if not os.path.isfile(param_file):
        sys.exit(f'ERROR: parameter file not found: {param_file}')

    # Parse orientation matrix
    orient_vals = [float(x) for x in args.orient.split()]
    if len(orient_vals) != 9:
        sys.exit(f'ERROR: expected 9 orientation values, got {len(orient_vals)}')
    print(f'\nOrientation matrix:')
    for i in range(3):
        print(f'  [{orient_vals[3*i]:.6f}  {orient_vals[3*i+1]:.6f}  '
              f'{orient_vals[3*i+2]:.6f}]')

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

    print(f'\nDataset parameters:')
    print(f'  FileStem: {file_stem}')
    print(f'  StartFileNr: {start_file_nr}, nScans: {n_scans}, '
          f'ScanStep: {scan_step}')
    print(f'  OmegaStart: {omega_start}, OmegaStep: {omega_step}')
    print(f'  SkipFrame: {skip_frame}, Padding: {padding}')

    patch_half = args.patchSize // 2
    ome_half = args.omePatch // 2

    # Get scan directories
    scan_dirs = get_scan_dirs(base_dir, start_file_nr, n_scans, scan_step)

    # ── Step 1: Forward simulation ──────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 1: Forward Simulation')
    print('='*60)

    out_dir = os.path.join(base_dir, args.outDir)
    os.makedirs(out_dir, exist_ok=True)

    if not args.skipFwdSim:
        # Create a working directory for the forward simulation
        fwd_dir = os.path.join(out_dir, 'fwd_sim')
        os.makedirs(fwd_dir, exist_ok=True)

        # Copy hkls.csv if it exists (ForwardSim needs it)
        hkls_src = os.path.join(base_dir, 'hkls.csv')
        if not os.path.isfile(hkls_src):
            # Try from a scan dir
            hkls_src = os.path.join(scan_dirs[0][2], 'hkls.csv')
        if os.path.isfile(hkls_src):
            shutil.copy2(hkls_src, os.path.join(fwd_dir, 'hkls.csv'))
        else:
            print('  WARNING: hkls.csv not found, GetHKLList will generate it')

        # Create positions.csv (single position at 0)
        with open(os.path.join(fwd_dir, 'positions.csv'), 'w') as f:
            f.write('0.0\n')

        # Create Grains.csv
        grains_csv = os.path.join(fwd_dir, 'Grains.csv')
        create_grains_csv(orient_vals, lattice_str, grains_csv)

        # Create modified parameter file
        fwd_param = os.path.join(fwd_dir, 'ps_fwd.txt')
        create_fwd_param_file(param_file, 'Grains.csv', fwd_param)

        # Run forward simulation
        spot_file = run_forward_simulation('ps_fwd.txt', args.nCPUs, fwd_dir)
        # Copy SpotMatrixGen.csv to output
        shutil.copy2(spot_file, os.path.join(out_dir, 'SpotMatrixGen.csv'))
    else:
        spot_file = os.path.join(out_dir, 'SpotMatrixGen.csv')
        if not os.path.isfile(spot_file):
            spot_file = os.path.join(base_dir, 'SpotMatrixGen.csv')
        if not os.path.isfile(spot_file):
            sys.exit(f'ERROR: SpotMatrixGen.csv not found')
        print(f'  Using existing SpotMatrixGen.csv: {spot_file}')

    # Parse spots
    spots = parse_spot_matrix_gen(spot_file)

    # ── Step 2: Build sinogram ──────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 2: Build Sinogram from Zip Files')
    print('='*60)

    sino, omegas = build_sinogram(
        spots, scan_dirs, file_stem, padding,
        patch_half=patch_half, ome_half=ome_half,
        skip_frame=skip_frame
    )

    print(f'\n  Sinogram shape: {sino.shape}')
    print(f'  Non-zero cells: {np.count_nonzero(sino)} / {sino.size} '
          f'({100*np.count_nonzero(sino)/sino.size:.1f}%)')
    print(f'  Max intensity: {sino.max():.1f}')

    # ── Step 3: Tomo reconstruction ─────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 3: Tomographic Reconstruction')
    print('='*60)

    tomo_work = os.path.join(out_dir, 'tomo_work')
    recon = run_tomo_recon(sino, omegas, tomo_work, n_cpus=args.nCPUs)

    # ── Save outputs ────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('SAVING OUTPUTS')
    print('='*60)

    save_outputs(sino, omegas, recon, spots, out_dir)

    print(f'\nAll done! Results in: {out_dir}')


if __name__ == '__main__':
    main()
