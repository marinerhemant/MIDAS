"""Automated test for pf-HEDM reconstruction pipeline.

Generates a synthetic microstructure, simulates multi-scan diffraction data
using ForwardSimulationCompressed, organizes output into the folder layout
expected by pf_MIDAS.py, and runs the full reconstruction pipeline.

Supports GPU parity testing: run with --gpu to compare GPU vs OMP output.

Usage:
    python tests/test_pf_hedm.py -nCPUs 8
    python tests/test_pf_hedm.py -nCPUs 8 --doTomo 1
    python tests/test_pf_hedm.py -nCPUs 8 --gpu --no-cleanup
"""

import argparse
import os
import sys
import subprocess
import shutil
import struct
import numpy as np
import zarr
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve MIDAS home relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent

sys.path.insert(0, str(MIDAS_HOME / 'utils'))

from calcMiso import (
    Euler2OrientMat, GetMisOrientationAngleOM, OrientMat2Euler
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FSTEM = 'pfhedm'
NGRAINS = 5
SIZE_UM = 50.0      # microstructure domain: ±25 µm
STEP_UM = 1.0       # EBSD voxel size in µm (fine GT resolution)
NSCANS = 15         # 15 scan positions → 15×15 = 225 reconstruction voxels
SCAN_RANGE_UM = 70.0  # scan range ±35 µm; adds ~1 empty voxel border outside domain
SEED = 42
PADDING = 6
# BeamSize = scan step = SCAN_RANGE_UM / (NSCANS - 1) = 5 µm
BEAMSIZE = SCAN_RANGE_UM / (NSCANS - 1)


def parse_args():
    parser = argparse.ArgumentParser(description='pf-HEDM reconstruction test')
    parser.add_argument('-nCPUs', type=int, default=8,
                        help='Number of CPUs for simulation and pipeline')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Skip cleanup of generated files after the test')
    parser.add_argument('--doTomo', type=int, default=0,
                        help='0=skip tomography (faster, default), 1=full tomo pipeline')
    parser.add_argument('--gpu', action='store_true',
                        help='Run GPU parity test: run OMP first, then GPU, compare results')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip generation and forward simulation. '
                             'Re-use existing zip data, still run peak search '
                             '(requires --no-cleanup from a prior run)')
    parser.add_argument('--skip-peaksearch', action='store_true',
                        help='Skip generation, forward sim, and peak search. '
                             'Start from indexing using existing data '
                             '(requires --no-cleanup from a prior run)')
    parser.add_argument('--compare-seeded', action='store_true',
                        help='After the normal pipeline, re-run indexer+refiner '
                             'seeded with GT orientations and compare results')
    return parser.parse_args()


def generate_microstructure(work_dir):
    """Generate synthetic microstructure using the utility script."""
    print('\n' + '='*70)
    print('  Step 1: Generating synthetic microstructure')
    print('='*70)

    cmd = [
        sys.executable,
        str(MIDAS_HOME / 'utils' / 'generate_pfhedm_microstructure.py'),
        '--outdir', str(work_dir),
        '--ngrains', str(NGRAINS),
        '--size', str(SIZE_UM),
        '--step', str(STEP_UM),
        '--nscans', str(NSCANS),
        '--beamsize', str(BEAMSIZE),
        '--scan_size', str(SCAN_RANGE_UM),
        '--seed', str(SEED),
    ]
    print(f'  Command: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=str(work_dir))
    if result.returncode != 0:
        print('ERROR: Microstructure generation failed.')
        sys.exit(1)

    # Verify outputs
    for fname in ['microstructure.ebsd', 'positions.csv',
                   'Parameters_pfhedm.txt', 'orientation_map.png']:
        fpath = work_dir / fname
        if not fpath.exists():
            print(f'ERROR: Expected output file not found: {fpath}')
            sys.exit(1)
    print('  Microstructure generation successful.')


def run_forward_simulation(work_dir, nCPUs):
    """Run ForwardSimulationCompressed to produce multi-scan zip files."""
    print('\n' + '='*70)
    print('  Step 2: Running forward simulation')
    print('='*70)

    bin_path = MIDAS_HOME / 'FF_HEDM' / 'bin' / 'ForwardSimulationCompressed'
    if not bin_path.exists():
        print(f'ERROR: {bin_path} not found. Please compile first.')
        sys.exit(1)

    param_file = work_dir / 'Parameters_pfhedm.txt'
    cmd = [str(bin_path), str(param_file), str(nCPUs)]
    print(f'  Command: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=str(work_dir))
    if result.returncode != 0:
        print('ERROR: ForwardSimulationCompressed failed.')
        sys.exit(1)

    # Verify zip files
    for scanNr in range(NSCANS):
        zip_name = work_dir / f'{FSTEM}_sim_scanNr_{scanNr}.zip'
        if not zip_name.exists():
            print(f'ERROR: Expected zip file not found: {zip_name}')
            sys.exit(1)
        print(f'  Scan {scanNr}: {zip_name.name} ({zip_name.stat().st_size / 1024:.0f} KB)')

    print(f'  Forward simulation produced {NSCANS} zip files.')


def enrich_zarr_metadata(zip_path, params_dict):
    """Inject analysis/measurement metadata into a Zarr zip.

    This mimics what ffGenerateZipRefactor does, providing the minimum
    metadata needed by the pf_MIDAS.py peak-search pipeline.
    """
    with zarr.ZipStore(str(zip_path), mode='a') as store:
        try:
            zRoot = zarr.group(store=store)
        except zarr.errors.GroupNotFoundError:
            zRoot = zarr.group(store=store, overwrite=True)

        # Ensure base structure
        sp_ana = zRoot.require_group('analysis/process/analysis_parameters')
        sp_pro = zRoot.require_group('measurement/process/scan_parameters')

        # datatype from data array
        data_dtype = str(zRoot['exchange/data'].dtype)
        dtype_map = {
            'uint16': 'uint16', 'int32': 'int32', 'uint32': 'uint32',
            'float32': 'float32', 'float64': 'float64',
        }
        dtype_str = dtype_map.get(data_dtype, data_dtype)
        sp_pro.create_dataset('datatype', data=np.bytes_(dtype_str.encode('UTF-8')),
                              overwrite=True)

        # Write analysis parameters using ffGenerateZipRefactor
        sys.path.insert(0, str(MIDAS_HOME / 'utils'))
        from ffGenerateZipRefactor import write_analysis_parameters
        z_groups = {
            'sp_pro_analysis': sp_ana,
            'sp_pro_meas': sp_pro,
        }
        write_analysis_parameters(z_groups, params_dict)


def parse_parameter_file(filepath):
    """Parse a MIDAS parameter file into a dict (same logic as test_ff_hedm)."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line_nc = line.split('#', 1)[0].strip()
            if not line_nc:
                continue
            parts = line_nc.split()
            if not parts:
                continue
            key, values = parts[0], parts[1:]
            processed = []
            for v in values:
                try:
                    processed.append(int(v))
                except ValueError:
                    try:
                        processed.append(float(v))
                    except ValueError:
                        processed.append(v)
            final = processed if len(processed) > 1 else (processed[0] if processed else '')
            if key not in params:
                params[key] = final
            else:
                if not isinstance(params[key], list):
                    params[key] = [params[key]]
                params[key].append(final)
    return params


def organize_for_pf_pipeline(work_dir, nCPUs):
    """Reorganize ForwardSimulationCompressed output into pf_MIDAS.py layout.

    pf_MIDAS expects:
      {topdir}/{startNrFirstLayer + (layerNr-1)*nrFilesPerSweep}/
        {fStem}_{startNr zero-padded to Padding}.MIDAS.zip

    With StartFileNrFirstLayer=1, NrFilesPerSweep=1, StartNr=1, Padding=6:
      1/pfhedm_000001.MIDAS.zip   (scan 0 = layer 1)
      2/pfhedm_000002.MIDAS.zip   (scan 1 = layer 2)
      ...
    """
    print('\n' + '='*70)
    print('  Step 3: Organizing files for pf_MIDAS.py')
    print('='*70)

    # Use ffGenerateZipRefactor's parser for proper multi-line param handling
    # (e.g. RingThresh creates [[1,10],[2,10],...] instead of ragged list)
    from ffGenerateZipRefactor import parse_parameter_file as parse_params_for_zarr
    params = parse_params_for_zarr(str(work_dir / 'Parameters_pfhedm.txt'))

    for scanNr in range(NSCANS):
        layerNr = scanNr + 1  # 1-indexed
        folder_name = str(layerNr)
        layer_dir = work_dir / folder_name

        # Create layer directory structure
        layer_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / 'Temp').mkdir(exist_ok=True)
        (layer_dir / 'output').mkdir(exist_ok=True)
        (layer_dir / 'midas_log').mkdir(exist_ok=True)

        # Source zip from ForwardSimulationCompressed
        src_zip = work_dir / f'{FSTEM}_sim_scanNr_{scanNr}.zip'

        # Target zip name: {fStem}_{thisStartNr padded}.MIDAS.zip
        # thisStartNr = startNrFirstLayer + (layerNr - 1) * nrFilesPerSweep
        startNrFirstLayer = params.get('StartFileNrFirstLayer', 1)
        nrFilesPerSweep = params.get('NrFilesPerSweep', 1)
        thisStartNr = startNrFirstLayer + (layerNr - 1) * nrFilesPerSweep
        padded = str(thisStartNr).zfill(PADDING)
        dst_zip = layer_dir / f'{FSTEM}_{padded}.MIDAS.zip'

        shutil.copy2(str(src_zip), str(dst_zip))
        print(f'  Scan {scanNr} → {folder_name}/{dst_zip.name}')

        # Enrich metadata
        enrich_zarr_metadata(dst_zip, params)

    # Create top-level output/Results dirs
    (work_dir / 'Output').mkdir(exist_ok=True)
    (work_dir / 'Results').mkdir(exist_ok=True)
    (work_dir / 'output').mkdir(exist_ok=True)

    print('  File organization and metadata enrichment complete.')


def run_pf_pipeline(work_dir, nCPUs, doTomo=0, useGPU=0, skip_peaksearch=False):
    """Run pf_MIDAS.py reconstruction pipeline."""
    print('\n' + '='*70)
    print(f'  Step 4: Running pf_MIDAS.py (doTomo={doTomo}, useGPU={useGPU}, skip_peaksearch={skip_peaksearch})')
    print('='*70)

    pf_script = MIDAS_HOME / 'FF_HEDM' / 'workflows' / 'pf_MIDAS.py'
    if not pf_script.exists():
        print(f'ERROR: {pf_script} not found.')
        sys.exit(1)

    do_peak = '0' if skip_peaksearch else '1'
    cmd = [
        sys.executable, str(pf_script),
        '-paramFile', 'Parameters_pfhedm.txt',
        '-nCPUs', str(nCPUs),
        '-nCPUsLocal', str(nCPUs),
        '-convertFiles', '0',
        '-doPeakSearch', do_peak,
        '-doTomo', str(doTomo),
        '-useGPU', str(useGPU),
        '-machineName', 'local',
        '-resultDir', str(work_dir),
    ]
    print(f'  Command: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=str(work_dir))

    if result.returncode != 0:
        print(f'WARNING: pf_MIDAS.py exited with return code {result.returncode}')
        # Don't exit — still try to validate partial output
    else:
        print('  pf_MIDAS.py completed successfully.')


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
def load_ground_truth(work_dir):
    """Load ground truth orientations from microstructure.ebsd.

    Returns dict with:
      'eulers_unique': ndarray (nGrains, 3) unique Euler angles in degrees
      'orient_mats':   list of 9-element OM arrays per grain
      'voxel_data':    ndarray (nVoxels, 6) with [x, y, z, e1, e2, e3]
      'nGrains':       int
      'sgnum':         int (from parameter file)
    """
    ebsd_path = work_dir / 'microstructure.ebsd'
    data = np.loadtxt(str(ebsd_path), skiprows=2)
    eulers_unique = np.unique(data[:, 3:6], axis=0)

    orient_mats = []
    for e in eulers_unique:
        om = Euler2OrientMat(np.radians(e))
        orient_mats.append(om)

    # Parse space group from parameter file
    sgnum = 225
    param_file = work_dir / 'Parameters_pfhedm.txt'
    if param_file.exists():
        with open(str(param_file)) as f:
            for line in f:
                if line.startswith('SpaceGroup'):
                    sgnum = int(line.split()[1])
                    break

    print(f'  Ground truth: {len(eulers_unique)} grains, SG={sgnum}')
    for i, e in enumerate(eulers_unique):
        print(f'    GT {i}: Euler=({e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f})°')

    return {
        'eulers_unique': eulers_unique,
        'orient_mats': orient_mats,
        'voxel_data': data,
        'nGrains': len(eulers_unique),
        'sgnum': sgnum,
    }


# ---------------------------------------------------------------------------
# Diagnostic 1: Forward Simulation Verification
# ---------------------------------------------------------------------------
def validate_forward_simulation(work_dir, nCPUs):
    """Verify that ForwardSimulationCompressed produces correct zip data.

    Runs a separate ForwardSim with WriteSpots=1 to get predicted spot
    positions, then checks that relevant pixels in the zip files contain
    non-zero intensity at those positions.
    """
    print('\n' + '='*70)
    print('  Diagnostic 1: Forward Simulation Verification')
    print('='*70)

    gt = load_ground_truth(work_dir)

    # --- Step 1: Run ForwardSim with WriteSpots=1 for a single scan ---
    diag_dir = work_dir / '_diag_fwdsim'
    diag_dir.mkdir(exist_ok=True)

    # Read original parameter file
    param_path = work_dir / 'Parameters_pfhedm.txt'
    with open(str(param_path)) as f:
        param_lines = f.readlines()

    # Parse lattice constant from params
    lattice_str = '3.5950 3.5950 3.5950 90 90 90'
    for line in param_lines:
        if line.startswith('LatticeConstant'):
            lattice_str = line.split(None, 1)[1].strip()
            break

    # Build a Grains.csv with all GT orientations
    grains_csv = diag_dir / 'Grains.csv'
    lat_parts = lattice_str.split()
    a, b, c = float(lat_parts[0]), float(lat_parts[1]), float(lat_parts[2])
    al, be, ga = float(lat_parts[3]), float(lat_parts[4]), float(lat_parts[5])

    with open(str(grains_csv), 'w') as f:
        f.write(f'%NumGrains {gt["nGrains"]}\n')
        f.write('%BeamCenter 0.000000\n')
        f.write('%BeamThickness 1000.000000\n')
        f.write('%GlobalPosition 0.000000\n')
        f.write('%NumPhases 1\n')
        f.write('%PhaseInfo\n')
        f.write(f'%\tSpaceGroup:{gt["sgnum"]}\n')
        f.write(f'%\tLattice Parameter: {a:.6f} {b:.6f} {c:.6f} '
                f'{al:.6f} {be:.6f} {ga:.6f}\n')
        f.write('%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t'
                'X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\n')
        for i, om in enumerate(gt['orient_mats']):
            f.write(f'{i+1}\t{chr(9).join(f"{v:.6f}" for v in om)}\t'
                    f'0.0\t0.0\t0.0\t'
                    f'{a:.6f}\t{b:.6f}\t{c:.6f}\t'
                    f'{al:.6f}\t{be:.6f}\t{ga:.6f}\n')

    # Build parameter file for WriteSpots mode
    fwd_param = diag_dir / 'ps_diag.txt'
    with open(str(fwd_param), 'w') as f:
        for line in param_lines:
            key = line.split()[0] if line.strip() and not line.strip().startswith('#') else ''
            if key == 'nScans':
                f.write('nScans 1\n')
            elif key == 'WriteSpots':
                f.write('WriteSpots 1\n')
            elif key == 'WriteImage':
                f.write('WriteImage 0\n')
            elif key in ('InFileName', 'InputFile'):
                f.write(f'InFileName {grains_csv}\n')
            elif key == 'PositionsFile':
                # Write single position
                pos_f = diag_dir / 'positions.csv'
                pos_f.write_text('0.0\n')
                f.write(f'PositionsFile {pos_f}\n')
            else:
                f.write(line)
        f.write('WriteSpots 1\n')
        f.write('WriteImage 0\n')
        f.write('nScans 1\n')
        f.write(f'InFileName {grains_csv}\n')

    # Copy hkls.csv if exists
    hkls_src = work_dir / 'hkls.csv'
    if not hkls_src.exists():
        # hkls.csv may be in a scan subdir
        hkls_src = work_dir / '1' / 'hkls.csv'
    if hkls_src.exists():
        shutil.copy2(str(hkls_src), str(diag_dir / 'hkls.csv'))

    bin_path = MIDAS_HOME / 'FF_HEDM' / 'bin' / 'ForwardSimulationCompressed'
    cmd = [str(bin_path), str(fwd_param), str(nCPUs)]
    print(f'  Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=str(diag_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: ForwardSim failed (rc={result.returncode})')
        print(f'  stderr: {result.stderr[-500:]}' if result.stderr else '')
        return

    # --- Step 2: Parse predicted spots ---
    spot_file = diag_dir / 'SpotMatrixGen.csv'
    if not spot_file.exists():
        print('  ERROR: SpotMatrixGen.csv not produced')
        return

    spot_data = np.genfromtxt(str(spot_file), skip_header=1, delimiter='\t')
    if spot_data.ndim == 1:
        spot_data = spot_data.reshape(1, -1)
    n_spots = len(spot_data)
    print(f'  Predicted spots: {n_spots}')

    # Per-grain spot counts
    # Cols: 0:GrainID 1:SpotID  2:Omega 3:DetHor 4:DetVert 5:OmeRaw 6:Eta
    #       7:RingNr 8:YLab 9:ZLab 10:Theta 11:StrainErr 12:ScanNr
    #       13:RingRad 14:omeBin
    print('\n  Per-grain predicted spots:')
    print(f'  {"Grain":>6} {"Spots":>6} {"Rings":>20}')
    print(f'  {"-"*6} {"-"*6} {"-"*20}')
    for gid in sorted(np.unique(spot_data[:, 0]).astype(int)):
        mask = spot_data[:, 0] == gid
        rings = sorted(int(r) for r in np.unique(spot_data[mask, 7]))
        print(f'  {gid:6d} {mask.sum():6d} {str(rings):>20}')

    # --- Step 3: Check zip data at predicted positions ---
    print('\n  Checking zip file intensity at predicted spot positions:')
    omega_step = 0.25
    for line in param_lines:
        if line.startswith('OmegaStep'):
            omega_step = float(line.split()[1])

    # Check a representative scan (center scan = scan 3, dir "4")
    center_scan = NSCANS // 2
    center_dir = work_dir / str(center_scan + 1)
    zip_files = list(center_dir.glob('*.MIDAS.zip'))
    if not zip_files:
        print('  ERROR: No zip file found in center scan directory')
        return

    zip_path = zip_files[0]
    store = zarr.storage.ZipStore(str(zip_path), mode='r')
    zg = zarr.open_group(store, mode='r')
    data_arr = zg['exchange/data']
    n_frames, nz, ny = data_arr.shape
    print(f'  Zip: {zip_path.name}, shape={data_arr.shape}')

    per_grain_found = {}
    per_grain_total = {}
    for gid in sorted(np.unique(spot_data[:, 0]).astype(int)):
        per_grain_found[gid] = 0
        per_grain_total[gid] = 0

    for row in spot_data:
        gid = int(row[0])
        ome_bin = int(row[14]) if len(row) > 14 else int(round((row[2] + 180) / omega_step))
        det_hor = int(round(row[3]))
        det_vert = int(round(row[4]))

        per_grain_total[gid] += 1

        # Check ±1 frame around predicted omega bin
        found = False
        for fi in range(max(0, ome_bin - 1), min(n_frames, ome_bin + 2)):
            # Detector coords: row=z, col=y
            r = max(0, min(nz - 1, det_vert))
            c = max(0, min(ny - 1, det_hor))
            # Sum a small 3x3 patch
            r0, r1 = max(0, r - 1), min(nz, r + 2)
            c0, c1 = max(0, c - 1), min(ny, c + 2)
            patch = np.array(data_arr[fi, r0:r1, c0:c1])
            if np.any(patch > 0):
                found = True
                break

        if found:
            per_grain_found[gid] += 1

    store.close()

    print(f'\n  Intensity check at center scan (scan {center_scan}):')
    print(f'  {"Grain":>6} {"Found":>6} {"Total":>6} {"Rate":>8}')
    print(f'  {"-"*6} {"-"*6} {"-"*6} {"-"*8}')
    all_ok = True
    for gid in sorted(per_grain_total.keys()):
        total = per_grain_total[gid]
        found = per_grain_found[gid]
        rate = found / total * 100 if total > 0 else 0
        status = '✓' if rate > 80 else '✗'
        print(f'  {gid:6d} {found:6d} {total:6d} {rate:7.1f}% {status}')
        if rate < 50:
            all_ok = False

    if all_ok:
        print('  ✓ Forward simulation zip data looks correct')
    else:
        print('  ✗ WARNING: Some grains have low signal in zip files')

    # Cleanup diag dir
    shutil.rmtree(str(diag_dir), ignore_errors=True)
    return gt


# ---------------------------------------------------------------------------
# Diagnostic 2: Peak Search Quality
# ---------------------------------------------------------------------------
def validate_peak_search(work_dir, gt, nCPUs):
    """Compare peak search output against forward-simulation predicted spots.

    Runs ForwardSim with WriteSpots=1 for the center scan to get predicted
    spot positions, then matches against InputAllExtraInfoFittingAll.
    """
    print('\n' + '='*70)
    print('  Diagnostic 2: Peak Search Quality')
    print('='*70)

    if gt is None:
        gt = load_ground_truth(work_dir)

    # --- Generate predicted spots for center scan ---
    diag_dir = work_dir / '_diag_peaksearch'
    diag_dir.mkdir(exist_ok=True)

    param_path = work_dir / 'Parameters_pfhedm.txt'
    with open(str(param_path)) as f:
        param_lines = f.readlines()

    # Parse needed params
    lattice_str = '3.5950 3.5950 3.5950 90 90 90'
    px_size = 200.0  # default
    omega_step = 0.25
    for line in param_lines:
        if line.startswith('LatticeConstant'):
            lattice_str = line.split(None, 1)[1].strip()
        elif line.startswith('px '):
            px_size = float(line.split()[1])
        elif line.startswith('OmegaStep'):
            omega_step = float(line.split()[1])

    # Build Grains.csv
    lat_parts = lattice_str.split()
    a, b, c = float(lat_parts[0]), float(lat_parts[1]), float(lat_parts[2])
    al, be, ga = float(lat_parts[3]), float(lat_parts[4]), float(lat_parts[5])

    grains_csv = diag_dir / 'Grains.csv'
    with open(str(grains_csv), 'w') as f:
        f.write(f'%NumGrains {gt["nGrains"]}\n')
        f.write('%BeamCenter 0.000000\n')
        f.write('%BeamThickness 1000.000000\n')
        f.write('%GlobalPosition 0.000000\n')
        f.write('%NumPhases 1\n')
        f.write('%PhaseInfo\n')
        f.write(f'%\tSpaceGroup:{gt["sgnum"]}\n')
        f.write(f'%\tLattice Parameter: {a:.6f} {b:.6f} {c:.6f} '
                f'{al:.6f} {be:.6f} {ga:.6f}\n')
        f.write('%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t'
                'X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\n')
        for i, om in enumerate(gt['orient_mats']):
            f.write(f'{i+1}\t{chr(9).join(f"{v:.6f}" for v in om)}\t'
                    f'0.0\t0.0\t0.0\t'
                    f'{a:.6f}\t{b:.6f}\t{c:.6f}\t'
                    f'{al:.6f}\t{be:.6f}\t{ga:.6f}\n')

    fwd_param = diag_dir / 'ps_diag.txt'
    with open(str(fwd_param), 'w') as f:
        for line in param_lines:
            key = line.split()[0] if line.strip() and not line.strip().startswith('#') else ''
            if key == 'nScans':
                f.write('nScans 1\n')
            elif key == 'WriteSpots':
                continue
            elif key == 'WriteImage':
                continue
            elif key in ('InFileName', 'InputFile'):
                continue
            elif key == 'PositionsFile':
                pos_f = diag_dir / 'positions.csv'
                pos_f.write_text('0.0\n')
                f.write(f'PositionsFile {pos_f}\n')
            else:
                f.write(line)
        f.write('WriteSpots 1\n')
        f.write('WriteImage 0\n')
        f.write('nScans 1\n')
        f.write(f'InFileName {grains_csv}\n')

    hkls_src = work_dir / 'hkls.csv'
    if not hkls_src.exists():
        hkls_src = work_dir / '1' / 'hkls.csv'
    if hkls_src.exists():
        shutil.copy2(str(hkls_src), str(diag_dir / 'hkls.csv'))

    bin_path = MIDAS_HOME / 'FF_HEDM' / 'bin' / 'ForwardSimulationCompressed'
    result = subprocess.run(
        [str(bin_path), str(fwd_param), str(nCPUs)],
        cwd=str(diag_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: ForwardSim for peak validation failed')
        shutil.rmtree(str(diag_dir), ignore_errors=True)
        return

    spot_file = diag_dir / 'SpotMatrixGen.csv'
    if not spot_file.exists():
        print('  ERROR: SpotMatrixGen.csv not produced')
        shutil.rmtree(str(diag_dir), ignore_errors=True)
        return

    pred = np.genfromtxt(str(spot_file), skip_header=1, delimiter='\t')
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    print(f'  Predicted spots from ForwardSim: {len(pred)}')

    # --- Match predicted spots against InputAllExtraInfoFittingAll for each scan ---
    lab_tol = 4 * px_size  # 4 pixels in µm
    ome_tol = 2 * abs(omega_step)  # 2 frames in degrees

    print(f'  Match tolerances: ±{lab_tol:.0f} µm (lab), ±{ome_tol:.3f}° (omega)')
    print()

    # Per-scan matching
    for scanNr in range(NSCANS):
        fn = work_dir / f'InputAllExtraInfoFittingAll{scanNr}.csv'
        if not fn.exists():
            print(f'  Scan {scanNr}: InputAll not found')
            continue

        import pandas as pd
        df = pd.read_csv(str(fn), delimiter=' ', skipinitialspace=True)
        ylab_col = '%YLab' if '%YLab' in df.columns else 'YLab'
        obs_y = df[ylab_col].values
        obs_z = df['ZLab'].values
        obs_ome = df['Omega'].values
        obs_ring = df['RingNumber'].values.astype(int)
        obs_matched = np.zeros(len(df), dtype=bool)

        # Per-grain matching
        per_grain = {}
        for gid in range(1, gt['nGrains'] + 1):
            per_grain[gid] = {'predicted': 0, 'matched': 0, 'dy': [], 'dz': [], 'dome': []}

        pred_matched = np.zeros(len(pred), dtype=bool)
        for i, row in enumerate(pred):
            gid = int(row[0])
            p_ylab, p_zlab, p_ome, p_ring = row[8], row[9], row[2], int(row[7])
            per_grain[gid]['predicted'] += 1

            # Find matching observed spot
            candidates = (
                (obs_ring == p_ring) &
                (np.abs(obs_y - p_ylab) <= lab_tol) &
                (np.abs(obs_z - p_zlab) <= lab_tol) &
                (np.abs(obs_ome - p_ome) <= ome_tol)
            )
            if np.any(candidates):
                idx = np.where(candidates)[0]
                # Pick closest
                dists = (obs_y[idx] - p_ylab)**2 + (obs_z[idx] - p_zlab)**2
                best = idx[np.argmin(dists)]
                pred_matched[i] = True
                obs_matched[best] = True
                per_grain[gid]['matched'] += 1
                per_grain[gid]['dy'].append(obs_y[best] - p_ylab)
                per_grain[gid]['dz'].append(obs_z[best] - p_zlab)
                per_grain[gid]['dome'].append(obs_ome[best] - p_ome)

        n_obs = len(df)
        n_pred_matched = pred_matched.sum()
        n_obs_matched = obs_matched.sum()
        n_missing = len(pred) - n_pred_matched
        n_extra = n_obs - n_obs_matched

        if scanNr == NSCANS // 2:  # center scan: full detail
            print(f'  Scan {scanNr} (CENTER) — {n_obs} observed spots:')
            print(f'  {"GT":>4} {"Pred":>5} {"Match":>5} {"Miss":>5} '
                  f'{"Rate":>6} {"AvgΔY":>8} {"AvgΔZ":>8} {"AvgΔΩ":>8}')
            print(f'  {"-"*4} {"-"*5} {"-"*5} {"-"*5} {"-"*6} {"-"*8} {"-"*8} {"-"*8}')
            for gid in sorted(per_grain.keys()):
                g = per_grain[gid]
                rate = g['matched'] / g['predicted'] * 100 if g['predicted'] > 0 else 0
                avg_dy = np.mean(np.abs(g['dy'])) if g['dy'] else 0
                avg_dz = np.mean(np.abs(g['dz'])) if g['dz'] else 0
                avg_dome = np.mean(np.abs(g['dome'])) if g['dome'] else 0
                print(f'  {gid:4d} {g["predicted"]:5d} {g["matched"]:5d} '
                      f'{g["predicted"]-g["matched"]:5d} '
                      f'{rate:5.1f}% {avg_dy:7.1f}µ {avg_dz:7.1f}µ {avg_dome:7.3f}°')

            print(f'  Total: {n_obs} observed, {n_pred_matched} matched to GT, '
                  f'{n_missing} GT missing, {n_extra} extra unmatched')
        else:
            print(f'  Scan {scanNr}: {n_obs} obs, '
                  f'{n_pred_matched}/{len(pred)} matched ({n_pred_matched/len(pred)*100:.0f}%), '
                  f'{n_extra} extra')

    print()
    shutil.rmtree(str(diag_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# Diagnostic 3: Indexer Solution Quality
# ---------------------------------------------------------------------------
def validate_indexer_output(work_dir, gt):
    """Analyze indexer solutions per voxel against ground truth.

    Reads IndexBest_all.bin and checks whether correct GT orientations
    are present, their confidence rank, and the gap to spurious solutions.
    Maps each pipeline voxel to its expected GT grain via the EBSD file.
    """
    print('\n' + '='*70)
    print('  Diagnostic 3: Indexer Solution Quality')
    print('='*70)

    if gt is None:
        gt = load_ground_truth(work_dir)

    sgnum = gt['sgnum']
    ebsd_data = gt['voxel_data']  # [x, y, z, e1, e2, e3]
    eulers_unique = gt['eulers_unique']

    # --- Build voxel → expected GT grain mapping from EBSD + positions ---
    positions = np.loadtxt(str(work_dir / 'positions.csv'))
    n_pos = len(positions)

    # Map each EBSD voxel Euler angle to a GT grain index
    def euler_to_gt(euler):
        for gi, gu in enumerate(eulers_unique):
            if np.allclose(euler, gu):
                return gi
        return -1

    # For the NxN pipeline grid, map voxel index → expected GT grain
    # voxel index = row * n_pos + col, where row/col index into positions
    expected_gt = {}
    for row in range(n_pos):
        y_pos = positions[row]
        for col in range(n_pos):
            x_pos = positions[col]
            vox_idx = row * n_pos + col
            # Find nearest EBSD voxel
            dists = (ebsd_data[:, 0] - x_pos)**2 + (ebsd_data[:, 1] - y_pos)**2
            nearest = np.argmin(dists)
            expected_gt[vox_idx] = euler_to_gt(ebsd_data[nearest, 3:6])

    # Print expected GT grid
    print(f'\n  Expected GT grain map ({n_pos}×{n_pos} grid):')
    for row in range(n_pos):
        line = f'    row {row} (y={positions[row]:7.1f}): '
        for col in range(n_pos):
            vox_idx = row * n_pos + col
            line += f'GT{expected_gt[vox_idx]} '
        print(line)

    # --- Read consolidated IndexBest_all.bin ---
    idx_bin = work_dir / 'Output' / 'IndexBest_all.bin'
    if not idx_bin.exists():
        print('  ERROR: IndexBest_all.bin not found')
        return

    with open(str(idx_bin), 'rb') as f:
        nVoxels = struct.unpack('i', f.read(4))[0]
        nSolArr = np.frombuffer(f.read(4 * nVoxels), dtype=np.int32)
        offArr = np.frombuffer(f.read(8 * nVoxels), dtype=np.int64)
        header_size = 4 + 4 * nVoxels + 8 * nVoxels
        all_data = np.frombuffer(f.read(), dtype=np.float64)

    print(f'\n  Voxels: {nVoxels}, Total solutions: {nSolArr.sum()}')

    # Binary layout per solution: 16 doubles
    # [0]:rowNr [1]:IA [2:11]:OM(9) [11]:pos_x [12]:pos_y [13]:pos_z
    # [14]:nTotal [15]:nMatched
    # Confidence = [15]/[14]

    # --- Per-voxel analysis ---
    stats = {
        'best_matches_expected': 0,
        'best_matches_any_gt': 0,
        'expected_found_anywhere': 0,
        'expected_in_top5': 0,
        'mislabeled': 0,
        'total_voxels': 0,
        'empty_voxels': 0,
    }

    rows = []
    mislabeled_details = []

    for v in range(nVoxels):
        n_sol = nSolArr[v]
        exp_gt = expected_gt.get(v, -1)

        if n_sol == 0:
            stats['empty_voxels'] += 1
            rows.append({
                'vox': v, 'row': v // n_pos, 'col': v % n_pos,
                'exp_gt': exp_gt, 'n_sol': 0,
                'best_conf': 0, 'picked_gt': -1, 'picked_miso': 999,
                'exp_found': False, 'exp_rank': -1, 'exp_conf': 0,
                'exp_miso': 999, 'mislabeled': True,
            })
            continue

        stats['total_voxels'] += 1
        offset_doubles = int((offArr[v] - header_size) // 8)
        sol_data = all_data[offset_doubles:offset_doubles + n_sol * 16].reshape(n_sol, 16)

        # Compute confidence for each solution
        confs = np.where(
            sol_data[:, 14] > 0,
            sol_data[:, 15] / sol_data[:, 14],
            0.0
        )
        n_total = sol_data[:, 14]
        n_matched = sol_data[:, 15]

        # Find best-by-confidence
        best_conf_idx = np.argmax(confs)
        best_conf_val = confs[best_conf_idx]

        # For each solution, compute misorientation to ALL GT grains
        miso_to_all_gt = np.full((n_sol, gt['nGrains']), 999.0)
        for i in range(n_sol):
            om = sol_data[i, 2:11]
            for gi, gt_om in enumerate(gt['orient_mats']):
                angle, _ = GetMisOrientationAngleOM(gt_om, om, sgnum)
                miso_to_all_gt[i, gi] = np.degrees(angle)

        # Nearest GT grain for the best-by-confidence solution
        picked_gt = np.argmin(miso_to_all_gt[best_conf_idx])
        picked_miso = miso_to_all_gt[best_conf_idx, picked_gt]

        # Check if the EXPECTED GT grain orientation exists anywhere
        if exp_gt >= 0:
            miso_to_expected = miso_to_all_gt[:, exp_gt]
            best_exp_sol_idx = np.argmin(miso_to_expected)
            best_exp_miso = miso_to_expected[best_exp_sol_idx]
            exp_found = best_exp_miso < 2.0

            # Rank of expected grain's best solution by confidence
            conf_order = np.argsort(-confs)
            exp_rank = -1
            exp_conf = 0.0
            if exp_found:
                exp_rank = int(np.where(conf_order == best_exp_sol_idx)[0][0]) + 1
                exp_conf = confs[best_exp_sol_idx]
        else:
            exp_found = False
            exp_rank = -1
            exp_conf = 0.0
            best_exp_miso = 999.0

        # Is the picked grain the expected one?
        best_matches_expected = (picked_miso < 2.0 and picked_gt == exp_gt)
        best_matches_any = picked_miso < 2.0
        is_mislabeled = not best_matches_expected

        if best_matches_expected:
            stats['best_matches_expected'] += 1
        if best_matches_any:
            stats['best_matches_any_gt'] += 1
        if exp_found:
            stats['expected_found_anywhere'] += 1
        if exp_rank > 0 and exp_rank <= 5:
            stats['expected_in_top5'] += 1
        if is_mislabeled:
            stats['mislabeled'] += 1

        r = {
            'vox': v, 'row': v // n_pos, 'col': v % n_pos,
            'exp_gt': exp_gt, 'n_sol': n_sol,
            'best_conf': best_conf_val,
            'picked_gt': picked_gt, 'picked_miso': picked_miso,
            'exp_found': exp_found, 'exp_rank': exp_rank,
            'exp_conf': exp_conf, 'exp_miso': best_exp_miso,
            'mislabeled': is_mislabeled,
            'n_total': int(n_total[best_conf_idx]),
            'n_matched': int(n_matched[best_conf_idx]),
        }
        rows.append(r)

        if is_mislabeled:
            # Collect additional detail for mislabeled analysis
            r_detail = dict(r)
            # Top 3 solutions by confidence
            top3 = conf_order[:min(3, n_sol)]
            r_detail['top3'] = []
            for rank_idx, sol_idx in enumerate(top3):
                nearest_gt = np.argmin(miso_to_all_gt[sol_idx])
                r_detail['top3'].append({
                    'rank': rank_idx + 1,
                    'conf': confs[sol_idx],
                    'nearest_gt': nearest_gt,
                    'miso': miso_to_all_gt[sol_idx, nearest_gt],
                    'miso_to_exp': miso_to_all_gt[sol_idx, exp_gt] if exp_gt >= 0 else 999,
                    'n_total': int(n_total[sol_idx]),
                    'n_matched': int(n_matched[sol_idx]),
                })
            if exp_found:
                r_detail['exp_sol_detail'] = {
                    'rank': exp_rank,
                    'conf': exp_conf,
                    'miso': best_exp_miso,
                    'n_total': int(n_total[best_exp_sol_idx]),
                    'n_matched': int(n_matched[best_exp_sol_idx]),
                }
            mislabeled_details.append(r_detail)

    # --- Print full voxel table ---
    n = stats['total_voxels']
    print(f'\n  Per-voxel indexer analysis ({n} voxels with solutions, '
          f'{stats["empty_voxels"]} empty):')
    print(f'  {"Vox":>4} {"R,C":>5} {"ExpGT":>5} {"nSol":>4} '
          f'{"Picked":>6} {"PkMiso":>6} {"PkConf":>6} '
          f'{"ExpFnd":>6} {"ExpRk":>5} {"ExpConf":>7} {"":>5}')
    print(f'  {"-"*4} {"-"*5} {"-"*5} {"-"*4} '
          f'{"-"*6} {"-"*6} {"-"*6} '
          f'{"-"*6} {"-"*5} {"-"*7} {"-"*5}')

    for r in rows:
        if r['n_sol'] == 0:
            print(f'  {r["vox"]:4d} {r["row"]},{r["col"]:1d}  GT{r["exp_gt"]}     0  (empty)')
            continue

        pk_str = f'GT{r["picked_gt"]}' if r['picked_miso'] < 2.0 else f'??{r["picked_gt"]}'
        exp_fnd = '✓' if r['exp_found'] else '✗'
        exp_rk = f'{r["exp_rank"]:3d}' if r['exp_rank'] > 0 else 'n/a'
        exp_cf = f'{r["exp_conf"]:.4f}' if r['exp_rank'] > 0 else '  n/a'
        flag = ''
        if r['mislabeled']:
            if r['picked_miso'] >= 2.0:
                flag = '✗ SPURIOUS'
            elif r['picked_gt'] != r['exp_gt']:
                flag = f'✗ WRONG (exp GT{r["exp_gt"]})'
            else:
                flag = '✗ MISS'

        print(f'  {r["vox"]:4d} {r["row"]},{r["col"]:1d}  GT{r["exp_gt"]}  {r["n_sol"]:4d} '
              f'{pk_str:>6} {r["picked_miso"]:5.1f}° {r["best_conf"]:.4f} '
              f'  {exp_fnd}   {exp_rk} {exp_cf} {flag}')

    # --- Summary ---
    print(f'\n  Summary ({n} voxels with solutions):')
    print(f'    Best-by-conf picks EXPECTED GT: {stats["best_matches_expected"]}/{n} '
          f'({stats["best_matches_expected"]/n*100:.0f}%)')
    print(f'    Best-by-conf picks ANY real GT: {stats["best_matches_any_gt"]}/{n} '
          f'({stats["best_matches_any_gt"]/n*100:.0f}%)')
    print(f'    Expected GT found in solutions: {stats["expected_found_anywhere"]}/{n} '
          f'({stats["expected_found_anywhere"]/n*100:.0f}%)')
    print(f'    Expected GT in top 5 by conf:   {stats["expected_in_top5"]}/{n} '
          f'({stats["expected_in_top5"]/n*100:.0f}%)')
    print(f'    MIS-LABELED voxels:             {stats["mislabeled"]}/{n} '
          f'({stats["mislabeled"]/n*100:.0f}%)')

    # Confidence statistics
    expected_confs = [r['exp_conf'] for r in rows if r['exp_found']]
    best_confs = [r['best_conf'] for r in rows if r['n_sol'] > 0]
    if expected_confs:
        print(f'\n  Confidence of expected GT solutions (when found):')
        print(f'    Mean: {np.mean(expected_confs):.4f}, '
              f'Max: {np.max(expected_confs):.4f}, '
              f'Min: {np.min(expected_confs):.4f}')
    print(f'  Confidence of best-by-confidence (picked):')
    print(f'    Mean: {np.mean(best_confs):.4f}, '
          f'Max: {np.max(best_confs):.4f}, '
          f'Min: {np.min(best_confs):.4f}')

    # --- Detailed mislabeled voxel analysis ---
    if mislabeled_details:
        print(f'\n  {"="*60}')
        print(f'  MIS-LABELED VOXELS — Detailed Analysis')
        print(f'  {"="*60}')

        for d in mislabeled_details:
            v = d['vox']
            print(f'\n  Voxel {v} (row={d["row"]}, col={d["col"]}):')
            print(f'    Expected grain: GT{d["exp_gt"]}')
            print(f'    Picked grain:   GT{d["picked_gt"]} '
                  f'(miso={d["picked_miso"]:.2f}°, '
                  f'conf={d["best_conf"]:.4f}, '
                  f'nMatched/nTotal={d["n_matched"]}/{d["n_total"]})')

            if d['exp_found']:
                ed = d['exp_sol_detail']
                print(f'    Expected GT{d["exp_gt"]} IS present:')
                print(f'      Rank: {ed["rank"]}/{d["n_sol"]}, '
                      f'conf={ed["conf"]:.4f}, '
                      f'miso={ed["miso"]:.2f}°, '
                      f'nMatched/nTotal={ed["n_matched"]}/{ed["n_total"]}')
                print(f'      Confidence gap: {d["best_conf"]:.4f} (picked) '
                      f'vs {ed["conf"]:.4f} (expected) '
                      f'= {d["best_conf"] - ed["conf"]:.4f}')
            else:
                print(f'    Expected GT{d["exp_gt"]} NOT found in any solution '
                      f'(closest miso: {d["exp_miso"]:.1f}°)')

            print(f'    Top 3 solutions by confidence:')
            for t in d['top3']:
                gt_match = '✓' if t['miso'] < 2.0 else ' '
                exp_match = '← EXPECTED' if (t['miso_to_exp'] < 2.0 and d['exp_gt'] >= 0) else ''
                print(f'      #{t["rank"]}: conf={t["conf"]:.4f}, '
                      f'nearest=GT{t["nearest_gt"]} (miso={t["miso"]:.2f}°){gt_match}, '
                      f'nM/nT={t["n_matched"]}/{t["n_total"]} {exp_match}')

    # --- Check UniqueOrientations.csv ---
    uq_path = work_dir / 'UniqueOrientations.csv'
    if uq_path.exists():
        uq = np.loadtxt(str(uq_path))
        if uq.ndim == 1:
            uq = uq.reshape(1, -1)
        n_unique = len(uq)
        print(f'\n  UniqueOrientations.csv: {n_unique} grains (GT has {gt["nGrains"]})')

        for i, row in enumerate(uq):
            om = row[5:14]
            best_miso = 999
            best_gt = -1
            for gi, gt_om in enumerate(gt['orient_mats']):
                angle, _ = GetMisOrientationAngleOM(gt_om, om, sgnum)
                angle_deg = np.degrees(angle)
                if angle_deg < best_miso:
                    best_miso = angle_deg
                    best_gt = gi
            status = '✓' if best_miso < 2.0 else '✗ PHANTOM'
            print(f'    Grain {i}: voxIdx={int(row[0])}, '
                  f'nearest GT={best_gt}, miso={best_miso:.4f}° {status}')

        n_phantom = sum(1 for row in uq if min(
            np.degrees(GetMisOrientationAngleOM(gt_om, row[5:14], sgnum)[0])
            for gt_om in gt['orient_mats']
        ) >= 2.0)
        n_matched = n_unique - n_phantom
        print(f'    Matched: {n_matched}/{gt["nGrains"]} GT grains found')
        if n_phantom > 0:
            print(f'    Phantom grains: {n_phantom}')


# ---------------------------------------------------------------------------
# Diagnostic 5: Indexer Confidence Per GT Orientation (all voxels)
# ---------------------------------------------------------------------------
def _build_grains_csv_and_params(work_dir, gt):
    """Write Grains_debug.csv and paramstest_debug.txt; return (grains_csv, debug_params, positions, num_scans)."""
    param_path = work_dir / 'Parameters_pfhedm.txt'
    lattice_str = '3.5950 3.5950 3.5950 90 90 90'
    with open(str(param_path)) as fh:
        for line in fh:
            if line.startswith('LatticeConstant'):
                lattice_str = line.split(None, 1)[1].strip()

    lat_parts = lattice_str.split()
    a, b, c = float(lat_parts[0]), float(lat_parts[1]), float(lat_parts[2])
    al, be, ga = float(lat_parts[3]), float(lat_parts[4]), float(lat_parts[5])
    sgnum = gt['sgnum']
    n_grains = gt['nGrains']

    grains_csv = work_dir / 'Grains_debug.csv'
    with open(str(grains_csv), 'w') as fh:
        fh.write(f'%NumGrains {n_grains}\n')
        fh.write('%BeamCenter 0.000000\n')
        fh.write('%BeamThickness 1000.000000\n')
        fh.write('%GlobalPosition 0.000000\n')
        fh.write('%NumPhases 1\n')
        fh.write('%PhaseInfo\n')
        fh.write(f'%\tSpaceGroup:{sgnum}\n')
        fh.write(f'%\tLattice Parameter: {a:.6f} {b:.6f} {c:.6f} '
                 f'{al:.6f} {be:.6f} {ga:.6f}\n')
        fh.write('%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t'
                 'X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\n')
        for i, om in enumerate(gt['orient_mats']):
            fh.write(f'{i+1}\t{chr(9).join(f"{v:.6f}" for v in om)}\t'
                     f'0.0\t0.0\t0.0\t'
                     f'{a:.6f}\t{b:.6f}\t{c:.6f}\t'
                     f'{al:.6f}\t{be:.6f}\t{ga:.6f}\n')

    params_src = work_dir / 'paramstest.txt'
    debug_params = work_dir / 'paramstest_debug.txt'
    with open(str(params_src)) as fh:
        lines = fh.readlines()
    # Redirect OutputFolder to a temp dir so we don't overwrite the pipeline's
    # consolidated files (IndexBest_all.bin etc.) in Output/
    debug_output = work_dir / '_debug_output'
    debug_output.mkdir(exist_ok=True)
    filtered = [ln for ln in lines
                if not ln.startswith('GrainsFile') and not ln.startswith('MicFile')
                and not ln.startswith('OutputFolder')]
    filtered.append(f'GrainsFile {grains_csv}\n')
    filtered.append(f'OutputFolder {debug_output}\n')
    with open(str(debug_params), 'w') as fh:
        fh.writelines(filtered)

    positions = np.loadtxt(str(work_dir / 'positions.csv'))
    num_scans = len(positions)
    return grains_csv, debug_params, positions, num_scans


def _parse_conf_debug(stdout):
    """Parse CONF_DEBUG lines into conf[voxNr][grainIdx] = conf_value."""
    conf = {}
    for ln in stdout.splitlines():
        if not ln.startswith('CONF_DEBUG'):
            continue
        parts = {}
        for tok in ln.split()[1:]:
            k, v = tok.split('=')
            parts[k] = v
        gi = int(parts['grainIdx'])
        vn = int(parts['voxNr'])
        nM = int(parts['nMatched'])
        nT = int(parts['nTspots'])
        cf = float(parts['conf'])
        conf.setdefault(vn, {})[gi] = {'nMatched': nM, 'nTspots': nT, 'conf': cf}
    return conf


def debug_indexer_single_voxel(work_dir, voxel_nr=25):
    """Run IndexerScanningOMP with GT Grains.csv for a single voxel."""
    print('\n' + '='*70)
    print(f'  Diagnostic 5: Indexer confidence per GT orientation (voxel {voxel_nr})')
    print('='*70)

    for f in [work_dir / 'Spots.bin', work_dir / 'hkls.csv',
              work_dir / 'positions.csv', work_dir / 'paramstest.txt']:
        if not f.exists():
            print(f'  ERROR: required file missing: {f}')
            return

    gt = load_ground_truth(work_dir)
    grains_csv, debug_params, positions, num_scans = _build_grains_csv_and_params(work_dir, gt)
    n_voxels = num_scans * num_scans

    if voxel_nr >= n_voxels:
        print(f'  ERROR: voxel_nr={voxel_nr} out of range (nVoxels={n_voxels})')
        debug_params.unlink(missing_ok=True)
    debug_output = work_dir / '_debug_output'
    if debug_output.exists():
        shutil.rmtree(str(debug_output), ignore_errors=True)
        return

    pos_sorted = np.sort(positions)
    row = voxel_nr // num_scans
    col = voxel_nr % num_scans
    print(f'  Grid: {num_scans}×{num_scans}, nVoxels={n_voxels}')
    print(f'  Voxel {voxel_nr}: row={row}, col={col}, '
          f'pos=({pos_sorted[row]:.2f}, {pos_sorted[col]:.2f}) µm')

    indexer_bin = MIDAS_HOME / 'FF_HEDM' / 'bin' / 'IndexerScanningOMP'
    cmd = [str(indexer_bin), str(debug_params),
           str(voxel_nr), str(n_voxels), str(num_scans), '1']
    print(f'\n  Running: {" ".join(cmd)}')
    env = dict(os.environ, MIDAS_DEBUG_INDEXER='1')
    result = subprocess.run(cmd, cwd=str(work_dir),
                            capture_output=True, text=True, timeout=120,
                            env=env)

    conf_all = _parse_conf_debug(result.stdout)
    conf_table = conf_all.get(voxel_nr, {})

    eulers = gt['eulers_unique']
    n_grains = gt['nGrains']
    print(f'\n  Confidence of each GT grain orientation at voxel {voxel_nr}:')
    print(f'  {"GrainIdx":>9} {"Euler (deg)":>30} {"nMatched":>9} {"nTspots":>9} {"Conf":>8}')
    print(f'  {"-"*9} {"-"*30} {"-"*9} {"-"*9} {"-"*8}')
    for gi in range(n_grains):
        e = eulers[gi]
        estr = f'({e[0]:.1f},{e[1]:.1f},{e[2]:.1f})'
        if gi in conf_table:
            r = conf_table[gi]
            print(f'  {gi:9d} {estr:>30} {r["nMatched"]:9d} {r["nTspots"]:9d} {r["conf"]:8.4f}')
        else:
            print(f'  {gi:9d} {estr:>30} {"—":>9} {"—":>9} {"0 matches":>9}')

    if result.returncode != 0:
        print(f'\n  WARNING: indexer exit code {result.returncode}')
    debug_params.unlink(missing_ok=True)
    debug_output = work_dir / '_debug_output'
    if debug_output.exists():
        shutil.rmtree(str(debug_output), ignore_errors=True)


def debug_indexer_all_voxels(work_dir, nCPUs=1):
    """Run IndexerScanningOMP with GT Grains.csv for ALL voxels in one pass.

    Reports per-voxel confidence for every GT grain orientation, highlights
    where the highest-confidence grain differs from the expected GT grain,
    and prints a grid summary and a detailed per-voxel table.

    Uses the existing Spots.bin / hkls.csv / positions.csv / paramstest.txt.
    """
    print('\n' + '='*70)
    print('  Diagnostic 5: Indexer confidence — ALL voxels')
    print('='*70)

    for f in [work_dir / 'Spots.bin', work_dir / 'hkls.csv',
              work_dir / 'positions.csv', work_dir / 'paramstest.txt']:
        if not f.exists():
            print(f'  ERROR: required file missing: {f}')
            print('  Run the full pipeline first to generate Spots.bin.')
            return

    gt = load_ground_truth(work_dir)
    n_grains = gt['nGrains']
    eulers   = gt['eulers_unique']
    ebsd_data = gt['voxel_data']

    grains_csv, debug_params, positions, num_scans = _build_grains_csv_and_params(work_dir, gt)
    n_voxels  = num_scans * num_scans
    pos_sorted = np.sort(positions)

    print(f'  Grid: {num_scans}×{num_scans} = {n_voxels} voxels, {n_grains} GT grains')
    print(f'  Running IndexerScanningOMP for all voxels (nBlocks=1, nProcs={nCPUs})…')

    # Build voxel→expected GT grain map (same logic as validate_indexer_output)
    def euler_to_gt(euler):
        for gi, gu in enumerate(eulers):
            if np.allclose(euler, gu):
                return gi
        return -1

    expected_gt = {}
    for vox in range(n_voxels):
        row = vox // num_scans
        col = vox % num_scans
        x_pos = pos_sorted[row]
        y_pos = pos_sorted[col]
        dists = (ebsd_data[:, 0] - x_pos)**2 + (ebsd_data[:, 1] - y_pos)**2
        nearest = np.argmin(dists)
        expected_gt[vox] = euler_to_gt(ebsd_data[nearest, 3:6])

    # Run indexer once for all voxels (blockNr=0, nBlocks=1)
    # Serial output (nProcs=1) avoids interleaved CONF_DEBUG lines
    indexer_bin = MIDAS_HOME / 'FF_HEDM' / 'bin' / 'IndexerScanningOMP'
    cmd = [str(indexer_bin), str(debug_params), '0', '1', str(num_scans), str(nCPUs)]
    print(f'  Command: {" ".join(cmd)}\n')
    env = dict(os.environ, MIDAS_DEBUG_INDEXER='1')
    result = subprocess.run(cmd, cwd=str(work_dir),
                            capture_output=True, text=True, timeout=300,
                            env=env)

    if result.returncode != 0:
        print(f'  WARNING: indexer exit code {result.returncode}')
        if result.stderr:
            print(f'  stderr: {result.stderr[-500:]}')

    # Parse CONF_DEBUG output
    conf_all = _parse_conf_debug(result.stdout)
    n_parsed = sum(len(v) for v in conf_all.values())
    print(f'  Parsed {n_parsed} CONF_DEBUG entries across {len(conf_all)} voxels')

    # ----------------------------------------------------------------
    # Build per-voxel report
    # ----------------------------------------------------------------
    THRESH = 0.1   # flag if best conf is below this (likely problem)

    winner_grid   = np.full((num_scans, num_scans), -1, dtype=int)
    winner_conf   = np.zeros((num_scans, num_scans))
    expected_grid = np.full((num_scans, num_scans), -1, dtype=int)
    mismatch_voxels = []   # voxels where winner != expected
    problem_voxels  = []   # voxels where expected GT has lowest confidence
    zero_conf_voxels = []  # voxels where expected GT has 0 matches

    voxel_rows = []
    for vox in range(n_voxels):
        row = vox // num_scans
        col = vox % num_scans
        exp_gt = expected_gt[vox]
        expected_grid[row, col] = exp_gt
        ct = conf_all.get(vox, {})

        if not ct:
            # No CONF_DEBUG at all for this voxel (0 matches for every grain)
            zero_conf_voxels.append(vox)
            voxel_rows.append({
                'vox': vox, 'row': row, 'col': col, 'exp_gt': exp_gt,
                'winner': -1, 'winner_conf': 0.0,
                'exp_conf': 0.0, 'exp_rank': -1,
                'confs': {},
            })
            continue

        # Rank grains by confidence (descending)
        ranked = sorted(ct.items(), key=lambda kv: kv[1]['conf'], reverse=True)
        winner_gi, winner_data = ranked[0]
        winner_grid[row, col] = winner_gi
        winner_conf[row, col] = winner_data['conf']

        exp_conf = ct.get(exp_gt, {}).get('conf', 0.0) if exp_gt >= 0 else 0.0
        exp_rank = next((i+1 for i, (gi, _) in enumerate(ranked) if gi == exp_gt), -1)

        if exp_gt >= 0 and winner_gi != exp_gt:
            mismatch_voxels.append(vox)

        # "Problem" = expected GT is ranked last among grains that were tested
        if exp_gt >= 0 and exp_rank == len(ct):
            problem_voxels.append(vox)

        if exp_gt >= 0 and exp_conf == 0.0:
            zero_conf_voxels.append(vox)

        voxel_rows.append({
            'vox': vox, 'row': row, 'col': col, 'exp_gt': exp_gt,
            'winner': winner_gi, 'winner_conf': winner_data['conf'],
            'exp_conf': exp_conf, 'exp_rank': exp_rank,
            'confs': {gi: d['conf'] for gi, d in ct.items()},
        })

    # ----------------------------------------------------------------
    # Print expected GT grid
    # ----------------------------------------------------------------
    print(f'\n  Expected GT grain grid ({num_scans}×{num_scans}):')
    print(f'  (row=x-axis sorted pos, col=y-axis sorted pos)')
    header = '       ' + ''.join(f'{pos_sorted[c]:8.1f}' for c in range(num_scans))
    print(f'  {header}')
    for r in range(num_scans):
        row_str = ''.join(f'     GT{expected_grid[r, c]}' for c in range(num_scans))
        print(f'  {pos_sorted[r]:6.1f} {row_str}')

    # ----------------------------------------------------------------
    # Print winner grid (highest-confidence grain per voxel)
    # ----------------------------------------------------------------
    print(f'\n  Winner grain grid (highest confidence):')
    print(f'  {header}')
    for r in range(num_scans):
        cells = []
        for c in range(num_scans):
            gi = winner_grid[r, c]
            exp = expected_grid[r, c]
            marker = '!' if gi != exp else ' '
            cells.append(f'  {marker}GT{gi}({winner_conf[r,c]:.2f})')
        print(f'  {pos_sorted[r]:6.1f}' + ''.join(cells))
    print('  (! = winner differs from expected GT)')

    # ----------------------------------------------------------------
    # Detailed per-voxel confidence table
    # ----------------------------------------------------------------
    print(f'\n  Per-voxel confidence for each GT grain:')
    header2 = (f'  {"Vox":>4} {"r":>2} {"c":>2} {"ExpGT":>5}  '
               + '  '.join(f'GT{gi}(conf)' for gi in range(n_grains))
               + '  Winner  ExpRank  Issue')
    print(f'  {"-"*len(header2)}')
    print(header2)
    print(f'  {"-"*len(header2)}')

    for vr in voxel_rows:
        vox = vr['vox']
        exp_gt = vr['exp_gt']
        winner = vr['winner']
        exp_rank = vr['exp_rank']
        ct = vr['confs']

        # Per-grain confidences in order
        grain_cols = '  '.join(
            f'{ct.get(gi, 0.0):6.3f}{"*" if gi == exp_gt else " "}'
            for gi in range(n_grains)
        )
        issue = ''
        if winner == -1:
            issue = 'NO_DATA'
        elif winner != exp_gt:
            issue = f'WRONG(winner=GT{winner})'
        elif exp_rank > 1:
            issue = f'OK(rank={exp_rank})'

        line = (f'  {vox:4d} {vr["row"]:2d} {vr["col"]:2d} '
                f'GT{exp_gt:1d}  {grain_cols}  '
                f'GT{winner:1d}  '
                f'rank={exp_rank:2d}  {issue}')
        print(line)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    n_correct  = sum(1 for vr in voxel_rows
                     if vr['winner'] == vr['exp_gt'] and vr['winner'] >= 0)
    n_mismatch = len(mismatch_voxels)
    n_zero     = len(zero_conf_voxels)

    print(f'\n  {"="*60}')
    print(f'  SUMMARY')
    print(f'  {"="*60}')
    print(f'  Total voxels       : {n_voxels}')
    print(f'  Winner == ExpGT    : {n_correct}/{n_voxels} ({100*n_correct/n_voxels:.1f}%)')
    print(f'  Mismatches         : {n_mismatch}  {mismatch_voxels}')
    print(f'  Zero-conf for ExpGT: {n_zero}  {zero_conf_voxels}')
    print(f'  Ranked-last issues : {len(problem_voxels)}  {problem_voxels}')

    if mismatch_voxels:
        print(f'\n  MISMATCH DETAIL:')
        for vox in mismatch_voxels:
            vr = next(r for r in voxel_rows if r['vox'] == vox)
            exp = vr['exp_gt']
            exp_c = vr['exp_conf']
            win_c = vr['winner_conf']
            pos_x = pos_sorted[vr['row']]
            pos_y = pos_sorted[vr['col']]
            e = eulers[exp] if exp >= 0 else None
            print(f'    Voxel {vox:2d} ({pos_x:7.2f},{pos_y:7.2f})µm: '
                  f'expected GT{exp} (conf={exp_c:.3f}, rank={vr["exp_rank"]}) '
                  f'but winner=GT{vr["winner"]} (conf={win_c:.3f})')
            if e is not None:
                print(f'         ExpGT Euler=({e[0]:.1f},{e[1]:.1f},{e[2]:.1f})°')
            # Print all confs for this voxel
            ct_sorted = sorted(vr['confs'].items(), key=lambda kv: kv[1], reverse=True)
            print(f'         All confs (ranked): '
                  + ', '.join(f'GT{gi}={c:.3f}{"←exp" if gi==exp else ""}' for gi, c in ct_sorted))

    debug_params.unlink(missing_ok=True)
    debug_output = work_dir / '_debug_output'
    if debug_output.exists():
        shutil.rmtree(str(debug_output), ignore_errors=True)
    return voxel_rows


# ---------------------------------------------------------------------------
# Diagnostic 6: Cross-compare GT vs IndexBest_all.bin vs CONF_DEBUG
# ---------------------------------------------------------------------------
def cross_compare_gt_indexer_debug(work_dir, nCPUs=1):
    """Side-by-side comparison of three grain-assignment sources per voxel.

    For every voxel prints:
      - GT:      expected grain from EBSD nearest-neighbor mapping
      - BIN:     best-by-confidence grain from IndexBest_all.bin (pipeline run)
      - DEBUG:   best-by-confidence grain from CONF_DEBUG (Grains.csv seeded run)

    This reveals whether the binary output faithfully records what the indexer
    computed, and whether both agree with the ground truth.
    """
    print('\n' + '='*70)
    print('  Diagnostic 6: Cross-compare GT vs IndexBest_all.bin vs CONF_DEBUG')
    print('='*70)

    for f in [work_dir / 'Spots.bin', work_dir / 'hkls.csv',
              work_dir / 'positions.csv', work_dir / 'paramstest.txt']:
        if not f.exists():
            print(f'  ERROR: required file missing: {f}')
            return

    idx_bin = work_dir / 'Output' / 'IndexBest_all.bin'
    if not idx_bin.exists():
        print('  ERROR: IndexBest_all.bin not found — pipeline must run first')
        return

    gt = load_ground_truth(work_dir)
    n_grains = gt['nGrains']
    sgnum    = gt['sgnum']
    eulers   = gt['eulers_unique']
    ebsd_data = gt['voxel_data']

    positions = np.loadtxt(str(work_dir / 'positions.csv'))
    num_scans = len(positions)
    n_voxels  = num_scans * num_scans
    pos_sorted = np.sort(positions)

    # --- 1. Build expected-GT map (same as other diagnostics) ---
    def euler_to_gt(euler):
        for gi, gu in enumerate(eulers):
            if np.allclose(euler, gu):
                return gi
        return -1

    expected_gt = {}
    for vox in range(n_voxels):
        row = vox // num_scans
        col = vox % num_scans
        x_pos = pos_sorted[row]
        y_pos = pos_sorted[col]
        dists = (ebsd_data[:, 0] - x_pos)**2 + (ebsd_data[:, 1] - y_pos)**2
        nearest = np.argmin(dists)
        expected_gt[vox] = euler_to_gt(ebsd_data[nearest, 3:6])

    # --- 2. Read IndexBest_all.bin ---
    with open(str(idx_bin), 'rb') as f:
        nVoxBin = struct.unpack('i', f.read(4))[0]
        nSolArr = np.frombuffer(f.read(4 * nVoxBin), dtype=np.int32)
        offArr  = np.frombuffer(f.read(8 * nVoxBin), dtype=np.int64)
        header_size = 4 + 4 * nVoxBin + 8 * nVoxBin
        all_data = np.frombuffer(f.read(), dtype=np.float64)

    print(f'  IndexBest_all.bin: {nVoxBin} voxels, {nSolArr.sum()} total solutions')

    # Per-voxel: best solution from the binary
    bin_results = {}   # vox → {gt_grain, conf, nMatched, nTotal, miso}
    for v in range(nVoxBin):
        n_sol = nSolArr[v]
        if n_sol == 0:
            bin_results[v] = {'gt_grain': -1, 'conf': 0.0,
                              'nMatched': 0, 'nTotal': 0, 'miso': 999.0, 'n_sol': 0}
            continue

        offset_doubles = int((offArr[v] - header_size) // 8)
        sol_data = all_data[offset_doubles:offset_doubles + n_sol * 16].reshape(n_sol, 16)

        confs = np.where(sol_data[:, 14] > 0, sol_data[:, 15] / sol_data[:, 14], 0.0)
        best_idx = np.argmax(confs)
        best_om  = sol_data[best_idx, 2:11]

        # Find nearest GT grain by misorientation
        best_miso = 999.0
        best_gi = -1
        for gi, gt_om in enumerate(gt['orient_mats']):
            angle, _ = GetMisOrientationAngleOM(gt_om, best_om, sgnum)
            miso_deg = np.degrees(angle)
            if miso_deg < best_miso:
                best_miso = miso_deg
                best_gi = gi

        bin_results[v] = {
            'gt_grain': best_gi, 'conf': confs[best_idx],
            'nMatched': int(sol_data[best_idx, 15]),
            'nTotal': int(sol_data[best_idx, 14]),
            'miso': best_miso, 'n_sol': n_sol,
        }

    # --- 3. Run CONF_DEBUG (seeded with GT Grains.csv) ---
    grains_csv, debug_params, _, _ = _build_grains_csv_and_params(work_dir, gt)

    indexer_bin = MIDAS_HOME / 'FF_HEDM' / 'bin' / 'IndexerScanningOMP'
    cmd = [str(indexer_bin), str(debug_params), '0', '1', str(num_scans), str(nCPUs)]
    print(f'  Running CONF_DEBUG: {" ".join(cmd)}')
    env = dict(os.environ, MIDAS_DEBUG_INDEXER='1')
    result = subprocess.run(cmd, cwd=str(work_dir),
                            capture_output=True, text=True, timeout=300,
                            env=env)

    conf_all = _parse_conf_debug(result.stdout)
    n_parsed = sum(len(v) for v in conf_all.values())
    print(f'  Parsed {n_parsed} CONF_DEBUG entries across {len(conf_all)} voxels')

    # Per-voxel: best grain from debug
    dbg_results = {}   # vox → {gt_grain, conf, nMatched, nTotal}
    for vox in range(n_voxels):
        ct = conf_all.get(vox, {})
        if not ct:
            dbg_results[vox] = {'gt_grain': -1, 'conf': 0.0,
                                'nMatched': 0, 'nTotal': 0}
            continue
        ranked = sorted(ct.items(), key=lambda kv: kv[1]['conf'], reverse=True)
        winner_gi, winner_data = ranked[0]
        dbg_results[vox] = {
            'gt_grain': winner_gi, 'conf': winner_data['conf'],
            'nMatched': winner_data['nMatched'],
            'nTotal': winner_data['nTspots'],
        }

    # --- 4. Print combined table ---
    print(f'\n  {"Vox":>4} {"r":>2} {"c":>2} '
          f'{"pos_x":>7} {"pos_y":>7}  '
          f'{"ExpGT":>5}  '
          f'{"BIN_grain":>9} {"BIN_conf":>8} {"BIN_M/T":>8} {"BIN_miso":>8}  '
          f'{"DBG_grain":>9} {"DBG_conf":>8} {"DBG_M/T":>8}  '
          f'{"Status":>12}')
    print(f'  {"-"*120}')

    n_all_match = 0
    n_bin_wrong = 0
    n_dbg_wrong = 0
    n_bin_dbg_disagree = 0
    disagreement_voxels = []

    for vox in range(n_voxels):
        row = vox // num_scans
        col = vox % num_scans
        x_pos = pos_sorted[row]
        y_pos = pos_sorted[col]
        exp = expected_gt[vox]

        br = bin_results.get(vox, {'gt_grain': -1, 'conf': 0, 'nMatched': 0,
                                    'nTotal': 0, 'miso': 999, 'n_sol': 0})
        dr = dbg_results.get(vox, {'gt_grain': -1, 'conf': 0, 'nMatched': 0,
                                    'nTotal': 0})

        bin_grain = br['gt_grain']
        dbg_grain = dr['gt_grain']

        # Status flags
        bin_ok = (bin_grain == exp) if exp >= 0 else True
        dbg_ok = (dbg_grain == exp) if exp >= 0 else True
        agree  = (bin_grain == dbg_grain)

        if bin_ok and dbg_ok and agree:
            status = 'OK'
            n_all_match += 1
        elif not agree:
            status = 'BIN!=DBG'
            n_bin_dbg_disagree += 1
            disagreement_voxels.append(vox)
        elif not bin_ok:
            status = 'BIN_WRONG'
            n_bin_wrong += 1
        elif not dbg_ok:
            status = 'DBG_WRONG'
            n_dbg_wrong += 1
        else:
            status = 'BOTH_WRONG'

        bin_mt = f'{br["nMatched"]}/{br["nTotal"]}' if br['nTotal'] > 0 else '—'
        dbg_mt = f'{dr["nMatched"]}/{dr["nTotal"]}' if dr['nTotal'] > 0 else '—'

        # Only print header-rows and problem voxels in detail
        print(f'  {vox:4d} {row:2d} {col:2d} '
              f'{x_pos:7.1f} {y_pos:7.1f}  '
              f'GT{exp:1d}  '
              f'    GT{bin_grain:1d}  {br["conf"]:8.4f} {bin_mt:>8} {br["miso"]:7.2f}°  '
              f'    GT{dbg_grain:1d}  {dr["conf"]:8.4f} {dbg_mt:>8}  '
              f'{status:>12}')

    # --- 5. Identify all mismatched voxels (BIN winner != expected GT) ---
    mismatch_voxels = []
    for vox in range(n_voxels):
        br = bin_results.get(vox, {'gt_grain': -1})
        if br['gt_grain'] != expected_gt[vox]:
            mismatch_voxels.append(vox)

    # --- 6. Read ALL solutions per voxel for mismatch detail (conf + IA) ---
    def _read_all_solutions(vox):
        """Return list of dicts for every solution at this voxel."""
        n_sol = nSolArr[vox]
        if n_sol == 0:
            return []
        off = int((offArr[vox] - header_size) // 8)
        sol = all_data[off:off + n_sol * 16].reshape(n_sol, 16)
        results = []
        for i in range(n_sol):
            nT = int(sol[i, 14])
            nM = int(sol[i, 15])
            conf = nM / nT if nT > 0 else 0.0
            ia = sol[i, 1]
            om = sol[i, 2:11]
            best_gi, best_miso = -1, 999.0
            for gi, gt_om in enumerate(gt['orient_mats']):
                angle, _ = GetMisOrientationAngleOM(gt_om, om, sgnum)
                m = np.degrees(angle)
                if m < best_miso:
                    best_miso, best_gi = m, gi
            results.append({'gi': best_gi, 'conf': conf, 'ia': ia,
                            'nM': nM, 'nT': nT, 'miso': best_miso,
                            'spotid': int(sol[i, 0])})
        results.sort(key=lambda r: (-r['conf'], r['ia']))
        return results

    # --- 7. Print summary ---
    print(f'\n  {"="*60}')
    print(f'  CROSS-COMPARISON SUMMARY')
    print(f'  {"="*60}')
    print(f'  Total voxels       : {n_voxels}')
    print(f'  All three agree    : {n_all_match}/{n_voxels}')
    print(f'  BIN wrong (vs GT)  : {n_bin_wrong}')
    print(f'  DBG wrong (vs GT)  : {n_dbg_wrong}')
    print(f'  BIN != DBG         : {n_bin_dbg_disagree}  {disagreement_voxels}')
    print(f'  Mismatched voxels  : {len(mismatch_voxels)}')

    # --- 8. Mismatch detail table: winner vs expected conf/IA ---
    if mismatch_voxels:
        print(f'\n  MISMATCH DETAIL — Winner vs Expected (confidence and IA):')
        print(f'  {"Vox":>4} {"Pos":>12} {"ExpGT":>5}  '
              f'{"WinGT":>5} {"WinConf":>8} {"WinIA":>6} {"WinM/T":>8}  '
              f'{"ExpConf":>8} {"ExpIA":>6} {"ExpM/T":>8}  '
              f'{"dConf":>7} {"dIA":>6} {"dSpots":>6}  {"Cause":>12}')
        print(f'  {"-"*130}')

        for vox in mismatch_voxels:
            row = vox // num_scans
            col = vox % num_scans
            x_pos = pos_sorted[row]
            y_pos = pos_sorted[col]
            exp = expected_gt[vox]
            sols = _read_all_solutions(vox)
            if not sols:
                continue

            # Find winner and expected in the solution list
            winner = sols[0]
            exp_sol = next((s for s in sols if s['gi'] == exp), None)

            w_conf = winner['conf']
            w_ia = winner['ia']
            w_mt = f'{winner["nM"]}/{winner["nT"]}'
            e_conf = exp_sol['conf'] if exp_sol else 0.0
            e_ia = exp_sol['ia'] if exp_sol else 0.0
            e_mt = f'{exp_sol["nM"]}/{exp_sol["nT"]}' if exp_sol else '—'
            d_conf = w_conf - e_conf
            d_ia = w_ia - e_ia if exp_sol else 0.0
            d_spots = winner['nM'] - (exp_sol['nM'] if exp_sol else 0)

            # Classify cause
            if abs(d_conf) < 1e-6:
                cause = 'IA_TIEBREAK'
            elif d_conf < 0.05:
                cause = 'MARGINAL'
            elif e_conf == 0:
                cause = 'NOT_FOUND'
            else:
                cause = 'BOUNDARY'

            print(f'  {vox:4d} ({x_pos:5.0f},{y_pos:5.0f})  '
                  f'GT{exp}  '
                  f'  GT{winner["gi"]}  {w_conf:8.4f} {w_ia:6.3f} {w_mt:>8}  '
                  f'{e_conf:8.4f} {e_ia:6.3f} {e_mt:>8}  '
                  f'{d_conf:+7.4f} {d_ia:+6.3f} {d_spots:+6d}  '
                  f'{cause:>12}')

        # Cause summary
        causes = {}
        for vox in mismatch_voxels:
            sols = _read_all_solutions(vox)
            if not sols:
                continue
            winner = sols[0]
            exp_sol = next((s for s in sols if s['gi'] == expected_gt[vox]), None)
            d_conf = winner['conf'] - (exp_sol['conf'] if exp_sol else 0.0)
            if abs(d_conf) < 1e-6:
                c = 'IA_TIEBREAK'
            elif d_conf < 0.05:
                c = 'MARGINAL'
            elif exp_sol is None or exp_sol['conf'] == 0:
                c = 'NOT_FOUND'
            else:
                c = 'BOUNDARY'
            causes[c] = causes.get(c, 0) + 1

        print(f'\n  Mismatch cause breakdown:')
        for cause, count in sorted(causes.items()):
            print(f'    {cause:15s}: {count}')

    # --- 9. Generate diagnostic plots ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Build grids for plotting (indexer results)
    expected_grid = np.full((num_scans, num_scans), -1, dtype=int)
    recon_grid    = np.full((num_scans, num_scans), -1, dtype=int)
    conf_grid     = np.zeros((num_scans, num_scans))
    nmatched_grid = np.zeros((num_scans, num_scans), dtype=int)

    for vox in range(n_voxels):
        r, c = vox // num_scans, vox % num_scans
        expected_grid[r, c] = expected_gt[vox]
        br = bin_results.get(vox, {'gt_grain': -1, 'conf': 0, 'nMatched': 0})
        recon_grid[r, c] = br['gt_grain']
        conf_grid[r, c] = br['conf']
        nmatched_grid[r, c] = br['nMatched']

    # Build grids for after-refinement results (microstrFull.csv)
    refined_grid      = np.full((num_scans, num_scans), -1, dtype=int)
    refined_comp_grid = np.full((num_scans, num_scans), np.nan)
    refined_miso_grid = np.full((num_scans, num_scans), np.nan)
    refined_poserr_grid = np.full((num_scans, num_scans), np.nan)
    refined_omeerr_grid = np.full((num_scans, num_scans), np.nan)
    refined_ia_grid     = np.full((num_scans, num_scans), np.nan)
    refined_mismatch  = []
    has_refinement = False

    csv_path = work_dir / 'Recons' / 'microstrFull.csv'
    if csv_path.exists():
        ref_data = np.genfromtxt(str(csv_path), delimiter=',', skip_header=1)
        if ref_data.ndim == 1:
            ref_data = ref_data.reshape(1, -1)
        valid_mask = ~np.isnan(ref_data[:, 26]) & (ref_data[:, 26] >= 0)
        ref_data = ref_data[valid_mask]
        if len(ref_data) > 0:
            has_refinement = True
            print(f'\n  Post-refinement: {len(ref_data)} valid rows in microstrFull.csv')
            for row in ref_data:
                om = row[1:10]
                rx, ry = row[11], row[12]
                completeness = row[26]
                pos_err = row[22]   # PosErr
                ome_err = row[23]   # OmeErr
                ia_err  = row[24]   # InternalAngle
                # Map (rx, ry) to nearest grid cell
                ri = np.argmin(np.abs(pos_sorted - rx))
                ci = np.argmin(np.abs(pos_sorted - ry))
                # Find nearest GT grain
                best_gi, best_miso = -1, 999.0
                for gi, gt_om in enumerate(gt['orient_mats']):
                    angle, _ = GetMisOrientationAngleOM(gt_om, om, sgnum)
                    m = np.degrees(angle)
                    if m < best_miso:
                        best_miso, best_gi = m, gi
                refined_grid[ri, ci] = best_gi
                refined_comp_grid[ri, ci] = completeness
                refined_miso_grid[ri, ci] = best_miso
                refined_poserr_grid[ri, ci] = pos_err
                refined_omeerr_grid[ri, ci] = ome_err
                refined_ia_grid[ri, ci] = ia_err
                if best_gi != expected_grid[ri, ci]:
                    refined_mismatch.append((ri, ci))

            n_ref_correct = np.sum(refined_grid == expected_grid)
            print(f'  Refined winner == ExpGT: {n_ref_correct}/{n_voxels} '
                  f'({100*n_ref_correct/n_voxels:.1f}%)')
            print(f'  Refined mismatches: {len(refined_mismatch)}')
    else:
        print(f'\n  microstrFull.csv not found — skipping refinement panels')

    # EBSD native grid
    xs_ebsd = np.unique(ebsd_data[:, 0])
    ys_ebsd = np.unique(ebsd_data[:, 1])
    ebsd_grid = np.full((len(xs_ebsd), len(ys_ebsd)), -1, dtype=int)
    for i, x in enumerate(sorted(xs_ebsd)):
        for j, y in enumerate(sorted(ys_ebsd)):
            mask = (ebsd_data[:, 0] == x) & (ebsd_data[:, 1] == y)
            if mask.any():
                ebsd_grid[i, j] = euler_to_gt(ebsd_data[mask, 3:6][0])

    grain_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    grain_cmap = ListedColormap(grain_colors[:n_grains])
    grain_norm = BoundaryNorm(np.arange(-0.5, n_grains, 1), grain_cmap.N)

    ebsd_half = STEP_UM / 2
    scan_half = BEAMSIZE / 2
    ext = [pos_sorted[0] - scan_half, pos_sorted[-1] + scan_half,
           pos_sorted[0] - scan_half, pos_sorted[-1] + scan_half]
    # EBSD plotted on the same extent as reconstruction; white outside domain
    ebsd_ext = [xs_ebsd.min() - ebsd_half, xs_ebsd.max() + ebsd_half,
                ys_ebsd.min() - ebsd_half, ys_ebsd.max() + ebsd_half]
    domain_half = SIZE_UM / 2

    def _add_domain_rect(ax):
        """Add EBSD domain boundary to any panel."""
        rect = plt.Rectangle((-domain_half, -domain_half), SIZE_UM, SIZE_UM,
                              linewidth=2, edgecolor='red', facecolor='none',
                              linestyle='--')
        ax.add_patch(rect)

    n_cols = 3 if has_refinement else 2
    n_rows = 3 if has_refinement else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))

    # ─── Row 0: Grain maps ─────────────────────────────────────
    # Panel (0,0): EBSD GT — same extent as reconstruction, white outside
    ax = axes[0, 0]
    im = ax.imshow(ebsd_grid.T, origin='lower', cmap=grain_cmap, norm=grain_norm,
                   extent=ebsd_ext)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title(f'EBSD Ground Truth ({len(xs_ebsd)}x{len(ys_ebsd)}, '
                 f'{STEP_UM}um)', fontsize=13)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    _add_domain_rect(ax)
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_grains))
    cbar.set_ticklabels([f'GT{i}' for i in range(n_grains)])

    # Panel (0,1): Indexer Winner grain map
    ax = axes[0, 1]
    im = ax.imshow(recon_grid.T, origin='lower', cmap=grain_cmap,
                   norm=grain_norm, extent=ext)
    ax.set_title(f'Indexer Winner ({num_scans}x{num_scans}, '
                 f'{BEAMSIZE:.1f}um)', fontsize=13)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    for vox in mismatch_voxels:
        r, c = vox // num_scans, vox % num_scans
        ax.plot(pos_sorted[r], pos_sorted[c], 'kx', markersize=8,
                markeredgewidth=2)
    _add_domain_rect(ax)
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_grains))
    cbar.set_ticklabels([f'GT{i}' for i in range(n_grains)])
    if mismatch_voxels:
        ax.plot([], [], 'kx', markersize=8, markeredgewidth=2,
                label='GT mismatch')
        ax.legend(loc='upper right', fontsize=9)

    # ─── Row 1: Confidence / nMatched ──────────────────────────
    # Panel (1,0): Indexer Confidence map
    ax = axes[1, 0]
    im = ax.imshow(conf_grid.T, origin='lower', cmap='RdYlGn',
                   vmin=0, vmax=1, extent=ext)
    ax.set_title('Indexer Confidence (nMatched/nTotal)', fontsize=13)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    plt.colorbar(im, ax=ax, label='Confidence')
    _add_domain_rect(ax)
    for r in range(num_scans):
        for c in range(num_scans):
            if 0 < conf_grid[r, c] < 0.5:
                ax.text(pos_sorted[r], pos_sorted[c],
                        f'{conf_grid[r, c]:.2f}', ha='center', va='center',
                        fontsize=6, color='black')

    # Panel (1,1): nMatched map
    ax = axes[1, 1]
    n_total_max = nmatched_grid.max() if nmatched_grid.max() > 0 else 116
    im = ax.imshow(nmatched_grid.T, origin='lower', cmap='viridis',
                   extent=ext)
    ax.set_title(f'nMatched (out of {n_total_max} total spots)', fontsize=13)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    plt.colorbar(im, ax=ax, label='nMatched')
    _add_domain_rect(ax)

    if has_refinement:
        # Panel (0,2): Refined grain map
        ax = axes[0, 2]
        im = ax.imshow(refined_grid.T, origin='lower', cmap=grain_cmap,
                       norm=grain_norm, extent=ext)
        ax.set_title(f'Refined Winner ({num_scans}x{num_scans})', fontsize=13)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        for ri, ci in refined_mismatch:
            ax.plot(pos_sorted[ri], pos_sorted[ci], 'kx', markersize=8,
                    markeredgewidth=2)
        _add_domain_rect(ax)
        cbar = plt.colorbar(im, ax=ax, ticks=range(n_grains))
        cbar.set_ticklabels([f'GT{i}' for i in range(n_grains)])
        if refined_mismatch:
            ax.plot([], [], 'kx', markersize=8, markeredgewidth=2,
                    label='GT mismatch')
            ax.legend(loc='upper right', fontsize=9)

        # Panel (1,2): Refined completeness
        ax = axes[1, 2]
        comp_display = np.where(np.isnan(refined_comp_grid), 0, refined_comp_grid)
        im = ax.imshow(comp_display.T, origin='lower', cmap='RdYlGn',
                       vmin=0, vmax=1, extent=ext)
        ax.set_title('Refined Completeness', fontsize=13)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.colorbar(im, ax=ax, label='Completeness')
        _add_domain_rect(ax)
        for r in range(num_scans):
            for c in range(num_scans):
                v = refined_comp_grid[r, c]
                if not np.isnan(v) and 0 < v < 0.5:
                    ax.text(pos_sorted[r], pos_sorted[c],
                            f'{v:.2f}', ha='center', va='center',
                            fontsize=6, color='black')

        # ─── Row 2: Error maps ─────────────────────────────────
        # Panel (2,0): Position Error
        ax = axes[2, 0]
        pe = np.where(np.isnan(refined_poserr_grid), 0, refined_poserr_grid)
        im = ax.imshow(pe.T, origin='lower', cmap='hot_r', extent=ext)
        ax.set_title('Position Error', fontsize=13)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.colorbar(im, ax=ax, label='PosErr')
        _add_domain_rect(ax)

        # Panel (2,1): Omega Error
        ax = axes[2, 1]
        oe = np.where(np.isnan(refined_omeerr_grid), 0, refined_omeerr_grid)
        im = ax.imshow(oe.T, origin='lower', cmap='hot_r', extent=ext)
        ax.set_title('Omega Error', fontsize=13)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.colorbar(im, ax=ax, label='OmeErr')
        _add_domain_rect(ax)

        # Panel (2,2): Internal Angle
        ax = axes[2, 2]
        ia = np.where(np.isnan(refined_ia_grid), 0, refined_ia_grid)
        im = ax.imshow(ia.T, origin='lower', cmap='hot_r', extent=ext)
        ax.set_title('Internal Angle', fontsize=13)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.colorbar(im, ax=ax, label='IA')
        _add_domain_rect(ax)

    plt.suptitle(f'pf-HEDM Reconstruction ({num_scans}x{num_scans}, '
                 f'{BEAMSIZE:.1f}um beam) vs GT '
                 f'({len(xs_ebsd)}x{len(ys_ebsd)}, {STEP_UM}um EBSD)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = work_dir / 'diagnostic_maps.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved diagnostic plot: {plot_path}')

    debug_params.unlink(missing_ok=True)
    debug_output = work_dir / '_debug_output'
    if debug_output.exists():
        shutil.rmtree(str(debug_output), ignore_errors=True)
    return {'n_voxels': n_voxels, 'n_all_match': n_all_match,
            'n_bin_dbg_disagree': n_bin_dbg_disagree,
            'mismatch_voxels': mismatch_voxels}


# ---------------------------------------------------------------------------
# Diagnostic 4: Sinogram Comparison
# ---------------------------------------------------------------------------
def validate_sinograms(work_dir, gt, nCPUs):
    """Compare pipeline sinograms against GT-derived sinograms from zip files.

    Builds sinograms using forward_sim_sinogram.py with GT orientations,
    then compares to the pipeline's sinos_*.bin output.
    """
    print('\n' + '='*70)
    print('  Diagnostic 4: Sinogram Comparison')
    print('='*70)

    if gt is None:
        gt = load_ground_truth(work_dir)

    # --- Step 1: Generate SpotMatrixGen for each GT grain ---
    diag_dir = work_dir / '_diag_sino'
    diag_dir.mkdir(exist_ok=True)

    param_path = work_dir / 'Parameters_pfhedm.txt'
    with open(str(param_path)) as f:
        param_lines = f.readlines()

    lattice_str = '3.5950 3.5950 3.5950 90 90 90'
    omega_step = 0.25
    px_size = 200.0
    for line in param_lines:
        if line.startswith('LatticeConstant'):
            lattice_str = line.split(None, 1)[1].strip()
        elif line.startswith('OmegaStep'):
            omega_step = float(line.split()[1])
        elif line.startswith('px '):
            px_size = float(line.split()[1])

    lat_parts = lattice_str.split()
    a, b, c = float(lat_parts[0]), float(lat_parts[1]), float(lat_parts[2])
    al, be, ga = float(lat_parts[3]), float(lat_parts[4]), float(lat_parts[5])

    # Write all GT grains to Grains.csv
    grains_csv = diag_dir / 'Grains.csv'
    with open(str(grains_csv), 'w') as f:
        f.write(f'%NumGrains {gt["nGrains"]}\n')
        f.write('%BeamCenter 0.000000\n')
        f.write('%BeamThickness 1000.000000\n')
        f.write('%GlobalPosition 0.000000\n')
        f.write('%NumPhases 1\n')
        f.write('%PhaseInfo\n')
        f.write(f'%\tSpaceGroup:{gt["sgnum"]}\n')
        f.write(f'%\tLattice Parameter: {a:.6f} {b:.6f} {c:.6f} '
                f'{al:.6f} {be:.6f} {ga:.6f}\n')
        f.write('%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t'
                'X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\n')
        for i, om in enumerate(gt['orient_mats']):
            f.write(f'{i+1}\t{chr(9).join(f"{v:.6f}" for v in om)}\t'
                    f'0.0\t0.0\t0.0\t'
                    f'{a:.6f}\t{b:.6f}\t{c:.6f}\t'
                    f'{al:.6f}\t{be:.6f}\t{ga:.6f}\n')

    fwd_param = diag_dir / 'ps_diag.txt'
    with open(str(fwd_param), 'w') as f:
        for line in param_lines:
            key = line.split()[0] if line.strip() and not line.strip().startswith('#') else ''
            if key == 'nScans':
                f.write('nScans 1\n')
            elif key in ('WriteSpots', 'WriteImage', 'InFileName', 'InputFile'):
                continue
            elif key == 'PositionsFile':
                pos_f = diag_dir / 'positions.csv'
                pos_f.write_text('0.0\n')
                f.write(f'PositionsFile {pos_f}\n')
            else:
                f.write(line)
        f.write('WriteSpots 1\nWriteImage 0\nnScans 1\n')
        f.write(f'InFileName {grains_csv}\n')

    hkls_src = work_dir / 'hkls.csv'
    if not hkls_src.exists():
        hkls_src = work_dir / '1' / 'hkls.csv'
    if hkls_src.exists():
        shutil.copy2(str(hkls_src), str(diag_dir / 'hkls.csv'))

    bin_path = MIDAS_HOME / 'FF_HEDM' / 'bin' / 'ForwardSimulationCompressed'
    result = subprocess.run(
        [str(bin_path), str(fwd_param), str(nCPUs)],
        cwd=str(diag_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: ForwardSim for sinogram validation failed')
        shutil.rmtree(str(diag_dir), ignore_errors=True)
        return

    spot_file = diag_dir / 'SpotMatrixGen.csv'
    if not spot_file.exists():
        print('  ERROR: SpotMatrixGen.csv not produced')
        shutil.rmtree(str(diag_dir), ignore_errors=True)
        return

    pred = np.genfromtxt(str(spot_file), skip_header=1, delimiter='\t')
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)

    # --- Step 2: Build GT sinograms from zip files ---
    # For each GT grain, extract intensity from zip at predicted positions
    gt_grain_ids = sorted(np.unique(pred[:, 0]).astype(int))
    print(f'  Building GT sinograms for {len(gt_grain_ids)} grains '
          f'from {NSCANS} scan zip files...')

    gt_sinos = {}  # grainID -> (n_spots, n_scans) array
    gt_omegas = {}

    for gid in gt_grain_ids:
        grain_mask = pred[:, 0] == gid
        grain_spots = pred[grain_mask]
        n_spots_grain = len(grain_spots)
        sino = np.zeros((n_spots_grain, NSCANS), dtype=np.float64)
        omegas = grain_spots[:, 2]  # omega angles

        for scanNr in range(NSCANS):
            scan_dir = work_dir / str(scanNr + 1)
            zip_files = list(scan_dir.glob('*.MIDAS.zip'))
            if not zip_files:
                continue

            store = zarr.storage.ZipStore(str(zip_files[0]), mode='r')
            zg = zarr.open_group(store, mode='r')
            data_arr = zg['exchange/data']
            n_frames, nz, ny = data_arr.shape

            for si, row in enumerate(grain_spots):
                ome_bin = int(row[14]) if len(row) > 14 else int(round((row[2] + 180) / omega_step))
                det_hor = int(round(row[3]))
                det_vert = int(round(row[4]))
                patch_half = 5

                total_int = 0.0
                for fi in range(max(0, ome_bin - 1), min(n_frames, ome_bin + 2)):
                    r0 = max(0, det_vert - patch_half)
                    r1 = min(nz, det_vert + patch_half + 1)
                    c0 = max(0, det_hor - patch_half)
                    c1 = min(ny, det_hor + patch_half + 1)
                    if r0 < r1 and c0 < c1:
                        patch = np.array(data_arr[fi, r0:r1, c0:c1])
                        total_int += float(np.sum(patch))

                sino[si, scanNr] = total_int

            store.close()

        gt_sinos[gid] = sino
        gt_omegas[gid] = omegas

    # Print GT sinogram stats
    print(f'\n  GT Sinograms from zip files:')
    for gid in gt_grain_ids:
        sino = gt_sinos[gid]
        nz_cells = np.count_nonzero(sino)
        print(f'    GT Grain {gid}: {sino.shape[0]} spots × {sino.shape[1]} scans, '
              f'non-zero: {nz_cells}/{sino.size} ({nz_cells/sino.size*100:.1f}%), '
              f'max: {sino.max():.0f}')

    # --- Step 3: Load pipeline sinograms ---
    import glob as glob_module
    sino_files = glob_module.glob(str(work_dir / 'sinos_*.bin'))
    # Filter to base file (sinos_N_M_S.bin)
    sino_files = [f for f in sino_files
                  if os.path.basename(f).count('_') == 3
                  and os.path.basename(f).split('_')[1].isdigit()]

    if not sino_files:
        print('  Pipeline sinograms not found (sinos_*.bin) — skipping comparison')
        shutil.rmtree(str(diag_dir), ignore_errors=True)
        return

    sino_fn = sino_files[0]
    parts = os.path.basename(sino_fn).replace('.bin', '').split('_')
    nGrs = int(parts[1])
    maxNHKLs = int(parts[2])
    nScans_sino = int(parts[3])

    pipeline_sinos = np.fromfile(sino_fn, dtype=np.float64,
                                  count=nGrs * maxNHKLs * nScans_sino
                                  ).reshape((nGrs, maxNHKLs, nScans_sino))

    # Load pipeline grain spot counts
    nrhkls_files = glob_module.glob(str(work_dir / f'nrHKLs_{nGrs}.bin'))
    if nrhkls_files:
        grain_spots_counts = np.fromfile(nrhkls_files[0], dtype=np.int32, count=nGrs)
    else:
        grain_spots_counts = np.full(nGrs, maxNHKLs)

    print(f'\n  Pipeline sinograms: {nGrs} grains, {maxNHKLs} max HKLs, '
          f'{nScans_sino} scans')

    # --- Match pipeline grains to GT grains ---
    uq_path = work_dir / 'UniqueOrientations.csv'
    if not uq_path.exists():
        print('  UniqueOrientations.csv not found — skipping comparison')
        shutil.rmtree(str(diag_dir), ignore_errors=True)
        return

    uq = np.loadtxt(str(uq_path))
    if uq.ndim == 1:
        uq = uq.reshape(1, -1)

    sgnum = gt['sgnum']

    print(f'\n  Pipeline vs GT Sinogram Comparison:')
    print(f'  {"PipeGr":>6} {"MatchGT":>7} {"Miso":>6} {"GTSpots":>7} '
          f'{"PipeSp":>6} {"NZ_GT":>6} {"NZ_Pipe":>7} {"Corr":>6}')
    print(f'  {"-"*6} {"-"*7} {"-"*6} {"-"*7} {"-"*6} {"-"*6} {"-"*7} {"-"*6}')

    for pi, row in enumerate(uq):
        om = row[5:14]
        # Find matching GT grain
        best_miso = 999
        best_gt = -1
        for gi, gt_om in enumerate(gt['orient_mats']):
            angle, _ = GetMisOrientationAngleOM(gt_om, om, sgnum)
            angle_deg = np.degrees(angle)
            if angle_deg < best_miso:
                best_miso = angle_deg
                best_gt = gi

        n_pipe_spots = grain_spots_counts[pi] if pi < len(grain_spots_counts) else 0
        pipe_sino = pipeline_sinos[pi, :n_pipe_spots, :]
        nz_pipe = np.count_nonzero(pipe_sino)

        if best_miso < 2.0 and (best_gt + 1) in gt_sinos:
            gt_sino = gt_sinos[best_gt + 1]  # grainID is 1-indexed
            nz_gt = np.count_nonzero(gt_sino)
            n_gt_spots = gt_sino.shape[0]

            # Compute correlation on the scan-axis sum
            gt_profile = gt_sino.sum(axis=0)  # sum over spots → (nScans,)
            pipe_profile = pipe_sino.sum(axis=0)  # sum over spots → (nScans,)
            if np.std(gt_profile) > 0 and np.std(pipe_profile) > 0:
                corr = np.corrcoef(gt_profile, pipe_profile)[0, 1]
            else:
                corr = 0.0

            print(f'  {pi:6d}   GT{best_gt:1d}  {best_miso:5.2f}° {n_gt_spots:7d} '
                  f'{n_pipe_spots:6d} {nz_gt:6d} {nz_pipe:7d} {corr:6.3f}')
        else:
            print(f'  {pi:6d}   none  {best_miso:5.1f}°       - '
                  f'{n_pipe_spots:6d}      - {nz_pipe:7d}   - ← PHANTOM')

    shutil.rmtree(str(diag_dir), ignore_errors=True)


def validate_results(work_dir, doTomo=0):
    """Validate reconstruction output."""
    print('\n' + '='*70)
    print('  Step 5: Validating results')
    print('='*70)

    passed = True

    if doTomo == 1:
        # Check microstructure H5
        h5_path = work_dir / 'Recons' / 'microstructure.hdf'
        if h5_path.exists():
            import h5py
            with h5py.File(str(h5_path), 'r') as f:
                if 'images' in f:
                    imgs = f['images'][:]
                    print(f'  ✓ microstructure.hdf exists ({h5_path.stat().st_size / 1024:.0f} KB)')
                    print(f'    images shape: {imgs.shape}')
                else:
                    print(f'  ✗ microstructure.hdf exists but has no images dataset')
                    passed = False
        else:
            print(f'  ✗ microstructure.hdf not found at {h5_path}')
            passed = False

    # Check microstrFull.csv (always produced)
    csv_path = work_dir / 'Recons' / 'microstrFull.csv'
    if csv_path.exists():
        data = np.genfromtxt(str(csv_path), delimiter=',', skip_header=1)
        n_valid = np.sum(~np.isnan(data[:, 26])) if len(data.shape) > 1 else 0
        print(f'  ✓ microstrFull.csv exists ({csv_path.stat().st_size / 1024:.0f} KB)')
        if len(data.shape) > 1:
            print(f'    Total entries: {data.shape[0]}, with valid completeness: {n_valid}')
        else:
            print(f'    (Single row or empty)')
    else:
        print(f'  ✗ microstrFull.csv not found at {csv_path}')
        passed = False

    # Check peak search outputs
    n_spot_files = 0
    for scanNr in range(NSCANS):
        fn = work_dir / f'InputAllExtraInfoFittingAll{scanNr}.csv'
        if fn.exists():
            n_spot_files += 1
    print(f'  Peak search CSVs: {n_spot_files}/{NSCANS}')
    if n_spot_files == 0:
        print(f'  ✗ No peak search outputs found')
        passed = False

    # Check Spots.bin
    spots_bin = work_dir / 'Spots.bin'
    if spots_bin.exists():
        print(f'  ✓ Spots.bin exists ({spots_bin.stat().st_size / 1024:.0f} KB)')
    else:
        print(f'  ✗ Spots.bin not found')

    # Check IndexBest_all.bin (consolidated indexer output)
    idx_bin = work_dir / 'Output' / 'IndexBest_all.bin'
    if idx_bin.exists():
        print(f'  ✓ IndexBest_all.bin exists ({idx_bin.stat().st_size / 1024:.0f} KB)')
    else:
        print(f'  ✗ IndexBest_all.bin not found')

    # Check Results CSVs (refinement output)
    results_dir = work_dir / 'Results'
    if results_dir.exists():
        result_csvs = list(results_dir.glob('*.csv'))
        print(f'  ✓ Results/ directory: {len(result_csvs)} CSV files')
    else:
        print(f'  ✗ Results/ directory not found')
        passed = False

    if doTomo == 1:
        # Check sinogram output
        sinos_dir = work_dir / 'Sinos'
        if sinos_dir.exists():
            sino_files = list(sinos_dir.glob('*.tif'))
            print(f'  ✓ Sinos/ directory: {len(sino_files)} TIF files')
        else:
            print(f'  ✗ Sinos/ directory not found')

        # Check recon output
        recons_dir = work_dir / 'Recons'
        if recons_dir.exists():
            recon_files = list(recons_dir.glob('recon_grNr_*.tif'))
            print(f'  ✓ Recons/ directory: {len(recon_files)} reconstruction TIF files')
        else:
            print(f'  ✗ Recons/ directory not found')

    print()
    if passed:
        print('  *** pf-HEDM RECONSTRUCTION TEST PASSED ***')
    else:
        print('  *** pf-HEDM RECONSTRUCTION TEST FAILED ***')
        print('  (Some output files are missing — check pipeline logs above)')

    return passed


def copy_outputs(work_dir, dest_dir):
    """Copy pipeline output directories for comparison."""
    for subdir in ['Output', 'Results', 'Recons']:
        src = work_dir / subdir
        dst = dest_dir / subdir
        if src.exists():
            if dst.exists():
                shutil.rmtree(str(dst))
            shutil.copytree(str(src), str(dst))


def compare_gpu_omp(omp_dir, gpu_dir):
    """Compare GPU and OMP pipeline outputs numerically.

    Returns True if all differences are within tolerance.
    """
    print('\n' + '='*70)
    print('  GPU vs OMP Parity Comparison')
    print('='*70)

    passed = True

    # Compare microstrFull.csv
    omp_csv = omp_dir / 'Recons' / 'microstrFull.csv'
    gpu_csv = gpu_dir / 'Recons' / 'microstrFull.csv'

    if not omp_csv.exists() or not gpu_csv.exists():
        print(f'  ✗ Cannot compare: OMP csv exists={omp_csv.exists()}, GPU csv exists={gpu_csv.exists()}')
        return False

    omp_data = np.genfromtxt(str(omp_csv), delimiter=',', skip_header=1)
    gpu_data = np.genfromtxt(str(gpu_csv), delimiter=',', skip_header=1)

    if omp_data.shape != gpu_data.shape:
        print(f'  ✗ Shape mismatch: OMP={omp_data.shape}, GPU={gpu_data.shape}')
        return False

    print(f'  Rows: OMP={omp_data.shape[0]}, GPU={gpu_data.shape[0]}')

    # Filter valid rows (both must have valid completeness at column 26)
    omp_valid = ~np.isnan(omp_data[:, 26]) if len(omp_data.shape) > 1 else np.array([])
    gpu_valid = ~np.isnan(gpu_data[:, 26]) if len(gpu_data.shape) > 1 else np.array([])
    both_valid = omp_valid & gpu_valid

    n_both = np.sum(both_valid)
    print(f'  Valid in both: {n_both}')

    if n_both == 0:
        print('  ✗ No valid rows to compare')
        return False

    omp_v = omp_data[both_valid]
    gpu_v = gpu_data[both_valid]

    # Orientation matrix: columns 1-9
    om_diff = np.max(np.abs(omp_v[:, 1:10] - gpu_v[:, 1:10]))
    om_ok = om_diff < 0.01
    print(f'  Orientation matrix max diff: {om_diff:.6f} {"✓" if om_ok else "✗"} (tol=0.01)')
    passed = passed and om_ok

    # Position: columns 11-13 (x, y, z in µm)
    pos_diff = np.max(np.abs(omp_v[:, 11:14] - gpu_v[:, 11:14]))
    pos_ok = pos_diff < 1.0
    print(f'  Position max diff: {pos_diff:.6f} µm {"✓" if pos_ok else "✗"} (tol=1.0 µm)')
    passed = passed and pos_ok

    # Lattice parameters: columns 15-20 (a, b, c, alpha, beta, gamma)
    # Use 0.01% relative tolerance
    omp_lat = omp_v[:, 15:21]
    gpu_lat = gpu_v[:, 15:21]
    lat_mask = np.abs(omp_lat) > 1e-10  # avoid division by zero
    if np.any(lat_mask):
        lat_rel_diff = np.max(np.abs((omp_lat[lat_mask] - gpu_lat[lat_mask]) / omp_lat[lat_mask])) * 100
        lat_ok = lat_rel_diff < 0.01
        print(f'  Lattice param max rel diff: {lat_rel_diff:.6f}% {"✓" if lat_ok else "✗"} (tol=0.01%)')
        passed = passed and lat_ok
    else:
        print('  Lattice params: skipped (all zero)')

    # Completeness: column 26
    comp_diff = np.max(np.abs(omp_v[:, 26] - gpu_v[:, 26]))
    comp_ok = comp_diff < 0.05
    print(f'  Completeness max diff: {comp_diff:.6f} {"✓" if comp_ok else "✗"} (tol=0.05)')
    passed = passed and comp_ok

    # Strain: columns 27-35
    strain_diff = np.max(np.abs(omp_v[:, 27:36] - gpu_v[:, 27:36]))
    strain_ok = strain_diff < 0.001
    print(f'  Strain tensor max diff: {strain_diff:.9f} {"✓" if strain_ok else "✗"} (tol=0.001)')
    passed = passed and strain_ok

    print()
    if passed:
        print('  *** GPU vs OMP PARITY TEST PASSED ***')
    else:
        print('  *** GPU vs OMP PARITY TEST FAILED ***')

    return passed


def cleanup(work_dir):
    """Remove generated test artifacts."""
    print('\n  Cleaning up test artifacts...')
    if work_dir.exists():
        shutil.rmtree(str(work_dir))
    print('  Cleanup complete.')


def run_gpu_binaries(work_dir, nCPUs):
    """Run GPU indexer and refiner directly on prepared data.

    Expects Spots.bin, Data.bin, nData.bin, paramstest.txt, hkls.csv,
    SpotsToIndex.csv, positions.csv to already exist in work_dir.
    """
    print('\n' + '='*70)
    print('  Running GPU binaries directly')
    print('='*70)

    bin_dir = MIDAS_HOME / 'FF_HEDM' / 'bin'

    # --- IndexerScanningGPU ---
    indexer = bin_dir / 'IndexerScanningGPU'
    if not indexer.exists():
        print(f'  ERROR: {indexer} not found')
        return False

    cmd = f"{indexer} paramstest.txt 0 1 {NSCANS} {nCPUs}"
    print(f'  IndexerScanningGPU: {cmd}')
    result = subprocess.run(cmd, shell=True, cwd=str(work_dir),
                            capture_output=True, text=True)
    print(f'  stdout: {result.stdout[:500]}' if result.stdout else '  (no stdout)')
    if result.stderr:
        print(f'  stderr: {result.stderr[:500]}')
    if result.returncode != 0:
        print(f'  ERROR: IndexerScanningGPU failed with return code {result.returncode}')
        return False

    # Check IndexBest_all.bin was produced
    idx_bin = work_dir / 'Output' / 'IndexBest_all.bin'
    if not idx_bin.exists():
        print(f'  ERROR: IndexBest_all.bin not produced by GPU indexer')
        return False
    print(f'  ✓ IndexBest_all.bin produced ({idx_bin.stat().st_size / 1024:.0f} KB)')

    # --- findSingleSolutionPFRefactored (same for both, uses IndexBest_all.bin) ---
    # Read params to get needed values
    params = parse_parameter_file(work_dir / 'paramstest.txt')
    sgnum = params.get('SpaceGroup', 225)
    maxang = 1.0
    tol_ome = params.get('MarginOme', 1.0)
    tol_eta = params.get('MarginEta', 1.0)

    fss = bin_dir / 'findSingleSolutionPFRefactored'
    cmd = f"{fss} {work_dir} {sgnum} {maxang} {NSCANS} {nCPUs} {tol_ome} {tol_eta} Parameters_pfhedm.txt 2 1 0"
    print(f'  findSingleSolutionPF: {cmd}')
    result = subprocess.run(cmd, shell=True, cwd=str(work_dir),
                            capture_output=True, text=True)
    if result.stdout:
        print(f'  stdout (last 300): ...{result.stdout[-300:]}')
    if result.returncode != 0:
        print(f'  WARNING: findSingleSolutionPFRefactored returned {result.returncode}')

    # --- FitOrStrainsScanningGPU ---
    spots_file = work_dir / 'SpotsToIndex.csv'
    if not spots_file.exists():
        print(f'  ERROR: SpotsToIndex.csv not found')
        return False
    with open(str(spots_file), 'r') as f:
        num_spots = len(f.readlines())
    print(f'  SpotsToIndex.csv: {num_spots} spots')

    refiner = bin_dir / 'FitOrStrainsScanningGPU'
    if not refiner.exists():
        print(f'  ERROR: {refiner} not found')
        return False

    cmd = f"{refiner} paramstest.txt 0 1 {num_spots} {nCPUs}"
    print(f'  FitOrStrainsScanningGPU: {cmd}')
    result = subprocess.run(cmd, shell=True, cwd=str(work_dir),
                            capture_output=True, text=True)
    if result.stdout:
        print(f'  stdout: {result.stdout[:500]}')
    if result.stderr:
        print(f'  stderr: {result.stderr[:500]}')
    if result.returncode != 0:
        print(f'  ERROR: FitOrStrainsScanningGPU failed with return code {result.returncode}')
        return False

    result_csvs = list((work_dir / 'Results').glob('*.csv'))
    print(f'  ✓ Results/: {len(result_csvs)} CSV files')
    return True


def compare_consolidated_bins(omp_dir, gpu_dir, label='IndexBest_all.bin'):
    """Compare consolidated binary files between OMP and GPU."""
    omp_file = omp_dir / 'Output' / label
    gpu_file = gpu_dir / 'Output' / label
    if not omp_file.exists() or not gpu_file.exists():
        print(f'  ✗ Cannot compare {label}: OMP exists={omp_file.exists()}, GPU exists={gpu_file.exists()}')
        return False

    omp_data = np.fromfile(str(omp_file), dtype=np.uint8)
    gpu_data = np.fromfile(str(gpu_file), dtype=np.uint8)
    if omp_data.shape != gpu_data.shape:
        print(f'  ✗ {label}: size mismatch OMP={omp_data.shape[0]} vs GPU={gpu_data.shape[0]} bytes')
        return False

    n_diff = np.sum(omp_data != gpu_data)
    if n_diff == 0:
        print(f'  ✓ {label}: binary identical ({omp_data.shape[0]} bytes)')
        return True
    else:
        print(f'  ≈ {label}: {n_diff}/{omp_data.shape[0]} bytes differ (reading as doubles for numerical comparison)')
        # Try numerical comparison on the data section
        # The header is: 4 bytes (nVoxels int32) + 4*nVoxels (nSolArr int32) + 8*nVoxels (offArr int64)
        # Then the rest is doubles
        with open(str(omp_file), 'rb') as f:
            nVox_omp = np.frombuffer(f.read(4), dtype=np.int32)[0]
        with open(str(gpu_file), 'rb') as f:
            nVox_gpu = np.frombuffer(f.read(4), dtype=np.int32)[0]
        print(f'    nVoxels: OMP={nVox_omp}, GPU={nVox_gpu}')
        return False  # Not identical, but report the diff


# ---------------------------------------------------------------------------
# Seeded vs Unseeded Comparison
# ---------------------------------------------------------------------------
def run_seeded_comparison(work_dir, nCPUs):
    """Re-run indexer+findSingleSolution+refiner seeded with GT orientations.

    Saves the unseeded pipeline results, reruns the pipeline with a
    Grains.csv built from GT, then compares per-voxel grain assignments
    and confidences side by side.
    """
    import glob as glob_module

    print('\n' + '#'*70)
    print('  SEEDED vs UNSEEDED COMPARISON')
    print('#'*70)

    for f in [work_dir / 'Spots.bin', work_dir / 'hkls.csv',
              work_dir / 'paramstest.txt', work_dir / 'positions.csv']:
        if not f.exists():
            print(f'  ERROR: required file missing: {f}')
            return

    gt = load_ground_truth(work_dir)
    n_grains = gt['nGrains']
    sgnum = gt['sgnum']
    eulers = gt['eulers_unique']
    ebsd_data = gt['voxel_data']

    positions = np.loadtxt(str(work_dir / 'positions.csv'))
    num_scans = len(positions)
    n_voxels = num_scans * num_scans
    pos_sorted = np.sort(positions)

    # --- Expected GT map ---
    def euler_to_gt(euler):
        for gi, gu in enumerate(eulers):
            if np.allclose(euler, gu):
                return gi
        return -1

    expected_gt = {}
    for vox in range(n_voxels):
        row, col = vox // num_scans, vox % num_scans
        x_pos, y_pos = pos_sorted[row], pos_sorted[col]
        dists = (ebsd_data[:, 0] - x_pos)**2 + (ebsd_data[:, 1] - y_pos)**2
        expected_gt[vox] = euler_to_gt(ebsd_data[np.argmin(dists), 3:6])

    # === 1. Save unseeded results ===
    print('\n  Saving unseeded results...')
    unseeded_dir = work_dir / '_unseeded_results'
    unseeded_dir.mkdir(exist_ok=True)
    copy_outputs(work_dir, unseeded_dir)
    for fname in ['SpotsToIndex.csv', 'UniqueOrientations.csv',
                  'singleSolution.mic']:
        src = work_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(unseeded_dir / fname))
    # Save microstrFull.csv
    mf_src = work_dir / 'Recons' / 'microstrFull.csv'
    if mf_src.exists():
        shutil.copy2(str(mf_src), str(unseeded_dir / 'microstrFull.csv'))

    # === 2. Create GT Grains.csv ===
    param_path = work_dir / 'Parameters_pfhedm.txt'
    lattice_str = '4.080000 4.080000 4.080000 90.000000 90.000000 90.000000'
    with open(str(param_path)) as fh:
        for line in fh:
            if line.startswith('LatticeConstant'):
                lattice_str = line.split(None, 1)[1].strip()
    lat_parts = lattice_str.split()
    a, b, c = float(lat_parts[0]), float(lat_parts[1]), float(lat_parts[2])
    al, be, ga = float(lat_parts[3]), float(lat_parts[4]), float(lat_parts[5])

    grains_csv = work_dir / 'Grains_GT.csv'
    with open(str(grains_csv), 'w') as fh:
        fh.write(f'%NumGrains {n_grains}\n')
        fh.write('%BeamCenter 0.000000\n')
        fh.write('%BeamThickness 1000.000000\n')
        fh.write('%GlobalPosition 0.000000\n')
        fh.write('%NumPhases 1\n')
        fh.write('%PhaseInfo\n')
        fh.write(f'%\tSpaceGroup:{sgnum}\n')
        fh.write(f'%\tLattice Parameter: {a:.6f} {b:.6f} {c:.6f} '
                 f'{al:.6f} {be:.6f} {ga:.6f}\n')
        fh.write('%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t'
                 'X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\n')
        for i, om in enumerate(gt['orient_mats']):
            fh.write(f'{i+1}\t{chr(9).join(f"{v:.6f}" for v in om)}\t'
                     f'0.0\t0.0\t0.0\t'
                     f'{a:.6f}\t{b:.6f}\t{c:.6f}\t'
                     f'{al:.6f}\t{be:.6f}\t{ga:.6f}\n')
    print(f'  Created {grains_csv} with {n_grains} GT orientations')

    # === 3. Modify paramstest.txt to add GrainsFile ===
    params_path = work_dir / 'paramstest.txt'
    with open(str(params_path)) as fh:
        lines = fh.readlines()
    lines = [l for l in lines if not l.startswith('GrainsFile')
             and not l.startswith('MicFile')]
    lines.append(f'GrainsFile {grains_csv}\n')
    with open(str(params_path), 'w') as fh:
        fh.writelines(lines)
    print(f'  Added GrainsFile to paramstest.txt')

    # === 4. Clear indexer/refiner outputs ===
    for pattern in ['Output/IndexBest*', 'Output/IndexBestFull*',
                    'SpotsToIndex.csv', 'UniqueOrientations.csv',
                    'singleSolution.mic', 'sinos_*.bin', 'omegas_*.bin',
                    'nrHKLs_*.bin', 'spotMapping_*.bin', 'spotMeta_*.bin',
                    'spotPositions_*.bin', 'patches_*.bin']:
        for f in glob_module.glob(str(work_dir / pattern)):
            os.remove(f)
    results_d = work_dir / 'Results'
    if results_d.exists():
        shutil.rmtree(str(results_d))
    results_d.mkdir(parents=True, exist_ok=True)

    # === 5. Run seeded pipeline ===
    bin_dir = MIDAS_HOME / 'FF_HEDM' / 'bin'

    # 5a. IndexerScanningOMP (seeded with GT)
    print(f'\n  Running seeded IndexerScanningOMP...')
    cmd = [str(bin_dir / 'IndexerScanningOMP'), 'paramstest.txt',
           '0', '1', str(num_scans), str(nCPUs)]
    print(f'  Command: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=str(work_dir), capture_output=True,
                            text=True, timeout=120)
    if result.returncode != 0:
        print(f'  ERROR: seeded indexer failed (rc={result.returncode})')
        print(result.stderr[-500:] if result.stderr else '')
        return

    idx_bin = work_dir / 'Output' / 'IndexBest_all.bin'
    if not idx_bin.exists():
        print('  ERROR: IndexBest_all.bin not produced')
        return
    print(f'  IndexBest_all.bin: {idx_bin.stat().st_size / 1024:.0f} KB')

    # 5b. findSingleSolutionPFRefactored
    print(f'\n  Running findSingleSolutionPFRefactored...')
    params = parse_parameter_file(str(params_path))
    # Strip trailing semicolons from MIDAS parameter values
    def _clean_param(val, default):
        if isinstance(val, list):
            val = val[0]
        if isinstance(val, str):
            val = val.rstrip(';')
            try:
                return float(val)
            except ValueError:
                return default
        return val
    tol_ome = _clean_param(params.get('MarginOme', 1.0), 1.0)
    tol_eta = _clean_param(params.get('MarginEta', 1.0), 1.0)
    cmd = [str(bin_dir / 'findSingleSolutionPFRefactored'),
           str(work_dir), str(sgnum), '1.0', str(num_scans), str(nCPUs),
           str(tol_ome), str(tol_eta), 'Parameters_pfhedm.txt', '2', '1', '0']
    print(f'  Command: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=str(work_dir), capture_output=True,
                            text=True, timeout=120)
    if result.returncode != 0:
        print(f'  WARNING: findSingleSolution returned {result.returncode}')

    # Print key lines from findSingleSolution output
    for line in result.stdout.splitlines():
        if any(k in line for k in ['unique orientations', 'Grain ', 'Voxel',
                                    'Total spots', 'Number of']):
            print(f'    {line}')

    # 5b2. Generate SpotsToIndex.csv from UniqueIndexSingleKey.bin
    #      (pf_MIDAS.py does this between findSingleSolution and refinement)
    key_bin = work_dir / 'Output' / 'UniqueIndexSingleKey.bin'
    spots_file = work_dir / 'SpotsToIndex.csv'
    if key_bin.exists():
        id_data = np.fromfile(str(key_bin), dtype=np.uintp,
                              count=n_voxels * 5).reshape((-1, 5))
        with open(str(spots_file), 'w') as fh:
            for vox in range(n_voxels):
                if id_data[vox, 1] != 0:
                    fh.write(f'{id_data[vox, 0]} {id_data[vox, 1]} '
                             f'{id_data[vox, 2]} {id_data[vox, 3]} '
                             f'{id_data[vox, 4]}\n')
        print(f'  Generated SpotsToIndex.csv from UniqueIndexSingleKey.bin')
    else:
        print(f'  ERROR: UniqueIndexSingleKey.bin not found')
        return

    # 5c. FitOrStrainsScanningOMP
    if not spots_file.exists():
        print('  ERROR: SpotsToIndex.csv not produced')
        return
    with open(str(spots_file)) as fh:
        n_spots_to_refine = len(fh.readlines())
    print(f'\n  Running FitOrStrainsScanningOMP ({n_spots_to_refine} spots)...')
    cmd = [str(bin_dir / 'FitOrStrainsScanningOMP'), 'paramstest.txt',
           '0', '1', str(n_spots_to_refine), str(nCPUs)]
    result = subprocess.run(cmd, cwd=str(work_dir), capture_output=True,
                            text=True, timeout=120)
    if result.returncode != 0:
        print(f'  WARNING: refiner returned {result.returncode}')

    # === 6. Read both IndexBest_all.bin and compare ===
    def _read_index_best(path):
        """Read IndexBest_all.bin, return per-voxel best grain + confidence."""
        with open(str(path), 'rb') as f:
            nV = struct.unpack('i', f.read(4))[0]
            nSolArr = np.frombuffer(f.read(4 * nV), dtype=np.int32)
            offArr = np.frombuffer(f.read(8 * nV), dtype=np.int64)
            header_size = 4 + 4 * nV + 8 * nV
            all_data = np.frombuffer(f.read(), dtype=np.float64)
        results = {}
        for v in range(nV):
            n_sol = nSolArr[v]
            if n_sol == 0:
                results[v] = {'gt_grain': -1, 'conf': 0, 'n_sol': 0,
                              'nMatched': 0, 'nTotal': 0, 'miso': 999}
                continue
            off = int((offArr[v] - header_size) // 8)
            sol = all_data[off:off + n_sol * 16].reshape(n_sol, 16)
            confs = np.where(sol[:, 14] > 0, sol[:, 15] / sol[:, 14], 0.0)
            best = np.argmax(confs)
            om = sol[best, 2:11]
            best_gi, best_miso = -1, 999.0
            for gi, gt_om in enumerate(gt['orient_mats']):
                angle, _ = GetMisOrientationAngleOM(gt_om, om, sgnum)
                m = np.degrees(angle)
                if m < best_miso:
                    best_miso, best_gi = m, gi
            results[v] = {'gt_grain': best_gi, 'conf': confs[best],
                          'n_sol': n_sol, 'miso': best_miso,
                          'nMatched': int(sol[best, 15]),
                          'nTotal': int(sol[best, 14])}
        return results

    unseeded_idx = _read_index_best(unseeded_dir / 'Output' / 'IndexBest_all.bin')
    seeded_idx = _read_index_best(work_dir / 'Output' / 'IndexBest_all.bin')

    # === 7. Read both microstrFull.csv (refinement results) ===
    def _read_refined(csv_path):
        """Read microstrFull.csv, return per-grid-cell grain + completeness."""
        grid_grain = np.full((num_scans, num_scans), -1, dtype=int)
        grid_comp = np.full((num_scans, num_scans), np.nan)
        if not csv_path.exists():
            return grid_grain, grid_comp
        data = np.genfromtxt(str(csv_path), delimiter=',', skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        valid = ~np.isnan(data[:, 26]) & (data[:, 26] >= 0)
        data = data[valid]
        for row in data:
            om = row[1:10]
            ri = np.argmin(np.abs(pos_sorted - row[11]))
            ci = np.argmin(np.abs(pos_sorted - row[12]))
            best_gi, best_miso = -1, 999.0
            for gi, gt_om in enumerate(gt['orient_mats']):
                angle, _ = GetMisOrientationAngleOM(gt_om, om, sgnum)
                m = np.degrees(angle)
                if m < best_miso:
                    best_miso, best_gi = m, gi
            grid_grain[ri, ci] = best_gi
            grid_comp[ri, ci] = row[26]
        return grid_grain, grid_comp

    # Regenerate microstrFull.csv for seeded run
    # (pf_MIDAS.py consolidation step — do it manually here)
    seeded_results = list((work_dir / 'Results').glob('*.csv'))
    if seeded_results:
        # Simple consolidation: read line 2 from each FitBest csv
        rows = []
        for csv_f in sorted(seeded_results):
            with open(str(csv_f)) as fh:
                lines_csv = fh.readlines()
            if len(lines_csv) >= 2:
                vals = lines_csv[1].strip().split('\t')
                rows.append(','.join(vals))
        seeded_mf = work_dir / 'Recons' / 'microstrFull_seeded.csv'
        (work_dir / 'Recons').mkdir(exist_ok=True)
        with open(str(seeded_mf), 'w') as fh:
            # Write header from any FitBest file
            with open(str(seeded_results[0])) as fh2:
                header = fh2.readline().strip().replace('\t', ',')
            fh.write(header + '\n')
            fh.write('\n'.join(rows) + '\n')
        seeded_ref_grain, seeded_ref_comp = _read_refined(seeded_mf)
    else:
        seeded_ref_grain = np.full((num_scans, num_scans), -1, dtype=int)
        seeded_ref_comp = np.full((num_scans, num_scans), np.nan)

    unseeded_mf = unseeded_dir / 'microstrFull.csv'
    unseeded_ref_grain, unseeded_ref_comp = _read_refined(unseeded_mf)

    # === 8. Print comparison table ===
    print(f'\n  {"="*100}')
    print(f'  SEEDED vs UNSEEDED — Per-voxel Comparison')
    print(f'  {"="*100}')
    print(f'  {"Vox":>4} {"r":>2} {"c":>2} {"Pos":>12} {"ExpGT":>5}  '
          f'{"UNS_idx":>7} {"UNS_cf":>6} {"UNS_M/T":>8}  '
          f'{"SEED_idx":>8} {"SEED_cf":>7} {"SEED_M/T":>8}  '
          f'{"UNS_ref":>7} {"SEED_ref":>8}  {"":>10}')
    print(f'  {"-"*120}')

    n_idx_agree = 0
    n_idx_seed_correct = 0
    n_idx_uns_correct = 0
    n_ref_seed_correct = 0
    n_ref_uns_correct = 0
    diff_voxels = []

    for vox in range(n_voxels):
        row, col = vox // num_scans, vox % num_scans
        x_pos, y_pos = pos_sorted[row], pos_sorted[col]
        exp = expected_gt[vox]

        u = unseeded_idx.get(vox, {'gt_grain': -1, 'conf': 0, 'nMatched': 0,
                                    'nTotal': 0})
        s = seeded_idx.get(vox, {'gt_grain': -1, 'conf': 0, 'nMatched': 0,
                                  'nTotal': 0})

        u_ref = unseeded_ref_grain[row, col]
        s_ref = seeded_ref_grain[row, col]

        idx_agree = (u['gt_grain'] == s['gt_grain'])
        if idx_agree:
            n_idx_agree += 1
        if u['gt_grain'] == exp:
            n_idx_uns_correct += 1
        if s['gt_grain'] == exp:
            n_idx_seed_correct += 1
        if u_ref == exp:
            n_ref_uns_correct += 1
        if s_ref == exp:
            n_ref_seed_correct += 1

        # Determine status
        if idx_agree and u['gt_grain'] == exp:
            status = ''
        elif not idx_agree and s['gt_grain'] == exp and u['gt_grain'] != exp:
            status = 'SEED_FIXES'
            diff_voxels.append(vox)
        elif not idx_agree and s['gt_grain'] != exp and u['gt_grain'] == exp:
            status = 'SEED_BREAKS'
            diff_voxels.append(vox)
        elif not idx_agree:
            status = 'BOTH_WRONG' if s['gt_grain'] != exp else 'SEED_FIXES'
            diff_voxels.append(vox)
        else:
            status = 'BOTH_WRONG'
            diff_voxels.append(vox)

        # Only print rows where indexer disagrees or either is wrong
        if not idx_agree or u['gt_grain'] != exp or s['gt_grain'] != exp:
            u_mt = f'{u["nMatched"]}/{u["nTotal"]}' if u['nTotal'] > 0 else '—'
            s_mt = f'{s["nMatched"]}/{s["nTotal"]}' if s['nTotal'] > 0 else '—'
            u_ref_s = f'GT{u_ref}' if u_ref >= 0 else '—'
            s_ref_s = f'GT{s_ref}' if s_ref >= 0 else '—'
            print(f'  {vox:4d} {row:2d} {col:2d} '
                  f'({x_pos:5.0f},{y_pos:5.0f})  '
                  f'GT{exp}  '
                  f'  GT{u["gt_grain"]:1d}  {u["conf"]:.4f} {u_mt:>8}  '
                  f'   GT{s["gt_grain"]:1d}  {s["conf"]:7.4f} {s_mt:>8}  '
                  f'  {u_ref_s:>5}   {s_ref_s:>6}  '
                  f'{status:>12}')

    # === 9. Summary ===
    print(f'\n  {"="*70}')
    print(f'  SUMMARY')
    print(f'  {"="*70}')
    print(f'  Total voxels             : {n_voxels}')
    print(f'  Indexer agree (same pick): {n_idx_agree}/{n_voxels}')
    print(f'  Indexer correct:')
    print(f'    Unseeded               : {n_idx_uns_correct}/{n_voxels} '
          f'({100*n_idx_uns_correct/n_voxels:.1f}%)')
    print(f'    Seeded                 : {n_idx_seed_correct}/{n_voxels} '
          f'({100*n_idx_seed_correct/n_voxels:.1f}%)')
    print(f'  Refined correct:')
    print(f'    Unseeded               : {n_ref_uns_correct}/{n_voxels} '
          f'({100*n_ref_uns_correct/n_voxels:.1f}%)')
    print(f'    Seeded                 : {n_ref_seed_correct}/{n_voxels} '
          f'({100*n_ref_seed_correct/n_voxels:.1f}%)')
    n_seed_fixes = sum(1 for v in diff_voxels
                       if seeded_idx.get(v, {}).get('gt_grain', -1) == expected_gt[v]
                       and unseeded_idx.get(v, {}).get('gt_grain', -1) != expected_gt[v])
    n_seed_breaks = sum(1 for v in diff_voxels
                        if seeded_idx.get(v, {}).get('gt_grain', -1) != expected_gt[v]
                        and unseeded_idx.get(v, {}).get('gt_grain', -1) == expected_gt[v])
    print(f'  Seeding fixes            : {n_seed_fixes} voxels')
    print(f'  Seeding breaks           : {n_seed_breaks} voxels')

    # === 10. Generate comparison plot ===
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    grain_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    grain_cmap = ListedColormap(grain_colors[:n_grains])
    grain_norm = BoundaryNorm(np.arange(-0.5, n_grains, 1), grain_cmap.N)

    scan_half = BEAMSIZE / 2
    ext = [pos_sorted[0] - scan_half, pos_sorted[-1] + scan_half,
           pos_sorted[0] - scan_half, pos_sorted[-1] + scan_half]
    domain_half = SIZE_UM / 2

    def _add_domain_rect(ax):
        rect = plt.Rectangle((-domain_half, -domain_half), SIZE_UM, SIZE_UM,
                              linewidth=2, edgecolor='red', facecolor='none',
                              linestyle='--')
        ax.add_patch(rect)

    # Build grids
    uns_idx_grid = np.full((num_scans, num_scans), -1, dtype=int)
    seed_idx_grid = np.full((num_scans, num_scans), -1, dtype=int)
    uns_conf_grid = np.zeros((num_scans, num_scans))
    seed_conf_grid = np.zeros((num_scans, num_scans))
    exp_grid = np.full((num_scans, num_scans), -1, dtype=int)

    for vox in range(n_voxels):
        r, c = vox // num_scans, vox % num_scans
        exp_grid[r, c] = expected_gt[vox]
        u = unseeded_idx.get(vox, {'gt_grain': -1, 'conf': 0})
        s = seeded_idx.get(vox, {'gt_grain': -1, 'conf': 0})
        uns_idx_grid[r, c] = u['gt_grain']
        seed_idx_grid[r, c] = s['gt_grain']
        uns_conf_grid[r, c] = u['conf']
        seed_conf_grid[r, c] = s['conf']

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # Row 0: grain maps
    for ax, grid, title, mismatch_grid in [
        (axes[0, 0], exp_grid, 'Expected GT', None),
        (axes[0, 1], uns_idx_grid,
         f'Unseeded Indexer ({n_idx_uns_correct}/{n_voxels})', exp_grid),
        (axes[0, 2], seed_idx_grid,
         f'Seeded Indexer ({n_idx_seed_correct}/{n_voxels})', exp_grid),
    ]:
        im = ax.imshow(grid.T, origin='lower', cmap=grain_cmap,
                       norm=grain_norm, extent=ext)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        _add_domain_rect(ax)
        cbar = plt.colorbar(im, ax=ax, ticks=range(n_grains))
        cbar.set_ticklabels([f'GT{i}' for i in range(n_grains)])
        if mismatch_grid is not None:
            for r in range(num_scans):
                for c in range(num_scans):
                    if grid[r, c] != mismatch_grid[r, c]:
                        ax.plot(pos_sorted[r], pos_sorted[c], 'kx',
                                markersize=7, markeredgewidth=2)

    # Row 1: refined grain maps + confidence difference
    for ax, grid, title, mismatch_grid in [
        (axes[1, 0], unseeded_ref_grain,
         f'Unseeded Refined ({n_ref_uns_correct}/{n_voxels})', exp_grid),
        (axes[1, 1], seeded_ref_grain,
         f'Seeded Refined ({n_ref_seed_correct}/{n_voxels})', exp_grid),
    ]:
        im = ax.imshow(grid.T, origin='lower', cmap=grain_cmap,
                       norm=grain_norm, extent=ext)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        _add_domain_rect(ax)
        cbar = plt.colorbar(im, ax=ax, ticks=range(n_grains))
        cbar.set_ticklabels([f'GT{i}' for i in range(n_grains)])
        if mismatch_grid is not None:
            for r in range(num_scans):
                for c in range(num_scans):
                    if grid[r, c] != mismatch_grid[r, c]:
                        ax.plot(pos_sorted[r], pos_sorted[c], 'kx',
                                markersize=7, markeredgewidth=2)

    # Confidence difference: seeded - unseeded
    ax = axes[1, 2]
    diff = seed_conf_grid - uns_conf_grid
    vmax = max(abs(diff.min()), abs(diff.max()), 0.1)
    im = ax.imshow(diff.T, origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax,
                   extent=ext)
    ax.set_title('Confidence: Seeded - Unseeded', fontsize=13)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    _add_domain_rect(ax)
    plt.colorbar(im, ax=ax, label='Conf difference')

    plt.suptitle(f'Seeded vs Unseeded Comparison ({num_scans}x{num_scans}, '
                 f'{BEAMSIZE:.1f}um beam)', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = work_dir / 'seeded_comparison.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved comparison plot: {plot_path}')

    # === 11. Restore paramstest.txt (remove GrainsFile) ===
    with open(str(params_path)) as fh:
        lines = fh.readlines()
    lines = [l for l in lines if not l.startswith('GrainsFile')]
    with open(str(params_path), 'w') as fh:
        fh.writelines(lines)


def main():
    args = parse_args()

    # Derive skip modes: --skip-peaksearch implies --skip-sim
    skip_sim = args.skip_sim or args.skip_peaksearch
    skip_peaksearch = args.skip_peaksearch

    # Working directory for the test
    work_dir = MIDAS_HOME / 'FF_HEDM' / 'Example' / 'pfhedm_test'
    if skip_sim:
        if not work_dir.exists():
            print(f'ERROR: --skip-sim/--skip-peaksearch requires existing data at {work_dir}')
            print('  Run the test first with --no-cleanup')
            sys.exit(1)
    else:
        if work_dir.exists():
            shutil.rmtree(str(work_dir))
        work_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nMIDAS Home: {MIDAS_HOME}')
    print(f'Working directory: {work_dir}')
    print(f'Configuration: {NGRAINS} grains, {SIZE_UM}×{SIZE_UM} µm, '
          f'{STEP_UM} µm EBSD step, {NSCANS} scans, {BEAMSIZE:.1f} µm beam')
    print(f'doTomo={args.doTomo}, gpu={args.gpu}')
    if skip_sim:
        print(f'  skip_sim={skip_sim}, skip_peaksearch={skip_peaksearch}')

    try:
        # Steps 1-3: Generate data (same for OMP and GPU)
        if not skip_sim:
            generate_microstructure(work_dir)
            run_forward_simulation(work_dir, args.nCPUs)
            organize_for_pf_pipeline(work_dir, args.nCPUs)

        if args.gpu:
            # GPU parity test: run full OMP pipeline, then re-run just
            # indexer + refiner with GPU binaries and compare
            print('\n' + '#'*70)
            print('  GPU PARITY TEST MODE')
            print('#'*70)

            # --- OMP run (full pipeline) ---
            print('\n  >>> Running full OMP pipeline...')
            run_pf_pipeline(work_dir, args.nCPUs, doTomo=args.doTomo, useGPU=0)
            omp_passed = validate_results(work_dir, doTomo=args.doTomo)

            if not omp_passed:
                print('\n  ✗ OMP pipeline failed — cannot do GPU comparison')
                passed = False
            else:
                # Save OMP Output/ and Results/
                omp_dir = work_dir / '_omp_results'
                omp_dir.mkdir(exist_ok=True)
                copy_outputs(work_dir, omp_dir)

                # Clear indexer/refiner outputs only (keep hkls.csv, etc. in Output/)
                import glob
                for pattern in ['Output/IndexBest*', 'Output/IndexBestFull*',
                                'SpotsToIndex.csv', 'UniqueOrientations.csv',
                                'singleSolution.mic', 'sinos_*.bin',
                                'omegas_*.bin', 'nrHKLs_*.bin',
                                'spotMapping_*.bin', 'spotMeta_*.bin',
                                'spotPositions_*.bin', 'patches_*.bin']:
                    for f in glob.glob(str(work_dir / pattern)):
                        os.remove(f)
                # Clear Results/ fully
                results_d = work_dir / 'Results'
                if results_d.exists():
                    shutil.rmtree(str(results_d))
                results_d.mkdir(parents=True, exist_ok=True)

                # --- GPU run (just indexer + refiner) ---
                print('\n  >>> Running GPU indexer + refiner on same data...')
                gpu_ok = run_gpu_binaries(work_dir, args.nCPUs)

                if gpu_ok:
                    # Compare consolidated Index files
                    compare_consolidated_bins(omp_dir, work_dir, 'IndexBest_all.bin')

                    # Compare microstrFull.csv if refinement produced results
                    parity_passed = compare_gpu_omp(omp_dir, work_dir)
                    passed = parity_passed
                else:
                    print('\n  ✗ GPU binaries failed — check error output above')
                    passed = False
        else:
            # Standard OMP test
            gt = None
            if not skip_sim:
                gt = validate_forward_simulation(work_dir, args.nCPUs)
            run_pf_pipeline(work_dir, args.nCPUs, doTomo=args.doTomo,
                            skip_peaksearch=skip_peaksearch)
            if not skip_sim and not skip_peaksearch:
                validate_peak_search(work_dir, gt, args.nCPUs)
            validate_indexer_output(work_dir, gt)
            debug_indexer_all_voxels(work_dir, nCPUs=args.nCPUs)
            cross_compare_gt_indexer_debug(work_dir, nCPUs=args.nCPUs)
            validate_sinograms(work_dir, gt, args.nCPUs)
            passed = validate_results(work_dir, doTomo=args.doTomo)
            if args.compare_seeded:
                run_seeded_comparison(work_dir, args.nCPUs)

    except Exception as e:
        print(f'\nERROR: Test failed with exception: {e}')
        import traceback
        traceback.print_exc()
        passed = False

    if not args.no_cleanup:
        cleanup(work_dir)

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()

