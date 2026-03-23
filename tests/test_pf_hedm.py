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
import numpy as np
import zarr
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve MIDAS home relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent

sys.path.insert(0, str(MIDAS_HOME / 'utils'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FSTEM = 'pfhedm'
NGRAINS = 5
SIZE_UM = 50.0
STEP_UM = 5.0
NSCANS = 7
SEED = 42
PADDING = 6
# BeamSize = step size = SIZE_UM / (NSCANS - 1)
BEAMSIZE = SIZE_UM / (NSCANS - 1)


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


def run_pf_pipeline(work_dir, nCPUs, doTomo=0, useGPU=0):
    """Run pf_MIDAS.py reconstruction pipeline."""
    print('\n' + '='*70)
    print(f'  Step 4: Running pf_MIDAS.py (doTomo={doTomo}, useGPU={useGPU})')
    print('='*70)

    pf_script = MIDAS_HOME / 'FF_HEDM' / 'workflows' / 'pf_MIDAS.py'
    if not pf_script.exists():
        print(f'ERROR: {pf_script} not found.')
        sys.exit(1)

    cmd = [
        sys.executable, str(pf_script),
        '-paramFile', 'Parameters_pfhedm.txt',
        '-nCPUs', str(nCPUs),
        '-nCPUsLocal', str(nCPUs),
        '-convertFiles', '0',
        '-doPeakSearch', '1',
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


def main():
    args = parse_args()

    # Working directory for the test
    work_dir = MIDAS_HOME / 'FF_HEDM' / 'Example' / 'pfhedm_test'
    if work_dir.exists():
        shutil.rmtree(str(work_dir))
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nMIDAS Home: {MIDAS_HOME}')
    print(f'Working directory: {work_dir}')
    print(f'Configuration: {NGRAINS} grains, {SIZE_UM}×{SIZE_UM} µm, '
          f'{STEP_UM} µm step, {NSCANS} scans, {BEAMSIZE} µm beam')
    print(f'doTomo={args.doTomo}, gpu={args.gpu}')

    try:
        # Steps 1-3: Generate data (same for OMP and GPU)
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

                # Clear Output/ and Results/ only (keep Spots.bin, Data.bin, etc.)
                for subdir in ['Output', 'Results']:
                    d = work_dir / subdir
                    if d.exists():
                        shutil.rmtree(str(d))
                    d.mkdir(parents=True, exist_ok=True)

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
            run_pf_pipeline(work_dir, args.nCPUs, doTomo=args.doTomo)
            passed = validate_results(work_dir, doTomo=args.doTomo)

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

