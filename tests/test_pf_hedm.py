"""Automated test for pf-HEDM reconstruction pipeline.

Generates a synthetic microstructure, simulates multi-scan diffraction data
using ForwardSimulationCompressed, organizes output into the folder layout
expected by pf_MIDAS.py, and runs the full reconstruction pipeline.

Usage:
    python tests/test_pf_hedm.py -nCPUs 8
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


def run_pf_pipeline(work_dir, nCPUs):
    """Run pf_MIDAS.py reconstruction pipeline."""
    print('\n' + '='*70)
    print('  Step 4: Running pf_MIDAS.py reconstruction pipeline')
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
        '-doTomo', '1',
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


def validate_results(work_dir):
    """Validate reconstruction output."""
    print('\n' + '='*70)
    print('  Step 5: Validating results')
    print('='*70)

    passed = True

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

    # Check microstrFull.csv
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


def cleanup(work_dir):
    """Remove generated test artifacts."""
    print('\n  Cleaning up test artifacts...')
    if work_dir.exists():
        shutil.rmtree(str(work_dir))
    print('  Cleanup complete.')


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

    try:
        generate_microstructure(work_dir)
        run_forward_simulation(work_dir, args.nCPUs)
        organize_for_pf_pipeline(work_dir, args.nCPUs)
        run_pf_pipeline(work_dir, args.nCPUs)
        passed = validate_results(work_dir)
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
