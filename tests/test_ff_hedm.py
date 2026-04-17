import argparse
import os
import sys
import subprocess
import shutil
import stat
import zarr
import numpy as np
from pathlib import Path
from test_common import (add_common_args, run_preflight, print_environment,
                         DiagnosticReporter, build_diff_histogram,
                         get_mismatch_samples, get_midas_home)

# Files/dirs that ship with the Example and must NOT be removed
PRESERVE = {
    'Parameters.txt', 'Parameters_px_overlaps.txt',
    'GrainsSim.csv',
    'consolidated_Output.h5', 'consolidated_Output_px_overlaps.h5',
    'positions.csv', 'Calibration',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Benchmark Testing Suite for FF_HEDM")
    parser.add_argument("-nCPUs", type=int, default=1, help="Number of CPUs to use for the test")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup of generated files after the test")
    parser.add_argument("--cleanup-only", action="store_true", help="Only cleanup generated files, don't run any tests")
    parser.add_argument("--px-overlap", action="store_true", help="Also run pixel-overlap peaksearch test")
    parser.add_argument("--dual-dataset", action="store_true", help="Also run dual-dataset refinement sanity test")
    parser.add_argument("--nGrains", type=int, default=0,
                        help="Generate N random grains instead of using existing GrainsSim.csv (0 = use existing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grain generation")
    add_common_args(parser)

    # Optional paramFN defaulting to the Example Parameters.txt relative to the script location
    default_param_fn = Path(__file__).resolve().parent.parent / "FF_HEDM" / "Example" / "Parameters.txt"
    parser.add_argument("-paramFN", type=str, default=str(default_param_fn), help="Path to the parameter file")

    return parser.parse_args()


def parse_parameter_file(filename):
    """Reads the parameter file into a dictionary."""
    params = {}
    lines = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                lines.append(line)
                line_no_comment = line.split('#', 1)[0]
                cleaned_line = line_no_comment.strip()
                if not cleaned_line: continue
                parts = cleaned_line.split()
                if not parts: continue
                key, values = parts[0], parts[1:]
                
                processed_values = []
                for v in values:
                    try:
                        processed_values.append(int(v))
                    except ValueError:
                        try:
                            processed_values.append(float(v))
                        except ValueError:
                            processed_values.append(v)
                
                final_value = processed_values if len(processed_values) > 1 else (processed_values[0] if processed_values else "")
                
                if key not in params:
                    params[key] = final_value
                else:
                    if not isinstance(params[key], list) or not any(isinstance(i, list) for i in (params[key] if isinstance(params[key], list) else [params[key]])):
                        params[key] = [params[key]]
                    params[key].append(final_value)
    except FileNotFoundError:
        print(f"Error: Parameter file not found at '{filename}'")
        sys.exit(1)
    return params, lines


def create_testing_env(param_path, work_dir):
    """Creates a temporary parameter file manipulating local paths for testing."""
    params, lines = parse_parameter_file(param_path)
    
    # Ensure InFileName (GrainsSim.csv) is correctly located
    if 'InFileName' in params:
        in_file = params['InFileName']
        full_in_file = param_path.parent / in_file
        if not full_in_file.exists():
            print(f"Error: InFileName {full_in_file} does not exist.")
            sys.exit(1)
    else:
        print("Error: No 'InFileName' defined in Parameters.txt")
        sys.exit(1)
        
    out_file_name = params.get('OutFileName', 'Au_FF_000001_pf')
    
    new_param_file = work_dir / f"test_{param_path.name}"
    
    # Rewrite with absolute paths
    with open(new_param_file, 'w') as f:
        for line in lines:
            line_parts = line.split()
            if not line_parts:
                f.write(line)
                continue
            key = line_parts[0]
            
            if key == 'InFileName':
                f.write(f"InFileName {str(param_path.parent / params['InFileName'])}\n")
            elif key == 'OutFileName':
                f.write(f"OutFileName {str(work_dir / out_file_name)}\n")
            elif key == 'PositionsFile': 
                 f.write(f"PositionsFile {str(param_path.parent / params['PositionsFile'])}\n")
            else:
                f.write(line)
                
    return new_param_file, params, out_file_name


def run_forward_simulation(param_file, nCPUs, work_dir):
    """Runs ForwardSimulationCompressed."""
    midas_home = os.environ.get('MIDAS_HOME', str(Path(__file__).resolve().parent.parent))
    bin_path = Path(midas_home) / 'FF_HEDM' / 'bin' / 'ForwardSimulationCompressed'
    
    if not bin_path.exists():
        print(f"Error: {bin_path} not found. Please compile first.")
        sys.exit(1)
        
    cmd = [str(bin_path), str(param_file), str(nCPUs)]
    print(f"Running Simulation: {' '.join(cmd)}")
    print(f"Working directory: {work_dir}")
    result = subprocess.run(cmd, cwd=str(work_dir))
    
    if result.returncode != 0:
        print("Error: ForwardSimulationCompressed failed.")
        sys.exit(1)


def enrich_zarr_metadata(zarr_file_path, params):
    """Injects parameter metadata into the Zarr following ffGenerateZipRefactor logic."""
    print(f"Enriching pure data Zarr zip {zarr_file_path} with analysis/scan metadata...")
    
    with zarr.ZipStore(str(zarr_file_path), mode='a') as store:
        try:
            zRoot = zarr.group(store=store)
        except zarr.errors.GroupNotFoundError:
            zRoot = zarr.group(store=store, overwrite=True)
            
        # Ensure base structure
        if 'analysis' not in zRoot:
            zRoot.create_group('analysis/process/analysis_parameters')
        if 'measurement' not in zRoot:
            zRoot.create_group('measurement/process/scan_parameters')
            
        sp_ana = zRoot.require_group('analysis/process/analysis_parameters')
        sp_pro = zRoot.require_group('measurement/process/scan_parameters')
        
        # Read dtype from the actual data array and write it for the C code
        dtype_map = {
            'uint16': 'uint16', 'int32': 'int32', 'uint32': 'uint32',
            'float32': 'float32', 'float64': 'float64',
        }
        data_dtype = str(zRoot['exchange/data'].dtype)
        dtype_str = dtype_map.get(data_dtype, data_dtype)
        sp_pro.create_dataset('datatype', data=np.bytes_(dtype_str.encode('UTF-8')))
        print(f"  Written datatype for C-code: '{dtype_str}' (from exchange/data dtype: {data_dtype})")

        z_groups = {
            'sp_pro_analysis': sp_ana,
            'sp_pro_meas': sp_pro
        }
        
        import sys
        midas_home = Path(os.environ.get('MIDAS_HOME', str(Path(__file__).resolve().parent.parent)))
        utils_dir = midas_home / "utils"
        if str(utils_dir) not in sys.path:
            sys.path.append(str(utils_dir))
            
        from ffGenerateZipRefactor import write_analysis_parameters
        write_analysis_parameters(z_groups, params)


def run_px_overlap_test(args, work_dir, original_zip):
    """Run pixel-overlap peaksearch test reusing the same simulated data.
    
    Copies the already-enriched zip from the main test and patches only the
    parameters that differ (UsePixelOverlap, doPeakFit).
    """
    px_param_path = work_dir / "Parameters_px_overlaps.txt"
    if not px_param_path.exists():
        print(f"Error: {px_param_path} not found. Skipping pixel-overlap test.")
        return

    # Parse px_overlaps params to find what differs
    px_params, _ = parse_parameter_file(px_param_path)

    # Create testing env from the px_overlaps parameter file
    test_param_file, _, out_file_base = create_testing_env(px_param_path, work_dir)

    # Copy the already-enriched zip (skip if same file, e.g. same OutFileName)
    px_zip_name = work_dir / f"{out_file_base}.analysis.MIDAS.zip"
    if px_zip_name.resolve() != Path(original_zip).resolve():
        shutil.copy2(str(original_zip), str(px_zip_name))

    # Patch only the changed parameters into the zip
    print("Patching pixel-overlap parameters into Zarr zip...")
    with zarr.ZipStore(str(px_zip_name), mode='a') as store:
        zRoot = zarr.group(store=store)
        sp_ana = zRoot.require_group('analysis/process/analysis_parameters')
        sp_meas = zRoot.require_group('measurement/process/scan_parameters')

        # UsePixelOverlap -> analysis_parameters
        if 'UsePixelOverlap' in sp_ana:
            del sp_ana['UsePixelOverlap']
        sp_ana.create_dataset('UsePixelOverlap', data=np.array([int(px_params.get('UsePixelOverlap', 1))], dtype=np.int32))

        # doPeakFit -> scan_parameters
        if 'doPeakFit' in sp_meas:
            del sp_meas['doPeakFit']
        sp_meas.create_dataset('doPeakFit', data=np.array([int(px_params.get('doPeakFit', 0))], dtype=np.int32))

    print("  Patched: UsePixelOverlap, doPeakFit")

    # Run the pipeline
    midas_home = Path(os.environ.get('MIDAS_HOME', str(Path(__file__).resolve().parent.parent)))
    ff_midas_script = midas_home / "FF_HEDM" / "workflows" / "ff_MIDAS.py"

    cmd = [
        sys.executable, str(ff_midas_script),
        "-paramFN", test_param_file.name,
        "-nCPUs", str(args.nCPUs),
        "-dataFN", px_zip_name.name,
        "-convertFiles", "0"
    ]
    print(f"Running px-overlap pipeline: {' '.join(cmd)}")
    pipeline_res = subprocess.run(cmd, cwd=str(work_dir))

    if pipeline_res.returncode != 0:
        print("Error: Pixel-overlap pipeline execution failed.")
        return

    print("\n*** Pixel-overlap pipeline executed successfully ***")

    # Regression comparison
    result_dir = work_dir / "LayerNr_1"
    ref_h5 = work_dir / "consolidated_Output_px_overlaps.h5"
    new_h5_files = list(result_dir.glob("*_consolidated.h5"))

    if new_h5_files and ref_h5.exists():
        compare_consolidated_hdf5(ref_h5, new_h5_files[0])
    elif not ref_h5.exists():
        print(f"\nSkipping px-overlap regression: reference file {ref_h5} not found.")
    else:
        print(f"\nSkipping px-overlap regression: no consolidated HDF5 generated in {result_dir}.")


def parse_grains_csv(path):
    """Parse a Grains.csv file, return list of grain dicts with OrientMatrix and Position."""
    grains = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            vals = line.split()
            if len(vals) < 22:  # GrainID + 9 orient + 3 pos + 9 strain = minimum
                continue
            try:
                grain = {
                    'id': int(vals[0]),
                    'orient': np.array([float(v) for v in vals[1:10]]).reshape(3, 3),
                    'pos': np.array([float(v) for v in vals[10:13]]),
                    'lattice': np.array([float(v) for v in vals[13:19]]),
                }
                grains.append(grain)
            except (ValueError, IndexError):
                continue
    return grains


def misorientation_angle_deg(R1, R2):
    """Compute misorientation angle in degrees between two orientation matrices."""
    dR = R1 @ R2.T
    trace = np.clip(np.trace(dR), -1.0, 3.0)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle_rad)


def compare_grains_csv(ref_path, new_path, pos_tol_um=1.0, orient_tol_deg=0.01):
    """Compare two Grains.csv files for grain count, position, and orientation.
    
    Args:
        ref_path: Path to reference Grains.csv (from single-dataset run)
        new_path: Path to new Grains.csv (from dual-dataset run)
        pos_tol_um: Position tolerance in micrometers
        orient_tol_deg: Orientation tolerance in degrees
    """
    print(f"\n{'='*70}")
    print(f"  Dual-Dataset Grains Comparison")
    print(f"  Reference: {ref_path}")
    print(f"  New:       {new_path}")
    print(f"  Tolerances: position={pos_tol_um} µm, orientation={orient_tol_deg}°")
    print(f"{'='*70}\n")
    
    ref_grains = parse_grains_csv(ref_path)
    new_grains = parse_grains_csv(new_path)
    
    n_pass = 0
    n_fail = 0
    
    # Check grain count
    if len(ref_grains) == len(new_grains):
        print(f"  [PASS ✅]  Grain count: {len(ref_grains)}")
        n_pass += 1
    else:
        print(f"  [FAIL ❌]  Grain count: ref={len(ref_grains)}, new={len(new_grains)}")
        n_fail += 1
    
    # Match grains by ID and compare
    ref_by_id = {g['id']: g for g in ref_grains}
    new_by_id = {g['id']: g for g in new_grains}
    
    common_ids = sorted(set(ref_by_id.keys()) & set(new_by_id.keys()))
    if not common_ids:
        print(f"  [FAIL ❌]  No common grain IDs found")
        n_fail += 1
    else:
        # Position comparison
        max_pos_diff = 0.0
        pos_ok = True
        for gid in common_ids:
            diff = np.linalg.norm(ref_by_id[gid]['pos'] - new_by_id[gid]['pos'])
            max_pos_diff = max(max_pos_diff, diff)
            if diff > pos_tol_um:
                pos_ok = False
        
        if pos_ok:
            print(f"  [PASS ✅]  Positions (max_diff={max_pos_diff:.4f} µm)")
            n_pass += 1
        else:
            print(f"  [FAIL ❌]  Positions (max_diff={max_pos_diff:.4f} µm, tol={pos_tol_um})")
            n_fail += 1
        
        # Orientation comparison
        max_orient_diff = 0.0
        orient_ok = True
        for gid in common_ids:
            angle = misorientation_angle_deg(ref_by_id[gid]['orient'], new_by_id[gid]['orient'])
            max_orient_diff = max(max_orient_diff, angle)
            if angle > orient_tol_deg:
                orient_ok = False
        
        if orient_ok:
            print(f"  [PASS ✅]  Orientations (max_misorientation={max_orient_diff:.6f}°)")
            n_pass += 1
        else:
            print(f"  [FAIL ❌]  Orientations (max_misorientation={max_orient_diff:.6f}°, tol={orient_tol_deg})")
            n_fail += 1
    
    print(f"\n{'='*70}")
    print(f"  Dual-Dataset Summary: {n_pass} PASS, {n_fail} FAIL")
    print(f"{'='*70}")
    
    if n_fail > 0:
        print("\n⚠️  Dual-dataset differences detected!")
    else:
        print("\n✅ Dual-dataset grains match the single-dataset reference.")
    
    return n_fail == 0


def run_dual_dataset_test(args, work_dir, final_zip_name):
    """Run dual-dataset refinement test using the same dataset twice with zero offsets.
    
    Feeds the same simulated Zarr as both datasets to ff_dual_datasets.py.
    The resulting grains should closely match the single-dataset run.
    """
    midas_home = Path(os.environ.get('MIDAS_HOME', str(Path(__file__).resolve().parent.parent)))
    dual_script = midas_home / "FF_HEDM" / "workflows" / "ff_dual_datasets.py"
    
    if not dual_script.exists():
        print(f"Error: {dual_script} not found. Skipping dual-dataset test.")
        return
    
    # Check that MapDatasets binary exists
    map_bin = midas_home / "FF_HEDM" / "bin" / "MapDatasets"
    if not map_bin.exists():
        print(f"Error: {map_bin} not found. Please build MapDatasets. Skipping.")
        return
    
    # Use a separate result folder to avoid clobbering the main test's LayerNr_1
    dual_res_dir = work_dir / "dual_dataset_test"
    if dual_res_dir.exists():
        shutil.rmtree(dual_res_dir)
    
    # The test parameter file with absolute paths
    test_param_file = work_dir / f"test_Parameters.txt"
    if not test_param_file.exists():
        print(f"Error: {test_param_file} not found. Run main test first.")
        return
    
    cmd = [
        sys.executable, str(dual_script),
        "-resultFolder", str(dual_res_dir),
        "-paramFN", str(test_param_file),
        "-dataFN", str(final_zip_name),
        "-dataFN2", str(final_zip_name),
        "-offsetX", "0", "-offsetY", "0", "-offsetZ", "0", "-offsetOmega", "0",
        "-nCPUs", str(args.nCPUs),
    ]
    
    print(f"Running dual-dataset pipeline: {' '.join(cmd)}")
    pipeline_res = subprocess.run(cmd, cwd=str(work_dir))
    
    if pipeline_res.returncode != 0:
        print("Error: Dual-dataset pipeline execution failed.")
        return
    
    print("\n*** Dual-dataset pipeline executed successfully ***")
    
    # Find the Grains.csv in the dual-dataset result
    # ff_dual_datasets.py puts results in dataset_1_analysis/
    dual_grains = dual_res_dir / "dataset_1_analysis" / "Grains.csv"
    ref_grains = work_dir / "LayerNr_1" / "Grains.csv"
    
    if dual_grains.exists() and ref_grains.exists():
        compare_grains_csv(ref_grains, dual_grains, pos_tol_um=1.0, orient_tol_deg=0.09)
    elif not ref_grains.exists():
        print(f"\nSkipping comparison: reference {ref_grains} not found.")
    else:
        print(f"\nSkipping comparison: dual-dataset {dual_grains} not found.")


def main():
    args = parse_args()
    param_path = Path(args.paramFN).resolve()
    if not param_path.exists():
        print(f"Error: Parameter file not found at {param_path}")
        sys.exit(1)

    print_environment()

    midas_home = get_midas_home()
    if not getattr(args, 'skip_preflight', False):
        run_preflight(
            required_binaries=["ForwardSimulationCompressed",
                               "PeaksFittingOMPZarrRefactor"],
            required_packages=["numpy", "zarr", "h5py"],
            required_data_files=[
                str(midas_home / "FF_HEDM" / "Example" / "Parameters.txt"),
                str(midas_home / "FF_HEDM" / "Example" / "GrainsSim.csv"),
                str(midas_home / "FF_HEDM" / "Example" / "consolidated_Output.h5"),
            ],
        )

    print(f"Starting FF_HEDM Benchmark using: {param_path}")
    print(f"Using CPUs: {args.nCPUs}")
    
    # 1. Prepare Workspace (always use FF_HEDM/Example/ as the working directory)
    work_dir = midas_home / "FF_HEDM" / "Example"
    work_dir.mkdir(exist_ok=True, parents=True)

    # Clean stale results directory to avoid crashes from leftover state
    layer_dir = work_dir / "LayerNr_1"
    if layer_dir.exists():
        print(f"Removing stale {layer_dir.name}/ directory...")
        shutil.rmtree(layer_dir)

    if args.cleanup_only:
        cleanup_work_dir(work_dir)
        return
    
    # 2. Modify Params & Resolve Paths
    test_param_file, params, out_file_base = create_testing_env(param_path, work_dir)
    print(f"Created temporary parameter environment: {test_param_file}")

    # 2b. Generate random grains if requested
    if args.nGrains > 0:
        from generate_grains import generate_grains_csv
        grains_path = work_dir / "GrainsSim.csv"
        backup = work_dir / "GrainsSim_original.csv"
        if not backup.exists() and grains_path.exists():
            shutil.copy2(str(grains_path), str(backup))
        rsample = params.get("Rsample", 2000)
        hbeam = params.get("Hbeam", 2000)
        beam_thickness = params.get("BeamThickness", 200)
        sg = params.get("SpaceGroup", 225)
        if isinstance(rsample, list): rsample = rsample[0]
        if isinstance(hbeam, list): hbeam = hbeam[0]
        if isinstance(beam_thickness, list): beam_thickness = beam_thickness[0]
        if isinstance(sg, list): sg = sg[0]
        lat_str = params.get("LatticeConstant", [4.08, 4.08, 4.08, 90, 90, 90])
        if not isinstance(lat_str, list):
            lat_str = [lat_str]
        lat = [float(v) for v in lat_str]
        if len(lat) < 6:
            lat = [4.08, 4.08, 4.08, 90.0, 90.0, 90.0]
        generate_grains_csv(
            grains_path, args.nGrains, lat,
            float(rsample), float(hbeam), float(beam_thickness),
            space_group=int(sg), seed=args.seed,
        )

    # 3. Execute Simulation
    # Note ForwardSimulationCompressed runs in cwd usually and might dump things locally too.
    # To be safe, we change to work_dir or give absolute path outputs.
    # OutFileName uses absolute paths, so executing here is fine.
    run_forward_simulation(test_param_file, args.nCPUs, work_dir)
    
    # Forward simulation produces _scanNr_0.zip
    generated_zip = work_dir / f"{out_file_base}_scanNr_0.zip"
    if not generated_zip.exists():
         print(f"Error: Expected simulation output {generated_zip} not found.")
         sys.exit(1)
         
    # 4. Enrich Zarr
    enrich_zarr_metadata(generated_zip, params)
    
    # 5. Rename to trigger ff_MIDAS correctly skipping generation
    # ff_MIDAS.py expects {Stem}.analysis.MIDAS.zip if bypassed.
    final_zip_name = work_dir / f"{out_file_base}.analysis.MIDAS.zip"
    shutil.move(str(generated_zip), str(final_zip_name))
    
    print(f"\nSimulation complete and Zarr generated: {final_zip_name}")
    print("\nNext step: Kickstarting ff_MIDAS.py workflow...")
    
    # 6. Kickstart ff_MIDAS.py
    midas_home = Path(os.environ.get('MIDAS_HOME', str(Path(__file__).resolve().parent.parent)))
    ff_midas_script = midas_home / "FF_HEDM" / "workflows" / "ff_MIDAS.py"
    
    if not ff_midas_script.exists():
        print(f"Error: {ff_midas_script} not found.")
        sys.exit(1)
        
    cmd = [
        sys.executable, str(ff_midas_script),
        "-paramFN", test_param_file.name,
        "-nCPUs", str(args.nCPUs),
        "-dataFN", final_zip_name.name,
        "-convertFiles", "0"
    ]
    
    print(f"Running pipeline: {' '.join(cmd)}")
    pipeline_res = subprocess.run(cmd, cwd=str(work_dir))
    
    if pipeline_res.returncode != 0:
         print("Error: Pipeline ff_MIDAS.py execution failed.")
         sys.exit(1)

    print("\n*** Automated FF_HEDM Benchmark Suite Executed Successfully ***")
    
    # 7. Regression comparison against reference consolidated HDF5
    result_dir = work_dir / "LayerNr_1"
    ref_h5 = work_dir / "consolidated_Output.h5"
    
    # Find the newly generated consolidated h5
    new_h5_files = list(result_dir.glob("*_consolidated.h5"))
    if new_h5_files and ref_h5.exists():
        compare_consolidated_hdf5(ref_h5, new_h5_files[0])
    elif not ref_h5.exists():
        print(f"\nSkipping regression comparison: reference file {ref_h5} not found.")
    else:
        print(f"\nSkipping regression comparison: no consolidated HDF5 generated in {result_dir}.")

    # 8. Optional pixel-overlap peaksearch test (before cleanup so zip exists)
    if args.px_overlap:
        print("\n" + "="*60)
        print("Running pixel-overlap peaksearch test...")
        print("="*60)
        run_px_overlap_test(args, work_dir, final_zip_name)

    # 8b. Optional dual-dataset refinement test
    if args.dual_dataset:
        print("\n" + "="*60)
        print("Running dual-dataset refinement test...")
        print("="*60)
        run_dual_dataset_test(args, work_dir, final_zip_name)

    # 9. Cleanup generated files
    if not args.no_cleanup:
        cleanup_work_dir(work_dir)
    else:
        print("\nSkipping cleanup (--no-cleanup specified).")


def cleanup_work_dir(work_dir):
    """Remove all test-generated files from the work directory, preserving originals."""
    print("\nCleaning up test-generated files...")
    removed = 0
    for item in sorted(work_dir.iterdir()):
        if item.name in PRESERVE or item.name == '.DS_Store':
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed += 1
        except Exception as e:
            print(f"  Warning: could not remove {item.name}: {e}")
    print(f"  Removed {removed} generated files/directories.")


def match_spots(ref_h5, new_h5, omega_tol=2.0, eta_tol=10.0, radius_tol=10.0):
    """Match spots between ref and new HDF5 by physical properties.

    Uses greedy nearest-neighbor matching on (RingNr, Omega, Eta).

    Returns:
        (matched_pairs, ref_unmatched, new_unmatched)
    """
    ref_ring = ref_h5["radius_data/RingNr"][()]
    ref_omega = ref_h5["radius_data/Omega"][()]
    ref_eta = ref_h5["radius_data/Eta"][()]
    ref_radius = ref_h5["radius_data/Radius"][()]
    new_ring = new_h5["radius_data/RingNr"][()]
    new_omega = new_h5["radius_data/Omega"][()]
    new_eta = new_h5["radius_data/Eta"][()]
    new_radius = new_h5["radius_data/Radius"][()]

    n_ref = len(ref_ring)
    n_new = len(new_ring)
    matched = []
    used_new = set()

    for ri in range(n_ref):
        best_ni = -1
        best_dist = float('inf')
        for ni in range(n_new):
            if ni in used_new:
                continue
            if new_ring[ni] != ref_ring[ri]:
                continue
            d_omega = abs(new_omega[ni] - ref_omega[ri])
            d_eta = abs(new_eta[ni] - ref_eta[ri])
            if d_eta > 180:
                d_eta = 360 - d_eta
            d_radius = abs(new_radius[ni] - ref_radius[ri])
            if d_omega > omega_tol or d_eta > eta_tol or d_radius > radius_tol:
                continue
            dist = d_omega + d_eta * 0.1 + d_radius * 0.01
            if dist < best_dist:
                best_dist = dist
                best_ni = ni
        if best_ni >= 0:
            matched.append((ri, best_ni))
            used_new.add(best_ni)

    ref_unmatched = [i for i in range(n_ref) if i not in {m[0] for m in matched}]
    new_unmatched = [i for i in range(n_new) if i not in used_new]
    return matched, ref_unmatched, new_unmatched


def compare_consolidated_hdf5(ref_path, new_path, atol=1e-6, rtol=1e-6,
                               max_extra_spots=3):
    """Compare consolidated HDF5 using order-independent spot matching.

    Spots are matched by (RingNr, Omega, Eta) proximity to handle
    platform-dependent ordering (arm64 vs x86_64). Matched spots are
    then compared with per-field tolerances.
    
    Args:
        ref_path: Path to the reference consolidated HDF5
        new_path: Path to the newly generated consolidated HDF5
        atol: Absolute tolerance for non-spot data
        rtol: Relative tolerance for non-spot data
        max_extra_spots: Maximum extra/missing spots before FAIL
    """
    import h5py

    print(f"\n{'='*70}")
    print(f"  Regression Comparison (order-independent)")
    print(f"  Reference: {ref_path}")
    print(f"  New:       {new_path}")
    print(f"  Tolerance: atol={atol}, rtol={rtol}, max_extra_spots={max_extra_spots}")
    print(f"{'='*70}\n")

    n_pass = 0
    n_fail = 0
    n_skip = 0

    try:
        ref = h5py.File(str(ref_path), 'r')
        new = h5py.File(str(new_path), 'r')
    except Exception as e:
        print(f"Error opening HDF5 files: {e}")
        sys.exit(1)

    try:
        # ── Phase 1: Spot matching via radius_data ────────────────────
        has_radius = ("radius_data/RingNr" in ref and "radius_data/RingNr" in new)
        if has_radius:
            matched, ref_unmatched, new_unmatched = match_spots(ref, new)
            n_ref_spots = ref["radius_data/RingNr"].shape[0]
            n_new_spots = new["radius_data/RingNr"].shape[0]
            n_matched = len(matched)
            n_extra = len(new_unmatched)
            n_missing = len(ref_unmatched)

            print(f"  Spot Matching: {n_matched} matched, "
                  f"{n_extra} extra, {n_missing} missing "
                  f"(ref={n_ref_spots}, new={n_new_spots})")

            if n_extra > 0:
                print(f"  Extra spots in new (indices): {new_unmatched[:10]}")
                for ni in new_unmatched[:5]:
                    ring = new["radius_data/RingNr"][()][ni]
                    omega = new["radius_data/Omega"][()][ni]
                    eta = new["radius_data/Eta"][()][ni]
                    imax = new["radius_data/IMax"][()][ni]
                    print(f"    [{ni}] Ring={ring:.0f}, Omega={omega:.2f}, "
                          f"Eta={eta:.2f}, IMax={imax:.1f}")
            if n_missing > 0:
                print(f"  Missing from new (ref indices): {ref_unmatched[:10]}")
                for ri in ref_unmatched[:5]:
                    ring = ref["radius_data/RingNr"][()][ri]
                    omega = ref["radius_data/Omega"][()][ri]
                    eta = ref["radius_data/Eta"][()][ri]
                    imax = ref["radius_data/IMax"][()][ri]
                    print(f"    [{ri}] Ring={ring:.0f}, Omega={omega:.2f}, "
                          f"Eta={eta:.2f}, IMax={imax:.1f}")
            print()
        else:
            matched = None

        # ── Per-field tolerances for matched spots ────────────────────
        spot_tolerances = {
            'IntegratedIntensity': {'rtol': 0.50, 'atol': 100},
            'Omega':              {'rtol': 0,    'atol': 2.0},
            'YCen':               {'rtol': 0,    'atol': 50.0},
            'ZCen':               {'rtol': 0,    'atol': 50.0},
            'IMax':               {'rtol': 0.50, 'atol': 100},
            'Radius':             {'rtol': 0.01, 'atol': 5.0},
            'Eta':                {'rtol': 0,    'atol': 10.0},
            'RingNr':             {'rtol': 0,    'atol': 0},
            'GrainRadius':        {'rtol': 0.50, 'atol': 10.0},
            'SigmaR':             {'rtol': 0.20, 'atol': 0.1},
            'SigmaEta':           {'rtol': 0.50, 'atol': 0.05},
            'SpotID':             None,  # IDs differ by design
        }

        # ── CalcRadius (order-independent) ────────────────────────────
        if has_radius and matched is not None:
            stage_ok = True
            stage_details = []

            if n_extra + n_missing > max_extra_spots:
                stage_details.append(
                    f"    Extra/missing spots: {n_extra + n_missing} > "
                    f"{max_extra_spots}")
                stage_ok = False
            else:
                stage_details.append(
                    f"    Spot count: {n_matched} matched"
                    + (f", {n_extra} extra, {n_missing} missing (OK)"
                       if n_extra + n_missing > 0 else ""))

            for field, tol in spot_tolerances.items():
                ds_path = f"radius_data/{field}"
                if ds_path not in ref or ds_path not in new:
                    continue
                if tol is None:
                    stage_details.append(
                        f"    {ds_path}: SKIP (IDs differ by design)")
                    continue

                ref_data = ref[ds_path][()]
                new_data = new[ds_path][()]
                max_diff = 0.0
                n_mismatch = 0
                worst = ""

                for ri, ni in matched:
                    rv, nv = float(ref_data[ri]), float(new_data[ni])
                    diff = abs(rv - nv)
                    if field == 'Eta' and diff > 180:
                        diff = 360 - diff
                    threshold = tol['atol'] + tol['rtol'] * abs(rv)
                    if diff > threshold:
                        n_mismatch += 1
                        if diff > max_diff:
                            worst = f"ref[{ri}]={rv:.4g} vs new[{ni}]={nv:.4g}"
                    max_diff = max(max_diff, diff)

                if n_mismatch > 0:
                    stage_details.append(
                        f"    {ds_path}: FAIL ({n_mismatch}/{n_matched} exceed "
                        f"tol, max_diff={max_diff:.2e}, worst: {worst})")
                    stage_ok = False
                else:
                    stage_details.append(
                        f"    {ds_path}: PASS (max_diff={max_diff:.2e})")

            status = "PASS ✅" if stage_ok else "FAIL ❌"
            n_pass += stage_ok
            n_fail += (not stage_ok)
            print(f"  [{status}]  CalcRadius (order-independent)")
            for d in stage_details:
                print(d)

        # ── PeaksFitting summary (order-independent by Omega/Eta) ─────
        ps_ok = True
        ps_details = []
        ps_path = "peaks/summary/data"
        if ps_path in ref and ps_path in new:
            r_ps = ref[ps_path][()]
            n_ps = new[ps_path][()]
            size_diff = abs(r_ps.shape[0] - n_ps.shape[0])
            if size_diff <= max_extra_spots:
                # Sort both by (Eta, Omega) for stable comparison
                # Columns: 0=SpotID, 1=IntInt, 2=Omega, 3=YCen, 4=ZCen,
                #          5=IMax, ..., 13=Eta
                n_cols = min(r_ps.shape[1], n_ps.shape[1])
                eta_col = min(13, n_cols - 1)
                ome_col = 2
                r_order = np.lexsort((r_ps[:, ome_col], r_ps[:, eta_col]))
                n_order = np.lexsort((n_ps[:, ome_col], n_ps[:, eta_col]))
                # Compare the min(len) sorted rows
                n_cmp = min(len(r_order), len(n_order))
                r_sorted = r_ps[r_order[:n_cmp]]
                n_sorted = n_ps[n_order[:n_cmp]]
                # Skip SpotID col (col 0) — IDs naturally differ
                if np.allclose(r_sorted[:, 1:], n_sorted[:, 1:],
                               atol=100, rtol=0.5, equal_nan=True):
                    ps_details.append(
                        f"    {ps_path}: PASS (sorted by Eta/Omega, "
                        f"{n_cmp} spots compared)")
                else:
                    diff = np.abs(r_sorted[:, 1:] - n_sorted[:, 1:])
                    n_mm = np.sum(~np.isclose(r_sorted[:, 1:], n_sorted[:, 1:],
                                              atol=100, rtol=0.5, equal_nan=True))
                    md = np.nanmax(diff)
                    ps_details.append(
                        f"    {ps_path}: FAIL (sorted, max_diff={md:.2e}, "
                        f"mismatched={n_mm}/{r_sorted[:, 1:].size})")
                    ps_ok = False
                if size_diff > 0:
                    ps_details.append(
                        f"    → spot count diff={size_diff} (within tol)")
            else:
                ps_details.append(
                    f"    {ps_path}: FAIL (ref={r_ps.shape[0]}, "
                    f"new={n_ps.shape[0]})")
                ps_ok = False
        else:
            ps_details.append(f"    {ps_path}: SKIP")
            n_skip += 1
        status = "PASS ✅" if ps_ok else "FAIL ❌"
        n_pass += ps_ok
        n_fail += (not ps_ok)
        print(f"  [{status}]  PeaksFitting (summary)")
        for d in ps_details:
            print(d)

        # ── ids_hash (direct compare) ─────────────────────────────────
        for stage_name, ds_path in [("ids_hash", "ids_hash/data")]:
            ok = True
            details = []
            if ds_path not in ref or ds_path not in new:
                details.append(f"    {ds_path}: SKIP")
                n_skip += 1
                print(f"  [SKIP]  {stage_name}")
                for d in details:
                    print(d)
                continue
            r, n = ref[ds_path][()], new[ds_path][()]
            if r.shape != n.shape:
                details.append(
                    f"    {ds_path}: FAIL (shape ref={r.shape}, new={n.shape})")
                ok = False
            elif np.issubdtype(r.dtype, np.floating):
                if np.allclose(r, n, atol=atol, rtol=rtol, equal_nan=True):
                    md = np.nanmax(np.abs(r - n)) if r.size > 0 else 0
                    details.append(f"    {ds_path}: PASS (max_diff={md:.2e})")
                else:
                    diff = np.abs(r - n)
                    md = np.nanmax(diff)
                    nm = np.sum(~np.isclose(r, n, atol=atol, rtol=rtol,
                                            equal_nan=True))
                    details.append(
                        f"    {ds_path}: FAIL (max_diff={md:.2e}, "
                        f"mismatched={nm}/{r.size})")
                    ok = False
            elif np.array_equal(r, n):
                details.append(f"    {ds_path}: PASS (exact)")
            else:
                nm = np.sum(r != n)
                details.append(f"    {ds_path}: FAIL ({nm}/{r.size} differ)")
                ok = False
            status = "PASS ✅" if ok else "FAIL ❌"
            n_pass += ok
            n_fail += (not ok)
            print(f"  [{status}]  {stage_name}")
            for d in details:
                print(d)

        # ── PeaksFitting per-frame (order-independent) ────────────────
        pf_path = "peaks/per_frame/data"
        pf_ok = True
        pf_details = []
        if pf_path in ref and pf_path in new:
            ref_pf, new_pf = ref[pf_path][()], new[pf_path][()]
            if matched is not None and ref_pf.shape[0] == n_ref_spots:
                n_cols = min(ref_pf.shape[1], new_pf.shape[1])
                n_ok = n_bad = 0
                for ri, ni in matched:
                    if ri < ref_pf.shape[0] and ni < new_pf.shape[0]:
                        if np.allclose(ref_pf[ri, :n_cols], new_pf[ni, :n_cols],
                                       atol=100, rtol=0.5, equal_nan=True):
                            n_ok += 1
                        else:
                            n_bad += 1
                if n_bad == 0:
                    pf_details.append(
                        f"    {pf_path}: PASS ({n_ok} matched spots OK)")
                else:
                    pf_details.append(
                        f"    {pf_path}: FAIL ({n_bad}/{n_ok + n_bad} "
                        f"exceed tolerance)")
                    pf_ok = False
            else:
                diff_n = abs(ref_pf.shape[0] - new_pf.shape[0])
                if diff_n <= max_extra_spots:
                    pf_details.append(
                        f"    {pf_path}: PASS (count diff={diff_n} within tol)")
                else:
                    pf_details.append(
                        f"    {pf_path}: FAIL (ref={ref_pf.shape}, "
                        f"new={new_pf.shape})")
                    pf_ok = False
        else:
            pf_details.append(f"    {pf_path}: SKIP")
            n_skip += 1
        status = "PASS ✅" if pf_ok else "FAIL ❌"
        n_pass += pf_ok
        n_fail += (not pf_ok)
        print(f"  [{status}]  PeaksFitting (per-frame)")
        for d in pf_details:
            print(d)

        # ── MergeOverlaps (count tolerance + id_rings) ────────────────
        merge_ok = True
        merge_details = []
        for ds_key in ["merge_map/MergedSpotID", "merge_map/FrameNr",
                        "merge_map/PeakID"]:
            if ds_key not in ref or ds_key not in new:
                continue
            r_d, n_d = ref[ds_key][()], new[ds_key][()]
            diff_n = abs(len(r_d) - len(n_d))
            if diff_n <= max_extra_spots:
                merge_details.append(
                    f"    {ds_key}: PASS (count diff={diff_n})")
            else:
                merge_details.append(
                    f"    {ds_key}: FAIL (ref={len(r_d)}, new={len(n_d)})")
                merge_ok = False

        for ds_key in ["id_rings/data"]:
            if ds_key in ref and ds_key in new:
                r_d, n_d = ref[ds_key][()], new[ds_key][()]
                if r_d.shape == n_d.shape:
                    if np.array_equal(r_d, n_d):
                        merge_details.append(
                            f"    {ds_key}: PASS (exact match)")
                    else:
                        nm = np.sum(r_d != n_d)
                        pct = (1 - nm / r_d.size) * 100
                        if pct >= 90:
                            merge_details.append(
                                f"    {ds_key}: PASS ({pct:.0f}% match)")
                        else:
                            merge_details.append(
                                f"    {ds_key}: FAIL ({pct:.0f}% match, "
                                f"{nm}/{r_d.size})")
                            merge_ok = False
                else:
                    merge_details.append(
                        f"    {ds_key}: shape ref={r_d.shape}, new={n_d.shape}")




        status = "PASS ✅" if merge_ok else "FAIL ❌"
        n_pass += merge_ok
        n_fail += (not merge_ok)
        print(f"  [{status}]  MergeOverlaps")
        for d in merge_details:
            print(d)

        # ── FitSetup (InputAll) — order-independent ──────────────────
        fit_ok = True
        fit_details = []

        # all_spots/data: cols 0=YLab,1=ZLab,2=Omega,3=GrainRadius,
        #                      4=SpotID,5=RingNr,6=Eta,...
        as_path = "all_spots/data"
        if as_path in ref and as_path in new:
            r_d, n_d = ref[as_path][()], new[as_path][()]
            size_diff = abs(r_d.shape[0] - n_d.shape[0])
            if size_diff <= max_extra_spots:
                n_cols = min(r_d.shape[1], n_d.shape[1])
                # Sort by (RingNr, Eta, Omega) — cols 5, 6, 2
                ring_col = min(5, n_cols - 1)
                eta_col = min(6, n_cols - 1)
                ome_col = 2
                r_order = np.lexsort((r_d[:, ome_col], r_d[:, eta_col],
                                      r_d[:, ring_col]))
                n_order = np.lexsort((n_d[:, ome_col], n_d[:, eta_col],
                                      n_d[:, ring_col]))
                n_cmp = min(len(r_order), len(n_order))
                r_sorted = r_d[r_order[:n_cmp]]
                n_sorted = n_d[n_order[:n_cmp]]
                # Skip SpotID col (col 4) — naturally differs
                skip_mask = np.ones(n_cols, dtype=bool)
                if n_cols > 4:
                    skip_mask[4] = False
                r_cmp = r_sorted[:, skip_mask]
                n_cmp_data = n_sorted[:, skip_mask]
                if np.allclose(r_cmp, n_cmp_data, atol=100, rtol=0.5,
                               equal_nan=True):
                    fit_details.append(
                        f"    {as_path}: PASS (sorted, {n_cmp} rows)")
                else:
                    diff = np.abs(r_cmp - n_cmp_data)
                    nm = np.sum(~np.isclose(r_cmp, n_cmp_data, atol=100,
                                            rtol=0.5, equal_nan=True))
                    md = np.nanmax(diff)
                    # Allow mismatches from the extra/missing spot(s):
                    # each extra spot can misalign ~n_cols values in sorted
                    n_allowed = max_extra_spots * (n_cols + 3)
                    if nm <= n_allowed:
                        fit_details.append(
                            f"    {as_path}: PASS (sorted, {n_cmp} rows, "
                            f"{nm} values within spot tolerance)")
                    else:
                        fit_details.append(
                            f"    {as_path}: FAIL (sorted, max_diff={md:.2e}, "
                            f"mismatched={nm}/{r_cmp.size})")
                        fit_ok = False
                if size_diff > 0:
                    fit_details.append(
                        f"    → row count diff={size_diff} (within tol)")
            else:
                fit_details.append(
                    f"    {as_path}: FAIL (ref={r_d.shape}, new={n_d.shape})")
                fit_ok = False
        else:
            fit_details.append(f"    {as_path}: SKIP")

        # spots_to_index: integer SpotIDs — sort and compare counts
        si_path = "spots_to_index/data"
        if si_path in ref and si_path in new:
            r_si, n_si = ref[si_path][()], new[si_path][()]
            if r_si.shape == n_si.shape:
                # SpotIDs will differ but count should match
                fit_details.append(
                    f"    {si_path}: PASS (count={r_si.shape[0]})")
            else:
                diff_n = abs(r_si.shape[0] - n_si.shape[0])
                if diff_n <= max_extra_spots:
                    fit_details.append(
                        f"    {si_path}: PASS (count diff={diff_n})")
                else:
                    fit_details.append(
                        f"    {si_path}: FAIL (ref={r_si.shape}, "
                        f"new={n_si.shape})")
                    fit_ok = False
        else:
            fit_details.append(f"    {si_path}: SKIP")
        status = "PASS ✅" if fit_ok else "FAIL ❌"
        n_pass += fit_ok
        n_fail += (not fit_ok)
        print(f"  [{status}]  FitSetup (InputAll)")
        for d in fit_details:
            print(d)

        # ── Grains (order-independent by position) ────────────────────
        grains_ok = True
        grains_details = []
        gs_key = "grains/summary"
        if gs_key in ref and gs_key in new:
            r_g, n_g = ref[gs_key][()], new[gs_key][()]
            if r_g.shape[0] != n_g.shape[0]:
                grains_details.append(
                    f"    Grain count: FAIL (ref={r_g.shape[0]}, "
                    f"new={n_g.shape[0]})")
                grains_ok = False
            else:
                n_grains = r_g.shape[0]
                grains_details.append(f"    Grain count: {n_grains} (match)")
                if r_g.shape[1] >= 13:
                    ref_pos = r_g[:, 10:13]
                    new_pos = n_g[:, 10:13]
                    grain_match = []
                    used = set()
                    for ri in range(n_grains):
                        best_ni, best_d = -1, float('inf')
                        for ni in range(n_grains):
                            if ni in used:
                                continue
                            d = np.linalg.norm(ref_pos[ri] - new_pos[ni])
                            if d < best_d:
                                best_d = d
                                best_ni = ni
                        if best_ni >= 0 and best_d < 100:
                            grain_match.append((ri, best_ni))
                            used.add(best_ni)
                            grains_details.append(
                                f"    Grain {ri} → {best_ni} "
                                f"(pos_diff={best_d:.2f} µm)")
                    if len(grain_match) == n_grains:
                        grains_details.append(
                            f"    All {n_grains} grains matched")
                    else:
                        grains_details.append(
                            f"    Only {len(grain_match)}/{n_grains} matched")
                        grains_ok = False

        for ds_key in ["spot_matrix/data", "grain_ids_key/data"]:
            if ds_key in ref and ds_key in new:
                r_d, n_d = ref[ds_key][()], new[ds_key][()]
                if r_d.shape == n_d.shape:
                    grains_details.append(
                        f"    {ds_key}: shape OK ({r_d.shape})")
                else:
                    grains_details.append(
                        f"    {ds_key}: shape ref={r_d.shape}, "
                        f"new={n_d.shape}")

        status = "PASS ✅" if grains_ok else "FAIL ❌"
        n_pass += grains_ok
        n_fail += (not grains_ok)
        print(f"  [{status}]  Grains (final)")
        for d in grains_details:
            print(d)

    finally:
        ref.close()
        new.close()

    print(f"\n{'='*70}")
    print(f"  Summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    print(f"{'='*70}")

    if n_fail > 0:
        print("\n⚠️  Regression differences detected!")
        sys.exit(1)
    else:
        print("\n✅ All pipeline stages match the reference output.")


# MIDAS version banner
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'utils'))
    from version import version_string as _vs
    print(_vs())
except Exception:
    pass

if __name__ == "__main__":
    main()
