import argparse
import os
import sys
import subprocess
import shutil
import stat
import zarr
import numpy as np
from pathlib import Path

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
        
        # Populate Scan Parameters (simplistic mappings from config to measurement)
        # Assuming missing physical metadata like temperature is irrelevant for raw simulations
        sp_pro.require_dataset('datatype', shape=(), dtype=str, data='uint16')

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


def main():
    args = parse_args()
    param_path = Path(args.paramFN).resolve()
    if not param_path.exists():
        print(f"Error: Parameter file not found at {param_path}")
        sys.exit(1)
        
    print(f"Starting FF_HEDM Benchmark using: {param_path}")
    print(f"Using CPUs: {args.nCPUs}")
    
    # 1. Prepare Workspace (always use FF_HEDM/Example/ as the working directory)
    midas_home = Path(os.environ.get('MIDAS_HOME', str(Path(__file__).resolve().parent.parent)))
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


def compare_consolidated_hdf5(ref_path, new_path, atol=1e-6, rtol=1e-6):
    """Compare a newly generated consolidated HDF5 against a reference file.
    
    Reports per-pipeline-stage PASS/FAIL with max absolute difference.
    
    Args:
        ref_path: Path to the reference consolidated HDF5
        new_path: Path to the newly generated consolidated HDF5
        atol: Absolute tolerance for floating-point comparison
        rtol: Relative tolerance for floating-point comparison
    """
    import h5py
    
    print(f"\n{'='*70}")
    print(f"  Regression Comparison")
    print(f"  Reference: {ref_path}")
    print(f"  New:       {new_path}")
    print(f"  Tolerance: atol={atol}, rtol={rtol}")
    print(f"{'='*70}\n")
    
    # Map pipeline stages to HDF5 paths
    stages = [
        ("PeaksFitting (summary)", [
            "peaks/summary/data",
        ]),
        ("PeaksFitting (per-frame)", [
            "peaks/per_frame/data",
        ]),
        ("MergeOverlaps", [
            "merge_map/MergedSpotID",
            "merge_map/FrameNr",
            "merge_map/PeakID",
            "id_rings/data",
            "ids_hash/data",
        ]),
        ("CalcRadius", [
            "radius_data/SpotID",
            "radius_data/IntegratedIntensity",
            "radius_data/Omega",
            "radius_data/YCen",
            "radius_data/ZCen",
            "radius_data/IMax",
            "radius_data/Radius",
            "radius_data/Eta",
            "radius_data/RingNr",
            "radius_data/GrainRadius",
            "radius_data/SigmaR",
            "radius_data/SigmaEta",
        ]),
        ("FitSetup (InputAll)", [
            "all_spots/data",
            "spots_to_index/data",
        ]),
        ("Grains (final)", [
            "grains/summary",
            "spot_matrix/data",
            "grain_ids_key/data",
        ]),
    ]
    
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
        for stage_name, datasets in stages:
            stage_ok = True
            stage_details = []
            
            for ds_path in datasets:
                if ds_path not in ref:
                    stage_details.append(f"    {ds_path}: SKIP (not in reference)")
                    n_skip += 1
                    continue
                if ds_path not in new:
                    stage_details.append(f"    {ds_path}: FAIL (not in new output)")
                    stage_ok = False
                    continue
                
                ref_data = ref[ds_path][()]
                new_data = new[ds_path][()]
                
                # Shape check
                if ref_data.shape != new_data.shape:
                    stage_details.append(
                        f"    {ds_path}: FAIL (shape mismatch: ref={ref_data.shape}, new={new_data.shape})")
                    stage_ok = False
                    continue
                
                # Data comparison
                if np.issubdtype(ref_data.dtype, np.floating):
                    if np.allclose(ref_data, new_data, atol=atol, rtol=rtol, equal_nan=True):
                        max_diff = np.nanmax(np.abs(ref_data - new_data)) if ref_data.size > 0 else 0.0
                        stage_details.append(f"    {ds_path}: PASS (max_diff={max_diff:.2e})")
                    else:
                        diff = np.abs(ref_data - new_data)
                        max_diff = np.nanmax(diff) if diff.size > 0 else 0.0
                        n_mismatch = np.sum(~np.isclose(ref_data, new_data, atol=atol, rtol=rtol, equal_nan=True))
                        stage_details.append(
                            f"    {ds_path}: FAIL (max_diff={max_diff:.2e}, mismatched={n_mismatch}/{ref_data.size})")
                        stage_ok = False
                elif np.issubdtype(ref_data.dtype, np.integer):
                    if np.array_equal(ref_data, new_data):
                        stage_details.append(f"    {ds_path}: PASS (exact match)")
                    else:
                        n_mismatch = np.sum(ref_data != new_data)
                        stage_details.append(
                            f"    {ds_path}: FAIL ({n_mismatch}/{ref_data.size} values differ)")
                        stage_ok = False
                else:
                    # String or other types — compare as-is
                    if np.array_equal(ref_data, new_data):
                        stage_details.append(f"    {ds_path}: PASS")
                    else:
                        stage_details.append(f"    {ds_path}: FAIL (values differ)")
                        stage_ok = False
            
            status = "PASS ✅" if stage_ok else "FAIL ❌"
            if stage_ok:
                n_pass += 1
            else:
                n_fail += 1
            
            print(f"  [{status}]  {stage_name}")
            for detail in stage_details:
                print(detail)
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


if __name__ == "__main__":
    main()
