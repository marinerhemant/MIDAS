import argparse
import os
import sys
import subprocess
import shutil
import stat
import zarr
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Benchmark Testing Suite for FF_HEDM")
    parser.add_argument("-nCPUs", type=int, default=1, help="Number of CPUs to use for the test")
    
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


def run_forward_simulation(param_file, nCPUs):
    """Runs ForwardSimulationCompressed."""
    midas_home = os.environ.get('MIDAS_HOME', str(Path(__file__).resolve().parent.parent))
    bin_path = Path(midas_home) / 'FF_HEDM' / 'bin' / 'ForwardSimulationCompressed'
    
    if not bin_path.exists():
        print(f"Error: {bin_path} not found. Please compile first.")
        sys.exit(1)
        
    cmd = [str(bin_path), str(param_file), str(nCPUs)]
    print(f"Running Simulation: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
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
    
    # 2. Modify Params & Resolve Paths
    test_param_file, params, out_file_base = create_testing_env(param_path, work_dir)
    print(f"Created temporary parameter environment: {test_param_file}")
    
    # 3. Execute Simulation
    # Note ForwardSimulationCompressed runs in cwd usually and might dump things locally too.
    # To be safe, we change to work_dir or give absolute path outputs.
    # OutFileName uses absolute paths, so executing here is fine.
    run_forward_simulation(test_param_file, args.nCPUs)
    
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

if __name__ == "__main__":
    main()
