#!/usr/bin/env python

import time
import parsl
import subprocess
import sys
import os
import argparse
import signal
import shutil
import logging
import numpy as np
from multiprocessing import Pool
from typing import List, Optional, Dict, Any, Union, Tuple, Generator
from functools import lru_cache, partial
from contextlib import contextmanager
import pwd
import getpass
from tqdm import tqdm

# --- SETUP: LOGGING AND DYNAMIC PATHS ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MIDAS-NF-MultiRes')
# Silence all Parsl loggers completely
logging.getLogger("parsl").setLevel(logging.CRITICAL)  # Only show critical errors
# Also silence these specific Parsl sub-loggers
for logger_name in ["parsl.dataflow.dflow", "parsl.dataflow.memoization", 
                    "parsl.process_loggers", "parsl.jobs.strategy",
                    "parsl.executors.threads"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

@lru_cache(maxsize=1)
def get_installation_dir() -> str:
    """Get the installation directory from the script's location. Cached for performance."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, '..', '..'))

install_dir = get_installation_dir()
utils_dir = os.path.join(install_dir, "utils")
v7_dir = os.path.join(install_dir, "NF_HEDM/v7")
bin_dir = os.path.join(install_dir, "NF_HEDM/bin")

sys.path.insert(0, utils_dir)
sys.path.insert(0, v7_dir)

from parsl.app.app import python_app

# --- CONSTANTS ---
SHM_FILES = ['SpotsInfo.bin', 'DiffractionSpots.bin', 'Key.bin', 'OrientMat.bin']

# --- HELPER FUNCTIONS: ENVIRONMENT, COMMANDS, PARSING ---

def get_midas_env() -> Dict[str, str]:
    """Get the environment variables for MIDAS, ensuring MIDAS_INSTALL_DIR and MIDAS_HOME are set."""
    env = dict(os.environ)
    if 'MIDAS_INSTALL_DIR' not in env:
        env['MIDAS_INSTALL_DIR'] = get_installation_dir()
    if 'MIDAS_HOME' not in env:
        env['MIDAS_HOME'] = get_installation_dir()
    return env

def run_command(cmd: str, working_dir: str, out_file: str, err_file: str) -> int:
    """Run a shell command with robust error handling and logging."""
    logger.info(f"Running: {cmd}")
    with open(out_file, 'w') as f_out, open(err_file, 'w') as f_err:
        f_out.write(cmd + "\n")
        process = subprocess.Popen(
            cmd, shell=True, stdout=f_out, stderr=f_err,
            cwd=working_dir, env=get_midas_env()
        )
        returncode = process.wait()
    if returncode != 0:
        with open(err_file, 'r') as f:
            error_content = f.read()
        error_msg = (f"Command failed with return code {returncode}:\n{cmd}\n"
                     f"Error output:\n{error_content}")
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    return returncode

def parse_parameters(param_file: str) -> Dict[str, Any]:
    """
    Parses the MIDAS parameter file into a dictionary. It reads a specific
    number of floats for designated multi-value keys and only the first value
    for all other keys, ignoring trailing comments or data.
    """
    params = {}
    # Define parameters that are exceptions and require multiple float values.
    # Format: { 'ParameterName': number_of_expected_floats }
    multi_value_float_exceptions = {
        'LatticeParameter': 6,
        'GridMask': 4,
        'BC': 2,
        'OmegaRange': 2,
        'BoxSize': 4,
        'BCTol': 2,
        'GridPoints': 12,
        'GridRefactor': 3 # StartingGridSize (float), ScalingFactor (float), NumLoops (int/float)
    }

    try:
        with open(param_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                
                key, value_string = parts
                values = value_string.split() # Splits by any whitespace (space, tab)

                if not values:
                    continue # Skip if there's a key but no value on the line

                # 1. Handle the special multi-value float parameters
                if key in multi_value_float_exceptions:
                    count = multi_value_float_exceptions[key]
                    if len(values) < count:
                        raise ValueError(f"Expected {count} values for key '{key}', but found {len(values)}.")
                    params[key] = [float(v) for v in values[:count]]

                # 2. Handle known single-value numeric parameters
                elif key in ['nDistances', 'RawStartNr']:
                    params[key] = int(values[0]) # Take only the first value and cast to int
                elif key in ['TomoPixelSize']:
                    params[key] = float(values[0]) # Take only the first value and cast to float

                # 3. Handle all other parameters (the general case)
                else:
                    params[key] = values[0] # Take only the first value and keep as string

    except FileNotFoundError:
        logger.error(f"Parameter file not found: {param_file}")
        sys.exit(1)
    except (ValueError, IndexError) as e:
        logger.error(f"Failed to parse line in {param_file}: '{line.strip()}'")
        logger.error(f"Reason: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing {param_file}: {e}")
        sys.exit(1)
        
    return params

def update_param_file(param_file: str, updates: Dict[str, str]):
    """Updates the parameter file with new values."""
    try:
        with open(param_file, 'r') as f:
            lines = f.readlines()
        
        with open(param_file, 'w') as f:
            for line in lines:
                key = line.split()[0] if line.split() else ""
                if key in updates:
                    # Preserve comments if possible? Simplified overwrite for robust replacement.
                    # We just write the new key value pair.
                    # Assuming standard formatting "Key Value ..."
                    f.write(f"{key} {updates[key]}\n")
                    del updates[key] # Mark as updated
                else:
                    f.write(line)
            
            # Append any new keys that weren't in the file
            for key, val in updates.items():
                f.write(f"{key} {val}\n")
                
    except Exception as e:
        logger.error(f"Failed to update parameter file {param_file}: {e}")
        raise

@contextmanager
def change_directory(new_dir: str) -> Generator[None, None, None]:
    """Context manager for safely changing the current working directory."""
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(old_dir)

# --- PARALLEL WORKER FUNCTIONS (PARSL & MULTIPROCESSING) ---

def create_app_with_retry(app_func):
    """Decorator to create a Parsl app with automatic retry logic."""
    @python_app
    def wrapped_app(*args, **kwargs):
        import time
        max_retries = 2  # Retry up to 2 times (3 attempts total)
        retry_delay = 10  # Seconds to wait between retries

        for attempt in range(max_retries + 1):
            try:
                # The original function is called here
                return app_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Parsl app {app_func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}. Error: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Parsl app {app_func.__name__} failed after all retries.")
                    raise  # Re-raise the final exception
    return wrapped_app

@create_app_with_retry
def median(psFN: str, distanceNr: int, numProcs: int, logDir: str, resultFolder: str, bin_dir: str) -> None:
    """Run median calculation remotely via Parsl."""
    import os, subprocess
    with open(f'{logDir}/median{distanceNr}_out.csv', 'w') as f, \
         open(f'{logDir}/median{distanceNr}_err.csv', 'w') as f_err:
        cmd = os.path.join(bin_dir, "MedianImageLibTiff") + f' {psFN} {distanceNr} {numProcs}'
        f.write(cmd)
        subprocess.call(cmd, shell=True, stdout=f, stderr=f_err, cwd=resultFolder, env=os.environ)

def median_local(distanceNr: int, psFN: str, numProcs: int, logDir: str, resultFolder: str, bin_dir: str) -> int:
    """Run median calculation locally. Made platform-independent by accepting all args."""
    import os, subprocess
    with open(f'{logDir}/median{distanceNr}_out.csv', 'w') as f, \
         open(f'{logDir}/median{distanceNr}_err.csv', 'w') as f_err:
        cmd = os.path.join(bin_dir, "MedianImageLibTiff") + f' {psFN} {distanceNr} {numProcs}'
        f.write(cmd)
        subprocess.call(cmd, shell=True, stdout=f, stderr=f_err, cwd=resultFolder, env=get_midas_env())
    return 1

@create_app_with_retry
def image(psFN: str, nodeNr: int, nNodes: int, numProcs: int, logDir: str, resultFolder: str, bin_dir: str) -> None:
    """Run image processing remotely via Parsl."""
    import os, subprocess
    with open(f'{logDir}/image{nodeNr}_out.csv', 'w') as f, \
         open(f'{logDir}/image{nodeNr}_err.csv', 'w') as f_err:
        cmd = os.path.join(bin_dir, "ImageProcessingLibTiffOMP") + f' {psFN} {nodeNr} {nNodes} {numProcs}'
        f.write(cmd)
        subprocess.call(cmd, shell=True, stdout=f, stderr=f_err, cwd=resultFolder, env=os.environ)

@create_app_with_retry
def fit(psFN: str, nodeNr: int, nNodes: int, numProcs: int, logDir: str, resultFolder: str, bin_dir: str) -> None:
    """Run orientation fitting remotely via Parsl."""
    import os, subprocess
    with open(f'{logDir}/fit{nodeNr}_out.csv', 'w') as f, \
         open(f'{logDir}/fit{nodeNr}_err.csv', 'w') as f_err:
        cmd = os.path.join(bin_dir, "FitOrientationOMP") + f' {psFN} {nodeNr} {nNodes} {numProcs}'
        f.write(cmd)
        subprocess.call(cmd, shell=True, stdout=f, stderr=f_err, cwd=resultFolder, env=os.environ)

# --- WORKFLOW STAGE FUNCTIONS ---

def run_preprocessing(args: argparse.Namespace, params: Dict, t0: float, skip_hex_grid: bool = False):
    """Handles HKLs, seed orientations, grid creation, and spot simulation."""
    logDir = params['logDir']
    resultFolder = params['resultFolder']

    logger.info("Making HKLs.")
    run_command(
        cmd=os.path.join(install_dir, "NF_HEDM/bin/GetHKLListNF") + f" {args.paramFN}",
        working_dir=resultFolder,
        out_file=f'{logDir}/hkls_out.csv',
        err_file=f'{logDir}/hkls_err.csv'
    )

    if args.ffSeedOrientations == 1:
        logger.info("Making seed orientations from far-field results.")
        # Ensure we have GrainsFile and SeedOrientations in params (refreshed)
        run_command(
            cmd=os.path.join(bin_dir, "GenSeedOrientationsFF2NFHEDM") + f" {params['GrainsFile']} {params['SeedOrientations']}",
            working_dir=resultFolder,
            out_file=f'{logDir}/seed_out.csv',
            err_file=f'{logDir}/seed_err.csv'
        )
        
    logger.info("Updating parameter file with orientation count.")
    try:
        nrOrientations = len(open(params['SeedOrientations']).readlines())
        with open(args.paramFN, 'r') as f:
            lines = [line for line in f if not line.strip().startswith('NrOrientations ')]
        lines.append(f'NrOrientations {nrOrientations}\n')
        with open(args.paramFN, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        raise RuntimeError(f"Failed to update parameter file with orientation count: {e}")

    if not skip_hex_grid:
        logger.info("Making and filtering reconstruction space.")
        run_command(
            cmd=os.path.join(bin_dir, "MakeHexGrid") + f" {args.paramFN}",
            working_dir=resultFolder,
            out_file=f'{logDir}/hex_out.csv',
            err_file=f'{logDir}/hex_err.csv'
        )

        if params.get('TomoImage') and len(params['TomoImage']) > 1:
            logger.info("Using tomo to filter reconstruction space.")
            run_command(
                cmd=os.path.join(bin_dir, "filterGridfromTomo") + f" {params['TomoImage']} {params['TomoPixelSize']}",
                working_dir=resultFolder,
                out_file=f'{logDir}/tomo_out.csv',
                err_file=f'{logDir}/tomo_err.csv'
            )
            shutil.move('grid.txt', 'grid_unfilt.txt'); shutil.move('gridNew.txt', 'grid.txt')
        elif params.get('GridMask') and len(params['GridMask']) > 0:
            logger.info("Applying grid mask.")
            mask = params['GridMask']
            gridpoints = np.genfromtxt('grid.txt', skip_header=1, delimiter=' ')
            gridpoints = gridpoints[
                (gridpoints[:, 2] >= mask[0]) & (gridpoints[:, 2] <= mask[1]) &
                (gridpoints[:, 3] >= mask[2]) & (gridpoints[:, 3] <= mask[3])
            ]
            logger.info(f'Filtered number of points: {gridpoints.shape[0]}')
            shutil.move('grid.txt', 'grid_old.txt')
            np.savetxt('grid.txt', gridpoints, fmt='%.6f', delimiter=' ', header=f'{gridpoints.shape[0]}', comments='')

    logger.info("Making simulated diffraction spots.")
    run_command(
        cmd=os.path.join(bin_dir, "MakeDiffrSpots") + f" {args.paramFN} {args.nCPUs}",
        working_dir=resultFolder,
        out_file=f'{logDir}/spots_out.csv',
        err_file=f'{logDir}/spots_err.csv'
    )
    logger.info(f"Preprocessing finished. Time taken: {time.time() - t0:.2f} seconds.")

def run_image_processing(args: argparse.Namespace, params: Dict, t0: float):
    """Handles median filtering and image processing stages."""
    logDir, resultFolder = params['logDir'], params['resultFolder']
    
    logger.info("Starting image processing stage.")
    try:
        cpusPerMedian = max(1, args.nCPUs // params['nDistances'])
        if args.machineName == 'local':
            logger.info(f"Computing median locally using multiprocessing ({cpusPerMedian} threads per process).")
            partial_median = partial(median_local, psFN=args.paramFN, numProcs=cpusPerMedian, logDir=logDir, resultFolder=resultFolder, bin_dir=bin_dir)
            with Pool(params['nDistances']) as p:
                work_items = range(1, params['nDistances'] + 1)
                list(tqdm(p.imap(partial_median, work_items), total=len(work_items), desc="Calculating Medians (Local)"))
        else:
            logger.info(f"Computing median remotely using Parsl ({cpusPerMedian} threads per task).")
            median_futures = [median(args.paramFN, i, cpusPerMedian, logDir, resultFolder, bin_dir) for i in range(1, params['nDistances'] + 1)]
            [f.result() for f in tqdm(median_futures, desc="Calculating Medians (Parsl)")]

        logger.info("Processing images in parallel.")
        image_futures = [image(args.paramFN, i, args.nNodes, args.nCPUs, logDir, resultFolder, bin_dir) for i in range(args.nNodes)]
        [f.result() for f in tqdm(image_futures, desc="Processing Images")]
        
    except Exception as e:
        logger.error("A failure occurred during the image processing stage. Aborting workflow.")
        logger.error(f"Details: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info(f"Image processing finished. Time taken: {time.time() - t0:.2f} seconds.")

def run_fitting_and_postprocessing(args: argparse.Namespace, params: Dict, t0: float):
    """Handles memory mapping, fitting, and final parsing."""
    logDir, resultFolder = params['logDir'], params['resultFolder']

    logger.info("Mapping image info to shared memory.")
    run_command(
        cmd=os.path.join(bin_dir, "MMapImageInfo") + f" {args.paramFN}",
        working_dir=resultFolder,
        out_file=f'{logDir}/map_out.csv',
        err_file=f'{logDir}/map_err.csv'
    )
    for f in SHM_FILES:
        shutil.copy2(f, f'/dev/shm/{f}')
    
    logger.info("Fitting orientations.")
    
    # Delete MicFileBinary if it exists to ensure a fresh start
    mic_file_binary = params.get('MicFileBinary')
    if mic_file_binary:
        mic_binary_path = os.path.join(resultFolder, mic_file_binary)
        if os.path.exists(mic_binary_path):
             logger.info(f"Deleting existing binary mic file: {mic_binary_path}")
             try:
                 os.remove(mic_binary_path)
             except OSError as e:
                 logger.warning(f"Could not delete {mic_binary_path}: {e}")
    
    try:
        if args.nNodes == 1:
            # --- Single Node: Monitor progress using fit0_out.csv line count ---
            grid_file = os.path.join(resultFolder, 'grid.txt')
            total_points = 0
            try:
                with open(grid_file, 'r') as f:
                    line = f.readline()
                    if line:
                        total_points = int(line.strip().split()[0])
            except Exception as e:
                logger.warning(f"Could not read total grid points from {grid_file}: {e}")
                total_points = None

            # Start the single fitting task (nodeNr=0)
            fit_future = fit(args.paramFN, 0, args.nNodes, args.nCPUs, logDir, resultFolder, bin_dir)
            
            outfile = os.path.join(logDir, 'fit0_out.csv')
            pbar = tqdm(total=total_points, desc="Fitting Orientations", unit="pts")
            last_count = 0
            
            while not fit_future.done():
                if os.path.exists(outfile):
                    try:
                        # Count lines (subtract 2 for header)
                        with open(outfile, 'rb') as f:
                            count = sum(1 for _ in f)
                        current_processed = max(0, count - 2)
                        if current_processed > last_count:
                            pbar.update(current_processed - last_count)
                            last_count = current_processed
                    except Exception:
                        pass
                time.sleep(1.0)
            
            # Final update ensures 100% if completed
            if os.path.exists(outfile):
                try:
                    with open(outfile, 'rb') as f:
                        count = sum(1 for _ in f)
                    current_processed = max(0, count - 2)
                    if current_processed > last_count:
                        pbar.update(current_processed - last_count)
                except Exception: pass
            
            pbar.close()
            fit_future.result() # Raise exception if task failed
        else:
            # --- Multi-Node: Standard Future completion tracking ---
            fit_futures = [fit(args.paramFN, i, args.nNodes, args.nCPUs, logDir, resultFolder, bin_dir) for i in range(args.nNodes)]
            [f.result() for f in tqdm(fit_futures, desc="Fitting Orientations")]
    except Exception as e:
        logger.error("A failure occurred during the orientation fitting stage. Aborting workflow.")
        logger.error(f"Details: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Parsing mic file.")
    run_command(
        cmd=os.path.join(bin_dir, "ParseMic") + f" {args.paramFN}",
        working_dir=resultFolder,
        out_file=f'{logDir}/parse_out.csv',
        err_file=f'{logDir}/parse_err.csv'
    )
    
    logger.info(f"Fitting stage finished. Time taken: {time.time() - t0:.2f} seconds.")

def run_multi_resolution_workflow(args, params, t0):
    """Orchestrates the multi-resolution looping."""
    starting_grid_size = params['GridRefactor'][0]
    scaling_factor = params['GridRefactor'][1]
    num_loops = int(params['GridRefactor'][2])
    mic_file_base = params['MicFileText']
    seed_file_base = params['SeedOrientations']
    
    # Capture runtime-injected keys that are NOT in the parameter file.
    # These must be re-injected after every parse_parameters() call.
    _logDir = params['logDir']
    _resultFolder = params['resultFolder']
    
    def reload_params():
        """Re-parse the parameter file and re-inject runtime keys."""
        p = parse_parameters(args.paramFN)
        p['logDir'] = _logDir
        p['resultFolder'] = _resultFolder
        return p
    
    # Validate required keys early
    if 'SeedOrientationsAll' not in params:
        logger.error("'SeedOrientationsAll' not found in parameter file. Required for multi-resolution workflow.")
        sys.exit(1)
    
    logger.info(f"Starting Multi-Resolution Workflow: {num_loops} loops, "
                f"StartingGridSize: {starting_grid_size}, Scaling: {scaling_factor}")

    # --- LOOP 0: Initial Run ---
    logger.info(">>> Running Loop 0 (Initial Coarse Pass)")
    
    # 1. Update MicFileText -> .0 and ensure GridSize is set to starting value 
    #    (recovers from interrupted runs where GridSize was overwritten)
    update_param_file(args.paramFN, {
        'MicFileText': f"{mic_file_base}.0",
        'GridSize': f"{starting_grid_size:.6f}"
    })
    
    # 2. Run standard workflow Unseeded
    args.ffSeedOrientations = 0 # Ensure unseeded
    params = reload_params()
    
    run_preprocessing(args, params, t0)
    if args.doImageProcessing == 1:
        run_image_processing(args, params, t0)
    run_fitting_and_postprocessing(args, params, t0)
    
    # 3. Backup SeedOrientationsAll
    seed_all_backup = f"{params['SeedOrientationsAll']}_Backup"
    shutil.copy2(params['SeedOrientationsAll'], seed_all_backup)
    logger.info(f"Backed up SeedOrientationsAll to {seed_all_backup}")
    
    current_mic_file = f"{mic_file_base}.0" # Output of Loop 0
    
    # --- REFINEMENT LOOPS ---
    for loop_idx in range(1, num_loops + 1):
        logger.info(f">>> Running Loop {loop_idx} (Refinement)")
        
        # Disable Image Processing for internal loops (reuses initial images)
        if args.doImageProcessing == 1:
            logger.info("Disabling image processing for refinement loops (images reused).")
            args.doImageProcessing = 0
        
        # a. Refine Grid
        new_grid_size = starting_grid_size / (scaling_factor ** loop_idx)
        logger.info(f"Refining GridSize to {new_grid_size}")
        update_param_file(args.paramFN, {'GridSize': f"{new_grid_size:.6f}"})
        
        # b. Generate Cluster Seeds from Previous Mic File
        grains_file = f"Grains.csv.{loop_idx}"
        prev_mic_path = os.path.join(params['resultFolder'], current_mic_file)
        grains_path = os.path.join(params['resultFolder'], grains_file)
        
        logger.info(f"Generating clustered seeds from {current_mic_file} -> {grains_file}")
        
        # Mic2GrainsList <ParamFile> <PrevMicFile> <GrainsFile>
        run_command(
            cmd=os.path.join(bin_dir, "Mic2GrainsList") + f" {args.paramFN} {prev_mic_path} {grains_path}",
            working_dir=params['resultFolder'],
            out_file=f"{params['logDir']}/mic2grains_{loop_idx}_out.csv",
            err_file=f"{params['logDir']}/mic2grains_{loop_idx}_err.csv"
        )
        
        # c. Pass 1: Seeded Run
        logger.info(f"--- Loop {loop_idx} Pass 1: Seeded Run ---")
        
        seed_orientations_loop = f"{seed_file_base}.{loop_idx}"
        target_mic_pass1 = f"{mic_file_base}.{loop_idx}"
        
        update_param_file(args.paramFN, {
            'GrainsFile': grains_path,
            'MicFileText': target_mic_pass1,
            'SeedOrientations': seed_orientations_loop # Will be created by GenSeed...
        })
        
        args.ffSeedOrientations = 1 # Enable seeded mode
        params = reload_params()
        
        run_preprocessing(args, params, t0, skip_hex_grid=False) # MAKE FULL HEX GRID
        
        # Capture the full grid for later (Pass 2 Needs exact columns)
        grid_pass1_map = {}
        with open(os.path.join(params['resultFolder'], 'grid.txt'), 'r') as f:
            for line in f:
                vals = line.split()
                if len(vals) < 5: continue
                # Key by X(2), Y(3) precise string or float?
                # Best to key by float with tolerance or format string.
                # ParseMic output X Y will be formatted.
                # Let's key by (float(x), float(y)) for robust lookup.
                try:
                    kx, ky = float(vals[2]), float(vals[3])
                    grid_pass1_map[(kx, ky)] = line.strip()
                except ValueError: continue

        if args.doImageProcessing == 1:
             run_image_processing(args, params, t0)
             
        run_fitting_and_postprocessing(args, params, t0)
        
        # d. Filter & Grid Update for "Bad" Solutions
        logger.info(f"Filtering bad solutions from Pass 1.")
        mic_pass1_path = os.path.join(params['resultFolder'], target_mic_pass1)
        
        params = reload_params()
        if 'MinConfidence' not in params:
            logger.warning("'MinConfidence' not found in parameter file. Using default 0.5.")
        min_conf = float(params.get('MinConfidence', '0.5'))
        
        bad_points_lines = []
        mic_pass1_good_lines = [] # Keep good lines for merge
        
        header_lines = []
        with open(mic_pass1_path, 'r') as f:
            for line in f:
                if line.startswith('%'):
                    header_lines.append(line)
                    continue
                vals = line.split()
                if len(vals) < 11: continue
                conf = float(vals[10]) # Column 10 is confidence
                
                x_val = float(vals[3])
                y_val = float(vals[4])
                
                if conf < min_conf:
                    # Lookup original grid line
                    # Note: float comparison equality is risky. Using small tolerance?
                    # Or matching based on proximity.
                    # Simple matching: find closest in map?
                    # The grid generation is deterministic. The Mic output should be exact representation of a grid point.
                    # We iterate through map? No, slow.
                    # Try direct match first.
                    found_line = grid_pass1_map.get((x_val, y_val))
                    if not found_line:
                        # Fallback: search with tolerance
                        for (gx, gy), gline in grid_pass1_map.items():
                             if abs(gx - x_val) < 1e-5 and abs(gy - y_val) < 1e-5:
                                 found_line = gline
                                 break
                    
                    if found_line:
                        bad_points_lines.append(found_line)
                    else:
                        logger.warning(f"Could not find original grid line for bad point ({x_val}, {y_val})")
                else:
                    mic_pass1_good_lines.append(line)

        logger.info(f"Found {len(bad_points_lines)} bad points. Writing to grid.txt.")
        
        if not bad_points_lines:
            logger.info("No bad points found. Skipping Pass 2 (Unseeded).")
            current_mic_file = target_mic_pass1
            continue
            
        with open(os.path.join(params['resultFolder'], 'grid.txt'), 'w') as f:
            f.write(f"{len(bad_points_lines)}\n")
            for p in bad_points_lines:
                f.write(f"{p}\n")
                
        # e. Pass 2: Unseeded Run (Bad Regions)
        logger.info(f"--- Loop {loop_idx} Pass 2: Unseeded Run (Bad Regions) ---")
        
        target_mic_pass2 = f"{mic_file_base}_all_solutions.{loop_idx}"
        
        update_param_file(args.paramFN, {
            'MicFileText': target_mic_pass2,
            'SeedOrientations': seed_all_backup # Use the full backup
        })
        
        args.ffSeedOrientations = 0
        params = reload_params()
        
        # SKIP MakeHexGrid
        run_preprocessing(args, params, t0, skip_hex_grid=True) 
        if args.doImageProcessing == 1:
             run_image_processing(args, params, t0)
        run_fitting_and_postprocessing(args, params, t0)
        
        # f. Merge Results
        logger.info("Merging Pass 1 and Pass 2 results.")
        
        final_lines = list(mic_pass1_good_lines) # Start with good from Pass 1
        
        # Read Pass 2 (Recovered bad)
        mic_pass2_path = os.path.join(params['resultFolder'], target_mic_pass2)
        if os.path.exists(mic_pass2_path):
            with open(mic_pass2_path, 'r') as f:
                for line in f:
                    if line.startswith('%'): continue
                    # Add all results from Pass 2 (filtered or unfiltered based on confidence? User suggested Mic2GrainsList handles filtering)
                    # We just merge everything.
                    final_lines.append(line)
        
        # Sort by Y (col 4), then X (col 3)
        def sort_key(line):
            v = line.split()
            # Ensure float parsing is robust
            try:
                return (float(v[4]), float(v[3])) 
            except:
                return (0.0, 0.0)
            
        final_lines.sort(key=sort_key)
        
        # Write merged file to a NEW filename so Pass 1 and Pass 2 are preserved
        current_mic_file = f"{mic_file_base}_merged.{loop_idx}"
        final_path = os.path.join(params['resultFolder'], current_mic_file)
        
        with open(final_path, 'w') as f:
            for h in header_lines:
                f.write(h)
            for l in final_lines:
                f.write(l)
                
        logger.info(f"Loop {loop_idx} complete. Merged result in {current_mic_file}.")

# --- SYSTEM UTILITIES AND CONFIGURATION ---

def check_shared_memory_files() -> bool:
    """
    Check for specific shared memory files used by this script and verify their ownership.
    """
    try:
        current_user = getpass.getuser()
        
        for filename in SHM_FILES:
            filepath = os.path.join('/dev/shm', filename)
            
            if os.path.exists(filepath):
                # File exists, now check the owner.
                try:
                    stat_info = os.stat(filepath)
                    owner_uid = stat_info.st_uid
                    owner_name = pwd.getpwuid(owner_uid).pw_name
                    
                    if owner_name != current_user:
                        logger.error(f"Conflict detected in shared memory.")
                        logger.error(f"File '{filepath}' already exists and is owned by '{owner_name}'.")
                        logger.error(f"The current user is '{current_user}'. Please have the other user clean up their files.")
                        return False
                except (KeyError, AttributeError):
                    logger.warning(f"Could not verify owner of existing file '{filepath}'. Proceeding with caution.")

    except Exception as e:
        logger.warning(f"An unexpected error occurred while checking shared memory files: {e}")
    
    return True

def load_machine_config(machine_name: str, n_nodes: int, num_procs: int) -> Tuple[int, int]:
    """Load machine configuration and set up Parsl."""
    configs = {
        'local': ('localConfig', 'localConfig', num_procs, 1),
        'orthrosnew': ('orthrosAllConfig', 'orthrosNewConfig', 32, 11),
        'orthrosall': ('orthrosAllConfig', 'orthrosAllConfig', 64, 5),
        'umich': ('uMichConfig', 'uMichConfig', 36, n_nodes),
        'marquette': ('marquetteConfig', 'marquetteConfig', 36, n_nodes),
        'purdue': ('purdueConfig', 'purdueConfig', 128, n_nodes)
    }
    if machine_name not in configs:
        raise ValueError(f"Unknown machine name: {machine_name}")

    module_name, config_name, procs, nodes = configs[machine_name]
    if machine_name not in ['local', 'orthrosnew', 'orthrosall']:
        os.environ['nNodes'] = str(nodes)
    
    # Dynamically import the configuration module
    module = __import__(module_name)
    parsl.load(getattr(module, config_name))
    return procs, nodes

class MyParser(argparse.ArgumentParser):
    """Custom argument parser to print help on error."""
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)

default_handler = None
def handler(num, frame):
    """Signal handler for Ctrl+C. The finally block in main handles cleanup."""
    logger.info("Ctrl-C pressed, exiting gracefully.")
    if default_handler:
        default_handler(num, frame)
    sys.exit(1)

# --- MAIN ORCHESTRATOR ---

def main():
    """Main function to set up and run the data processing workflow."""
    t0 = time.time()
    
    # --- 1. Initial Setup and Argument Parsing ---
    if not check_shared_memory_files(): sys.exit(1)
    
    os.environ['MIDAS_INSTALL_DIR'] = install_dir
    os.environ.setdefault('MIDAS_HOME', install_dir)
    
    global default_handler
    default_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    
    parser = MyParser(
        description=(
            'Near-field HEDM analysis using MIDAS (Multi-Resolution). V7.0.0, contact hsharma@anl.gov\n\n'
            'Additional parameter file keys for multi-resolution mode:\n'
            '  GridRefactor <StartingGridSize> <ScalingFactor> <NumLoops>  # e.g. GridRefactor 5.0 2.0 3\n'
            '      - StartingGridSize: The initial grid size value (preserved across interrupted runs).\n'
            '      - ScalingFactor: Factor by which StartingGridSize is divided in each loop.\n'
            '      - NumLoops: Number of refinement iterations to perform.\n'
            '  SeedOrientationsAll <path>               # Full seed orientations file (backed up internally as <path>_Backup)\n'
            '                                           # NOTE: Should be distinct from SeedOrientations.\n'
            '  MinConfidence <value>                    # Confidence threshold for filtering bad solutions (default: 0.5)\n'
            '  GridSize <value>                         # Grid size (overwritten during workflow; use StartingGridSize in GridRefactor)\n'
            '  MicFileText <basename>                   # Base name for mic output (suffixed with .0, .1, ... per loop)\n'
            '  SeedOrientations <path>                  # Seed orientations file (unique copy created per loop: <path>.N)\n'
            '                                           # NOTE: The workflow will generate <path>.N files, preserving original.\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-paramFN', type=str, required=True, help='Parameter file name.')
    parser.add_argument('-nCPUs', type=int, default=10, help='Number of CPU cores to use if running locally.')
    parser.add_argument('-machineName', type=str, default='local', help='Machine name for execution.')
    parser.add_argument('-nNodes', type=int, default=1, help='Number of nodes for execution.')
    parser.add_argument('-ffSeedOrientations', type=int, default=0, help='Use seed orientations from far-field results (1=yes, 0=no).')
    parser.add_argument('-doImageProcessing', type=int, default=1, help='Perform image processing (1=yes, 0=no).')
    args = parser.parse_args()

    # --- 2. Configuration from Parsed Arguments and Files ---
    params = parse_parameters(args.paramFN)
    resultFolder = params.get('DataDirectory')
    if not resultFolder:
        logger.error("DataDirectory not found in parameter file.")
        sys.exit(1)
    
    logDir = os.path.join(resultFolder, 'midas_log')
    params['logDir'] = logDir
    params['resultFolder'] = resultFolder

    os.makedirs(logDir, exist_ok=True)

    # Setup file logging to capture all output permanently
    log_file_path = os.path.join(logDir, 'midas_nf_workflow_multires.log')
    file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(file_handler) # Add handler to the root logger
    logger.info(f"Logging to console and to file: {log_file_path}")

    os.environ['MIDAS_SCRIPT_DIR'] = resultFolder
    
    try:
        args.nCPUs, args.nNodes = load_machine_config(args.machineName, args.nNodes, args.nCPUs)
    except Exception as e:
        logger.error(f"Failed to load machine configuration: {e}", exc_info=True)
        sys.exit(1)
    
    os.makedirs(logDir, exist_ok=True)
    
    # --- 3. Workflow Execution with Guaranteed Cleanup ---
    with change_directory(resultFolder):
        # Delete stale MicFileBinary if it exists from a previous run
        mic_bin = params.get('MicFileBinary')
        if mic_bin:
            mic_bin_path = os.path.join(resultFolder, mic_bin)
            if os.path.exists(mic_bin_path):
                os.remove(mic_bin_path)
                logger.info(f"Removed stale MicFileBinary: {mic_bin_path}")

        # Ensure the reduced data subdirectory exists
        reduced_fn = params.get('ReducedFileName', '')
        reduced_subdir = os.path.dirname(reduced_fn)
        if reduced_subdir:
            reduced_dir_path = os.path.join(resultFolder, reduced_subdir)
            os.makedirs(reduced_dir_path, exist_ok=True)
            logger.info(f"Ensured reduced data directory exists: {reduced_dir_path}")

        try:
            # Check for GridRefactor parameter to determine workflow
            if 'GridRefactor' in params:
                 run_multi_resolution_workflow(args, params, t0)
            else:
                 run_preprocessing(args, params, t0)
                 if args.doImageProcessing == 1:
                     run_image_processing(args, params, t0)
                 run_fitting_and_postprocessing(args, params, t0)

        finally:
            logger.info("Initiating final cleanup of shared memory and Parsl.")
            for f in SHM_FILES:
                try:
                    path = f'/dev/shm/{f}'
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Could not remove /dev/shm/{f}: {e}")
            parsl.dfk().cleanup()

    logger.info(f"Workflow completed successfully. Total time taken: {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
