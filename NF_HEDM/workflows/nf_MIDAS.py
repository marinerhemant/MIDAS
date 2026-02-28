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
logger = logging.getLogger('MIDAS-NF')
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

import midas_config
midas_config.run_startup_checks()

from parsl.app.app import python_app

# --- CONSTANTS ---

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
        'GridPoints': 12
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

def run_preprocessing(args: argparse.Namespace, params: Dict, t0: float):
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
    """Handles combined median filtering and image processing in a single pass per layer."""
    logDir, resultFolder = params['logDir'], params['resultFolder']
    
    logger.info("Starting combined image processing stage (median + peak extraction).")
    try:
        cpusPerLayer = max(1, args.nCPUs)
        work_items = range(1, params['nDistances'] + 1)
        
        for distanceNr in tqdm(work_items, desc="Processing Layers (Combined)"):
            cmd = os.path.join(bin_dir, "ProcessImagesCombined") + f' {args.paramFN} {distanceNr} {cpusPerLayer}'
            run_command(
                cmd=cmd,
                working_dir=resultFolder,
                out_file=f'{logDir}/combined_image{distanceNr}_out.csv',
                err_file=f'{logDir}/combined_image{distanceNr}_err.csv'
            )
        
    except Exception as e:
        logger.error("A failure occurred during the image processing stage. Aborting workflow.")
        logger.error(f"Details: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info(f"Image processing finished. Time taken: {time.time() - t0:.2f} seconds.")

def run_fitting_and_postprocessing(args: argparse.Namespace, params: Dict, t0: float):
    """Handles memory mapping, fitting, and final parsing."""
    logDir, resultFolder = params['logDir'], params['resultFolder']

    logger.info("Mapping image info to memory-mapped files.")
    run_command(
        cmd=os.path.join(bin_dir, "MMapImageInfo") + f" {args.paramFN}",
        working_dir=resultFolder,
        out_file=f'{logDir}/map_out.csv',
        err_file=f'{logDir}/map_err.csv'
    )
    
    if args.refineParameters == 0:
        logger.info("Fitting orientations.")
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
    elif args.refineParameters == 1:
        logger.info("Refining parameters...")
        if args.multiGridPoints == 0:
            try:
                pos_input = input("Enter the x,y coordinates to optimize (e.g., 1.2,3.4): ")
                x, y = map(float, pos_input.split(','))
                
                grid_data = np.genfromtxt('grid.txt', skip_header=1)
                distances_sq = (grid_data[:, 2] - x)**2 + (grid_data[:, 3] - y)**2
                closest_index = np.argmin(distances_sq)
                grid_point_nr = closest_index + 1
                logger.info(f"Closest grid point to ({x},{y}) is #{grid_point_nr}.")

                out_file_path = f'{logDir}/fit_singlepoint_out.csv'
                run_command(
                    cmd=os.path.join(bin_dir, "FitOrientationParameters") + f' {args.paramFN} {grid_point_nr} {args.nCPUs}',
                    working_dir=resultFolder,
                    out_file=out_file_path,
                    err_file=f'{logDir}/fit_singlepoint_err.csv'
                )
                try:
                    with open(out_file_path, 'r') as f:
                        results = f.read().strip()
                    if results:
                        print("\n" + "="*25 + " REFINEMENT RESULTS " + "="*25)
                        print(results)
                        print("="*70 + "\n")
                        logger.info("Successfully displayed parameter refinement results to the console.")
                    else:
                        logger.warning(f"Refinement output file is empty: {out_file_path}")
                except FileNotFoundError:
                    logger.error(f"Could not find refinement output file to display: {out_file_path}")
            except Exception as e:
                logger.error(f"Failed during single-point parameter refinement: {e}", exc_info=True)
                sys.exit(1)
        else:
            logger.info("Refining parameters on multiple grid points defined in parameter file.")
            out_file_path = f'{logDir}/fit_multipoint_out.csv'
            run_command(
                cmd=os.path.join(bin_dir, "FitOrientationParametersMultiPoint") + f' {args.paramFN} {args.nCPUs}',
                working_dir=resultFolder,
                out_file=out_file_path,
                err_file=f'{logDir}/fit_multipoint_err.csv'
            )
            try:
                with open(out_file_path, 'r') as f:
                    results = f.read().strip()
                if results:
                    print("\n" + "="*25 + " REFINEMENT RESULTS " + "="*25)
                    print(results)
                    print("="*70 + "\n")
                    logger.info("Successfully displayed parameter refinement results to the console.")
                else:
                    logger.warning(f"Refinement output file is empty: {out_file_path}")
            except FileNotFoundError:
                logger.error(f"Could not find refinement output file to display: {out_file_path}")
    
    logger.info(f"Fitting stage finished. Time taken: {time.time() - t0:.2f} seconds.")


# --- SYSTEM UTILITIES AND CONFIGURATION ---
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
    
    os.environ['MIDAS_INSTALL_DIR'] = install_dir
    os.environ.setdefault('MIDAS_HOME', install_dir)
    
    global default_handler
    default_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    
    parser = MyParser(
        description='Near-field HEDM analysis using MIDAS. Contact: hsharma@anl.gov',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-paramFN', type=str, required=True, help='Parameter file name.')
    parser.add_argument('-nCPUs', type=int, default=10, help='Number of CPU cores to use if running locally.')
    parser.add_argument('-machineName', type=str, default='local', help='Machine name for execution.')
    parser.add_argument('-nNodes', type=int, default=1, help='Number of nodes for execution.')
    parser.add_argument('-ffSeedOrientations', type=int, default=0, help='Use seed orientations from far-field results (1=yes, 0=no).')
    parser.add_argument('-doImageProcessing', type=int, default=1, help='Perform image processing (1=yes, 0=no).')
    parser.add_argument('-refineParameters', type=int, default=0, help='Refine setup parameters (1=yes, 0=no).')
    parser.add_argument('-multiGridPoints', type=int, default=0, help='If refining parameters, use multiple grid points (1=yes, 0=no).')
    args = parser.parse_args()

    # --- 2. Configuration from Parsed Arguments and Files ---
    params = parse_parameters(args.paramFN)
    resultFolder = params.get('OutputDirectory', params.get('DataDirectory'))
    if not resultFolder:
        logger.error("Neither OutputDirectory nor DataDirectory found in parameter file.")
        sys.exit(1)
    if 'OutputDirectory' in params:
        logger.info(f"Using OutputDirectory for results: {resultFolder}")
    
    logDir = os.path.join(resultFolder, 'midas_log')
    params['logDir'] = logDir
    params['resultFolder'] = resultFolder

    os.makedirs(logDir, exist_ok=True)

    # Setup file logging to capture all output permanently
    log_file_path = os.path.join(logDir, 'midas_nf_workflow.log')
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
            run_preprocessing(args, params, t0)
            
            if args.doImageProcessing == 1:
                run_image_processing(args, params, t0)
                
            run_fitting_and_postprocessing(args, params, t0)

        finally:
            logger.info("Initiating Parsl cleanup.")
            parsl.dfk().cleanup()

    logger.info(f"Workflow completed successfully. Total time taken: {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()