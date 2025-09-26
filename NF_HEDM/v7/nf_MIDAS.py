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

# Silence the noisy Parsl loggers to keep the output clean.
for logger_name in ["parsl", "parsl.dataflow.dflow", "parsl.executors.threads"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

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
    """Get the environment variables for MIDAS, ensuring MIDAS_INSTALL_DIR is set."""
    env = dict(os.environ)
    if 'MIDAS_INSTALL_DIR' not in env:
        env['MIDAS_INSTALL_DIR'] = get_installation_dir()
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
    """Parses the MIDAS parameter file into a dictionary, handling different data types."""
    params = {}
    try:
        with open(param_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                key, value = parts
                # Type casting for known numeric parameters
                if key in ['nDistances', 'RawStartNr']:
                    params[key] = int(value)
                elif key in ['TomoPixelSize']:
                    params[key] = float(value)
                elif key == 'GridMask':
                    params[key] = [float(v) for v in value.split()]
                else:
                    params[key] = value # Default to string
    except FileNotFoundError:
        logger.error(f"Parameter file not found: {param_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to parse parameter file {param_file}: {e}")
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
def median(psFN: str, distanceNr: int, logDir: str, resultFolder: str, bin_dir: str) -> None:
    """Run median calculation remotely via Parsl."""
    import os, subprocess
    with open(f'{logDir}/median{distanceNr}_out.csv', 'w') as f, \
         open(f'{logDir}/median{distanceNr}_err.csv', 'w') as f_err:
        cmd = os.path.join(bin_dir, "MedianImageLibTiff") + f' {psFN} {distanceNr}'
        f.write(cmd)
        subprocess.call(cmd, shell=True, stdout=f, stderr=f_err, cwd=resultFolder, env=os.environ)

def median_local(distanceNr: int, psFN: str, logDir: str, resultFolder: str, bin_dir: str) -> int:
    """Run median calculation locally. Made platform-independent by accepting all args."""
    import os, subprocess
    with open(f'{logDir}/median{distanceNr}_out.csv', 'w') as f, \
         open(f'{logDir}/median{distanceNr}_err.csv', 'w') as f_err:
        cmd = os.path.join(bin_dir, "MedianImageLibTiff") + f' {psFN} {distanceNr}'
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
        cmd=os.path.join(install_dir, "FF_HEDM/bin/GetHKLListNF") + f" {args.paramFN}",
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
        cmd=os.path.join(bin_dir, "MakeDiffrSpots") + f" {args.paramFN}",
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
        if args.machineName == 'local':
            logger.info("Computing median locally using multiprocessing.")
            partial_median = partial(median_local, psFN=args.paramFN, logDir=logDir, resultFolder=resultFolder, bin_dir=bin_dir)
            with Pool(params['nDistances']) as p:
                work_items = range(1, params['nDistances'] + 1)
                list(tqdm(p.imap(partial_median, work_items), total=len(work_items), desc="Calculating Medians (Local)"))
        else:
            logger.info("Computing median remotely using Parsl.")
            median_futures = [median(args.paramFN, i, logDir, resultFolder, bin_dir) for i in range(1, params['nDistances'] + 1)]
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
    
    if args.refineParameters == 0:
        logger.info("Fitting orientations.")
        try:
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

                run_command(
                    cmd=os.path.join(bin_dir, "FitOrientationParametersMultiPoint") + f' {args.paramFN} {grid_point_nr}',
                    working_dir=resultFolder,
                    out_file=f'{logDir}/fit_singlepoint_out.csv',
                    err_file=f'{logDir}/fit_singlepoint_err.csv'
                )
            except Exception as e:
                logger.error(f"Failed during single-point parameter refinement: {e}", exc_info=True)
                sys.exit(1)
        else:
            logger.info("Refining parameters on multiple grid points defined in parameter file.")
            run_command(
                cmd=os.path.join(bin_dir, "FitOrientationParameters") + f' {args.paramFN}',
                working_dir=resultFolder,
                out_file=f'{logDir}/fit_multipoint_out.csv',
                err_file=f'{logDir}/fit_multipoint_err.csv'
            )
    
    logger.info(f"Fitting stage finished. Time taken: {time.time() - t0:.2f} seconds.")

# --- SYSTEM UTILITIES AND CONFIGURATION ---

def check_shared_memory_files() -> bool:
    """
    Check for specific shared memory files used by this script and verify their ownership.
    
    This is a more robust check that only targets the files this workflow creates,
    avoiding conflicts with other legitimate processes.
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
                    # Fallback for cases where UID might not be found in the password database
                    logger.warning(f"Could not verify owner of existing file '{filepath}'. Proceeding with caution.")

    except Exception as e:
        logger.warning(f"An unexpected error occurred while checking shared memory files: {e}")
        # It's safer to proceed with a warning than to halt the script on a check failure.
    
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
    
    global default_handler
    default_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    
    parser = MyParser(
        description='Near-field HEDM analysis using MIDAS. V7.0.0, contact hsharma@anl.gov',
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
    resultFolder = params.get('DataDirectory')
    if not resultFolder:
        logger.error("DataDirectory not found in parameter file.")
        sys.exit(1)
    
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
        try:
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