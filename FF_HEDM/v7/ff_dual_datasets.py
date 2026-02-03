#!/usr/bin/env python

import parsl
import subprocess
import sys
import os
import time
import argparse
import signal
import shutil
import re
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union, Generator
from functools import lru_cache
from contextlib import contextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MIDAS_DualDataset')
# Silence all Parsl loggers completely
logging.getLogger("parsl").setLevel(logging.CRITICAL)
for logger_name in ["parsl.dataflow.dflow", "parsl.dataflow.memoization", 
                    "parsl.process_loggers", "parsl.jobs.strategy",
                    "parsl.executors.threads"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Set paths dynamically using script location
@lru_cache(maxsize=1)
def get_installation_dir() -> str:
    """Get the installation directory from the script's location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    install_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    return install_dir

# Get paths
install_dir = get_installation_dir()
utils_dir = os.path.join(install_dir, "utils")
v7_dir = os.path.join(install_dir, "FF_HEDM/v7")
bin_dir = os.path.join(install_dir, "FF_HEDM/bin")

# Add paths to sys.path
sys.path.insert(0, utils_dir)
sys.path.insert(0, v7_dir)

from parsl.app.app import python_app
pytpath = sys.executable

@contextmanager
def change_directory(new_dir: str) -> Generator[None, None, None]:
    """Context manager for changing directory."""
    if not os.path.exists(new_dir):
        raise FileNotFoundError(f"Directory does not exist: {new_dir}")
        
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        try:
            os.chdir(old_dir)
        except Exception as e:
            logger.error(f"Failed to change back to original directory {old_dir}: {e}")
            try:
                os.chdir(os.path.expanduser("~"))
            except:
                pass

@contextmanager
def cleanup_context():
    """Context manager for cleanup on exit."""
    try:
        yield
    finally:        
        try:
            parsl.dfk().cleanup()
        except Exception as e:
            logger.error(f"Failed to clean up Parsl: {e}")

def safely_run_command(cmd: str, working_dir: str, out_file: str, err_file: str, 
                       task_name: str = "Command") -> int:
    """Run a shell command with improved error handling."""
    logger.info(f"Running: {cmd}")
    
    try:
        with open(out_file, 'w') as f_out, open(err_file, 'w') as f_err:
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=f_out, 
                stderr=f_err, 
                cwd=working_dir,
                env=get_midas_env()
            )
            returncode = process.wait()
            
        if returncode != 0:
            with open(err_file, 'r') as f:
                error_content = f.read()
            error_msg = f"{task_name} failed with return code {returncode}:\n{cmd}\nError output:\n{error_content}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        return returncode
    except Exception as e:
        if not isinstance(e, RuntimeError):
            logger.error(f"Exception during {task_name}: {e}")
            raise RuntimeError(f"{task_name} failed: {str(e)}")
        raise

def get_midas_env() -> Dict[str, str]:
    """Get the environment variables for MIDAS."""
    env = dict(os.environ)
    if 'MIDAS_INSTALL_DIR' not in env:
        env['MIDAS_INSTALL_DIR'] = get_installation_dir()
    return env

def read_parameter_file(psFN: str) -> Dict[str, str]:
    """Read parameters from file into a dictionary."""
    if not os.path.exists(psFN):
        raise FileNotFoundError(f"Parameter file not found: {psFN}")
        
    params = {}
    with open(psFN, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                params[parts[0]] = parts[1].strip()
    return params

def parse_int_param(param_dict: Dict[str, str], name: str, default: int = 0) -> int:
    """Parse an integer parameter from the parameter dictionary."""
    if name not in param_dict:
        return default
    try:
        return int(param_dict[name])
    except ValueError:
        logger.warning(f"Invalid integer value for {name}: {param_dict[name]}, using default {default}")
        return default

def generateZip(
    resFol: str,
    pfn: str,
    dfn: str,
    nchunks: int = -1,
    preproc: int = -1,
    outf: str = 'ZipOut.txt',
    errf: str = 'ZipErr.txt'
) -> Optional[str]:
    """Generate ZIP file from data."""
    # Assuming layer number is always 1 for this workflow
    layerNr = 1
    params = read_parameter_file(pfn)
    numFilesPerScan = parse_int_param(params, 'NrFilesPerSweep', default=1)

    cmd = f"{pytpath} {os.path.join(utils_dir, 'ffGenerateZipRefactor.py')} -resultFolder {resFol} -paramFN {pfn} -LayerNr {layerNr} -dataFN {dfn}"
    
    if nchunks != -1:
        cmd += f' -numFrameChunks {nchunks}'
    if preproc != -1:
        cmd += f' -preProcThresh {preproc}'
    if numFilesPerScan > 1:
        cmd += f' -numFilesPerScan {numFilesPerScan}'
        
    outf_path = f"{resFol}/output/{outf}"
    errf_path = f"{resFol}/output/{errf}"
    
    try:
        print(cmd)
        safely_run_command(cmd, resFol, outf_path, errf_path, task_name="ZIP generation")
        
        with open(outf_path, 'r') as f:
            lines = f.readlines()
            
        if lines and lines[-1].startswith('OutputZipName'):
            return lines[-1].split()[1]
        else:
            logger.error("Could not find OutputZipName in the output")
            return None
    except Exception as e:
        logger.error(f"Failed to generate ZIP: {e}")
        return None

def create_app_with_retry(app_func):
    """Decorator to create a Parsl app with retry logic."""
    @python_app
    def wrapped_app(*args, **kwargs):
        import logging, time
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger(f'MIDAS_{app_func.__name__}')
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                return app_func(*args, **kwargs, logger=logger)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    raise
    return wrapped_app

def _peaks_impl(resultDir: str, zipFN: str, numProcs: int, bin_dir: str, blockNr: int = 0, numBlocks: int = 1, logger=None):
    """Implementation of peak search function."""
    import subprocess, os
    if logger is None:
        import logging
        logger = logging.getLogger('MIDAS_peaks')
    os.makedirs(f'{resultDir}/output', exist_ok=True)
    outfile = f'{resultDir}/output/peaksearch_out{blockNr}.csv'
    errfile = f'{resultDir}/output/peaksearch_err{blockNr}.csv'
    env = dict(os.environ)
    logger.info(f"Running PeaksFittingOMPZarrRefactor in {resultDir} for block {blockNr}/{numBlocks}")
    with open(outfile, 'w') as f, open(errfile, 'w') as f_err:
        cmd = f"{os.path.join(bin_dir, 'PeaksFittingOMPZarrRefactor')} {zipFN} {blockNr} {numBlocks} {numProcs}"
        logger.info(f"Executing command: {cmd}")
        process = subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=f_err, cwd=resultDir)
        returncode = process.wait()
        if returncode != 0:
            with open(errfile, 'r') as err_reader: error_content = err_reader.read()
            raise RuntimeError(f"Peak search failed with return code {returncode}. Error output:\n{error_content}")
        logger.info(f"PeaksFittingOMPZarrRefactor completed successfully for block {blockNr}/{numBlocks}")
peaks = create_app_with_retry(_peaks_impl)

def _index_impl(resultDir: str, numProcs: int, bin_dir: str, blockNr: int = 0, numBlocks: int = 1, logger=None):
    """Implementation of indexing function."""
    import subprocess, os
    if logger is None:
        import logging
        logger = logging.getLogger('MIDAS_index')
    with open(os.path.join(resultDir, "SpotsToIndex.csv"), "r") as f: num_lines = len(f.readlines())
    logger.info(f"Found {num_lines} spots to index")
    outfile = f'{resultDir}/output/indexing_out{blockNr}.csv'
    errfile = f'{resultDir}/output/indexing_err{blockNr}.csv'
    logger.info(f"Running IndexerOMP in {resultDir} for block {blockNr}/{numBlocks}")
    with open(outfile, 'w') as f, open(errfile, 'w') as f_err:
        cmd = f"{os.path.join(bin_dir, 'IndexerOMP')} paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}"
        logger.info(f"Executing command: {cmd}")
        process = subprocess.Popen(cmd, shell=True, env=dict(os.environ), stdout=f, stderr=f_err, cwd=resultDir)
        returncode = process.wait()
        if returncode != 0:
            with open(errfile, 'r') as err_reader: error_content = err_reader.read()
            raise RuntimeError(f"Indexing failed with return code {returncode}. Error output:\n{error_content}")
        logger.info(f"IndexerOMP completed successfully for block {blockNr}/{numBlocks}")
index = create_app_with_retry(_index_impl)

def _refine_impl(resultDir: str, numProcs: int, bin_dir: str, blockNr: int = 0, numBlocks: int = 1, logger=None):
    """Implementation of refinement function."""
    import subprocess, os, resource
    if logger is None:
        import logging
        logger = logging.getLogger('MIDAS_refine')
    with open(os.path.join(resultDir, "SpotsToIndex.csv"), "r") as f: num_lines = len(f.readlines())
    logger.info(f"Found {num_lines} spots to refine")
    outfile = f'{resultDir}/output/refining_out{blockNr}.csv'
    errfile = f'{resultDir}/output/refining_err{blockNr}.csv'
    logger.info(f"Running FitPosOrStrainsOMP in {resultDir} for block {blockNr}/{numBlocks}")
    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    with open(outfile, 'w') as f, open(errfile, 'w') as f_err:
        cmd = f"{os.path.join(bin_dir, 'FitPosOrStrainsOMP')} paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}"
        logger.info(f"Executing command: {cmd}")
        process = subprocess.Popen(cmd, shell=True, env=dict(os.environ), stdout=f, stderr=f_err, cwd=resultDir)
        returncode = process.wait()
        if returncode != 0:
            with open(errfile, 'r') as err_reader: error_content = err_reader.read()
            raise RuntimeError(f"Refinement failed with return code {returncode}. Error output:\n{error_content}")
        logger.info(f"FitPosOrStrainsOMP completed successfully for block {blockNr}/{numBlocks}")
refine = create_app_with_retry(_refine_impl)

# Signal handler for cleanup
default_handler = None
def handler(num, frame):
    """Handle Ctrl+C by cleaning up and exiting."""
    try:
        logger.info("Ctrl-C was pressed, cleaning up.")
        parsl.dfk().cleanup()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        return default_handler(num, frame)

class MyParser(argparse.ArgumentParser):
    """Custom argument parser with better error handling."""
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)

def load_machine_config(machine_name: str, n_nodes: int, num_procs: int) -> Tuple[int, int]:
    """Load machine configuration and set up Parsl."""
    if machine_name == 'local':
        import localConfig
        parsl.load(config=localConfig.localConfig)
        return num_procs, 1
    elif machine_name in ['orthrosnew', 'orthrosall', 'umich', 'marquette', 'purdue']:
        if machine_name == 'orthrosnew':
            import orthrosAllConfig
            parsl.load(config=orthrosAllConfig.orthrosNewConfig)
            return 32, 11
        elif machine_name == 'orthrosall':
            import orthrosAllConfig
            parsl.load(config=orthrosAllConfig.orthrosAllConfig)
            return 64, 5
        elif machine_name == 'umich':
            import uMichConfig
            os.environ['nNodes'] = str(n_nodes)
            parsl.load(config=uMichConfig.uMichConfig)
            return 36, n_nodes
        elif machine_name == 'marquette':
            import marquetteConfig
            os.environ['nNodes'] = str(n_nodes)
            parsl.load(config=marquetteConfig.marquetteConfig)
            return 36, n_nodes
        elif machine_name == 'purdue':
            import purdueConfig
            os.environ['nNodes'] = str(n_nodes)
            parsl.load(config=purdueConfig.purdueConfig)
            return 128, n_nodes
    else:
        raise ValueError(f"Unknown machine name: {machine_name}")

def setup_output_directories(result_dir: str) -> None:
    """Set up output directories."""
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/output", exist_ok=True)
    os.makedirs(f"{result_dir}/Temp", exist_ok=True)

def process_dataset_until_binning(
    dataset_id: int, top_res_dir: str, ps_fn: str, data_fn: str, 
    num_procs: int, n_nodes: int, n_chunks: int, preproc: int, bin_directory: str
) -> str:
    """
    Processes a single dataset through all steps up to and including data binning.
    Returns the path to the result directory for this dataset.
    """
    result_dir = os.path.join(top_res_dir, f'dataset_{dataset_id}_analysis')
    logger.info(f"Processing Dataset {dataset_id}: results will be saved in {result_dir}")
    
    setup_output_directories(result_dir)
    shutil.copy2(ps_fn, result_dir)

    # Check for PanelShiftsFile and copy if exists
    try:
        params = read_parameter_file(ps_fn)
        if 'PanelShiftsFile' in params:
             psShiftFile = params['PanelShiftsFile']
             if os.path.exists(psShiftFile):
                 shutil.copy2(psShiftFile, result_dir)
                 logger.info(f"[{dataset_id}] Copied PanelShiftsFile {psShiftFile} to {result_dir}")
             else:
                 logger.warning(f"[{dataset_id}] PanelShiftsFile specified {psShiftFile} but does not exist.")
    except Exception as e:
        logger.error(f"[{dataset_id}] Failed to copy PanelShiftsFile: {e}")
    
    t0 = time.time()
    
    logger.info(f"[{dataset_id}] Generating combined MIDAS file...")
    outFStem = generateZip(result_dir, ps_fn, dfn=data_fn, nchunks=n_chunks, preproc=preproc)
    if not outFStem:
        raise RuntimeError(f"[{dataset_id}] Failed to generate ZIP file")
    
    cmdUpd = f'{pytpath} {os.path.join(utils_dir, "updateZarrDset.py")} -fn {os.path.basename(outFStem)} -folder {result_dir} -keyToUpdate analysis/process/analysis_parameters/ResultFolder -updatedValue {result_dir}/'
    try:
        subprocess.check_call(cmdUpd, shell=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[{dataset_id}] Failed to update zarr dataset: {e}")
            
    logger.info(f"[{dataset_id}] Generating HKLs. Time till now: {time.time() - t0:.2f} seconds.")
    cmd_hkl = f"{os.path.join(bin_directory, 'GetHKLListZarr')} {outFStem}"
    safely_run_command(cmd_hkl, result_dir, f'{result_dir}/output/hkls_out.csv', f'{result_dir}/output/hkls_err.csv', "HKL generation")

    logger.info(f"[{dataset_id}] Doing PeakSearch. Time till now: {time.time() - t0:.2f} seconds.")
    res = [peaks(result_dir, outFStem, num_procs, bin_directory, blockNr=nodeNr, numBlocks=n_nodes) for nodeNr in range(n_nodes)]
    [i.result() for i in res]
    
    logger.info(f"[{dataset_id}] Merging peaks...")
    cmd_merge = f"{os.path.join(bin_directory, 'MergeOverlappingPeaksAllZarr')} {outFStem}"
    safely_run_command(cmd_merge, result_dir, f'{result_dir}/output/merge_out.csv', f'{result_dir}/output/merge_err.csv', "Peak merging")
    
    logger.info(f"[{dataset_id}] Calculating Radii...")
    cmd_radius = f"{os.path.join(bin_directory, 'CalcRadiusAllZarr')} {outFStem}"
    safely_run_command(cmd_radius, result_dir, f'{result_dir}/output/radius_out.csv', f'{result_dir}/output/radius_err.csv', "Radius calculation")

    logger.info(f"[{dataset_id}] Transforming data...")
    cmd_setup = f"{os.path.join(bin_directory, 'FitSetupZarr')} {outFStem}"
    safely_run_command(cmd_setup, result_dir, f'{result_dir}/output/fit_setup_out.csv', f'{result_dir}/output/fit_setup_err.csv', "Data transformation")

    logger.info(f"[{dataset_id}] Binning data. Time till now: {time.time() - t0:.2f} seconds.")
    cmd_bin = f"{os.path.join(bin_directory, 'SaveBinData')}"
    safely_run_command(cmd_bin, result_dir, f'{result_dir}/output/binning_out.csv', f'{result_dir}/output/binning_err.csv', "Data binning")
    
    logger.info(f"[{dataset_id}] Finished pre-processing and binning. Total time: {time.time() - t0:.2f} seconds.")
    return result_dir

def process_mapped_dataset_from_indexing(
    result_dir: str, num_procs: int, n_nodes: int, bin_directory: str
):
    """
    Runs the final analysis steps (indexing, refinement, grain processing)
    on a mapped dataset.
    """
    t0 = time.time()
    logger.info(f"Starting final analysis in {result_dir}")

    with change_directory(result_dir):
        logger.info(f"Indexing. Time till now: {time.time() - t0:.2f} seconds.")
        res_index = [index(result_dir, num_procs, bin_directory, blockNr=nodeNr, numBlocks=n_nodes) for nodeNr in range(n_nodes)]
        [i.result() for i in res_index]

        logger.info(f"Refining. Time till now: {time.time() - t0:.2f} seconds.")
        res_refine = [refine(result_dir, num_procs, bin_directory, blockNr=nodeNr, numBlocks=n_nodes) for nodeNr in range(n_nodes)]
        [i.result() for i in res_refine]
        
        logger.info(f"Making grains list. Time till now: {time.time() - t0:.2f} seconds.")
        # NOTE: Using the non-Zarr version as the mapped data is not a single Zarr file
        cmd_grains = f"{os.path.join(bin_directory, 'ProcessGrains')} {result_dir}/paramstest.txt"
        safely_run_command(cmd_grains, result_dir, f'{result_dir}/output/grains_out.csv', f'{result_dir}/output/grains_err.csv', "Grain processing")

    logger.info(f"Done with mapped dataset. Total time for final steps: {time.time() - t0:.2f} seconds.")

def main():
    """Main function to process two datasets."""        
    global default_handler
    default_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    
    parser = MyParser(
        description='Far-field HEDM analysis for two datasets using MIDAS. V7.0.0, contact hsharma@anl.gov', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Arguments for the two-dataset workflow ---
    parser.add_argument('-resultFolder', type=str, required=True, help='Folder where you want to save all results.')
    parser.add_argument('-paramFN', type=str, required=True, help='Parameter file name, will be used for both datasets.')
    parser.add_argument('-dataFN', type=str, required=True, help='Data file name for the FIRST dataset (e.g., .h5 file).')
    parser.add_argument('-dataFN2', type=str, required=True, help='Data file name for the SECOND dataset (e.g., .h5 file).')
    parser.add_argument('-offsetX', type=float, required=True, default=0.0, help='Offset in X to map dataset 2 to dataset 1 (micrometers).')
    parser.add_argument('-offsetY', type=float, required=True, default=0.0, help='Offset in Y to map dataset 2 to dataset 1 (micrometers).')
    parser.add_argument('-offsetZ', type=float, required=True, default=0.0, help='Offset in Z to map dataset 2 to dataset 1 (micrometers).')
    parser.add_argument('-offsetOmega', type=float, required=True, default=0.0, help='Offset in Omega to map dataset 2 to dataset 1 (micrometers).')

    # --- Standard configuration arguments ---
    parser.add_argument('-nCPUs', type=int, default=10, help='Number of CPU cores to use per node.')
    parser.add_argument('-machineName', type=str, default='local', help='Machine name for execution: local, orthrosnew, orthrosall, umich, marquette, purdue.')
    parser.add_argument('-nNodes', type=int, default=1, help='Number of nodes for execution.')
    parser.add_argument('-numFrameChunks', type=int, default=-1, help='If low on RAM, process data in chunks. -1 to disable.')
    parser.add_argument('-preProcThresh', type=int, default=-1, help='Threshold for saving dark-corrected data. -1 to disable.')
    
    args = parser.parse_args()
    
    # --- Setup and Configuration ---
    top_res_dir = os.path.abspath(args.resultFolder)
    os.makedirs(top_res_dir, exist_ok=True)
    os.environ['MIDAS_SCRIPT_DIR'] = top_res_dir
    
    try:
        num_procs, n_nodes = load_machine_config(args.machineName, args.nNodes, args.nCPUs)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # --- Main Processing Workflow ---
    with cleanup_context():
        try:
            # Step 1: Process the first dataset up to binning
            result_dir1 = process_dataset_until_binning(
                dataset_id=1, top_res_dir=top_res_dir, ps_fn=args.paramFN, data_fn=args.dataFN,
                num_procs=num_procs, n_nodes=n_nodes, n_chunks=args.numFrameChunks,
                preproc=args.preProcThresh, bin_directory=bin_dir
            )
            
            # Step 2: Process the second dataset up to binning
            result_dir2 = process_dataset_until_binning(
                dataset_id=2, top_res_dir=top_res_dir, ps_fn=args.paramFN, data_fn=args.dataFN2,
                num_procs=num_procs, n_nodes=n_nodes, n_chunks=args.numFrameChunks,
                preproc=args.preProcThresh, bin_directory=bin_dir
            )

            # Step 3: Add the mapping information to the parameter file of the first dataset
            paramstest_path = os.path.join(result_dir1, "paramstest.txt")
            dataset2_line = f"Dataset2Folder {result_dir2} {args.offsetX} {args.offsetY} {args.offsetZ} {args.offsetOmega}\n"
            # read the minNrSpots line from paramFN
            minLine = "MinNrSpots 2\n"
            with open(args.paramFN, "r") as f:
                for line in f:
                    if line.startswith("MinNrSpots"):
                        minLine = line
                        break
            logger.info(f"Appending to {paramstest_path}: {dataset2_line.strip()}")
            with open(paramstest_path, "a") as f:
                f.write(dataset2_line)
                f.write(minLine)

            # Step 4: Map the two datasets together
            logger.info(f"Mapping datasets: {result_dir1} and {result_dir2}")
            map_cmd = f"{os.path.join(bin_dir, 'MapDatasets')} {result_dir1} {result_dir2} {args.offsetOmega} {num_procs}"
            safely_run_command(
                map_cmd, result_dir1,
                out_file=os.path.join(result_dir1, 'output', 'map_out.txt'),
                err_file=os.path.join(result_dir1, 'output', 'map_err.txt'),
                task_name="Dataset Mapping"
            )

            # Step 5: Run indexing and subsequent steps on the mapped data in the first folder
            process_mapped_dataset_from_indexing(
                result_dir=result_dir1, num_procs=num_procs, n_nodes=n_nodes, bin_directory=bin_dir
            )

        except Exception as e:
            logger.error(f"An error occurred during the dual dataset workflow: {e}")
            sys.exit(1)
            
    logger.info("Dual dataset processing completed successfully.")

if __name__ == "__main__":
    main()