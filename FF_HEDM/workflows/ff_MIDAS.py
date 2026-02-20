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
logger = logging.getLogger('MIDAS')
# Silence all Parsl loggers completely
logging.getLogger("parsl").setLevel(logging.CRITICAL)  # Only show critical errors
# Also silence these specific Parsl sub-loggers
for logger_name in ["parsl.dataflow.dflow", "parsl.dataflow.memoization", 
                    "parsl.process_loggers", "parsl.jobs.strategy",
                    "parsl.executors.threads"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Set paths dynamically using script location
@lru_cache(maxsize=1)
def get_installation_dir() -> str:
    """Get the installation directory from the script's location.
    Cached for performance.
    
    Returns:
        Installation directory path
    """
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
    """Context manager for changing directory.
    
    Args:
        new_dir: Directory to change to
        
    Yields:
        None
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
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
            # Try to change to home directory as fallback
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
        # Clean up Parsl
        try:
            parsl.dfk().cleanup()
        except Exception as e:
            logger.error(f"Failed to clean up Parsl: {e}")

def safely_run_command(cmd: str, working_dir: str, out_file: str, err_file: str, 
                       task_name: str = "Command") -> int:
    """Run a shell command with improved error handling.
    
    Args:
        cmd: Command to run
        working_dir: Directory to run command in
        out_file: Path to save stdout
        err_file: Path to save stderr
        task_name: Description of the task for error messages
        
    Returns:
        Return code from the command
    
    Raises:
        RuntimeError: If command fails with error details
    """
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
            
        # Check if command failed
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
    
    # Set MIDAS_INSTALL_DIR environment variable if not already set
    if 'MIDAS_INSTALL_DIR' not in env:
        env['MIDAS_INSTALL_DIR'] = get_installation_dir()
    
    return env

def read_parameter_file(psFN: str) -> Dict[str, str]:
    """Read parameters from file into a dictionary.
    
    Args:
        psFN: Parameter file name
        
    Returns:
        Dictionary of parameters
        
    Raises:
        FileNotFoundError: If parameter file doesn't exist
    """
    if not os.path.exists(psFN):
        raise FileNotFoundError(f"Parameter file not found: {psFN}")
        
    params = {}
    with open(psFN, 'r') as f:
        for line in f:
            # remove comments
            line = line.split('#', 1)[0]
            # strip whitespace
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ', 1)
            if len(parts) == 2:
                params[parts[0]] = parts[1].strip()
                
    return params

def update_parameter_file(psFN: str, updates: Dict[str, str]) -> None:
    """Update parameter file with new values. More robust implementation.
    
    Args:
        psFN: Parameter file name
        updates: Dictionary of parameter names and values to update
        
    Raises:
        FileNotFoundError: If parameter file doesn't exist
    """
    if not os.path.exists(psFN):
        raise FileNotFoundError(f"Parameter file not found: {psFN}")
        
    # Read current content
    with open(psFN, 'r') as f:
        lines = f.readlines()
        
    # Track parameters that have been updated
    updated_params = set()
    
    # Update existing parameters
    with open(psFN, 'w') as f:
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                f.write(line)
                continue
                
            parts = line_stripped.split(' ', 1)
            if len(parts) == 2 and parts[0] in updates:
                f.write(f"{parts[0]} {updates[parts[0]]}\n")
                updated_params.add(parts[0])
            else:
                f.write(line)
                
        # Add parameters that weren't in the file
        for param, value in updates.items():
            if param not in updated_params:
                f.write(f"{param} {value}\n")

def parse_float_param(param_dict: Dict[str, str], name: str, default: float = 0.0) -> float:
    """Parse a float parameter from the parameter dictionary.
    
    Args:
        param_dict: Parameter dictionary
        name: Parameter name
        default: Default value
        
    Returns:
        Parsed float value
    """
    if name not in param_dict:
        return default
        
    try:
        return float(param_dict[name])
    except ValueError:
        logger.warning(f"Invalid float value for {name}: {param_dict[name]}, using default {default}")
        return default

def parse_int_param(param_dict: Dict[str, str], name: str, default: int = 0) -> int:
    """Parse an integer parameter from the parameter dictionary.
    
    Args:
        param_dict: Parameter dictionary
        name: Parameter name
        default: Default value
        
    Returns:
        Parsed integer value
    """
    if name not in param_dict:
        return default
        
    try:
        return int(param_dict[name])
    except ValueError:
        logger.warning(f"Invalid integer value for {name}: {param_dict[name]}, using default {default}")
        return default

def validate_paths(paths: List[str]) -> bool:
    """Validate that all paths exist.
    
    Args:
        paths: List of paths to validate
        
    Returns:
        True if all paths exist, False otherwise
    """
    for path in paths:
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            return False
    return True

def validate_layer_range(start: int, end: int) -> bool:
    """Validate layer range.
    
    Args:
        start: Start layer number
        end: End layer number
        
    Returns:
        True if valid, False otherwise
    """
    if start < 1:
        logger.error(f"Start layer number must be >= 1, got {start}")
        return False
        
    if end < start:
        logger.error(f"End layer number must be >= start layer number, got {end} < {start}")
        return False
        
    return True

class ProgressTracker:
    """Track progress of a multi-step operation."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """Initialize the progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Description of the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, step: int = None, message: str = None):
        """Update progress.
        
        Args:
            step: Current step (increments by 1 if None)
            message: Additional message to log
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        elapsed = time.time() - self.start_time
        percent = (self.current_step / self.total_steps) * 100
        
        log_msg = f"{self.description}: {self.current_step}/{self.total_steps} ({percent:.1f}%), "
        log_msg += f"elapsed: {elapsed:.1f}s"
        
        if message:
            log_msg += f" - {message}"
            
        logger.info(log_msg)

def generateZip(
    resFol: str,
    pfn: str,
    layerNr: int,
    dfn: str = '',
    dloc: str = '',
    nchunks: int = -1,
    preproc: int = -1,
    outf: str = 'ZipOut.txt',
    errf: str = 'ZipErr.txt',
    numFilesPerScan: int = 1
) -> Optional[str]:
    """Generate ZIP file from data.
    
    Args:
        resFol: Result folder
        pfn: Parameter file name
        layerNr: Layer number
        dfn: Data file name
        dloc: Data location
        nchunks: Number of frame chunks
        preproc: Pre-processing threshold
        outf: Output file name
        errf: Error file name
        numFilesPerScan: Number of files per scan
    Returns:
        ZIP file name if successful, None otherwise
    """
    cmd = f"{pytpath} {os.path.join(utils_dir, 'ffGenerateZipRefactor.py')} -resultFolder {resFol} -paramFN {pfn} -LayerNr {layerNr}"
    
    if dfn:
        cmd += f' -dataFN {dfn}'
    if dloc:
        cmd += f' -dataLoc {dloc}'
    if nchunks != -1:
        cmd += f' -numFrameChunks {nchunks}'
    if preproc != -1:
        cmd += f' -preProcThresh {preproc}'
    if numFilesPerScan > 1:
        cmd += f' -numFilesPerScan {numFilesPerScan}'
        
    outf_path = f"{resFol}/midas_log/{outf}"
    errf_path = f"{resFol}/midas_log/{errf}"
    
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
    """Decorator to create a Parsl app with retry logic.
    
    Args:
        app_func: Function to decorate
        
    Returns:
        Decorated function
    """
    @python_app
    def wrapped_app(*args, **kwargs):
        import logging
        import time
        import os
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(f'MIDAS_{app_func.__name__}')
        
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                return app_func(*args, **kwargs, logger=logger)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    raise
    
    return wrapped_app

def _peaks_impl(resultDir: str, zipFN: str, numProcs: int, bin_dir: str, blockNr: int = 0, 
               numBlocks: int = 1, logger=None):
    """Implementation of peak search function.
    
    Args:
        resultDir: Result directory
        zipFN: ZIP file name
        numProcs: Number of processors
        bin_dir: Path to the bin directory
        blockNr: Block number
        numBlocks: Number of blocks
        logger: Logger instance
    """
    import subprocess
    import os
    import sys
    
    if logger is None:
        import logging
        logger = logging.getLogger('MIDAS_peaks')
    
    # Make sure output directory exists
    os.makedirs(f'{resultDir}/midas_log', exist_ok=True)
    
    outfile = f'{resultDir}/midas_log/peaksearch_out{blockNr}.csv'
    errfile = f'{resultDir}/midas_log/peaksearch_err{blockNr}.csv'
    
    # Copy all environment variables
    env = dict(os.environ)
    
    # If MIDAS_INSTALL_DIR is not set, try to get from script location
    if 'MIDAS_INSTALL_DIR' not in env:
        # Best effort to set MIDAS_INSTALL_DIR inside the app
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(script_dir):
            # Go up two levels to get to the installation directory
            install_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
            env['MIDAS_INSTALL_DIR'] = install_dir
    
    logger.info(f"Running PeaksFittingOMPZarrRefactor in {resultDir} for block {blockNr}/{numBlocks}")
    
    with open(outfile, 'w') as f, open(errfile, 'w') as f_err:
        cmd = f"{os.path.join(bin_dir, 'PeaksFittingOMPZarrRefactor')} {zipFN} {blockNr} {numBlocks} {numProcs}"
        logger.info(f"Executing command: {cmd}")
        
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            env=env, 
            stdout=f, 
            stderr=f_err, 
            cwd=resultDir
        )
        returncode = process.wait()
        
        if returncode != 0:
            f_err.flush()
            with open(errfile, 'r') as err_reader:
                error_content = err_reader.read()
            error_msg = f"Peak search failed with return code {returncode}. Error output:\n{error_content}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"PeaksFittingOMPZarrRefactor completed successfully for block {blockNr}/{numBlocks}")

# Create retry-capable app
peaks = create_app_with_retry(_peaks_impl)

def _index_impl(resultDir: str, numProcs: int, bin_dir: str, blockNr: int = 0, 
               numBlocks: int = 1, logger=None):
    """Implementation of indexing function.
    
    Args:
        resultDir: Result directory
        numProcs: Number of processors
        bin_dir: Path to the bin directory
        blockNr: Block number
        numBlocks: Number of blocks
        logger: Logger instance
    """
    import subprocess
    import os
    import sys
    
    if logger is None:
        import logging
        logger = logging.getLogger('MIDAS_index')
    
    # Ensure we're in the correct directory
    os.chdir(resultDir)
    
    # Make sure output directory exists
    os.makedirs(f'{resultDir}/midas_log', exist_ok=True)
    
    # Copy all environment variables
    env = dict(os.environ)
    
    # If MIDAS_INSTALL_DIR is not set, try to get from script location
    if 'MIDAS_INSTALL_DIR' not in env:
        # Best effort to set MIDAS_INSTALL_DIR inside the app
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(script_dir):
            # Go up two levels to get to the installation directory
            install_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
            env['MIDAS_INSTALL_DIR'] = install_dir
    
    # Count lines in SpotsToIndex.csv
    try:
        with open(os.path.join(resultDir, "SpotsToIndex.csv"), "r") as f:
            num_lines = len(f.readlines())
            logger.info(f"Found {num_lines} spots to index")
    except Exception as e:
        error_msg = f"Failed to read SpotsToIndex.csv: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    outfile = f'{resultDir}/midas_log/indexing_out{blockNr}.csv'
    errfile = f'{resultDir}/midas_log/indexing_err{blockNr}.csv'
    
    logger.info(f"Running IndexerOMP in {resultDir} for block {blockNr}/{numBlocks}")
    
    with open(outfile, 'w') as f, open(errfile, 'w') as f_err:
        cmd = f"{os.path.join(bin_dir, 'IndexerOMP')} paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}"
        logger.info(f"Executing command: {cmd}")
        
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            env=env, 
            stdout=f, 
            stderr=f_err, 
            cwd=resultDir
        )
        returncode = process.wait()
        
        if returncode != 0:
            f_err.flush()
            with open(errfile, 'r') as err_reader:
                error_content = err_reader.read()
            error_msg = f"Indexing failed with return code {returncode}. Error output:\n{error_content}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"IndexerOMP completed successfully for block {blockNr}/{numBlocks}")

# Create retry-capable app
index = create_app_with_retry(_index_impl)

def _refine_impl(resultDir: str, numProcs: int, bin_dir: str, blockNr: int = 0, 
                numBlocks: int = 1, logger=None):
    """Implementation of refinement function.
    
    Args:
        resultDir: Result directory
        numProcs: Number of processors
        bin_dir: Path to the bin directory
        blockNr: Block number
        numBlocks: Number of blocks
        logger: Logger instance
    """
    import subprocess
    import os
    import sys
    import resource
    
    if logger is None:
        import logging
        logger = logging.getLogger('MIDAS_refine')
    
    # Ensure we're in the correct directory
    os.chdir(resultDir)
    
    # Make sure output directory exists
    os.makedirs(f'{resultDir}/midas_log', exist_ok=True)
    
    # Copy all environment variables
    env = dict(os.environ)
    
    # If MIDAS_INSTALL_DIR is not set, try to get from script location
    if 'MIDAS_INSTALL_DIR' not in env:
        # Best effort to set MIDAS_INSTALL_DIR inside the app
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(script_dir):
            # Go up two levels to get to the installation directory
            install_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
            env['MIDAS_INSTALL_DIR'] = install_dir
    
    # Count lines in SpotsToIndex.csv
    try:
        with open(os.path.join(resultDir, "SpotsToIndex.csv"), "r") as f:
            num_lines = len(f.readlines())
            logger.info(f"Found {num_lines} spots to refine")
    except Exception as e:
        error_msg = f"Failed to read SpotsToIndex.csv: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    outfile = f'{resultDir}/midas_log/refining_out{blockNr}.csv'
    errfile = f'{resultDir}/midas_log/refining_err{blockNr}.csv'
    
    logger.info(f"Running FitPosOrStrainsOMP in {resultDir} for block {blockNr}/{numBlocks}")
    
    # Enable core dumps
    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    
    with open(outfile, 'w') as f, open(errfile, 'w') as f_err:
        cmd = f"{os.path.join(bin_dir, 'FitPosOrStrainsOMP')} paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}"
        logger.info(f"Executing command: {cmd}")
        
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            env=env, 
            stdout=f, 
            stderr=f_err, 
            cwd=resultDir
        )
        returncode = process.wait()
        
        if returncode != 0:
            f_err.flush()
            with open(errfile, 'r') as err_reader:
                error_content = err_reader.read()
            error_msg = f"Refinement failed with return code {returncode}. Error output:\n{error_content}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"FitPosOrStrainsOMP completed successfully for block {blockNr}/{numBlocks}")

# Create retry-capable app
refine = create_app_with_retry(_refine_impl)

# Signal handler for cleanup
default_handler = None

def handler(num, frame):
    """Handle Ctrl+C by cleaning up and exiting."""
    try:
        logger.info("Ctrl-C was pressed, cleaning up.")
        # Add parsl cleanup
        parsl.dfk().cleanup()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        return default_handler(num, frame)

class MyParser(argparse.ArgumentParser):
    """Custom argument parser with better error handling."""
    def error(self, message):
        """Print error message and exit."""
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)

def load_machine_config(machine_name: str, n_nodes: int, num_procs: int) -> Tuple[int, int]:
    """Load machine configuration and set up Parsl.
    
    Args:
        machine_name: Name of the machine to use
        n_nodes: Number of nodes to use
        
    Returns:
        Tuple of (num_processors, num_nodes)
        
    Raises:
        ValueError: If machine_name is unknown
    """
    if machine_name == 'local':
        import localConfig
        parsl.load(config=localConfig.localConfig)
        return num_procs, 1  # Default for local
    elif machine_name == 'orthrosnew':
        import orthrosAllConfig
        parsl.load(config=orthrosAllConfig.orthrosNewConfig)
        return 32, 11
    elif machine_name == 'orthrosall':
        import orthrosAllConfig
        parsl.load(config=orthrosAllConfig.orthrosAllConfig)
        return 64, 5
    elif machine_name == 'umich':
        os.environ['nNodes'] = str(n_nodes)
        import uMichConfig
        parsl.load(config=uMichConfig.uMichConfig)
        return 36, n_nodes
    elif machine_name == 'marquette':
        os.environ['nNodes'] = str(n_nodes)
        import marquetteConfig
        parsl.load(config=marquetteConfig.marquetteConfig)
        return 36, n_nodes
    elif machine_name == 'purdue':
        os.environ['nNodes'] = str(n_nodes)
        import purdueConfig
        parsl.load(config=purdueConfig.purdueConfig)
        return 128, n_nodes
    else:
        raise ValueError(f"Unknown machine name: {machine_name}")

def setup_output_directories(result_dir: str) -> None:
    """Set up output directories.
    
    Args:
        result_dir: Result directory path
    """
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/midas_log", exist_ok=True)
    os.makedirs(f"{result_dir}/Output", exist_ok=True)
    os.makedirs(f"{result_dir}/Temp", exist_ok=True)

def process_layer(layer_nr: int, top_res_dir: str, ps_fn: str, data_fn: str, num_procs: int, 
                 n_nodes: int, n_chunks: int, preproc: int, inp_file_name: str, 
                 provide_input_all: int, convert_files: int, do_peak_search: int, 
                 peak_search_only: int, bin_directory: str, grains_file: str = '') -> None:
    """Process a single layer.
    
    Args:
        layer_nr: Layer number
        top_res_dir: Top result directory
        ps_fn: Parameter file name
        data_fn: Data file name
        num_procs: Number of processors
        n_nodes: Number of nodes
        n_chunks: Number of frame chunks
        preproc: Pre-processing threshold
        inp_file_name: Input file name
        provide_input_all: Whether to provide input all
        convert_files: Whether to convert files
        do_peak_search: Whether to do peak search
        peak_search_only: Whether to do peak search only
        bin_directory: Directory containing binaries
        grains_file: Optional grains file
    """
    # Determine output directory name
    if len(inp_file_name) <= 1:
        output_dir_stem = f'LayerNr_{layer_nr}/'
    else:
        ext = '.' + '.'.join(inp_file_name.split('_')[-1].split('.')[1:])
        filestem = '_'.join(inp_file_name.split('_')[:-1])
        file_nr = int(inp_file_name.split('_')[-1].split('.')[0])
        layer_nr = file_nr
        padding = len(inp_file_name.split('_')[-1].split('.')[0])
        inp_fstm = inp_file_name.split('.')[0]
        output_dir_stem = f'analysis_{inp_fstm}'
        
        # If param file exists, update parameters for specific input file
        if os.path.exists(ps_fn):
            updates = {
                'Ext': ext,
                'FileStem': filestem,
                'StartFileNrFirstLayer': '1'
            }
            update_parameter_file(ps_fn, updates)

    result_dir = f'{top_res_dir}/{output_dir_stem}'
    logger.info(f"Processing Layer Nr: {layer_nr}, results will be saved in {result_dir}")
    
    # Set up directories and copy parameter file
    setup_output_directories(result_dir)
    if os.path.exists(ps_fn):
        shutil.copy2(ps_fn, result_dir)
        
        # Check for PanelShiftsFile and copy if exists
        try:
            params = read_parameter_file(ps_fn)
            if 'PanelShiftsFile' in params:
                 psShiftFile = params['PanelShiftsFile']
                 if os.path.exists(psShiftFile):
                     shutil.copy2(psShiftFile, result_dir)
                     logger.info(f"Copied PanelShiftsFile {psShiftFile} to {result_dir}")
                 else:
                     logger.warning(f"PanelShiftsFile specified {psShiftFile} but does not exist.")
        except Exception as e:
            logger.error(f"Failed to copy PanelShiftsFile: {e}")
    
    t0 = time.time()
    
    # Process based on input type
    outFStem = None
    if provide_input_all == 0:
        if convert_files == 1:
            params = read_parameter_file(ps_fn)
            NrFilesPerLayer = parse_int_param(params, 'NrFilesPerSweep',default=1)
            print("NrFilesPerLayer:", NrFilesPerLayer)
            if len(data_fn) > 0:
                logger.info("Generating combined MIDAS file from HDF and ps files.")
            else:
                logger.info("Generating combined MIDAS file from GE and ps files.")
            outFStem = generateZip(result_dir, ps_fn, layer_nr, dfn=data_fn, nchunks=n_chunks, preproc=preproc,numFilesPerScan=NrFilesPerLayer)
            if not outFStem:
                raise RuntimeError("Failed to generate ZIP file")
        else:
            if len(data_fn) > 0:
                outFStem = f'{result_dir}/{data_fn}'
                if not os.path.exists(outFStem):
                    shutil.copy2(data_fn, result_dir)
            else:
                # Extract file information from parameter file
                params = read_parameter_file(ps_fn)
                fStem = params.get('FileStem')
                startFN = parse_int_param(params, 'StartFileNrFirstLayer')
                NrFilesPerLayer = parse_int_param(params, 'NrFilesPerSweep',default=1)
                
                if not all([fStem, startFN is not None, NrFilesPerLayer is not None]):
                    raise ValueError("Missing required parameters in parameter file")
                    
                thisFileNr = startFN + (layer_nr - 1) * NrFilesPerLayer
                outFStem = f'{result_dir}/{fStem}_{str(thisFileNr).zfill(6)}.MIDAS.zip'
                
                if not os.path.exists(outFStem) and data_fn:
                    shutil.copy2(data_fn, result_dir)
                    
            # Update zarr dataset
            cmdUpd = f'{pytpath} {os.path.join(utils_dir, "updateZarrDset.py")} -fn {os.path.basename(outFStem)} -folder {result_dir} -keyToUpdate analysis/process/analysis_parameters/ResultFolder -updatedValue {result_dir}/'
            logger.info(cmdUpd)
            
            try:
                subprocess.check_call(cmdUpd, shell=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to update zarr dataset: {e}")
                
        logger.info(f"Generating HKLs. Time till now: {time.time() - t0} seconds.")
        
        try:
            f_hkls_out = f'{result_dir}/midas_log/hkls_out.csv'
            f_hkls_err = f'{result_dir}/midas_log/hkls_err.csv'
            cmd = f"{os.path.join(bin_directory, 'GetHKLListZarr')} {outFStem}"
            safely_run_command(cmd, result_dir, f_hkls_out, f_hkls_err, task_name="HKL generation")
        except Exception as e:
            raise RuntimeError(f"Failed to generate HKLs: {e}")
    else:
        # Handle InputAll case
        with change_directory(result_dir):
            logger.info(f"Generating HKLs. Time till now: {time.time() - t0} seconds.")
            
            try:
                f_hkls_out = f'{result_dir}/midas_log/hkls_out.csv'
                f_hkls_err = f'{result_dir}/midas_log/hkls_err.csv'
                cmd = f"{os.path.join(bin_directory, 'GetHKLList')} {ps_fn}"
                safely_run_command(cmd, result_dir, f_hkls_out, f_hkls_err, task_name="HKL generation")
            except Exception as e:
                raise RuntimeError(f"Failed to generate HKLs: {e}")
        
        # Handle InputAll data - this part from the original function
        process_inputall_data(result_dir, top_res_dir, ps_fn)
                    
    # Process peaks if required
    if provide_input_all == 0:
        if do_peak_search == 1:
            logger.info(f"Doing PeakSearch. Time till now: {time.time() - t0} seconds.")
            
            try:
                res = []
                for nodeNr in range(n_nodes):
                    res.append(peaks(result_dir, outFStem, num_procs, bin_directory, blockNr=nodeNr, numBlocks=n_nodes))
                outputs = [i.result() for i in res]
                logger.info(f"PeakSearch done. Time till now: {time.time() - t0}")
            except Exception as e:
                raise RuntimeError(f"Failed during peak search: {e}")
        else:
            logger.info("Peaksearch results were supplied. Skipping peak search.")
            
        if peak_search_only == 1:
            return
            
        logger.info("Merging peaks.")
        
        try:
            f_merge_out = f'{result_dir}/midas_log/merge_overlaps_out.csv'
            f_merge_err = f'{result_dir}/midas_log/merge_overlaps_err.csv'
            cmd = f"{os.path.join(bin_directory, 'MergeOverlappingPeaksAllZarr')} {outFStem}"
            safely_run_command(cmd, result_dir, f_merge_out, f_merge_err, task_name="Peak merging")
        except Exception as e:
            raise RuntimeError(f"Failed to merge peaks: {e}")
        
        logger.info(f"Calculating Radii. Time till now: {time.time() - t0}")
        
        try:
            f_radius_out = f'{result_dir}/midas_log/calc_radius_out.csv'
            f_radius_err = f'{result_dir}/midas_log/calc_radius_err.csv'
            cmd = f"{os.path.join(bin_directory, 'CalcRadiusAllZarr')} {outFStem}"
            safely_run_command(cmd, result_dir, f_radius_out, f_radius_err, task_name="Radius calculation")
        except Exception as e:
            raise RuntimeError(f"Failed to calculate radii: {e}")
        
        logger.info(f"Transforming data. Time till now: {time.time() - t0}")
        
        try:
            f_setup_out = f'{result_dir}/midas_log/fit_setup_out.csv'
            f_setup_err = f'{result_dir}/midas_log/fit_setup_err.csv'
            cmd = f"{os.path.join(bin_directory, 'FitSetupZarr')} {outFStem}"
            safely_run_command(cmd, result_dir, f_setup_out, f_setup_err, task_name="Data transformation")
        except Exception as e:
            raise RuntimeError(f"Failed to transform data: {e}")

    # Add grains file to parameters if needed
    with change_directory(result_dir):
        if grains_file:
            try:
                with open(f"{result_dir}/paramstest.txt", "a") as paramstestF:
                    paramstestF.write(f"GrainsFile {grains_file}\n")
            except Exception as e:
                raise RuntimeError(f"Failed to add grainsFile parameter to paramstest.txt: {e}")

        # Propagate RingsToExcludeFraction
        try:
            with open(ps_fn, 'r') as pf:
                for line in pf:
                    if line.strip().startswith('RingsToExcludeFraction'):
                         with open(f"{result_dir}/paramstest.txt", "a") as paramstestF:
                             paramstestF.write(line)
        except Exception as e:
            logger.error(f"Failed to propagate RingsToExcludeFraction: {e}")

        # Bin data
        logger.info(f"Binning data. Time till now: {time.time() - t0}, workingdir: {result_dir}")
        try:
            f_bin_out = f'{result_dir}/midas_log/binning_out.csv'
            f_bin_err = f'{result_dir}/midas_log/binning_err.csv'
            cmd = f"{os.path.join(bin_directory, 'SaveBinData')}"
            safely_run_command(cmd, result_dir, f_bin_out, f_bin_err, task_name="Data binning")
        except Exception as e:
            raise RuntimeError(f"Failed to bin data: {e}")
            
        # Run indexing
        logger.info(f"Indexing. Time till now: {time.time() - t0}")
        
        try:
            res_index = []
            for nodeNr in range(n_nodes):
                res_index.append(index(result_dir, num_procs, bin_directory, blockNr=nodeNr, numBlocks=n_nodes))
            output_index = [i.result() for i in res_index]
        except Exception as e:
            raise RuntimeError(f"Failed during indexing: {e}")
            
        # Run refinement
        logger.info(f"Refining. Time till now: {time.time() - t0}")
        
        try:
            res_refine = []
            for nodeNr in range(n_nodes):
                res_refine.append(refine(result_dir, num_procs, bin_directory, blockNr=nodeNr, numBlocks=n_nodes))
            output_refine = [i.result() for i in res_refine]
        except Exception as e:
            raise RuntimeError(f"Failed during refinement: {e}")
                        
        # Process grains
        logger.info(f"Making grains list. Time till now: {time.time() - t0}")
        
        try:
            f_grains_out = f'{result_dir}/midas_log/process_grains_out.csv'
            f_grains_err = f'{result_dir}/midas_log/process_grains_err.csv'
            
            if provide_input_all == 0:
                if grains_file:
                    cmd = f"{os.path.join(bin_directory, 'ProcessGrainsZarr')} {outFStem} 1"
                else:
                    cmd = f"{os.path.join(bin_directory, 'ProcessGrainsZarr')} {outFStem}"
            else:
                cmd = f"{os.path.join(bin_directory, 'ProcessGrains')} {result_dir}/paramstest.txt"
                
            safely_run_command(cmd, result_dir, f_grains_out, f_grains_err, task_name="Grain processing")
        except Exception as e:
            raise RuntimeError(f"Failed to process grains: {e}")
        get_grains_info(result_dir)
            
        logger.info(f"Done Layer {layer_nr}. Total time elapsed: {time.time() - t0}")

def get_grains_info(result_dir: str) -> None:
    """Read and print the first line of the Grains.csv file.
    
    Args:
        result_dir: Result directory containing the Grains.csv file
    """
    grains_file = os.path.join(result_dir, "Grains.csv")
    
    logger.info(f"Attempting to read Grains.csv from {result_dir}")
    
    if not os.path.exists(grains_file):
        logger.warning(f"No Grains.csv file could be read from {result_dir}")
        print(f"No Grains.csv file could be read from {result_dir}")
        return
    
    try:
        with open(grains_file, 'r') as f:
            first_line = f.readline().strip()
            
        if first_line:
            logger.info(f"First line of Grains.csv: {first_line}")
            print(f"First line of Grains.csv: {first_line}")
        else:
            logger.warning(f"Grains.csv exists but is empty in {result_dir}")
            print(f"Grains.csv exists but is empty in {result_dir}")
    except Exception as e:
        logger.error(f"Error reading Grains.csv: {e}")
        print(f"Error reading Grains.csv: {e}")

def process_inputall_data(result_dir: str, top_res_dir: str, ps_fn: str) -> None:
    """Process InputAll data.
    
    Args:
        result_dir: Result directory
        top_res_dir: Top level result directory
        ps_fn: Parameter file name
    """
    try:
        os.chdir(result_dir)
        shutil.copy2(f'{top_res_dir}/InputAllExtraInfoFittingAll.csv', f'{result_dir}/InputAll.csv')
        shutil.copy2(f'{top_res_dir}/InputAllExtraInfoFittingAll.csv', f'{result_dir}/.')
        
        # Read parameter file to get rings to index
        params = read_parameter_file(ps_fn)
        ring2Index = parse_float_param(params, 'OverAllRingToIndex')
        min2Index = parse_float_param(params, 'MinOmeSpotIDsToIndex')
        max2Index = parse_float_param(params, 'MaxOmeSpotIDsToIndex')
        
        # Process spots
        sps = np.genfromtxt(f'{result_dir}/InputAll.csv', skip_header=1)
        sps_filt = sps[sps[:,5] == ring2Index,:]
        
        if len(sps_filt.shape) < 2:
            raise ValueError("No IDs for indexing due to no spots for ring2index")
            
        sps_filt2 = sps_filt[sps_filt[:,2] >= min2Index,:]
        
        if len(sps_filt2.shape) < 2:
            raise ValueError("No IDs for indexing due to no spots above minOmeSpotsToIndex")
            
        sps_filt3 = sps_filt2[sps_filt2[:,2] <= max2Index,:]
        
        if len(sps_filt3.shape) < 2:
            raise ValueError("No IDs for indexing due to no spots below maxOmeSpotsToIndex")
            
        IDs = sps_filt3[:,4].astype(np.int32)
        np.savetxt(f'{result_dir}/SpotsToIndex.csv', IDs, fmt="%d")
        
        # Copy and update paramstest.txt
        shutil.copy2(f'{top_res_dir}/{ps_fn}', f'{result_dir}/paramstest.txt')                    
        ringNrs = []
        with open(f'{result_dir}/paramstest.txt', 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('RingThresh '):
                ringNrs.append(int(line.split(' ')[1]))
        
        # What we need extra: RingRadii and RingNumbers, first read hkls.csv
        ringRads = np.zeros((len(ringNrs),2))
        hkls = np.genfromtxt(f'{result_dir}/hkls.csv',skip_header=1)
        unq, locs = np.unique(hkls[:,4],return_index=True)
        for rN in range(len(ringNrs)):
            ringNr = ringNrs[rN]
            for tp in range(len(unq)):
                if ringNr == int(unq[tp]):
                    ringRads[rN] = np.array([ringNr,hkls[locs[tp],-1]])
        
        # Write updated parameter file
        with open(f'{result_dir}/paramstest.txt', 'w') as paramstestF:
            for nr in range(len(ringRads)):
                paramstestF.write(f'RingRadii {ringRads[nr,1]}\n')
                paramstestF.write(f'RingNumbers {int(ringRads[nr,0])}\n')
            paramstestF.write(f'OutputFolder {result_dir}/Output\n')
            paramstestF.write(f'ResultFolder {result_dir}/Results\n')
            paramstestF.write('SpotsFileName InputAll.csv\n')
            paramstestF.write('IDsFileName SpotsToIndex.csv\n')
            paramstestF.write('RefinementFileName InputAllExtraInfoFittingAll.csv\n')
            for line in lines:
                paramstestF.write(line)
                    
        os.makedirs(f'{result_dir}/midas_log', exist_ok=True)
        os.makedirs(f'{result_dir}/Output', exist_ok=True)
        os.makedirs(f'{result_dir}/Results', exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to process InputAll data: {e}")

def main():
    """Main function to process data."""        
    # Set up signal handler
    global default_handler
    default_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    
    # Set up argument parser
    parser = MyParser(
        description='Far-field HEDM analysis using MIDAS. Contact: hsharma@anl.gov', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('-resultFolder', type=str, required=False, default='', 
                        help='Folder where you want to save results. If nothing is provided, it will default to the current folder.')
    parser.add_argument('-paramFN', type=str, required=False, default='', 
                        help='Parameter file name. Provide either paramFN and/or dataFN.')
    parser.add_argument('-dataFN', type=str, required=False, default='', 
                        help='Data file name. This is if you have either h5 or zip files. Provide either paramFN and/or dataFN (in case zip exists).')
    parser.add_argument('-nCPUs', type=int, required=False, default=10, 
                        help='Number of CPU cores to use if running locally.')
    parser.add_argument('-machineName', type=str, required=False, default='local', 
                        help='Machine name for execution, local, orthrosnew, orthrosall, umich, marquette, purdue.')
    parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, 
                        help='If low on RAM, it can process parts of the dataset at the time. -1 will disable.')
    parser.add_argument('-preProcThresh', type=int, required=False, default=-1, 
                        help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
    parser.add_argument('-nNodes', type=int, required=False, default=-1, 
                        help='Number of nodes for execution, omit if want to automatically select.')
    parser.add_argument('-fileName', type=str, required=False, default='', 
                        help='If you specify a fileName, this will run just that file. If you provide this, it will override startLayerNr and endLayerNr')
    parser.add_argument('-startLayerNr', type=int, required=False, default=1, 
                        help='Start LayerNr to process')
    parser.add_argument('-endLayerNr', type=int, required=False, default=1, 
                        help='End LayerNr to process')
    parser.add_argument('-convertFiles', type=int, required=False, default=1, 
                        help='If want to convert to zarr, if zarr files exist already, put to 0.')
    parser.add_argument('-peakSearchOnly', type=int, required=False, default=0, 
                        help='If want to do peakSearchOnly, nothing more, put to 1.')
    parser.add_argument('-doPeakSearch', type=int, required=False, default=1, 
                        help="If don't want to do peakSearch, put to 0.")
    parser.add_argument('-provideInputAll', type=int, required=False, default=0, 
                        help="If want to provide InputAllExtraInfoFittingAll.csv, put to 1. MUST provide all the parameters in the paramFN. The resultFolder must exist and contain the InputAlExtraInfoFittingAll.csv")
    parser.add_argument('-rawDir', type=str, required=False, default='', 
                        help='If want override the rawDir in the Parameter file.')
    parser.add_argument('-grainsFile', type=str, required=False, default='', 
                        help='Optional input file containing seed grains to use for grain finding. If not provided, grains will be determined from scratch.')
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        logger.error("MUST PROVIDE EITHER paramFN or dataFN")
        sys.exit(1)
        
    args, unparsed = parser.parse_known_args()
    
    # Set variables from arguments
    result_dir = args.resultFolder
    ps_fn = args.paramFN
    data_fn = args.dataFN
    num_procs = args.nCPUs
    machine_name = args.machineName
    n_nodes = args.nNodes
    n_chunks = args.numFrameChunks
    preproc = args.preProcThresh
    start_layer_nr = args.startLayerNr
    end_layer_nr = args.endLayerNr
    convert_files = args.convertFiles
    peak_search_only = args.peakSearchOnly
    do_peak_search = args.doPeakSearch
    raw_dir = args.rawDir
    inp_file_name = args.fileName
    provide_input_all = args.provideInputAll
    grains_file = args.grainsFile
    
    # Basic validation
    if not validate_layer_range(start_layer_nr, end_layer_nr):
        sys.exit(1)
        
    # Handle input file name
    if len(inp_file_name) > 1 and len(data_fn) < 1 and '.h5' in inp_file_name:
        data_fn = inp_file_name
        
    # Set number of nodes
    if n_nodes == -1:
        n_nodes = 1
        
    # Set defaults if neither paramFN nor dataFN is provided
    if not ps_fn and not data_fn:
        logger.error("Either paramFN or dataFN must be provided")
        sys.exit(1)
        
    # Update raw directory if provided
    if len(raw_dir) > 1 and ps_fn:
        try:
            # Read parameter file
            params = read_parameter_file(ps_fn)
            
            # Extract required parameters
            ring2Index = parse_float_param(params, 'OverAllRingToIndex')
            min2Index = parse_float_param(params, 'MinOmeSpotIDsToIndex')
            max2Index = parse_float_param(params, 'MaxOmeSpotIDsToIndex')
                    
            # Update RawFolder and Dark
            updates = {'RawFolder': raw_dir}
            
            # Find the dark file name and update its path
            if 'Dark' in params:
                dark_name = params['Dark'].split('/')[-1]
                updates['Dark'] = f'{raw_dir}/{dark_name}'
                    
            update_parameter_file(ps_fn, updates)
            
        except Exception as e:
            logger.error(f"Failed to update raw directory: {e}")
            sys.exit(1)
    
    # Handle grains file
    if grains_file and ps_fn and os.path.exists(ps_fn):
        try:
            params = read_parameter_file(ps_fn)
            
            # Update MinNrSpots to 1 if grains_file is provided
            updates = {'MinNrSpots': '1'}
            update_parameter_file(ps_fn, updates)
        except Exception as e:
            logger.error(f"Failed to update parameters for grains file: {e}")
            sys.exit(1)
    
    # Set up environment
    env = get_midas_env()
    
    # Set up result directory
    if len(result_dir) == 0 or result_dir == '.':
        result_dir = os.getcwd()
    if result_dir[0] == '~':
        result_dir = os.path.expanduser(result_dir)
    if result_dir[0] != '/':
        result_dir = os.getcwd() + '/' + result_dir
        
    os.makedirs(result_dir, exist_ok=True)
    os.environ['MIDAS_SCRIPT_DIR'] = result_dir
    
    # Load configuration based on machine name
    try:
        num_procs, n_nodes = load_machine_config(machine_name, n_nodes, num_procs)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Run for each layer
    orig_dir = os.getcwd()
    top_res_dir = result_dir
    
    # Use cleanup context manager
    with cleanup_context():
        progress = ProgressTracker(end_layer_nr - start_layer_nr + 1, "Layer processing")
        
        for layer_nr in range(start_layer_nr, end_layer_nr + 1):
            try:
                process_layer(
                    layer_nr=layer_nr, 
                    top_res_dir=top_res_dir, 
                    ps_fn=ps_fn, 
                    data_fn=data_fn, 
                    num_procs=num_procs, 
                    n_nodes=n_nodes, 
                    n_chunks=n_chunks, 
                    preproc=preproc, 
                    inp_file_name=inp_file_name, 
                    provide_input_all=provide_input_all, 
                    convert_files=convert_files, 
                    do_peak_search=do_peak_search, 
                    peak_search_only=peak_search_only,
                    bin_directory=bin_dir,
                    grains_file=grains_file
                )
                
                progress.update(message=f"Layer {layer_nr} completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to process layer {layer_nr}: {e}")
                sys.exit(1)
            finally:
                # Return to original directory after each layer
                os.chdir(orig_dir)
    
    logger.info("All layers processed successfully")


if __name__ == "__main__":
    main()