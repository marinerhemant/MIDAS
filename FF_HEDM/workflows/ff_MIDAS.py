#!/usr/bin/env python

import parsl
import subprocess
import sys
import os
import time
import argparse
import signal
import shutil
import glob
import re
import logging
import numpy as np
import h5py
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


def discover_layer_files(raw_folder: str, ext: str, padding: int,
                         start_file_nr: int, end_file_nr: int) -> list:
    """Scan raw_folder for files with numbers in [start_file_nr, end_file_nr].

    Expected filename pattern: {stem}_{zero-padded-number}{ext}
    The last ``padding`` digits before the extension are treated as the
    file number.  Files whose basename (without extension) starts with
    ``dark_`` are skipped.

    Args:
        raw_folder: Directory containing raw data files.
        ext: File extension including leading dot (e.g. '.ge3', '.tif').
        padding: Number of digits used for zero-padding.
        start_file_nr: First file number to include.
        end_file_nr: Last file number to include.

    Returns:
        Sorted list of ``(file_nr, filestem)`` tuples where *filestem*
        is everything before ``_NNNNNN`` (the number segment).
    """
    import re
    if not os.path.isdir(raw_folder):
        logger.error(f"RawFolder does not exist: {raw_folder}")
        return []

    # Build regex: stem is captured as group 1, number is group 2
    # Example with padding=6: (.+)_(\d{6})\.ge3$
    escaped_ext = re.escape(ext)
    pattern = re.compile(rf'^(.+?)_(\d{{{padding}}}){escaped_ext}$')

    found = []
    n_darks = 0
    all_files = os.listdir(raw_folder)
    for fname in all_files:
        m = pattern.match(fname)
        if not m:
            continue
        stem = m.group(1)
        file_nr = int(m.group(2))

        # Skip files outside the requested range
        if file_nr < start_file_nr or file_nr > end_file_nr:
            continue

        # Skip dark files
        if stem.lower().startswith('dark_'):
            n_darks += 1
            continue

        found.append((file_nr, stem))

    found.sort(key=lambda x: x[0])

    n_missing = (end_file_nr - start_file_nr + 1) - len(found) - n_darks
    logger.info(
        f"Batch discovery in {raw_folder}: "
        f"{len(found)} data files, {n_darks} darks skipped, "
        f"{max(0, n_missing)} file numbers missing"
    )
    if found:
        logger.info(f"  File numbers: {found[0][0]} .. {found[-1][0]}")
        # Show unique stems
        stems = sorted(set(s for _, s in found))
        if len(stems) <= 5:
            logger.info(f"  Unique stems: {stems}")
        else:
            logger.info(f"  Unique stems: {len(stems)} different stems")

    return found

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
        logger.info(f"Running: {cmd}")
        # Stream output: stdout to file, stderr to terminal (for tqdm progress bar)
        with open(outf_path, 'w') as f_out:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=None,  # inherit terminal for tqdm progress bar
                cwd=resFol,
                env=get_midas_env(),
                bufsize=1,
                universal_newlines=True
            )
            for line in process.stdout:
                f_out.write(line)
                if 'done:' in line or 'OutputZipName' in line or 'Processing' in line:
                    logger.info(f"[ZIP] {line.rstrip()}")
            process.wait()

        if process.returncode != 0:
            error_msg = f"ZIP generation failed with return code {process.returncode}:\n{cmd}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        with open(outf_path, 'r') as f:
            lines = f.readlines()
            
        if lines and lines[-1].startswith('OutputZipName'):
            return lines[-1].split()[1]
        else:
            logger.error("Could not find OutputZipName in the output")
            return None
    except Exception as e:
        if not isinstance(e, RuntimeError):
            logger.error(f"Failed to generate ZIP: {e}")
            raise RuntimeError(f"ZIP generation failed: {str(e)}")
        raise

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
            # print("NrFilesPerLayer:", NrFilesPerLayer)
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
        
        # Generate consolidated HDF5 output with full provenance
        try:
            generate_consolidated_hdf5(result_dir, outFStem)
        except Exception as e:
            logger.warning(f"Failed to generate consolidated HDF5: {e}")
            
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


def generate_consolidated_hdf5(result_dir: str, zarr_path: str) -> None:
    """Generate a consolidated HDF5 file with full grain-to-peak provenance.
    
    Creates {stem}_consolidated.h5 with:
      - /parameters/: all analysis + scan parameters
      - /all_spots/: full InputAllExtraInfoFittingAll table
      - /radius_data/: full Radius CSV (21 cols)
      - /merge_map/: full MergeMap.csv (all 26 _PS.csv cols per constituent peak)
      - /grains/grain_NNNN/spots/spot_MMMM/constituent_peaks/: per-spot raw peaks
      - /raw_data_ref/: path to zarr.zip
    
    Also appends provenance data to the zarr.zip under analysis/provenance/.
    
    Args:
        result_dir: Result directory containing Grains.csv, SpotMatrix.csv, etc.
        zarr_path: Path to the zarr.zip file
    """
    import zarr
    from collections import defaultdict
    
    stem = os.path.splitext(os.path.basename(zarr_path))[0] if zarr_path else 'output'
    h5_path = os.path.join(result_dir, f'{stem}_consolidated.h5')
    
    # ---------- Read all input files ----------
    grains_file = os.path.join(result_dir, 'Grains.csv')
    spotmatrix_file = os.path.join(result_dir, 'SpotMatrix.csv')
    merge_map_file = os.path.join(result_dir, 'MergeMap.csv')
    extra_info_file = os.path.join(result_dir, 'InputAllExtraInfoFittingAll.csv')
    params_file = os.path.join(result_dir, 'paramstest.txt')
    
    # Find Radius file
    radius_file = None
    for f in os.listdir(result_dir):
        if f.startswith('Radius_StartNr_') and f.endswith('.csv'):
            radius_file = os.path.join(result_dir, f)
            break
    
    for required in [grains_file, spotmatrix_file]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Required file not found: {required}")
    
    # Parse Grains.csv (skip 9-line header)
    grains_header_lines = 9
    with open(grains_file, 'r') as f:
        header_info = [f.readline().strip() for _ in range(grains_header_lines)]
    grains_data = np.genfromtxt(grains_file, skip_header=grains_header_lines)
    if grains_data.ndim == 1:
        grains_data = grains_data.reshape(1, -1)
    grains_cols = [
        'GrainID', 'O11', 'O12', 'O13', 'O21', 'O22', 'O23', 'O31', 'O32', 'O33',
        'X', 'Y', 'Z', 'a', 'b', 'c', 'alpha', 'beta', 'gamma',
        'eFab11', 'eFab12', 'eFab13', 'eFab21', 'eFab22', 'eFab23',
        'eFab31', 'eFab32', 'eFab33',
        'eKen11', 'eKen12', 'eKen13', 'eKen21', 'eKen22', 'eKen23',
        'eKen31', 'eKen32', 'eKen33',
        'RMSErrorStrain', 'Confidence', 'Reserved1', 'Reserved2',
        'PhaseNr', 'Radius', 'Eul0', 'Eul1', 'Eul2', 'Reserved3', 'Reserved4'
    ]
    
    # Parse SpotMatrix.csv (skip 1-line header)
    spot_data = np.genfromtxt(spotmatrix_file, skip_header=1)
    if spot_data.ndim == 1:
        spot_data = spot_data.reshape(1, -1)
    spot_cols = ['GrainID', 'SpotID', 'Omega', 'DetY', 'DetZ', 'OmeRaw',
                 'Eta', 'RingNr', 'YLab', 'ZLab', 'Theta', 'StrainError']
    
    # Parse MergeMap.csv (tab-separated, skip header) — now 3 columns: MergedSpotID, FrameNr, PeakID
    merge_map_data = None
    if os.path.exists(merge_map_file):
        merge_map_data = np.genfromtxt(merge_map_file, skip_header=1, delimiter='\t', dtype=int)
        if merge_map_data.ndim == 1:
            merge_map_data = merge_map_data.reshape(1, -1)
    
    merge_map_cols = ['MergedSpotID', 'FrameNr', 'PeakID']
    
    ps_cols = [
        'SpotID', 'IntegratedIntensity', 'Omega', 'YCen', 'ZCen', 'IMax',
        'Radius', 'Eta', 'SigmaR', 'SigmaEta', 'NrPixels',
        'TotalNrPixelsInPeakRegion', 'nPeaks', 'maxY', 'maxZ', 'diffY', 'diffZ',
        'rawIMax', 'returnCode', 'retVal', 'BG', 'SigmaGR', 'SigmaLR',
        'SigmaGEta', 'SigmaLEta', 'MU'
    ]
    
    # Parse InputAllExtraInfoFittingAll.csv
    extra_info_data = None
    if os.path.exists(extra_info_file):
        extra_info_data = np.genfromtxt(extra_info_file, skip_header=1)
        if extra_info_data.ndim == 1:
            extra_info_data = extra_info_data.reshape(1, -1)
    extra_info_cols = [
        'YLab', 'ZLab', 'Omega', 'GrainRadius', 'SpotID', 'RingNumber',
        'Eta', 'Ttheta', 'OmegaIni', 'YOrig', 'ZOrig',
        'YOrigDetCor', 'ZOrigDetCor', 'OmegaOrigDetCor', 'IntegratedIntensity'
    ]
    
    # Parse Radius CSV
    radius_data_arr = None
    radius_cols = [
        'SpotID', 'IntegratedIntensity', 'Omega', 'YCen', 'ZCen', 'IMax',
        'MinOme', 'MaxOme', 'Radius', 'Theta', 'Eta', 'DeltaOmega', 'NImgs',
        'RingNr', 'GrainVolume', 'GrainRadius', 'PowderIntensity',
        'SigmaR', 'SigmaEta', 'NrPx', 'NrPxTot'
    ]
    if radius_file and os.path.exists(radius_file):
        radius_data_arr = np.genfromtxt(radius_file, skip_header=1)
        if radius_data_arr.ndim == 1:
            radius_data_arr = radius_data_arr.reshape(1, -1)
    
    # Parse paramstest.txt
    params_dict = {}
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            for line in f:
                line = line.strip().rstrip(';')
                if line and not line.startswith('#') and not line.startswith('%'):
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        key = parts[0]
                        if key in params_dict:
                            if isinstance(params_dict[key], list):
                                params_dict[key].append(parts[1])
                            else:
                                params_dict[key] = [params_dict[key], parts[1]]
                        else:
                            params_dict[key] = parts[1]
    
    # Parse hkls.csv
    hkls_file = os.path.join(result_dir, 'hkls.csv')
    hkls_data = None
    hkls_cols = ['h', 'k', 'l', 'D-spacing', 'RingNr', 'g1', 'g2', 'g3', 'Theta', '2Theta', 'Radius']
    if os.path.exists(hkls_file):
        hkls_data = np.genfromtxt(hkls_file, skip_header=1)
        if hkls_data.ndim == 1:
            hkls_data = hkls_data.reshape(1, -1)
    
    # Parse Result_StartNr_*.csv (PeaksFitting summary)
    result_file = None
    for f in os.listdir(result_dir):
        if f.startswith('Result_StartNr_') and f.endswith('.csv'):
            result_file = os.path.join(result_dir, f)
            break
    result_data = None
    result_cols = [
        'SpotID', 'IntegratedIntensity', 'Omega', 'YCen', 'ZCen', 'IMax',
        'MinOme', 'MaxOme', 'SigmaR', 'SigmaEta', 'NrPx', 'NrPxTot', 'Radius', 'Eta'
    ]
    if result_file and os.path.exists(result_file):
        result_data = np.genfromtxt(result_file, skip_header=1)
        if result_data.ndim == 1:
            result_data = result_data.reshape(1, -1)
    
    # Parse IDRings.csv
    idrings_file = os.path.join(result_dir, 'IDRings.csv')
    idrings_data = None
    idrings_cols = ['RingNumber', 'OriginalID', 'NewID']
    if os.path.exists(idrings_file):
        idrings_data = np.genfromtxt(idrings_file, skip_header=1)
        if idrings_data.ndim == 1:
            idrings_data = idrings_data.reshape(1, -1)
    
    # Parse IDsHash.csv (no header, 4 columns: RingNr StartID EndID D-spacing)
    idshash_file = os.path.join(result_dir, 'IDsHash.csv')
    idshash_data = None
    idshash_cols = ['RingNr', 'StartID', 'EndID', 'D-spacing']
    if os.path.exists(idshash_file):
        idshash_data = np.genfromtxt(idshash_file)
        if idshash_data.ndim == 1:
            idshash_data = idshash_data.reshape(1, -1)
    
    # Parse SpotsToIndex.csv (single column, no header)
    spots2index_file = os.path.join(result_dir, 'SpotsToIndex.csv')
    spots2index_data = None
    if os.path.exists(spots2index_file):
        spots2index_data = np.genfromtxt(spots2index_file, dtype=int)
        if spots2index_data.ndim == 0:
            spots2index_data = spots2index_data.reshape(1)
    
    # Parse GrainIDsKey.csv (variable-width rows, padded array)
    grainidskey_file = os.path.join(result_dir, 'GrainIDsKey.csv')
    grainidskey_data = None
    if os.path.exists(grainidskey_file):
        raw_rows = []
        with open(grainidskey_file, 'r') as f:
            for line in f:
                vals = [int(x) for x in line.strip().split() if x]
                if vals:
                    raw_rows.append(vals)
        if raw_rows:
            max_len = max(len(r) for r in raw_rows)
            grainidskey_data = np.full((len(raw_rows), max_len), -1, dtype=int)
            for ri, row in enumerate(raw_rows):
                grainidskey_data[ri, :len(row)] = row
    
    # Build merge map index: merged_spot_id -> list of (FrameNr, PeakID) tuples
    merge_idx = defaultdict(list)
    if merge_map_data is not None:
        for row_i in range(merge_map_data.shape[0]):
            merged_id = int(merge_map_data[row_i, 0])
            frame_nr = int(merge_map_data[row_i, 1])
            peak_id = int(merge_map_data[row_i, 2])
            merge_idx[merged_id].append((frame_nr, peak_id))
    
    # Load ALL _PS.csv files from Temp/ directory
    # Build cache: (frame_nr, peak_id) -> full 26-column peak row
    # Also collect all rows for the flat /peaks/per_frame/ dataset
    temp_dir = os.path.join(result_dir, 'Temp')
    # _PS.csv files use the full zarr basename (incl. .zip) as their stem
    # e.g., 'Au_FF_000001_pf.analysis.MIDAS.zip_000004_PS.csv'
    file_stem = os.path.basename(zarr_path) if zarr_path else ''
    ps_cache = {}  # (frame_nr, peak_id) -> np.array of 26 values
    all_ps_rows = []  # all rows across all frames, with frame_nr prepended
    if os.path.isdir(temp_dir):
        import re
        ps_files = sorted(f for f in os.listdir(temp_dir) if f.endswith('_PS.csv'))
        logger.info(f"Loading _PS.csv data for {len(ps_files)} frames from {temp_dir}")
        for ps_file in ps_files:
            # Extract frame number from filename: *_NNNNNN_PS.csv
            match = re.search(r'_(\d{6})_PS\.csv$', ps_file)
            if not match:
                continue
            frame_nr = int(match.group(1))
            ps_path = os.path.join(temp_dir, ps_file)
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    ps_data = np.genfromtxt(ps_path, skip_header=1)
                if ps_data.size == 0:
                    continue
                if ps_data.ndim == 1:
                    ps_data = ps_data.reshape(1, -1)
                for row in ps_data:
                    peak_id = int(row[0])
                    ps_cache[(frame_nr, peak_id)] = row
                    # Prepend frame_nr to each row for the flat table
                    all_ps_rows.append(np.concatenate([[frame_nr], row]))
            except Exception as e:
                logger.warning(f"Error reading {ps_path}: {e}")
        logger.info(f"Loaded {len(ps_cache)} peak entries from _PS.csv files")
    
    # Build radius index: spot_id -> row index
    radius_idx = {}
    if radius_data_arr is not None:
        for row_i in range(radius_data_arr.shape[0]):
            radius_idx[int(radius_data_arr[row_i, 0])] = row_i
    
    # ---------- Write HDF5 ----------
    logger.info(f"Writing consolidated HDF5 to {h5_path}")
    with h5py.File(h5_path, 'w') as h5:
        # /parameters/
        pg = h5.create_group('parameters')
        for key, val in params_dict.items():
            try:
                if isinstance(val, list):
                    # Multi-valued params (e.g. RingNumbers, OmegaRange)
                    float_vals = []
                    for v in val:
                        float_vals.extend([float(x) for x in v.split()])
                    pg.create_dataset(key, data=np.array(float_vals))
                else:
                    parts = val.split()
                    if len(parts) == 1:
                        try:
                            pg.create_dataset(key, data=float(val))
                        except ValueError:
                            pg.create_dataset(key, data=val)
                    else:
                        try:
                            pg.create_dataset(key, data=np.array([float(x) for x in parts]))
                        except ValueError:
                            pg.create_dataset(key, data=val)
            except Exception:
                pg.attrs[key] = str(val)
        
        # /all_spots/
        if extra_info_data is not None:
            sg = h5.create_group('all_spots')
            sg.create_dataset('data', data=extra_info_data)
            sg.attrs['column_names'] = extra_info_cols
        
        # /radius_data/
        if radius_data_arr is not None:
            rg = h5.create_group('radius_data')
            for ci, col_name in enumerate(radius_cols):
                if ci < radius_data_arr.shape[1]:
                    rg.create_dataset(col_name, data=radius_data_arr[:, ci])
            rg.attrs['column_names'] = radius_cols
        
        # /merge_map/
        if merge_map_data is not None:
            mg = h5.create_group('merge_map')
            for ci, col_name in enumerate(merge_map_cols):
                if ci < merge_map_data.shape[1]:
                    mg.create_dataset(col_name, data=merge_map_data[:, ci])
            mg.attrs['column_names'] = merge_map_cols
        
        # /grains/
        gg = h5.create_group('grains')
        gg.create_dataset('summary', data=grains_data)
        gg.attrs['column_names'] = grains_cols[:min(len(grains_cols), grains_data.shape[1])]
        
        for grain_row in grains_data:
            grain_id = int(grain_row[0])
            grain_grp = gg.create_group(f'grain_{grain_id:04d}')
            grain_grp.create_dataset('grain_id', data=grain_id)
            grain_grp.create_dataset('orientation', data=grain_row[1:10].reshape(3, 3))
            grain_grp.create_dataset('position', data=grain_row[10:13])
            if grains_data.shape[1] > 43:
                grain_grp.create_dataset('euler_angles', data=grain_row[43:46])
            if grains_data.shape[1] > 18:
                grain_grp.create_dataset('lattice_params_fit', data=grain_row[13:19])
            if grains_data.shape[1] > 27:
                grain_grp.create_dataset('strain_fable', data=grain_row[19:28].reshape(3, 3))
            if grains_data.shape[1] > 36:
                grain_grp.create_dataset('strain_kenesei', data=grain_row[28:37].reshape(3, 3))
            if grains_data.shape[1] > 37:
                grain_grp.create_dataset('rms_strain_error', data=grain_row[37])
            if grains_data.shape[1] > 38:
                grain_grp.create_dataset('confidence', data=grain_row[38])
            if grains_data.shape[1] > 41:
                grain_grp.create_dataset('phase_nr', data=int(grain_row[41]))
            if grains_data.shape[1] > 42:
                grain_grp.create_dataset('radius', data=grain_row[42])
            
            # Get spots for this grain
            grain_spots = spot_data[spot_data[:, 0] == grain_id]
            spots_grp = grain_grp.create_group('spots')
            spots_grp.create_dataset('n_spots', data=len(grain_spots))
            
            if len(grain_spots) > 0:
                for ci, col in enumerate(spot_cols[1:], start=1):
                    if ci < grain_spots.shape[1]:
                        spots_grp.create_dataset(col.lower(), data=grain_spots[:, ci])
                
                # Per-spot subgroups with constituent peaks
                for spot_row in grain_spots:
                    spot_id = int(spot_row[1])
                    sp_grp = spots_grp.create_group(f'spot_{spot_id:06d}')
                    sp_grp.create_dataset('spot_id', data=spot_id)
                    for ci, col in enumerate(spot_cols[2:], start=2):
                        if ci < spot_row.shape[0]:
                            sp_grp.create_dataset(col.lower(), data=spot_row[ci])
                    
                    # Add radius-derived properties
                    if spot_id in radius_idx:
                        ri = radius_idx[spot_id]
                        rrow = radius_data_arr[ri]
                        for ri_name in ['MinOme', 'MaxOme', 'Theta', 'DeltaOmega',
                                        'NImgs', 'GrainVolume', 'GrainRadius',
                                        'PowderIntensity']:
                            ri_ci = radius_cols.index(ri_name)
                            if ri_ci < rrow.shape[0]:
                                ds_name = ri_name.lower()
                                # Avoid name collision with SpotMatrix columns
                                if ds_name in sp_grp:
                                    ds_name = f'radius_{ds_name}'
                                sp_grp.create_dataset(ds_name, data=rrow[ri_ci])
                    
                    # Constituent peaks from MergeMap + _PS.csv data
                    if spot_id in merge_idx:
                        constituents = merge_idx[spot_id]
                        cp_grp = sp_grp.create_group('constituent_peaks')
                        cp_grp.create_dataset('n_constituent_peaks', data=len(constituents))
                        frame_nrs = [c[0] for c in constituents]
                        peak_ids = [c[1] for c in constituents]
                        cp_grp.create_dataset('frame_nr', data=np.array(frame_nrs, dtype=int))
                        cp_grp.create_dataset('peak_id', data=np.array(peak_ids, dtype=int))
                        # Look up full peak data from _PS.csv cache
                        peak_rows = []
                        for fn, pid in constituents:
                            if (fn, pid) in ps_cache:
                                peak_rows.append(ps_cache[(fn, pid)])
                        if peak_rows:
                            peak_arr = np.array(peak_rows)
                            for ci, col in enumerate(ps_cols):
                                if ci < peak_arr.shape[1]:
                                    cp_grp.create_dataset(col.lower(), data=peak_arr[:, ci])
        
        # /spot_matrix/ - flat table of SpotMatrix.csv
        sm = h5.create_group('spot_matrix')
        sm.create_dataset('data', data=spot_data)
        sm.attrs['column_names'] = spot_cols
        
        # /hkls/
        if hkls_data is not None:
            hg = h5.create_group('hkls')
            hg.create_dataset('data', data=hkls_data)
            hg.attrs['column_names'] = hkls_cols[:min(len(hkls_cols), hkls_data.shape[1])]
        
        # /peaks/summary/ - Result_StartNr_*.csv
        pg = h5.create_group('peaks')
        if result_data is not None:
            ps_sum = pg.create_group('summary')
            ps_sum.create_dataset('data', data=result_data)
            ps_sum.attrs['column_names'] = result_cols[:min(len(result_cols), result_data.shape[1])]
        
        # /peaks/per_frame/ - all _PS.csv rows with frame_nr prepended
        if all_ps_rows:
            pf_data = np.array(all_ps_rows)
            pf = pg.create_group('per_frame')
            pf.create_dataset('data', data=pf_data)
            pf_cols = ['FrameNr'] + ps_cols
            pf.attrs['column_names'] = pf_cols[:min(len(pf_cols), pf_data.shape[1])]
        
        # /id_rings/
        if idrings_data is not None:
            ir = h5.create_group('id_rings')
            ir.create_dataset('data', data=idrings_data)
            ir.attrs['column_names'] = idrings_cols[:min(len(idrings_cols), idrings_data.shape[1])]
        
        # /ids_hash/
        if idshash_data is not None:
            ih = h5.create_group('ids_hash')
            ih.create_dataset('data', data=idshash_data)
            ih.attrs['column_names'] = idshash_cols[:min(len(idshash_cols), idshash_data.shape[1])]
        
        # /spots_to_index/
        if spots2index_data is not None:
            si = h5.create_group('spots_to_index')
            si.create_dataset('data', data=spots2index_data)
        
        # /grain_ids_key/ - padded array (-1 = padding)
        if grainidskey_data is not None:
            gk = h5.create_group('grain_ids_key')
            gk.create_dataset('data', data=grainidskey_data)
            gk.attrs['description'] = 'Each row is a grain. Values are alternating (SpotID, LocalIndex) pairs. -1 indicates padding.'
        
        # /raw_data_ref/
        rr = h5.create_group('raw_data_ref')
        rr.create_dataset('zarr_path', data=os.path.abspath(zarr_path) if zarr_path else '')
    
    logger.info(f"Consolidated HDF5 written: {h5_path}")
    
    # ---------- Append provenance to zarr.zip ----------
    if zarr_path and os.path.exists(zarr_path):
        try:
            store = zarr.ZipStore(zarr_path, mode='a')
            root = zarr.open(store, mode='a')
            prov = root.require_group('analysis/provenance')
            
            # merge_map (3-column: MergedSpotID, FrameNr, PeakID)
            if merge_map_data is not None:
                mm_grp = prov.require_group('merge_map')
                for ci, col in enumerate(merge_map_cols):
                    if ci < merge_map_data.shape[1]:
                        ds_name = col.lower()
                        if ds_name in mm_grp:
                            del mm_grp[ds_name]
                        mm_grp.create_dataset(ds_name, data=merge_map_data[:, ci],
                                              chunks=True, overwrite=True)
            
            # grain_spots mapping
            gs_grp = prov.require_group('grain_spots')
            if 'grain_ids' in gs_grp:
                del gs_grp['grain_ids']
            if 'spot_ids' in gs_grp:
                del gs_grp['spot_ids']
            gs_grp.create_dataset('grain_ids', data=spot_data[:, 0].astype(int),
                                  chunks=True, overwrite=True)
            gs_grp.create_dataset('spot_ids', data=spot_data[:, 1].astype(int),
                                  chunks=True, overwrite=True)
            
            store.close()
            logger.info(f"Provenance appended to zarr: {zarr_path}")
        except Exception as e:
            logger.warning(f"Failed to append provenance to zarr: {e}")

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

def reprocess_results(result_dir: str) -> None:
    """Re-run MergeOverlappingPeaksAllZarr and generate consolidated HDF5.

    Use this to regenerate MergeMap.csv and the consolidated HDF5 file for
    datasets that were processed before these features were added.

    Args:
        result_dir: Path to an existing analysis result directory
                    (e.g., LayerNr_1/) containing a .MIDAS.zip and Grains.csv.
    """
    result_dir = os.path.abspath(result_dir)
    logger.info(f"Reprocessing results in {result_dir}")

    # Find the .MIDAS.zip file
    zip_files = glob.glob(os.path.join(result_dir, '*.MIDAS.zip'))
    if not zip_files:
        raise FileNotFoundError(
            f"No .MIDAS.zip file found in {result_dir}. "
            f"This directory must contain a Zarr-ZIP archive from a previous analysis.")
    outFStem = zip_files[0]
    logger.info(f"Found Zarr-ZIP: {outFStem}")

    # Determine bin directory
    bin_directory = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'bin')
    if not os.path.isdir(bin_directory):
        # Try MIDAS_HOME
        midas_home = os.environ.get('MIDAS_HOME', '')
        if midas_home:
            bin_directory = os.path.join(midas_home, 'FF_HEDM', 'bin')
    logger.info(f"Using bin directory: {bin_directory}")

    # Ensure log directory exists
    log_dir = os.path.join(result_dir, 'midas_log')
    os.makedirs(log_dir, exist_ok=True)

    # Step 1: Re-run MergeOverlappingPeaksAllZarr to generate MergeMap.csv
    merge_bin = os.path.join(bin_directory, 'MergeOverlappingPeaksAllZarr')
    if not os.path.exists(merge_bin):
        raise FileNotFoundError(f"MergeOverlappingPeaksAllZarr binary not found at {merge_bin}")

    logger.info("Step 1/2: Running MergeOverlappingPeaksAllZarr (generating MergeMap.csv)...")
    f_merge_out = os.path.join(log_dir, 'merge_overlaps_reprocess_out.csv')
    f_merge_err = os.path.join(log_dir, 'merge_overlaps_reprocess_err.csv')
    cmd = f"{merge_bin} {outFStem}"
    safely_run_command(cmd, result_dir, f_merge_out, f_merge_err,
                       task_name="Peak merging (reprocess)")

    merge_map = os.path.join(result_dir, 'MergeMap.csv')
    if os.path.exists(merge_map):
        logger.info(f"MergeMap.csv generated: {merge_map}")
    else:
        logger.warning("MergeMap.csv was not generated — check merge log for errors")

    # Step 2: Generate consolidated HDF5
    logger.info("Step 2/2: Generating consolidated HDF5...")
    try:
        generate_consolidated_hdf5(result_dir, outFStem)
    except Exception as e:
        logger.error(f"Failed to generate consolidated HDF5: {e}")
        raise

    logger.info("Reprocessing complete.")


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
    parser.add_argument('-reprocess', type=int, required=False, default=0,
                        help='Set to 1 to re-run peak merging (MergeMap.csv) and consolidated HDF5 generation on existing results. Only needs -resultFolder (or runs in current dir).')
    parser.add_argument('-batchMode', type=int, required=False, default=0,
                        help='Auto-detect files in RawFolder. Handles varying FileStem values across layers, skips darks and missing file numbers. Only for NrFilesPerSweep=1.')
    
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
    if not ps_fn and not data_fn and args.reprocess != 1:
        logger.error("Either paramFN or dataFN must be provided")
        sys.exit(1)
    
    # Handle reprocess mode
    if args.reprocess == 1:
        reprocess_dir = result_dir if result_dir else os.getcwd()
        # If layer dirs exist, reprocess each
        layer_dirs = sorted(glob.glob(os.path.join(reprocess_dir, 'LayerNr_*')))
        if layer_dirs:
            for ld in layer_dirs:
                try:
                    reprocess_results(ld)
                except Exception as e:
                    logger.error(f"Reprocess failed for {ld}: {e}")
        else:
            reprocess_results(reprocess_dir)
        return
        
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
        if args.batchMode == 1 and ps_fn:
            # ── Batch mode: auto-detect files with varying stems ──────────
            params = read_parameter_file(ps_fn)
            raw_folder = params.get('RawFolder', '.')
            ext = params.get('Ext', '.ge3')
            if not ext.startswith('.'):
                ext = '.' + ext
            padding = parse_int_param(params, 'Padding', default=6)
            start_fn = parse_int_param(params, 'StartFileNrFirstLayer', default=1)

            start_file_nr = start_fn + (start_layer_nr - 1)
            end_file_nr = start_fn + (end_layer_nr - 1)

            discovered = discover_layer_files(
                raw_folder, ext, padding, start_file_nr, end_file_nr
            )

            if not discovered:
                logger.error("Batch mode: no valid files found in the specified range.")
                sys.exit(1)

            progress = ProgressTracker(len(discovered), "Batch layer processing")

            for file_nr, filestem in discovered:
                layer_nr = file_nr - start_fn + 1
                logger.info(f"Batch: file_nr={file_nr}, stem={filestem}, layer_nr={layer_nr}")

                # Update FileStem in the parameter file for this layer
                update_parameter_file(ps_fn, {'FileStem': filestem})

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

                    progress.update(message=f"Layer {layer_nr} (file {file_nr}, {filestem}) completed")

                except Exception as e:
                    logger.error(f"Failed to process layer {layer_nr} (file {file_nr}, {filestem}): {e}")
                    sys.exit(1)
                finally:
                    os.chdir(orig_dir)
        else:
            # ── Standard mode: fixed FileStem, sequential layers ──────────
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