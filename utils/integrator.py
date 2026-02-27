#!/usr/bin/env python3

"""
MIDAS Data Integration Tool

This script processes scientific data files for MIDAS software, integrating
multiple data files with support for parallel processing.

Features:
- Parallel processing of multiple data files
- Conversion between various data formats
- Integration with MIDAS detector mapping and analysis tools
- Progress reporting for long-running operations
- Graceful error handling and shutdown

Author: Hemant Sharma
"""

import argparse
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, Any, Iterator, TypeVar

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.warning("tqdm library not available. Progress bars disabled. Install with 'pip install tqdm'.")

import fsspec
import numpy as np
import zarr
import scipy.io

# Type variables for generics
T = TypeVar('T')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("integrator")

# Set paths dynamically using script location
from functools import lru_cache

@lru_cache(maxsize=1)
def get_installation_dir() -> Path:
    """Get the installation directory from the script's location.
    Cached for performance.
    
    Returns:
        Path: Installation directory path
    """
    script_dir = Path(__file__).resolve().parent
    install_dir = script_dir.parent
    return install_dir

# Path configuration
MIDAS_HOME = get_installation_dir()
MIDAS_UTILS = MIDAS_HOME / "utils"
MIDAS_BIN = MIDAS_HOME / "FF_HEDM" / "bin"

# Add MIDAS utils to path
sys.path.insert(0, str(MIDAS_UTILS))
import midas_config
midas_config.run_startup_checks()
try:
    from midas2zip import Hdf5ToZarr
except ImportError:
    logger.error("Could not import MIDAS utilities. Make sure MIDAS is properly installed.")
    raise


class ProcessingError(Exception):
    """Base exception for all processing-related errors."""
    pass


class IntegrationError(ProcessingError):
    """Exception raised for errors during the integration process."""
    pass


class FileOperationError(ProcessingError):
    """Exception raised for errors during file operations."""
    pass


class ConfigurationError(ProcessingError):
    """Exception raised for errors in configuration."""
    pass


class ProgressStatus(Enum):
    """Enumeration of possible progress statuses."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class ProgressInfo:
    """Information about progress of a processing task."""
    file_number: int
    status: ProgressStatus
    message: str = ""
    output_file: Optional[Path] = None
    error: Optional[Exception] = None


@dataclass
class ProcessingParameters:
    """Container for file processing parameters"""
    result_dir: Path
    param_file: Path
    input_file_pattern: str
    dark_file: Optional[Path] = None
    data_loc: str = 'exchange/data'
    dark_loc: str = 'exchange/dark'
    frame_chunks: int = -1
    pre_proc: int = -1
    convert_files: bool = True
    write_mat: bool = False
    log_dir: str = 'stdout'
    skip_existing: bool = False
    progress_callback: Optional[Callable[[ProgressInfo], None]] = None
    log_level: int = logging.INFO
    nCPUsLocal: int = 4


class FileProcessor:
    """Handles processing of data files for integration"""
    
    def __init__(self, params: ProcessingParameters):
        """Initialize with processing parameters
        
        Args:
            params: Processing parameters configuration
        
        Raises:
            ConfigurationError: If essential directories can't be created
        """
        self.params = params
        self._configure_logging()
        try:
            self._ensure_dirs_exist()
        except Exception as e:
            raise ConfigurationError(f"Failed to create required directories: {str(e)}")
    
    def _configure_logging(self) -> None:
        """Configure logging based on parameters"""
        logger.setLevel(self.params.log_level)
    
    def _ensure_dirs_exist(self) -> None:
        """Ensure necessary directories exist
        
        Raises:
            FileOperationError: If directories can't be created
        """
        try:
            self.params.result_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.params.result_dir / self.params.log_dir
            log_path.mkdir(exist_ok=True)
        except Exception as e:
            raise FileOperationError(f"Could not create required directories: {str(e)}")
    
    def _get_file_path(self, file_nr: int, start_file_nr: int) -> Path:
        """
        Generate full file path for a given file number.
        
        Args:
            file_nr: File number offset
            start_file_nr: Starting file number
            
        Returns:
            Path to the file
        """
        padded_file_nr = str(start_file_nr + file_nr).zfill(6)
        start_file_nr_str = str(start_file_nr).zfill(6)
        file_path = Path(self.params.input_file_pattern.replace(start_file_nr_str, padded_file_nr))
        return file_path
    
    def _report_progress(self, progress_info: ProgressInfo) -> None:
        """Report progress to callback if provided
        
        Args:
            progress_info: Progress information
        """
        if self.params.progress_callback:
            self.params.progress_callback(progress_info)
    
    @contextmanager
    def _run_command(
        self, 
        cmd: List[str],
        out_file: Path,
        err_file: Path,
        error_msg: str
    ) -> Iterator[subprocess.CompletedProcess]:
        """Run a command with proper error handling
        
        Args:
            cmd: Command to run (list of arguments)
            out_file: File to capture stdout
            err_file: File to capture stderr
            error_msg: Error message prefix for exceptions
            
        Yields:
            Completed process information
            
        Raises:
            IntegrationError: If command execution fails
        """
        cmd_str = ' '.join(cmd)
        try:
            with open(out_file, 'w') as out_f, open(err_file, 'w') as err_f:
                result = subprocess.run(cmd_str, shell=True, stdout=out_f, stderr=err_f)
                
            if result.returncode != 0:
                try:
                    with open(err_file, 'r') as f:
                        err_content = f.read().strip()
                    full_error = f"{error_msg}\nError log:\n{err_content}"
                except Exception as e:
                    full_error = f"{error_msg}\nCouldn't read error log: {str(e)}"
                    
                logger.error(full_error)
                raise IntegrationError(full_error)
                
            yield result
            
        except subprocess.SubprocessError as e:
            full_error = f"{error_msg}\nSubprocess error: {str(e)}"
            logger.error(full_error)
            raise IntegrationError(full_error)
    
    def generate_zip(self, data_file: Path) -> Path:
        """
        Generate ZIP file from data.
        
        Args:
            data_file: Input data file
            
        Returns:
            Path to generated ZIP file
        
        Raises:
            IntegrationError: If errors are encountered during ZIP generation
        """
        if not data_file.exists():
            raise FileOperationError(f"Input file does not exist: {data_file}")
            
        cmd = [
            sys.executable,
            str(MIDAS_UTILS / 'ffGenerateZipRefactor.py'),
            '-resultFolder', str(self.params.result_dir),
            '-paramFN', str(self.params.param_file)
        ]
        
        if self.params.dark_file:
            cmd.extend(['-darkFN', str(self.params.dark_file)])
        
        cmd.extend(['-dataFN', str(data_file)])
        
        if self.params.data_loc:
            cmd.extend(['-dataLoc', self.params.data_loc])
        if self.params.dark_loc:
            cmd.extend(['-darkLoc', self.params.dark_loc])
        if self.params.frame_chunks != -1:
            cmd.extend(['-numFrameChunks', str(self.params.frame_chunks)])
        if self.params.pre_proc != -1:
            cmd.extend(['-preProcThresh', str(self.params.pre_proc)])
        
        # Output file paths
        data_basename = data_file.name
        log_path = self.params.result_dir / self.params.log_dir
        out_file = log_path / f"{data_basename}_ZipOut.txt"
        err_file = log_path / f"{data_basename}_ZipErr.txt"
        
        # Run command
        with self._run_command(
            cmd, 
            out_file, 
            err_file, 
            f"ZIP generation failed for {data_file}"
        ):
            pass
        
        # Extract output ZIP filename
        try:
            with open(out_file, 'r') as f:
                lines = f.readlines()
                if lines and lines[-1].startswith('OutputZipName'):
                    return Path(lines[-1].split()[1])
                else:
                    error_msg = f"Could not find output ZIP filename in {out_file}"
                    logger.error(error_msg)
                    raise IntegrationError(error_msg)
        except Exception as e:
            error_msg = f"Failed to read output file {out_file}: {str(e)}"
            logger.error(error_msg)
            raise IntegrationError(error_msg)
    
    def convert_hdf_to_zarr(self, hdf_file: Path, input_zip_file: Path) -> Path:
        """
        Convert HDF5 file to Zarr ZIP format.
        
        Args:
            hdf_file: Input HDF5 file
            input_zip_file: Original Input Zarr file from which metadata will be copied
            
        Returns:
            Path to output Zarr ZIP file
            
        Raises:
            IntegrationError: If conversion fails
        """
        if not hdf_file.exists():
            raise FileOperationError(f"HDF file does not exist: {hdf_file}")
            
        output_zip = Path(f"{hdf_file}.zarr.zip")
        
        # Backup existing file if it exists
        if output_zip.exists():
            try:
                backup_path = Path(f"{output_zip}.old")
                shutil.move(str(output_zip), str(backup_path))
                logger.info(f"Backed up existing file to {backup_path}")
            except Exception as e:
                logger.warning(f"Could not backup existing file {output_zip}: {str(e)}")
        
        try:
            # Convert HDF5 to Zarr
            with fsspec.open(str(hdf_file), mode='rb', anon=False, 
                            requester_pays=True, default_fill_cache=False) as f:
                with zarr.ZipStore(str(output_zip)) as store_zip:
                    h5_chunks_zip = Hdf5ToZarr(f, store_zip)
                    h5_chunks_zip.translate()
            
            # Enrich Zarr with metadata from initial zarr
            self._enrich_zarr_with_metadata(input_zip_file, output_zip)

            # Verify the file was created successfully
            if not output_zip.exists() or output_zip.stat().st_size == 0:
                raise IntegrationError(f"Failed to create valid Zarr ZIP file: {output_zip}")
                
            return output_zip
        except Exception as e:
            error_msg = f"Failed to convert HDF5 file {hdf_file} to Zarr ZIP: {str(e)}"
            logger.error(error_msg)
            raise IntegrationError(error_msg) from e

    def _copy_group_recursive(self, group_in, group_out, total_frames, omega_sum_frames, exclude_keys=None, path_prefix=''):
        """
        Recursively copy all datasets from a Zarr group to an output group.
        1D arrays with length == total_frames are averaged per OmegaSumFrames chunk.
        
        Args:
            group_in: Source Zarr group
            group_out: Destination Zarr group
            total_frames: Number of frames in the original data
            omega_sum_frames: Number of frames summed per chunk
            exclude_keys: Set of keys to skip (only at top level)
            path_prefix: Current path for logging
        """
        import math
        if exclude_keys is None:
            exclude_keys = set()
        
        for key in group_in.keys():
            if not path_prefix and key in exclude_keys:
                continue
            
            full_path = f"{path_prefix}/{key}" if path_prefix else key
            item = group_in[key]
            
            try:
                if isinstance(item, zarr.hierarchy.Group):
                    # Recurse into sub-groups
                    sub_out = group_out.require_group(key)
                    self._copy_group_recursive(item, sub_out, total_frames, omega_sum_frames, path_prefix=full_path)
                else:
                    # It's a dataset — read it
                    data_in = item[()]
                    
                    # If 1D array matching frame count and we're summing frames, average it
                    if (isinstance(data_in, np.ndarray) and data_in.ndim == 1 
                            and len(data_in) == total_frames 
                            and omega_sum_frames > 1 and total_frames > 0):
                        num_chunks = math.ceil(total_frames / omega_sum_frames)
                        chunked_data = np.zeros(num_chunks, dtype=data_in.dtype)
                        for i in range(num_chunks):
                            start_idx = i * omega_sum_frames
                            end_idx = min(start_idx + omega_sum_frames, total_frames)
                            chunked_data[i] = np.mean(data_in[start_idx:end_idx])
                        data_to_write = chunked_data
                        logger.debug(f"Averaged '{full_path}' from {data_in.shape} to {data_to_write.shape}")
                    else:
                        data_to_write = data_in
                    
                    # Write to output (overwrite if exists)
                    if key in group_out:
                        del group_out[key]
                    group_out.create_dataset(key, data=data_to_write)
                    print(f"  Copied: {full_path} (shape={data_to_write.shape if isinstance(data_to_write, np.ndarray) else 'scalar'})")
                    logger.info(f"Copied metadata '{full_path}' to output Zarr.")
            except Exception as e:
                logger.warning(f"Failed to copy '{full_path}': {e}")

    def _enrich_zarr_with_metadata(self, input_zip_file: Path, output_zip_file: Path) -> None:
        """
        Copy metadata from the input Zarr to the output Zarr:
          - measurement/process/scan_parameters (excluding datatype, start, step)
          - instrument/ (all keys recursively)
        
        1D arrays with length == total_frames are averaged per OmegaSumFrames chunk.
        
        Args:
            input_zip_file: Path to original input Zarr.zip
            output_zip_file: Path to output Zarr.zip
        """
        try:
            with zarr.open(str(input_zip_file), mode='r') as z_in:
                # Get total frames
                total_frames = 0
                if 'exchange/data' in z_in:
                    total_frames = z_in['exchange/data'].shape[0]
                    
                # Get OmegaSumFrames
                omega_sum_frames = 1
                try:
                    if 'analysis/process/analysis_parameters/OmegaSumFrames' in z_in:
                        omega_sum_frames = int(z_in['analysis/process/analysis_parameters/OmegaSumFrames'][0])
                except Exception as e:
                    logger.debug(f"Could not read OmegaSumFrames, defaulting to 1: {e}")
                
                with zarr.open(str(output_zip_file), mode='a') as z_out:
                    # Copy scan_parameters
                    if 'measurement/process/scan_parameters' in z_in:
                        sp_in = z_in['measurement/process/scan_parameters']
                        sp_out = z_out.require_group('measurement/process/scan_parameters')
                        self._copy_group_recursive(
                            sp_in, sp_out, total_frames, omega_sum_frames,
                            exclude_keys={'datatype', 'start', 'step'},
                            path_prefix='measurement/process/scan_parameters'
                        )
                    else:
                        logger.info("No scan parameters found in input Zarr.")
                    
                    # Copy instrument/ group
                    if 'instrument' in z_in:
                        inst_in = z_in['instrument']
                        inst_out = z_out.require_group('instrument')
                        self._copy_group_recursive(
                            inst_in, inst_out, total_frames, omega_sum_frames,
                            path_prefix='instrument'
                        )
                        logger.info("Copied instrument metadata to output Zarr.")
                    else:
                        logger.info("No instrument group found in input Zarr.")
                        
        except Exception as e:
            logger.error(f"Error enriching Zarr with metadata: {e}")
    
    def save_as_matlab(self, zarr_file_path: Path) -> Optional[Path]:
        """
        Save Zarr file as MATLAB .mat file
        
        Args:
            zarr_file_path: Path to Zarr file
            
        Returns:
            Path to the created .mat file or None if operation failed
            
        Raises:
            FileOperationError: If the operation fails critically
        """
        if not zarr_file_path.exists():
            raise FileOperationError(f"Zarr file does not exist: {zarr_file_path}")
            
        output_path = Path(f"{zarr_file_path}.mat")
        
        try:
            with zarr.open(str(zarr_file_path), mode='r') as zarr_file:
                # Create a dictionary with proper handling of zarr arrays
                data_dict = {}
                for key in zarr_file.keys():
                    try:
                        # Handle different kinds of zarr arrays with appropriate slicing
                        if isinstance(zarr_file[key], zarr.core.Array):
                            data_dict[key] = zarr_file[key][...]  # Use [...] for full slice
                        else:
                            # For zarr groups, store metadata about the group
                            data_dict[key] = f"Group: {str(zarr_file[key])}"
                    except Exception as inner_e:
                        logger.warning(f"Could not extract key '{key}' from zarr file: {str(inner_e)}")
                
                # Save as .mat file
                scipy.io.savemat(str(output_path), data_dict)
                logger.info(f"Successfully created MATLAB file: {output_path}")
                return output_path
                
        except Exception as e:
            # Log the error but don't raise to allow processing to continue
            logger.error(f"Failed to create MATLAB file for {zarr_file_path}: {str(e)}")
            
            # If this is a critical error that should stop processing, raise it
            if isinstance(e, (MemoryError, KeyboardInterrupt)):
                raise FileOperationError(f"Critical error during MATLAB file creation: {str(e)}") from e
                
            return None
    
    def run_integrator(self, zip_file: Path) -> Path:
        """
        Run IntegratorZarr on a ZIP file
        
        Args:
            zip_file: Input ZIP file
            
        Returns:
            Path to the output HDF file
            
        Raises:
            IntegrationError: If integration fails
        """
        if not zip_file.exists():
            raise FileOperationError(f"ZIP file does not exist: {zip_file}")
            
        log_path = self.params.result_dir / self.params.log_dir
        out_log = log_path / f"{zip_file.name}_integrator_out.csv"
        err_log = log_path / f"{zip_file.name}_integrator_err.csv"
        
        integrator_path = MIDAS_BIN / 'IntegratorZarrOMP'
        integrator_cmd = [str(integrator_path), str(zip_file),str(self.params.nCPUsLocal)]
        
        with self._run_command(
            integrator_cmd, 
            out_log, 
            err_log,
            f"IntegratorZarr failed for {zip_file}"
        ):
            pass
        
        # Check if the HDF file was created
        expected_hdf = Path(f"{zip_file}.caked.hdf")
        if not expected_hdf.exists():
            error_msg = f"The expected HDF file {expected_hdf} was not created"
            logger.error(error_msg)
            raise IntegrationError(error_msg)
        
        return expected_hdf
    
    def run_detector_mapper(self, zip_file: Path) -> None:
        """
        Run detector mapper on ZIP file.
        
        Args:
            zip_file: Input ZIP file
            
        Raises:
            IntegrationError: If mapping fails
        """
        if not zip_file.exists():
            raise FileOperationError(f"ZIP file does not exist: {zip_file}")
            
        log_path = self.params.result_dir / self.params.log_dir
        out_log = log_path / "map_out.csv"
        err_log = log_path / "map_err.csv"
        
        mapper_path = MIDAS_BIN / 'DetectorMapperZarr'
        mapper_cmd = [str(mapper_path), str(zip_file)]
        
        with self._run_command(
            mapper_cmd, 
            out_log, 
            err_log,
            f"DetectorMapperZarr failed for {zip_file}"
        ):
            pass
    
    def process_file(self, file_nr: int, start_file_nr: int) -> Optional[Path]:
        """
        Process a single data file.
        
        Args:
            file_nr: File number index
            start_file_nr: Starting file number
            
        Returns:
            Path to output Zarr ZIP file or None if file doesn't exist
            
        Raises:
            IntegrationError: If processing fails
        """
        # Generate filename for this file number
        file_path = self._get_file_path(file_nr, start_file_nr)
        file_num = start_file_nr + file_nr
        
        # Report starting progress
        self._report_progress(ProgressInfo(
            file_number=file_num,
            status=ProgressStatus.PENDING,
            message=f"Starting processing for file {file_path}"
        ))
        
        # Check if the file exists
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}. Skipping.")
            self._report_progress(ProgressInfo(
                file_number=file_num,
                status=ProgressStatus.SKIPPED,
                message=f"File does not exist: {file_path}"
            ))
            return None
        
        logger.info(f"Processing file {file_path}")
        self._report_progress(ProgressInfo(
            file_number=file_num,
            status=ProgressStatus.RUNNING,
            message=f"Processing file {file_path}"
        ))
        
        try:
            # Generate ZIP if needed
            if file_nr > 0 and self.params.convert_files:
                zip_file = self.generate_zip(file_path)
            else:
                if not str(file_path).endswith('.zip'):
                    file_path = Path(f"{file_path}.analysis.MIDAS.zip")
                zip_file = self.params.result_dir / file_path.name
                logger.info(f'Using existing zip file: {zip_file}')
            
            # Run integrator
            hdf_file = self.run_integrator(zip_file)
            
            # Convert HDF to Zarr
            out_zip = self.convert_hdf_to_zarr(hdf_file, zip_file)
            
            # Save as MATLAB file if requested
            if self.params.write_mat:
                self.save_as_matlab(out_zip)
            
            # Report success
            self._report_progress(ProgressInfo(
                file_number=file_num,
                status=ProgressStatus.COMPLETED,
                message=f"Successfully processed file {file_path}",
                output_file=out_zip
            ))
            
            return out_zip
            
        except Exception as e:
            # Report failure
            self._report_progress(ProgressInfo(
                file_number=file_num,
                status=ProgressStatus.FAILED,
                message=f"Error processing file {file_path}: {str(e)}",
                error=e
            ))
            
            # Re-raise the exception
            if isinstance(e, (IntegrationError, FileOperationError)):
                raise
            else:
                raise IntegrationError(f"Error processing file {file_path}: {str(e)}") from e


class MidasIntegrator:
    """Main integrator class that orchestrates the data processing"""
    
    def __init__(self):
        """Initialize the integrator"""
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Initialize state
        self.args, self.overrides = self._parse_arguments()
        
        # Handle parameter file overrides
        self.temp_param_file = None
        if self.overrides:
            try:
                self.temp_param_file = self._create_temp_param_file(Path(self.args.paramFN), self.overrides)
                logger.info(f"Created temporary parameter file with overrides: {self.temp_param_file}")
                # Update args to point to temp file
                self.args.paramFN = str(self.temp_param_file)
            except Exception as e:
                logger.error(f"Failed to create temporary parameter file: {e}")
                sys.exit(1)
        
        self.params = self._create_processing_parameters()
        self.processor = FileProcessor(self.params)
        self.completed_files = set()
        self._shutdown_requested = False
        
    def _create_temp_param_file(self, original_path: Path, overrides: Dict[str, str]) -> Path:
        """Create a temporary parameter file with overridden values.
        
        Args:
             original_path: Path to original parameter file
             overrides: Dictionary of parameter overrides
             
        Returns:
            Path to temporary parameter file
        """
        import tempfile
        
        # Read original content
        with open(original_path, 'r') as f:
            lines = f.readlines()
            
        # Create a map of existing keys to line numbers for replacement
        key_map = {}
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts:
                key_map[parts[0]] = i
        
        # Apply overrides
        new_lines = list(lines)
        for key, value in overrides.items():
            if key in key_map:
                # Replace existing line
                idx = key_map[key]
                # Preserve original commenting if present, though simplistic
                original_comment = ""
                if '#' in new_lines[idx]:
                    original_comment = " #" + new_lines[idx].split('#', 1)[1].strip()
                new_lines[idx] = f"{key} {value}{original_comment}\n"
                logger.info(f"Overriding parameter '{key}' with '{value}'")
            else:
                # Append new line
                new_lines.append(f"{key} {value}\n")
                logger.info(f"Adding new parameter '{key}' with '{value}'")
                
        # Write to temp file
        fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix='midas_param_', text=True)
        with os.fdopen(fd, 'w') as f:
            f.writelines(new_lines)
            
        return Path(temp_path)

    def _cleanup_temp_file(self):
        """Remove temporary parameter file if it exists"""
        if self.temp_param_file and self.temp_param_file.exists():
            try:
                os.unlink(self.temp_param_file)
                logger.info(f"Cleaned up temporary parameter file: {self.temp_param_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary parameter file: {e}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        # Store original handlers
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        
        def handler(sig, frame):
            self._cleanup_temp_file()
            self._handle_shutdown_signal(sig, frame)
            
        # Set new handlers
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    def _handle_shutdown_signal(self, sig: int, frame: Any) -> None:
        """Handle shutdown signals gracefully
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        if self._shutdown_requested:
            # Second signal received, use original handler
            logger.warning("Second shutdown signal received. Forcing exit.")
            self._cleanup_temp_file()
            if sig == signal.SIGINT and self._original_sigint:
                self._original_sigint(sig, frame)
            elif sig == signal.SIGTERM and self._original_sigterm:
                self._original_sigterm(sig, frame)
            else:
                # Default behavior: exit
                sys.exit(1)
        else:
            logger.warning(
                "Shutdown signal received. Finishing current tasks and exiting. "
                "Press Ctrl+C again to force immediate exit."
            )
            self._shutdown_requested = True
    
    def _parse_arguments(self) -> Tuple[argparse.Namespace, Dict[str, str]]:
        """Parse command line arguments including overrides
        
        Returns:
            Tuple of (Parsed arguments namespace, Dictionary of overrides)
        """
        parser = argparse.ArgumentParser(
            description='MIDAS Data Integration Tool',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument('-resultFolder', type=str, default='.', 
                          help='Folder where you want to save results.')
        parser.add_argument('-paramFN', type=str, required=True, 
                          help='Parameter file name.')
        parser.add_argument('-dataFN', type=str, required=True, 
                          help='DataFileName for first file, this should have the full path if not in the current folder.')
        parser.add_argument('-darkFN', type=str, default='', 
                          help='DarkFileName, full path.')
        parser.add_argument('-dataLoc', type=str, default='exchange/data', 
                          help='Data location.')
        parser.add_argument('-darkLoc', type=str, default='exchange/dark', 
                          help='Dark location.')
        parser.add_argument('-numFrameChunks', type=int, default=-1, 
                          help='Number of chunks to use when reading the data file if RAM is smaller than expanded data. -1 will disable.')
        parser.add_argument('-preProcThresh', type=int, default=-1, 
                          help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
        parser.add_argument('-startFileNr', type=int, default=-1, 
                          help='Which fileNr to start from. Default is -1, which means that fileNr in dataFN is read.')
        parser.add_argument('-endFileNr', type=int, default=-1, 
                          help='End fileNr. Default is -1, which means a single file is processed.')
        parser.add_argument('-convertFiles', type=int, default=1, 
                          help='Whether want to convert files to ZarrZip format or not.')
        parser.add_argument('-mapDetector', type=int, default=1, 
                          help='Whether want to generate map of detector or not. If unsure, put to 1. If already have the CORRECT Map.bin and nMap.bin, put it to 0.')
        parser.add_argument('-nCPUs', type=int, default=1, 
                          help='If you want to use multiple CPUs.')
        parser.add_argument('-nCPUsLocal', type=int, default=4, 
                          help='If you want to use multiple CPUs locally for each integrator instance. Default is 4, so each integrator instance will use 4 CPUs.')
        parser.add_argument('-writeMat', type=int, default=0, 
                          help='If you want to write a matlab .mat file.')
        parser.add_argument('-skipExisting', action='store_true',
                          help='Skip processing of files that have already been processed.')
        parser.add_argument('-logLevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                          default='INFO', help='Set the logging level')
        
        # Parse known args and capture the rest
        args, unknown = parser.parse_known_args()
        
        # Process unknown args as overrides
        overrides = {}
        i = 0
        while i < len(unknown):
            key = unknown[i]
            if key.startswith('-'):
                key = key.lstrip('-')
                
            if i + 1 < len(unknown):
                value = unknown[i+1]
                overrides[key] = value
                i += 2
            else:
                logger.warning(f"Ignoring duplicate/orphan argument: {key}")
                i += 1
                
        return args, overrides
        
    def _create_processing_parameters(self) -> ProcessingParameters:        
        """Create processing parameters from command line arguments
        
        Returns:
            Processing parameters
        
        Raises:
            ConfigurationError: If parameter file doesn't exist
        """
        # Convert string log level to logging constant
        log_level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        log_level = log_level_map.get(self.args.logLevel, logging.INFO)
        
        result_dir = Path(self.args.resultFolder)
        if str(result_dir) == '.' or not result_dir.is_absolute():
            result_dir = Path.cwd() / self.args.resultFolder
        
        param_file = Path(self.args.paramFN)
        if not param_file.exists():
            raise ConfigurationError(f"Parameter file does not exist: {param_file}")
        
        return ProcessingParameters(
            result_dir=result_dir,
            param_file=param_file,
            input_file_pattern=self.args.dataFN,
            dark_file=Path(self.args.darkFN) if self.args.darkFN else None,
            data_loc=self.args.dataLoc,
            dark_loc=self.args.darkLoc,
            frame_chunks=self.args.numFrameChunks,
            pre_proc=self.args.preProcThresh,
            convert_files=bool(self.args.convertFiles),
            write_mat=bool(self.args.writeMat),
            skip_existing=bool(self.args.skipExisting),
            progress_callback=self._on_progress_update,
            log_level=log_level,
            nCPUsLocal=self.args.nCPUsLocal,
        )
    
    def _on_progress_update(self, progress_info: ProgressInfo) -> None:
        """Default progress callback
        
        Args:
            progress_info: Progress information
        """
        # In the base implementation, just log the progress
        status_str = progress_info.status.name
        if progress_info.status == ProgressStatus.COMPLETED and progress_info.output_file:
            logger.info(
                f"File {progress_info.file_number} [{status_str}]: "
                f"{progress_info.message} -> {progress_info.output_file}"
            )
        else:
            logger.info(f"File {progress_info.file_number} [{status_str}]: {progress_info.message}")

    def _find_start_file_number(self) -> int:
        """Find the start file number from dataFN if not provided
        
        Returns:
            Start file number
            
        Raises:
            ConfigurationError: If start file number can't be determined
        """
        if self.args.startFileNr != -1:
            return self.args.startFileNr
            
        match = re.search(r'\d{6}', self.args.dataFN)
        if not match:
            error_msg = "Could not find 6 padded fileNr in dataFN."
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
            
        return int(match.group(0))
        
    def _get_completed_files(self, start_file_nr: int, end_file_nr: int) -> Set[int]:
        """Get set of file numbers that have already been processed.
        
        Args:
            start_file_nr: Starting file number
            end_file_nr: Ending file number
            
        Returns:
            Set of file numbers that already have zarr.zip files
        """
        completed = set()
        
        # Pattern to match 6-digit file numbers in filenames
        file_nr_pattern = re.compile(r'(\d{6})')
        
        # Look for all .caked.hdf.zarr.zip files in the result directory
        zarr_files = list(self.params.result_dir.glob("*.caked.hdf.zarr.zip"))
        
        # Get the base names of the files we're processing
        start_file_nr_str = str(start_file_nr).zfill(6)
        file_pattern_base = os.path.basename(self.params.input_file_pattern)
        pattern_parts = file_pattern_base.split(start_file_nr_str)
        
        if len(pattern_parts) != 2:
            logger.warning("Could not reliably identify file pattern, skipping completed files check")
            return set()
            
        prefix, suffix = pattern_parts
        
        # Check each zarr.zip file
        for zarr_file in zarr_files:
            zarr_filename = zarr_file.name
            
            # Get the file number from the filename
            if prefix in zarr_filename and suffix in zarr_filename:
                match = file_nr_pattern.search(zarr_filename)
                if match:
                    file_nr = int(match.group(1))
                    if start_file_nr <= file_nr <= end_file_nr:
                        completed.add(file_nr)
        
        logger.info(f"Found {len(completed)} already processed files")
        return completed
    
    def _validate_inputs(self) -> bool:
        """Validate input parameters
        
        Returns:
            True if all inputs are valid, False otherwise
        """
        # Check if parameter file exists
        if not self.params.param_file.exists():
            logger.error(f"Parameter file does not exist: {self.params.param_file}")
            return False
            
        # Check if dark file exists if provided
        if self.params.dark_file and not self.params.dark_file.exists():
            logger.warning(f"Dark file does not exist: {self.params.dark_file}")
            # Not a fatal error, just a warning
            
        return True
    
    def run(self) -> None:
        """Run the integration process
        
        Raises:
            ConfigurationError: If configuration is invalid
            IntegrationError: If integration fails
        """
        try:
            # Validate inputs
            if not self._validate_inputs():
                raise ConfigurationError("Invalid configuration. See previous error messages.")
            
            # Find start file number
            start_file_nr = self._find_start_file_number()
            logger.info(f'Processing starting from file number: {start_file_nr}')
            
            # Determine end file number
            end_file_nr = self.args.endFileNr
            if end_file_nr == -1:
                end_file_nr = start_file_nr
            
            # Calculate number of files to process
            nr_files = end_file_nr - start_file_nr + 1
            logger.info(f"Processing {nr_files} files from {start_file_nr} to {end_file_nr}")
            
            # Check for already processed files if skip_existing is enabled
            if self.params.skip_existing:
                self.completed_files = self._get_completed_files(start_file_nr, end_file_nr)
                if self.completed_files:
                    logger.info(f"Will skip {len(self.completed_files)} already processed files")
            
            # Process first file separately to generate detector map if needed
            file_nr = 0
            file_num = start_file_nr + file_nr
            
            # Get the first file path
            file_path = self.processor._get_file_path(file_nr, start_file_nr)
            
            # Check if first file exists
            if not file_path.exists():
                logger.error(f"First file does not exist: {file_path}")
                # Find the first file that exists
                for i in range(nr_files):
                    test_file_path = self.processor._get_file_path(i, start_file_nr)
                    if test_file_path.exists() and (not self.params.skip_existing or start_file_nr + i not in self.completed_files):
                        file_nr = i
                        file_num = start_file_nr + file_nr
                        file_path = test_file_path
                        logger.info(f"Found first existing file at index {file_nr}: {file_path}")
                        break
                else:
                    error_msg = "No existing files found to process!"
                    logger.error(error_msg)
                    raise FileOperationError(error_msg)
            
            # If first file is in completed files and skip_existing is enabled, find the first non-completed file
            if self.params.skip_existing and file_num in self.completed_files:
                logger.info(f"First file {file_num} already processed, finding first non-processed file...")
                for i in range(nr_files):
                    test_file_path = self.processor._get_file_path(i, start_file_nr)
                    if test_file_path.exists() and start_file_nr + i not in self.completed_files:
                        file_nr = i
                        file_num = start_file_nr + file_nr
                        file_path = test_file_path
                        logger.info(f"Found non-processed file at index {file_nr}: {file_path}")
                        break
                else:
                    logger.info("All files already processed, nothing to do!")
                    return
            
            if self.args.convertFiles:
                zip_file = self.processor.generate_zip(file_path)
            else:
                if not str(file_path).endswith('.zip'):
                    file_path = Path(f"{file_path}.analysis.MIDAS.zip")
                zip_file = self.params.result_dir / file_path.name
                logger.info(f'Using existing zip file: {zip_file}')
            
            # Check if Map.bin and nMap.bin already exist in the result folder
            map_file = self.params.result_dir / "Map.bin"
            nmap_file = self.params.result_dir / "nMap.bin"
            
            # Determine if we need to run detector mapper
            map_files_exist = map_file.exists() and nmap_file.exists()
            
            # Run detector mapper if requested or if map files don't exist (but skip if files exist)
            if self.args.mapDetector or not map_files_exist:
                if map_files_exist:
                    logger.info("Map.bin and nMap.bin already exist in result folder. Skipping detector mapping.")
                elif not self.args.mapDetector:
                    logger.warning("Map.bin and nMap.bin do not exist in result folder but are required. Running detector mapper despite mapDetector=0.")
                else:
                    logger.info("Running detector mapper...")
                
                if not map_files_exist:
                    self.processor.run_detector_mapper(zip_file)
            
            # Create list of files to process (excluding completed files if skip_existing is enabled)
            files_to_process = []
            for file_nr in range(nr_files):
                file_num = start_file_nr + file_nr
                # Check if file exists before adding to processing list
                test_file_path = self.processor._get_file_path(file_nr, start_file_nr)
                if test_file_path.exists() and (not self.params.skip_existing or file_num not in self.completed_files):
                    files_to_process.append(file_nr)
                elif not test_file_path.exists():
                    logger.warning(f"File does not exist: {test_file_path}. Skipping.")
            
            if len(files_to_process) == 0:
                logger.info("No files to process after checking existence and skipping existing files.")
                return
                
            logger.info(f"Will process {len(files_to_process)} of {nr_files} total files")
            
            # Process all files
            results = self._process_files(files_to_process, start_file_nr)
            
            # Print summary of results
            self._print_summary(results)
                    
        except Exception as e:
            logger.error(f"An error occurred during processing: {str(e)}")
            raise
        finally:
            self._cleanup_temp_file()
    
    def _process_files(self, files_to_process: List[int], start_file_nr: int) -> List[Path]:
        """Process multiple files either serially or in parallel
        
        Args:
            files_to_process: List of file indices to process
            start_file_nr: Starting file number
            
        Returns:
            List of output file paths for successfully processed files
            
        Raises:
            IntegrationError: If processing fails
        """
        results = []
        
        # Check if shutdown was requested before starting
        if self._shutdown_requested:
            logger.warning("Shutdown requested before processing started. Aborting.")
            return results
        
        if self.args.nCPUs == 1:
            # Serial processing
            logger.info("Processing files serially")
            
            # Use tqdm if available, otherwise fallback to simple counter
            if TQDM_AVAILABLE:
                progress_bar = tqdm(
                    total=len(files_to_process),
                    desc="Processing files",
                    unit="file"
                )
            else:
                logger.info(f"Starting processing of {len(files_to_process)} files")
            
            # Process each file
            for i, file_nr in enumerate(files_to_process):
                # Check for shutdown request
                if self._shutdown_requested:
                    logger.warning(f"Shutdown requested. Processed {i} of {len(files_to_process)} files.")
                    break
                    
                file_num = start_file_nr + file_nr
                try:
                    # Print progress if tqdm not available
                    if not TQDM_AVAILABLE:
                        logger.info(f"Processing file {i+1}/{len(files_to_process)} (file number: {file_num})")
                    
                    out_zip = self.processor.process_file(file_nr, start_file_nr)
                    if out_zip:
                        results.append(out_zip)
                    
                    # Update progress bar if using tqdm
                    if TQDM_AVAILABLE:
                        progress_bar.update(1)
                    elif out_zip:
                        logger.info(f'Successfully processed: {out_zip}')
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_num}: {str(e)}")
                    # Update progress bar even on error
                    if TQDM_AVAILABLE:
                        progress_bar.update(1)
                    
                    # Re-raise critical errors
                    if isinstance(e, (KeyboardInterrupt, MemoryError)):
                        raise
            
            # Close progress bar if using tqdm
            if TQDM_AVAILABLE:
                progress_bar.close()
        else:
            # Parallel processing
            logger.info(f"Processing files in parallel using {self.args.nCPUs} CPUs")
            
            with ProcessPoolExecutor(max_workers=self.args.nCPUs) as executor:
                # Submit tasks
                futures = {
                    executor.submit(self.processor.process_file, file_nr, start_file_nr): file_nr
                    for file_nr in files_to_process
                }
                
                # Process results as they complete with progress bar if tqdm available
                if TQDM_AVAILABLE:
                    with tqdm(total=len(futures), desc="Processing files", unit="file") as progress_bar:
                        for future in as_completed(futures):
                            # Check for shutdown request
                            if self._shutdown_requested:
                                # Cancel pending futures
                                for f in futures:
                                    if not f.done():
                                        f.cancel()
                                logger.warning("Shutdown requested. Cancelling pending tasks.")
                                break
                                
                            file_nr = futures[future]
                            file_num = start_file_nr + file_nr
                            try:
                                out_zip = future.result()
                                if out_zip:
                                    results.append(out_zip)
                                progress_bar.update(1)
                            except Exception as e:
                                logger.error(f"Error processing file {file_num}: {str(e)}")
                                progress_bar.update(1)
                                
                                # Re-raise critical errors
                                if isinstance(e, (KeyboardInterrupt, MemoryError)):
                                    # Cancel pending futures
                                    for f in futures:
                                        if not f.done():
                                            f.cancel()
                                    raise
                else:
                    # Without tqdm, just log each completion
                    processed_count = 0
                    for future in as_completed(futures):
                        # Check for shutdown request
                        if self._shutdown_requested:
                            # Cancel pending futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            logger.warning("Shutdown requested. Cancelling pending tasks.")
                            break
                            
                        file_nr = futures[future]
                        file_num = start_file_nr + file_nr
                        processed_count += 1
                        try:
                            out_zip = future.result()
                            if out_zip:
                                results.append(out_zip)
                                logger.info(f'[{processed_count}/{len(futures)}] Successfully processed file {file_num}: {out_zip}')
                            else:
                                logger.info(f'[{processed_count}/{len(futures)}] File {file_num} was skipped (doesn\'t exist)')
                        except Exception as e:
                            logger.error(f'[{processed_count}/{len(futures)}] Error processing file {file_num}: {str(e)}')
                            
                            # Re-raise critical errors
                            if isinstance(e, (KeyboardInterrupt, MemoryError)):
                                # Cancel pending futures
                                for f in futures:
                                    if not f.done():
                                        f.cancel()
                                raise
        
        return results
    
    def _print_summary(self, results: List[Path]) -> None:
        """Print summary of processing results
        
        Args:
            results: List of output file paths
        """
        if not results:
            logger.info("No files were successfully processed.")
            return
            
        logger.info(f"Successfully processed {len(results)} files")
        
        # Print details for first few results
        max_detail = min(3, len(results))
        for i, out_zip in enumerate(results[:max_detail]):
            try:
                logger.info(f'Output file {i+1}: {out_zip}')
                try:
                    with zarr.open(str(out_zip)) as z:
                        logger.info(f'File structure: {z.tree(level=2)}')
                except Exception as e:
                    logger.error(f"Error displaying tree structure: {str(e)}")
            except Exception as e:
                logger.error(f"Error displaying result information: {str(e)}")
                
        # If more than max_detail results, just summarize the rest
        if len(results) > max_detail:
            logger.info(f"... and {len(results) - max_detail} more files")


if __name__ == "__main__":
    try:
        integrator = MidasIntegrator()
        integrator.run()
    except KeyboardInterrupt:
        logger.critical("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        sys.exit(1)