#!/usr/bin/env python3

"""
MIDAS Data Integration Tool

This script processes scientific data files for MIDAS software, integrating
multiple data files with support for parallel processing.

Author: Hemant Sharma
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

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
        Installation directory path
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
try:
    from midas2zip import Hdf5ToZarr
except ImportError:
    logger.error("Could not import MIDAS utilities. Make sure MIDAS is properly installed.")
    raise


class IntegrationError(Exception):
    """Exception raised for errors during the integration process."""
    pass


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


class FileProcessor:
    """Handles processing of data files for integration"""
    
    def __init__(self, params: ProcessingParameters):
        """Initialize with processing parameters"""
        self.params = params
        self._ensure_dirs_exist()
    
    def _ensure_dirs_exist(self) -> None:
        """Ensure necessary directories exist"""
        self.params.result_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.params.result_dir / self.params.log_dir
        log_path.mkdir(exist_ok=True)
    
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
        cmd = [
            sys.executable,
            str(MIDAS_UTILS / 'ffGenerateZip.py'),
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
        with open(out_file, 'w') as out_f, open(err_file, 'w') as err_f:
            result = subprocess.run(' '.join(cmd), shell=True, stdout=out_f, stderr=err_f)
            
        # Check for errors
        if result.returncode != 0:
            with open(err_file, 'r') as f:
                err_content = f.read().strip()
            error_msg = f"ZIP generation failed for {data_file}\nError log:\n{err_content}"
            logger.error(error_msg)
            raise IntegrationError(error_msg)
        
        # Extract output ZIP filename
        with open(out_file, 'r') as f:
            lines = f.readlines()
            if lines and lines[-1].startswith('OutputZipName'):
                return Path(lines[-1].split()[1])
            else:
                error_msg = f"Could not find output ZIP filename in {out_file}"
                logger.error(error_msg)
                raise IntegrationError(error_msg)
    
    def convert_hdf_to_zarr(self, hdf_file: Path) -> Path:
        """
        Convert HDF5 file to Zarr ZIP format.
        
        Args:
            hdf_file: Input HDF5 file
            
        Returns:
            Path to output Zarr ZIP file
            
        Raises:
            IntegrationError: If conversion fails
        """
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
            
            # Verify the file was created successfully
            if not output_zip.exists() or output_zip.stat().st_size == 0:
                raise IntegrationError(f"Failed to create valid Zarr ZIP file: {output_zip}")
                
            return output_zip
        except Exception as e:
            error_msg = f"Failed to convert HDF5 file {hdf_file} to Zarr ZIP: {str(e)}"
            logger.error(error_msg)
            raise IntegrationError(error_msg) from e
    
    def save_as_matlab(self, zarr_file_path: Path) -> Path:
        """
        Save Zarr file as MATLAB .mat file
        
        Args:
            zarr_file_path: Path to Zarr file
            
        Returns:
            Path to the created .mat file
            
        Raises:
            IntegrationError: If saving fails
        """
        output_path = Path(f"{zarr_file_path}.mat")
        
        try:
            zarr_file = zarr.open(str(zarr_file_path), mode='r')
            
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
            logger.error(f"Failed to create MATLAB file for {zarr_file_path}: {str(e)}")
            # Don't raise here, just return None to allow processing to continue
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
        log_path = self.params.result_dir / self.params.log_dir
        out_log = log_path / f"{zip_file.name}_integrator_out.csv"
        err_log = log_path / f"{zip_file.name}_integrator_err.csv"
        
        integrator_path = MIDAS_BIN / 'IntegratorZarr'
        integrator_cmd = f"{integrator_path} {zip_file}"
        
        with open(out_log, 'w') as f_out, open(err_log, 'w') as f_err:
            result = subprocess.run(integrator_cmd, shell=True, stdout=f_out, stderr=f_err)
        
        # Check for errors
        if result.returncode != 0:
            with open(err_log, 'r') as f:
                err_content = f.read().strip()
            error_msg = f"IntegratorZarr failed for {zip_file}\nError log:\n{err_content}"
            logger.error(error_msg)
            raise IntegrationError(error_msg)
        
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
        log_path = self.params.result_dir / self.params.log_dir
        out_log = log_path / "map_out.csv"
        err_log = log_path / "map_err.csv"
        
        mapper_path = MIDAS_BIN / 'DetectorMapperZarr'
        mapper_cmd = f"{mapper_path} {zip_file}"
        
        with open(out_log, 'w') as f_out, open(err_log, 'w') as f_err:
            result = subprocess.run(mapper_cmd, shell=True, stdout=f_out, stderr=f_err)
        
        # Check for errors
        if result.returncode != 0:
            with open(err_log, 'r') as f:
                err_content = f.read().strip()
            error_msg = f"DetectorMapperZarr failed for {zip_file}\nError log:\n{err_content}"
            logger.error(error_msg)
            raise IntegrationError(error_msg)
    
    def process_file(self, file_nr: int, start_file_nr: int) -> Optional[Path]:
        """
        Process a single data file.
        
        Args:
            file_nr: File number index
            start_file_nr: Starting file number
            
        Returns:
            Path to output Zarr ZIP file or None if file doesn't exist
        """
        # Generate filename for this file number
        file_path = self._get_file_path(file_nr, start_file_nr)
        
        # Check if the file exists
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}. Skipping.")
            return None
        
        logger.info(f"Processing file {file_path}")
        
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
        out_zip = self.convert_hdf_to_zarr(hdf_file)
        
        # Save as MATLAB file if requested
        if self.params.write_mat:
            self.save_as_matlab(out_zip)
        
        return out_zip


class MidasIntegrator:
    """Main integrator class that orchestrates the data processing"""
    
    def __init__(self):
        """Initialize the integrator"""
        self.args = self._parse_arguments()
        self.params = self._create_processing_parameters()
        self.processor = FileProcessor(self.params)
        self.completed_files = set()
    
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments"""
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
        parser.add_argument('-writeMat', type=int, default=0, 
                          help='If you want to write a matlab .mat file.')
        parser.add_argument('-skipExisting', action='store_true',
                          help='Skip processing of files that have already been processed.')
        
        return parser.parse_args()
    
    def _create_processing_parameters(self) -> ProcessingParameters:
        """Create processing parameters from command line arguments"""
        result_dir = Path(self.args.resultFolder)
        if str(result_dir) == '.' or not result_dir.is_absolute():
            result_dir = Path.cwd() / self.args.resultFolder
        
        return ProcessingParameters(
            result_dir=result_dir,
            param_file=Path(self.args.paramFN),
            input_file_pattern=self.args.dataFN,
            dark_file=Path(self.args.darkFN) if self.args.darkFN else None,
            data_loc=self.args.dataLoc,
            dark_loc=self.args.darkLoc,
            frame_chunks=self.args.numFrameChunks,
            pre_proc=self.args.preProcThresh,
            convert_files=bool(self.args.convertFiles),
            write_mat=bool(self.args.writeMat),
            skip_existing=bool(self.args.skipExisting),
        )
    
    def _find_start_file_number(self) -> int:
        """Find the start file number from dataFN if not provided"""
        if self.args.startFileNr != -1:
            return self.args.startFileNr
            
        match = re.search(r'\d{6}', self.args.dataFN)
        if not match:
            logger.error("Could not find 6 padded fileNr in dataFN.")
            raise ValueError("Could not find 6 padded fileNr in dataFN.")
            
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
    
    def run(self) -> None:
        """Run the integration process"""
        try:
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
                    logger.error("No existing files found to process!")
                    return
            
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
            if self.args.nCPUs == 1:
                # Serial processing
                logger.info("Processing files serially")
                results = []
                
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
                            
                    except IntegrationError as e:
                        logger.error(f"Error processing file {file_num}: {str(e)}")
                        # Update progress bar even on error
                        if TQDM_AVAILABLE:
                            progress_bar.update(1)
                
                # Close progress bar if using tqdm
                if TQDM_AVAILABLE:
                    progress_bar.close()
            else:
                # Parallel processing
                logger.info(f"Processing files in parallel using {self.args.nCPUs} CPUs")
                results = []
                
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
                    else:
                        # Without tqdm, just log each completion
                        processed_count = 0
                        for future in as_completed(futures):
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
            
            # Print summary of results
            logger.info(f"Successfully processed {len(results)} of {len(files_to_process)} files")
            # for out_zip in results:
            #     try:
            #         logger.info(f'Output file {out_zip} tree structure:')
            #         logger.info(zarr.open(str(out_zip)).tree())
            #     except Exception as e:
            #         logger.error(f"Error displaying tree structure for {out_zip}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"An error occurred during processing: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        integrator = MidasIntegrator()
        integrator.run()
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        sys.exit(1)