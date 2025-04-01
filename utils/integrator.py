#!/usr/bin/env python3


import argparse
import os
import re
import shutil
import subprocess
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import fsspec
import zarr
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
import scipy

# Path configuration
MIDAS_HOME = os.path.expanduser("~/.MIDAS")
MIDAS_UTILS = os.path.expanduser('~/opt/MIDAS/utils/')
MIDAS_BIN = os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin")

# Add MIDAS utils to path
sys.path.insert(0, MIDAS_UTILS)
from midas2zip import Hdf5ToZarr

# Environment setup
def setup_environment():
    """Configure environment variables for MIDAS executables"""
    env = dict(os.environ)
    lib_paths = [
        f'{MIDAS_HOME}/BLOSC/lib64',
        f'{MIDAS_HOME}/FFTW/lib',
        f'{MIDAS_HOME}/HDF5/lib',
        f'{MIDAS_HOME}/LIBTIFF/lib',
        f'{MIDAS_HOME}/LIBZIP/lib64',
        f'{MIDAS_HOME}/NLOPT/lib',
        f'{MIDAS_HOME}/ZLIB/lib',
    ]
    
    # Preserve existing LD_LIBRARY_PATH if set
    existing_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
    if existing_lib_path:
        lib_paths.append(existing_lib_path)
    
    env['LD_LIBRARY_PATH'] = ':'.join(lib_paths)
    return env


class CustomArgumentParser(argparse.ArgumentParser):
    """Custom argument parser with improved error handling"""
    def error(self, message):
        sys.stderr.write(f'Error: {message}\n')
        self.print_help()
        sys.exit(2)


def generate_zip(result_dir, param_file, data_file='', dark_file='', 
                data_loc='', dark_loc='', frame_chunks=-1, pre_proc=-1,
                log_dir='stdout'):
    """
    Generate ZIP file from data.
    
    Args:
        result_dir: Directory to store results
        param_file: Parameter file
        data_file: Input data file
        dark_file: Dark field file
        data_loc: Data location within file
        dark_loc: Dark field location within file
        frame_chunks: Number of frame chunks to process
        pre_proc: Pre-processing threshold
        log_dir: Directory for log files
        
    Returns:
        Path to generated ZIP file
    
    Raises:
        SystemExit: If errors are encountered during ZIP generation
    """
    cmd = [
        sys.executable,
        os.path.join(MIDAS_UTILS, 'ffGenerateZip.py'),
        '-resultFolder', result_dir,
        '-paramFN', param_file
    ]
    
    if dark_file:
        cmd.extend(['-darkFN', dark_file])
    if data_file:
        cmd.extend(['-dataFN', data_file])
    if data_loc:
        cmd.extend(['-dataLoc', data_loc])
    if dark_loc:
        cmd.extend(['-darkLoc', dark_loc])
    if frame_chunks != -1:
        cmd.extend(['-numFrameChunks', str(frame_chunks)])
    if pre_proc != -1:
        cmd.extend(['-preProcThresh', str(pre_proc)])
    
    # Output file paths
    data_basename = os.path.basename(data_file)
    out_file = f'{result_dir}/{log_dir}/{data_basename}_ZipOut.txt'
    err_file = f'{result_dir}/{log_dir}/{data_basename}_ZipErr.txt'
    
    # Run command
    with open(out_file, 'w') as out_f, open(err_file, 'w') as err_f:
        subprocess.run(' '.join(cmd), shell=True, stdout=out_f, stderr=err_f)
    
    # Check for errors
    with open(err_file, 'r') as f:
        err_content = f.read().strip()
        if err_content:
            print(f"ERROR: ZIP generation failed for {data_file}")
            print(f"Error log contents from {err_file}:")
            print("-" * 40)
            print(err_content)
            print("-" * 40)
            sys.exit(1)
    
    # Extract output ZIP filename
    with open(out_file, 'r') as f:
        lines = f.readlines()
        if lines and lines[-1].startswith('OutputZipName'):
            return lines[-1].split()[1]
        else:
            print(f"ERROR: Could not find output ZIP filename in {out_file}")
            sys.exit(1)
    
    return None


def convert_hdf_to_zarr(hdf_file, output_zip):
    """
    Convert HDF5 file to Zarr ZIP format.
    
    Args:
        hdf_file: Input HDF5 file
        output_zip: Output Zarr ZIP file
        
    Returns:
        Path to output Zarr ZIP file
        
    Raises:
        Exception: If any error occurs during the conversion process
    """
    # Backup existing file if it exists
    zip_path = Path(output_zip)
    if zip_path.exists():
        try:
            shutil.move(output_zip, f"{output_zip}.old")
        except Exception as e:
            print(f"WARNING: Could not backup existing file {output_zip}: {str(e)}")
    
    try:
        # Convert HDF5 to Zarr
        with fsspec.open(hdf_file, mode='rb', anon=False, requester_pays=True, default_fill_cache=False) as f:
            with zarr.ZipStore(output_zip) as store_zip:
                h5_chunks_zip = Hdf5ToZarr(f, store_zip)
                h5_chunks_zip.translate()
        
        # Verify the file was created successfully
        if not os.path.exists(output_zip) or os.path.getsize(output_zip) == 0:
            raise Exception(f"Failed to create valid Zarr ZIP file: {output_zip}")
            
        return output_zip
    except Exception as e:
        print(f"ERROR: Failed to convert HDF5 file {hdf_file} to Zarr ZIP {output_zip}")
        print(f"Exception: {str(e)}")
        raise


def process_single_file(args_tuple):
    """
    Process a single data file with parameters passed as a tuple.
    
    Args:
        args_tuple: Tuple containing all the arguments
        
    Returns:
        Path to output Zarr ZIP file
    """
    # Unpack arguments
    (file_nr, input_file_pattern, start_file_nr, result_dir, 
     param_file, frame_chunks, pre_proc, convert_files, 
     env, data_loc, dark_loc, dark_file, log_dir, write_mat) = args_tuple
    
    # Generate filename for this file number
    padded_file_nr = str(start_file_nr + file_nr).zfill(6)
    start_file_nr_str = str(start_file_nr).zfill(6)
    this_file = input_file_pattern.replace(start_file_nr_str, padded_file_nr)
    
    # Generate ZIP if needed
    if file_nr > 0 and convert_files == 1:
        zip_file = generate_zip(result_dir, param_file, data_file=this_file, 
                               dark_file=dark_file, data_loc=data_loc, 
                               dark_loc=dark_loc, frame_chunks=frame_chunks, 
                               pre_proc=pre_proc, log_dir=log_dir)
    else:
        if not this_file.endswith('.zip'):
            this_file += '.analysis.MIDAS.zip'
        zip_file = os.path.join(result_dir, os.path.basename(this_file))
        print(f'Processing file: {zip_file}')
    
    # Run integrator
    out_log = f'{result_dir}/{log_dir}/{os.path.basename(zip_file)}_integrator_out.csv'
    err_log = f'{result_dir}/{log_dir}/{os.path.basename(zip_file)}_integrator_err.csv'
    
    with open(out_log, 'w') as f_out, open(err_log, 'w') as f_err:
        integrator_cmd = f"{os.path.join(MIDAS_BIN, 'IntegratorZarr')} {zip_file}"
        subprocess.run(integrator_cmd, shell=True, env=env, stdout=f_out, stderr=f_err)
    
    # Check for errors in integrator
    with open(err_log, 'r') as f:
        err_content = f.read().strip()
        if err_content:
            print(f"ERROR: IntegratorZarr failed for {zip_file}")
            print(f"Error log contents from {err_log}:")
            print("-" * 40)
            print(err_content)
            print("-" * 40)
            sys.exit(1)
    
    # Convert HDF to Zarr
    final_file = f'{zip_file}.caked.hdf'
    
    # Check if the HDF file was created
    if not os.path.exists(final_file):
        print(f"ERROR: The expected HDF file {final_file} was not created.")
        print(f"Check the integrator output log: {out_log}")
        sys.exit(1)
        
    out_zip = f'{final_file}.zarr.zip'
    
    try:
        out_zip = convert_hdf_to_zarr(final_file, out_zip)
    except Exception as e:
        print(f"ERROR: Failed to convert HDF to Zarr for {final_file}")
        print(f"Exception: {str(e)}")
        sys.exit(1)
    
    # Save as MATLAB file if requested
    if write_mat:
        try:
            zarr_file = zarr.open(out_zip, mode='r')
            
            # Create a dictionary with proper handling of zarr arrays
            data_dict = {}
            for key in zarr_file.keys():
                try:
                    # Handle different kinds of zarr arrays with appropriate slicing
                    if isinstance(zarr_file[key], zarr.core.Array):
                        # Get the array shape to create proper slice objects
                        shape = zarr_file[key].shape
                        if shape:  # Only try to slice if there's actually data
                            data_dict[key] = zarr_file[key][...]  # Use [...] instead of [:]
                        else:
                            # For empty arrays, create an empty numpy array with the right shape
                            data_dict[key] = np.array([])
                    else:
                        # For zarr groups, store metadata about the group
                        data_dict[key] = f"Group: {str(zarr_file[key])}"
                except Exception as inner_e:
                    print(f"WARNING: Could not extract key '{key}' from zarr file: {str(inner_e)}")
                    # Continue with other keys instead of failing
            
            # Save as .mat file
            scipy.io.savemat(f"{out_zip}.mat", data_dict)
            print(f"Successfully created MATLAB file: {out_zip}.mat")
        except Exception as e:
            print(f"ERROR: Failed to create MATLAB file for {out_zip}")
            print(f"Exception: {str(e)}")
            # Don't exit on this error, just continue without the .mat file
            print("Continuing without creating the MATLAB file.")
    
    return out_zip


def run_detector_mapper(zip_file, env, result_dir, log_dir='stdout'):
    """
    Run detector mapper on ZIP file.
    
    Args:
        zip_file: Input ZIP file
        env: Environment variables
        result_dir: Directory to store results
        log_dir: Directory for log files
        
    Raises:
        SystemExit: If errors are encountered during detector mapping
    """
    out_log = f'{result_dir}/{log_dir}/map_out.csv'
    err_log = f'{result_dir}/{log_dir}/map_err.csv'
    
    with open(out_log, 'w') as f_out, open(err_log, 'w') as f_err:
        mapper_cmd = f"{os.path.join(MIDAS_BIN, 'DetectorMapperZarr')} {zip_file}"
        subprocess.run(mapper_cmd, shell=True, env=env, stdout=f_out, stderr=f_err)
    
    # Check for errors in detector mapper
    with open(err_log, 'r') as f:
        err_content = f.read().strip()
        if err_content:
            print(f"ERROR: DetectorMapperZarr failed for {zip_file}")
            print(f"Error log contents from {err_log}:")
            print("-" * 40)
            print(err_content)
            print("-" * 40)
            sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = CustomArgumentParser(
        description='Code to integrate files. Contact: hsharma@anl.gov',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-resultFolder', type=str, default='.', 
                       help='Folder where you want to save results.')
    parser.add_argument('-paramFN', type=str, required=True, 
                       help='Parameter file name.')
    parser.add_argument('-dataFN', type=str, required=True, default='', 
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
    
    return parser.parse_args()


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Normalize paths
    result_dir = args.resultFolder
    if len(result_dir) == 0 or result_dir == '.':
        result_dir = os.getcwd()
    if not result_dir.startswith('/'):
        result_dir = os.path.join(os.getcwd(), result_dir)
    result_dir = os.path.join(result_dir, '')  # Ensure trailing slash
    
    # Create necessary directories
    log_dir = 'stdout'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, log_dir), exist_ok=True)
    
    # Setup environment
    env = setup_environment()
    
    # Find start file number
    start_file_nr = args.startFileNr
    start_file_nr_str = str(start_file_nr).zfill(6)
    
    if start_file_nr == -1:
        match = re.search(r'\d{6}', args.dataFN)
        if not match:
            print("Could not find 6 padded fileNr. Exiting.")
            sys.exit(1)
        start_file_nr_str = match.group(0)
        start_file_nr = int(start_file_nr_str)
        print(f'Processing file number: {start_file_nr}')
    
    # Determine end file number
    end_file_nr = args.endFileNr
    if end_file_nr == -1:
        end_file_nr = start_file_nr
    
    # Calculate number of files to process
    nr_files = end_file_nr - start_file_nr + 1
    
    # Process first file
    file_nr = 0
    this_file = args.dataFN.replace(start_file_nr_str, str(start_file_nr + file_nr).zfill(6))
    
    if args.convertFiles == 1:
        zip_file = generate_zip(
            result_dir, args.paramFN, data_file=this_file, dark_file=args.darkFN,
            data_loc=args.dataLoc, dark_loc=args.darkLoc, 
            frame_chunks=args.numFrameChunks, pre_proc=args.preProcThresh,
            log_dir=log_dir
        )
    else:
        if not this_file.endswith('.zip'):
            this_file += '.analysis.MIDAS.zip'
        zip_file = os.path.join(result_dir, this_file)
        print(f'Processing file: {zip_file}')
    
    # Run detector mapper if requested
    if args.mapDetector == 1:
        run_detector_mapper(zip_file, env, result_dir, log_dir)
    
    # Process all files
    if args.nCPUs == 1:
        # Serial processing
        for file_nr in range(nr_files):
            # Create arguments tuple for process_single_file
            process_args = (
                file_nr, args.dataFN, start_file_nr, result_dir, args.paramFN,
                args.numFrameChunks, args.preProcThresh, args.convertFiles, env,
                args.dataLoc, args.darkLoc, args.darkFN, log_dir, args.writeMat
            )
            out_zip = process_single_file(process_args)
            print(f'Output file {out_zip} tree structure:')
            print(zarr.open(out_zip).tree())
    else:
        # Parallel processing
        print(f"Starting {args.nCPUs} parallel jobs.")
        
        # Create a list of argument tuples for each file to process
        process_args_list = []
        for file_nr in range(nr_files):
            process_args = (
                file_nr, args.dataFN, start_file_nr, result_dir, args.paramFN,
                args.numFrameChunks, args.preProcThresh, args.convertFiles, env,
                args.dataLoc, args.darkLoc, args.darkFN, log_dir, args.writeMat
            )
            process_args_list.append(process_args)
        
        # Use Pool to process files in parallel
        with Pool(args.nCPUs) as pool:
            results = pool.map(process_single_file, process_args_list)
            
            for out_zip in results:
                print(f'Output file {out_zip} tree structure:')
                print(zarr.open(out_zip).tree())


if __name__ == "__main__":
    main()