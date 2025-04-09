#!/usr/bin/env python
"""
FitGrain script to find tx parameter for grain fitting.
This script processes grain data, sorts by specified properties,
and optimizes a subset of grains for refinement.
"""

import os
import sys
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from multiprocessing import Pool
import argparse
import pathlib
import multiprocessing
import shutil
import logging
import time
import datetime

# Determine installation path (one directory up from script location)
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
INSTALL_DIR = SCRIPT_DIR.parent
UTILS_DIR = os.path.join(INSTALL_DIR, 'utils')
sys.path.insert(0, str(UTILS_DIR))

# Configure logging
def setup_logging(log_file=None):
    """Set up logging configuration with console and file handlers."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    # Remove any existing handlers (in case this function is called multiple times)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
    
    return root_logger

class MyParser(argparse.ArgumentParser):
    """Custom argument parser with improved error handling."""
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

# Global variables
IDs = []

def check_dependencies():
    """
    Check if required dependencies and binaries exist.
    
    Returns:
        bool: True if all dependencies are found, False otherwise
    """
    dependencies_ok = True
    
    # Check for FitGrain executable
    fit_grain_bin = os.path.join(INSTALL_DIR, 'FF_HEDM/bin/FitGrain')
    if not os.path.isfile(fit_grain_bin):
        print(f"Error: FitGrain executable not found at {fit_grain_bin}")
        dependencies_ok = False
    
    # Check for GetHKLList executable
    get_hkl_bin = os.path.join(INSTALL_DIR, 'FF_HEDM/bin/GetHKLList')
    if not os.path.isfile(get_hkl_bin):
        print(f"Error: GetHKLList executable not found at {get_hkl_bin}")
        dependencies_ok = False
    
    # Check for numpy and matplotlib
    try:
        import numpy
        import matplotlib.pyplot
    except ImportError as e:
        print(f"Error: Missing Python dependencies: {str(e)}")
        dependencies_ok = False
    
    return dependencies_ok

def run_fit_grain_one(work_data):
    """
    Run FitGrain on a single grain.
    
    Args:
        work_data: List containing [grain_id, folder_path, ps_file_name]
        
    Returns:
        float: tx value or np.nan if not found
    """
    grain_id = work_data[0]
    folder = work_data[1]
    ps_file = work_data[2]
    
    fit_grain_bin = os.path.join(INSTALL_DIR, 'FF_HEDM/bin/FitGrain')
    output_file = f'{folder}/fitGrain/fitGrainOut{grain_id}.txt'
    
    # Check if FitGrain executable exists
    if not os.path.isfile(fit_grain_bin):
        logging.error(f"FitGrain executable not found at {fit_grain_bin}")
        return np.nan
    
    try:
        cmd = f'{fit_grain_bin} {folder} {ps_file} {grain_id}'
        logging.debug(f"Running command: {cmd}")
        
        start_time = time.time()
        process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, text=True)
        elapsed = time.time() - start_time
        
        # Check if the process executed successfully
        if process.returncode != 0:
            logging.error(f"Grain {grain_id}: FitGrain failed with code {process.returncode} (in {elapsed:.2f}s)")
            logging.error(f"Error message: {process.stderr}")
            return np.nan
            
        # Write the output to file
        with open(output_file, 'w') as out_f:
            out_f.write(process.stdout)
        
        # Check if output file exists and has content
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logging.error(f"Output file for grain {grain_id} is empty or missing")
            return np.nan
            
        # Parse the output to extract tx value
        tx_found = False
        with open(output_file, 'r') as in_f:
            for line in in_f:
                if line.startswith('Input tx:'):
                    tx_found = True
                    try:
                        tx_value = float(line.rstrip().split()[-1])
                        logging.debug(f"Grain {grain_id}: Found tx = {tx_value} (in {elapsed:.2f}s)")
                        return tx_value
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error parsing tx value for grain {grain_id}: {str(e)}")
                        return np.nan
        
        if not tx_found:
            logging.error(f"'Input tx:' not found in output for grain {grain_id}")
            
    except (subprocess.SubprocessError, IOError) as e:
        logging.error(f"Error processing grain {grain_id}: {str(e)}")
    
    return np.nan

def main():
    """Main function to run the grain fitting process."""
    global IDs
    
    # Create timestamp for log file and results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parser = MyParser(
        description='Fit Grains to find tx, contact hsharma@anl.gov',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-resultFolder', type=str, required=True, 
                        help='Folder where the results live')
    parser.add_argument('-psFN', type=str, required=True, 
                        help='ps file that will be used. This is temporary. Will be replaced by zarr soon.')
    parser.add_argument('-fractionGrains', type=int, required=False, default=20, 
                        help='1/fractionGrains will be used optimized. Typically ~20.')
    parser.add_argument('-numProcs', type=int, required=False, default=8, 
                        help='Number of processors to use.')
    parser.add_argument('-propertyToSortGrains', type=str, required=False, default='DiffOme', 
                        help='Which property to choose to sort grains, choose either DiffOme or DiffPos.')
    parser.add_argument('-debug', action='store_true', 
                        help='Enable debug logging')
    parser.add_argument('-noplot', action='store_true',
                        help='Disable plotting (useful for batch processing)')
    
    args, unparsed = parser.parse_known_args()
    folder = args.resultFolder
    ps_file = args.psFN
    frac = args.fractionGrains
    num_procs = args.numProcs
    sort_property = args.propertyToSortGrains
    debug_mode = args.debug
    no_plot = args.noplot
    
    # Set up logging with a log file in the output directory
    os.makedirs(f'{folder}/fitGrain/', exist_ok=True)
    log_file = f'{folder}/fitGrain/ff_fitgrain_{timestamp}.log'
    log_level = logging.DEBUG if debug_mode else logging.INFO
    setup_logging(log_file)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    logging.info(f"Starting ff_fitgrain.py (version 2.0)")
    logging.info(f"Log file: {log_file}")
    
    # Record command line arguments
    logging.info(f"Arguments: {vars(args)}")
    
    # Check dependencies before proceeding
    logging.info("Checking dependencies...")
    if not check_dependencies():
        logging.error("Missing required dependencies. Please check installation.")
        sys.exit(1)
    
    # Check if result folder exists
    if not os.path.isdir(folder):
        logging.error(f"Result folder '{folder}' does not exist")
        sys.exit(1)
    
    # Check if ps file exists
    if not os.path.isfile(ps_file):
        logging.error(f"PS file '{ps_file}' does not exist")
        sys.exit(1)
        
    # Validate fraction value
    if frac <= 0:
        logging.error(f"fractionGrains must be positive, got {frac}")
        sys.exit(1)
        
    # Check and adjust numProcs
    available_cpus = multiprocessing.cpu_count()
    if num_procs > available_cpus:
        logging.warning(f"Requested {num_procs} processors, but only {available_cpus} are available.")
        logging.warning(f"Adjusting to use {available_cpus} processors.")
        num_procs = available_cpus
    
    # Create timeout handler for long-running processes
    def timeout_handler(timeout=300):  # 5 minutes default timeout
        def decorator(func):
            def wrapper(*args, **kwargs):
                import signal
                
                def handle_timeout(signum, frame):
                    raise TimeoutError(f"Process timed out after {timeout} seconds")
                
                # Set timeout handler
                original_handler = signal.signal(signal.SIGALRM, handle_timeout)
                signal.alarm(timeout)
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    # Reset timeout handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, original_handler)
                
                return result
            return wrapper
        return decorator
    
    # Define output paths
    fit_grain_dir = f'{folder}/fitGrain/'
    grains_file = os.path.join(folder, 'Grains.csv')
    hkls_output = os.path.join(folder, 'hklsout.csv')
    results_file = os.path.join(folder, f'tx_results_{timestamp}.csv')
    plot_file = os.path.join(folder, f'tx_values_{timestamp}.png')
    
    # Record started time for performance tracking
    start_time_total = time.time()
    
    # Change to result folder
    original_dir = os.getcwd()
    os.chdir(folder)
    logging.info(f"Changed working directory to: {folder}")
    
    # Determine which column to use for sorting
    if sort_property == 'DiffOme':
        col_val = 20
    elif sort_property == 'DiffPos':
        col_val = 19
    else:  # Default to DiffOme
        col_val = 20
        logging.warning(f"Unknown property '{sort_property}', defaulting to DiffOme (column {col_val})")

    # Generate HKL list
    try:
        get_hkl_bin = os.path.join(INSTALL_DIR, 'FF_HEDM/bin/GetHKLList')
        logging.info(f"Running GetHKLList: {get_hkl_bin} {ps_file}")
        
        start_time = time.time()
        process = subprocess.run(f'{get_hkl_bin} {ps_file}', shell=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed = time.time() - start_time
        logging.info(f"GetHKLList completed in {elapsed:.2f} seconds")
        
        # Check if GetHKLList executed successfully
        if process.returncode != 0:
            logging.error(f"GetHKLList failed with code {process.returncode}")
            logging.error(f"Error message: {process.stderr}")
            os.chdir(original_dir)
            sys.exit(1)
            
        # Write output to file
        with open(hkls_output, 'w') as hkl_out:
            hkl_out.write(process.stdout)
            
        # Verify the output file has content
        if not os.path.exists(hkls_output) or os.path.getsize(hkls_output) == 0:
            logging.error(f"HKL output file is empty or missing")
            os.chdir(original_dir)
            sys.exit(1)
            
        logging.info(f"HKL list generated and saved to {hkls_output}")
            
    except subprocess.SubprocessError as e:
        logging.error(f"Error generating HKL list: {str(e)}")
        os.chdir(original_dir)
        sys.exit(1)
    
    # Read and sort grains
    try:
        # Check if Grains.csv exists
        if not os.path.exists(grains_file):
            logging.error(f"Grains file '{grains_file}' does not exist")
            os.chdir(original_dir)
            sys.exit(1)
            
        # Check if file has content
        if os.path.getsize(grains_file) == 0:
            logging.error(f"Grains file '{grains_file}' is empty")
            os.chdir(original_dir)
            sys.exit(1)
            
        # Read grain data
        logging.info(f"Reading grain data from {grains_file}")
        grains = np.genfromtxt(grains_file, skip_header=9)
        
        # Check if data was loaded correctly
        if grains.size == 0:
            logging.error(f"No grain data found in '{grains_file}'")
            os.chdir(original_dir)
            sys.exit(1)
            
        # Check if column index is valid
        if grains.ndim < 2 or col_val >= grains.shape[1]:
            logging.error(f"Invalid column index {col_val} for sorting. File has fewer columns.")
            os.chdir(original_dir)
            sys.exit(1)
            
        # Sort grains
        logging.info(f"Sorting grains by column {col_val} ({sort_property})")
        grains_sorted = grains[grains[:, col_val].argsort()]
        n_grains = grains_sorted.shape[0]
        logging.info(f"Found {n_grains} grains in total")
        
        # Calculate number of grains to refine
        n_spots_to_refine = n_grains // frac  # We will take 1/nth of grains and refine them
        
        # Check if we have enough grains to process
        if n_spots_to_refine <= 0:
            logging.error(f"Not enough grains to process with fraction {frac}")
            os.chdir(original_dir)
            sys.exit(1)
            
        logging.info(f"Will process {n_spots_to_refine} grains (1/{frac} of total)")
            
    except (IOError, ValueError) as e:
        logging.error(f"Error reading or processing grain file: {str(e)}")
        os.chdir(original_dir)
        sys.exit(1)
    
    # Prepare work data for multiprocessing
    IDs = [int(grain_id) for grain_id in grains_sorted[:n_spots_to_refine, 0]]
    work_data = [[grain_id, folder, ps_file] for grain_id in IDs]
    tx_values = []
    
    # Process grains in parallel
    try:
        logging.info(f"Processing {n_spots_to_refine} grains using {num_procs} processors...")
        
        # Create progress tracking variables
        successful_fits = 0
        failed_fits = 0
        start_time_mp = time.time()
        
        with Pool(processes=num_procs) as pool:
            # Process grains with progress reporting
            results = []
            
            # Process all grains
            results = pool.map(run_fit_grain_one, work_data)
            tx_values = list(results)
            
        # Calculate elapsed time
        elapsed_mp = time.time() - start_time_mp
        processing_rate = n_spots_to_refine / elapsed_mp
            
        # Count successful and failed fits
        for val in tx_values:
            if np.isnan(val):
                failed_fits += 1
            else:
                successful_fits += 1
                
        # Report statistics
        logging.info(f"Parallel processing completed in {elapsed_mp:.2f} seconds")
        logging.info(f"Processing rate: {processing_rate:.2f} grains/second")
        logging.info(f"Results: {successful_fits} successful fits, {failed_fits} failed fits")
        
        # Warning if too many failures
        failure_rate = failed_fits / n_spots_to_refine if n_spots_to_refine > 0 else 0
        if failure_rate > 0.3:  # More than 30% failed
            logging.warning(f"High failure rate ({failed_fits}/{n_spots_to_refine}, {failure_rate:.1%}) in grain fitting")
            if failure_rate > 0.7:  # More than 70% failed - critical warning
                logging.error("CRITICAL: Most grain fits failed! Check FitGrain executable and input parameters")
            
    except Exception as e:
        logging.error(f"Error during parallel processing: {str(e)}")
        os.chdir(original_dir)
        sys.exit(1)
    
    # Convert to numpy array for analysis
    tx_values = np.array(tx_values)
    valid_values = tx_values[~np.isnan(tx_values)]
    
    # Save raw data to a file for future reference
    try:
        with open(results_file, 'w') as f:
            f.write("# ff_fitgrain.py results generated on {}\n".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write("# Command line arguments: {}\n".format(" ".join(sys.argv)))
            f.write("# Grain ID, tx Value\n")
            for idx, grain_id in enumerate(IDs):
                f.write(f"{grain_id}, {tx_values[idx]}\n")
        logging.info(f"Raw results saved to {results_file}")
    except IOError as e:
        logging.warning(f"Could not save raw results to file: {str(e)}")
    
    if len(valid_values) > 0:
        # Statistical analysis
        mean_tx = np.mean(valid_values)
        median_tx = np.median(valid_values)
        std_tx = np.std(valid_values)
        min_tx = np.min(valid_values)
        max_tx = np.max(valid_values)
        
        logging.info('Statistical summary:')
        logging.info(f'  Mean tx: {mean_tx:.6f}')
        logging.info(f'  Median tx: {median_tx:.6f}')
        logging.info(f'  Std dev: {std_tx:.6f}')
        logging.info(f'  Range: {min_tx:.6f} to {max_tx:.6f}')
        
        # Create plot
        if not no_plot:
            try:
                plt.figure(figsize=(12, 8))
                plt.plot(valid_values, 'b.-', label='Valid tx values')
                
                # Add statistical indicators
                plt.axhline(y=mean_tx, color='r', linestyle='-', linewidth=2, label=f'Mean: {mean_tx:.6f}')
                plt.axhline(y=median_tx, color='g', linestyle='--', linewidth=2, label=f'Median: {median_tx:.6f}')
                
                # Add standard deviation lines
                plt.axhline(y=mean_tx + std_tx, color='orange', linestyle=':', linewidth=1.5, 
                            label=f'Mean + 1σ: {mean_tx + std_tx:.6f}')
                plt.axhline(y=mean_tx - std_tx, color='orange', linestyle=':', linewidth=1.5, 
                            label=f'Mean - 1σ: {mean_tx - std_tx:.6f}')
                
                plt.title(f'tx Values for Processed Grains - {os.path.basename(folder)}', fontsize=14)
                plt.xlabel('Grain Index', fontsize=12)
                plt.ylabel('tx Value', fontsize=12)
                plt.grid(True)
                plt.legend(loc='best', fontsize=10)
                
                # Add text with summary statistics
                stats_text = (f"Mean: {mean_tx:.6f}\n"
                              f"Median: {median_tx:.6f}\n"
                              f"Std Dev: {std_tx:.6f}\n"
                              f"Range: [{min_tx:.6f}, {max_tx:.6f}]\n"
                              f"Success Rate: {len(valid_values)}/{n_spots_to_refine} "
                              f"({len(valid_values)/n_spots_to_refine*100:.1f}%)")
                
                plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                            fontsize=10)
                
                # Save plot
                plt.tight_layout()
                plt.savefig(plot_file, dpi=300)
                logging.info(f"Plot saved to {plot_file}")
                
                # Only show plot if not in batch mode
                if not no_plot:
                    plt.show()
            except Exception as e:
                logging.error(f"Could not create plot: {str(e)}")
        else:
            logging.info("Plot generation skipped (--noplot option)")
    else:
        logging.error("No valid tx values were found. Check the FitGrain execution outputs.")
    
    # Display total runtime
    total_runtime = time.time() - start_time_total
    logging.info(f"Total runtime: {total_runtime:.2f} seconds")
    
    # Return to original directory
    os.chdir(original_dir)
    logging.info("Processing complete")
    
    return 0

if __name__ == "__main__":
    main()