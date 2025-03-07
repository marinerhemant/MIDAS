#!/usr/bin/env python3
"""
Diffraction Analyzer - Command Line Interface
--------------------------------------------
Command line tool for processing diffraction images, applying data integration,
and fitting Voigt profiles to the results.

This script provides a user-friendly interface to the diffraction image processing
functionality defined in the midas_integrator.core module.

Author: Hemant Sharma
Date: 2025/03/06
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, Optional
import time
import numpy as np
from pathlib import Path

# Import the core module
from midas_integrator.core import (
    DiffractionConfig, DiffractionProcessor, GPUUtils
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"


class ProgressBar:
    """
    Simple terminal progress bar.
    """
    def __init__(self, total_length: int = 50):
        """Initialize the progress bar."""
        self.total_length = total_length
        self.current_message = ""
        self.last_progress = -1
    
    def update(self, message: str, progress: float) -> None:
        """
        Update the progress bar.
        
        Parameters:
        -----------
        message : str
            Status message
        progress : float
            Progress as a fraction (0.0 to 1.0)
        """
        # Ensure progress is between 0 and 1
        progress = max(0, min(1, progress))
        
        # Only update if progress has changed significantly
        if progress - self.last_progress < 0.01 and message == self.current_message:
            return
        
        self.last_progress = progress
        self.current_message = message
        
        # Calculate completed length
        completed_length = int(self.total_length * progress)
        
        # Create the progress bar
        bar = "[" + "=" * completed_length + " " * (self.total_length - completed_length) + "]"
        
        # Format percentage
        percentage = int(progress * 100)
        
        # Print the progress bar
        sys.stdout.write(f"\r{message} {bar} {percentage}%")
        sys.stdout.flush()
        
        # Print newline if complete
        if progress >= 1.0:
            sys.stdout.write("\n")
            sys.stdout.flush()


def parse_args(args=None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Parameters:
    -----------
    args : list, optional
        Command line arguments to parse. If None, sys.argv[1:] is used.
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Process diffraction images and fit Voigt profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required parameters
    parser.add_argument("image_path", help="Path to the diffraction image")
    
    # Optional parameters with defaults
    parser.add_argument("--dark", help="Path to the dark image for background subtraction")
    parser.add_argument("--map", default="Map.bin", help="Path to the pixel map binary file")
    parser.add_argument("--nmap", default="nMap.bin", help="Path to the pixel count map binary file")
    parser.add_argument("--rmin", type=float, default=10, help="Minimum radius for binning")
    parser.add_argument("--rmax", type=float, default=100, help="Maximum radius for binning")
    parser.add_argument("--rbin", type=float, default=0.25, help="Size of each radial bin")
    parser.add_argument("--etamin", type=float, default=-180, help="Minimum azimuthal angle")
    parser.add_argument("--etamax", type=float, default=180, help="Maximum azimuthal angle")
    parser.add_argument("--etabin", type=float, default=1, help="Size of each azimuthal bin")
    parser.add_argument("--badpx", type=float, default=-1, help="Value that marks bad pixels")
    parser.add_argument("--gappx", type=float, default=-2, help="Value that marks gap pixels")
    parser.add_argument("--output", help="Path to save the plot, disabled if not provided")
    
    # Advanced options
    parser.add_argument("--peaks", type=int, default=1, help="Number of peaks to fit")
    parser.add_argument("--cpu", action="store_true", help="Force CPU processing (no GPU)")
    parser.add_argument("--benchmark", action="store_true", help="Run CPU vs GPU benchmark")
    parser.add_argument("--cache", action="store_true", help="Cache integration results")
    parser.add_argument("--config", help="Load parameters from a JSON config file")
    parser.add_argument("--save-config", help="Save parameters to a JSON config file")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    
    # New option to save integrated data
    parser.add_argument("--save-data", help="Path to save the integrated radius and intensity data (CSV format)")
    
    return parser.parse_args(args)


def config_from_args(args: argparse.Namespace) -> DiffractionConfig:
    """
    Create a configuration object from parsed arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns:
    --------
    DiffractionConfig
        Configuration for diffraction processing
    """
    # If a config file is provided, load it
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        config = DiffractionConfig.load(args.config)
        
        # Override with command line arguments where provided
        if args.image_path:
            config.image_path = args.image_path
        if args.dark:
            config.dark_path = args.dark
        if args.map:
            config.map_path = args.map
        if args.nmap:
            config.n_map_path = args.nmap
        if args.output:
            config.output_file = args.output
        if args.save_data:
            config.save_data_file = args.save_data
        
        # Override eta parameters if provided
        if args.etamin is not None:
            config.eta_min = args.etamin
        if args.etamax is not None:
            config.eta_max = args.etamax
        if args.etabin is not None:
            config.eta_bin_size = args.etabin
        
        return config
    
    # Otherwise, create a new config from the arguments
    return DiffractionConfig(
        image_path=args.image_path,
        dark_path=args.dark,
        map_path=args.map,
        n_map_path=args.nmap,
        r_min=args.rmin,
        r_max=args.rmax,
        r_bin_size=args.rbin,
        eta_min=args.etamin,
        eta_max=args.etamax,
        eta_bin_size=args.etabin,
        bad_px_intensity=args.badpx,
        gap_intensity=args.gappx,
        output_file=args.output,
        save_data_file=args.save_data,
        use_gpu=not args.cpu,
        num_peaks=args.peaks,
        cache_results=args.cache,
        benchmark=args.benchmark
    )


def save_integrated_data(data: np.ndarray, filepath: str) -> None:
    """
    Save integrated radius and intensity data to a CSV file.
    
    Parameters:
    -----------
    data : np.ndarray
        Integrated data with radius and intensity values
    filepath : str
        Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Save data
    header = "Radius,Intensity"
    np.savetxt(
        filepath, 
        data, 
        delimiter=',', 
        header=header, 
        fmt='%.6f',
        comments=''
    )
    
    logger.info(f"Integrated data saved to {filepath}")


def main(args=None):
    """
    Main function to run the program.
    
    Parameters:
    -----------
    args : list, optional
        Command line arguments to parse. If None, sys.argv[1:] is used.
        
    Returns:
    --------
    int
        Exit code (0 for success, 1 for failure)
    """
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_args(args)
    
    # Create configuration
    config = config_from_args(args)
    
    # Save configuration if requested
    if args.save_config:
        config.save(args.save_config)
        logger.info(f"Configuration saved to {args.save_config}")
    
    # Check GPU availability
    if config.use_gpu:
        if GPUUtils.cuda_available():
            GPUUtils.print_gpu_info()
        else:
            logger.warning("CUDA is not available. Using CPU processing.")
            config.use_gpu = False
    
    # Create processor
    processor = DiffractionProcessor(config)
    
    # Create progress bar
    progress_bar = None if args.no_progress else ProgressBar()
    
    # Define callback function for progress updates
    def progress_callback(message, progress):
        if progress_bar:
            progress_bar.update(message, progress)
    
    # Process the diffraction image
    try:
        result, params = processor.process(progress_callback)
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        
        # Print fitted parameters
        if config.num_peaks == 1:
            amp, bg, mix, cen, width = params
            logger.info(f"Fitted parameters: Amplitude={amp:.2f}, Background={bg:.2f}, "
                       f"Mix={mix:.2f}, Center={cen:.2f}, Width={width:.2f}")
        else:
            logger.info(f"Fitted {config.num_peaks} peaks:")
            for i in range(config.num_peaks):
                amp, bg, mix, cen, width = params[i*5:(i+1)*5]
                logger.info(f"  Peak {i+1}: Amplitude={amp:.2f}, Background={bg:.2f}, "
                           f"Mix={mix:.2f}, Center={cen:.2f}, Width={width:.2f}")
        
        # Save integrated data if requested
        if args.save_data:
            save_integrated_data(result, args.save_data)
        
        return 0
    except Exception as e:
        logger.error(f"Error processing diffraction image: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())