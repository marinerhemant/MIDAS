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


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure an argument parser for diffraction processing.
    
    Returns:
    --------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Process diffraction images')
    
    # Required arguments
    parser.add_argument('--image', dest='image_path', help='Path to the diffraction image')
    
    # Configuration file
    parser.add_argument('--config', dest='config_file', help='Path to JSON configuration file')
    
    # Optional arguments
    parser.add_argument("--dark", help="Path to the dark image for background subtraction")
    parser.add_argument("--map", help="Path to the pixel map binary file")
    parser.add_argument("--nmap", help="Path to the pixel count map binary file")
    parser.add_argument("--rmin", type=float, help="Minimum radius for binning")
    parser.add_argument("--rmax", type=float, help="Maximum radius for binning")
    parser.add_argument("--rbin", type=float, help="Size of each radial bin")
    parser.add_argument("--etamin", type=float, help="Minimum azimuthal angle")
    parser.add_argument("--etamax", type=float, help="Maximum azimuthal angle")
    parser.add_argument("--etabin", type=float, help="Size of each azimuthal bin")
    parser.add_argument("--badpx", type=float, help="Value that marks bad pixels")
    parser.add_argument("--gappx", type=float, help="Value that marks gap pixels")
    parser.add_argument("--output", dest='output_file', help="Path to save the plot, disabled if not provided")
    
    # Advanced options
    parser.add_argument("--save-config", help="Save parameters to a JSON config file")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument('--save-data', dest='save_data', help='Path to save the integrated data')
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--cpu', dest='use_gpu', action='store_false', help='Force CPU processing')
    parser.add_argument('--peaks', dest='num_peaks', type=int, help='Number of peaks to fit')
    parser.add_argument('--cache', dest='cache_results', action='store_true', help='Cache integration results')
    parser.add_argument('--benchmark', dest='benchmark', action='store_true', help='Run performance benchmark')
    
    return parser

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

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("diffraction-cli")

    # Set up argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Check for config file or image path
    if not args.config_file and not args.image_path:
        parser.error("Either --config or --image must be specified")
    
    # Create configuration
    try:
        config = DiffractionConfig.from_args_and_config(args, args.config_file)
        logger.info("Created configuration successfully")
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        sys.exit(1)
    
    # Display final configuration
    logger.debug("Final configuration:")
    for key, value in config.to_dict().items():
        logger.debug(f"  {key}: {value}")

    # Ensure we have an image path
    if not config.image_path:
        parser.error("Image path must be specified either in config file or as --image argument")
    
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