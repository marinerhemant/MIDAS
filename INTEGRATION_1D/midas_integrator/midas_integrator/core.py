#!/usr/bin/env python3
"""
Diffraction Image Processing and Analysis - Core Module
------------------------------------------------------
Core functionality for processing diffraction images, applying data integration,
and fitting Voigt profiles to the results.

This module provides the classes and functions necessary for working with
diffraction images from 2D detectors. It handles data loading, integration,
and profile fitting with both CPU and GPU acceleration options.

Author: Hemant Sharma
Date: 2025/03/06
"""

import os
import time
import numpy as np
from numba import njit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image
import struct
from math import ceil
from typing import Tuple, Optional, List, Dict, Any, Union, Callable
import logging
import json
from dataclasses import dataclass, asdict
from functools import lru_cache

try:
    from numba import cuda
    CUDA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUDA_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DiffractionConfig:
    """Configuration class for diffraction image processing parameters."""
    image_path: str
    dark_path: Optional[str] = None
    map_path: str = 'Map.bin'
    n_map_path: str = 'nMap.bin'
    r_min: float = 10.0
    r_max: float = 100.0
    r_bin_size: float = 0.25
    eta_min: float = -180.0
    eta_max: float = 180.0
    eta_bin_size: float = 1.0
    bad_px_intensity: float = -1.0
    gap_intensity: float = -2.0
    output_file: Optional[str] = None
    save_data_file: Optional[str] = None
    use_gpu: bool = True
    num_peaks: int = 1
    cache_results: bool = False
    benchmark: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DiffractionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'DiffractionConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))

class VoigtFitter:
    """Class for Voigt profile fitting operations."""
    
    @staticmethod
    def func_voigt(x: np.ndarray, amp: float, bg: float, mix: float, cen: float, width: float) -> np.ndarray:
        """
        Calculate the Voigt profile, which is a convolution of Gaussian and Lorentzian profiles.
        
        Parameters:
        -----------
        x : np.ndarray
            The x values at which to calculate the profile
        amp : float
            Amplitude of the profile
        bg : float
            Background level
        mix : float
            Mixing parameter between Lorentzian and Gaussian (0 = pure Gaussian, 1 = pure Lorentzian)
        cen : float
            Center position of the profile
        width : float
            Width parameter of the profile
            
        Returns:
        --------
        np.ndarray
            Calculated Voigt profile values
        """
        dx = x - cen
        # Correct Gaussian normalization
        g = np.exp(-0.5 * (dx / width)**2) / (width * np.sqrt(2 * np.pi))
        # Lorentzian component
        l = 1 / (np.pi * width * (1 + (dx / width)**2))
        return bg + amp * (mix * l + (1 - mix) * g)
    
    @staticmethod
    def multi_voigt(x: np.ndarray, *params) -> np.ndarray:
        """
        Calculate multiple Voigt profiles and sum them.
        
        Parameters:
        -----------
        x : np.ndarray
            The x values at which to calculate the profiles
        *params : float
            Parameters for each Voigt profile (5 parameters per peak)
            
        Returns:
        --------
        np.ndarray
            Sum of all Voigt profiles
        """
        result = np.zeros_like(x, dtype=float)
        num_peaks = len(params) // 5
        
        for i in range(num_peaks):
            peak_params = params[i*5:(i+1)*5]
            result += VoigtFitter.func_voigt(x, *peak_params)
            
        return result
    
    @staticmethod
    def fit_single_voigt(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a single Voigt profile to the provided data.
        
        Parameters:
        -----------
        x : np.ndarray
            The x values of the data points
        y : np.ndarray
            The y values of the data points
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Optimized parameters and covariance matrix
        """
        # Set reasonable bounds for parameters
        y_max = np.max(y)
        y_median = np.median(y)
        x_at_max = x[np.argmax(y)]
        width_guess = len(y) / 20  # More reasonable width guess
        
        bounds = ([0, 0, 0, 0, 0], [y_max * 10, y_max, 1, np.max(x), len(y) / 4])
        
        # Initial parameter guesses
        p0 = [y_max, y_median, 0.5, x_at_max, width_guess]
        
        try:
            # Perform curve fitting with more robust settings
            params, params_cov = curve_fit(
                VoigtFitter.func_voigt, x, y, 
                p0=p0, 
                bounds=bounds,
                maxfev=10000,  # More iterations
                method='trf'   # Trust Region Reflective - more robust
            )
            return params, params_cov
        except RuntimeError as e:
            logger.warning(f"Curve fitting failed: {e}")
            logger.info("Returning initial guess parameters")
            return np.array(p0), np.zeros((5, 5))
    
    @staticmethod
    def fit_multi_voigt(x: np.ndarray, y: np.ndarray, num_peaks: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit multiple Voigt profiles to the provided data.
        
        Parameters:
        -----------
        x : np.ndarray
            The x values of the data points
        y : np.ndarray
            The y values of the data points
        num_peaks : int
            Number of peaks to fit
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Optimized parameters and covariance matrix
        """
        if num_peaks <= 0:
            raise ValueError("Number of peaks must be positive")
        
        if num_peaks == 1:
            return VoigtFitter.fit_single_voigt(x, y)
        
        # For multi-peak fitting, we need to identify potential peak locations
        # Use a simple peak finding algorithm
        y_smooth = np.convolve(y, np.ones(5)/5, mode='same')  # Simple smoothing
        
        # Find local maxima
        peak_indices = []
        for i in range(1, len(y_smooth)-1):
            if y_smooth[i] > y_smooth[i-1] and y_smooth[i] > y_smooth[i+1]:
                peak_indices.append(i)
        
        # Sort by peak height and take the num_peaks highest
        peak_indices.sort(key=lambda i: y_smooth[i], reverse=True)
        peak_indices = peak_indices[:num_peaks]
        peak_indices.sort()  # Sort by position for clarity
        
        # Create initial guesses for each peak
        p0 = []
        y_bg = np.min(y)
        
        for idx in peak_indices:
            amp = y[idx] - y_bg
            cen = x[idx]
            width = len(y) / 20
            p0.extend([amp, y_bg, 0.5, cen, width])
        
        # If we didn't find enough peaks, add some default ones
        while len(p0) < num_peaks * 5:
            # Add a small peak at a random position
            random_pos = np.random.randint(0, len(x))
            amp = (y[random_pos] - y_bg) / 2
            cen = x[random_pos]
            p0.extend([amp, y_bg, 0.5, cen, width])
        
        # Ensure we have exactly num_peaks sets of parameters
        p0 = p0[:num_peaks * 5]
        
        # Define the fitting function with the right number of parameters
        def fitting_func(x, *params):
            return VoigtFitter.multi_voigt(x, *params)
        
        try:
            # Perform the curve fitting
            params, params_cov = curve_fit(
                fitting_func, x, y, 
                p0=p0,
                maxfev=20000,  # More iterations for multi-peak fitting
                method='trf'
            )
            return params, params_cov
        except RuntimeError as e:
            logger.warning(f"Multi-peak curve fitting failed: {e}")
            logger.info("Returning initial guess parameters")
            return np.array(p0), np.zeros((len(p0), len(p0)))


class BinaryUtils:
    """Utilities for handling binary data files."""
        
    @staticmethod
    def load_pixel_maps(map_path: str, n_map_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load pixel mapping data from binary files.
        
        Parameters:
        -----------
        map_path : str
            Path to the pixel map binary file
        n_map_path : str
            Path to the pixel count map binary file
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Pixel list, fractional values, and pixel count list
        """
        # Check if files exist
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file not found: {map_path}")
        if not os.path.exists(n_map_path):
            raise FileNotFoundError(f"NMap file not found: {n_map_path}")
        
        # Load pixel list
        logger.info(f"Loading pixel map from {map_path}")
        px_list = np.fromfile(map_path, dtype=np.int32)
        
        # Validate pixel list dimensions
        if px_list.size % 4 != 0:
            raise ValueError("The total number of elements in pxList is not divisible by 4.")
        
        # Reshape pixel list and create fractional values
        px_list = px_list.reshape(-1, 4)
        logger.info(f"Reading area fractions in map from {map_path}")
        frac_values = np.fromfile(map_path, dtype=np.float64).reshape(-1, 2)[:,1]
                
        # Load pixel count list
        logger.info(f"Loading pixel count map from {n_map_path}")
        n_px_list = np.fromfile(n_map_path, dtype=np.int32)
        
        logger.info(f"Loaded {px_list.shape[0]} pixels and {n_px_list.size} bin entries")
        
        return px_list, frac_values, n_px_list


class ImageUtils:
    """Utilities for handling image data."""
    
    @staticmethod
    def load_image_data(image_path: str, dark_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess image data.
        
        Parameters:
        -----------
        image_path : str
            Path to the diffraction image
        dark_path : Optional[str]
            Path to the dark image for background subtraction
            
        Returns:
        --------
        Tuple[np.ndarray, int]
            Processed image data and number of pixels in Y dimension
        """
        # Check if image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Loading image from {image_path}")
        
        # Load image
        with Image.open(image_path) as img:
            image = np.array(img)
        
        # Apply dark correction if dark image is provided
        if dark_path:
            if not os.path.exists(dark_path):
                raise FileNotFoundError(f"Dark image file not found: {dark_path}")
            
            logger.info(f"Applying dark correction from {dark_path}")
            with Image.open(dark_path) as img:
                dark = np.array(img)
            
            # Validate dimensions
            if dark.shape != image.shape:
                raise ValueError(f"Dark image dimensions {dark.shape} don't match image dimensions {image.shape}")
            
            # Subtract dark image and clip negative values
            image = np.clip(image - dark, 0, None)
        
        # Convert to float
        image = image.astype(np.float32)
        
        logger.info(f"Image loaded with shape {image.shape}")
        
        return image, image.shape[0]


class GPUUtils:
    """Utilities for GPU operations."""
    
    @staticmethod
    def cuda_available() -> bool:
        """
        Check if CUDA is available.
        
        Returns:
        --------
        bool
            True if CUDA is available, False otherwise
        """
        return CUDA_AVAILABLE
    
    @staticmethod
    def get_optimal_block_size() -> int:
        """
        Get the optimal block size for the current GPU.
        
        Returns:
        --------
        int
            Optimal number of threads per block
        """
        if not GPUUtils.cuda_available():
            return 256  # Default fallback
        
        device = cuda.get_current_device()
        # A good balance is often a multiple of 32 (warp size) up to the hardware limit
        return min(256, device.MAX_THREADS_PER_BLOCK)
    
    @staticmethod
    def print_gpu_info() -> None:
        """Print information about available GPUs."""
        if not GPUUtils.cuda_available():
            logger.info("No CUDA-capable devices found")
            return
        
        device = cuda.get_current_device()
        logger.info(f"Using GPU: {device.name}")
        logger.info(f"Compute capability: {device.compute_capability}")
        logger.info(f"Max threads per block: {device.MAX_THREADS_PER_BLOCK}")
        logger.info(f"Max shared memory per block: {device.MAX_SHARED_MEMORY_PER_BLOCK} bytes")
        logger.info(f"Warp size: {device.WARP_SIZE}")


# Import and define CUDA kernels only if CUDA is available
if CUDA_AVAILABLE:
    @cuda.jit
    def integrate_image_kernel(image, px_list, n_px_list, frac_values, result, 
                              n_r_bins, n_eta_bins, r_min, r_bin_size, 
                              bad_px_intensity, gap_intensity, nr_pixels_y):
        """
        CUDA kernel for diffraction image integration.
        
        This kernel processes one radial bin per thread block, with threads within
        a block processing different azimuthal bins.
        """
        # Get thread and block indices
        i = cuda.blockIdx.x  # radial bin index
        j = cuda.threadIdx.x  # azimuthal bin index
        
        # Check if this thread is responsible for a valid bin
        if i < n_r_bins and j < n_eta_bins:
            # Shared memory for block-level reduction
            shared_intensity = cuda.shared.array(shape=(1024), dtype=np.float32)
            shared_counts = cuda.shared.array(shape=(1024), dtype=np.int32)
            
            # Initialize shared memory
            shared_intensity[j] = 0.0
            shared_counts[j] = 0
            cuda.syncthreads()
            
            # Process the bin
            pos = i * n_eta_bins + j
            n_pixels = n_px_list[2 * pos]
            data_pos = n_px_list[2 * pos + 1]
            
            intensity = 0.0
            tot_area = 0.0
            
            # Process all pixels in this bin
            for k in range(n_pixels):
                idx = data_pos + k
                pixel_y = px_list[idx][0]
                pixel_x = px_list[idx][1]
                test_pos = pixel_y, pixel_x
                
                # Check if pixel is valid
                pixel_val = image[test_pos]
                if pixel_val != bad_px_intensity and pixel_val != gap_intensity:
                    frac = frac_values[idx]
                    intensity += pixel_val * frac
                    tot_area += frac
            
            # Normalize intensity if area is positive
            if tot_area > 0:
                intensity /= tot_area
                shared_intensity[j] = intensity
                shared_counts[j] = 1
            
            cuda.syncthreads()
            
            # Reduction within block (only thread 0 does this)
            if j == 0:
                total_int = 0.0
                total_count = 0
                
                for jj in range(n_eta_bins):
                    total_int += shared_intensity[jj]
                    total_count += shared_counts[jj]
                
                if total_count > 0:
                    r_mean = r_min + (i + 0.5) * r_bin_size
                    result[i, 0] = r_mean
                    result[i, 1] = total_int / total_count


    @cuda.jit
    def integrate_image_kernel_optimized(image, px_list, n_px_list, frac_values, result, 
                                        n_r_bins, n_eta_bins, r_min, r_bin_size, 
                                        bad_px_intensity, gap_intensity, nr_pixels_y):
        """
        Optimized CUDA kernel for diffraction image integration.
        
        This version uses a more efficient memory access pattern and reduction algorithm.
        """
        # Get thread and block indices
        i = cuda.blockIdx.x  # radial bin
        tx = cuda.threadIdx.x  # thread index within block
        
        # Shared memory for reduction
        shared_intensity = cuda.shared.array(shape=(1024), dtype=np.float32)
        shared_counts = cuda.shared.array(shape=(1024), dtype=np.int32)
        
        # Initialize shared memory
        shared_intensity[tx] = 0.0
        shared_counts[tx] = 0
        cuda.syncthreads()
        
        # Process multiple azimuthal bins per thread using a grid-stride loop
        stride = cuda.blockDim.x
        
        for j in range(tx, n_eta_bins, stride):
            if i < n_r_bins and j < n_eta_bins:
                pos = i * n_eta_bins + j
                n_pixels = n_px_list[2 * pos]
                data_pos = n_px_list[2 * pos + 1]
                
                intensity = 0.0
                tot_area = 0.0
                
                # Process all pixels in this bin
                for k in range(n_pixels):
                    idx = data_pos + k
                    pixel_y = px_list[idx][0]
                    pixel_x = px_list[idx][1]
                    test_pos = pixel_y, pixel_x
                    
                    # Check if pixel is valid
                    pixel_val = image[test_pos]
                    if pixel_val != bad_px_intensity and pixel_val != gap_intensity:
                        frac = frac_values[idx]
                        intensity += pixel_val * frac
                        tot_area += frac
                
                # Normalize intensity if area is positive
                if tot_area > 0:
                    intensity /= tot_area
                    cuda.atomic.add(shared_intensity, tx, intensity)
                    cuda.atomic.add(shared_counts, tx, 1)
        
        cuda.syncthreads()
        
        # Parallel reduction in shared memory
        s = cuda.blockDim.x // 2
        while s > 0:
            if tx < s:
                shared_intensity[tx] += shared_intensity[tx + s]
                shared_counts[tx] += shared_counts[tx + s]
            cuda.syncthreads()
            s //= 2
        
        # Write final result
        if tx == 0 and shared_counts[0] > 0:
            r_mean = r_min + (i + 0.5) * r_bin_size
            result[i, 0] = r_mean
            result[i, 1] = shared_intensity[0] / shared_counts[0]


class PlotUtils:
    """Utilities for plotting results."""
    
    @staticmethod
    def plot_results(result: np.ndarray, params: np.ndarray, num_peaks: int = 1, output_file: Optional[str] = None) -> None:
        """
        Plot the integrated data and fitted Voigt profile(s).
        
        Parameters:
        -----------
        result : np.ndarray
            Integrated diffraction data
        params : np.ndarray
            Fitted Voigt profile parameters
        num_peaks : int
            Number of peaks in the fit
        output_file : Optional[str]
            Path to save the plot (if None, display plot instead)
        """
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.plot(result[:, 0], result[:, 1], 'o', markersize=3, label='Integrated Data')
        
        # Plot fitted curve(s)
        x = result[:, 0]
        
        if num_peaks == 1:
            # Single peak
            y_fit = VoigtFitter.func_voigt(x, params[0], params[1], params[2], params[3], params[4])
            plt.plot(x, y_fit, 'r-', linewidth=2, label='Voigt Fit')
            
            # Add peak parameters to the plot
            amp, bg, mix, cen, width = params
            plt.annotate(f'Peak: A={amp:.1f}, BG={bg:.1f}, M={mix:.2f}, C={cen:.1f}, W={width:.1f}',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        else:
            # Multiple peaks
            colors = plt.cm.tab10.colors
            y_fit_total = np.zeros_like(x, dtype=float)
            
            for i in range(num_peaks):
                peak_params = params[i*5:(i+1)*5]
                y_fit_i = VoigtFitter.func_voigt(x, *peak_params)
                y_fit_total += y_fit_i
                
                # Plot individual peak
                plt.plot(x, y_fit_i, '--', color=colors[i % len(colors)], linewidth=1, 
                         label=f'Peak {i+1}')
                
                # Add peak parameters
                amp, bg, mix, cen, width = peak_params
                plt.annotate(f'Peak {i+1}: A={amp:.1f}, C={cen:.1f}, W={width:.1f}',
                            xy=(0.05, 0.95 - 0.05*i), xycoords='axes fraction',
                            color=colors[i % len(colors)],
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Plot total fit
            plt.plot(x, y_fit_total, 'r-', linewidth=2, label='Total Fit')
        
        # Add labels and legend
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Diffraction Data with Voigt Profile Fit')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()
    
class ImageIntegrator:
    """Class for performing image integration."""
    
    @staticmethod
    def integrate_image_cuda(image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                           r_min, r_bin_size, bad_px_intensity, gap_intensity, nr_pixels_y):
        """
        CUDA-accelerated version of integrate_image.
        
        Parameters match the original integrate_image function.
        
        Returns:
        --------
        np.ndarray
            Integrated 1D profile with radius and intensity values
        """
        if not GPUUtils.cuda_available():
            raise RuntimeError("CUDA is not available. Use integrate_image_cpu instead.")
            
        # Ensure all arrays are contiguous for optimal memory access
        image = np.ascontiguousarray(image, dtype=np.float32)
        px_list = np.ascontiguousarray(px_list, dtype=np.int32)
        n_px_list = np.ascontiguousarray(n_px_list, dtype=np.int32)
        frac_values = np.ascontiguousarray(frac_values, dtype=np.float64)
        
        # Initialize result array
        result = np.zeros((n_r_bins, 2), dtype=np.float32)
        
        # Create stream for concurrent operations
        stream = cuda.stream()
        
        try:
            # Transfer data to GPU using pinned memory for faster transfers
            with cuda.pinned_array(image.shape, dtype=np.float32) as pinned_image:
                pinned_image[:] = image
                d_image = cuda.to_device(pinned_image, stream=stream)
            
            d_px_list = cuda.to_device(px_list, stream=stream)
            d_n_px_list = cuda.to_device(n_px_list, stream=stream)
            d_frac_values = cuda.to_device(frac_values, stream=stream)
            d_result = cuda.to_device(result, stream=stream)
            
            # Define grid and block dimensions
            # Each radial bin is processed by one block
            # Each azimuthal bin is processed by one thread within the block
            threads_per_block = min(1024, n_eta_bins)  # Max 1024 threads per block
            blocks_per_grid = n_r_bins
            
            # Launch kernel
            integrate_image_kernel[blocks_per_grid, threads_per_block, stream](
                d_image, d_px_list, d_n_px_list, d_frac_values, d_result,
                n_r_bins, n_eta_bins, r_min, r_bin_size,
                bad_px_intensity, gap_intensity, nr_pixels_y
            )
            
            # Transfer result back to host
            result = d_result.copy_to_host(stream=stream)
            
            # Synchronize stream to ensure all operations are complete
            stream.synchronize()
            
        finally:
            # Clean up resources
            cuda.close()
        
        return result
    
    @staticmethod
    def integrate_image_cuda_optimized(image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                                      r_min, r_bin_size, bad_px_intensity, gap_intensity, nr_pixels_y):
        """
        Optimized CUDA-accelerated version of integrate_image.
        
        This version uses memory optimization techniques and a more efficient reduction algorithm.
        
        Returns:
        --------
        np.ndarray
            Integrated 1D profile with radius and intensity values
        """
        if not GPUUtils.cuda_available():
            raise RuntimeError("CUDA is not available. Use integrate_image_cpu instead.")
            
        # Convert to the right types and ensure contiguity
        image = np.ascontiguousarray(image, dtype=np.float32)
        px_list = np.ascontiguousarray(px_list, dtype=np.int32)
        n_px_list = np.ascontiguousarray(n_px_list, dtype=np.int32)
        frac_values = np.ascontiguousarray(frac_values, dtype=np.float64)
        
        # Initialize result array
        result = np.zeros((n_r_bins, 2), dtype=np.float32)
        
        # Create stream for concurrent operations
        stream = cuda.stream()
        
        try:
            # Transfer data to GPU using pinned memory for faster transfers
            with cuda.pinned_array(image.shape, dtype=np.float32) as pinned_image:
                pinned_image[:] = image
                d_image = cuda.to_device(pinned_image, stream=stream)
            
            d_px_list = cuda.to_device(px_list, stream=stream)
            d_n_px_list = cuda.to_device(n_px_list, stream=stream)
            d_frac_values = cuda.to_device(frac_values, stream=stream)
            d_result = cuda.to_device(result, stream=stream)
            
            # Define grid and block dimensions
            # Get optimal block size for current GPU
            threads_per_block = GPUUtils.get_optimal_block_size()
            blocks_per_grid = n_r_bins
            
            logger.info(f"CUDA kernel configuration: {blocks_per_grid} blocks with {threads_per_block} threads each")
            
            # Launch kernel
            integrate_image_kernel_optimized[blocks_per_grid, threads_per_block, stream](
                d_image, d_px_list, d_n_px_list, d_frac_values, d_result,
                n_r_bins, n_eta_bins, r_min, r_bin_size,
                bad_px_intensity, gap_intensity, nr_pixels_y
            )
            
            # Transfer result back to host
            result = d_result.copy_to_host(stream=stream)
            
            # Synchronize stream to ensure all operations are complete
            stream.synchronize()
            
        finally:
            # Clean up resources
            cuda.close()
        
        return result
    
    @staticmethod
    @njit
    def integrate_image_cpu(image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                           r_min, r_bin_size, bad_px_intensity, gap_intensity, nr_pixels_y):
        """
        CPU version of the integrate_image function using Numba.
        Used as a fallback if CUDA is not available.
        """
        result = np.zeros((n_r_bins, 2), dtype=np.float32)
        
        for i in range(n_r_bins):
            int_1d = 0
            n1ds = 0
            r_mean = (r_min + (i + 0.5) * r_bin_size)
            
            for j in range(n_eta_bins):
                pos = i * n_eta_bins + j
                n_pixels = n_px_list[2 * pos + 0]
                data_pos = n_px_list[2 * pos + 1]
                
                intensity = 0.
                tot_area = 0.
                
                if n_pixels == 0:
                    continue
                    
                for k in range(n_pixels):
                    this_val = px_list[data_pos + k]
                    test_pos = this_val[0], this_val[1]
                    
                    if image[test_pos] == bad_px_intensity or image[test_pos] == gap_intensity:
                        continue
                        
                    this_int = image[test_pos]
                    intensity += this_int * frac_values[data_pos + k]
                    tot_area += frac_values[data_pos + k]
                    
                if tot_area == 0:
                    continue
                    
                intensity /= tot_area
                int_1d += intensity
                n1ds += 1
                
            if n1ds == 0:
                continue
                
            int_1d /= n1ds
            result[i, 0] = r_mean
            result[i, 1] = int_1d
            
        return result
    
    @staticmethod
    def integrate_image(image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                       r_min, r_bin_size, bad_px_intensity, gap_intensity, nr_pixels_y, use_gpu=True):
        """
        High-level function that chooses the appropriate integration method based on device availability.
        
        This function tries to use the optimized CUDA version first, falls back to the basic CUDA version,
        and finally falls back to the CPU version if CUDA is not available.
        
        Parameters:
        -----------
        image : np.ndarray
            Image data
        px_list : np.ndarray
            Pixel list
        n_px_list : np.ndarray
            Pixel count list
        frac_values : np.ndarray
            Fractional values
        n_r_bins : int
            Number of radial bins
        n_eta_bins : int
            Number of azimuthal bins
        r_min : float
            Minimum radius
        r_bin_size : float
            Size of each radial bin
        bad_px_intensity : float
            Value that marks bad pixels
        gap_intensity : float
            Value that marks gap pixels
        nr_pixels_y : int
            Number of pixels in Y dimension
        use_gpu : bool
            Whether to use GPU acceleration if available
            
        Returns:
        --------
        np.ndarray
            Integrated 1D profile with radius and intensity values
        """
        # If GPU is not requested or not available, use CPU
        if not use_gpu or not GPUUtils.cuda_available():
            logger.info("Using CPU integration")
            return ImageIntegrator.integrate_image_cpu(
                image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                r_min, r_bin_size, bad_px_intensity, gap_intensity, nr_pixels_y
            )
        
        # Otherwise, try GPU
        logger.info("Using GPU integration")
        try:
            # Try optimized CUDA version first
            logger.info("Attempting optimized CUDA integration")
            return ImageIntegrator.integrate_image_cuda_optimized(
                image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                r_min, r_bin_size, bad_px_intensity, gap_intensity, nr_pixels_y
            )
        except Exception as e1:
            logger.warning(f"Optimized CUDA integration failed: {e1}")
            try:
                # Fall back to basic CUDA version
                logger.info("Falling back to basic CUDA integration")
                return ImageIntegrator.integrate_image_cuda(
                    image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                    r_min, r_bin_size, bad_px_intensity, gap_intensity, nr_pixels_y
                )
            except Exception as e2:
                logger.warning(f"Basic CUDA integration failed: {e2}")
                logger.info("Falling back to CPU integration")
                # Fall back to CPU version
                return ImageIntegrator.integrate_image_cpu(
                    image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                    r_min, r_bin_size, bad_px_intensity, gap_intensity, nr_pixels_y
                )


class DiffractionProcessor:
    """Main class for diffraction image processing and analysis."""
    
    def __init__(self, config: DiffractionConfig):
        """
        Initialize the processor with configuration.
        
        Parameters:
        -----------
        config : DiffractionConfig
            Configuration for processing
        """
        self.config = config
        self._result_cache = {}
    
    def process(self, callback: Optional[Callable[[str, float], None]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a diffraction image and fit Voigt profile(s) to the integrated data.
        
        Parameters:
        -----------
        callback : Optional[Callable[[str, float], None]]
            Callback function for progress updates (message, progress)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Integrated data and fitted parameters
        """
        # Check if results are cached
        cache_key = (
            self.config.image_path, 
            self.config.dark_path, 
            self.config.map_path, 
            self.config.n_map_path,
            self.config.r_min,
            self.config.r_max,
            self.config.r_bin_size,
            self.config.eta_min,
            self.config.eta_max,
            self.config.eta_bin_size
        )
        
        if self.config.cache_results and cache_key in self._result_cache:
            logger.info("Using cached integration results")
            result = self._result_cache[cache_key]
            # Still do the fitting
            return self._fit_profiles(result, callback)
        
        # Progress callback
        if callback:
            callback("Loading image data...", 0.1)
        
        # Load and process image
        image, nr_pixels_y = ImageUtils.load_image_data(
            self.config.image_path, 
            self.config.dark_path
        )
        
        # Calculate the number of bins
        n_r_bins = int(ceil((self.config.r_max - self.config.r_min) / self.config.r_bin_size))
        n_eta_bins = int(ceil((self.config.eta_max - self.config.eta_min) / self.config.eta_bin_size))
        
        if callback:
            callback("Loading pixel maps...", 0.2)
        
        # Load pixel maps
        px_list, frac_values, n_px_list = BinaryUtils.load_pixel_maps(
            self.config.map_path, 
            self.config.n_map_path
        )
        
        if callback:
            callback("Integrating image...", 0.4)
        
        # Benchmark integration times if requested
        if self.config.benchmark and GPUUtils.cuda_available():
            cpu_time, basic_cuda_time, opt_cuda_time = self._run_benchmark(
                image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                nr_pixels_y
            )
            
            # Plot benchmark results
            benchmark_output = None
            if self.config.output_file:
                base, ext = os.path.splitext(self.config.output_file)
                benchmark_output = f"{base}_benchmark{ext}"
            
            PlotUtils.plot_benchmark_results(
                cpu_time, basic_cuda_time, opt_cuda_time, 
                benchmark_output
            )
        
        # Integrate the image
        result = ImageIntegrator.integrate_image(
            image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
            self.config.r_min, self.config.r_bin_size, 
            self.config.bad_px_intensity, self.config.gap_intensity, 
            nr_pixels_y, self.config.use_gpu
        )
        
        # Cache results if enabled
        if self.config.cache_results:
            self._result_cache[cache_key] = result
        
        # Save integrated data if requested
        if self.config.save_data_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.config.save_data_file)), exist_ok=True)
            
            # Save data
            header = "Radius,Intensity"
            np.savetxt(
                self.config.save_data_file, 
                result, 
                delimiter=',', 
                header=header, 
                fmt='%.6f',
                comments=''
            )
            logger.info(f"Integrated data saved to {self.config.save_data_file}")
        
        # Fit profile and return results
        return self._fit_profiles(result, callback)
    
    def _fit_profiles(self, result: np.ndarray, callback: Optional[Callable[[str, float], None]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit Voigt profile(s) to the integrated data.
        
        Parameters:
        -----------
        result : np.ndarray
            Integrated diffraction data
        callback : Optional[Callable[[str, float], None]]
            Callback function for progress updates
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Integrated data and fitted parameters
        """
        if callback:
            callback("Fitting Voigt profile...", 0.8)
        
        # Fit Voigt profile
        if self.config.num_peaks > 1:
            params, _ = VoigtFitter.fit_multi_voigt(
                result[:, 0], result[:, 1], self.config.num_peaks
            )
        else:
            params, _ = VoigtFitter.fit_single_voigt(result[:, 0], result[:, 1])
        
        if self.config.output_file:
            if callback:
                callback("Plotting results...", 0.9)
            
            # Plot results
            PlotUtils.plot_results(result, params, self.config.num_peaks, self.config.output_file)
        
        if callback:
            callback("Processing complete", 1.0)
        
        return result, params
    
    def _run_benchmark(self, image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins, nr_pixels_y) -> Tuple[float, float, float]:
        """
        Benchmark different integration methods.
        
        Parameters:
        -----------
        Various arrays needed for integration
            
        Returns:
        --------
        Tuple[float, float, float]
            CPU, basic CUDA, and optimized CUDA integration times in seconds
        """
        logger.info("Running integration benchmark...")
        
        # CPU integration
        start = time.time()
        _ = ImageIntegrator.integrate_image_cpu(
            image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
            self.config.r_min, self.config.r_bin_size, 
            self.config.bad_px_intensity, self.config.gap_intensity, 
            nr_pixels_y
        )
        cpu_time = time.time() - start
        logger.info(f"CPU integration time: {cpu_time:.3f} seconds")
        
        # Initialize CUDA times
        basic_cuda_time = float('inf')
        opt_cuda_time = float('inf')
        
        if GPUUtils.cuda_available():
            # Basic CUDA integration
            try:
                start = time.time()
                _ = ImageIntegrator.integrate_image_cuda(
                    image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                    self.config.r_min, self.config.r_bin_size, 
                    self.config.bad_px_intensity, self.config.gap_intensity, 
                    nr_pixels_y
                )
                basic_cuda_time = time.time() - start
                logger.info(f"Basic CUDA integration time: {basic_cuda_time:.3f} seconds")
            except Exception as e:
                logger.warning(f"Basic CUDA benchmark failed: {e}")
            
            # Optimized CUDA integration
            try:
                start = time.time()
                _ = ImageIntegrator.integrate_image_cuda_optimized(
                    image, px_list, n_px_list, frac_values, n_r_bins, n_eta_bins,
                    self.config.r_min, self.config.r_bin_size, 
                    self.config.bad_px_intensity, self.config.gap_intensity, 
                    nr_pixels_y
                )
                opt_cuda_time = time.time() - start
                logger.info(f"Optimized CUDA integration time: {opt_cuda_time:.3f} seconds")
            except Exception as e:
                logger.warning(f"Optimized CUDA benchmark failed: {e}")
        
        return cpu_time, basic_cuda_time, opt_cuda_time


# Plot utility functions
@staticmethod
def plot_benchmark_results(cpu_time: float, basic_cuda_time: float, opt_cuda_time: float, 
                          output_file: Optional[str] = None) -> None:
        """
        Plot benchmark results comparing CPU and GPU integration times.
        
        Parameters:
        -----------
        cpu_time : float
            CPU integration time in seconds
        basic_cuda_time : float
            Basic CUDA integration time in seconds
        opt_cuda_time : float
            Optimized CUDA integration time in seconds
        output_file : Optional[str]
            Path to save the plot (if None, display plot instead)
        """
        methods = ['CPU', 'Basic CUDA', 'Optimized CUDA']
        times = [cpu_time, basic_cuda_time, opt_cuda_time]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, times, color=['blue', 'green', 'red'])
        
        # Add speedup labels
        if cpu_time > 0:
            for i, (method, time) in enumerate(zip(methods[1:], times[1:])):
                if time > 0:
                    speedup = cpu_time / time
                    plt.text(i+1, time + 0.1, f'{speedup:.1f}x faster than CPU', 
                             ha='center', va='bottom', rotation=0)
        
        plt.ylabel('Time (seconds)')
        plt.title('Integration Performance Comparison')
        plt.grid(axis='y', alpha=0.3)
        
        # Add time values on top of bars
        for bar, time in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, time + 0.05, 
                     f'{time:.3f}s', ha='center', va='bottom')
        
        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Benchmark plot saved to {output_file}")
        else:
            plt.show()