#!/usr/bin/env python3
"""
Example script demonstrating how to use the midas_integrator package programmatically.

Author: Hemant Sharma
Date: 2025/03/06
"""

import os
import matplotlib.pyplot as plt
from midas_integrator import DiffractionConfig, DiffractionProcessor, GPUUtils
from midas_integrator.utils import plot_log_scale_profile, set_publication_style

def main():
    # Set publication style for plots
    set_publication_style()
    
    # Print GPU information if available
    if GPUUtils.cuda_available():
        print("GPU support is available")
        GPUUtils.print_gpu_info()
    else:
        print("GPU support is not available, will use CPU")
    
    # Define the configuration
    config = DiffractionConfig(
        image_path="sample_data/diffraction_image.tif",
        dark_path="sample_data/dark_image.tif",
        map_path="sample_data/Map.bin",
        n_map_path="sample_data/nMap.bin",
        r_min=20.0,
        r_max=120.0,
        r_bin_size=0.2,
        eta_min=-90.0,    # Custom azimuthal angle range
        eta_max=90.0,     # Custom azimuthal angle range
        eta_bin_size=0.5, # Custom azimuthal bin size
        output_file="results/diffraction_fit.png",
        save_data_file="results/integrated_data.csv", # Save the integrated data
        num_peaks=2,  # Try to fit two peaks
        benchmark=True,  # Run a benchmark
        cache_results=True  # Cache the integration results
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(config.output_file), exist_ok=True)
    
    # Create a processor with this configuration
    processor = DiffractionProcessor(config)
    
    # Define a simple callback for progress updates
    def progress_callback(message, progress):
        print(f"{message} - {progress*100:.1f}%")
    
    # Process the diffraction image
    result, params = processor.process(progress_callback)
    
    # Print the fitted parameters
    print("\nFitted Parameters:")
    for i in range(config.num_peaks):
        peak_params = params[i*5:(i+1)*5]
        print(f"Peak {i+1}:")
        print(f"  Amplitude: {peak_params[0]:.2f}")
        print(f"  Background: {peak_params[1]:.2f}")
        print(f"  Mix (Gaussian/Lorentzian): {peak_params[2]:.2f}")
        print(f"  Center: {peak_params[3]:.2f}")
        print(f"  Width: {peak_params[4]:.2f}")
    
    # Access the raw integrated data
    radii = result[:, 0]
    intensities = result[:, 1]
    
    # Create a log-scale plot using the utility function
    log_output = os.path.join(os.path.dirname(config.output_file), "diffraction_fit_log.png")
    plot_log_scale_profile(
        radii, 
        intensities,
        title="Diffraction Profile (Log Scale)",
        output_file=log_output
    )
    print(f"Log-scale plot saved to {log_output}")
    
    # You can also perform additional analysis or create custom plots here
    from midas_integrator.core import VoigtFitter
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(radii, intensities, 'o', markersize=3, label="Data")
    
    # Plot the individual peaks and total fit in log scale
    x = radii
    y_fit_total = 0
    for i in range(config.num_peaks):
        peak_params = params[i*5:(i+1)*5]
        y_fit_i = VoigtFitter.func_voigt(x, *peak_params)
        y_fit_total += y_fit_i
        plt.semilogy(x, y_fit_i, '--', linewidth=1, label=f"Peak {i+1}")
    
    plt.semilogy(x, y_fit_total, 'r-', linewidth=2, label="Total Fit")
    
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Intensity (log scale)')
    plt.title('Custom Diffraction Plot with Voigt Profile Fit (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the custom plot
    custom_log_output = os.path.join(os.path.dirname(config.output_file), "custom_log_plot.png")
    plt.savefig(custom_log_output, dpi=300, bbox_inches='tight')
    print(f"Custom log-scale plot saved to {custom_log_output}")
    
    print("\nProcessing complete!")
    
    # Print information about saved integrated data
    if config.save_data_file:
        print(f"Integrated data saved to: {config.save_data_file}")
        print("First few rows of the saved data:")
        with open(config.save_data_file, 'r') as f:
            for i, line in enumerate(f):
                print(f"  {line.strip()}")
                if i >= 5:  # Show first 5 rows
                    break


if __name__ == "__main__":
    main()
