# MIDAS INTRGRATOR User Guide

This guide provides detailed information on how to use the `midas_integrator` package for processing diffraction images and fitting Voigt profiles.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Command Line Interface](#command-line-interface)
3. [Python API](#python-api)
4. [Configuration Options](#configuration-options)
5. [Working with Data](#working-with-data)
6. [GPU Acceleration](#gpu-acceleration)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

## Getting Started

After [installing](INSTALLATION.md) the package, you can use it either through the command line interface or the Python API.

### Basic Example

```python
from midas_integrator import DiffractionConfig, DiffractionProcessor

# Create a configuration
config = DiffractionConfig(
    image_path="diffraction_image.tif",
    dark_path="dark_image.tif",
    output_file="result.png"
)

# Create a processor
processor = DiffractionProcessor(config)

# Process the image
result, params = processor.process()

# Print fitted parameters
amp, bg, mix, cen, width = params
print(f"Amplitude: {amp}, Background: {bg}, Center: {cen}, Width: {width}")
```

## Command Line Interface

The package provides a convenient command line interface for quick processing:

```bash
midas_integrator image.tif --dark dark.tif --output result.png
```

### Command Line Options

```
positional arguments:
  image_path           Path to the diffraction image

optional arguments:
  -h, --help           show this help message and exit
  --dark DARK          Path to the dark image for background subtraction
  --map MAP            Path to the pixel map binary file (default: Map.bin)
  --nmap NMAP          Path to the pixel count map binary file (default: nMap.bin)
  --rmin RMIN          Minimum radius for binning (default: 10)
  --rmax RMAX          Maximum radius for binning (default: 100)
  --rbin RBIN          Size of each radial bin (default: 0.25)
  --etamin ETAMIN      Minimum azimuthal angle (default: -180)
  --etamax ETAMAX      Maximum azimuthal angle (default: 180)
  --etabin ETABIN      Size of each azimuthal bin (default: 1)
  --badpx BADPX        Value that marks bad pixels (default: -1)
  --gappx GAPPX        Value that marks gap pixels (default: -2)
  --output OUTPUT      Path to save the plot
  --peaks PEAKS        Number of peaks to fit (default: 1)
  --cpu                Force CPU processing (no GPU)
  --benchmark          Run CPU vs GPU benchmark
  --cache              Cache integration results
  --config CONFIG      Load parameters from a JSON config file
  --save-config CONFIG Save parameters to a JSON config file
  --no-progress        Disable progress bar
  --save-data          Path to save the integrated radius and intensity data (CSV format)
```

### Examples

#### Basic processing with dark correction:
```bash
midas_integrator image.tif --dark dark.tif --output result.png
```

#### Fitting multiple peaks:
```bash
midas_integrator image.tif --peaks 2 --output multi_peak_result.png
```

#### Custom azimuthal range:
```bash
midas_integrator image.tif --etamin -90 --etamax 90 --etabin 0.5 --output custom_eta.png
```

#### Saving integrated data:
```bash
midas_integrator image.tif --output result.png --save-data integrated_data.csv
```

#### Running a benchmark:
```bash
midas_integrator image.tif --output result.png --benchmark
```

## Python API

The Python API provides more flexibility and control over the processing pipeline.

### Core Classes

- `DiffractionConfig`: Configuration class for processing parameters
- `DiffractionProcessor`: Main processing class that handles the workflow
- `VoigtFitter`: Utilities for Voigt profile fitting
- `GPUUtils`: Utilities for GPU acceleration
- `ImageUtils`: Utilities for image handling
- `BinaryUtils`: Utilities for binary data handling

### Plotting Utilities

The package includes several plotting utilities for visualizing results:

```python
from midas_integrator.utils import (
    plot_diffraction_profile,
    plot_log_scale_profile,
    plot_peaks,
    set_publication_style
)

# Set publication-quality plotting style
set_publication_style()

# Plot the data
plot_diffraction_profile(radii, intensities, output_file="profile.png")

# Log-scale plot for better visibility of small features
plot_log_scale_profile(radii, intensities, output_file="log_profile.png")

# Plot with fitted peaks
plot_peaks(radii, intensities, params, num_peaks=2, output_file="peaks.png")
```

## Configuration Options

The `DiffractionConfig` class provides numerous options for controlling the processing:

```python
config = DiffractionConfig(
    # Required parameters
    image_path="image.tif",
    
    # Data paths
    dark_path="dark.tif",        # Dark image for background subtraction
    map_path="Map.bin",          # Pixel mapping file
    n_map_path="nMap.bin",       # Pixel count mapping file
    
    # Binning parameters
    r_min=10.0,                  # Minimum radius
    r_max=100.0,                 # Maximum radius
    r_bin_size=0.25,             # Radial bin size
    eta_min=-180.0,              # Minimum azimuthal angle
    eta_max=180.0,               # Maximum azimuthal angle
    eta_bin_size=1.0,            # Azimuthal bin size
    
    # Pixel handling
    bad_px_intensity=-1.0,       # Value marking bad pixels
    gap_intensity=-2.0,          # Value marking gap pixels
    
    # Output options
    output_file="result.png",    # Path to save the plot
    save_data_file="data.csv",   # Path to save integrated data
    
    # Processing options
    use_gpu=True,                # Use GPU acceleration if available
    num_peaks=1,                 # Number of peaks to fit
    cache_results=False,         # Cache integration results
    benchmark=False              # Run benchmark comparison
)
```

### Saving and Loading Configurations

You can save and load configurations to/from JSON files:

```python
# Save configuration
config.save("my_config.json")

# Load configuration
config = DiffractionConfig.load("my_config.json")
```

## Working with Data

### Input Data Requirements

The package works with the following input data:

1. **Diffraction Image**: 2D image from a detector (TIFF, PNG, etc.)
2. **Dark Image** (optional): Background image for subtraction
3. **Pixel Mapping Files**: Binary files that map detector pixels to radial/azimuthal bins

### Output Data

The package can produce the following outputs:

1. **Plot File**: Visualization of the integrated profile with fitted peak(s)
2. **Integrated Data**: CSV file with radius and intensity values
3. **Fitted Parameters**: Parameters of the Voigt profile(s)

### Handling Integrated Data

The integration result is a NumPy array with two columns:
- Column 0: Radius values
- Column 1: Corresponding intensity values

```python
# Access the integrated data
radii = result[:, 0]
intensities = result[:, 1]

# Perform additional analysis
peak_radius = radii[np.argmax(intensities)]
mean_intensity = np.mean(intensities)
```

## GPU Acceleration

The package includes GPU acceleration for faster processing:

### Checking GPU Availability

```python
from midas_integrator import GPUUtils

if GPUUtils.cuda_available():
    print("GPU acceleration is available")
    GPUUtils.print_gpu_info()
else:
    print("GPU acceleration is not available, using CPU")
```

### Controlling GPU Usage

```python
# Force CPU processing
config = DiffractionConfig(
    image_path="image.tif",
    output_file="result.png",
    use_gpu=False  # Disable GPU acceleration
)
```

### Benchmarking

You can benchmark CPU vs. GPU performance:

```python
config = DiffractionConfig(
    image_path="image.tif",
    output_file="result.png",
    benchmark=True  # Run CPU vs. GPU benchmark
)
```

## Advanced Usage

### Multi-Peak Fitting

```python
# Fit two peaks
config = DiffractionConfig(
    image_path="image.tif",
    output_file="result.png",
    num_peaks=2
)
processor = DiffractionProcessor(config)
result, params = processor.process()

# Interpret parameters (5 parameters per peak)
for i in range(config.num_peaks):
    peak_params = params[i*5:(i+1)*5]
    amp, bg, mix, cen, width = peak_params
    print(f"Peak {i+1}: A={amp}, BG={bg}, M={mix}, C={cen}, W={width}")
```

### Custom Azimuthal Integration

You can integrate over specific azimuthal angle ranges:

```python
# Integrate over top half only
config = DiffractionConfig(
    image_path="image.tif",
    output_file="result.png",
    eta_min=0,    # Start at 0 degrees
    eta_max=180,  # End at 180 degrees
    eta_bin_size=1.0
)
```

### Progress Tracking

For long-running operations, you can provide a callback function:

```python
def progress_callback(message, progress):
    print(f"{message} - {progress*100:.1f}%")

processor = DiffractionProcessor(config)
result, params = processor.process(progress_callback)
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure all file paths are correct and the files exist
2. **GPU Acceleration Errors**: If GPU integration fails, the package will automatically fall back to CPU
3. **Fitting Errors**: If peak fitting fails, check your data and try adjusting the r_min and r_max values

### Debug Logging

You can enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tips

1. Use GPU acceleration for large images
2. Enable caching for repeated operations on the same data
3. Adjust bin sizes appropriately for your data
4. Consider reducing the azimuthal range if you don't need the full 360 degrees
