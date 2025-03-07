# Diffraction Image Processing

## Overview

This package provides tools for processing diffraction images from 2D detectors, performing data integration to obtain 1D diffraction profiles, and fitting Voigt profiles to the results. The code is optimized with both CPU (Numba) and GPU (CUDA) acceleration.

## Features

- **High-performance image integration** with both CPU and GPU acceleration
- **Automatic fallback** between optimized CUDA, basic CUDA, and CPU implementations
- **Multi-peak fitting** with Voigt profiles
- **Comprehensive configuration** through command-line or programmatic interface
- **Progress reporting** for long-running operations
- **Result caching** to avoid redundant calculations
- **Performance benchmarking** to compare CPU and GPU implementations
- **Export of integrated data** in CSV format
- **Azimuthal angle control** for integration
- **Comprehensive error handling** and detailed logging

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- Pillow (PIL)
- Numba

Optional:
- CUDA Toolkit (for GPU acceleration)

## Installation

Clone this repository:

```bash
git clone https://github.com/username/diffraction-processing.git
cd diffraction-processing
```

Install the dependencies:

```bash
pip install numpy scipy matplotlib pillow numba
```

For GPU acceleration (optional):

```bash
pip install numba cudatoolkit
```

## Usage

### Command Line Interface

The package provides a command-line interface for easy use:

```bash
python diffraction_cli.py image.tif --dark dark.tif --map Map.bin --nmap nMap.bin --output output.png
```

#### Command Line Options

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

### Programmatic Usage

You can also use the package programmatically in your own Python code:

```python
from diffraction_core import DiffractionConfig, DiffractionProcessor

# Create a configuration
config = DiffractionConfig(
    image_path="diffraction_image.tif",
    dark_path="dark_image.tif",
    map_path="Map.bin",
    n_map_path="nMap.bin",
    r_min=10.0,
    r_max=100.0,
    r_bin_size=0.25,
    eta_min=-90.0,     # Custom azimuthal angle range
    eta_max=90.0,      # Custom azimuthal angle range
    eta_bin_size=0.5,  # Custom azimuthal bin size
    output_file="diffraction_fit.png",
    save_data_file="integrated_data.csv",  # Save the integrated data
    num_peaks=1
)

# Create a processor with this configuration
processor = DiffractionProcessor(config)

# Process the diffraction image
result, params = processor.process()

# Access the fitted parameters
amp, bg, mix, cen, width = params
print(f"Amplitude: {amp}, Background: {bg}, Mix: {mix}, Center: {cen}, Width: {width}")

# Access the raw integrated data
radii = result[:, 0]
intensities = result[:, 1]
```

See `example_script.py` for a more complete example.

### Saving Integrated Data

The package can now save the integrated 1D profile (radius vs. intensity) to a CSV file:

```bash
python diffraction_cli.py image.tif --output plot.png --save-data integrated_data.csv
```

This creates a CSV file with headers "Radius,Intensity" containing the integrated diffraction profile.

### Controlling Azimuthal Integration Range

You can control the azimuthal integration range using the eta parameters:

```bash
python diffraction_cli.py image.tif --output plot.png --etamin -90 --etamax 90 --etabin 0.5
```

This integrates only over azimuthal angles from -90° to 90° with 0.5° bin size.

## Performance Optimization

The code automatically selects the best available implementation:

1. **Optimized CUDA implementation** (if CUDA is available)
2. **Basic CUDA implementation** (fallback if optimized implementation fails)
3. **CPU implementation with Numba** (fallback if CUDA is not available)

For best performance:

- Use a GPU with CUDA support
- Enable result caching for repeated operations on the same data
- Use appropriate bin sizes for your specific use case

## Running Tests

The package includes a set of unit tests to verify functionality:

```bash
python -m unittest test_diffraction.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See the LICENSE file for details.