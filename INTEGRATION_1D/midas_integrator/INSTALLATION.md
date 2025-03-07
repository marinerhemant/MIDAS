# Installation Guide

This guide provides instructions for installing the `midas_integrator` package on different platforms.

## Prerequisites

Before installing the package, ensure you have the following prerequisites:

- Python 3.6 or higher
- pip (Python package installer)

For GPU acceleration (optional but recommended for performance):
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (compatible with your GPU), needs to be installed using conda, pip does not have cudatoolkit.

## Basic Installation

### Installing from Source

If you want to install the latest development version from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/marinerhemant/MIDAS.git
   cd MIDAS/INTEGRATION_1D/midas_integrator
   ```

2. Install the package in development mode:
   ```bash
   pip install -r requirements.txt
   ```

3. To include GPU support when installing from source:
   ```bash
   pip install -r requirements.txt
   conda install cudatoolkit
   ```

4. Install the package:
   ```bash
   python setup.py install
   ```

## Platform-Specific Instructions

### Windows

On Windows, the installation process is generally the same as above, but there are a few considerations:

1. It's recommended to use Anaconda or Miniconda for managing Python environments:
   ```
   conda create -n midas python=3.9
   conda activate midas
   pip install midas_integrator
   ```

2. For GPU support, you might need to install CUDA Toolkit and cuDNN separately.

### macOS

On macOS, the package can be installed using the procedure above as described above.

Note: GPU acceleration is not available on macOS as CUDA is not supported on this platform.

### Linux

On Linux, follow the basic installation instructions. For GPU support:

1. Ensure you have the NVIDIA drivers installed:
   ```bash
   nvidia-smi
   ```

2. Install CUDA Toolkit if not already installed:
   ```bash
   # Example for Ubuntu (adjust based on your distribution)
   sudo apt update
   sudo apt install nvidia-cuda-toolkit
   ```

3. Rest follow the instructions above.

## Verifying Installation

To verify that the package is installed correctly, run:

```python
import midas_integrator
print(midas_integrator.__version__)
```

To verify GPU support:

```python
from midas_integrator import GPUUtils
print(f"CUDA Available: {GPUUtils.cuda_available()}")
if GPUUtils.cuda_available():
    GPUUtils.print_gpu_info()
```

## Troubleshooting

If you encounter issues during installation:

1. **Missing dependencies**: Ensure all dependencies are installed. You can try:
   ```bash
   pip install -r requirements.txt
   ```

2. **GPU issues**: If you're having problems with GPU acceleration:
   - Check that CUDA is properly installed: `nvcc --version`
   - Ensure your GPU is detected: `nvidia-smi`
   - Try updating your GPU drivers

3. **Performance issues**: If the package is running slowly:
   - Check if GPU acceleration is being used
   - Adjust binning parameters for better performance
   - Consider using the caching feature for repeated operations

## Development Installation

For developers who want to contribute to the package:

```bash
git clone https://github.com/marinerhemant/MIDAS.git
cd MIDAS/INTEGRATION_1D/midas_integrator
python setup.py install
```

This installs the package in development mode with additional development dependencies.

To run the tests:

```bash
pytest
```
