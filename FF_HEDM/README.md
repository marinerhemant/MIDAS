# MIDAS/FF_HEDM

MIDAS/FF_HEDM is a software suite for the analysis of Far-Field High-Energy Diffraction Microscopy (FF-HEDM) data. It provides tools for calibration, indexing, grain mapping, and strain analysis.

## Overview

This repository contains a collection of C executables for processing FF-HEDM data. The code is designed to run on multi-core CPUs with optional CUDA acceleration for some components.

## Features

- Detector calibration
- Peak detection and fitting
- Grain indexing and refinement
- Strain analysis
- Multi-layer analysis
- Grain tracking
- 3D visualization support

## Build System

MIDAS/FF_HEDM now supports a modern CMake build system that offers better portability and maintainability.

## Requirements

- C compiler (GCC recommended)
- CMake 3.16 or higher
- CUDA toolkit (optional, for CUDA-accelerated components)
- Python 3.x with packages listed in `requirements.txt`
- Various libraries (NLOPT, HDF5, TIFF, etc.) - automatically downloaded by build system

## Quick Start

```bash
# Clone the repository
git clone https://github.com/marinerhemant/MIDAS.git
cd MIDAS/FF_HEDM

# Create build directory
mkdir build && cd build

# Configure and build
cmake -DUSE_CUDA=OFF ..
cmake --build . -j $(nproc)

# Install
cmake --install .

# Update after doing git pull and if you have MIDAS already installled
cmake --build . --target midas_executables -j $(nproc)

```

## Configuration Options

| Option                 | Default   | Description                                   |
|------------------------|-----------|-----------------------------------------------|
| CMAKE_INSTALL_PREFIX   | build/install | Installation directory                    |
| MIDAS_CONFIG_DIR       | $HOME/.MIDAS | Configuration directory                    |
| USE_USER_HOME          | ON        | Use home directory for config                 |
| USE_CUDA               | ON        | Build CUDA components                         |
| BUILD_OMP              | ON        | Build OpenMP components                       |
| BUILD_SHARED_LIBS      | ON        | Build shared libraries instead of static      |
| DOWNLOAD_DEPENDENCIES  | ON        | Download and build dependencies automatically |
| SYSTEM_DEPENDENCIES    | OFF       | Use system dependencies if available          |
| APPLE_RPATH_FIX        | OFF       | Apply rpath fix for macOS                     |
| INSTALL_PYTHON_DEPS    | ON        | Install Python dependencies                   |

## License

Copyright (c) 2014, UChicago Argonne, LLC  
See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Directory Structure

```
MIDAS/FF_HEDM/
├── bin/                # Generated executable files
├── cmake/              # CMake modules and templates
│   ├── deps/           # Dependency configuration files
│   ├── FindModules/    # Custom find modules
│   └── templates/      # Template files
├── v7/                 # Cluster scripts and utilities
├── src/                # Source code
├── CMakeLists.txt      # Main CMake configuration
└── requirements.txt    # Python dependencies
```

## Contact

For questions or support, please contact hsharma@anl.gov