# Installation Guide for MIDAS/FF_HEDM

This document provides detailed instructions for installing MIDAS/FF_HEDM using the CMake build system.

## Prerequisites

### Required Software

- **C Compiler**: GCC 7.0+ or Clang 6.0+
- **CMake**: Version 3.16 or higher
- **Git**: For cloning the repository
- **Python 3.x**: For scripts and visualization tools

### Optional Software

- **CUDA Toolkit**: For CUDA-accelerated components
- **OpenMP**: For multi-threading (usually included with compilers)

## Installation Methods

### Method 1: Standard Installation (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/marinerhemant/MIDAS.git
   cd MIDAS/FF_HEDM
   ```

2. **Create a build directory**:
   ```bash
   mkdir build
   cd build
   ```

3. **Configure the build**:
   ```bash
   cmake ..
   ```

4. **Build the software**:
   ```bash
   cmake --build . -j $(nproc)  # Use parallel build
   ```

5. **Install the software**:
   ```bash
   cmake --install .
   ```

6. **Set up the environment**:
   ```bash
   source install/bin/setup.sh
   ```

### Method 2: Custom Installation

For a custom installation with specific options:

1. **Follow steps 1-2 from Method 1**

2. **Configure with custom options**:
   ```bash
   cmake .. \
     -DCMAKE_INSTALL_PREFIX=/path/to/install \
     -DMIDAS_CONFIG_DIR=/path/to/config \
     -DBUILD_CUDA=ON \
     -DBUILD_OMP=ON \
     -DBUILD_SHARED_LIBS=ON \
     -DINSTALL_PYTHON_DEPS=ON \
     -DPYTHON_EXECUTABLE=/path/to/python
   ```

3. **Follow steps 4-6 from Method 1**

## Configuration Options

| Option                 | Default   | Description                                   |
|------------------------|-----------|-----------------------------------------------|
| CMAKE_INSTALL_PREFIX   | build/install | Installation directory                    |
| MIDAS_CONFIG_DIR       | $HOME/.MIDAS | Configuration directory                    |
| USE_USER_HOME          | ON        | Use home directory for config                 |
| BUILD_CUDA             | OFF       | Build CUDA components                         |
| BUILD_OMP              | ON        | Build OpenMP components                       |
| BUILD_SHARED_LIBS      | ON        | Build shared libraries instead of static      |
| DOWNLOAD_DEPENDENCIES  | ON        | Download and build dependencies automatically |
| SYSTEM_DEPENDENCIES    | OFF       | Use system dependencies if available          |
| APPLE_RPATH_FIX        | OFF       | Apply rpath fix for macOS                     |
| INSTALL_PYTHON_DEPS    | ON        | Install Python dependencies                   |

## Platform-Specific Instructions

### Linux

The default settings should work on most Linux distributions.

### macOS

For macOS, we recommend using GCC from Homebrew rather than Apple Clang:

```bash
# Install GCC from Homebrew
brew install gcc

# HIGHLY SUGGESTED TO USE SYSTEM DEPENDENCIES ON MAC AND USE HOMEBREW
brew install hdf5 libtiff fftw libzip nlopt

# Configure with GCC and macOS-specific options
cmake .. \
  -DCMAKE_C_COMPILER=$(brew --prefix gcc)/bin/gcc-14 \
  -DAPPLE_RPATH_FIX=ON
```

### Cluster Environments

For cluster environments, you may need to load modules first:

```bash
# Example for a cluster with module system
module load gcc/9.3.0
module load cmake/3.20.0
module load cuda/11.4.0
module load python/3.8.5

# Then proceed with the standard installation
```

## Dependency Management

The CMake build system automatically handles all dependencies:

```bash
# With all dependencies downloaded and built
cmake .. -DDOWNLOAD_DEPENDENCIES=ON

# Using system dependencies where available
cmake .. -DSYSTEM_DEPENDENCIES=ON

# HIGHLY SUGGESTED TO USE SYSTEM DEPENDENCIES ON MAC AND USE HOMEBREW
brew install hdf5 libtiff fftw libzip nlopt

```

## Python Dependencies

Python dependencies are automatically installed if `INSTALL_PYTHON_DEPS` is enabled:

```bash
# To manually install Python dependencies
python -m pip install --user -r requirements.txt
```

## Troubleshooting

### Common Issues

1. **CMake not found**: Ensure CMake is installed and in your PATH
   ```bash
   which cmake
   cmake --version
   ```

2. **Compiler issues**: Check your compiler installation
   ```bash
   gcc --version
   # or
   clang --version
   ```

3. **Library path issues**: Ensure libraries can be found
   ```bash
   # Check if environment setup was successful
   echo $LD_LIBRARY_PATH
   # or on macOS
   echo $DYLD_LIBRARY_PATH
   ```

4. **Missing Python packages**: Check Python package installation
   ```bash
   python -m pip list
   ```

### Getting Help

If you encounter issues not covered here, please:

1. Check the README.md file for additional information
2. Check GitHub issues for similar problems
3. Contact hsharma@anl.gov
4. In CMakeLists.txt, edit the project line: uncomment line 2 and comment line 3 in case CUDA is causing installation issues.