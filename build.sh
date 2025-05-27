#!/bin/bash

# Build script for the MIDAS project

# Default values
BUILD_TYPE="Release"
CMAKE_GENERATOR="Unix Makefiles"
BUILD_DIR="build"
INSTALL_DIR=""
ENABLE_CUDA="ON" # Matches top-level CMakeLists.txt option default
ENABLE_OMP="ON"  # Matches top-level CMakeLists.txt option default
DOWNLOAD_DEPS="ON" # Matches top-level CMakeLists.txt option default
USE_SYSTEM_DEPS="OFF" # Matches top-level CMakeLists.txt option default
PYTHON_EXEC=""
INSTALL_PY_SCRIPTS="ON"
INSTALL_PY_DEPS="ON"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2) # Number of parallel jobs
USE_NINJA=0
CLEAN_BUILD=0

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -d, --debug               Build in Debug mode"
    echo "  -r, --release             Build in Release mode (default: Release)"
    echo "  -b, --build-dir DIR       Set build directory (default: build)"
    echo "  -i, --install-dir DIR     Set CMAKE_INSTALL_PREFIX"
    echo "  --cuda ON|OFF             Enable/Disable CUDA components (default: $ENABLE_CUDA)"
    echo "  --omp ON|OFF              Enable/Disable OpenMP components (default: $ENABLE_OMP)"
    echo "  --download-deps ON|OFF    Enable/Disable downloading dependencies (default: $DOWNLOAD_DEPS)"
    echo "  --system-deps ON|OFF      Prefer system-installed dependencies (default: $USE_SYSTEM_DEPS)"
    echo "  --python PATH             Specify Python executable for installing deps"
    echo "  --install-py-scripts ON|OFF Install Python utility scripts (default: $INSTALL_PY_SCRIPTS)"
    echo "  --install-py-deps ON|OFF  Install Python dependencies via pip (default: $INSTALL_PY_DEPS)"
    echo "  -j, --jobs N              Number of parallel jobs for build (default: auto)"
    echo "  --ninja                   Use Ninja generator instead of Makefiles"
    echo "  --clean                   Clean the build directory before building"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help ;;
        -d|--debug) BUILD_TYPE="Debug"; shift ;;
        -r|--release) BUILD_TYPE="Release"; shift ;;
        -b|--build-dir) BUILD_DIR="$2"; shift 2 ;;
        -i|--install-dir) INSTALL_DIR="$2"; shift 2 ;;
        --cuda) ENABLE_CUDA="$2"; shift 2 ;;
        --omp) ENABLE_OMP="$2"; shift 2 ;;
        --download-deps) DOWNLOAD_DEPS="$2"; shift 2 ;;
        --system-deps) USE_SYSTEM_DEPS="$2"; shift 2 ;;
        --python) PYTHON_EXEC="$2"; shift 2 ;;
        --install-py-scripts) INSTALL_PY_SCRIPTS="$2"; shift 2 ;;
        --install-py-deps) INSTALL_PY_DEPS="$2"; shift 2 ;;
        -j|--jobs) JOBS="$2"; shift 2 ;;
        --ninja) USE_NINJA=1; CMAKE_GENERATOR="Ninja"; shift ;;
        --clean) CLEAN_BUILD=1; shift ;;
        *) echo "Unknown option: $1"; show_help ;;
    esac
done

# CMake options list
CMAKE_OPTIONS=()
CMAKE_OPTIONS+=("-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
CMAKE_OPTIONS+=("-DUSE_CUDA=${ENABLE_CUDA}")
CMAKE_OPTIONS+=("-DBUILD_OMP=${ENABLE_OMP}")
CMAKE_OPTIONS+=("-DDOWNLOAD_DEPENDENCIES=${DOWNLOAD_DEPS}")
CMAKE_OPTIONS+=("-DUSE_SYSTEM_DEPS=${USE_SYSTEM_DEPS}")
CMAKE_OPTIONS+=("-DINSTALL_PYTHON_SCRIPTS=${INSTALL_PY_SCRIPTS}")
CMAKE_OPTIONS+=("-DINSTALL_PYTHON_DEPENDENCIES=${INSTALL_PY_DEPS}")

if [ -n "$INSTALL_DIR" ]; then
    CMAKE_OPTIONS+=("-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}")
fi

if [ -n "$PYTHON_EXEC" ]; then
    CMAKE_OPTIONS+=("-DPYTHON_EXECUTABLE=${PYTHON_EXEC}")
fi

# Platform-specific options (example from FF_HEDM, may or may not be needed globally)
# if [ "$(uname)" == "Darwin" ]; then
#     CMAKE_OPTIONS+=("-DAPPLE_RPATH_FIX=ON") # This was an FF_HEDM option, check if needed globally
# fi

# Clean build directory if requested
if [ "$CLEAN_BUILD" -eq 1 ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || { echo "Error: Could not enter build directory $BUILD_DIR"; exit 1; }

# Configure
echo "Configuring CMake with options: \${CMAKE_OPTIONS[*]}"
cmake .. \
    -G "$CMAKE_GENERATOR" \
    "${CMAKE_OPTIONS[@]}"

# Build
echo "Building with $JOBS jobs..."
cmake --build . -j "$JOBS"

# Installation message
BUILD_SUCCESS=$?
if [ $BUILD_SUCCESS -eq 0 ]; then
    echo ""
    echo "Build completed successfully!"
else
    echo ""
    echo "Build failed."
    exit 1
fi

cd ..
