#!/bin/bash

# Simple build script for MIDAS/FF_HEDM using CMake

# Default values
BUILD_TYPE="Release"
CMAKE_GENERATOR="Unix Makefiles"
BUILD_DIR="build"
INSTALL_DIR=""
CONFIG_DIR=""
ENABLE_CUDA=OFF
ENABLE_OMP=ON
PYTHON_EXEC=""
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
USE_NINJA=0

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help             Show this help message"
    echo "  -d, --debug            Build in Debug mode"
    echo "  -r, --release          Build in Release mode (default)"
    echo "  -b, --build-dir DIR    Set build directory (default: build)"
    echo "  -i, --install-dir DIR  Set installation directory"
    echo "  -c, --config-dir DIR   Set configuration directory"
    echo "  --cuda                 Enable CUDA components"
    echo "  --no-omp               Disable OpenMP components"
    echo "  --python PATH          Specify Python executable"
    echo "  -j, --jobs N           Number of parallel jobs (default: auto)"
    echo "  --ninja                Use Ninja generator instead of Makefiles"
    echo "  --clean                Clean the build directory before building"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -i|--install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -c|--config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --cuda)
            ENABLE_CUDA=ON
            shift
            ;;
        --no-omp)
            ENABLE_OMP=OFF
            shift
            ;;
        --python)
            PYTHON_EXEC="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --ninja)
            USE_NINJA=1
            CMAKE_GENERATOR="Ninja"
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Set install options
INSTALL_OPTIONS=""
if [ -n "$INSTALL_DIR" ]; then
    INSTALL_OPTIONS="-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
fi

# Set config options
CONFIG_OPTIONS=""
if [ -n "$CONFIG_DIR" ]; then
    CONFIG_OPTIONS="-DMIDAS_CONFIG_DIR=${CONFIG_DIR} -DUSE_USER_HOME=OFF"
fi

# Set Python options
PYTHON_OPTIONS=""
if [ -n "$PYTHON_EXEC" ]; then
    PYTHON_OPTIONS="-DPYTHON_EXECUTABLE=${PYTHON_EXEC}"
fi

# Handle platform-specific options
PLATFORM_OPTIONS=""
if [ "$(uname)" == "Darwin" ]; then
    PLATFORM_OPTIONS="-DAPPLE_RPATH_FIX=ON"
    
    # Check if GCC is available from Homebrew
    if command -v brew &> /dev/null; then
        GCC_PREFIX=$(brew --prefix gcc 2>/dev/null)
        if [ -n "$GCC_PREFIX" ] && [ -x "$GCC_PREFIX/bin/gcc-14" ]; then
            PLATFORM_OPTIONS="$PLATFORM_OPTIONS -DCMAKE_C_COMPILER=$GCC_PREFIX/bin/gcc-14"
        fi
    fi
fi

# Clean build directory if requested
if [ -n "$CLEAN_BUILD" ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

# Configure
echo "Configuring CMake..."
cmake .. \
    -G "$CMAKE_GENERATOR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_CUDA="$ENABLE_CUDA" \
    -DBUILD_OMP="$ENABLE_OMP" \
    $INSTALL_OPTIONS \
    $CONFIG_OPTIONS \
    $PYTHON_OPTIONS \
    $PLATFORM_OPTIONS

# Build
echo "Building with $JOBS jobs..."
cmake --build . -j "$JOBS"

# Installation message
echo "Build completed successfully!"
echo "To install, run: cmake --install ."
echo "Then run: source $([ -n "$INSTALL_DIR" ] && echo "$INSTALL_DIR" || echo "$(pwd)/install")/bin/setup.sh"