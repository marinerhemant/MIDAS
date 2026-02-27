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
RUN_TESTS=""  # Empty means no tests; can be 'ff', 'nf', or 'all'

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
    echo "  --test ff|nf|calib|all   Run benchmark tests after build (ff, nf, calib, or all)"
    exit 0
}

check_and_download_seeds() {
    local SEED_DIR="NF_HEDM/seedOrientations"
    local REQUIRED_SEEDS=("cubicSeed.txt" "hexagonalSeed.txt" "monoclinicSeed.txt" "orthorhombicSeed.txt" "tetragonalSeed.txt" "triclinicSeed.txt" "trigonalSeed.txt")
    local MISSING_SEEDS=0

    # Check if directory exists
    if [ ! -d "$SEED_DIR" ]; then
        MISSING_SEEDS=1
    else
        # Check for each file
        for seed in "${REQUIRED_SEEDS[@]}"; do
            if [ ! -f "$SEED_DIR/$seed" ]; then
                MISSING_SEEDS=1
                break
            fi
        done
    fi

    # Download and extract if missing
    if [ "$MISSING_SEEDS" -eq 1 ]; then
        echo "Missing seed files in $SEED_DIR. Downloading..."
        mkdir -p "$SEED_DIR"
        
        # Download
        if command -v curl >/dev/null 2>&1; then
            curl -L -o "$SEED_DIR/seed.zip" "https://github.com/marinerhemant/MIDAS/releases/download/v9.1-data/seed.zip"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "$SEED_DIR/seed.zip" "https://github.com/marinerhemant/MIDAS/releases/download/v9.1-data/seed.zip"
        else
            echo "Error: Neither curl nor wget found. Cannot download seed files."
            exit 1
        fi

        # Unzip
        if command -v unzip >/dev/null 2>&1; then
            unzip -o "$SEED_DIR/seed.zip" -d "$SEED_DIR"
        else
             echo "Error: unzip command not found."
             exit 1
        fi

        # Cleanup
        rm "$SEED_DIR/seed.zip"
        if [ -d "$SEED_DIR/__MACOSX" ]; then
            rm -rf "$SEED_DIR/__MACOSX"
        fi
        echo "Seed files downloaded and extracted successfully."
    else
        echo "Seed files already exist."
    fi
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
        --test) RUN_TESTS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; show_help ;;
    esac
done

# Check and download seeds
check_and_download_seeds

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

    # Touch the update-check timestamp so the 14-day reminder resets
    touch ".last_update_check"

    # Install git post-merge hook (reminds users to rebuild after git pull)
    HOOK_SRC="../hooks/post-merge"
    HOOK_DST="../.git/hooks/post-merge"
    if [ -f "$HOOK_SRC" ] && [ -d "../.git/hooks" ]; then
        cp "$HOOK_SRC" "$HOOK_DST"
        chmod +x "$HOOK_DST"
        echo "Installed git post-merge hook."
    fi
else
    echo ""
    echo "Build failed."
    exit 1
fi

cd ..

# Run benchmarks if requested
if [ -n "$RUN_TESTS" ]; then
    echo ""
    echo "Running benchmark tests..."
    
    # Detect Python
    if [ -n "$PYTHON_EXEC" ]; then
        PY_CMD="$PYTHON_EXEC"
    else
        PY_CMD="python"
    fi

    TEST_CPUS="$JOBS"

    if [ "$RUN_TESTS" = "ff" ] || [ "$RUN_TESTS" = "all" ]; then
        echo ""
        echo "=== Running FF-HEDM Benchmark ==="
        $PY_CMD utils/test_ff_hedm.py -nCPUs "$TEST_CPUS"
        if [ $? -ne 0 ]; then
            echo "FF-HEDM benchmark FAILED."
            exit 1
        fi
    fi

    if [ "$RUN_TESTS" = "nf" ] || [ "$RUN_TESTS" = "all" ]; then
        echo ""
        echo "=== Running NF-HEDM Benchmark ==="
        $PY_CMD utils/test_nf_hedm.py -nCPUs "$TEST_CPUS"
        if [ $? -ne 0 ]; then
            echo "NF-HEDM benchmark FAILED."
            exit 1
        fi
    fi

    if [ "$RUN_TESTS" = "calib" ] || [ "$RUN_TESTS" = "all" ]; then
        echo ""
        echo "=== Running FF-HEDM Calibration Benchmark ==="
        $PY_CMD utils/test_ff_calibration.py -nCPUs "$TEST_CPUS"
        if [ $? -ne 0 ]; then
            echo "FF-HEDM Calibration benchmark FAILED."
            exit 1
        fi
    fi

    echo ""
    echo "All requested benchmarks completed successfully."
fi
