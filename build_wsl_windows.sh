#!/bin/bash
# 1. Update and install all required development libraries
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential cmake pkg-config python3-dev \
    zlib1g-dev libjpeg-dev libtiff-dev libpng-dev libssl-dev libbz2-dev \
    liblzma-dev libzstd-dev liblz4-dev libzip-dev \
    libhdf5-dev libblosc-dev libblosc2-dev libnlopt-dev libfftw3-dev

# 2. Fix WSL clock sync (prevents "Clock skew detected" errors)
sudo hwclock -s

# 3. Setup Project Directory
mkdir -p ~/opt && cd ~/opt
[ -d "MIDAS" ] || git clone https://github.com/marinerhemant/MIDAS
cd MIDAS

# 4. Neutralize "Custom" build scripts that force x86/Intel optimizations
# This forces the project to use the ARM-compatible system libraries instead
mkdir -p cmake/deps/backup
mv cmake/deps/*_custom.cmake cmake/deps/backup/ 2>/dev/null || true

# 5. Clean and Configure Build
rm -rf build && mkdir build && cd build

# We use CMAKE_C_STANDARD_LIBRARIES and CMAKE_CXX_STANDARD_LIBRARIES 
# to ensure the linker order is correct (libraries at the end of the command)
HDF5_INC="/usr/include/hdf5/serial"
HDF5_LIB_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial"
LIBS="-L${HDF5_LIB_DIR} -lhdf5_hl -lhdf5 -ltiff -lblosc2 -lblosc -lfftw3 -lnlopt -lz -lm -ldl"

cmake .. \
  -DUSE_SYSTEM_DEPS=ON \
  -DDOWNLOAD_DEPENDENCIES=OFF \
  -DINSTALL_PYTHON_DEPENDENCIES=OFF \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF \
  -DCMAKE_C_FLAGS="-I${HDF5_INC}" \
  -DCMAKE_CXX_FLAGS="-I${HDF5_INC}" \
  -DCMAKE_C_STANDARD_LIBRARIES="${LIBS}" \
  -DCMAKE_CXX_STANDARD_LIBRARIES="${LIBS}" \
  -Wno-dev

# 6. Build using all available CPU cores
make -j$(nproc)
