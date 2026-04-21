#!/usr/bin/env bash
# Install C build deps on the cibuildwheel macOS runner.
# Invoked via [tool.cibuildwheel.macos] before-all.
#
# Required for building MIDASCalibrant:
#   - HDF5 C + HL (FileReader.c)
#   - libTIFF     (FileReader.c)
#   - libomp      (OpenMP; not bundled with Apple clang)
#   - NLopt       — built via FetchContent by CMakeLists.txt, no system install.

set -euxo pipefail

brew update
brew install hdf5 libtiff libomp

LIBOMP_PREFIX="$(brew --prefix libomp)"
HDF5_PREFIX="$(brew --prefix hdf5)"
LIBTIFF_PREFIX="$(brew --prefix libtiff)"

{
    echo "OpenMP_ROOT=${LIBOMP_PREFIX}"
    echo "HDF5_ROOT=${HDF5_PREFIX}"
    echo "CMAKE_PREFIX_PATH=${LIBOMP_PREFIX}:${HDF5_PREFIX}:${LIBTIFF_PREFIX}"
} >> "${GITHUB_ENV:-/dev/null}" || true

export OpenMP_ROOT="${LIBOMP_PREFIX}"
export HDF5_ROOT="${HDF5_PREFIX}"
export CMAKE_PREFIX_PATH="${LIBOMP_PREFIX}:${HDF5_PREFIX}:${LIBTIFF_PREFIX}:${CMAKE_PREFIX_PATH:-}"
