#!/usr/bin/env bash
# Install C build deps for cibuildwheel macOS.
# Invoked via [tool.cibuildwheel.macos] before-all.

set -euxo pipefail

brew update
brew install hdf5 libtiff libomp c-blosc2 libzip

LIBOMP_PREFIX="$(brew --prefix libomp)"
HDF5_PREFIX="$(brew --prefix hdf5)"
LIBTIFF_PREFIX="$(brew --prefix libtiff)"
BLOSC2_PREFIX="$(brew --prefix c-blosc2)"
LIBZIP_PREFIX="$(brew --prefix libzip)"

{
    echo "OpenMP_ROOT=${LIBOMP_PREFIX}"
    echo "HDF5_ROOT=${HDF5_PREFIX}"
    echo "CMAKE_PREFIX_PATH=${LIBOMP_PREFIX}:${HDF5_PREFIX}:${LIBTIFF_PREFIX}:${BLOSC2_PREFIX}:${LIBZIP_PREFIX}"
} >> "${GITHUB_ENV:-/dev/null}" || true

export OpenMP_ROOT="${LIBOMP_PREFIX}"
export HDF5_ROOT="${HDF5_PREFIX}"
export CMAKE_PREFIX_PATH="${LIBOMP_PREFIX}:${HDF5_PREFIX}:${LIBTIFF_PREFIX}:${BLOSC2_PREFIX}:${LIBZIP_PREFIX}:${CMAKE_PREFIX_PATH:-}"
