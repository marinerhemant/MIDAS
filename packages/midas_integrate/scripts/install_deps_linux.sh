#!/usr/bin/env bash
# Install C build deps inside manylinux_2_28 for cibuildwheel.
# Invoked via [tool.cibuildwheel.linux] before-all.
#
# Required for MIDASIntegrator + MIDASDetectorMapper:
#   - HDF5 C + HL  (FileReader.c)
#   - libTIFF      (FileReader.c)
#   - BLOSC2       (ZarrReader.c)
#   - libzip       (ZarrReader.c)
#   - OpenMP       (gcc libgomp, ships with manylinux_2_28's gcc-toolset)
#   - NLopt        — FetchContent'd by CMakeLists.txt, no system install.

set -euxo pipefail

dnf install -y epel-release || true
dnf install -y \
    hdf5-devel \
    libtiff-devel \
    blosc2-devel \
    libzip-devel

ldconfig
