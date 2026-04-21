#!/usr/bin/env bash
# Install C build deps inside the manylinux_2_28 container used by cibuildwheel.
# Invoked via [tool.cibuildwheel.linux] before-all.
#
# Required for building MIDASCalibrant:
#   - HDF5 C + HL (FileReader.c)
#   - libTIFF    (FileReader.c)
#   - OpenMP     (gcc's libgomp, ships with manylinux_2_28's gcc-toolset)
#   - NLopt     — built via FetchContent by CMakeLists.txt, no system install.

set -euxo pipefail

dnf install -y epel-release || true
dnf install -y hdf5-devel libtiff-devel

ldconfig
