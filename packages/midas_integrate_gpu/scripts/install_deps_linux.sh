#!/usr/bin/env bash
# Install C build deps inside the CUDA 12.4 dev container used by cibuildwheel.
# Invoked via [tool.cibuildwheel.linux] before-all.
#
# The manylinux image for this package is nvidia/cuda:12.4.1-devel-ubuntu22.04,
# which already bundles nvcc + the CUDA runtime. We still need host-side
# packages: NLopt is FetchContent'd by CMake, but we need build tools +
# OpenMP headers.

set -euxo pipefail

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgomp1 \
    libpthread-stubs0-dev \
    pkg-config \
    python3-dev

# Ensure libstdc++ is available for any C++ bits the CUDA toolkit needs.
ldconfig
