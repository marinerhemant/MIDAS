<p align="center">
  <img src="logos/midas_logo.png" alt="MIDAS Logo" width="300">
</p>

# MIDAS — Microstructural Imaging using Diffraction Analysis Software

[![License](https://img.shields.io/badge/License-UChicago_Argonne-blue.svg)](LICENSE)

**MIDAS** is an open-source suite for reconstructing three-dimensional microstructures from High-Energy Diffraction Microscopy (HEDM) data. Developed at the [Advanced Photon Source](https://www.aps.anl.gov/) at Argonne National Laboratory, it supports the complete data-reduction pipeline — from raw detector frames to grain maps, strain tensors, spatially resolved orientation fields, and tomographic reconstructions.

**Contact:** [Hemant Sharma](mailto:hsharma@anl.gov?subject=[MIDAS]%20From%20Github) (hsharma@anl.gov)

---

## Key Capabilities

| Technique | What It Produces | Detector Distance |
|-----------|-----------------|-------------------|
| **Far-Field HEDM (FF-HEDM)** | Grain centroids, average orientations, full elastic strain tensors | ≈ 1 m |
| **Near-Field HEDM (NF-HEDM)** | Spatially resolved 3D orientation maps, grain morphology, grain boundary networks | ≈ 5–10 mm |
| **Point-Focus HEDM (PF-HEDM)** | High-resolution grain orientations from focused beam | ≈ 1 m |
| **Radial Integration (Caking)** | 1D intensity vs. 2θ profiles for Rietveld refinement (GSAS-II) | — |
| **Tomography (CT)** | Absorption-contrast cross-sections via gridrec algorithm | — |

---

## Repository Layout

```
MIDAS/
├── FF_HEDM/          # Far-field HEDM analysis codes (calibration, indexing, fitting, integration)
├── NF_HEDM/          # Near-field HEDM reconstruction codes
├── DT/               # Data transformation utilities
├── TOMO/             # Tomographic reconstruction (gridrec CT engine)
├── utils/            # Python utilities (integrator, auto-calibration, workflow drivers)
├── gui/              # Interactive visualization GUI
├── manuals/          # Comprehensive documentation (see below)
├── cmake/            # CMake build configuration and dependency management
├── build.sh          # Build script (Linux / macOS)
├── build_wsl_windows.sh  # Build script (Windows via WSL)
├── environment.yml   # Conda environment specification
├── CMakeLists.txt    # Top-level CMake configuration
└── LICENSE           # UChicago Argonne open-source license
```

---

## Documentation

Full manuals are in the **[manuals/](manuals/)** directory. Start with the [manuals README](manuals/README.md) for an overview of all HEDM techniques, coordinate systems, and a getting-started checklist.

| Manual | Topic |
|--------|-------|
| [FF_calibration](manuals/FF_calibration.md) | FF-HEDM geometry calibration |
| [FF_Analysis](manuals/FF_Analysis.md) | FF-HEDM grain indexing and fitting |
| [FF_RadialIntegration](manuals/FF_RadialIntegration.md) | Radial integration / caking |
| [FF_Interactive_Plotting](manuals/FF_Interactive_Plotting.md) | Interactive FF-HEDM visualization |
| [FF_dual_datasets](manuals/FF_dual_datasets.md) | Dual-dataset FF-HEDM analysis |
| [PF_Analysis](manuals/PF_Analysis.md) | Point-Focus HEDM analysis |
| [NF_calibration](manuals/NF_calibration.md) | NF-HEDM detector calibration |
| [NF_Analysis](manuals/NF_Analysis.md) | NF-HEDM reconstruction workflow |
| [NF_MultiResolution_Analysis](manuals/NF_MultiResolution_Analysis.md) | Multi-resolution NF-HEDM |
| [NF_gui](manuals/NF_gui.md) | NF-HEDM interactive GUI |
| [ForwardSimulationManual](manuals/ForwardSimulationManual.md) | Forward simulation for validation |
| [GSAS-II_Integration](manuals/GSAS-II_Integration.md) | Importing MIDAS output into GSAS-II |
| [Tomography_Reconstruction](manuals/Tomography_Reconstruction.md) | Absorption-contrast CT reconstruction |

---

## Installation

### Prerequisites

| Platform | Requirements |
|----------|-------------|
| **macOS** | Homebrew, LLVM, libomp, GCC, CMake, jemalloc |
| **Linux** | GCC ≥ 9, CMake ≥ 3.16 |
| **Windows** | WSL with Ubuntu |

MIDAS automatically downloads and builds these C/C++ dependencies during compilation: [NLOPT](https://nlopt.readthedocs.io/en/latest/), [LIBTIFF](http://www.libtiff.org/), [FFTW](http://www.fftw.org/), [HDF5](https://www.hdfgroup.org/solutions/hdf5/), [BLOSC](https://github.com/Blosc/c-blosc), [BLOSC-2](https://github.com/Blosc/c-blosc2), [ZLIB](https://zlib.net/), [LIBZIP](https://libzip.org/).

### Clone

```bash
git clone https://github.com/marinerhemant/MIDAS.git
cd MIDAS
```

### macOS

1. Install Homebrew (if not already installed):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
    Without sudo access, install to your home directory:
    ```bash
    mkdir homebrew && curl -L https://github.com/Homebrew/brew/tarball/main | tar xz --strip-components 1 -C homebrew
    eval "$(homebrew/bin/brew shellenv)"
    brew update --force --quiet
    chmod -R go-w "$(brew --prefix)/share/zsh"
    ```
    Add Homebrew to your PATH:
    ```bash
    echo 'eval $(/opt/homebrew/bin/brew shellenv)' >> ~/.zshrc
    source ~/.zshrc
    ```

2. Install dependencies:
    ```bash
    brew install llvm libomp gcc cmake jemalloc
    ```

3. Configure environment variables:
    ```bash
    echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
    echo 'export LDFLAGS="-L/opt/homebrew/opt/llvm/lib $LDFLAGS"' >> ~/.zshrc
    echo 'export CPPFLAGS="-I/opt/homebrew/opt/llvm/include $CPPFLAGS"' >> ~/.zshrc
    echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib $LDFLAGS"' >> ~/.zshrc
    echo 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include $CPPFLAGS"' >> ~/.zshrc
    echo 'CC=/opt/homebrew/opt/gcc/bin/gcc-15' >> ~/.zshrc
    echo 'export CC' >> ~/.zshrc
    echo 'CXX=/opt/homebrew/opt/gcc/bin/g++-15' >> ~/.zshrc
    echo 'export CXX' >> ~/.zshrc
    source ~/.zshrc
    ```

4. Build:
    ```bash
    ./build.sh
    ```

### Linux

```bash
./build.sh
```

### Windows (WSL)

```bash
sudo ./build_wsl_windows.sh
```

### Disabling CUDA

```bash
./build.sh --cuda OFF
```

### Building a Single Target

```bash
cd build
cmake --build . --target <TargetName>
```

### Python Environment

```bash
conda env create -f environment.yml
conda activate midas_env
```

---

## Quick Start

1. **Calibrate** the detector geometry → [FF_calibration](manuals/FF_calibration.md)
2. **Run FF-HEDM** grain indexing and fitting → [FF_Analysis](manuals/FF_Analysis.md)
3. **Visualize** results interactively → [FF_Interactive_Plotting](manuals/FF_Interactive_Plotting.md)
4. **Reconstruct NF-HEDM** orientation maps → [NF_Analysis](manuals/NF_Analysis.md)
5. **Validate** with forward simulation → [ForwardSimulationManual](manuals/ForwardSimulationManual.md)

See the [manuals README](manuals/README.md) for the full step-by-step checklist.

---

## Acknowledgments

- [SGInfo](http://cci.lbl.gov/sginfo/) library for HKL calculations
- [ODFPF](https://anisotropy.mae.cornell.edu/onr/Matlab/matlab-functions.html) package (Cornell) for misorientation functions

---

## License

MIDAS is released under the [UChicago Argonne open-source license](LICENSE).

Copyright © 2012, UChicago Argonne, LLC. All rights reserved.

> This product includes software produced by UChicago Argonne, LLC under Contract No. DE-AC02-06CH11357 with the Department of Energy.
