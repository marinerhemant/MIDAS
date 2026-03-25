# GPU Acceleration Guide

**Version:** 11.0
**Contact:** hsharma@anl.gov

MIDAS supports GPU-accelerated computation across all major analysis pipelines using NVIDIA CUDA. This guide covers building with GPU support, available GPU-accelerated executables, and usage.

---

## 1. Building with CUDA Support

GPU support requires the NVIDIA CUDA Toolkit (version 11.0 or later recommended).

### CMake Configuration

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
make -j$(nproc)
```

To target specific GPU architectures:

```bash
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
```

Common architecture values:

| Architecture | GPUs |
|---|---|
| 70 | V100 |
| 80 | A100, A30 |
| 86 | RTX 3090, A40 |
| 89 | RTX 4090, L40 |
| 90 | H100 |

The build system compiles the following CUDA targets when `USE_CUDA=ON`:

| Target | Module | Description |
|---|---|---|
| `IndexerGPU` | FF-HEDM | GPU-accelerated indexer |
| `FitPosOrStrainsGPU` | FF-HEDM | GPU-accelerated strain fitting |
| `IndexerScanningGPU` | PF-HEDM | GPU scanning-mode indexer |
| `FitOrStrainsScanningGPU` | PF-HEDM | GPU scanning-mode strain fitter |
| `FitOrientationGPU` | NF-HEDM | GPU orientation fitting |
| `IntegratorFitPeaksGPUStream` | Integration | GPU-accelerated radial integration with peak fitting |
| `tomo_gpu` (linked into MIDAS_TOMO) | Tomography | GPU-accelerated gridrec reconstruction |

All CUDA targets are compiled with `-Xcompiler=-fopenmp` for hybrid GPU+OpenMP parallelism.

---

## 2. FF-HEDM GPU Acceleration

### GPU Indexing and Fitting

Enable GPU acceleration in the FF-HEDM pipeline:

```bash
python FF_HEDM/workflows/ff_MIDAS.py -paramFN params.txt -useGPU 1
```

The `-useGPU 1` flag routes indexing through `IndexerGPU` and strain fitting through `FitPosOrStrainsGPU`.

**IndexerGPU** implements a two-pass funnel screening approach:

1. **Pass 1 (coarse):** Single-layer bitfield prefilter using a 32×32 tile occupancy grid (~1.5 MB, fits in L2 cache). Uses `__restrict__` pointers, `__ldg` texture loads, break-on-miss early termination, and loop unrolling.
2. **Pass 2 (fine):** Full multi-layer verification of Pass 1 candidates with post-filter diagonal approach.

**FitPosOrStrainsGPU** ports the NLOPT Nelder-Mead simplex algorithm to GPU, running per-grain refinement in parallel with device-side spot computation. Features dynamic spot reassignment and full strain tensor fitting.

### GPU Screening Only

To run only the screening pass (Phase 1) without refinement:

```bash
export MIDAS_SCREEN_ONLY=1
python FF_HEDM/workflows/ff_MIDAS.py -paramFN params.txt -useGPU 1
```

### Verbose Output

```bash
export MIDAS_VERBOSE=1
```

Enables per-voxel diagnostic output for debugging.

---

## 3. PF/Scanning HEDM GPU Acceleration

Enable GPU acceleration for scanning HEDM:

```bash
python FF_HEDM/workflows/pf_MIDAS.py -paramFN params.txt -useGPU 1
```

**IndexerScanningGPU** supports three indexing modes:

1. **Spot-driven** — with beam proximity filter for spatial awareness
2. **MicFile-seeded** — seeded from previous reconstruction
3. **GrainsFile-seeded** — seeded from Grains.csv

**FitOrStrainsScanningGPU** reads consolidated indexer output (`IndexBest_all.bin`, `IndexKey_all.bin`) and performs per-voxel Nelder-Mead refinement on GPU.

Both GPU executables use the consolidated binary I/O format, reducing filesystem overhead from ~30K+ small files to 3 binary files per scan.

---

## 4. NF-HEDM GPU Acceleration

Enable GPU-accelerated NF-HEDM orientation fitting:

```bash
python NF_HEDM/workflows/nf_MIDAS.py -paramFN params.txt -gpuFit 1
```

**FitOrientationGPU** accelerates both screening (Phase 1: discrete orientation search) and fitting (Phase 2: Nelder-Mead continuous refinement).

Features:

- Shared GPU math library (`nf_gpu.h`) with device functions for orientation matrix operations, diffraction spot calculation, and fractional overlap computation
- Port of NLOPT Nelder-Mead algorithm to GPU for exact CPU/GPU parity
- Batch processing of multiple voxels and orientations
- Constant memory for HKL tables, global memory for large arrays
- Optional double-precision mode for exact numerical parity

The `-gpuFit` flag works with both single-resolution (`nf_MIDAS.py`) and multi-resolution (`nf_MIDAS_Multiple_Resolutions.py`) workflows.

---

## 5. Radial Integration GPU Streaming

The GPU integrator provides real-time radial integration with peak fitting:

```bash
python FF_HEDM/workflows/integrator_batch_process.py -paramFN params.txt
```

**IntegratorFitPeaksGPUStream** features:

- Socket-based architecture for continuous data streaming
- 4 CUDA streams for overlapped computation
- Warp shuffle reductions for efficient summation
- GSAS-II area-normalized pseudo-Voigt peak fitting
- Integration with `live_viewer.py` for real-time visualization

Supports both folder-based file input and PVA (Process Variable Access) streaming from EPICS.

See [FF_Radial_Integration.md](FF_Radial_Integration.md) for full documentation.

---

## 6. Tomographic Reconstruction GPU

GPU-accelerated gridrec tomographic reconstruction is automatically used when built with CUDA support.

Features:

- Multi-pair batched reconstruction with dynamic batch sizing (capped at 50 pairs to limit pinned memory)
- Double-buffered pipeline with pthread overlap for compute/transfer
- 3-stream CUDA overlap for kernel execution
- Pinned memory for efficient host-device transfers
- OMP-parallel sinogram reads for GPU batch dispatch
- Pre-allocated per-thread scratch buffers
- mmap-based sinogram input for zero-copy parallel reads (both CPU and GPU paths)
- GPU-side Pad + reconCentering + getRecons kernels
- Stripe artifact removal on GPU path (Vo et al. 2018 algorithms)

See [Tomography_Reconstruction.md](Tomography_Reconstruction.md) for full documentation.

---

## 7. Precision Control

By default, GPU computations use single precision (float32) for performance. For applications requiring higher precision:

```bash
export MIDAS_GPU_DOUBLE=1
```

This enables double-precision computation in the GPU kernels. The performance impact depends on the GPU architecture — consumer GPUs (RTX series) have significantly reduced double-precision throughput compared to data-center GPUs (A100, H100).

Double precision has been verified to achieve exact parity with CPU results across all GPU-accelerated modules.

---

## 8. Environment Variables Summary

| Variable | Description |
|---|---|
| `MIDAS_GPU_DOUBLE=1` | Enable double-precision GPU computation |
| `MIDAS_GPU_FIT=1` | Enable GPU Phase 2 (fitting) — used internally |
| `MIDAS_SCREEN_ONLY=1` | Run only Phase 1 screening, skip fitting |
| `MIDAS_VERBOSE=1` | Enable per-voxel diagnostic output |

## 9. CLI Flags Summary

| Flag | Pipeline | Description |
|---|---|---|
| `-useGPU 1` | FF-HEDM, PF-HEDM | Route indexing and fitting through GPU executables |
| `-gpuFit 1` | NF-HEDM | Enable GPU orientation fitting (screening + refinement) |

---

## 10. Performance Notes

- GPU acceleration provides the largest speedup for NF-HEDM (thousands of voxels × thousands of orientations) and PF/scanning HEDM (many scan positions)
- FF-HEDM GPU indexing benefits from large grid sizes and many diffraction rings
- The GPU integrator is optimized for real-time streaming use cases
- Tomography GPU acceleration scales with the number of sinogram pairs and reconstruction size
- Memory usage: GPU executables pre-allocate scratch buffers and use pinned memory for efficient transfers
- All GPU modules maintain full CPU/GPU parity — results are identical (within floating-point precision for float32 mode, exact for float64 mode)

---

## 11. Testing GPU Parity

MIDAS includes benchmark and parity tests for GPU modules:

```bash
# NF-HEDM GPU parity
python tests/test_nf_hedm.py -nCPUs 4 --gpu-fit

# Tomography GPU vs CPU parity
python tests/test_tomo_parity.py --phantom-size 256 --plot

# PF-HEDM GPU
python tests/test_pf_hedm.py -nCPUs 4 -useGPU
```

Analysis scripts for parity debugging are in `NF_HEDM/Example/`:
- `analyze_mismatches.py` — per-voxel misorientation comparison with `--all` flag
- `parity_maps.py` — spatial confidence diff and misorientation maps

---

## See Also

- [FF_Analysis.md](FF_Analysis.md) — FF-HEDM analysis pipeline
- [NF_Analysis.md](NF_Analysis.md) — NF-HEDM reconstruction
- [PF_Analysis.md](PF_Analysis.md) — PF/scanning HEDM analysis
- [FF_Radial_Integration.md](FF_Radial_Integration.md) — Radial integration with GPU streaming
- [Tomography_Reconstruction.md](Tomography_Reconstruction.md) — Tomographic reconstruction
- [README.md](README.md) — MIDAS manual index

---

If you encounter any issues or have questions, please open an issue on this repository.
