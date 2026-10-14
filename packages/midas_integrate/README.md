# midas-integrate

Pure-Python radial integration for area X-ray detectors. A drop-in,
pip-installable replacement for the MIDAS C/CUDA radial integration
pipeline — no compilers, no native libraries, no CMake.

```bash
pip install midas-integrate
```

## What's in the box

| C/CUDA source                          | Python module                          |
| -------------------------------------- | -------------------------------------- |
| `MapperCore.c`, `DetectorGeometry.c`, `DetectorMapper.c` | `midas_integrate.detector_mapper`, `midas_integrate.geometry` |
| `IntegratorFitPeaksGPUStream.cu` (GPU streaming) | `midas_integrate.kernels`, `midas_integrate.server`, `midas_integrate.pipeline` |
| `IntegratorZarrOMP.c` (CPU OMP, bilinear) | `midas_integrate.kernels` (`mode='bilinear'`) |
| `PeakFit.c`                            | `midas_integrate.peakfit`              |
| `PeakFitIO.c`                          | `midas_integrate.peak_io`              |
| `Map.bin` / `nMap.bin`                 | `midas_integrate.bin_io`               |

## CPU/GPU selection

Everything that touches arrays accepts a `device` argument that is forwarded
to PyTorch. CPU and CUDA are first-class:

```python
from midas_integrate import build_csr, integrate, profile_1d, load_map

pixmap = load_map('Map.bin', 'nMap.bin')
geom_cpu  = build_csr(pixmap, n_r=990, n_eta=72, n_pixels_y=1475, n_pixels_z=1679, device='cpu')
geom_cuda = build_csr(pixmap, ..., device='cuda')

import torch
img = torch.from_numpy(image_2d)
profile = profile_1d(integrate(img, geom_cuda, mode='bilinear'), geom_cuda)
```

## Three integration modes (full parity with C codes)

| `mode`      | Equivalent C kernel                                   | Use when |
|-------------|--------------------------------------------------------|----------|
| `'floor'`   | `integrate_noMapMask` in `IntegratorFitPeaksGPUStream.cu` | streaming, max throughput |
| `'bilinear'`| pixel loop in `IntegratorZarrOMP.c` lines 1733–1744    | offline analysis, max accuracy |
| `'gradient'`| `GradientCorrection=1` branch in the GPU stream         | strong tilt + small R |

## Pixel weighting (`pixel_weights`)

By default (`pixel_weights="area"`) a pixel counts in a bin by its overlap area with the bin in the projected plane, so a bin value is the
area-weighted mean of the pixels it covers. `build_csr(..., pixel_weights="fraction")`, or `--pixel-weights fraction` on `midas-integrate` and
`midas-integrate-server`, gives every pixel the same total weight (its weights over all bins sum to 1), the convention of pyFAI's split pixel and of
azint. The two profiles are identical for a constant image (tested) and, for a synthetic ring that does not vary with azimuth, differ by at most 0.15 microstrain in our test; they differ more when the ring position or intensity varies with azimuth.
We measured the size on one real frame at 14 degrees tilt (differences up to about 1 microstrain) and on a synthetic ring whose position varies by
500 microstrain around the azimuth (about 10 microstrain); both are provisional measurements (one frame, one synthetic family), not a bound.
Pixels cut by the radial range limits are renormalised to 1 inside the range, so the bins at the very ends of the range differ from pyFAI's. The
option is applied by `build_csr` only; `build_fused_geometry` always uses area weights. Absolute intensities also differ between integrators by a
solid-angle normalisation constant of order `1/cos^2(alpha) - 1` with `cos(alpha) = cos(ty) cos(tz)` (3.1 % for a pure 10 degree tilt and 13.2 % for
20 degrees; 3.6 % and 15.5 % for tilts of ty = 10, tz = -4 and ty = 20, tz = -8 degrees), so compare profiles from different tools after fixing that
constant.

## CLI

Three entry points mirror the C binaries:

```bash
# 1. Build Map.bin / nMap.bin from a parameter file (one-shot, slow):
midas-detector-mapper params.txt -j 8

# 2. Integrate one frame (one-shot, fast):
midas-integrate params.txt --image frame.tif --device cuda

# 3. Streaming socket server (matches the C wire protocol on port 60439):
midas-integrate-server params.txt --device cuda --num-streams 4
```

## Numerical parity

- DetectorMapper output (`Map.bin`/`nMap.bin`): byte-equivalent to the C
  version (entry counts, sums of `frac` and `areaWeight` per bin agree to
  ULP; entry order within a bin may differ).
- Per-frame integration: float32 ULP-level (median 1.7e-8 relative error
  vs the C/CUDA `IntegratorFitPeaksGPUStream` binary on PILATUS3 2M with
  CeO₂ data; max 2.1e-7 relative).
- Peak fitting: same model (pseudo-Voigt + global background, SNIP
  background subtraction), different optimizer (scipy LM vs NLopt
  Nelder-Mead). Fit parameters typically agree to ~1e-5 relative on
  noisy real data.

## Performance (PILATUS3 2M, 1475×1679, NVIDIA H100)

|                                              | Throughput       |
|----------------------------------------------|------------------|
| C MIDAS GPU stream (per the paper)           | ~1,600 fps       |
| **midas-integrate (PyTorch CSR, FP32, CUDA)**| **~3,250 fps**   |
| midas-integrate (PyTorch CSR, FP64, CPU)     | ~675 fps         |
| C MIDAS CPU (per the paper)                  | ~262 fps         |
| pyFAI CSR-cython (per the paper)             | ~7 fps           |

## Using a calibration from `midas-calibrate-v2`

`midas-calibrate-v2` (the differentiable Bayesian rewrite of the
calibration framework) exposes parameters under canonical names
(`iso_R2`, `iso_R4`, `iso_R6`, `a1`/`phi1`, …, `a6`/`phi6`) instead of
v1's `p0`..`p14`. Two routes exist for getting a v2 result into this
package:

```python
# (a) In-memory: drop the v2 unpacked dict straight onto a v1 template
from midas_integrate.params import parse_params
from midas_integrate.compat.from_v2 import params_from_v2_unpacked

template = parse_params("seed_paramstest.txt")  # carries binning, masks, …
ip = params_from_v2_unpacked(res.unpacked, template=template)
```

```python
# (b) High-level: spline + δr_k sidecars + paramstest in one call
from midas_calibrate_v2.compat.to_integrate import to_integrate_params

ip = to_integrate_params(
    res,                      # PVCalibrationResult or FourStageResult
    template=template,
    output_dir="./integrate_in",
    ring_d_spacing_A=ring_d, ring_two_theta_deg=ring_tt,  # for δr_k JSON
)
```

What the adapter does for you:

- Remaps `iso_R*` / `a*` / `phi*` to v1's `p0`..`p14` slots.
- Carries `Lsd`, `BC_y/BC_z`, `tx/ty/tz`, `Parallax`, `pxY/pxZ`,
  `Wavelength` through unchanged.
- Warns when v2-only parameters are dropped (per-panel blocks → use
  `midas_calibrate_v2.compat.to_v1.write_panel_shifts_file`; per-ring
  `δr_k` → use the JSON sidecar from `to_integrate_params` and apply at
  the peak-fit / Rietveld stage, since v1's radial map has no per-ring
  concept).
- For a `FourStageResult`, evaluates the Stage 4 thin-plate spline on
  the detector grid and writes the binary `ΔR(Y, Z)` lookup that
  `IntegrationParams.ResidualCorrectionMap` consumes.

The cache hash (`Map.bin` / `nMap.bin`) covers all 15 distortion
coefficients and `Parallax` / `Wavelength` unconditionally, so changing
any v2 parameter will correctly invalidate stale caches.

## Requirements

- Python ≥ 3.9
- numpy, scipy, torch, tifffile, h5py, joblib

CUDA support is automatic if your installed `torch` has CUDA. macOS Metal (MPS)
works for the integration kernel; sparse-CSR support on MPS is partial as of
torch 2.7.

## License

BSD-3-Clause. See LICENSE.
