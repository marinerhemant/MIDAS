# Migrating from pyFAI to `midas-integrate-v2`

If you currently use [pyFAI](https://github.com/silx-kit/pyFAI) for
azimuthal / radial integration of 2-D detector frames, this guide
shows you the equivalent `midas-integrate-v2` calls and what's
genuinely different.

## TL;DR

| pyFAI | `midas-integrate-v2` | Notes |
|-------|----------------------|-------|
| `AzimuthalIntegrator(dist=, poni1=, poni2=, ...)` | `IntegrationSpec` | Uses pixel-based BC instead of PONI. **Conversion: `poni = (BC + 0.5) · pixel_size`** — pyFAI is pixel-corner, MIDAS is pixel-centre (the 0.5 px gotcha). See "Coordinate convention" below. |
| `ai.integrate1d(image, n_r)` | `integrate_polygon(image, geom)` then `prof = int2d.mean(dim=0)` | v2 returns 2D `(η, R)` first; collapse over η for a 1D profile. |
| `ai.integrate2d(image, n_r, n_eta)` | `integrate_polygon(image, geom)` returns 2D directly | Same shape, same convention. |
| `ai.set_mask(mask)` | `geom = PolygonBinGeometry.from_spec(spec, mask=mask)` | Mask applied at build time. |
| `error_model='poisson'` | `integrate_polygon_with_variance(image, geom)` | Returns `(mean, σ)` per bin. |
| `pyFAI.calibrant.Calibrant("CeO2")` | `from midas_integrate_v2 import CALIBRANTS; CALIBRANTS["ceo2"]` | List of d-spacings. |
| (manually flag bad pixels) | `LearnableMask` + train | **Differentiation**: v2 learns the mask from your calibrant frame. |
| `ai.calib2()` (interactive GUI) | `estimate_initial_spec(...)` + notebook 03 refinement | v2 is non-interactive; calibration runs in code/notebook. |

## Conceptual differences

### Coordinate convention — **pixel-corner vs pixel-centre (the 0.5 px gotcha)**

This is the most common source of "I get a different answer" when
moving between pyFAI and MIDAS. **MIDAS BC and pyFAI PONI differ by
exactly 0.5 px**:

- **pyFAI's PONI** uses the **pixel-corner** convention. Pixel index
  `(0, 0)` corresponds to the corner of the first pixel; the centre
  of pixel `(i, j)` is at coordinate `(i + 0.5, j + 0.5)`.
- **MIDAS's BC** uses the **pixel-centre** convention. Pixel index
  `(0, 0)` IS the centre of the first pixel; the centre of pixel
  `(i, j)` is at coordinate `(i, j)`.

**Forward conversion (BC → PONI):**

```python
poni1_m = (BC_y_px + 0.5) * pxY_um * 1e-6
poni2_m = (BC_z_px + 0.5) * pxZ_um * 1e-6
```

**Reverse (PONI → BC):**

```python
BC_y_px = poni1_m / (pxY_um * 1e-6) - 0.5
BC_z_px = poni2_m / (pxZ_um * 1e-6) - 0.5
```

If you skip the `+ 0.5` you'll get a calibration that's off by half a
pixel in beam centre — small but systematic, enough to shift Bragg
peaks visibly at high R and to break apparent vs. true d-spacings at
the per-mille level. The MIDAS calibration paper uses this convention
explicitly when comparing to pyFAI baselines (the 0.5 px shift is
applied at the demo / parity-test stage in
``midas_calibrate_v2/dev/demo/demo_pyfai_vs_v2.py``).

This convention difference applies *only* to BC ↔ PONI conversion.
Everything else (Lsd in metres, wavelength in metres, distortion
parameters) round-trips between the two packages without a shift.

For a tilted detector the PONI definition gets fiddly because pyFAI
parametrises tilts via `rot1/rot2/rot3` (Tait-Bryan around the PONI),
while MIDAS uses `tx/ty/tz` (rotations of the detector plane). For
small tilts (< few degrees) they coincide; for larger tilts you may
want to refine MIDAS tilts from scratch on the same image rather than
algebraically converting.

### Distortion model

- pyFAI supports radial distortion via separate `splineFile=` files
  (Fit2D-format spline calibrations, typically detector-vendor
  supplied).
- `midas-integrate-v2` uses the v2 distortion (15 parameters:
  isotropic radial `iso_R2/R4/R6` plus 6 azimuthal harmonics with
  amplitude+phase). All refinable jointly with geometry.

To translate a pyFAI spline correction to v2:
- If you have a vendor spline, evaluate it on the detector grid →
  binary `ΔR(Y, Z)` lookup; pass to integrate via the
  `RBFResidualCorrection` class (notebook 03).
- For new calibrations, fit the v2 distortion model directly via
  `midas_calibrate_v2.autocalibrate_pv` (calibrate-v2's notebook
  series).

### Integration kernels

- **pyFAI's "splitpixel"** mode does subpixel oversampling (their
  defaults are 8×8). Equivalent: `SubpixelBinGeometry.from_spec(spec, K=8)`.
- **pyFAI's "splitbbox"** mode uses an axis-aligned bounding box
  approximation. Equivalent: roughly `HardBinGeometry`.
- **MIDAS's polygon-area kernel** (`PolygonBinGeometry`): exact
  Green's-theorem polygon-arc-arc intersection. **No equivalent in
  pyFAI.** This is the MIDAS differentiator — sub-bin precision in
  peak position without any oversampling artefact.

### Variance / uncertainty

- pyFAI: `error_model='poisson'` returns `(I, sigma)` per bin.
- v2: `integrate_polygon_with_variance(image, geom)` returns the same
  thing. The math is identical (standard error propagation).

### Mask

- pyFAI: `ai.set_mask(mask_array)` — static input; user-managed.
- v2: same static-mask path via `from_spec(..., mask=mask)`. **PLUS**
  v2 has `LearnableMask` — the mask is auto-learned from the
  calibrant frame at calibration time (notebook 10). No equivalent in
  pyFAI.

### Multi-image / sweep mode

- pyFAI: integrate one frame at a time; user manages the loop and
  output collection.
- v2: `integrate_stream(spec, FrameSource, ...)` handles the loop,
  normalisation, cosmic-ray rejection, output writing. CLI:
  `midas-integrate-v2-batch` (notebook 11).

### CLI

- pyFAI CLI is `pyFAI-integrate`, plus `pyFAI-saxs`, etc.
- v2 CLI: `midas-integrate-v2` (single frame), `midas-integrate-v2-batch`
  (sweep mode), `midas-integrate-v2-write-map` (emit v1-format
  Map.bin).

## Translation cookbook

### Single frame, default polygon mode

pyFAI:
```python
import pyFAI
ai = pyFAI.load("calibration.poni")
result = ai.integrate1d(image, 1000, error_model='poisson')
np.savetxt("out.dat", np.column_stack([result.radial, result.intensity, result.sigma]))
```

`midas-integrate-v2`:
```python
from midas_integrate_v2 import (
    spec_from_v1_paramstest, PolygonBinGeometry,
    integrate_polygon_with_variance, write_dat, build_provenance,
)

spec = spec_from_v1_paramstest("paramstest.txt")
geom = PolygonBinGeometry.from_spec(spec, n_jobs=-1)
img = torch.from_numpy(image)
mean2d, sigma2d = integrate_polygon_with_variance(img, geom)
prof  = mean2d.mean(dim=0).numpy()
sigma = (sigma2d ** 2).mean(dim=0).sqrt().numpy() / np.sqrt(spec.n_eta_bins)

# Q-axis for PDF tools
Q = (4 * np.pi / float(spec.Wavelength)) * np.sin(0.5 * np.arctan(
    r_axis * spec.pxY / float(spec.Lsd)))
write_dat("out.dat", q_axis_invA=Q, intensity=prof, sigma=sigma,
          metadata=build_provenance(spec, integrate_mode="polygon"))
```

A few extra lines but you get embedded provenance metadata, polygon-
exact area, and the same numerical result.

### With a static mask

pyFAI:
```python
ai.set_mask(mask)
result = ai.integrate1d(image, 1000, error_model='poisson')
```

`midas-integrate-v2`:
```python
geom = PolygonBinGeometry.from_spec(spec, mask=mask, n_jobs=-1)
mean2d, sigma2d = integrate_polygon_with_variance(image_t, geom)
```

### Bad-pixel detection (v2 only)

pyFAI requires you to threshold a flat-field image manually. v2
learns the mask:

```python
from midas_integrate_v2 import LearnableMask, sparsity_prior, EtaUniformityLoss

mask = LearnableMask(spec.NrPixelsZ, spec.NrPixelsY, init_weight=0.9)
opt = torch.optim.Adam(mask.parameters(), lr=0.5)
for _ in range(500):
    opt.zero_grad()
    int2d = integrate_with_corrections(image_t, spec, learnable_mask=mask)
    L = EtaUniformityLoss()(int2d) + sparsity_prior(mask, weight=1e-4)
    L.backward(); opt.step()

hard_mask = mask.extract_hard_mask(threshold=0.5)
# now plug `hard_mask` into your production integration geometry
```

Notebook 10 walks through the full demo with planted hot pixels.

### Sweep-mode batch

pyFAI:
```python
import glob
profiles = []
for path in sorted(glob.glob("frames/*.tif")):
    img = imread(path)
    result = ai.integrate1d(img, 1000)
    profiles.append(result.intensity)
np.save("profiles.npy", profiles)
```

`midas-integrate-v2`:
```python
from midas_integrate_v2 import (
    TIFFGlobSource, integrate_stream, write_h5, build_provenance,
)
source = TIFFGlobSource("frames/*.tif")
result = integrate_stream(spec, source, mode="polygon")
write_h5("profiles.h5", profiles=result["profiles"],
          r_axis=result["r_axis_px"], frame_ids=result["frame_ids"],
          metadata=build_provenance(spec, integrate_mode="polygon"))
```

Or just `midas-integrate-v2-batch paramstest.txt --image-glob 'frames/*.tif' --out-dir profiles/`.

## What pyFAI has that v2 doesn't (yet)

- **Interactive calibration GUI** (`pyFAI-calib2`). v2 is code-first;
  you bootstrap with `estimate_initial_spec(image, …)` and refine via
  notebook 03 / 06.
- **Non-flat detector geometry** (curved imaging plates, e.g.). v2
  assumes a flat detector with rigid-body tilt; that's almost all
  modern HEDM detectors.
- **GPU integration** via OpenCL. v2 supports CUDA today (the
  integration kernels are pure torch); OpenCL is not a goal.
- **Decades of community plugins** (Bragg-imaging, GISAXS, etc.). v2
  is HEDM-focused; the package boundary is "calibrant + sample
  azimuthal integration with autograd".

## What v2 has that pyFAI doesn't

- **Exact polygon-area kernel** (no subpixel approximation, no smooth
  kernel).
- **Differentiable mask** (`LearnableMask`).
- **Autograd through every refinable parameter** (geometry,
  distortion, parallax, wavelength).
- **Multi-image loss aggregators** for joint multi-frame calibration.
- **Bayesian uncertainty quantification** (Laplace approximation;
  notebook 07).
- **Pure-Python codepath** (no numba required for the polygon kernel
  or any v2-only path).
- **Self-describing output files** with embedded provenance metadata.

## Performance comparison

The `bench/` directory has a benchmark script comparing v2's polygon
mode to pyFAI's splitpixel on a Pilatus-sized detector. Headline:
v2's polygon build is ~5-10× slower than pyFAI's splitpixel build
(once); v2's per-frame integrate is ~comparable; v2's autograd-
enabled path is ~3× slower than the raw integration but unique in
capability.

For batch integration of >100 frames, the build cost amortises and
v2 is competitive on throughput while delivering the differentiability
that enables joint-refinement workflows pyFAI can't do.
