# midas-calibrate-v2

Fully differentiable detector calibration for MIDAS.

**v1 still ships** as the C-backed reference implementation under
`midas-calibrate`.  v2 coexists; pick whichever fits the job.

## What v2 adds over v1

- **Fit anything** — every input (Lsd, BC, tilts, p₀…p₁₄, parallax, λ, **pxY**, **pxZ**, **panel shifts**) is a differentiable Parameter.  Promoting a fixed input to refined is a one-line spec change.
- **Multi-image / multi-distance joint** — share intrinsics (pxY/Z, distortion, panels) across multiple beam positions or distances.  Wright2022's grid panel calibration is a special case.  This is what *operationally* unlocks pixel-size and d-spacing fitting (rank-deficient with a single image).
- **Bayesian calibration** — Laplace at MAP (cheap), pyro mean-field VI, or pyro NUTS for full posterior.  Per-parameter 1σ + full covariance for downstream uncertainty propagation.
- **NN-augmented residual** — small conv NN ΔR(y, z) augmenter, two-stage training, smoothness-regularised so it doesn't absorb harmonics.
- **Joint forward cake** — predict the (R, η) cake intensity directly from `(θ_geom, θ_shape)` and fit raw cake values.  Sidesteps v1's centroid + Newton inversion break.
- **Downstream HEDM coupling** — sensitivity diagnostic + auxiliary loss that uses per-grain strain noise to validate calibration.
- **Multi-calibrant exposures** — CeO2 *and* LaB6 (or any mix) on one frame. The ring table carries both phases, blended rings are excluded automatically, and the residual is reported **per phase** so the calibrants can be checked against each other.
- **Identifiability gates** — refusing to let you refine what the data cannot determine: azimuthal coverage vs the distortion harmonics, RhoD scaling vs the radial terms, and a per-parameter σ audit that flags coefficients consistent with zero or sitting on a bound.

## Two calibrants on one exposure

```python
from midas_calibrate_v2 import calibrate

res = calibrate(
    image, wavelength=0.153443, pxY=200.0,
    calibrant=["CeO2", "LaB6"],      # seeding uses the FIRST one
    min_ring_separation_px=12.0,     # drop rings that collide in radius
)
print(res.calibrants)                # ['CeO2', 'LaB6']
print(res.sigma["Lsd"])              # 1σ, not just a point estimate
print(res.unconstrained)             # refined but |value| < 1σ — freeze these
print(res.at_bounds)                 # refined but pinned on a bound
for d in res.diagnostics:
    print(d.severity, d.name, d.message)
```

or from the CLI:

```bash
midas-calibrate-v2 ps.txt --mode ff --image cal.h5 \
    --calibrant CeO2 --calibrant LaB6 --min-ring-separation 12
```

**What a second calibrant does and does not buy.** It adds rows to the
Jacobian, not a new direction: expect ~√N tighter σ and a genuine cross-check
between the two powders, but *no* new wavelength identifiability (both phases
enter only through their d-spacings, so λ↔Lsd stays degenerate — use
`lsd_offsets_um` for that) and *no* help with azimuthal harmonics (both
powders illuminate the same arc of the detector).

**Different sample positions** ("two capillaries") are modelled by passing the
frame twice with one calibrant each and `mode="same_detector"`, which shares
the tilts and leaves `Lsd`/`BC` per phase:

```python
ms = build_multi_spec([v1_ceo2, v1_lab6], mode="same_detector")
```

Note this offset is exactly degenerate with that phase's lattice constant on a
single frame. See `notebooks/25_two_calibrants_one_exposure.ipynb`.

## Installation

```bash
pip install midas-calibrate-v2
# Bayesian extras (pyro):
pip install "midas-calibrate-v2[bayesian]"
# Everything:
pip install "midas-calibrate-v2[all]"
```

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_calibrate_v2/notebooks).

## Quickstart

### Drop-in replacement for v1

```python
from midas_calibrate.params import CalibrationParams
from midas_calibrate_v2.pipelines.single import autocalibrate
import tifffile

v1 = CalibrationParams.from_file("paramstest.txt")
image = tifffile.imread("ceria.tif")
result = autocalibrate(v1, image, n_iter=5)
print(result.history[-1].mean_strain_uE)
```

### Fitting pixel size from multi-distance data

```python
from midas_calibrate.params import CalibrationParams
from midas_calibrate_v2.pipelines.multi import build_multi_spec, autocalibrate_multi

v1s = [CalibrationParams.from_file(p) for p in paramfiles]
images = [tifffile.imread(p) for p in image_files]

multi = build_multi_spec(v1s)        # pxY, pxZ shared across images
multi.shared["pxY"].refined = True   # opt in: refine pixel size
multi.shared["pxZ"].refined = True

result = autocalibrate_multi(v1s, images, multi_spec=multi, n_iter=10)
print(f"refined pxY = {result.shared_unpacked['pxY']:.4f} μm")
```

### Bayesian calibration with Laplace 1σ

```python
from midas_calibrate_v2.pipelines.bayesian import autocalibrate_bayesian

result = autocalibrate_bayesian(v1, image, mode="laplace")
print(result.laplace.sigma_per_dim)      # marginal 1σ per refined param
print(result.laplace.cov)                # full covariance
```

### NN-augmented residual

```python
from midas_calibrate_v2.pipelines.nn_residual import autocalibrate_nn

result = autocalibrate_nn(v1, image, mode="two_stage")
print(result.harmonic_drift)             # should be ~0 if NN is well-behaved
```

### Joint forward cake

```python
from midas_calibrate_v2.pipelines.joint_cake import autocalibrate_joint

result = autocalibrate_joint(v1, image)
```

### Sensitivity to downstream HEDM

```python
from midas_calibrate_v2.pipelines.downstream import sensitivity_diagnostic

def my_hedm_evaluator(unpacked):
    # User-supplied differentiable HEDM forward model.  Returns per-grain
    # strain residual norms or a scalar fitness.
    ...

report = sensitivity_diagnostic(v1, image, my_hedm_evaluator)
print(report.parameter_names, report.sensitivity_signed)
```

## Architecture

```
midas_calibrate_v2/
├── parameters/   Parameter, CalibrationSpec, MultiImageSpec, pack/unpack, transforms
├── forward/      geometry, panels, distortion, parallax, bragg, cake, peak_shape, nn_residual
├── loss/         pseudo_strain, prior, multi_image, nn_regularizer, downstream_strain
├── inference/    lm, lbfgs, adam, laplace, vi (pyro), hmc (pyro)
├── pipelines/    single, multi, bayesian, nn_residual, joint_cake, downstream
└── compat/       from_v1 (read v1 paramstest.txt), to_v1 (write back)
```

Forward primitives compose: `pixel_to_REta` consumes pxY, pxZ as tensors,
applies optional per-panel rigid body, runs the harmonic distortion basis,
applies always-on parallax, and returns `(R, η)` differentiable in every
refined input.

## Migrating from v1

v1 parameter files are read directly:

```python
from midas_calibrate_v2.compat import spec_from_v1_file
spec = spec_from_v1_file("paramstest.txt")
```

Output is v1-compatible so downstream MIDAS HEDM tools need no changes:

```python
from midas_calibrate_v2.compat import write_v1_paramstest
write_v1_paramstest(result.unpacked, v1_template, "paramstest_v2.txt")
```

### Handing off to `midas-integrate`

For the radial integration pipeline (`midas-integrate` ≥ 0.4.0) there is
a one-call adapter that handles distortion remap, the Stage 4 spline,
and the per-ring `δr_k` sidecar in a single step:

```python
from midas_integrate.params import parse_params
from midas_calibrate_v2.compat.to_integrate import to_integrate_params

template = parse_params("seed_paramstest.txt")
ip = to_integrate_params(
    res,                        # PVCalibrationResult or FourStageResult
    template=template,
    output_dir="./integrate_in",
    ring_d_spacing_A=ring_d, ring_two_theta_deg=ring_tt,
)
# ip is now ready for midas_integrate.detector_mapper.build_map
```

What's not representable in `midas-integrate` v1 (per-panel shifts and
per-ring `δr_k`) is dropped from the `IntegrationParams` and routed to
sidecar files instead — `write_panel_shifts_file` and
`write_per_ring_offsets_json` respectively. Per-ring offsets need to be
applied at the downstream peak-fit / Rietveld stage; v1's radial map
has no per-ring concept.

## Status

- v0.1.0 — alpha.  Single, multi, bayesian (Laplace + VI), NN-residual,
  joint-cake pipelines wired and runnable.  Parity testing against v1
  ongoing.
- v0.2 — production-grade tests, multi-image HDF5 spec, performance pass on
  the joint-cake LM (Schur complement via `lm_solve_arrowhead`).
- v0.3 — downstream HEDM coupling once the differentiable HEDM forward
  model lands in `midas_diffract` / `midas_grain_odf`.
