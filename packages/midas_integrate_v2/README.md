# midas-integrate-v2

Differentiable, autograd-aware radial integration. Companion to
[`midas-integrate`](../midas_integrate/README.md) — v2 sits *alongside*
v1, not in place of it.

```bash
pip install midas-integrate-v2
```

## When to use which

| | `midas-integrate` (v1) | `midas-integrate-v2` |
|---|---|---|
| Production batch integration of detector frames | ✅ Use this | overkill |
| Refining geometry / corrections jointly with integrated profile | needs sidecar machinery | ✅ Use this |
| Joint refinement with `midas-calibrate-v2` | export → re-import | ✅ Native loop |
| Stage-4 thin-plate spline as a refinable layer | baked binary lookup | ✅ `nn.Module` |
| Per-ring `δr_k` (F2 fix) inside the radial map | sidecar JSON for downstream | ✅ Native, refinable |
| Bit-identical hot path | ✅ CSR kernel | ✅ same CSR kernel |
| Exact polygon-arc-arc bin overlap (no subpixel, no smooth-kernel approx) | ✅ numba kernel | ✅ pure-numpy/torch kernel |
| Hand-holding student notebooks | — | ✅ student notebooks, self-contained |

v2 reuses v1's CSR sparse-matmul integration kernel for the forward pass
(so the hard-binning path stays bit-identical), and adds a parallel
soft-binning forward path that's differentiable end-to-end.

## Architecture

```
midas_integrate_v2/
  spec.py              # IntegrationSpec — v2-native (iso_R*, a*/phi*) torch tensors
  forward/             # pixel_to_REta from a spec (re-exports calibrate_v2)
  binning/             # build_map (bridges to v1 numba) + MapCache
  kernels/             # hard-bin integrate + profile_1d (v1 parity)
  diff/                # soft-bin integrate (linear interp; differentiable)
  corrections/         # δr_k, RBF spline, polarization, solid-angle, Q-bins
  compat/              # v1 IntegrationParams ⇄ v2 IntegrationSpec
```

## Quickstart

### 1. Bit-identical to v1, with a v2-native parameter dataclass

```python
from midas_integrate.params import parse_params
from midas_integrate_v2 import (
    spec_from_v1_params, build_geometry, integrate, profile_1d,
)

p = parse_params("paramstest.txt")
spec = spec_from_v1_params(p)                       # v2-native
geom = build_geometry(spec, dtype=torch.float64)    # CSR + cached map
int2d = integrate(image, geom, mode="floor")        # bit-identical to v1
prof = profile_1d(int2d, geom)
```

### 2. Joint refinement of geometry against an integrated-profile loss

```python
from midas_integrate_v2 import (
    spec_from_v1_params, integrate_with_corrections,
    PolarizationCorrection, SolidAngleCorrection, PerRingOffsets,
    RBFResidualCorrection,
    EtaUniformityLoss, ProfileMSELoss, GaussianPriorLoss,
)

spec = spec_from_v1_params(p, requires_grad=True)
pol = PolarizationCorrection(pol_fraction=0.99, refinable=False)
sa  = SolidAngleCorrection()
delta_rk = PerRingOffsets(n_rings=12)               # F2 fix, refinable
spline = RBFResidualCorrection(centres, weights)    # Stage-4 spline

opt = torch.optim.Adam([
    spec.Lsd, spec.BC_y, spec.BC_z, spec.ty, spec.tz,
    *delta_rk.parameters(),
    *spline.parameters(),
], lr=1e-3)

eta_loss = EtaUniformityLoss(intensity_floor=1.0)
prior = GaussianPriorLoss({"Lsd": (Lsd_seed, 100.0)})

for _ in range(200):
    opt.zero_grad()
    int2d = integrate_with_corrections(
        image, spec,
        residual=spline, per_ring_offsets=delta_rk,
        ring_R_centres_px=ring_centres,
        polarization=pol, solid_angle=sa,
    )
    loss = eta_loss(int2d) + 0.01 * prior(spec)
    loss.backward()
    opt.step()
```

### 2b. "Build once, integrate many" pure-torch path

```python
from midas_integrate_v2 import (
    spec_from_v1_paramstest, SoftBinGeometry,
    integrate_soft, integrate_soft_batch,
)

spec = spec_from_v1_paramstest("paramstest.txt", requires_grad=False)
geom = SoftBinGeometry.from_spec(spec)        # precompute once
profiles = integrate_soft_batch(images_3d, geom)   # (N, n_eta, n_r)
```

No numba in the call path — useful when you've already imported torch
and want to avoid the OpenMP runtime conflict with v1's numba mapper.

### 3. Hand off back to v1 for batch integration

```python
from midas_integrate_v2 import v1_params_from_spec
from midas_integrate.detector_mapper import build_and_write_map

p_v1 = v1_params_from_spec(spec)                    # tensor → scalar
build_and_write_map(p_v1, output_dir="run/")        # v1 CLI then takes over
```

## Design choices

- **Implicit gradient strategy**: hard-binning forward keeps bit-parity
  with v1; gradient flows through a parallel soft-binning kernel
  (linear interpolation in R and η). Bin assignments are not
  themselves differentiated — the upstream `(R, η) = pixel_to_REta(...)`
  is, which is the slope you want for refinement.
- **Map cache**: hashes the same fields as v1's `compute_param_hash` so
  v1 and v2 share `Map.bin` caches and never diverge.
- **Pixel pitch and R** (non-square or hex pixels): v2's R axis (`RMin + RBinSize * (i + 0.5)`, the `reta_map`, a profile's `r_axis`)
  is in units of the MEAN pitch, `spec.mean_pixel_um()`. Convert with `tan(2θ) = R * spec.mean_pixel_um() / Lsd`; `spec.pxY`
  alone is wrong whenever `pxY != pxZ` (it equals the mean for square pixels). `pxZ` left at 0 means square. The v1 mapper
  (`midas_integrate`, and the v1-bridged `build_geometry`) is a single-pitch model that uses `pxY` for both axes; it now
  warns when `pxY != pxZ`, and `write_map_bin_from_geometry` warns because Map.bin readers assume `pxY` units.
- **Corrections from the paramstest** (since 2026-10-11): `SolidAngleCorrection` /
  `PolarizationCorrection` (with `PolarizationFraction`, `PolarizationPlaneEtaDeg`) set on the spec
  are applied by the hard, subpixel and polygon geometries, their `*_with_variance` integrators and
  `write_map_bin_from_geometry`, as `I / (SA * POL)` with the per-bin normaliser unchanged, exactly as
  `integrate_with_corrections` and v1 do. The soft geometry (`SoftBinGeometry`, `integrate_soft`, CLI `--mode soft`)
  applies them too, and its divisor keeps its gradient to the spec so refinement sees the correction's dependence
  on the geometry. An explicit per-pixel `correction=` passed to a variance integrator replaces the spec-derived
  one (never stacked). Opt out for one geometry with `geom.pixel_divisor = None`. Earlier versions ignored the
  flags silently.
- **Which binning kernel to use for ring positions (2026-10-14, measured on noiseless synthetic scenes)**: prefer `PolygonBinGeometry`. With the true
  tilted plane supplied it was flat with tilt up to 60 deg (worst sector <= ~8 microstrain at rings >= 3 deg on noiseless synthetic
  scenes, no tilt-dependent error above ~5 microstrain detectable). `SubpixelBinGeometry` puts the centre of each K x K sub-square
  wholly into one bin and has a narrow sampling-aliasing comb at radial bin widths w = m/K px (K=4: 0.25 / 0.5 / 0.75 / 1.0 px)
  and sqrt(2)-related widths: ring-position sector scatter of tens of microstrain (K=4 at 0.25 px: 63/22/11 microstrain at
  rings 1.5/3/4.5 deg vs 19/4.7/2.5 at 0.20 px; rms over azimuth sectors, `tilt4way/logs/binwidth_midas.log`). It raises `SubpixelResonanceWarning` for 2 <= K <= 8 on those widths; azint (same
  algorithm) shows the same behaviour. Larger K lowers the comb but does not remove it (K=16 at 0.25 px: 13/3.6/1.8 microstrain rms), so polygon remains the recommendation.
- **Distance normalisation (opt-in, default off)**: `spec.IntensityRefDistance_um = L_ref` (> 0) multiplies the output by
  `(Lsd / L_ref)^2`. Raw counts per pixel scale as `1/Lsd^2`; `SolidAngleCorrection` is normalised to 1 at normal incidence, so it
  removes the angle dependence but not that factor. `L_ref = 1e6` reproduces GSAS-II, which multiplies by `Lsd[m]^2`, and makes
  intensities from different sample-to-detector distances comparable. It is applied in the same divisor as the flags above (hard,
  subpixel, polygon, variance, soft, `integrate_with_corrections`, Map.bin), works with or without them, and is v2-only (a v1 paramstest
  has no key for it; `v1_params_from_spec` warns). When any correction is on, the profile header lists them under `corrections`.
  Provisional: on one real 1-14 deg lineout set, SA + polarization (P = 1) + `L_ref = 1e6` matched the GSAS-II lineouts to 0.5-1 %.
- **`nn.Module` corrections**: `δr_k`, the Stage-4 RBF spline, and the
  polarization/solid-angle factors are all torch modules with
  `requires_grad`-controllable parameters. Mix and match as the
  optimisation problem demands.
- **`IntegrationSpec`** uses v2 distortion names (`iso_R2, a1, phi1, …`);
  the `from_v1`/`to_v1` adapters round-trip the chaotic legacy `p0..p14`
  naming losslessly.

## What's in v0.1.0 (the first release)

Tested end-to-end against the v1 production pipeline on real Pilatus +
Varex Aero CeO₂ data; a full unit-test suite and student notebooks (see `tests/` and `notebooks/`).

### Math correctness

- **Exact polygon-area pixel-bin overlap kernel** (Green's theorem on
  circular-arc + radial-segment intersections). No subpixel
  approximation, no smooth-kernel blur — the differentiator vs pyFAI /
  dxchange / DPDAK / nika.
- **Exact tilt-aware solid-angle correction** (`Lsd² · (n̂·r) / |r|³`).
  Bit-identical to v1 at fp64 on any detector pose.
- **Exact thin-plate-spline kernel** (`r² log r` with the analytic
  `r=0` limit handled cleanly).
- **Polarisation correction**: standard `1 − PF · sin²(2θ) · cos²(η − plane)`.
- **Parallax correction**: `R + parallax · sin(2θ) / px` (matches v1).
- **Q ↔ R bin edge conversion**: `2θ = 2 arcsin(λ/2d)`, `R = (Lsd/px) tan(2θ)`.
- **BC ↔ PONI 0.5 px convention** for pyFAI interop pinned in
  `compat.pyfai`; the `make_pyfai_integrator(spec)` helper makes it
  impossible to drop the half-pixel shift.

### Five binning kernels (each clearly labeled)

| Kernel | Math | Differentiable in geometry? | Use for |
|---|---|---|---|
| `PolygonBinGeometry` | **Exact** polygon-arc-arc | No | Production batch + calibration accuracy |
| `HardBinGeometry` | Hard floor (one sample per pixel) | No | Max throughput, fixed geometry |
| `SubpixelBinGeometry` | K×K oversampling of hard | No | Mid-fidelity fast path |
| `SoftBinGeometry` / `integrate_diff` | Linear-interp soft binning | **Yes** | Refinement / autograd |
| `MapCache` (wraps v1) | Same as v1 | No | v1-cache interop |

### Differentiable refinement

- Autograd through every refinable parameter (Lsd, BC, tilts,
  Parallax, wavelength, all 15 distortion coefficients).
- All 4 v2 corrections as `nn.Module`s with refinable parameters:
  per-ring δr_k, Stage-4 thin-plate spline, polarisation, solid-angle.
- 9 loss families: profile MSE / weighted, η-uniformity, peak-position,
  Gaussian prior, multi-image, batched-spec, η-slice, wedge, ring-masked.

### Differentiable bad-pixel mask (`LearnableMask`)

The MIDAS differentiator no other azimuthal integrator has. Per-pixel
inclusion weight is a learnable parameter; train jointly with the
calibration loss + a sparsity prior, and bad pixels (hot, dead,
cosmic-ray-prone) get auto-zeroed while good pixels stay at weight ≈ 1.
Notebook 10 walks through the demo with planted hot pixels.

### Production-deployable pipeline

- **Streaming**: `TIFFGlobSource` / `HDF5FrameSource` / `ZarrFrameSource`
  iterators; `FrameNormalizer` (monitor / exposure / transmission);
  `reject_cosmic_rays` (per-pixel temporal sigma-clip);
  `integrate_stream` (out-of-core, memory constant in N-frames).
- **Variance propagation**: every binning kernel has an
  `integrate_*_with_variance` variant returning `(mean, σ)` per bin.
  Default Poisson; user-supplied variance images supported.
- **Output writers** with embedded provenance metadata (package
  version, geometry hash, mask fraction, source file names): CSV,
  XYE (Rietveld), FXYE (GSAS), DAT (PDF), 2D-CSV, HDF5.
- **Per-pixel masks** in every binning kernel — applied at *build time*,
  so masked pixels never enter the integration.
- **CLI scripts** (see `[project.scripts]`): `midas-integrate-v2` (single frame),
  `midas-integrate-v2-batch` (sweep mode), `midas-integrate-v2-write-map`
  (emit v1-format Map.bin / nMap.bin without numba), plus `-server`, `-pdf` and `midas-bootstrap`.

### Pedagogical material (student notebooks, ~3.5 hrs end-to-end)

01 First Diffraction Pattern → 02 Geometry Intuition → 03 Joint
Refinement → 04 Multi-Distance Calibration → 05 calibrate-v2 ↔
integrate-v2 Handoff → 06 Custom Losses → 07 Bayesian UQ → 08 PDF
Analysis → 09 Production Workflow → 10 Differentiable Mask → 11
Sweep-mode Batch Processing.

Each notebook is self-contained, executes end-to-end, and includes
"try it yourself" exercises. They live in `notebooks/` and are **not shipped
with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_integrate_v2/notebooks).

### Ecosystem

- **pyFAI migration guide** (`docs/MIGRATING_FROM_PYFAI.md`).
- **Performance benchmark** script (`bench/bench_integrate.py`).
- **Bootstrap helpers** (`estimate_BC_from_image`,
  `estimate_initial_spec`) for users without a starting paramstest.
- **Ring auto-detect** (`detect_rings`, `suggest_material`) with built-in
  CeO₂ / LaB₆ / Si / Cr₂O₃ d-spacings (Cr₂O₃ uses JCPDS 38-1479).

## Roadmap

- **v0.2** — Polygon kernel GPU port (vectorise the scalar Python
  loops onto torch+CUDA, keeping the math exact). Right answer for
  sub-pixel `RBinSize` builds where the trivial fast path doesn't fire.
- **v0.3** — Multi-GPU integrate; NeXus-strict HDF5 output;
  integration with the wider HEDM pipeline (peak fitting, indexing).

## License

BSD-3-Clause.
