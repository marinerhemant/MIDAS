# midas-dt

Diffraction / X-ray computed tomography (XRD-CT): raw detector frames to
per-voxel diffraction patterns and the maps derived from them.

```python
from midas_dt import (
    Channel, DTScan, assemble, detect_snake, find_centre,
    geometry_from_legacy_params, run_recon_then_fit,
)
from midas_dt.reduce import FrameReducer

geo  = geometry_from_legacy_params("ps_dt.txt")
scan = DTScan.from_stem("/data", "sample", 161, 215, dark_file="dark.raw")
chan = Channel(105, 125, r_bin=0.5)

inten, var = zip(*(FrameReducer(geo, chan, dark=scan.dark())
                   .reduce_translation(scan, t)
                   for t in range(scan.n_translations)))
stack = assemble(np.stack(inten), np.stack(var), scan.omega_deg, chan,
                 snake=detect_snake(profiles)[0])
result = run_recon_then_fit(stack, shift=find_centre(stack).shift)
```

or from the shell:

```bash
midas-dt --params ps_dt.txt --raw-dir /data --stem sample \
         --start 161 --end 215 --dark dark.raw \
         --r-min 105 --r-max 125 --out ./maps
```

## Scope

**In:** powder-like XRD-CT. A pencil or line beam, the sample translated
across it and rotated, one area-detector frame per (translation, rotation).
Rings are integrated azimuthally, reconstructed per (Q, η) bin, and fitted.

**Out:** scanning-3DXRD. If the rings break into discrete spots the sample is
coarse-grained and this is the wrong tool — use `midas_index`'s PF mode with
`midas_pf_odf`. The dividing line is operational: continuous at your working
bin size, or not. Check it on a frame before committing to a reduction —
`packages/midas_dt/scripts/look_at_frame.py` in the MIDAS repo does this, and
`midas_dt.rings.find_rings` gives the quantitative version.

One trap worth repeating from that script: on a Pilatus, the module gaps read
as azimuthal structure and make **every** ring look spotty. Mask them first.

## The three branches

All ship, they share one `Channel` list, and `compare()` measures the gap
between any two of them on your data.

**A — `run_fit_then_recon`** fits each projection, then reconstructs the
parameter sinograms. 12 reconstructions per channel, independent of binning —
cheap.

**B — `run_recon_then_fit`** reconstructs every bin, then fits per voxel.
Exact, at `n_r × n_eta` reconstructions.

**C — `run_direct`** never reconstructs. It builds a differentiable forward
map (voxel peak parameters → per-voxel pattern → line integral → the measured
sinogram) and solves for the voxel parameters by gradient descent, so the peak
model is enforced *inside* the inversion. σ then comes from the curvature of
the loss rather than from repeated reconstructions. Needs
`pip install midas-dt[direct]`.

> **No performance claim.** Whether C beats B on accuracy at matched compute
> has **not been tested**, and nothing in this package asserts it. That claim
> is gated on a preregistered comparison followed by an adversarial check; if
> it loses, that gets reported too. What *is* verified is correctness on
> synthetic data with known ground truth — the projector matches an
> independent reference, its adjoint passes the dot-product test, autograd
> matches finite differences, and the solver recovers planted peak centres to
> **0.019 px** (0.0005 px at 2000 steps). Use C because you want error bars
> from curvature or a model-constrained inversion, not because it is assumed
> to be better.

### When branch A is valid

Radon inversion is linear; peak fitting is not. Only quantities that **add
along a ray** may be back-projected directly:

- `TotalIntensity`, `TotalIntensityBackgroundCorr`, `FitIntegratedIntensity` — yes.
- `RMEAN`, `SigmaG`, `SigmaL`, `MixFactor` — **no**. A projection's fitted
  `RMEAN` is the intensity-weighted mean along the ray, not the sum, so
  back-projecting it gives a number with no physical meaning that looks
  entirely reasonable.

So `weighting="intensity"` (the default) reconstructs the moments instead:

```
RMEAN_voxel = recon(RMEAN_proj × I_proj) / recon(I_proj)
```

Both terms add, so this is correct wherever the single-peak / small-shift
linearisation holds. `weighting="none"` reproduces the legacy behaviour and
marks its outputs `approximate` in the result and its provenance.

Measured on a phantom whose peak position varies across the sample:
`TotalIntensity` (additive) agrees between branches to **0.0**; `RMEAN` (not
additive) to **0.0085**. That difference is the reason the distinction exists.

## Conventions it pins

`midas_dt.conventions` is the single place for the things that silently
produce wrong answers. All are tested.

| | |
|---|---|
| fit-output order | 12 canonical channels; `MaxIntensityObs` is slot **5** |
| additive outputs | only 3 of the 12 may be back-projected |
| `RECON_SIGN` | **+1** — measured against a planted object. It was −1, copied from the 2023 script, and that inverted every map |
| omega | negated once (1-ID aerotech), in `DTScan.from_stem` |
| first frame | dropped (1-ID writes a throwaway) |
| snake | **detected** from the data, not read from a flag |

**Reading pre-2026 MIDAS DT output:** every legacy Python driver omits
`MaxIntensityObs` from slot 5, shifting each label from index 5 on — a file
named `*_BGFit_*` actually holds `MaxIntensityObs`. Index by position and take
the name from `FIT_OUTPUT_NAMES`. Indices 0–4 are unaffected.

## Error bars

`midas_integrate_v2` propagates Poisson σ through integration; `sinogram`
carries it; `reconstruct(variance_samples=K)` gives per-voxel σ by Monte
Carlo. It is opt-in because each sample costs a full extra reconstruction.

There is deliberately no cheap "push the variance sinogram through FBP"
option: for a linear operator `A` that computes `A @ var`, not `A² @ var`, and
it can go negative through the ramp filter's lobes. Manufacturing an error bar
is worse than not having one.

Branch C offers a second route — `laplace_sigma()`, from the curvature of the
loss. Two things to know before using it:

- It is **block-diagonal**: every other voxel is held fixed while each 4×4
  block is computed, because the exact Hessian is over all 4 × n_voxel
  parameters at once. Neighbouring voxels are strongly correlated through the
  shared rays, so this **understates** the uncertainty. Rank voxels by
  confidence with it; do not quote it as a calibrated interval.
- The default `noise_var` is `1/N`, **not** the converged loss. The loss here
  is already weighted by `1/variance`, so passing the loss counts the noise
  twice and inflates σ by exactly `sqrt(loss × N)` — measured, that turned a
  0.035 px error bar into 446 px on a 20 px window.

## Reconstruction: gridrec, SIRT or TV

`reconstruct()` uses `midas-tomo`'s gridrec — filtered back projection, one
pass, and the right default. Its failure mode matters for XRD-CT, where angle
counts are often low because every projection costs a full detector frame at
every translation: below Nyquist, FBP does not degrade gracefully, it puts
**streaks** through the image that a per-voxel peak fit will happily fit.

`sirt()` and `tv_reconstruct()` trade compute for artefact suppression, over
all bins at once (one sparse mat-mul per iteration regardless of channel
count). Neither is a free win:

- both are iterative, so the answer depends on where you stop
- SIRT is biased toward smoothness, TV toward piecewise-constant images
- neither has analytic noise propagation, so σ needs Monte Carlo — the gridrec
  path's `variance_samples=K` is cheaper and better understood

**`tv_weight` is not a free parameter.** Set it too high and TV does exactly
what it is designed to do — produce a piecewise-constant image — turning a
noisy continuum into flat regions with invented edges, which a downstream fit
then reports as real structure with small scatter. Start at 0, confirm you
reproduce gridrec, and raise it only while streaks fall faster than features.

Use gridrec unless streaking is visibly limiting, then compare.

## Self-absorption

The beam is attenuated reaching a voxel and the diffracted beam is attenuated
leaving it, so an uncorrected map reads as though the sample were richer at its
surface. The attenuation sits **inside** the ray sum, weighting each voxel
differently — which is why it biases along the ray and why no rescaling of the
sinogram can undo it.

```python
from midas_dt import attenuation_factors, uniform_mu, run_direct

mu = uniform_mu(sample_mask, mu_per_pixel=0.05)   # or mu_from_transmission(...)
A  = attenuation_factors(mu, scan.omega_deg, two_theta_deg=1.2)

result = run_direct(stack, attenuation=A)          # exact
```

**Exact (branch C):** fold the factors into the forward operator and solve with
them in place. The model then describes the measurement, so nothing needs
undoing. This is the real payoff of having a forward model.

**Approximate (branches A and B):** `correct_reconstruction()` divides each
voxel by the attenuation it suffered averaged over the rotation. That removes
the bulk near/far bias but not the distortion absorption already caused in the
reconstruction, because the ω-dependence of the attenuation violates the Radon
model the reconstruction assumed. Marked approximate.

**Not provided:** dividing the sinogram by a per-ray factor. It looks like a
correction and cannot be one — a single number per ray cannot invert a
position-dependent weighting inside the sum.

μ comes from the transmission channel (`mu_from_transmission`, an absorption-CT
reconstruction through `midas-tomo`) or a uniform value over a sample mask.
The outgoing ray is treated as parallel to the incoming one when the in-plane
deflection is under 5° — decided from the geometry, not assumed. At 90 keV with
rings at 100–500 px that deflection is ~0.5–2.7°, so it costs almost nothing;
at low energy or high angle the exact path is used instead.

## What it does not correct

Attached to every result via `ScanKnownLimits` and written into
`provenance.json`, so a map cannot be separated from its caveats:

- **self-absorption** — correctable now (above), but **not corrected by
  default**: it needs a μ you have to supply. Uncorrected phase fractions stay
  biased toward the sample surface and qualitative.
- **texture** — the η-integrated pattern is a powder pattern only if the voxel
  is randomly oriented
- **phase fractions** are relative: uncorrected for structure factor,
  Lorentz-polarisation and absorption
- **a single-channel strain map** is one projection of the tensor along the
  scattering vector, not the tensor

## Texture tomography on an orientation-node basis

> **Status: checked on simulation only; not a measured result, and the engineering claim is PROVISIONAL.** A four-angle adversarial
> check (2026-10-28) reproduced the numbers below and found no error in the physics, but left statistics, reproduction and artifacts
> open; it found seed-fragile tests and tests that missed 9 of 21 deliberately broken variants of the code, which are fixed here and
> have not been re-checked by a fresh adversarial pass. Validation was on a cubic (Cu-like) sample, noise-free, one generator.

Per-voxel ODFs as **non-negative weights on a list of orientation nodes** (Gaussian kernels on SO(3), summed over the crystal symmetry),
fitted to ring-binned rotation-series data `data[omega, translation, eta, ring]`. Needs `pip install midas-dt[texture]`; the names are
imported lazily.

```python
import torch
from midas_dt import (NodeBasis, NodeOperator, TextureScan, make_rings, ring_normalise,
                      fista_nonneg, voxel_odf, merge_nodes)

scan  = TextureScan(n_omega=360, omega_start_deg=0.0, omega_step_deg=0.5, n_eta=180, n_trans=103,
                    wavelength_a=0.248)                       # MIDAS lab frame by default
rings = make_rings([(1, 1, 1), (2, 0, 0), (2, 2, 0)], (4.0493,) * 3 + (90.0,) * 3, 225, scan.wavelength_a)
basis = NodeBasis(node_orientations, sigma_deg=0.4, space_group=225)    # (K, 3, 3), crystal -> sample
op    = NodeOperator(scan, rings, basis, size=99, mask=sample_mask,
                     sub_omega=4, sub_eta=4, node_batch=128)           # bin-integrated sampling
data, _ = ring_normalise(data, q99=71.9)
res   = fista_nonneg(op, data, n_iter=100, loss="huber", delta=30.0)   # c >= 0; the iteration count is part of the model
odf   = voxel_odf(res.coef[v], basis.orientations, space_group=225, floor=0.05)   # the floor is required
odf.medoid(), odf.mass_within(reference, 1.5), odf.w1(reference)
```

* **Batched over nodes.** `NodeOperator` radon-projects the weight maps of `node_batch` nodes at a time (the sparse operator of
  `direct.projection_matrix`) and scatters them through a sparse, omega-block-diagonal pole-figure matrix, so working memory does not
  grow with `K`. The matrix itself is cached (`cache=True`) at `~K x spots-per-node x footprint` entries; `cache=False` rebuilds it per call.
* **Sample each bin more than once.** The pole figure is averaged over `sub_omega x sub_eta` points per detector bin. With one point
  per bin and a node narrower than the bin, a node's response depends on where its spot falls inside the bin: the operator aliases.
  Use about `bin / sigma` sub-samples.
* **The weight floor is an argument, not a default.** A recovered node ODF is a cloud of small atoms; W1 and the atom count depend
  strongly on the floor. `voxel_odf(..., floor=...)` has none by default, `readout_table` takes several. Merge near-duplicate nodes
  (`merge_nodes`, `NodeBasis.merge`, `VoxelODF.merged`) before counting atoms.
* **Limits.** Cubic only (other symmetries are wired through midas-hkls but untested). The kernel is a von Mises-Fisher density per
  normal, the first-order pole figure of a rotation-vector Gaussian, so it is an approximation for kernels wider than tens of degrees.
  No Lorentz or Jacobian factor is modelled (they cancel in ring-normalised data; ring normalisation is exact only for an untextured
  sample). The Radon projection is taken at the frame centre (translation smear within one frame is ignored). Kernels are cut off at
  `cutoff_sigma` = 4.5 sigma, which costs about 4e-5 of the mass in bulk bins but ~2e-3 relative in the bins of slow, grazing
  crossings (raise `cutoff_sigma` if those matter). The iteration count of the solver is the other regulariser: report it.
* **How it was checked.** The operator was compared with a brute-force Ewald solution (same convention, so a check of the code, not
  of the convention), with crystallites sampled from the Gaussian (Monte Carlo pole figure), with the exact kernel sampled on the same
  grid, and numerically with the released diffractom operator in a development script that is not part of this package (0.63 %
  relative L2 difference over all bins with 3 random nodes per voxel, 0.94-1.05 % with one node per voxel). The design was written after reading
  diffractom's documented frame conventions; no code was copied. The tests in `tests/test_texture_nodes.py` and
  `tests/test_odf_readout.py` are the gates that ship.
* **Not in this release:** strain, more than one phase, node discovery from spotty data, and any change to the GSH operator.

Conventions are in the docstring of `midas_dt.texture_nodes`: lengths in voxels (one voxel = one translation bin), angles in degrees,
wavelength in angstrom, orientation matrices map crystal -> sample, `omega` an active right-handed rotation of the sample about the
rotation axis.

## Installing

```bash
pip install midas-dt            # the full workflow: frames -> sinograms -> maps
pip install midas-dt[direct]    # + branch C, SIRT/TV, absorption (midas-invert)
pip install midas-dt[indexing]  # + ring indexing against candidate phases
pip install midas-dt[texture]   # + GSH / node-basis texture modules (midas-hkls, midas-stress)
pip install midas-dt[full]      # all extras
```

The base install does the documented workflow end to end. `midas-integrate-v2`
(integration with Poisson variance) and `scipy` are **core**, not extras:
`FrameReducer`, ring finding and every per-voxel fit need them, so a package
without them could not build a sinogram. `midas-integrate-v2` brings `torch`.

`midas-tomo` supplies the reconstruction engine; it compiles its C at install
time and falls back to a Python-only path when the toolchain is absent, so it
never breaks the install.

`[direct]` adds `midas-invert` for branch C, SIRT/TV and the absorption
operator. `[indexing]` adds `midas-hkls` for matching rings against candidate
phases.

## License

BSD-3-Clause.
