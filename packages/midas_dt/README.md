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
`packages/midas_dt/dev/look_at_frame.py` in the MIDAS repo does this, and
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
`pip install midas-dt[direct]` (torch).

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
| `RECON_SIGN` | −1: `doLog=0` back-projects intensity, so the result is negative-going |
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

## What it does not correct

Attached to every result via `ScanKnownLimits` and written into
`provenance.json`, so a map cannot be separated from its caveats:

- **self-absorption** — phase fractions are biased toward the sample surface
  and are qualitative
- **texture** — the η-integrated pattern is a powder pattern only if the voxel
  is randomly oriented
- **phase fractions** are relative: uncorrected for structure factor,
  Lorentz-polarisation and absorption
- **a single-channel strain map** is one projection of the tensor along the
  scattering vector, not the tensor

## Installing

```bash
pip install midas-dt          # scan, channels, sinograms, branches A and B
pip install midas-dt[direct]  # + branch C (torch, midas-invert)
pip install midas-dt[full]    # everything
```

`midas-tomo` supplies the reconstruction engine and is always installed; it
compiles its C at install time and falls back to a Python-only path when the
toolchain is absent, so it never breaks the install.

`[direct]` adds torch and `midas-invert` for branch C. It is separate because
torch is large and branches A and B do not need it. `[full]` also adds
`midas-integrate-v2` (integration with variance), `midas-peakfit`,
`midas-hkls` (ring indexing) and `midas-stress`.

## License

BSD-3-Clause.
