# Phase 1b — reconstruct: cake to per-voxel patterns

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Goal:** a per-voxel diffraction pattern on a 2-D voxel grid, with the rotation axis found,
the sign right, and a branch chosen for a stated reason.

---

## 1b.1 Assemble the sinogram

```python
from midas_dt import assemble
stack = assemble(intensity, variance, ...)   # (n_translations, n_frames, n_eta, n_r) in
```

`SinogramStack.intensity` is **`(n_bins, n_omega, n_translations)`** with `bin_shape =
(n_eta, n_r)` flattened. Two different layouts in two adjacent objects, so read
`bin_shape` rather than inferring from a length.

**Variance is propagated, not re-estimated.** `midas_integrate_v2` carries Poisson σ through
integration. This matters because η bins have very different counts: a σ guessed from the
summed intensity is wrong wherever the illumination is uneven, which is everywhere on a real
scan.

## 1b.2 Find the rotation axis — and make the two methods agree

```python
from midas_dt import find_centre
res = find_centre(stack, method="com", cross_check=True)   # cross_check is the default
print(res.describe())          # "axis shift +2.135 px (com+sweep)"
print(res.method, res.well_determined, res.detail["com_shift"], res.detail["sweep_shift"])
```

**`method="com+sweep"` is not a valid input** — `midas_dt` 0.5.0 raises
`ValueError: method must be 'com' or 'sweep'`. `"com+sweep"` is what the *result* reports in
`res.method` once the cross-check has run. Pass `method="com"`; `cross_check=True` is the
default and is what runs the confirming sweep.

Two independent estimators, deliberately:

* **`com`** — centre of mass of the sinogram against its geometric centre. Fast, and it also
  reports how *well-defined* the centre is.
* **`sweep`** — reconstruct across a range of shifts and score each by image variance. Slower
  by the number of trials.

`cross_check` runs both and **flags disagreement** rather than silently preferring one. Take a
disagreement seriously: it usually means a low-contrast or near-symmetric object where the
centre genuinely is not well determined, and the reconstruction will be soft no matter which
value you pick.

**Two traps in reading the result.** If `com` reports itself not `well_determined`, the
function returns **early, without running the sweep** — so an agreement you never got can look
like one you did. And `_sweep` is *seeded at the COM value* with `step=0.5`, so its grid
contains the seed: on the DAC Ti S1 set `com_shift` and `sweep_shift` came back
**bit-identical** at −0.114 px, which confirms nothing finer than 0.5 px. `well_determined`
only asserts |sweep − com| ≤ 1.0 px. Do not quote the third decimal.

A wrong shift blurs or doubles every voxel. It does not error.

## 1b.3 `RECON_SIGN` is +1

`midas_dt.conventions.RECON_SIGN` is **+1**, measured against a planted object.

**It was −1**, copied from the 2023 driver script, and that **inverted every map**. An inverted
map is not obviously wrong — it is a plausible microstructure with the contrast reversed. This
is why the constant lives in `conventions` with a test, rather than in whoever's script.

## 1b.4 Choose a branch, and know which outputs are allowed

Three branches ship. `compare()` measures the gap between any two **on your data** — run it
once rather than trusting a general claim.

| | What it does | Cost |
|---|---|---|
| **A** `run_fit_then_recon` | fit each projection, then reconstruct the parameter sinograms | 12 reconstructions per channel — cheap |
| **B** `run_recon_then_fit` | reconstruct every bin, then fit per voxel | `n_r × n_eta` reconstructions — exact |
| **C** `run_direct` | never reconstructs; differentiable forward map solved by gradient descent, so the peak model is enforced *inside* the inversion | needs `midas-dt[direct]` |

### ★ The trap: only additive quantities may be back-projected

Radon inversion is linear; peak fitting is not. Branch A is valid **only** for quantities that
add along a ray:

| Output | Additive? |
|---|---|
| `TotalIntensity`, `TotalIntensityBackgroundCorr`, `FitIntegratedIntensity` | **yes** |
| `RMEAN`, `SigmaG`, `SigmaL`, `MixFactor` | **NO** |

A projection's fitted `RMEAN` is the **intensity-weighted mean** along the ray, not the sum. A
weighted mean does not add, so back-projecting it gives a number with no physical meaning **that
looks entirely reasonable**.

Measured between branches on a phantom whose peak position varies across the sample:
`TotalIntensity` agrees to **0.0**; `RMEAN` to **0.0085**. And on *real* data the equivalent
comparison gave a between-branch correlation of **0.03** — i.e. essentially unrelated answers.

So `weighting="intensity"` (the default) reconstructs the moments instead:

```
RMEAN_voxel = recon(RMEAN_proj × I_proj) / recon(I_proj)
```

Both terms add, so this is correct wherever the single-peak / small-shift linearisation holds.
`weighting="none"` reproduces the legacy behaviour and marks its outputs `approximate` in the
result and in its provenance — if you see that label, do not quote the number without saying so.

### No performance claim between B and C

Whether C beats B on accuracy at matched compute **has not been tested**, and nothing in the
package asserts it. What *is* verified is correctness on synthetic ground truth: the projector
matches an independent reference, its adjoint passes the dot-product test, autograd matches
finite differences, and the solver recovers planted peak centres to **0.019 px**. Choose C
because you want σ from loss curvature or a model-constrained inversion — not because it is
assumed better.

## 1b.5 Reconstruction algorithm

`reconstruct()` (gridrec-style) is the default. `midas_dt.iterative` adds:

* **`sirt`** — iterative, better on limited or gappy ω coverage;
* **`tv_reconstruct`** — total-variation regularised, for piecewise-uniform samples.

Both need torch. **A different algorithm is a different answer**, not a refinement — if you
switch, re-run `compare()` and say which one produced the numbers you report.

## 1b.6 Absorption — predict it before invoking it

`midas_dt.absorption`: `uniform_mu`, `mu_from_transmission`, `attenuation_factors`,
`attenuated_projection_matrix`, `correct_reconstruction`.

**Quantify the expected effect before blaming absorption for anything.** On the CeO₂ scan the
predicted absorption effect on azimuthal intensity was **0.000 %**, which removed it from the
suspect list for a spurious texture signal — and that was worth far more than a correction
would have been.

Absorption *suppresses the centre* of a reconstruction and would **widen** a measured
profile — so it is the wrong explanation for a diameter that comes out too small. Check the
direction of the effect against your symptom.

## 1b.7 Index the rings before extracting anything

```python
from midas_dt import index_rings, CEO2
result = index_rings(radii_px, geometry, candidates=[CEO2])
```

Needs `midas-dt[indexing]` (`midas-hkls`). Phase 1a's calibration already returns an hkl list
if you calibrated against a known calibrant — use it. For CeO₂ at 106.9 keV, nine reflections
between 350 and 1200 px:

| hkl | 2θ (°) | R (px) |
|---|---|---|
| 111 | 2.1264 | 404.03 |
| 200 | 2.4554 | 466.61 |
| 220 | 3.4728 | 660.35 |
| 311 | 4.0724 | 774.72 |
| 222 | 4.2536 | 809.31 |
| 400 | 4.9120 | 935.16 |
| 331 | 5.3531 | 1019.60 |
| 420 | 5.4922 | 1046.27 |
| 422 | 6.0169 | 1146.92 |

Provenance: `calib/geometry_for_dt.json` on `11idc`.

**An incomplete ring list is the fastest way to fail to index a correct cell.** `find_rings`
uses a *rolling* baseline for exactly this reason — powder background falls steeply with radius,
so a global threshold finds the strong inner rings, misses most of the outer ones, and returns
a list that looks plausible.

## 1b.8 Write the maps

```python
from midas_dt import strain_map, d_spacing_map, phase_fraction_map, write_maps_hdf5
```

Write the geometry, the branch, the weighting mode, the axis shift and the algorithm **into the
output file**. A map without those is not re-derivable, and phase 5 will ask for them.

## 1b.9 The CLI, for the standard path

```bash
midas-dt --params ps_dt.txt --raw-dir <dir> --stem <stem> --start N --end M \
         --out <dir> --branch recon-fit --r-min 350 --r-max 1200 --r-bin 1 \
         --eta-bin 10 --compare --n-cpus 8
```

`--compare` runs **both** branches and reports the per-output discrepancy. On a new dataset,
run it once: it is the cheapest way to find out whether your non-additive outputs are being
back-projected when they should not be.

`--shift` is estimated from the data if omitted — but print what it chose (§1b.2).

## 1b.10 What to hand forward

```
Sinogram:   shape, bin_shape (n_eta, n_r), variance propagated?
Axis:       shift in px, method, and whether com and sweep AGREED
Sign:       RECON_SIGN applied (+1)
Branch:     A / B / C, weighting mode, and WHY
Compare:    per-output discrepancy from --compare; any output marked `approximate`
Algorithm:  gridrec / SIRT / TV
Absorption: predicted magnitude (not just "corrected" or "ignored")
Rings:      indexed hkl list with R and 2theta
Maps:       output path, with geometry + branch + shift written IN
```

Then `phase-2-extract.md` for per-azimuth area and centroid.
