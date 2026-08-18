# Phase 4 — topotomography: alignment, conditioning, shape, and the intragranular field

**Goal:** one grain's 3-D shape and, if the geometry allows it, its interior orientation
field.

TT is the only route here to what is happening *inside* a grain. It is also the technique
where the decisive limit is set **before any photons are collected**, so §4.1 comes first and
is not optional.

## 4.1 Conditioning FIRST — what the scan can determine at all

A single TT scan cannot see rotation about its own scattering vector. The rocking condition is
disturbed by lattice rotation about `a = (k_in × G)/|k_in × G|`, and since `G` is aligned to
the rotation axis, as ψ sweeps the sensitivity direction traces the great circle perpendicular
to `G` and **never acquires a component along it**. Rotation about `G` does not change the
Bragg condition at all — that component is not weakly constrained, it is *exactly null*.

For unit scattering vectors `g_i`, the sensitivity moment is

```
M = (1/N) sum_i (I - g_i g_i^T) / 2
```

and for two reflections separated by γ its eigenvalues are exactly

```
(1 - cos g)/4 ,   (1 + cos g)/4 ,   1/2
```

so the weakest component scales as **γ²/8** for small γ. Close reflections do **not** rescue
the null, however many you take.

```python
from midas_dct_tt import rotation_conditioning, best_reachable_pair, separation_for_conditioning
ev, ratio = rotation_conditioning([g1, g2])
```

**The measured case.** A real campaign's grain scanned on two reflections **13.3°** apart gives
`[0.0067, 0.4933, 0.5]` — the third rotation component **75×** less constrained than the other
two. The analytic law reproduces that experiment's actual 90+90 view sampling to four decimal
places, so this is a design fact, not an estimate.

**Two normalisations exist; do not conflate them.** `rotation_conditioning` returns
`λ_min/λ_max`, which for a *pair* saturates at **0.5** (the largest eigenvalue is always ½).
Reporting `λ_min/λ_mid = (1−cos γ)/(1+cos γ)` instead gives 1.0 at 90°. A 66.1° pair is 0.423
in the second and 0.297 in the first. **Quote the separation γ, which is unambiguous.**

## 4.2 Can the stage even reach a good pair?

Conditioning says what you need; the goniometer says what you can have.

```python
from midas_dct_tt import (reciprocal_basis, instrument_transformation,
                          reachable_reflections, best_reachable_pair)
B = reciprocal_basis(a, a, c, 90, 90, 120)
T = instrument_transformation(samrx_off, samry_off, omega_off)
best = best_reachable_pair(U, B, wavelength_A, envelope=30.0, T=T)   # (hkl_a, hkl_b, gamma, ratio)
```

Audited across 55 topotomographically scanned grains of a published campaign:

| tilt envelope | median γ | γ ≥ 60° | median ratio¹ |
|---|---|---|---|
| ±15° | 29.8° | 0/55 | 0.071 |
| ±25° | 55.6° | 9/55 | 0.278 |
| **±30°** | **66.1°** | **51/55** | **0.424** |
| ±35° | 80.8° | 55/55 | 0.726 |
| ±45° | 90.0° | 55/55 | 1.000 |
| *envelope actually used* | 33.8° | **0/55** | 0.092 |

¹ `(1−cos γ)/(1+cos γ)`, the §4.1 second normalisation.

**±30° is the requirement**, and nothing below ±25° works for any grain. The binding
constraint is the **stage**, not the reflection choice — the experimenters could not have
picked better within the range they used.

Two honesty notes that must travel with this table:

* The "envelope actually used" is the range that campaign *used*, which need not be its
  hardware limit. If the stage reached further, this is a planning oversight rather than an
  instrument bound. The deposited data cannot distinguish them.
* Systematic absences are **not** applied by `reachable_reflections`, so every figure is an
  optimistic bound — filtering absences can only shrink the reachable set.

## 4.3 Solve the tilts

```python
from midas_dct_tt import topotomo_tilts, tilt_branches
up, low = topotomo_tilts(G_sample, T)
```

Validated against **74 independent real goniometer settings**: median residual **0.043°**
(up) and **0.050°** (low), against a random-grain null of **25–40°** — discrimination
**985× / 526×**, with **0 of 200** null draws beating the truth. That single test exercises the
whole chain: orientation convention, reciprocal basis, instrument offsets, alignment solution.

**The solution is two-fold degenerate:** `(up, low) ≡ (up + 180°, −low)`. Since `atan` returns
the principal branch, the sibling always lands at `|up| ≥ 90°` — measured minimum **90.0°
across 18 810 grain × reflection cases** — so no sub-90° stage can reach it.
`reachable_reflections` tests both branches anyway, so the conclusion never depends on which
one you happened to look at.

**Friedel partners give identical tilts.** `G` and `−G` produce the same `(up, low)`, because
numerator and denominator both negate. Enumerating `+hkl` and `−hkl` separately is harmless
but redundant.

## 4.4 Masks: segment the grain, and know which criterion you used

Three segmentations are available and they are not equivalent:

| criterion | when |
|---|---|
| intensity threshold | fast, and the least trustworthy — a topograph's intensity varies with path length and extinction |
| **rocking-peak position** | segment by *where* the rocking curve peaks, not how bright it is |
| **rigidity** | segment by what moves as a rigid body across ψ |

Prefer the latter two. Intensity alone conflates the grain with its illumination.

## 4.5 Reconstruct the volume

Both a hull/algebraic route and a differentiable route exist (`midas_dct_tt.recon`,
`.inverse`). Validate as in phase 3: a null, and a comparison against a no-shape floor.

> **Landmine, documented.** In the research scripts for this campaign, the grain volume is
> returned *pre-multiplied* by one scan's alignment while the projectors then apply the
> other's on top. The composed rotation is **required by the data** but arrives **by accident**:
> "fixing" it collapses the held-out score from **0.81 to 0.62**. This is a property of those
> dev scripts, not of the shipped library — the shipped projector takes no alignment argument,
> so a library user cannot reach the double application. If you are reading those scripts,
> read this paragraph first.

## 4.6 The intragranular field

`midas_dct_tt.field_inverse` recovers a 12-component deformation field (9 of the deformation
gradient, 3 translation) per voxel.

```python
F, info = fit_deformation_field(obs, grain, alignments, psi, ..., lambda_smooth=lam,
                                lr_schedule="cosine", return_best=True)
```

**Two optimiser settings are load-bearing, not cosmetic.** Without a schedule the fit ends
*above* its own best evaluated iterate; the effect on a reconstruction is under 1 %, but it is
**fatal for model selection**, because the argmin then moves with the hyperparameter and
cross-λ comparison becomes meaningless. Anneal to zero (`eta_min=0`) — a floor of `lr/100`
still leaves the last steps moving at full rate. `info["settled"]` is **budget-relative and is
not a convergence certificate.**

**Choose λ by the discrepancy principle against a measured noise floor**, not by eye.

### The controls that make a field believable

| control | result | what it establishes |
|---|---|---|
| held-out views | NCC **0.860** | it predicts data it never saw |
| shuffled-training baseline | +0.139 ± 0.084 | the score is not a fitting artefact |
| constant field | −0.010 | structure, not a global offset |
| **wrong support** | **0.810** vs 0.860, fields agree **+0.940** on 79 % overlap | **the data determine the field, not the domain** |
| disjoint halves (23 / 22 views) | 0.856 / 0.853, cross-half **+0.903** | *determined*, not merely reproducible |

The wrong-support row is the one that changes behaviour: **a fit converging beautifully on
your mask is not evidence the mask is right.** A wrong support scores well largely because it
*contains* the sampled region.

### Resolution — measure it, do not assume it

Against a **polynomial ceiling** (the best a smooth global function attains on the same
planted residual while carrying no per-voxel information), recovery is above the ceiling only
over about one octave: the window is **1.2–2.0 µm** (2–4 voxels), peaking at 1.5 µm. Above
~2.8 µm a low-order polynomial does *better* — meaning a few dozen numbers would have
reproduced the "field". **Report a field with its transfer function, or it is unbounded.**

### Registration is not free

Sub-pixel misregistration manufactures amplitude. At 0 / 0.1 / 0.3 px jitter the correlation
falls **0.246 / 0.176 / 0.042** while `|H|/|H_true|` *rises* **0.24 → 0.39 → 1.06**: the fit
compensates by inventing magnitude. Register deliberately (`orbit_register`), and treat a
field whose amplitude matches truth suspiciously well as a registration artefact until shown
otherwise.

## 4.7 What TT does not give you here

**Absolute strain has not been demonstrated on real data by this pipeline.** The tilts depend
on the *direction* of `G`, not its length, so goniometer settings identify only the axial
ratio `c/a` — scaling `a` and `c` together changes the residual by nothing to four decimal
places. Treat any absolute-strain claim as unsupported until a separate demonstration exists.

## 4.8 Exit criteria

- [ ] reflection-pair separation γ and the sensitivity eigenvalues, stated **before** any field claim
- [ ] tilt solution validated against settings that were not fitted
- [ ] segmentation criterion named (intensity / rocking-peak / rigidity)
- [ ] field carries: held-out score, shuffled and constant baselines, **wrong-support control**
- [ ] field carries its **transfer function** (the length scale above which a polynomial wins)
- [ ] registration jitter considered before believing an amplitude

**Halt** if γ < 60° and a rotation *tensor* is the deliverable. You can still report the
well-determined components; you cannot report the tensor. That is a fact about the experiment,
not about the fit.
