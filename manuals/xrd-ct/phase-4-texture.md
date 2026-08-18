# Phase 4 — texture: the per-voxel ODF

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Goal:** a per-voxel orientation distribution, or a defensible bound saying there is not one.

**Entry gate.** Do not start until:

1. phase 2 has reported **peak/background** per ring, and `ENVELOPE.md` §0 says texture is
   answerable at that contrast;
2. rings are **vetted singlets** with per-azimuth SNR;
3. you have run `scripts/odf_positive_control.py` at the measured contrast and read its verdict.

**No XRD-CT dataset here has yet produced a positive per-voxel ODF result.** One was refuted,
one null was itself refuted, one is parked. The operator is validated three independent ways;
the *application* has no positive real-data result. Read that as calibration for your priors,
not as a reason to skip the phase.

---

## 4.1 Choose the model, and default to the small one

| Model | Params/voxel | Non-negativity | Use when |
|---|---|---|---|
| **Uniaxial squared-modulus** (`odf_uniaxial`) | **4** | **free** | Default. Any contrast below ~0.3, or a physically justified single fibre axis |
| General symmetry-adapted GSH (`gsh`) | 23–61 at L=6–10 | not enforced | High contrast, many rings, and an identifiability count that clears §4.4 |
| Non-negative radial basis | grid-dependent | pointwise | **Untested here.** Our comparison was INVALID (dof mismatch). `texture_kernel` is a simulator, not an inversion basis |

Squaring a real expansion buys two things a general GSH solve does not have: non-negativity
with **no** cone projection and **no** L1 penalty, and only even orders — so the odd-`l` ghost
subspace never arises. And 4 parameters instead of 61 matters enormously at low contrast, where
a 61-parameter fit will absorb background error as texture and hand you a map.

```
I(q̂) = | Σ_{l=0,2,4,6} a_l c_l P_l(û · q̂) |²,      c_l = √((2l+1)/4π)
```

## 4.2 Symmetry — the Laue group, from the space group

```python
from midas_hkls import proper_rotations_from_space_group
group = proper_rotations_from_space_group("P6_3/mmc", (2.9505, 2.9505, 4.6826, 90, 90, 120))
basis = SymGSH(L=6, group=group, lattice=(2.9505, 2.9505, 4.6826, 90, 90, 120))
fam = basis.families((1, 0, 1))
```

**Three things this gets right that a hand-rolled table does not:**

1. **Improper operations map to `-R`, not discarded.** Friedel ⇒ the recoverable symmetry is
   the Laue group. Discarding them under-symmetrises the **73** space groups with improper
   operations but no inversion centre (of 230: 92 centrosymmetric, 65 Sohncke, 73 neither).
2. **The lattice matters for plane normals.** In cubic, `(hkl)` doubles as a Cartesian
   direction; **in every other system it does not**. In hcp the angle between (10-10) and
   (0001) is not what the raw index triple suggests. `gsh.hkl_family` is the **cubic-only**
   shortcut.
3. **The setting is preserved** — monoclinic unique-axis b vs c, and the rhombohedral
   settings, come out right because the group is derived from the space group rather than
   looked up by crystal system.

It also raises loudly rather than returning a plausible non-group if the lattice is
inconsistent with the space group.

**Note on symmetry and expressiveness:** cubic has `M(2) = 0` — no l=2 invariant — while 622
has one. So a cubic-only implementation **cannot represent hcp basal texture at all**.

## 4.3 The operator is closed form. No mesh.

The fibre integral of a GSH mode is `(4π/(2l+1))·conj(Y_l^m(y))·Y_l^n(h)` — Bunge's classical
pole-figure equation. So the operator is a table of spherical harmonic values, and the
tetrahedral Rodrigues mesh a handoff plan called its critical-path item is unnecessary.

Validated three independent ways (`LAB_NOTEBOOK.md` §2): 5.6e-15 against brute-force fibre
quadrature, corr 0.99968 against our own Monte-Carlo, corr 0.999992 against **TexTOM**.

**Only even `l` is measurable, for every scan design.** `Y_l^n(-h) = (-1)^l Y_l^n(h)` and
diffraction cannot separate `h` from `-h`. Odd `l` is the classical ghost subspace. Quote
`basis.ghost_dimension()` rather than hiding it — no amount of extra data recovers it, only a
positivity constraint does.

## 4.4 Count identifiability BEFORE fitting

Unknowns grow roughly as `L³`; the row count does not. Measured for 156 voxels and 24192 rows,
unknowns/rows: L=6 → 0.30×, L=8 → 0.52×, L=10 → 0.79×, **L=12 → 1.43×**, L=14 → 1.81×.

**An underdetermined least-squares fit drives its residual toward zero for free.** So "the
residual reached the noise floor at L ≥ 12" is not evidence of anything. **Report the ratio
with every residual.**

Required `L` rises with crystallites per voxel: L=6 at ~1000, L=8 at ~3000, L=10 at ~10⁴ — in
the overdetermined regime. A favourable count is **necessary, never sufficient**.

## 4.5 Regularisation: discrepancy principle, and know its limits

Choose λ by the **discrepancy principle against a measured photon-noise floor**, never by
L-curve.

And know exactly what that buys: **it fixes the WEIGHT, not the PRIOR.** Three different priors
reached the same residual floor, and the run with the **best** residual (0.0606) gave among the
**worst** reconstructions (MAE 0.250 against 0.056). **Never rank reconstructions by residual
alone.**

## 4.5a Set the refute line from a control at YOUR contrast, not from a constant

Measured with a realistically estimated background, a planted ~25 %-amplitude fibre produces:

| peak/bg | 0.50 | 0.20 | 0.10 | 0.05 | **0.02** |
|---|---|---|---|---|---|
| global improvement over uniform null | 30.8 % | 30.6 % | 24.0 % | 12.1 % | **2.6 %** |

**At peak/bg = 0.02, genuine texture yields under 5 %.** So a refute line fixed at 5 % — as the
DAC Ti analysis registered — would reject real texture at the bottom of that contrast range. Run
the control at your measured contrast and set the line from what it can actually deliver.

## 4.6 Run the ladder, not a single fit

```python
model = UniaxialODFModel(design, rays, good, data, weights)
res = fit_uniaxial_ladder(model)
print(res.verdict(xy, refute_pct=5.0, confirm_pct=20.0))
```

Three nested models:

| rung | params | answers |
|---|---|---|
| uniform null | 1/voxel | is there any azimuthal signal at all? |
| **one globally shared texture** | +3 total | is there texture *anywhere*, at any spatial scale? |
| per-voxel | +4/voxel | is it spatially resolved? |

**The middle rung is the point.** A per-voxel fit that fails is ambiguous on its own — it can
mean "no texture" or "texture present but not spatially resolvable". Three shared parameters
separate them. On the DAC Ti scan the global rung bought **0.11 %** and per-voxel **0.17 %**, so
the absence was not a resolution limit.

`chi2_null ≥ chi2_global ≥ chi2_pervoxel` holds by nesting; a violation means the optimiser
stalled, and `fit_uniaxial_ladder` warns rather than reporting a worse model.

**Fix the thresholds before you look at the numbers.** Use `/preregister`.

## 4.7 The three checks a positive result must pass

1. **Residual improvement** over the uniform null, above your pre-registered line.
2. **Polynomial `r²`** below ~0.5 (`explained_by_polynomial`). Above it, the map is a smooth
   instrumental/absorption field. **Three retractions in this project.**
3. **Per-ring agreement** — the same physical texture must be seen by every vetted ring;
   pairwise correlation below ~0.3 means it is not one ODF.

Plus, before any of them count: **the positive control at the measured contrast**.

## 4.8 The positive control, and how to read it

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/odf_positive_control.py \
    --contrasts <your measured peak/bg> --n-crystallites 40000 --background both
```

It plants **discrete crystallites** and bins their {hkl} normals — no Legendre polynomials, no
squared modulus, nothing the fit uses. Then it lays real peaks on a background and runs the
**same** extraction and fit.

**Read three things, in this order:**

1. **Plant quality.** Only ~6 % of normals land near the diffraction condition, so a low
   crystallite count leaves single-digit counts per bin and a *planted* pole figure that is
   40 % noise. Below ~15 % plant noise before believing anything about the pipeline.
2. **The two background modes.** `known` subtracts an exact pedestal (Poisson noise only);
   `estimated` runs the real chain, so the background must be modelled. **The gap between them
   is the background-model error**, which dominates real low-contrast data and does *not*
   average down with more frames.
3. **Two separate verdicts.** *Detect* (does the global rung find planted texture?) and
   *resolve* (does per-voxel `S` correlate with the plant?). These support different claims:
   detect-only means a **global** null is interpretable but a **per-voxel** null is not, and
   you should report a sample-average bound rather than a map.

**Measured on this geometry (5 hcp rings, 50 azimuths, 11 translations × 12 ω, 69 voxels):
detect YES, resolve NO.** Global improvement reaches 31 % while per-voxel `|corr|` never exceeds
**0.35** with a realistic background. And per-voxel recovery is **non-monotonic in SNR** —
better at peak/bg 0.1 than at 0.5 — because at high SNR the limit is **model mismatch**, not
noise: the plant is a discrete-crystallite fibre distribution and the fit is a 4-parameter
squared-modulus expansion, so the fit chases a shape it cannot represent. Noise was regularising
it. Do not read "more signal" as "better map" here.

**A sign trap worth knowing before it alarms you.** The recovered quantity is the **pole-figure**
order parameter, not the crystal-axis one. Prism normals in hcp are perpendicular to *c*, so a
c-axis fibre appears as a *negative* pole-figure `S`. Scoring recovered pole-figure `S` against
planted c-axis `S` reads corr = −0.75 and looks like total failure; it is correct. Compare like
with like — the script reports both and warns when the signs differ.

## 4.9 Two ceilings that look like data limits

* **Hermans `S` saturates near 0.61** with only `a₂` free, and then *decreases*: +0.59 at
  `a₂ = 1`, max +0.611 near 1.35, +0.586 at 2, +0.447 at 5. A fit stalling near `S ~ 0.6` may
  be at this ceiling. A sharper axial texture needs `a₄` and `a₆`.
* **Kernel truncation** (if you use `texture_kernel` as a positivity basis): a **sharp** kernel
  loses most of its amplitude at low `L` — 8° keeps 5.8 % at L=6 — while a 40° kernel keeps
  100 %. Cause is symmetry, not bandwidth: `M(2) = 0` for cubic annihilates the l=2 term, the
  largest coefficient for any kernel sharper than ~30°.

## 4.10 The scope limit to state explicitly

`fibre_cos_theta` fixes the fibre along the **rotation axis**, justified because
`n_s·ẑ = cos θ_B sin η` has no ω dependence.

**If the sample's unique axis is not the rotation axis** — a DAC loaded in radial geometry, say
— the texture varies with ω and this model **cannot fit it by construction**. A null then means
nothing.

So a negative result from this model is: **refuted for a fibre about the rotation axis; NOT
tested for any other axis.** Establishing which applies needs the loading geometry — a fact
about the experiment, not the data. Ask before spending more compute.

And do **not** try to settle it from the ω behaviour: an axial fibre is *necessarily* static in
ω, so "static in ω therefore instrumental" is wrong. That inference was made here and withdrawn.

## 4.11 What to hand forward

```
Model:          uniaxial (4 par) | general GSH (n_coef = ?), and WHY
Symmetry:       space group, Laue class, group order, lattice used
Identifiability: n_coef, rows, unknowns/rows ratio, ghost_dimension()
Ladder:         chi2 null / global / per-voxel, with parameter counts
Checks:         improvement %, polynomial r^2, per-ring agreement
Control:        plant noise, both background modes, detect + resolve verdicts
Scope:          fibre axis assumed, and whether the loading geometry confirms it
Result:         a map (with all three checks passed) OR a bound (with its softener)
```

Then `phase-5-report.md`.
