# solve-cell — the envelope: what these data cannot determine

Read this **before** promising an answer. Each limit is tiered:

* **intrinsic** — no amount of counting, precision or cleverness moves it. Say so and stop.
* **configured** — a different scan (wedge, step, energy) would move it. Say what to change.
* **measured here** — a number from one dataset; re-measure on yours.

---

## 1. An a/b splitting when every reflection has h = k or h = k = 0 — INTRINSIC

`|G|² = h²/a² + k²/b² + l²/c²` is **invariant under a↔b** whenever `|h| = |k|` (and trivially
when both are 0). A reflection set built only from such families carries **zero** information
about the splitting. **No precision fixes this**, and no null will tell you — the fit will
happily return a splitting, drawn entirely from noise.

Check before running: `ab_separable(hkl)` (rank ≥ 2). If it is False, the answer is
"not obtainable from this set", full stop. Configured escape: a wider ω wedge that brings in
the (k,h) partners — measured on one DAC sample as needing **±50° to ±90°**, not a marginal
extension.

## 2. The SIGN of any a/b splitting — INTRINSIC (a gauge)

A 90° rotation about c* exchanges a and b and refits equally well. The sign you get is
inherited from the seed: swapping the seed moved "a > b" from 57.3 % of domains to 45.3 %.
**Never quote a sign.** A sign that IS consistent across domains is evidence of an artifact,
not of physics — see limit 3.

## 3. A splitting measured on an asymmetric index distribution — MEASURED / configured

In a limited ω wedge the observed indices are not symmetric in h and k. Measured across one
raster: `|k| > |h|` in 5004 reflections against `|h| > |k|` in 1009 — a **5× asymmetry**, with
(0,1) at 1818 spots and (2,0) at 17. `1/a²` then rides on a sparse design column
(`σ(1/a²)/σ(1/b²)` median 5.46, > 2 in 71.8 % of domains), any radial systematic pushes `a`
one way, and the fitted splitting emerges **with a consistent sign**. Run `index_asymmetry`
and report it beside any splitting.

## 4. Which member of a structural series, when the reflections carry no c — INTRINSIC

c is often the ONLY parameter separating members of a series (Ruddlesden-Popper n = 2 vs
n = 3 differ in-plane by 0.31 %, absorbed by any free scale). Measured on six unexplained
spots: the c-term was **0.00–5.61 % of 1/d²**. Structurally undeterminable. Compute each
candidate's c-leverage before running the comparison, and if it is this small, say the
comparison cannot decide and stop.

## 5. Counting statistics on a splitting — CONFIGURED, and usually the real wall

At perfect geometry with an orthorhombic constraint, bootstrap `σ(split) = 0.417 pp` on one
sample. To resolve a 0.34 % target at 3σ needs **N ≈ 1055 reflections; that domain had 71.**
Compute the required N from your own bootstrap before promising a measurement. Free-triclinic
ab initio on the same data gave `σ(δ) = 0.58 %` — an apparent +0.96 % split is **1.7σ**
against a tetragonal truth, i.e. not significant.

## 6. A cell fitted to a spot list contaminated by powder — MEASURED

`flag_powder` flagged **199 of 442 spots (45 %)** as ring fragments on one sample (19 spots at
one |q| spread over 310° of azimuth). Removing them changed the cell **entirely**
(3.6318/3.6551/19.2598 vs the contaminated answer). Always separate powder before refining,
and report how many were removed.

## 7. Distinguishing finite size from faulting, from widths alone — INTRINSIC

`1/w` and `1/(πw)` differ by 3.14 on one measurement. Node widths cannot choose. See the
`defect` doc set, `ENVELOPE.md` §15 — this belongs to that chain, not this one, but it reaches
back here whenever a coherence length is quoted alongside a cell.

## 8. A pressure inverted through a reference ladder, outside the ladder's range — INTRINSIC

Two compounding failures, both measured:

* **The interpolant can BE the result.** On the same 5 reference points and the same measured
  cell, one axis inverted to 40.6 / 38.3 / 30.2 / undefined / 5.1 / 71.9 GPa depending only on
  the fitted form. Interpolant-choice range **35.5 GPa**, against a claimed effect of 37.0.
* **One anchor can carry it.** Dropping a single provenance-mixed anchor (300 K, different
  phase, spliced into a 40 K in-situ ladder) moved that gauge from 40.6 to 5.1 GPa.

**Before quoting anything inverted through a fit:** refit with ≥3 defensible forms and quote
the span; leave out each anchor in turn; confirm the value is on a monotonic physical branch
(a root past the fitted vertex is not a measurement); and check no fitted parameter railed.
If the sample sits outside the ladder's range, the honest statement is "the reference does not
reach here", not "the sample is anomalous".

## 9. A pressure from a gasket is not the sample's pressure — INTRINSIC

A gasket gauge reads the **gasket**, and in axial DAC geometry (diffraction vector ~85° to the
load axis) with σ_axial > σ_radial it is a **lower bound**. Do not silently promote it to a
chamber pressure. Quote the model-free compression (V/V₀) alongside the pressure, since the
equation of state and the ambient reference each add several GPa of spread on their own.

## 10. Anything from a gate or subset that contains the answer — INTRINSIC

A subset selected with `h+k+l even` returns a body-centred verdict at 97.5 % because the
selector imposed it. A gate of the form `|x_new/x_seed − 1| < tol` is a truncation band around
the regressor of the obvious test: measured, a 1 % seed-referenced gate produced
**+0.128 ± 0.042** under a null with zero effect by construction, while the same gate
referenced to a fixed nominal gave **+0.002 ± 0.053**. And a "re-derived" cell from a pipeline
that hardcodes the hand-index cell as its seed confirms nothing — an unseeded re-index of the
same data landed **0.35 %** away in both axes.

## 11. A symmetry verdict and a significance test that share one σ — INTRINSIC

If the σ that decides "is the split significant" is the same σ that decided "is the lattice
tetragonal", those are **one test, not two**, and the agreement between them is circular.
With a measured *radial* σ instead, one dataset's "tetragonal, split not significant" became
"monoclinic, split at 3.3–3.7σ". Also test stability: on that data **17 of 71 single-spot
deletions flipped the crystal system.**
