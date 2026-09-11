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
"not obtainable from this set", full stop. **If it is True, that is not yet permission** — rank without
partners does not protect the splitting (§14). Configured escape: a wider ω wedge that brings in
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

**Which σ goes into that N.** The 0.417 pp is a bootstrap over 71 partner-supported reflections. A bootstrap
over a handful of reflections does not measure σ: on a held-out S5 position (dry run 2026-09-10, p = 378,
7 reflections that fail §14's partner gate) the manual bootstrap gave sd 0.065 % and the packaged one 0.142 %,
2× apart, and feeding either into the 3σ rule says N ≈ 2–10 would suffice — against the 1055 measured here.
Until your own set passes §14 and is of comparable size, take σ from the 71-reflection measurement,
scaled by √(71/N), as the floor; never from a bootstrap over the same few reflections.

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

## 12. An a/b splitting from d-spacing SCATTER, without indexing — INTRINSIC

When a multi-grain sample cannot be indexed it is tempting to read the splitting off the within-family
spread of d, since a↔b sends d(h,k,l) to d(k,h,l). **It does not work, and it fails by producing
SIGNIFICANT false positives on a crystal with a = b exactly.** In a dense reciprocal lattice the
splitting acts as a free position-adjuster: raising δ spreads the predicted branches and improves the fit
to scatter of any origin.

Measured on La3Ni2O7 with three successive estimators, each refuted on the same tetragonal control:
per-family variance regression; the same with a d-dependent radial-error floor; and a mixture maximum
likelihood over all spots, which returned **δ = 0.630 %, χ² = 6.67, p = 0.0098 on the tetragonal
control**. Planting δ = 0 exactly and varying only the scatter, it returned δ = 0.190 % at p = 0.013 for
σ = 0.20 %. All three recovered a PLANTED δ faithfully on clean synthetic data, so the failure is not the
algebra: strain scatter and splitting are not separable without knowing which spot is (200) and which
(020), and that is exactly what an orientation provides.

**Consequence:** an a/b splitting in a multi-grain sample needs enough reflections per grain to index it
— a wider ω range — not a cleverer statistic. The structural degeneracy ships as a test,
`midas_defect/tests/test_ab_from_dscatter_is_refuted.py`.

## 13. tx from a free fit over a short rotation — CONFIGURED

Over a few degrees of ω the crystal orientation absorbs most of any tx (about 80 % over S5's 12°, to a
0.021° residual), and tx IS the rotation-axis tilt about the beam, so "freeing tx does not improve the fit"
says nothing. Force the hand-indexed targets instead and read the parameter they demand; measure tx from
Friedel ω-splitting. A wider rotation range moves this limit. (`phase-1-geometry.md`)

## 14. An a/b splitting from a set with rank but no partners — MEASURED here, and it looks significant

`ab_separable` asks only for rank ≥ 2 of the (h², k²) design. It does not ask for an (h,k)/(k,h) partner —
the only pair that compares a with b at the same nominal |G|, where a radial systematic cancels. Measured in
this doc set's own test run on 2604 domain 1: `ab_separable` True, `partner_multiplicity` {} (zero partners),
`index_asymmetry` 12 : 0, full-metric fit condition number 1395 — and the joint fit returned
**δ = 1.80 % with a bootstrap interval [0.17, 10.36] that excludes zero.** That interval is the §3 artifact,
not a measurement. **Gate: rank AND partners (with their multiplicity) AND the asymmetry, before any δ is
computed.** On S5 the same run gave δ = 0.94 % [0.10, 9.11] from 12 reflections, 7 of them called sensitive —
but each partner pair rested on ONE spot on one side ((0,1) 2 : 1, (1,2) 1 : 3), one blob-finding decision from
nothing, and even past that gate §5 applies: that N cannot resolve a sub-percent splitting.

## 15. A pressure gauge "found" in a dense ring list — MEASURED here

Line matching gets cheap when a frame carries 33–60 rings between 2° and 23°: within ±0.04° of some ring lies
**18.6 %** of the 3–21° band on 2604 and **11.2 %** on S5, so a scan over the lattice parameter finds a few
coincidences somewhere. This doc set's own test harness then counted matched RINGS as lines, and returned
**"Pt, a = 3.7775 Å, 3 lines, 41.7 GPa" on 2604** — 2 distinct lines, one of them (220) sitting on the Re
gasket's (110), Pt's strong (200) absent, and no Pt gauge anywhere in the 2604 record — and **"Re,
a = 2.5875 Å, 4 lines, 108.9 GPa" on S5** — 3 distinct lines with Re's strongest line (101) absent, 90 GPa
from the Pt gauge on the same frames. For comparison the 2604 gasket fit (a = 2.669 Å) matched 5 of its 6
lines. **Gate: count DISTINCT lines; require the strongest expected lines; never let a line another phase
already explains count again; and compare with the chance rate — the ring coverage above, or the identical
scan for a phase known to be absent.** (project record; `phase-4-phase-id.md`)

## 16. A crystal system decided before the a/b gate — MEASURED here

`holohedry_from_fit` on the 2604 set of §14 said **orthorhombic** (tolerance 0.0126) — on a set that cannot
protect a from b — and on S5 **tetragonal** (tolerance 0.108) from 12 reflections, while a hand
`rel_tol = 1e-3` said triclinic on both. A σ taken from an ill-conditioned fit is small exactly along the
direction the data do not constrain. Decide the crystal system only after §14's gate, and remember §11: the
symmetry verdict and the splitting test share one σ.
