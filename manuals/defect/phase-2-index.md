# Phase 2 — orientation from the cloud, and the completeness audit

You need a predicted reciprocal lattice before any voxel can be attributed. If you already
have trustworthy grains from `ff-hedm`, use them and go straight to the audit.

## Indexing from the cloud

```python
from midas_defect.seed_index import find_seed_orientation, refine_U_lattice
res = find_seed_orientation(qx, qy, qz, intensity, crystal=my_crystal,
                            n_bright=30, tol_q_rel=0.02, tol_angle_deg=3.0)
```

Pair-voting over bright cores: for every pair of non-collinear centroids and every compatible
(hkl, hkl) assignment, build a candidate `U` and score it by how many *other* centroids it
explains. Phase-agnostic — it takes any `midas_hkls.Crystal`.

**The tolerance is split on purpose.** `tol_q_rel` and `tol_angle_deg` are separate because a
single scalar tolerance in q-space mixes two different errors: a ~1 px radial error, and an
angular error set by the ω step. On one dataset splitting them took a solution from 7 to 12
reflections **without loosening the cell**.

**Do not rank seed families by d-spacing** when a principal axis lies near the beam: the
largest-d families are then the ones with zero observed spots, and the brightest spots in the
dataset can sit in a family that never gets tried.

**Before indexing, settle the orientation convention on this cloud.**
`bragg_diffuse.check_orientation_convention` decides `OM` versus `OM.T` from the data. Getting
it wrong does not fail loudly — it produces an orientation that indexes *something*. See
`phase-1` step 5 for why it cannot be looked up.

## When the bright-core seeder cannot see the domain

`find_seed_orientation` votes over the brightest cores. A minority domain 64–88× fainter than
the dominant one is not in that set: on La₃Ni₂O₇ it needed `n_bright=150` to appear at all, and
found nothing at 20. `rows.py` is the second route in, and it does not rank by intensity.

### A lattice row

```python
from midas_defect.rows import (hkl_box_from_geometry, candidate_row_spacings,
                               find_lattice_rows, index_from_row)
hmax, lmax = hkl_box_from_geometry(geom, wavelength_A, a, c)   # do NOT guess these
rows = find_lattice_rows(q, intensity, spacings=candidate_row_spacings(a, c, sgnum=139))
U, n, cell = index_from_row(q, intensity, rows[0], a=a, c=c)
```

A **row** is collinear spots at regular spacing through the origin — the (00L) ladder is one
instance, and `find_lattice_rows` handles any direction. A row identifies a lattice *direction*
without seeding from intensity, so it reaches a domain a bright-core seeder cannot, and it
measures that direction's spacing directly, fixing two of three orientation degrees of freedom
and leaving a 1-D scan (`row_scan_range`).

Four things about rows were wrong until verification caught them:

* **Parity is per direction, from the space group** — not a blanket "even multiples only". With
  the blanket rule, synthetic rows along (1,0,0), (1,1,1), (0,1,2), (1,1,2) and (1,2,1) returned
  nothing; 1 of 8 directions worked. `allowed_multiple_parity` does it per direction.
* **The spacing window must cover the observed `n·|G|`**, not `|G|`.
* **`candidate_row_spacings` is orbit-reduced, one entry per symmetry orbit.** Listing (0,1,1)
  and (1,0,1) separately made every mixed row look ambiguous — best and runner-up both at
  rel = 0.0108 — so identification refused on rows it should have taken.
* **A spacing estimate must resist sub-multiples.** Letting `d/g` win a "most rungs" contest
  gave 16 rungs at a half-spacing; `refine_row_spacing` fits an LS slope through the origin and
  `_robust_common_factor` collapses sub-multiples. Its robust pass runs **post-convergence** —
  during iteration it trimmed high-|n| rungs, mistaking the fit trend for outliers.

### A pair, once the cell is known

```python
from midas_defect.rows import index_from_pairs, refine_to_convergence
U, n, seed_pair, margin = index_from_pairs(q, I, B_seed, a=a, c=c, tol_q=0.05,
                                           min_reflections=4)
```

Anchors are **stratified** — half the brightest, half sampled across the intensity range —
because an all-brightest anchor set finds the dominant domain again. The margin is
symmetry-reduced (422 here); without that, two descriptions of the same crystal look like
rivals up to 90° apart and every margin test on a well-determined domain reads zero.

### Every search needs a null, and the obvious two are wrong

```python
from midas_defect.rows import search_null
threshold, _ = search_null(q, I, B, my_search, n_rep=40, alpha=0.05)
```

A search that scores *N* orientations and keeps the best has *N* chances at a coincidence. The
question is never "does this orientation explain 12 reflections" but "does this **search** find
12 on structureless data of the same kind".

* The **runner-up** fails the moment a second real grain exists — each grain's nearest rival is
  the other, the margin collapses, and the search stops after one domain.
* Randomising the **orientation** while keeping the real spots is not a null either: with a few
  thousand tries one lands within a degree of a real grain, and the "null" then holds 67–79
  against 96 for real grains.

So randomise the **spots**: keep every |q| exactly and replace every direction. That preserves
the radial structure a search exploits — shell occupancy, candidate hkl per spot — and destroys
only the mutual angles a crystal imposes. `search` must return an int.

### Refine and re-match together, or the cell belongs to something else

```python
res = refine_to_convergence(q, U_seed, a0=a_seed, c0=c_seed, avail=unclaimed)
res.lat        # IS the least-squares fit to q[res.claim] / res.hkl[res.claim]
```

A seeded search must bootstrap its first match from another domain's cell — there is no cell for
the new domain yet. **What it must not do is stop there.** Two distinct bugs come from stopping,
both of which reached a production analysis and neither of which the suite caught:

* **Seed-cell anchoring.** Choose the reflection set with a neighbour's cell, never re-choose,
  and the fitted cell can be pulled toward the neighbour's. This one is a *correctness* argument,
  not a measured one: every attempt to size it on this dataset has been confounded (see the
  warning below), so iterate because selecting reflections with a foreign cell is wrong, not
  because a number says so.
* **A reported cell never fitted to the reported reflections.** Assign the candidate *inside* the
  loop and you report a cell fitted to the previous iteration's claim, or to a rejected round.
  **26 %** of one run's stored `c` values were not the LS fit to their own stored hkl list,
  median **0.0041 Å** — against a 0.035 Å effect under study.

`refine_to_convergence` iterates on the domain's own cell, keeps a round only on a strict gain
(more reflections, or an equal count at lower residual, which makes the loop monotone and safe
to run blind), and **refits once on the converged set before returning**. That postcondition is
the point of the function, and `test_refine_to_convergence_cell_matches_returned_reflections`
pins it.

**Do NOT claim this fixes anchoring — and read this before measuring anchoring at all.**
The obvious experiment (regress a seeded domain's `c` on its position's seed `c`, before and
after) is **confounded by the acceptance gate**, and the trap is worth more than the fix:

> A gate of the form `|c/c_seed − 1| < tol` is a **truncation band around the regressor**. It
> manufactures a positive c-on-c slope with **zero** anchoring present. Under a y-permutation
> null (no anchoring by construction) a 1 % seed-referenced gate produced **+0.128 ± 0.042**,
> while a gate referenced to a fixed nominal cell produced **+0.002 ± 0.053**.

So a seed-referenced gate fabricates the very correlation it would then be read as evidence of.

**The 2x2, run one change at a time** (233 positions, all four arms re-run; paired bootstrap):

|                     | OLD `\|c/c_seed-1\|<1%` | NEW `\|c/c_nom-1\|<1.5%` |
|---|---|---|
| **no loop** (pre-fix) | +0.2260 | **+0.0836** |
| **loop + refit**      | +0.2073 | +0.1247 |

Loop alone: **−0.019** [−0.098, +0.055] old gate, **+0.041** [−0.045, +0.135] new gate — zero,
and *positive* on the new gate. Gate alone: **−0.142** [−0.236, −0.059], larger than the whole
apparent improvement. The buggy code with only the gate swapped (+0.084) beats the "fixed"
pipeline (+0.125). Gate-corrected anchoring does not fall at all: **+0.095 → +0.123**.

**Why iterate at all, then.** Because the loop does what it is for: it moves cells OFF the seed,
and under a seed-referenced gate that gets them rejected — the loop loses 43 pair domains
(532 → 489) on the old gate. And 4.9 % of *ungated* independently-seeded domains sit more than
1 % from the seed cell, so the old bound was clipping a real population. The loop and the
nominal gate are a package; the loop is not usable without it. What the loop is **not** is the
reason the measured slope fell.

**Two rules follow.** Reference any cell-plausibility gate to a **fixed nominal cell**, never to
a neighbour's — a seed-referenced gate is not merely imprecise, it is circular. And when you
change a gate and a loop together, you cannot attribute the result to either; change one at a
time, or re-run one arm with the other's gate so the arms differ only in the thing you mean to
test. The true size of seed-cell anchoring on this dataset is **currently unknown**.

## Before you report an a/b splitting from an indexed set

The count that gets quoted — "N reflections are sensitive to a vs b" — hides three things,
and each of them produced a wrong answer on La3Ni2O7.

```python
from midas_hkls import (ab_sensitive_mask, partner_multiplicity,
                        ab_separable, shear_separable, distortion_condition,
                        index_asymmetry)

sens  = ab_sensitive_mask(hkl)          # per reflection: |h| != |k| AND its (k,h) partner present
pairs = partner_multiplicity(hkl)       # {(min,max): (n_fwd, n_rev)} -- how many spots per pair
```

**1. A pair supported on one spot is one blob-finding decision away from nothing.**
On 2604 domain 2, six reflections were called sensitive — but the (1,0) side of the
(0,1)/(1,0) pair was a **single spot**. Remove it and all six stop being sensitive. Report
`partner_multiplicity`, not just the count.

**2. `ab_sensitive_mask` is blind to a γ shear, which may be the real distortion.**
It credits `(1,0)/(0,1)` and rejects `(1,1)/(1,-1)` because `|h| == |k|`. But in the RP
**subcell** an Fmmm supercell distortion is a **γ shear** — γ = 89.613° vs 90.387° for the two
variants, with a and b IDENTICAL — so the pair that actually splits is `(1,1)/(1,-1)`, the one
the mask rejects, while `(1,0)/(0,1)` does not split at all. Use `shear_separable` (rank 3)
whenever a shear is possible; `ab_separable` (rank ≥ 2) is not enough.

**3. The mask is the rule for a |G|-only analysis and understates a vector pipeline.**
With full 3-D q vectors the in-plane azimuth `atan2(k/b, h/a)` depends on a/b for ANY h, k both
non-zero, no partner needed. Do not use the mask to argue a vector pipeline is insensitive.

**Then check the design, not just the count.** `index_asymmetry` measures how unevenly a and b
are constrained by the observed indices. In a limited ω wedge they are not symmetric: measured
across the 2604 raster, |k| > |h| in 5004 reflections against |h| > |k| in 1009 — a 5× asymmetry.
`1/a²` then rides on a sparse design column, any radial systematic pushes `a` one way, and the
fitted splitting comes out **with a consistent sign across domains**. Since a↔b labelling is a
per-domain gauge, a common sign is physically impossible and is the signature of this artifact.

**A gate seeded on the answer is not a confirmation.** `repro/gate_packaged.py` in the
nickelate analysis hardcodes `CELL_A, CELL_C` from the hand index and refines from it — so its
agreement with that hand index measures nothing. A genuinely unseeded re-index of the same data
landed 0.35 % away in both axes. If you quote a cell as independently confirmed, check what
seeded the run.
