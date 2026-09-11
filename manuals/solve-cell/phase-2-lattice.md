# Phase 2 — finding the lattice

Three routes, in the order to try them when nothing is known.

## Route A — ab initio, no cell supplied

```python
from midas_hkls.ab_initio import index_ab_initio
res = index_ab_initio(g, two_pi=True, sigma_g=SIGMA_G, min_reflections=20)
```

**`sigma_g` must be MEASURED, not guessed** — it sets what counts as a significant excess and
therefore what the routine will call a lattice. Measured on one dataset: `sigma_g = 2.76e-3
Å⁻¹`, corresponding to a 1.18 px residual, and that value implied `σ(δ) = 0.58 %` on any a/b
splitting — enough to say up front that a 0.34 % target was unreachable.

**On a centred lattice the answer is the PRIMITIVE cell.** For I4/mmm (body-centred), a
correct result reads 3.632 / 3.705 / 9.947 Å at 79.6 / 79.4 / 88.4°, V = 129.4 — the third
axis being the centring vector (a/2, a/2, c/2). Truth was 3.612 / 3.612 / 9.959, 79.6 / 79.6 /
90.0, V = 125.6: errors 0.6 % / 2.6 % / 0.1 %, V 3.1 %. Compared against the CONVENTIONAL cell
(V = 251.1) that same correct answer looks like a "×0.52 failure". **Convert before judging:**

```python
from midas_hkls.conventional import to_conventional
conv = to_conventional(res.cell)
```

**`success=True` is not a verdict.** On the delivered S5 position `index_ab_initio` reported success on a
cell with a **1.466 Å** axis and 13 % of spots indexed — physically impossible for any oxide. Check the
shortest length (no lattice translation in a real crystal is much under ~2 Å), the volume against the
formula volume and the indexed fraction before converting or seeding anything from it. On the delivered
2604 position it succeeded properly (81 of 415 spots, primitive 3.657/3.774/9.902 Å) — and `sigma_g`
changed nothing on either position (identical cells with `None` and `2.76e-3`).

**`to_conventional` decides the crystal system ONCE, at one tolerance.** On the delivered 2604 position the
primitive cell above converted to **triclinic** at the default tolerance, and to a non-standard
**orthorhombic-P** at `rel_tol = 0.0768` — derived from the cell's own σ, and identically from
`tolerance_from_fit` on a UB fit — never to the body-centred cell that the row repeat and the declared hand
index describe. Do not let the conversion choose the lattice: take c from a row repeat (Route B), test a
DECLARED candidate cell (Route C), and decide the symmetry after the a/b gate (`ENVELOPE.md` §16).

## Route B — from reciprocal-lattice rows

Works when ab initio will not, and it is the route to prefer when one axis is obvious (a
visible (00L) ladder) but the in-plane indexing is not.

```python
from midas_defect.rows import find_lattice_rows, index_from_row, candidate_row_spacings
rows = find_lattice_rows(q, I, a=A, c=C, two_pi=True, space_group_number=SG,
                         identify=False)          # identify=False -> cell-free repeat
```

`identify=False` gives the **fundamental repeat with no cell assumed** — the honest first
answer. Measured on one sample: 9.5978 Å over 7 rungs, which is c/2 = 19.20 for even-L or
9.60 for all-L; the latter is unphysical for that structure, so the assignment is forced, and
**that reasoning must be written down** because it is the step that sets c.

**PASS `a`, `c` AND `space_group_number` EXPLICITLY.** They default to La3Ni2O7
(3.6116 / 19.2516 / 139). A call that omits them is silently seeded with a nickelate.

With `identify=False` the finder looks only for a repeat. On the delivered 2604 position, with no cell
supplied, it found a **13-rung row repeating every 9.6188 Å** — c/2, so c ≈ 19.24 Å (an independent route
gave 9.5978 Å, 0.2 % away). On S5, where c* lies near the beam, it found only 3-rung rows: there is no
ladder to find, which is itself the answer.

## Route C — seeded from bright spots

```python
from midas_defect.seed_index import find_seed_orientation
```

**Do not rank seed families by d-spacing** when c* lies near the beam. The largest-d families
are (00L), which have **zero observed spots** in that geometry, while the brightest spots in
the dataset sat in a family ranked **18th of 23** and were never tried. Rank by observed
intensity, or sweep all families.

### The packaged route: `index_from_cloud`, with the ω sign scanned

Contract in `README.md` (a callable cloud for `resolve_conventions`, one set of tolerances spelled two ways).
What it measured on the delivered positions: the ω sign was decisive on both (2604 +1: 58 against −1: 1;
S5 +1: 11 against 2); 2604 came to **58 INDEXED / 0 MISSED / 5 MASKED / 37 ABSENT** of 100 predicted at
d_min 0.93 Å (the recorded gate: 51 / 0 / 5 / 16 of 72 at 1.10 Å), S5 to 11 / 0 / 2 / 22.

**Feed the sign scan only seedable spots.** `resolve_conventions` seeds from the `n_bright` brightest spots it
is given. On a held-out 2604 position (dry run 2026-09-10) those were gasket and anvil spots: the scan returned
−1 at 4 : 1, "decisive" under the ratio rule, and wrong — the crystal's own reflections gave +1 at 38 : 0
(`find_domains`) and the cell-free rows 10 : 2. Pass `~pw & ~stationary` spots, and note that
`ConventionScan.decisive` now also requires the winner to reach `min_assigned` (8): a 4 : 1 is not a verdict.

**A declared seed confirms nothing about the cell** (README trap 2). These runs start from the hand-indexed
or the collaborators' cell, so they test the orientation and the completeness, not the lattice.

**Before 2026-09-10, `index_from_cloud` converged every crystal's cell with I4/mmm extinctions** — it did
not pass the crystal's space group on (`PACKAGE_NOTES.md` §10). On S5 (Fmmm) the fix took INDEXED 9 → 11
and the median residual 2.05 → 0.61 px; on 2604 (I4/mmm) the log is byte-identical. Re-run anything
non-I-centred indexed through it before then.

**More than one domain at a position: `midas_defect.domains.find_domains`** (contract in the defect
manual, `phase-2-index.md`): row-seeded domains first, then pair-seeded ones against a whole-search null,
with the cell and space group required. Where c* lies near the beam and no row of three rungs exists — S5
at 30 K — pass `seed_from_nominal=True`, or the call returns nothing without saying why; a held-out S5
reader (2026-09-10) had to rebuild the pair search by hand for that reason. `index_from_cloud` above finds
ONE orientation.

**`frame` is the RAW fractional frame index.** `resolve_conventions` rebuilds ω as
`omega_first_deg + omega_step_deg * frame`, so map the ingest's live-stack centroid through the live
indices first.

## Rules that apply to all three routes

**1. Never ring-filter before a 3-D indexer.** It cost 17 of 45 reflections on one sample.
Remove powder with `flag_powder` on the SPOTS (a 3-D, azimuth-aware test), not by cutting
radii.

**2. Separate powder first, and count what you removed.** `flag_powder` flagged 199 of 442
spots (45 %) on one sample; removing them changed the refined cell entirely.

**3. Set every search window from the model's own residual distribution.** An eyeballed 2°
window — 14× the median residual — manufactured a false positive that was retracted the same
day.

**4. Split the tolerance.** A single scalar tolerance in G-space mixes a ~1 px RADIAL error
with an ANGULAR error set by the ω step. A split radial/angular criterion took one sample from
7 to 12 reflections **without loosening the cell**. Beware also that a 3-D q-vector angle
folds an ω error in with an azimuthal one: two reflections accepted that way were later found
to be **6.16° and 3.34° off in ω** — 6 and 3 frames on a 1° step.

**5. Beating a null is necessary, not sufficient.** Run a **decoy** — a deliberately wrong but
plausible model — through the same machinery. On one sample a decoy cell also beat the ω null,
and only a structural test separated them.

**6. Re-run every null with the floor lowered.** A refiner returning `None` below 6 matches
made the null AND the decoy report 0, producing a bogus "p = 0.000, decoy 0". Uncensored: null
median 3, max 5; decoy 4.

**7. The h↔k permutation null for an a/b splitting is INVALID — it cannot fail.** Do not use
it. See `ENVELOPE.md` §1.

## Completeness, as soon as you have a candidate

```python
from midas_defect.completeness import audit_completeness, window_from_residuals   # INDEXED / MISSED / MASKED / ABSENT
```

**MISSED is the number that matters** — predicted, unmasked, and not observed. A good result
looks like 51 INDEXED / **0 MISSED** / 5 MASKED / 16 ABSENT of 72. A weak one looks like
4 INDEXED / 26 ABSENT of 32, and that sample never did produce a quotable cell.
**A spot count is not a completeness.**


On this doc set's own test run: 2604 **58 INDEXED / 0 MISSED** / 5 MASKED / 37 ABSENT of 100; S5 11 / 0 / 2 /
22 of 35 after the P12 fix. Zero MISSED on both — and on S5 that is 11 reflections, which no splitting survives.