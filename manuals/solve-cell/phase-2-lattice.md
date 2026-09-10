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

## Route C — seeded from bright spots

```python
from midas_defect.seed_index import find_seed_orientation
```

**Do not rank seed families by d-spacing** when c* lies near the beam. The largest-d families
are (00L), which have **zero observed spots** in that geometry, while the brightest spots in
the dataset sat in a family ranked **18th of 23** and were never tried. Rank by observed
intensity, or sweep all families.

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
from midas_defect.completeness import ...     # INDEXED / MISSED / MASKED / ABSENT
```

**MISSED is the number that matters** — predicted, unmasked, and not observed. A good result
looks like 51 INDEXED / **0 MISSED** / 5 MASKED / 16 ABSENT of 72. A weak one looks like
4 INDEXED / 26 ABSENT of 32, and that sample never did produce a quotable cell.
**A spot count is not a completeness.**
