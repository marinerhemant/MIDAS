# Phase 2 — DCT: spots → Friedel pairs → self-calibration → grains

**Goal:** a list of grains, each with an orientation and a position, every acceptance
threshold justified by a null.

## 2.1 Segment

Segment diffraction spots frame by frame. The output is a spot table: centroid `(y, z)`,
frame/ω, integrated intensity, size.

Two calibration points from the real run: **71 554 spots** at an 8σ threshold, of which
**29 295** (82 %) went on to pair. Lowering the threshold *does* find more real spots — but on
this data the extra ones were **unpairable**, because their ω+180° partners were too weak.
More sensitivity did not become more grains. Test that before spending on it.

## 2.2 Friedel pairing — and why it is the load-bearing step

A grain reflection at ω has a partner at ω+180° on the opposite side. Pairing them is what
makes everything else possible:

```
(y + y')/2 - c    ->  the ring radius, with grain-position blur CANCELLED
(z - z')/2        ->  the out-of-plane coordinate a point grain on the axis would give
```

The cancellation is exact in the algebra, and you can see it in the data: **paired ring radii
are sharp (widths 0.9–2.9 px) where unpaired radii are broad.** That contrast is the check
that pairing is working — use it rather than trusting the pair count.

**Pairing breaks at high grain density**, and it broke here. The matcher was genuinely wrong
until the ring constraint was applied *during* matching rather than after: on-ring pairs went
**5 679 → 16 282**. If your pair yield is poor, suspect the matcher before concluding the data
are weak.

## 2.3 Self-calibrate from the pairs

With pairs in hand, fit lattice type, `λ/2a` and distance (phase 1.3). This is
self-calibration — no external standard is involved — so quote rings used and free parameters
with the result.

## 2.4 Index: feed the indexer *virtual* spots

**Do not feed raw spots.** Use the pair-derived virtual spots from §2.2. The reason is
concrete: rings here are **84 px apart**, while grain position shifts a raw spot by up to
**150 px**. Ring assignment on raw spots is not merely noisy, it is impossible. With virtual
spots the sample-radius and beam-height parameters go to the floor and ring assignment becomes
unambiguous.

Practical traps in the MIDAS indexer, each of which cost time here:

| Trap | Reality |
|---|---|
| `IndexBest.bin` column meanings | **col 13 = `n_t_spots`, col 14 = `n_matches`.** The module docstring says the opposite; `_seed_record` is authoritative. Reading col 13 makes every seed look perfect |
| `OutputFolder` | The directory **holding** `Spots.bin`, *not* its parent |
| Margin units | **µm against the ring radius.** At an 880 µm ring radius, a 150 µm margin is a ~10° window — everything matches |

## 2.5 Set the tolerance from a null — this is the part that gets skipped

**The ω-scrambled null.** Permute ω across the spot table and re-run the identical pipeline. A
real signal indexes; a tolerance artefact indexes just as well scrambled.

What happened here, in order:

| Margin | Real | ω-scrambled null | Verdict |
|---|---|---|---|
| loose (4× the adopted) | 2761 of 2902 seeds, completeness 0.250 | **2761 of 2902, completeness 0.250** | **Indistinguishable.** Retracted |
| adopted (0.52° on ring 1) | 488 seeds (ring-1 seeding), 814 (all-ring) | **0 seeds** | Real |

The adopted acceptance — minimum match fraction **0.09** — sits above the null's *maximum*
completeness of **0.069**. That is how a threshold should be chosen: read off the null's tail,
not off a plot of the real data.

Two further checks worth running:

* **Internal angle.** Indexed seeds agreed to **0.33°**, which is the measurement scale here.
* **Cross-check the misorientation implementation.** Local cubic-symmetry misorientation
  agreed with `midas_stress.misorientation` to **5.7e-14°** — after the frame fix below.

## 2.6 The symmetry-frame trap

With `v_sample = U v_crystal`, symmetry equivalents are `U·S`, so the misorientation between
two grains is

```
Ua^T Ub S          <- S on the RIGHT
```

Putting `S` in the middle (`Uaᵀ S Ub`) reported **29.8°** for a pair genuinely **0.33°**
apart, and left the seeds refusing to cluster: **367 clusters from 488 seeds**, against **205**
after the fix. It was caught only because the independent `midas_stress` cross-check
disagreed. Always run that cross-check.

## 2.7 Positions are in the SAMPLE frame

Each Friedel pair flashes at its **own** ω, so solving for one shared *lab* position across a
grain's pairs is the wrong model. Include `Rz(σω)` in the design matrix so each pair
contributes in the frame it was measured in.

Effect: triangulation residual **52 → 41 µm**, and the grain-centre cloud changed from a smear
into a recognisable sample cross-section — compact, roughly circular, ~350 µm across in the
top view, spanning z ∈ [−62, 58] µm inside a 220 µm beam. **That physical agreement only
appeared after the fix**, and is the check that the frame is now right.

## 2.8 Accept grains on two independent criteria

Accept jointly on:

1. **match count** against the null, and
2. **the position the grain's own Friedel pairs imply** — an independent quantity.

A grain that indexes well but implies an impossible position is not a grain.

## 2.9 Expect a detection floor, and recognise it

The run here reached **843 grains** where a space-filling structure at the measured grain size
would hold ~2010. Every stage was tested and each was either ruled out or already at its
limit: segmentation threshold (extra spots real but unpairable), the matcher (broken, fixed),
match-count acceptance (never binding), seeding coverage (exhaustive — 1.05 M of 1.05 M
combinations), duplicate grains (ruled out: close pairs misoriented 57–60°), attenuation
breaking pairs (ruled out: r = +0.003 with depth).

The informative negative: **pair-free voxel indexing**, which removes the need for Friedel
pairs entirely by asserting a position, found 256 grains in unmeasured space — of which
**220 were rediscoveries** of grains already in the list and only **36 new**, of which **19**
cleared the same acceptance every other grain cleared. Net: **843 → 862, +2.3 %**.

Two things follow. First, the deficit is a **detection-sensitivity limit** reached from four
independent directions, not an indexing-strategy problem. Second, those **220 rediscoveries
are the strongest validation in the whole reconstruction**: an independent route, using only
unassigned spots, re-derived orientations found by the pair-based pipeline.

## 2.10 Exit criteria

- [ ] pair yield, plus the **paired-vs-unpaired ring width** contrast
- [ ] self-calibration with rings and free parameters quoted
- [ ] every tolerance traced to a null, with the null's number written down
- [ ] misorientation cross-checked against `midas_stress`
- [ ] positions solved in the sample frame, with the residual before and after
- [ ] grains accepted on match count **and** implied position

**Halt** if any threshold cannot be tied to a null. A tolerance chosen by eye is the single
documented way this pipeline has produced confident, entirely fictitious grains.
