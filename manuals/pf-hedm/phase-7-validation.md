# Phase 7 — is the map real? The ω-shuffle null and the chance ceiling

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).
> Read this **before quoting any per-voxel result, grain count, or acceptance
> threshold.** It is cheap (one re-index, no re-prep) and it is the only thing
> that separates a map from a plausible-looking fit to a dense spot cloud.

Every other phase asks whether the pipeline ran. This one asks whether the
answer carries information about the sample. On a dense scanning dataset those
are genuinely different questions: **completeness saturates**, and a saturated
completeness cannot tell a grain from a coincidence.

---

## 7.1 The null

**Permute the ω columns (`Omega`, `OmegaIni`, `OmegaDetCor`, kept consistent
with each other) within each ring, independently per scan file.** Re-bin,
re-index, change nothing else.

That preserves exactly:

- every spot's detector position, radius, η and ring assignment
- the per-scan per-ring **spot count** and **ω multiset**
- therefore the beam-position gate statistics — each voxel is offered the same
  number of spots as in the real run

and destroys only the **position–ω joint structure**, which is what encodes
orientation.

**Per-scan is load-bearing, not tidiness.** In PF mode ω enters the beam gate
(`newY = x·sin ω + y·cos ω`, kept when `|newY − ypos[scan]| ≤ scanTol`), so
shuffling ω *across* scans changes how many spots each voxel may see and
confounds the null with a gating change.

**Both arms must run with the same binary and the same `ScanPosTol`.** See
[`DIAGNOSIS.md`](DIAGNOSIS.md) "Solution counts differ between a pipeline run
and a hand-run of the same binary" — a missing `ScanPosTol` silently widens the
gate by `0.1/2` µm and moves every count.

Script: `scripts/pf_null_shuffle.py` in the campaign tree (`--arm real|null`,
`--n-scans`, `--n-voxels`). It asserts the multisets are unchanged and that no
non-ω column moved, and refuses to run if either fails.

## 7.2 What to compare

Winner selection must be the production routine (`per_voxel_cluster`) — the same
call `find_grains` makes — not `argmax`, which is a different answer.

| statistic | why |
|---|---|
| voxels with any solution | coarse, but the first thing to move |
| **best completeness per voxel** | what `find_grains` selects on |
| **max best-completeness over all null voxels** | **the chance ceiling** — see §7.3 |
| best IA (`CalcAvgIA`) | separates where completeness saturates |
| **distinct winner orientations ÷ voxels** | threshold-free, see §7.5 |

## 7.3 The chance ceiling, and why the shipped gate sits below it

**The chance ceiling is the highest best-completeness any null voxel reaches.**
Below it, real and chance overlap and completeness cannot discriminate.

**The ceiling is NOT predictable from spot density. Measure it per layer.**
Five layers, four samples, both arms at the correct 0.75 µm gate, banked real
arm against a re-run null:

| layer | spots | spots/voxel | real best-comp median | **null max = ceiling** | real voxels in the chance band |
|---|---|---|---|---|---|
| A | 22 848 | 282 | 0.6481 | **none — the null found NOTHING** | **0 %** |
| B | 418 783 | 5 170 | 0.9423 | **0.5333** | **0.0 %** |
| C | 935 620 | 5 536 | 0.9231 | **0.6957** | 24.5 % |
| D | 1 294 542 | 3 586 | 0.8943 | **0.8333** | 39.8 % |
| E | 1 391 165 | 8 231 | 0.9600 | **0.7500** | 14.2 % |

⚠ **A four-point version of this table looked cleanly monotonic in spot count
and it was over-read.** Layer E — the densest, and the one with the most spots
per voxel — came in *below* layer D on both the ceiling and the contaminated
fraction, against an explicit prediction that it would be highest. Neither total
spots nor spots-per-voxel orders this table. Density is the *mechanism* (a sparse
enough list has no chance floor at all, layer A), but it is **not a predictor you
may substitute for the measurement.**

**An empty null is a real and common outcome, not an error.** On the 23 k-spot
layer the shuffled list produced *zero* accepted solutions in *any* voxel:
there is no chance floor to clear, and a completeness of 0.65 there is worth
more than 0.92 on a saturated layer. Handle `null.size == 0` explicitly — the
first version of these scripts crashed on it.

⚠ **Do not port a ceiling between layers, and do not assume the low-completeness
layers are the suspect ones.** The 23 k-spot layer above was flagged for review
precisely *because* 61 % of its voxels sat below a **different** layer's ceiling
— and it turned out to be the cleanest layer in the campaign. Sparse layers have
no chance floor; dense ones do.

⚠ **The max is the noisiest statistic in this table.** It is one voxel's extreme
and it is quantized (0.8333 = 5/6, e.g. 45 of 54 reflections). Tightening the
gate from 0.80 to 0.75 µm moved the medium layer's ceiling 0.7083 → 0.6957 but
left the dense layer's at 0.8333 exactly, even though that null's bulk did move
(148 → 139 voxels, p90 0.7500 → 0.7393, p99 0.7981 → 0.7731). **Report the null's
p99 alongside the max**, and state which gate both arms ran at.

**No null voxel reached completeness 1.0 on any layer**, against real medians of
0.65–0.96. The separation is real: **the PF per-voxel path passes its null at
campaign thresholds, with no change to the recipe.** That is the headline, and
it is what the beam-position gate buys (§7.6).

But the shipped acceptance gate is **`MinMatchesToAcceptFrac 0.500000`**, at or
below the ceiling on **every layer tested**, so the band `[0.50, ceiling]` is
accepted and reachable by chance:

| gate | medium (936 k) real / null | dense (1.29 M) real / null |
|---|---|---|
| **0.50 (shipped)** | 143 / 73 → **0.51** | 354 / 139 → **0.39** |
| 0.60 | 121 / 43 → 0.36 | 307 / 81 → 0.26 |
| 0.7083 | 108 / **0** → **0.000** | 267 / 33 → 0.124 |
| 0.75 | — | 251 / 6 → 0.024 |
| 0.80 | — | 227 / 1 → 0.004 |
| 0.90 | — | 174 / **0** → **0.000** |

Voxels whose winner sits at or below the ceiling: **24.5 %** (medium), **39.8 %**
(dense) — and **0 %** on both sparse layers. They may still be real; this test
cannot say so.

> ⚠ **Measure the ceiling per layer.** Porting layer C's 0.6957 onto layer D
> still admits 33 null voxels (null/real 0.124). **Never quote one layer's
> ceiling for another** — and note the failure runs both ways: the layer with the
> *lowest* completeness in the campaign (A, median 0.6481) turned out to have no
> chance floor at all, while the one that looked healthiest by completeness (E,
> median 0.9600) carries a 0.75 ceiling.

## 7.4 IA separates where completeness saturates

On the same layer, production winner selection:

| | best completeness | **best IA** |
|---|---|---|
| real | 0.9258 | **0.1393** |
| null | 0.6200 | **0.3669** |

IA separates 2.6×, independently of completeness. This is the measurement that
justifies promoting `CalcAvgIA` from a hidden tiebreak inside `per_voxel_cluster`
to a reported per-voxel quantity — worth doing precisely in the dense regime
where completeness has no dynamic range left.

**It does not always work.** In the merged-FF arm (§7.6) IA did *not* separate —
the null was *better* (0.2896 vs 0.3243). IA is a discriminator here because the
gate keeps the search sparse, not because IA is intrinsically robust.

## 7.5 Spatial coherence — a threshold-free screen for every banked layer

**distinct winner orientations at 1° ÷ voxels with a solution.**

A real grain spans several voxels, so real winners repeat. Chance fits do not —
each voxel invents its own orientation and the ratio goes to 1.

```
real:  144 voxels →  59 distinct   ratio 0.41
null:   77 voxels →  75 distinct   ratio 0.97
```

It needs no threshold, no chance ceiling and no re-indexing — it reads straight
off banked `Results/Result_OrientPos_voxel_*.csv`. Across 32 banked layers of the
reference campaign the ratio ran **0.035 – 0.490 (median 0.305)**; none
approached the null's 0.97.

**It is a screen, not a verdict, and low is not automatically good.** One edge
layer gave 284 voxels → **10** distinct orientations (0.035) ≈ 28 voxels/grain
≈ 42 µm — two orders of magnitude above that sample's ~0.29 µm primary particle
size. The reading is not "excellent map": it is that **`OneSolPerVox` maps only
the largest grains**, which win voxels on matched-reflection count. That is goal 1
working as designed, and it means **these maps are not a grain census** and must
never be quoted as one.

Script: `scripts/layer_coherence_qc.py`.

## 7.6 Why merged-FF fails this test, and why regular PF does not

`IndexerUnified.c:1005`:

```c
int doScanFilter = (nScans_ > 1);
```

and inside the **matching** loop (1094-1101), for every candidate observed spot:

```c
if (doScanFilter) {
    dy = fabs(yRot - ypos[scannrobs]);
    if (softMode == 0) { if (!(dy < scanTol)) continue; }   /* rejected from MATCHING */
}
```

So in regular PF the beam-position gate restricts **what a theoretical spot may
match**, per candidate grain, per reflection — not merely which spots seed.
**merged-FF writes a 1-row `positions.csv`, so `nScans_ == 1` and `doScanFilter`
is 0: the gate is off entirely** and every theoretical spot may match anything in
its (ring, η, ω) bin.

Measured consequence, merged-FF on a 935 k-spot merged list, 10 000 matched seeds:

| | seeds w/ solution | completeness med | IA med | distinct @1° |
|---|---|---|---|---|
| real | 97.1 % | 1.0000 | 0.3243 | 6 086 |
| **null** | **97.5 %** | **1.0000** | **0.2896** | **7 652** |

**The null beat the real arm on every axis.** A merged-FF "grain count" from a
scanning dataset at campaign thresholds carries no information about the sample.

Raising the peak thresholds until the merged list is sparse *does* restore
information (79.5 % vs 3.2 %; 3 741 vs 564 distinct) — but only by discarding
92 % of the spots, i.e. exactly the weak small-grain spots pf-HEDM exists to
capture.

> **The two sparsifiers are not equivalent.** Thresholding is **lossy** — every
> discarded spot is real. The scan gate is **information-preserving** — no spot is
> discarded, each is merely offered only to the voxels whose beam could have lit
> it. Collapsing the scans into one FF pattern deletes the `scannrobs` column that
> makes the search well-posed, so **merged-FF is structurally wrong for scanning
> data, not merely badly tuned.** It also costs 5.6× more core-hours.

For comparison, a genuine single-illumination far-field run of the same layer
passes cleanly: 786 solutions real, **0** null.

## 7.7 What the null does not cover

- It shares the real arm's **geometry, calibration and peak list**. A wrong ω
  sign or `positions.csv` convention mirrors *both* arms identically and the null
  will not see it. Those stay phase-1 halt conditions.
- It validates the **method on this layer**, not the specific banked numbers. If
  the arms were produced by different binaries or different `ScanPosTol`, the
  comparison is void — see DIAGNOSIS.
- A ceiling measured with a **wider gate than the run used** is an over-estimate,
  so a gate set from it is conservative rather than wrong. State which gate the
  ceiling was measured at.

## 7.8 Cost

Both arms are a re-index only — prep, peakfit and binning are untouched, and the
null's binning is regenerated from the shuffled CSVs in seconds. On the reference
campaign: **~95 min/arm at 30 cores** (169 voxels) and **~3.5 h/arm at 54 cores**
(361 voxels). The banked real arm can serve as one arm **only if** it used the
same binary and the same `ScanPosTol`.
