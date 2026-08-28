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

Measured on the NMC811 1-ID campaign, both arms re-run fresh with one binary:

| layer | scans / voxels | real: vox w/ sol | real best-comp median | **null max = ceiling** | real voxels at 1.0 | gate both arms ran at |
|---|---|---|---|---|---|---|
| sparse (13 scans, 169 vox, 936 k spots) | 13 / 169 | 143 | 0.9231 | **0.6957** | 22.4 % | 0.75 µm (matched) |
| dense (19 scans, 361 vox, 1.29 M spots) | 19 / 361 | 354 | 0.8943 | *0.8333* ⚠ | 26.3 % | 0.80 µm — **over-estimate, re-measurement pending** |

⚠ **Quote the gate the ceiling was measured at.** The sparse row was first
measured with both arms at a 0.80 µm gate (the `ScanPosTol` fallback, see
DIAGNOSIS) and gave 0.7083; re-running the null at the correct 0.75 brought it
to **0.6957**. A wider gate admits more chance matches, so a ceiling measured
too wide is an **over-estimate** and a gate set from it is conservative, not
wrong. The dense row is still the 0.80 figure and will move down.

No null voxel reached completeness 1.0 in either. **The separation is real —
the PF per-voxel path passes its null at campaign thresholds, with no change to
the recipe.** That is the headline, and it is what the beam-position gate buys
(§7.6).

But the shipped acceptance gate is **`MinMatchesToAcceptFrac 0.500000`**, which
is *below both ceilings*, so the band `[0.50, ceiling]` is accepted and
reachable by chance:

| gate | sparse (matched, 0.75) real / null | dense (0.80 ⚠) real / null |
|---|---|---|
| **0.50 (shipped)** | 143 / 73 → **0.51** | 354 / 148 → **0.42** |
| 0.60 | 121 / 43 → 0.36 | 308 / 86 → 0.28 |
| 0.7083 | 108 / **0** → **0.000** | 271 / 42 → 0.155 |
| 0.75 | — | 253 / 14 → 0.055 |
| 0.80 | — | 228 / 1 → 0.004 |

**Voxels whose winner sits at or below the ceiling: 24.5 % (sparse), 39.5 %
(dense).** They may still be real; this test cannot say so.

> ⚠ **The ceiling is density-dependent — measure it per layer.** It rose from
> 0.6957 to 0.8333 with 1.4× the spot count. Porting the sparse number to the
> dense layer still admits 42 null voxels. **Never quote one layer's ceiling for
> another**, and never assume the denser layers are the safer ones — they are the
> more contaminated ones.

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
