# FF-HEDM Lab Notebook — `bt_1id_jul26` / `Au3_cubes_ff_000008`

**Companion to `FF_HEDM_Handbook.md`.** The handbook says what to do; this records what
was actually found, how it was measured, and what turned out to be wrong. They are kept
apart on purpose: the handbook has to stay short enough to follow, and this has to stay
honest enough to stop a refuted idea coming back.

Dataset throughout: APS 1-ID, GE5 (ADEPT) 2048², 200 µm pixels, 95.0 keV, Au cubes,
`Au3_cubes_ff_000008`, 1440 used frames, ω 180 → −179.75 at −0.25°/frame. `§n` without a
qualifier means a section of *this* file; handbook sections are named as such.

**Read §4 before re-opening any question here** — three attractive claims are recorded
there as retracted, with the measurement that killed each one.

---

## 1. What this campaign established

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | `midas_peakfit` batched-LM grouping was timing-dependent | FIXED 0.4.6 | §2a |
| 2 | `midas_transforms` `calc_radius` used CUDA float atomics | FIXED 0.8.2 | §2b |
| 3 | `midas_process_grains` averaged GrainRadius over the wrong SpotID space (5.5× low) | FIXED 0.6.1 | §3a |
| 4 | `midas_fit_grain` returned seed positions in fp32 — a *scaling* bug, not precision | FIXED 0.6.0 | §3c |
| 5 | That failure was silent — nothing downstream could tell | FIXED 0.6.0 / pipeline 0.7.0 | §3c |
| 6 | The specimen is one grain plus its Σ3 annealing twin | VERIFIED vs raw frames | §3d |
| 7 | C `ProcessGrains` over-segments those 2 orientations into 6 grains | ESTABLISHED | §3d |
| 8 | The recon is **COMPLETE** — 2 grains explain every credible reflection; the "91 % unindexed" spot list is ~98 % noise, padding and haloes of those same 2 grains | RESOLVED | §4d |
| 9 | Spot list degrades after detection; brightness (SNR) is the discriminator, **not** ω multiplicity — 45.9 % of credible spots are single-frame and 8 indexed ones reach SNR 2511. `RingThresh 10` is close to what both criteria recommend | RESOLVED — `MinPeakSNR` implemented 2026-08-01 (§6d). Needs `midas-zipper >= 0.1.5` or the key is dropped when the params are zipped | §6b, §6c, §6d |
| 10 | "Background varies ~20σ around a ring band" — RETRACTED, an artefact of a hand-rolled band mask (13.4 % pixel overlap with the real band) | RETRACTED | §6a |

Every one of defects 1–5 produced **wrong numbers and no error message**. That is the
through-line of this campaign, and the reason the handbook now insists on cross-checking
against the C reference and against the raw frames: on this pipeline, nothing tells you
the answer is wrong.

---

## 2. Reproducibility — two nondeterminism defects

### 2a. Fixed: `midas_peakfit` — batched-LM grouping was timing-dependent

Symptom: three runs of the same FF pipeline, same `Parameters.txt`, same host, gave three
different `Grains.csv`. The first diverging artifact was `Temp/AllPeaks_PS.bin`, while
`Temp/AllPeaks_PX.bin` (the raw connected-pixel sets) was byte-identical — so thresholding
and connected components were fine and the **peak fit** was not. Isolated: two runs of
`midas_peakfit.orchestrator.run` alone on one fixed zarr moved **1167 of 8599 peaks**, and
not in the last bit — one intensity column differed by 344 counts.

Cause. `lm_solve` is mathematically independent per region but not numerically: batching
`B` regions into one call selects a different cuBLAS/MAGMA batched-GEMM and Cholesky
kernel, so a region's fit depends on *which regions it was solved alongside*.
`RegionPool` made that grouping vary run to run in two ways — it derived the batch quantum
from **live** free VRAM and host `MemAvailable` and re-keyed the cache on the live bucket
count, and the consumer pulled *every* queued entry rather than one quantum, so chunk
boundaries landed wherever the consumer thread happened to be scheduled. On top of that
`lm.py` set `torch.backends.cuda.matmul.allow_tf32 = True` **at import, process-wide**;
that flag's design note only covers the fp64 path, but with the FF default
`--dtype float32` it also caught the plain fp32 `Jt @ J`, assembling the normal equations
at a 10-bit mantissa.

Fixed in `midas_peakfit/pool.py` (quantum decided once per `(n_peaks, m_pixels)`, quantized
to a power of two, consumer pulls exactly one quantum including at drain; host residency
moved to a separate global backstop that logs loudly if it ever fires) and
`midas_peakfit/lm.py` + `lm_generic.py` (TF32 scoped to the fp64 matmul that asked for it,
never a global import side effect). Locked by
`midas_peakfit/tests/test_pool_determinism.py` — 6 of its 7 CPU tests fail on the pre-fix
source. No throughput cost: 34.1 frames/s after vs 30.8 before.

### 2b. Fixed: `midas_transforms` — `calc_radius` used floating-point atomics

With peakfit deterministic, one divergence remained. `Result_StartNr_*.csv` (merge output)
was bit-identical while `Radius_StartNr_*.csv` was not, and exactly three columns moved:

| column | max relative difference between two runs |
|---|---|
| `PowderIntensity` | 2.23e-07 |
| `GrainVolume` | 3.75e-07 |
| `GrainRadius` | 4.49e-07 |

That is float32 epsilon, and it is the whole chain: `powder_int` → `GrainVolume` →
`GrainRadius`. `radius/core.py` summed per-ring intensity with
`powder_int.scatter_add_(0, spot_match, ...)`, which on CUDA lowers to floating-point
`atomicAdd` — arbitrary summation order per launch. It surfaced in `Grains.csv` as
`GrainRadius` 20.775146 vs 20.775148 µm and nothing else.

Fixed by replacing it with a per-ring masked `sum` (`torch.sum` is a fixed-order tree
reduction and deterministic on every backend). The reduction is over the number of
**configured rings**, so this costs nothing. Locked by
`midas_transforms/tests/test_calc_radius_determinism.py`, which also checks the value
against an independent numpy reference so determinism can't be bought with a wrong answer.

**General rule for this codebase:** `scatter_add_`, `index_add_`, and `index_put_` with
`accumulate=True` on float CUDA tensors are all nondeterministic. If a scientific output
depends on one, it is not reproducible.

### 2c. Verified

Three independent full-pipeline runs on `Au3_cubes_ff_000008`, `rm -rf results/` between
each, on chutoro: **bit-identical at all 27 checkpointed artifacts**, `Grains.csv` md5
`0449046c4a1eaa698d447fa480f10671` all three times.

### 2d. Grain position is weakly determined — but far less than first reported

**CORRECTED 2026-07-31.** An earlier revision of this section reported cluster members
spanning **356/193/553 µm** and called the reported position "a tie-break, not a
measurement". Those numbers were measured with the **broken fp32 refiner** (§3c), which
was not refining position at all — so most of that spread was the bug, not the data.
Re-measured on the same seeds after the fix:

| candidate cluster | pre-fix (fp32) span X/Y/Z | **after the fix** |
|---|---|---|
| cluster 0 (8 members) | 245 / 333 / 534 µm | **47 / 71 / 238 µm** |
| cluster 1 (9 members) | 642 / 294 / 517 µm | **133 / 56 / 283 µm** |

So position determinacy improved 3–5× in X and Y. It is still not tight: the candidates
within a cluster disagree by ~50–130 µm in X/Y and ~240–280 µm in Z, and they all sit at
completeness 1.0000, so `Completeness` cannot separate them — with
`MarginRadius`/`MarginRadial`/`MarginEta` at 500 µm (2.5 px at this pixel size) a grain
can move a long way with every spot still inside the matching window. The remaining
discriminator is `DiffPos`, and its minimum is shallow.

**What this means in practice:** quote grain positions to ~100 µm on this dataset, not to
the six decimals `Grains.csv` prints. If you need better, tighten the matching margins
toward 1 px — do NOT touch `Rsample`/`Hbeam`, which are a search bound (hard rule 9).

### 2e. Which candidate becomes "the grain" — and the two modes disagree

**In `--mode spot_aware` (what `midas-pipeline --scan-mode ff` runs), the grain IS one
candidate.** `midas_process_grains/pipeline.py:416-417` picks
`rep_pos = argmin(ias[members])` — the member with the smallest **internal-angle**
residual, `OrientPosFit.bin` **column 24** — and then copies that one candidate's
`position`, `orient_mat`, `lattice`, `grain_radius` and `confidence` straight into the
output (lines 452-456, 524-528). There is no averaging of any kind. Confirmed numerically
on this run: `Grains.csv` ID 80 = (10.882587, 47.793549, −108.281662) and ID 185 =
(−100.816216, −8.876349, 51.627029) are exactly the argmin members, and are **not** the
cluster medians (65.44, 23.35, −120.27) and (−34.62, 5.92, −8.66).

**`--mode physics` does the opposite.** `v4_pipeline.py:723` reports
`np.median(positions[members])` with a rotation-mean orientation, and its comment argues
that a single representative gives a >20 % rate of grains whose stored OM fails to
re-predict its own spots. So the two code paths take opposite positions on the same
question. That is unresolved in this tree — do not assume the mode you ran did what the
other one's comment describes.

**The C reference settles which one is the reference.** `FF_HEDM/src/ProcessGrains.c`
picks `BestGrainPos` by minimum internal angle (lines 688-695) and then copies that
member's row verbatim — `FinalMatrix[kk][jj+1] = OPs[rown_l][jj]` (line 1041) — for
orientation, position and lattice. Only the strain is recomputed, from the cluster's
pooled spots. So **`spot_aware` matches the C reference and `physics` deviates from it.**
Verified by running the freshly-built C binary on the identical refinement output: the two
grains it shares with `spot_aware` agree to six decimals in X/Y/Z, lattice parameter,
DiffPos, DiffOme and DiffAngle. See Handbook §8a.

For this dataset the trade is measurable. Per cluster, member position sd is
(73, 106, 193) µm; the bootstrap SE of the median over 9 members is (31, 53, 99) µm. So
the median **would** be the more precise position estimator here — at the cost of
publishing an X/Y/Z that no single fit actually produced. `spot_aware`'s single-candidate
choice is self-consistent but inherits the full member scatter.

**Knock-on from the 0.5.6 refiner bug (Handbook §8a):** since the rep is chosen by `argmin(col 24)`,
and 0.5.6 wrote `col 22 = mean_angle, col 23 = mean_pos, col 24 = mean_ome` (0.5.7 writes
`pos, ome, angle`), a pre-0.5.7 tree selected the representative by **mean ω error** while
believing it was internal angle. That changes which candidate becomes the grain, not just
a printed label. One more reason not to run < 0.5.7.

### 2f. Why the answer used to jump between runs

Before the fixes, a single flipped peak renumbered the spot IDs, which changed cluster
membership and so changed which candidate won the `argmin` — which is why one run reported
(18.28, −5.46, 150.11) µm and another (10.89, 47.83, −108.04) µm for the same grain. Both
were members of the same candidate cluster.

Two separate defects fed that: the peak list was not reproducible (§2a, §2b), and the
fp32 refiner was not moving position at all so the candidates stayed scattered across the
indexer's coarse position grid (§3c). With both fixed the run is bit-reproducible and the
candidate spread is 3–5× tighter (§2d).

If you want the position tighter still, tighten `MarginRadius` / `MarginRadial` /
`MarginEta` from 500 µm (2.5 px) toward ~1 px so `Completeness` regains discriminating
power, and re-check the candidate spread by reading `Results/OrientPosFit.bin` (cols 11-13
are position; col 26 > 0 selects the alive candidates). Do **not** reach for
`Rsample`/`Hbeam` — shrinking the envelope narrows the spread only by clamping candidates
against the bound, replacing an honest ambiguity with a fabricated pile-up (hard rule 9).

---


---

## 3. Cross-check against the C reference — findings

The build-and-run recipe lives in **Handbook §13**; this is what running it turned up.

### 3a. What that comparison found (Au3_cubes_ff_000008, 2026-07-30)

Identical refinement input, C vs python `spot_aware`:

| | C `ProcessGrains` | python `spot_aware` |
|---|---|---|
| grains | **6** | **2** |
| shared grains' X/Y/Z, a, DiffPos/Ome/Angle | — | identical to 6 decimals |
| GrainRadius (grain 80) | 114.620659 µm | 20.775146 µm ← **was wrong** |

1. **The reduction rule agrees.** Both take the minimum-internal-angle member and copy its
   fit verbatim. Nothing to fix.
2. **The clustering does not — and C is the one that is wrong (§3d).** C walks the
   *shared-spot adjacency* in `ProcessKey.bin` and merges neighbours with misorientation
   < 0.4° (`FindInternalAngles`, `ProcessGrains.c:140`), `MinNrSpots` defaulting to 1.
   python Phase-1 clusters globally on misorientation at `MisoriTol` 0.5°, then runs a
   Pass-A spot-overlap merge. C keeps 6 where python keeps 2 — but those 6 are only **two
   orientations**, and python's 2 are the physical answer. Resolved in §3d.
3. **GrainRadius was a genuine python bug — now fixed.** `midas_process_grains` built its
   per-spot radius lookup from `Radius_*.csv`. That file and `ExtraInfo.bin` hold the same
   spots numbered 1..N but in **different orders** (`calc_radius` renumbers, then
   `bin_data` sorts by `(RingNumber, Omega, Eta)` and renumbers again), and every id
   downstream of the binner is in the ExtraInfo space. The join therefore averaged ~112
   arbitrary spots, so every grain came out near the *global* mean radius (~22 µm) instead
   of its own. Fixed to read `ExtraInfo.bin` col 3 keyed by col 4 — the same source the
   refiner uses. python now reports 114.620677 / 99.962755 µm against C's
   114.620659 / 99.962738. Locked by
   `midas_process_grains/tests/test_spot_radius_id_space.py`.

**Reported grain sizes from any run before this fix are too small — by 5.5× on this
dataset.** The error is not a constant factor; it is "your grain's radius was replaced by
the sample-wide average", so it compresses the whole size distribution toward the mean.

### 3b. If `IndexerOMP` is slow, your binned files are in the wrong format

**`IndexerOMP` is not slow.** On this dataset it indexes 189 seeds in **2.03 s** on 16
threads (~94 seeds/s), comparable to the unified `midas_indexer` at 3 s, and finds the
same 20 candidates. Full legacy chain end to end: index 2.03 s → `FitPosOrStrainsOMP`
0.29 s → `ProcessGrains` 0.023 s.

If you see it take minutes, the cause is Handbook §13b — the pipeline's binned files are in the
PF/unified layout and legacy FF C reads a narrower one. **Three widths differ, and
converting only one of them leaves the indexer reading garbage:**

| file | PF / unified (what the pipeline writes) | legacy FF (what `IndexerOMP` reads) |
|---|---|---|
| `Spots.bin` | `(N, 10)` float64 | `(N, 9)` float64 — col 9 = `ScanNr` |
| `nData.bin` | `(B, 2)` **int64** (count, offset) | `(B, 2)` **int32** |
| `Data.bin` | `(T, 2)` **int64** (rowno, scanno) | `(T,)` **int32** rowno |

`nData.bin` is the one that bites. `IndexerOMP.c:122` does `nspots = ndata[Pos*2]` on an
`int *`; against an int64 array a bin lookup lands on the wrong bin and frequently reads
an **offset as a count** — up to 220,925 here instead of ≤24. The inner loop then scans
~10⁴× too many rows. That is both the slowdown *and* the reason nothing matches, from one
cause. Diagnose it by checking the file against the bin count:

```python
total_bins = n_ring_bins * ceil(360/EtaBinSize) * ceil(360/OmeBinSize)
nd = np.fromfile("nData.bin", dtype=np.int64).reshape(-1, 2)
assert nd.shape[0] == total_bins          # wrong dtype if this fails
assert nd[:, 0].sum() == n_data_entries   # counts must sum to Data.bin's length
```

Read at the correct width the numbers are self-consistent (counts ≤ 24 summing to 220,925,
offsets non-decreasing); at the wrong width they are absurd (counts up to 220,925 summing
to 9.4e12, all offsets 0). Converter: `utils/pf_to_ff_bins.py` — it asserts all three invariants before
writing.

**Corrected 2026-07-30.** An earlier revision of this section claimed indexing cost scaled
with `Rsample`/`Hbeam` and tabulated 16 s at 200 µm vs >420 s at 2000 µm. Those timings
were measured on the malformed `nData.bin` and are **void** — the apparent envelope
sensitivity was the corrupt bin lookup, not the search space. With correct input the
indexer takes 2.03 s at `Rsample 2000`. Do not use runtime as an argument for touching the
envelope; see hard rule 9.

**Resolved:** legacy `IndexerOMP` indexing 0/189 had the same single cause. With correctly
converted binaries it indexes 20/189 — the same 20 the unified indexer finds.

### 3c. The refiner returned its input in float32 — a SCALING bug, not precision

Same seeds in (the C `IndexerOMP`'s `IndexBest.bin`), three refiners out:

| refiner | \|pos\| mean | DiffPos med | median \|Δpos\| vs C | moved from seed |
|---|---|---|---|---|
| C `FitPosOrStrainsOMP` | 52.5 µm | 193.89 | — | 158.3 µm |
| py **float64** (cpu & cuda) | 71.0 / 69.5 µm | 199.1 | **13.4 µm** | 149.7 µm |
| py **float32** (cpu & cuda) | 227.1 µm | 231.9 | 158.2 µm | **0.0 — 20/20** |

In float32 the refiner did not refine position **at all** — it emitted the
seed. cpu and cuda agree to 3 s.f. within each dtype.

**The cause is parameter scaling, not arithmetic.** The optimizer carries
position as `pos_scaled = pos / pos_scale`, and the shipped `pos_scale = 100`
left the gradient blocks wildly unbalanced. Measured on the synthetic fixture:

| `pos_scale` | \|g\|position | \|g\|euler | ratio | fp32 error vs truth |
|---|---|---|---|---|
| **1e2 (shipped)** | 95.8 | 1.47e5 | **1537** | **154.27 µm** |
| 1e3 | 958 | 1.47e5 | 154 | 0.75 µm |
| 1e4 | 9581 | 1.47e5 | 15.4 | 0.013 µm |
| **1e5** | 9.58e4 | 1.47e5 | **1.5** | **0.004 µm** |

L-BFGS applies **one step length to the whole concatenated vector**, so
position advanced ~1500× less per step than orientation. fp64 has the mantissa
headroom to keep resolving that; fp32, whose gradient carries ~1e-4 relative
rounding noise (~600× eps), does not — the position component of each step
lands under the noise. After the first orientation-dominated step the
strong-Wolfe line search finds no further descent and returns t = 0 forever:

```
f64: 177.35 → 49.08 → 5.02 → ... → 2.27e-08   (descends; final error 0.00 µm)
f32: 177.36 → 70.25 → 70.25 ×9                (frozen EXACTLY; error 154.27 µm)
```

Everything else is clean, which is how the scaling was isolated: the fp32
gradient **direction is exact** (`cos(g64,g32) = +1.00000000` on every block),
fp32 **resolves** loss decreases down to t = 1e-8 and position steps of
0.0001 µm, and **LM fails too** — so it is not specific to L-BFGS. The error
is also *not* Lsd-driven (tested at 10 / 100 / 1000 mm: ~1e-4 throughout).

**Fix.** `midas_fit_grain.refine_block` now derives the scale from the entry
gradient — `s = |g_other| / |g_pos_µm|`, the value that makes the ratio 1 —
instead of a fixed 100. It is a pure reparameterization, so the sample-cylinder
clamp (which divides the µm bounds by `pos_scale`) stays consistent. Result,
over three seed offsets:

| seed offset | f64 old | f64 auto | f32 old | f32 auto |
|---|---|---|---|---|
| (90, −60, 110) | 0.0031 | **0.0002** | 154.27 | **0.0032** |
| (−200, 150, −80) | 0.0261 | **0.0002** | 67.62 | **0.0041** |
| (15, −5, 25) | 0.1054 | **0.0002** | 29.58 | **0.0049** |

fp64 improves too (15–500×).

**Confirmed on the real dataset** (189 C-indexer seeds, vs the C reference):

| config | median \|Δposition\| vs C | DiffPos median |
|---|---|---|
| fp32, fixed `pos_scale`=100 | **158.24 µm** | 231.9 |
| fp64, fixed `pos_scale`=100 | 13.38 µm | 199.07 |
| **fp32, auto** | **13.65 µm** | **196.94** |
| **fp64, auto** | 13.96 µm | 199.11 |
| *(C reference itself)* | — | *193.89* |

End to end through the pipeline, the two reported grains moved to where they
should be — near the rotation centre, and onto the C chain's answer:

| grain | before the fix | after | full C chain |
|---|---|---|---|
| 80 / 1000 | (10.9, 47.8, **−108.3**) | **(2.7, 2.0, −5.3)** | (−4.6, 4.8, −20.8) |
| 185 / 1177 | (−100.8, −8.9, 51.6) | **(−10.3, −22.5, −3.5)** | (−7.3, −27.8, 4.0) |

Distance from the C reference fell 99 → 17 µm and 107 → 10 µm, and DiffPos to
190.1 / 182.2 against C's 188.7 / 181.0. Refinement also got *faster*
(51.9 s → 16.4 s at fp64) because balanced blocks converge sooner.

`RefinementConfig.dtype` still defaults to **float64** — cheap at this scale and
the conservative choice — but fp32 is now a supported trade for throughput on
large runs, not a correctness risk.

**The silence was the second bug**, and it is fixed independently:
`refine_block` reports `max_position_move_um` / `median_position_move_um` /
`n_unmoved_position`, `midas_fit_grain/driver.py` emits `UNREFINED-POSITIONS: …`
when no grain moved more than **px/1000**, and
`midas_pipeline/stages/refinement.py` re-surfaces that into the run log.
(px/1000, not one pixel — a healthy fp64 fit moved only 0.77 px here, so a
one-pixel threshold would flag good fits; fp32 moved 2.5e-06 px.)

**If you see that warning, the grain positions in that run are indexer seeds,
not fits.** Do not quote them.

NOTE the PF scanning path (`refine.py::refine_grain`, used by `scan_driver`)
is deliberately left on the fixed scale: `position_mode="fixed"` locks the
voxel to the scan grid there, so position is not a free parameter, and PF
carries C-parity gates. Apply the same equilibration if you enable
`position_mode="voxel_bounded"`.

### 3d. RESOLVED — C's 6 grains are 2 orientations, and one is a Σ3 twin

C reports 6, python `spot_aware` reports 2, from identical refinement. Measuring pairwise
misorientation settles it: the 6 are **two families**, not six orientations.

```
           1000      899     1014      918     1177     1176
 1000         -    0.638    0.079    0.203   59.973   59.850
  899     0.638        -    0.570    0.448   59.918   59.959
 1014     0.079    0.570        -    0.127   59.995   59.882
  918     0.203    0.448    0.127        -   59.977   59.899
 1177    59.973   59.918   59.995   59.977        -    0.620
 1176    59.850   59.959   59.882   59.899    0.620        -
```

**C is over-segmenting.** 1000 and 1014 differ by **0.079°** and share **70 %** of their
spots (Jaccard 0.697); 1000/1014/918 are mutually < 0.21°, comfortably inside C's own 0.4°
merge threshold. They stayed separate only because `FindInternalAngles` requires
misorientation < 0.4° **AND** shared-spot adjacency in `ProcessKey.bin` — near-identical
orientations that miss that adjacency link never get compared. The duplicates are also the
worse fits: the two python-matching grains (1000, 1177) have the lowest DiffPos
(188.7, 181.0) and Conf 1.000, while the extras run 206–333 with Conf 0.875–0.983.

python's two land on C's best members at **0.031°** and **0.018°**, and no C grain is more
than 1° from a python counterpart. **python `spot_aware` gives the right answer here.**

**The cross-family ~60° is a Σ3 annealing twin**, and it is about ⟨111⟩ — every one of the
eight cross pairs, several within 0.01–0.11° of the ideal axis. So the specimen is one
grain plus its twin.

#### Verified against the raw frames

Orientation algebra alone would not settle this — a phase can score highly on reflections
it *borrows* from another (exactly what killed a Zn/Cu epitaxy claim in the Fuller Laue
campaign). The honest test is whether the twin has spots that are **its own**:

```
parent 1000: 112 spots     twin 1177: 116 spots     shared: 43
                                    -> 73 spots belong to the twin alone
```

Those 73 were checked directly in the dark-subtracted frames, at the predicted pixel:

| spot set | SNR > 5 | median SNR | min |
|---|---|---|---|
| **twin-only** | **30/30** | **1718** | 351 |
| parent-only | 30/30 | 1634 | 135 |
| shared | 30/30 | 1736 | 820 |

The pixel convention was **not assumed** — it was calibrated on spots already known to be
real, and came out unambiguous: `img[z, y]` with `row = BCZ + z/px`, `col = BCY − y/px`
gives median SNR **1469** where all seven alternative axis/sign combinations give ~1.0.
Residual registration is ~1 px (median +0.77 px in column, 1.03 px radial), because this
check ignores the distortion terms p0–p14 and the detector tilt; the ±6 px signal box
absorbs that.

Figure + generators: `twin_vs_frames.png`, `twin_vs_frames.py`, `twin_frames_fig.py`
(beamtime `analysis/au3_cubes_ff_000008/`).

**Method worth reusing:** when two codes disagree on grain count, ask (a) pairwise
misorientation — are these even different orientations? (b) spot-set Jaccard — are they
explaining the same observations? and (c) for any orientation you want to believe, does it
have *exclusive* spots with real intensity? (a)+(b) separate over-segmentation from real
grains; (c) stops you believing a phase that only ever borrows peaks.

---

## 4. Open questions, and claims that were retracted

### 4a. What is and is not still open

The `Au3_cubes_ff_000008` run completes all 13 stages, is bit-reproducible, and yields
**2 grains** — a parent and its Σ3 twin, both with independent evidence in the raw frames
(§3d). As of 2026-07-31 it is a **complete** reconstruction: those two explain every
credible reflection, and the rest of the spot list is noise (§4d).

**Closed:**

- ~~"2 grains explain only 185 of 2076 spots (8.9 %)"~~ — the denominator was ~98 %
  non-diffraction. Do not quote the 8.9 % or "91 % unindexed" figures; see §4d.
- ~~indexing-parameter work on `Completeness`/`MinNrSpots`/`OverAllRingToIndex`~~ — there
  is nothing left to index. Tuning these would be fitting the indexer to noise.

**Still open:**

- **Grain positions are good to ~100 µm, not better** (§2d). Quote them accordingly.
- 899 of 1176 candidate pairs sit at 0.45–0.64°, just outside C's 0.4° merge threshold.
  Whether that is same-grain spread or real substructure is **not resolved**.
- **`RingThresh` is set too low for this specimen** — the peak finder is working deep in
  the noise and over-segmenting strong reflections (§4d). Peak-finder side, not indexer.
- `Rsample`/`Hbeam` being generous is *correct* and must stay that way (handbook hard
  rule 9). It is not a pending item.

Current output, for reference:

```
 ID          X          Y          Z    DiffPos   GrainRadius
 80    2.685097   1.982347  -5.349646   190.089      114.6207
185  -10.289690 -22.492229  -3.493707   182.207       99.9628
```

Both grains sit within ~25 µm of the rotation centre, and DiffPos (190.1 / 182.2 µm) is
within ~1 % of the C reference's own residual (188.7 / 181.0 µm).

### 4d. RESOLVED — the "91 % unindexed" figure is noise; the reconstruction is complete

*Measured 2026-07-31. Scripts:
`~/Desktop/analysis/bt_1id_jul26_calib/spot_{noise_audit,noise_null,snr_all,frames_fixed}.py`
+ `halo_check.py` + `residual44_check.py`; outputs under `<run>/spot_noise_audit/`.*

**Bottom line: `Au3_cubes_ff_000008` is a COMPLETE reconstruction.** The specimen is one
grain and its Σ3 twin (§3d), and those two explain every credible reflection in the data.
The "2 grains explain only 8.9 % of the spots" statement that sat in this document as the
headline open item was an artifact of a spot list that is ~98 % non-diffraction.

The spot list has 2076 rows (not 2077). Classifying every row by its own-frame SNR at the
recorded detector pixel — measured on the raw frames, one read per frame, all 2076 rows:

| class | n | % | what it is |
|---|---|---|---|
| indexed | 185 | 8.9 % | the 2 grains; own-frame SNR median **1989**, 100 % > 5 |
| unindexed, SNR > 5, **within 20 px of an indexed spot** | 333 | 16.0 % | over-segmented tails/haloes of the SAME two grains |
| unindexed, SNR ≤ 5 | 1309 | 63.1 % | noise excursions the peak finder crossed threshold on |
| ring-0 padding | 205 | 9.9 % | zero-intensity rows; not spots at all |
| unindexed, SNR > 5, isolated (≥ 50 px) | 44 | 2.1 % | **also noise — see below** |

88.3 % of the SNR > 5 unindexed spots sit within 20 px of an indexed spot at |Δω| < 2°,
and the distance distribution is cleanly bimodal (identical counts at the 20 px and 50 px
cuts), so the halo classification is not a threshold artifact.

**The residual 44 are a noise tail, not small grains.** Raising the SNR cut settles it —
a real population thins slowly and keeps a size mode; a noise tail piled against the cut
collapses:

| SNR cut | n | median implied radius | median SNR | NImgs==1 |
|---|---|---|---|---|
| > 5 | 44 | 7.0 µm | 6.4 | 64 % |
| > 8 | 10 | 7.2 µm | 11.1 | 90 % |
| > 10 | 6 | 7.2 µm | 11.6 | 100 % |
| > 15 | **0** | — | — | — |

Three things damn them:

1. **They vanish by SNR 15.** The indexed set holds all 185 at every cut through SNR 100,
   and 166 of 185 even at SNR 1000.
2. **The implied radius does not move** — 7.0 → 7.2 µm while the median SNR nearly doubles
   (6.4 → 11.6). For real grains, keeping brighter spots must select *bigger* grains. A
   size that ignores the intensity cut is a threshold artifact, and it also means the
   fitted `IntegratedIntensity` behind `GrainRadius` is decoupled from the raw signal —
   itself a junk signature.
3. **The implied size distribution is absurdly narrow**: IQR 5.5–7.6 µm, against 89–124 µm
   for the indexed grains. Real grain populations are not that monodisperse; a threshold
   cut on a noise distribution is.

44 spots scattered over 5 rings could not form a coherent grain in any case, and the
indexer already had all 2076 spots and returned only the two.

Discriminators, and what each is worth:

- **Own-frame SNR** is the one that carries the result. Median 1989 (indexed) vs 2.7
  (unindexed single-frame).
- **ω-localization** (same pixel ±90° away) confirms the padding rows are inert
  (ratio 1.0) but does **not** separate real from noise: unindexed spots score ~2.2×,
  which is exactly what *selection bias* produces — the pixel was chosen because it was a
  local maximum on that frame. Do not read that 2.2× as evidence of diffraction.
- **NImgs**: 1389 of 1686 ring-assigned unindexed spots are single-frame vs 8 of 185
  indexed. **Suggestive only — single-frame does NOT imply noise** (§6c): those 8
  indexed single-frame spots run to SNR 2511, and 45.9 % of all credible spots are
  single-frame. Do not use ω multiplicity as a quality criterion.
- **Hot pixels**: 26.8 % of unindexed spots fall in repeated 2-px detector cells (worst
  cell fires 19× at different ω) vs **0 %** of indexed.
- **Friedel pairing — REJECTED as evidence.** Unindexed spots pair at 77.8 % against a
  25 % shuffled null, which looks like strong support for them being real. It is not
  usable: mirrored detector artifacts and bad regions are symmetric about the beam centre,
  so they pair systematically, and the shuffle null does not model that. Do not resurrect
  this argument without a detector-structure-aware null.

**Implied grain size.** `calc_radius` converts a spot's intensity into `GrainRadius` by
ratio against the ring powder intensity (meaningless for a noise event — it answers "if
real, how big?", not "is it real?"). Indexed spots imply **96.8 µm** median radius. The 44
isolated credible spots imply **7.0 µm** (p25 5.5, p75 7.6) — ~14× smaller in radius,
~2600× smaller in volume — at median SNR 6, i.e. right at the detection limit.

**What this changes.** There is no missing-grain problem and no indexing-parameter problem.
`Completeness`, `MinNrSpots` and `OverAllRingToIndex` do **not** need revisiting for this
dataset — chasing them would be tuning the indexer to fit noise. Two real issues remain,
and both are peak-finder-side, not indexer-side:

- **Over-segmentation around strong reflections** inflates the spot count by ~16 % (333
  spots). That is a `RingThresh` / peak-splitting question.
- **The threshold may be too low for this specimen** — 1309 sub-SNR-5 rows plus 205
  zero-intensity padding rows were admitted. But §6b resolves this: detection at
  `RingThresh 10` is reasonably clean and the population degrades at **merge/fit**. The
  threshold is close to what both criteria recommend; the open item is `merge_overlaps`.

**Remaining caveat.** The 20 px halo criterion is physically reasonable (the saturated
blobs are ~10–15 px across) and the bimodal distance distribution supports it, but it was
not independently validated. This does not affect the conclusion: the 44 spots *outside*
that radius were tested on their own and fail independently, so the completeness verdict
does not rest on where the halo cut is drawn.

### 4b. Could not verify — do not upgrade these

- Whether `DetZ`'s +181 mm offset is stable across the beamtime. Measured at one distance
  only.
- Whether the 95-vs-96 keV strain gap (19.4 vs 72.7 µε) is a genuine energy discriminator
  or partly an artifact of the distortion harmonics re-fitting. The distortion-frozen
  control was not run.
- `ImTransOpt 0` is a **convention**, not a measurement: the calibration was fitted on the
  untransformed `exchange/data` array so the recon must match it, but a self-consistent
  calibration + recon pair can still be globally *mirrored*. The absolute handedness is
  unestablished until something external pins it.

### 4c. RETRACTED — do not resurrect

Earlier revisions of the handbook asserted all three of these. Each was an artifact of a
defect that has since been fixed. If one of them starts to look true again, the first
suspect is a regression in the named fix, not a rediscovery.

| Retracted claim | What actually caused it | Fixed in |
|---|---|---|
| "Only one of the two grains reproduces across runs" | the two nondeterminism defects — peak list was not reproducible | §2a, §2b |
| "The reported position is a tie-break over a ~500 µm span" | the fp32 refiner returned seed positions, so the spread was the indexer's coarse grid | §3c |
| "Indexing cost scales with `Rsample`/`Hbeam`" | a malformed `nData.bin` — int64 offsets read as int32 counts | §3b |

The third is the most dangerous, because it invites exactly the change that handbook hard
rule 9 forbids: shrinking `Rsample`/`Hbeam` to the real sample size, which plants grains on
the faces of the bounding box. Indexing was **2.03 s** for 189 seeds once the file format
was right (§3b).

---

## 5. Measurement ledger — what was verified, and how

Kept here rather than in the handbook so that each entry keeps its provenance.
Handbook §11 is the one-paragraph summary of this table.

| What | How it was established | Handbook § |
|---|---|---|
| ω sign (`aero` is CW, negate every ω) | all **7297** `bt_1id_jul26` FF par rows; rule shared with NF | 2 |
| Throwaway first frame | ~1.5 % baseline offset measured in three separate files | 3e |
| `SkipFrame` is a **consumer-side** skip | read the three `midas_peakfit` call sites; the zipper's first-file exemption and back-dated `start_omega` are correct as shipped; changing it produces the double-skip (1439-frame) failure. Locked by `midas_zipper/tests/test_skipframe.py` | 3 |
| `DetZ` − `Lsd` = **+181 mm** | assignment-free ring-ratio measurement | 5b |
| Energy **95.0 keV** | three instrument records + beamline confirmation | 4 |
| CeO₂ 0/180 repeatability | `Lsd`/`BC` repeat to 0.01 % / 0.01 px across an independent 180° repeat | 5f |
| Shared env missing `matplotlib`, `scikit-image` | import failure on chutoro | 1 |
| `darkLoc` vs `darkDataset` | the zipper reads `config['darkLoc']` (`ff_zip.py:334`) and writes an all-zero dark when unset. Confirmed by reading `zarr["exchange/dark"]` before (max 0) and after (mean **1870.55**) | 3d |
| `midas-fit-grain` 0.5.6 rotated the residual columns; 0.5.7 fixes it | the same grain's ω residual went from **223.87°** to **0.054°** | 8a |
| `RingThresh` sensitivity | measured on this dataset | 6b |
| Pipeline is bit-reproducible | 3 runs identical at all 27 artifacts; `Grains.csv` md5 `0449046c4a1eaa698d447fa480f10671` | 12 |
| `GrainRadius` agrees with C | 114.620677 / 99.962755 µm vs C's 114.620659 / 99.962738 | 13 |
| Refined position agrees with C | 13.4–14.0 µm median over 189 identical C-indexer seeds | 13 |

---

## 6. `RingThresh` calculator (2026-08-01) — and a premise of mine that was wrong

Built `midas_peakfit.background` + `midas_peakfit.ring_thresh` (CLI
`midas-ring-thresh`); see Handbook §6b. Two findings matter more than the code.

### 6a. RETRACTED — "the background varies by ~20 sigma around a ring band"

I measured per-azimuthal-sector background inside the ring bands and reported a spread of
**90–139 counts against a noise sigma of ~5**, and used it to argue that no single absolute
`RingThresh` can serve a band. **That was wrong.** It came from a band mask I built myself
from a plain radius-from-beam-centre, rather than the distortion-corrected `Rt` the
production path uses after `apply_image_transformations` + `transpose_square`.

Head to head on `Au3_cubes_ff_000008`:

| | production band (`Rt`) | naive band |
|---|---|---|
| pixel overlap with production band | — | **13.4 %** |
| background spread / noise sigma | **0.4** | 165–199 counts |

The naive mask sits off the true ring and straddles its steep radial edge, which
manufactures the apparent azimuthal variation. Through the real geometry **the band is
flat** and an absolute per-ring threshold is perfectly adequate on this dataset.

This is the *same* error, third time: hand-rolling the band mask also produced blob counts
67× off the production pipeline, and made me briefly doubt Handbook §6b's (correct) table.
**Never rebuild the band mask — call `compute_rt_eta` / `compute_good_coords`.**

Local background subtraction was still implemented (`BgSubtract 1`, default **0**, so the
legacy path is bit-identical and is locked by a test). It is justified where a background
genuinely varies; it is *not* justified here, and this dataset must not be cited as the
motivation.

### 6b. RESOLVED — the noise is mostly manufactured DOWNSTREAM of detection

§4d found 1309 of 2076 recorded spots below SNR 5 at `RingThresh 10`, while the calculator
scored per-frame blobs at that same threshold as clean. Those differed in *two* ways at
once — the objects (per-frame blobs at detection vs merged/fitted spots after
`merge_overlaps`) and the SNR estimator — so both were varied independently, on the same
30 frames (`utils/spot_audit/stage_vs_estimator.py`):

| | E1 in-band annulus (calculator) | E2 81×81 box on raw (audit) |
|---|---|---|
| **stage A — detected blobs** | 90.8 % | **53.4 %** |
| **stage B — merged/fitted spots** | 28.8 % | 28.8 % |

*(fraction of objects with SNR > 5, n = 131 blobs / 59 spots)*

**Both effects are real, and they are separable.**

1. **The stage effect dominates and is the answer.** Under the *same* estimator E2, the
   clean fraction roughly halves from detection to the recorded spot list
   (53.4 % → 28.8 %). Detection is not where most of the noise enters —
   `merge_overlaps` + peak fitting is. At stage B the two estimators agree *exactly*
   (28.8 % / 28.8 %), which is a strong check that this is a property of the objects and
   not of how they are scored.
2. **My criterion A was over-optimistic, and is now fixed.** E1 restricted its background
   annulus to in-band pixels — a strip that carries spot wings and elevated ring
   background — so it understated the noise floor (90.8 % vs 53.4 % on identical blobs).
   `blob_snr` now measures on a corrected frame built WITHOUT the band mask, so the
   background is real everywhere. After the fix the reference dataset reports 20 % clean at
   threshold 5 where the old form said 52 %, and the recommendation becomes per-ring:
   `10 / 20 / 20 / 10 / 10`.

**Consequence for §4d.** "`RingThresh` is too low" is *not* supported as the main cause.
The threshold the run used is close to what both criteria independently recommend. The
open item moves to the stage that actually degrades the population:

- **`merge_overlaps` and the peak fit are where the spot list goes bad.** A blob that is
  clean on its own frame becomes a recorded spot at SNR ≤ 5. Candidate mechanisms, none
  yet tested: merging chains together weak single-frame detections that should not join;
  the fit placing a centroid off the intensity when it fails to converge (58.4 % of
  unindexed spots have `ReturnCode != -1` vs 9.2 % of indexed, §4d); or sub-threshold
  wings being pulled into a merged spot's footprint.

That is the next thing to measure, and it is a `merge_overlaps`/fitting question, not a
threshold one.

### 6c. RESOLVED — the mechanism is *un-merged single-frame detections*, not merging

§6b established the spot list degrades between detection and the recorded spots. Testing
the three candidate mechanisms (`utils/spot_audit/merge_fit_degradation.py`, 1871
ring-assigned spots; frame tests on 30 frames):

**The dominant one — ω multiplicity.** Cleanliness tracks `NImgs` monotonically and
steeply:

| NImgs | n | median SNR | frac SNR>5 |
|---|---|---|---|
| 1 | 1397 | 2.7 | **18.5 %** |
| 2 | 184 | 3.9 | 39.7 % |
| 3–5 | 105 | 24.5 | 60.0 % |
| ≥ 6 | 185 | 1972.1 | **90.8 %** |

This **inverts the hypothesis in §6b**. `merge_overlaps` is not manufacturing noise by
chaining weak detections together — merging is acting as a *filter that works*, and the
dirt is the 1397 detections (75 % of the list) that never merged with anything. A real
Bragg reflection sweeps through the Bragg condition over finite ω and is seen on several
frames; a noise excursion is seen once. Spots that merge across ≥6 frames are 90.8 % clean.

**Secondary — fit divergence.** 54.1 % of spots carry `ReturnCode != -1`. They are dirtier
(24.2 % clean vs 36.9 %), and 52.5 % of recorded spots sit **more than 5 px from any
detected blob** on their own frame — of those, 87.1 % diverged and only 19.4 % are clean.
So a failed fit really does deposit the centroid where there is no intensity. But it is not
the whole story: even *converged* spots are only 36.9 % clean.

**RETRACTED — do NOT filter on `NImgs`.** An earlier revision of this section recommended
`NImgs >= 2` as the spot-list filter (474 kept, 64.1 % clean, 177/185 indexed retained).
That recommendation is **wrong and must not be implemented.**

ω width is set by mosaicity, beam divergence and energy bandwidth — **not** by grain size.
A small or undeformed grain can satisfy the Bragg condition inside a single 0.25° frame,
so single-frame is a perfectly ordinary property of real signal. Measured here:

- **8 of the 185 indexed spots are single-frame, at SNR 424, 1620, 1686, 1788, 1990, 2071,
  2188, 2511** (median IMax 10719). These are unambiguous Bragg spots that `NImgs >= 2`
  deletes.
- **258 of the 562 credible (SNR > 5) spots — 45.9 % — are single-frame**, up to SNR 2752.
- The loss is concentrated where it matters most: `NImgs >= 2` discards **85.6 % of the
  SNR 5–10 spots**, 56.4 % of SNR 10–100, and only 10.3 % of SNR > 100. It preferentially
  destroys the *weak but real* population, which is exactly the small-grain signal.

`NImgs` only appeared to work because this dataset's real spots come from **two large
mosaic grains** that happen to span 12+ frames each. It was acting as a proxy for "belongs
to one of the two big grains" — circular reasoning when the goal is to find *other* grains,
and not generalisable to a fine-grained, annealed or undeformed sample, nor to a coarser ω
step. **A two-grain dataset cannot establish an ω-multiplicity rule.**

**Use SNR instead — it strictly dominates.**

| filter | kept | indexed kept |
|---|---|---|
| `NImgs >= 2` | 474 | 177/185 (95.7 %) |
| **`SNR > 5`** | **562** | **185/185 (100 %)** |

SNR keeps *more* spots and *all* of the real ones, and it makes no assumption about
mosaicity. The `NImgs` correlation in the table above is a *consequence* of real spots
being bright, not an independent criterion.

**Actionable — the real gap.** MIDAS does not compute a per-spot SNR anywhere.
`MinIntegratedIntensity` (`midas_transforms` fit_setup, default 0 = off) is the closest
existing knob, but integrated intensity is not SNR: it carries no local background or noise
estimate, so it cannot separate a weak real spot on a quiet patch of detector from a noise
excursion on a hot one. The right change is to compute local SNR per spot during the peak
search — the frame is already in hand, and `midas_peakfit.ring_thresh.blob_snr` already
does exactly this — and expose a `MinPeakSNR` filter. That is physically principled,
costs nothing extra to compute, and does not assume anything about ω width.

For reference, what the recorded columns give (**not** a recommendation — see above):

 Evaluated against both things that
matter — noise removed, and real (indexed) spots lost:

| filter | kept | clean % | indexed kept |
|---|---|---|---|
| (none) | 1871 | 30.0 % | 185/185 (100 %) |
| **`NImgs >= 2`** | 474 | **64.1 %** | **177/185 (95.7 %)** |
| `NImgs >= 3` | 290 | 79.7 % | 171/185 (92.4 %) |
| `ReturnCode == -1` | 859 | 36.9 % | 168/185 (90.8 %) |
| `NImgs>=2 AND FitRMSE<2000` | 290 | 62.1 % | 78/185 (42.2 %) |

Read this table only as evidence that *brightness* tracks cleanliness. The `NImgs` rows
are retracted as a filter for the reasons above; the 4.3 % of indexed spots they cost are
real single-frame Bragg spots, not an acceptable rounding error.

**`FitRMSE` is a TRAP — do not use it as a quality cut.** It is an *absolute* residual, so
it scales with peak intensity: the brightest, most certainly-real spots have the largest
RMSE. Cutting at `FitRMSE < 2000` throws away **58 % of the indexed spots** while barely
improving cleanliness. If a residual cut is ever wanted it must be normalised by intensity
first.

**Not yet implemented.** MIDAS has no `NImgs` filter today; the analogue is
`MinIntegratedIntensity` in `midas_transforms` fit_setup (default 0 = off). A `MinNrFrames`
parameter applied at the same point would implement this directly.

### 6d. `MinPeakSNR` implemented (2026-08-01)

`midas_peakfit.background.region_snr` / `filter_regions_by_snr`, wired into both peak-search
paths (in-process and worker), exposed as **`MinPeakSNR <float>`**, default **0 = off**.
Shared by FF *and* PF, since both use `midas_peakfit`.

`SNR = (peak - cell_median) / cell_sigma` against the region's own
(ring, azimuthal sector) cell. The cell statistics are robust (median, 1.4826·MAD) over
the thousands of pixels in a 10° arc, so Bragg spots inside the cell cannot inflate the
background against themselves — the failure that makes a small per-spot annulus
over-optimistic (§6b). Measured on the **ungated** frame; on a thresholded frame every
sub-threshold pixel is 0, the MAD collapses, and SNR is meaningless.

**Effect on Au3 at detection** (167 blobs, 40 frames): 94 % survive at `MinPeakSNR 5`,
80 % at 10, 46 % at 20. That is a *weak* filter here — as §6b predicts, detection is
already reasonably clean on this dataset and the degradation happens later. It is not
evidence the filter is ineffective in general; it is evidence Au3's problem is downstream.

**Two caveats, both open:**

1. **Not validated for real-spot retention.** Only 2 of 167 sampled blobs could be matched
   to a known indexed spot (indexed spots are recorded at their intensity-weighted *mean*
   ω, so they rarely coincide with an individually sampled frame). n = 2 establishes
   nothing. Proper validation needs a full re-run with the filter on, comparing the final
   grain list — not done.
2. **The two SNR estimators disagree and neither is established as correct.** The cell
   estimator (this filter) and the 81×81-box-on-raw estimator (the audit) rank spots
   differently: at detection the cell form says 94 % clean at SNR 5 where the box form says
   53 %. The cell form is local in both radius and azimuth and robust to spots; the box
   form spans the ring's radial profile, so its σ absorbs real structure and is likely
   *over*-conservative. Which is right decides where a sensible `MinPeakSNR` sits, so
   **treat any specific recommended value as unestablished** until they are reconciled.

**Where it should matter more: pf-HEDM.** Spurious signal being admitted at detection is
exactly the failure this addresses, and PF shares this peak search. Unvalidated on PF data.

**Natural extension, not implemented:** the same SNR could be applied *post-merge* to the
recorded spot list, which is where §6b/§6c located the degradation on this dataset.

---

## 7. Refiner cross-implementation parity (2026-08-03)

Six implementations of the same refinement, all started from the **same** C
`IndexerOMP` seeds, the same matched spots and the same geometry, so the refiner
is the only variable. Harness: `utils/ff_refiner/refiner_crosscheck.py` +
`xcheck_analyze.py`. Of 189 seeds, **20 are refined by every implementation**
(the rest never indexed); comparison is over those 20.

| pair | Δposition µm (med / p95 / max) | misorientation ° (med / max) | Δa Å (max) |
|---|---|---|---|
| py-f64-cpu vs py-f64-gpu | 0.002 / 0.009 / **0.012** | 0.0000 / 0.0000 | 9.5e-07 |
| py-f32-cpu vs py-f32-gpu | 0.346 / 0.965 / 1.875 | 0.0007 / 0.0018 | 4.5e-04 |
| py-f64 vs py-f32 | 0.221 / 3.41 / **8.5** | 0.0138 / 0.0573 | 2.3e-04 |
| c-orig vs c-omp | 5.26 / 24.6 / **60.5** | 0.0310 / 0.1441 | 9.3e-04 |
| C vs python (any) | 12–14 / 73–79 / **85** | 0.017–0.031 / 0.088–0.155 | 2.7e-03 |

**Orientation and lattice agree everywhere** — worst-case misorientation 0.155°
across all six, worst Δa 2.7e-3 Å (6.6e-4 relative). No implementation is
finding a different grain.

**Position is the loose axis, and the C implementations are not the reference.**
`c-orig` and `c-omp` disagree with *each other* by up to 60 µm — nearly as much
as either disagrees with python (85 µm). Python is not the outlier; the DiffPos
minimum is genuinely shallow (§2d), so all of this sits inside the ~100 µm
position uncertainty the method actually has. Do not treat any one
implementation's position as ground truth at the tens-of-µm level.

**fp64 CPU vs GPU is effectively exact** (12 nm). fp32 costs ~0.2–0.4 µm
typical and up to 8.5 µm worst case — acceptable given the ~100 µm envelope,
and only with the `pos_scale` auto-equilibration in place (§3c); without it
fp32 did not refine position at all.

Runtime, 189 seeds / 16 threads: c-omp 0.1 s, c-orig 0.3 s, py-f32-cpu 4.4 s,
py-f32-gpu 5.3 s, py-f64-cpu 8.9 s, py-f64-gpu 15.7 s. The C path is 30–150×
faster here; GPU does not pay off at this problem size.

**Row-count discrepancy:** the C implementations emit 188 rows where python
emits 189. Unexplained, and flagged by the harness rather than silently
truncated. Small, but it means the two families do not agree on how many seeds
exist — worth resolving before anyone diffs these files row-by-row.

Two traps this exercise walked into, both now guarded by tests:

- **`OrientPosFit.bin` is 27 doubles per row**, with orientation at cols 1:10,
  position 11:14, lattice 15:21. Assuming a compact packing makes the row count
  non-integer and the file reads as "no output" — which looked like all six
  implementations failing. Use `io_binary.read_orient_pos_fit`, don't reimplement.
- **Orientation must be compared with symmetry.** A raw matrix angle reported a
  median of exactly 120.000° for every pair — the cubic symmetry angle. Use
  `midas_stress.orientation.misorientation_om_batch` (returns **radians**).

CI-runnable half: `packages/midas_fit_grain/tests/test_backend_parity.py`
(fp32/fp64 × cpu/gpu on the synthetic fixture, skips absent backends). Note it
must drive `refine_block`, not `refine_grain` — the per-grain entry point keeps
a fixed `pos_scale` by design and rejects `"auto"`.

### 7a. Scored against KNOWN truth — the C bridge (2026-08-03)

§7 concluded "the C implementations are not the reference" from C-vs-C
disagreement alone. That argument is suggestive but circular: it shows they
cannot *both* be right, not that neither is. This closes it non-circularly.

Pre-registered before any simulated data existed:
`PREREGISTER_refiner_gold_standard.md` (git-excluded, local). Effect size fixed
in advance at **>5 µm median per-grain position error**, justified from the
FCC-parent median `GrainRadius` of 4.08 µm and the 8.96 µm the two C codes already
differ by. R1 — "no gold standard exists" — was named in advance as the most
likely outcome.

**The bridge.** `cbridge.py` writes simulated grains of known position,
orientation and lattice into the three files the C codes read — `ExtraInfo.bin`
(16 doubles/row, **SpotID must equal row+1**, `FitPosOrStrainsOMP.c:2541`),
`IndexBest.bin` (15 doubles/row), `IndexBestFull.bin` (**stride `MaxNHKLS` =
5000 × 2 doubles**, not the per-seed spot count) — inside the *real* the FCC parent
geometry, reusing its own `paramstest_comp.txt` and `hkls.csv`. Writing only
the legacy seed pair feeds all three implementations: c-omp probes for
`IndexBest_all.bin` first and falls back (`FitUnified.c:1466-1481`), and the
python driver does the same (`driver.py:411`).

Geometry drops out by construction, which is worth stating because it was
suspected for a while: **the c-orig fitting path never touches tx/ty/tz or
p0..p14** — both codes consume ExtraInfo's already detector-corrected
YLab/ZLab. That is the same reason `_build_model` sets `apply_tilts=False`
(`driver.py:232`).

**Gates run before interpreting anything.** Zero noise, seed displaced 133 µm:
all four arms land within 0.12 µm of truth (c-orig 0.00, c-omp 0.12,
py:all_at_once 0.00, py:iterative 0.00). The layouts are right and the arms
are commensurable. A bridge that failed this would have measured only its
author's file-format bugs.

**Result** — median per-grain |position − truth|, µm, 200 grains,
115 spots/grain (the real run has ~108):

| noise px | c-orig | c-omp | py all_at_once | py iterative | seed |
|---|---|---|---|---|---|
| 0.05 | 2.03 | 2.69 | 26.43 | **1.82** | 125.6 |
| 0.2 | 8.09 | 9.20 | 42.14 | **7.19** | 137.6 |
| 0.5 | 20.58 | 20.50 | 21.01 | **18.76** | 133.3 |
| 1.6 (real) | 70.70 | 69.02 | **55.38** | 60.41 | 138.7 |

1.6 px is not a round number — it is the measured FCC-parent value, 315.6 µm
vertical residual over a 200 µm pixel, fixed in the pre-registration.

**Both refutation criteria fire.** py:iterative wins 3 of 4 levels but by
0.21 / 0.90 / 1.74 µm — every margin inside the 5 µm effect size (**R1**), and
the ranking inverts at the real noise level (**R2**).

**So c-orig is not a gold standard.** It is statistically indistinguishable
from c-omp and py:iterative below the real noise level, and at the real noise
level it is the *worst* of the four. Ten years of use establishes that it is
useful, not that it is accurate. §7's warning stands and is now demonstrated.

Corollary: **"make c-omp match c-orig" is not a well-posed goal.** The 8.96 µm
median per-grain C-vs-C disagreement is *smaller* than either code's own error
against truth at the real noise level (~70 µm). They disagree because position
is underdetermined there, not because one is wrong.

Scope it precisely — R1 covers three arms, not four. **py:all_at_once is
genuinely worse at low noise** (26.4 and 42.1 µm vs ~2 and ~8), far outside the
effect size. That is the under-refinement defect of §3c, and it is real.

### 7b. What the bridge did NOT explain — still open

On the real FCC-parent dataset c-orig reaches robust Z σ 38.84 µm and the best python arm 63.76
µm. On identical clean input they are equal. Measuring the *same statistic* in
both settings (robust Z σ of refined positions; simulated truth Z is uniform in
±1 µm so it is essentially pure error) localises this sharply:

| arm | simulated @1.6 px | the real FCC-parent dataset | ratio |
|---|---|---|---|
| c-orig | 37.94 | 38.84 | 1.02 |
| c-omp | 34.69 | 38.33 | 1.10 |
| py:iterative | 33.57 | 63.76 | **1.90** |
| py:all_at_once | 32.27 | 157.78 | **4.89** |

**The simulation reproduces both C codes on real data to 2–10 %** — an
independent validation of the whole setup, since 1.6 px was fixed beforehand
from an unrelated measurement. It does not reproduce python. The deficit is
therefore **real-data-specific and python-specific**: the simulation is not
missing something that hurts everyone.

Ruled out as the cause: the optimizer (equal in simulation), the loss (matching
C's `internal_angle` changes nothing — see below), staging alone (`iterative`
gets to 63.8 µm and stops), geometry (shared), and **dynamic spot
reassignment**. That last one was the prime suspect — c-orig re-assigns spots
during refinement when `Spots.bin`/`Data.bin` are present
(`FitPosOrStrainsOMP.c:2209-2245`) and python has no equivalent — and it is
**refuted**: deleting those files changes nothing at all. c-orig 38.84 → 38.84,
c-omp 38.33 → 38.33, same 1441 seeds, same 192.79 µm median move, with the C
logging "Spots.bin not found" to confirm the feature was off. Bit-identical.

**Cause unexplained.** Recorded as unexplained deliberately: the reading was
fixed before the reassignment test ran, and a third story fitted to the same
number afterwards would not be evidence.

### 7c. Loss/mode matrix on the real FCC-parent dataset (2000 seeds, target c-orig 38.84 µm)

| solver | loss | mode | Z σ µm | moved µm |
|---|---|---|---|---|
| lbfgs | full3d | all_at_once | 157.78 | 2.94 |
| lbfgs | full3d | **iterative** | **63.76** | **162.31** |
| lbfgs | internal_angle | all_at_once | 160.83 | 0.00 |
| lbfgs | internal_angle | iterative | 160.83 | 0.00 |
| lbfgs | angular | all_at_once | 160.83 | 0.00 |
| lbfgs | angular | iterative | 160.83 | 0.00 |
| lm | ×3 losses | ×2 modes | all 6 FAILED (exit 1; one −9 = OOM) | — |
| nelder_mead | internal_angle | iterative | no result — timed out at 3600 s | — |

**Staging is the whole effect; the loss does nothing.** This retracts an
earlier guess of mine that C's internal-angle objective was most of the gap —
matching it does not help at all. Two defects fall out:

- **`internal_angle` and `angular` do not refine position at all** — bit-
  identical 160.83 µm and exactly 0.00 µm movement in every mode. The
  `BARELY-REFINED-POSITIONS` guard fires on all four cells.
- **Every `lm` cell fails on real data** while `lm` works on the small
  fixture. Untriaged.

`full3d` is vindicated: it is the only loss that refines position at all.

### 7d. Two latent defects found by reading the source — both INERT on the FCC parent

Recorded because they are real; flagged inert because that was checked, not
assumed. Neither can explain any measurement above.

1. `FitPosOrStrainsOMP.c:2481` reads `RingNr` from `ExtraInfo` column **9**
   (`YOrig`, a lab coordinate in µm) instead of column **5** (`RingNumber`);
   everywhere else in the same file uses col 5. Only reachable through
   `RingsToReject`, and the FCC parent sets none, so the loop body never executes.
2. `driver.py:427` keeps reflections out to `2·atan(RhoD/Lsd)` in 2θ — **twice**
   c-orig's `0.5·atan(RhoD/Lsd)` in θ (`.c:2283`). c-orig is right: ring radius
   = Lsd·tan(2θ). Inert here because the `RingNumbers` 1–5 filter bites first
   (ring 5 sits at 2θ = 5.01°, both cutoffs pass it).

Harnesses: `~/Desktop/analysis/bt_1id_jul26c/{cbridge,reassign,matrix}.py`, deployed
to `copland:~/`; logs `copland:~/{cbridge,reassign,matrix}.log`; per-grain
arrays under `.../bt_1id_jul26c/cbridge/*_err.npy`.

### 7g. Independent replication on shade_LSHR (2026-08-04)

A second real dataset, different material (Ni superalloy LSHR, a=3.585, SG 225),
different beamtime (1-ID `bt_1id_nov20`), different geometry (Lsd 754 mm, 7
rings): `shade_LSHR_voi_ff_000299.ge3`, layer 1. The whole FF chain was re-run
from raw with `FF_HEDM/workflows/ff_MIDAS.py` on MIDAS **v11.0 (8be17019)** —
690 s end to end from a 12 GB `.ge3`, peaksearch 47 s — giving 22327 seeds.
Every refiner then started from those same seeds with **byte-identical**
parameters, c-orig re-run rather than reused so the reference shares them.

| arm | n | Z robust sd µm | moved µm | DiffPos µm | \|ΔPos\| vs c-orig µm | wall s |
|---|---|---|---|---|---|---|
| c-orig | 21785 | 3.92 | 263.47 | 137.34 | — | 172.5 |
| c-omp | 21785 | 4.23 | 263.13 | 139.43 | **2.81** | 78.2 |
| py-f64 | 21785 | 22.47 | 246.40 | 327.10 | **41.36** | 718.9 |
| py-f32 | 21785 | 22.47 | 246.40 | 327.10 | 41.36 | 362.5 |
| py-f64, ω placeholder | 21785 | 22.47 | 246.40 | 327.10 | 41.36 | 720.2 |

Per-axis median |difference|: c-omp vs c-orig **1.47 / 1.45 / 0.73** (X/Y/Z);
py-f64 vs c-orig **20.24 / 20.23 / 13.75**.

**This is a cleaner replication than the FCC parent.** There the two C codes disagreed by
8.96 µm, which muddied any claim about python. Here they agree to **2.81 µm**
and python sits **41.4 µm** away — 15× the C-vs-C difference. Z IQR 30.3 µm
against 5.3 µm. So the python position deficit of §7b reproduces on an
independent dataset, and on this one the C codes are mutually consistent.

Unlike the FCC parent, python here is worse by its **own** metric too (DiffPos 327 vs
137). On the FCC parent python reached a marginally BETTER residual while landing
elsewhere; here it simply fits worse. Caveat: DiffPos is each implementation's
self-assessment and may not be defined identically across them — the
implementation-independent numbers are |ΔPos| and the Z spread.

**fp32 = fp64.** Identical to 5 significant figures on every metric, at 2× the
speed. Precision is not a factor in the deficit — consistent with §7 and with
the the FCC parent matrix. (The fp32 frozen-position defect of §3c is fixed and stays
fixed here.)

**The ω placeholder is INERT — measured, not assumed.** The pipeline's
`paramstest.txt` writes no `OmegaStart`/`OmegaStep`/`NrFrames`/`NrPixelsY/Z`,
so `_build_model` substitutes `omega_start=-180, omega_step=+0.25`
(`driver.py:215`) where this 1-ID aero scan is truly `180 / -0.25`. An arm
carrying the placeholder values was run against an arm carrying the true ones:
**21785 / 21785 rows bit-identical, max |ΔPos| exactly 0.000e+00.** The covered
ω interval is ±180 either way and the refiner takes observed ω from ExtraInfo
and predicted ω from crystallography, so the keys never enter the residual.

Scope that carefully: inert **for the refiner**. Anything that maps frames to ω
— peaksearch, transforms — reads the master parameter file, where `OmegaStep
-0.25` is correct. This does not license leaving the keys out.

**Z is NOT an accuracy metric on this dataset**, unlike the FCC parent's 2 µm focused
beam. `BeamSize 400` / `Hbeam 800` are search bounds, and grains rail at the
±400 µm limit (c-orig min −399.9, max 400.0), so the true illuminated thickness
is unknown. Z spread is used here only to compare implementations.

Harness `~/Desktop/analysis/shade_LSHR/bench_refiners.py` →
`chutoro:~/bench_refiners.py`, log `chutoro:~/bench.log`, per-arm dirs under
`.../sharma_work/shade_LSHR/bench/`.

### 7h. Then vs now — the same raw layer through three MIDAS eras

Same raw file, same parameter file, three code eras:

| run | grains | DiffPos median µm |
|---|---|---|
| 2021 original beamtime | 3775 | 157.37 |
| 2024 v7 re-analysis | 3486 | 157.93 |
| **2026 today (v11.0)** | **3484** | **114.10** |

Grain-by-grain, mutual-nearest-neighbour matched (symmetry-aware, no
registration — same lab frame throughout):

| comparison | matched | median \|ΔPos\| µm | median misorientation |
|---|---|---|---|
| 2026 vs 2024 | 3443 (98.8 %) | 10.56 | 0.042° |
| 2026 vs 2021 | 3457 (99.2 %) | 10.27 | 0.040° |
| 2024 vs 2021 | 3464 (99.4 %) | 10.96 | 0.044° |

**Grain finding is stable across five years** — ~99 % of grains recur and
orientation agrees to 0.04°, lattice to 1.8e-3 Å. The count step 3775 → 3486
happened between 2021 and 2024 and nothing has moved since.

**But position scatters ~10.5 µm between ANY two runs**, including 2026 vs 2024.
If code changes drove it, the closer-in-time pair would differ less; it does
not (10.56 vs 10.27). Zero of ~3450 pairs are bit-identical, 0.3 % agree within
1 µm. That is the shallow-position-minimum result of §7 reproduced on a
different sample and beamtime, and it lands on the same scale as the the FCC parent
C-vs-C figure (8.96 µm). The scatter is in-plane: X 5.68, Y 5.45, Z 2.10 µm.

**One real improvement:** DiffPos 157.9 → 114.1 µm (−28 %) between the 2024
build and v11.0, with the Z spread of grain centres tightening 7.70 → 5.78 µm.
2021 and 2024 are indistinguishable (157.4 / 157.9). Column layout was checked
byte-identical across all three files, so this is not a column-shift artifact.

Harness `~/Desktop/analysis/shade_LSHR/compare_runs.py`.

### 7i. Full python chain vs full C chain (2026-08-04)

Both orchestrators, end to end, same raw `.ge3`, same parameter file.
Classical = `FF_HEDM/workflows/ff_MIDAS.py`, all C binaries. New =
`midas-pipeline 0.7.0 run --scan-mode ff` with `--indexer-backend python
--refine-backend python`, CUDA.

| stage | new pipeline (python) | classical (C) | ratio |
|---|---|---|---|
| peakfit | 1374.6 s | 47 s | **29×** |
| transforms | 51.4 s | 8 s | 6× |
| binning | 2.3 s | 4 s | 0.6× |
| indexing | 2420.5 s | 425 s | **5.7×** |
| refinement | OOM on GPU (see below) | 173 s | — |

Grain-level, after completing the python chain with refinement on CPU:

| | new pipeline | classical |
|---|---|---|
| grains | **4136** | 3484 |
| DiffPos median | **315.47 µm** | 114.10 µm |
| mutually matched | 3425 (82.8 % of newpipe) | |
| \|ΔPos\| on matched | **33.00 µm** median, p95 108.4 | |
| misorientation | 0.034° median | |
| per-axis \|Δ\| | X 16.4, Y 16.2, Z 11.3 µm | |
| newpipe grains with no partner | **711** | |

Consistent with §7g's refiner-only 41.4 µm (smaller here because this compares
post-`ProcessGrains` merged grains, not raw refined seeds). The 711 extra grains
are unexplained — genuine extra detections or spurious splits, not determined.

**Two integration defects found doing this:**

1. **`midas-pipeline --refine-backend python --device cuda` cannot complete an
   FF layer.** `RefinementConfig.dtype` is pinned to float64 and deliberately
   NOT inherited from `--dtype` (`config.py:197`), and there is no
   `--refine-dtype` flag, so the refiner always runs fp64. At B = 22328 that
   OOMs: `_rematch_batch` builds `(B, S, K, M)` = (21785, 244, 2, 168) tensors
   — **13.31 GiB each at fp64**, several of them — against a 47.4 GiB A6000.
   The pipeline passes `0 1` for block_nr/n_blocks, and there is no
   memory-aware batching anywhere: `n_blocks` is a work-splitting argument
   (`driver.py:396`), and `refine_block` takes the whole block as one padded
   batch. Padding is not the issue (median 240 spots vs S_max 244).
   The pin's own docstring records that fp32 ≡ fp64 since the `pos_scale`
   equilibration fix — reproduced in §7g to 5 significant figures — so the pin
   is over-conservative *and* is what breaks the GPU path.
2. **`ff_MIDAS.py -useTorchRefiner` is dead.** Its `-refineLoss` still offers
   `pixel|angular|internal_angle` defaulting to `pixel`, but the 2-D `pixel`
   loss was DISABLED 2026-05-20 (`residuals.py:125`, it raises) and the 3-D
   loss renamed `full3d`. `pixel` is rejected by `midas_fit_grain --loss`;
   `full3d` is rejected by ff_MIDAS's own argparse; `angular`/`internal_angle`
   pass both but do not refine position at all (§7c). No setting works.
   `midas-pipeline` tracked the rename; the classical workflow did not.

### 7j. Why python fails — three pre-registered tests, and an answer

**Cross-grain coupling: REFUTED.** `refine_block` optimises a whole block under
ONE `torch.optim.LBFGS` — one scalar loss, one line search, `history_size=10`
(rank-20) against a B×12 = 261 420-dimensional parameter vector whose true
Hessian is block diagonal. A compelling argument, and wrong. Median |ΔPos| vs
c-orig on 400 fixed grains:

| B=400 | B=100 | B=25 | B=5 | **B=1** | B=400, iter ×8 |
|---|---|---|---|---|---|
| 43.314 | 43.361 | 43.231 | 43.309 | **43.309** | 43.314 |

**B=1 is one optimizer per grain — total independence — and it changes the
answer by 0.005 µm.** The iteration control is null too: 8× the budget gives a
bit-identical result in the same wall-clock, i.e. the budget was never consumed.
(`PREREGISTER_grain_coupling.md`.)

**Matched-spot selection: REFUTED.** 779 grains, sets read from each code's own
`FitBest.bin`: median Jaccard **1.0000**, **92 % of grains bit-identical**,
worst 1 % still 99.1 %. Spearman(set disagreement, |ΔPos|) = **+0.057**; least-
vs most-disagreeing quartile |ΔPos| 40.89 vs 41.83 µm. The codes use the same
spots. (`PREREGISTER_matching.md`.)

**Cross-evaluation: DESCENT FAILURE — the positive result.** Both answers
scored under both objectives by one evaluator, 387 grains:

| objective | L(c-orig) | L(python) | ratio | python wins |
|---|---|---|---|---|
| **full3d — python's OWN** | 9.7415e+02 | 1.6567e+03 | **1.6432** | **2.1 %** |
| internal_angle — C's | 1.5203e-03 | 1.7837e-03 | 1.1309 | 2.6 % |

**python's answer is worse under BOTH objectives on ~98 % of grains**, and
64 % worse under its own. So this is not a modelling choice: c-orig reaches a
point that python's own loss overwhelmingly prefers, and python stops short.

> **The torch refiner does not minimise the function it is minimising.**

Which matches the failure the codebase already documents for fp32
(`refine_block.py`): the strong-Wolfe line search returns t = 0, the loss
repeats bit-for-bit, the ftol counter trips, and the solver reports
convergence. Converged-by-its-own-criterion, batch-independent and
budget-independent are all consistent with that. (`PREREGISTER_crosseval.md`.)

**Excluded to date** for this deficit: optimizer batching · iteration budget ·
precision (fp32 ≡ fp64) · loss selection · staging alone · geometry · dynamic
spot reassignment · frozen lattice block · matched-spot selection · the
objective definition itself.

Harnesses: `~/Desktop/analysis/shade_LSHR/{batch_sweep,match_diff,crosseval,
exit_state}.py`.

### 7k. RETRACTION + the actual C algorithm, read out of the source (2026-08-04)

**RETRACTED — "C minimises an INTERNAL ANGLE."** Stated in §7 and repeated
through this investigation. It is backwards for three of the four stages in
C's default (`FitAllAtOnce=0`) path. Read from `FitPosOrStrainsOMP.c`:

| stage | params **optimised** | objective (error fn) | what is **kept** |
|---|---|---|---|
| 1. `FitPositionIni` | **12** — pos + euler + lattice | **2-D (Δy, Δz)** (`FitErrorsPosT`) | **position only** |
| 2. `FitOrientIni` | 9 — euler + lattice, pos fixed | **internal angle** (`FitErrorsOrientStrains`) | **euler only** |
| 3. `FitStrainIni` | 6 — lattice, pos + euler fixed | **2-D (Δy, Δz)** (`FitErrorsStrains`) | lattice |
| 4. `FitPosSec` | 3 — position | **2-D (Δy, Δz)** weighted (`FitErrorsPosSec`) | position |
| *(FitAllAtOnce=1)* | 12 | internal angle (`FitErrorsOrientStrainsPos`) | all |

Only the ORIENTATION stage uses the g-vector internal angle. Position and
strain are both fitted against 2-D detector distance —
`Error += CalcNorm2(PosObs[0]-PosTheor[0], PosObs[1]-PosTheor[1])` — with **no
omega term**. Stage 4's `wgt` combines `WeightMask` and
`exp(-fRMSE·WeightFitRMSE)`; both are inert on shade_LSHR (1.0 and 0.0).

This retro-explains §7c, where forcing python's loss to `internal_angle` froze
position at exactly 0.00 µm movement: C never fits position with that
objective, so the result was never evidence about C's method.

**Two structural details that a naive port would miss** — and that were caught
by the user querying my first reading, not by me:

- **Stages 1 and 2 deliberately OVER-PARAMETERISE and discard.** Stage 1
  optimises all 12 parameters and the caller then resets euler and lattice to
  the seed (`XFit[i+3] = Euler0[i]`, `XFit[i+6] = LatCin[i]`); stage 2
  optimises 9 and keeps only the 3 euler. The discarded blocks act as nuisance
  parameters absorbing misfit during the fit. This is NOT the same as freezing
  them, and "position-only stage 1" — my first reading, from the function name
  — is wrong.
- **Every stage runs Nelder-Mead TWICE, warm-started**, `ftol_rel = xtol_rel =
  1e-5`, and spots are re-mapped (`CalcAngleErrors`) plus optionally
  re-assigned (`ReassignSpotsFromBins`) between stages.

Bounds: position seed ± `Rsample`, clamped to |X|,|Y| ≤ `Rsample` and
|Z| ≤ `Hbeam`/2 (stage 4 uses ± `Rsample`/2); euler ± `MargOme2` = 2°;
lattice a·(1 ± `MargABC`/100), angles ± `MargABG`/100.

Note python's 2-D `pixel` loss — disabled 2026-05-20 for omitting omega and
letting orientation drift ~20° — is exactly the objective C uses for position.
The loss is not unsound; using it while orientation is FREE is. C only ever
applies it with orientation either fixed (stages 3, 4) or discarded
afterwards (stage 1).

### 7l. The PF refiner is a DIFFERENT algorithm — port spec for both

`FitOrStrainsScanningOMP.c` (PF / scanning) is not the FF sequence with a
tweak. Per voxel it runs only TWO stages:

| stage | params | objective | bounds | initial simplex |
|---|---|---|---|---|
| 1 | euler (3) | **(Δη, Δω)** — `FitErrors3DOrient`, `sqrt(dη² + dω²)` | Euler0 ± `MargOme2` | 0.05° each |
| 2 | lattice (6), pos+euler fixed | **2-D (Δy, Δz)** — `FitErrors12D` | ± `MargABC`/`MargABG` % | 0.001 Å (abc), 0.01° (angles) |

**PF never refines position at all** — the voxel position is the scan-grid
position. `obj_12D`, `obj_9D` and `obj_3D` exist in that file but are never
called; only `obj_3DOrient` and `obj_6D` are. Matches the python config default
`position_mode="fixed"`.

**So the orientation objective differs between the two modes:** FF uses the
g-vector internal angle (`FitErrorsOrientStrains`), PF uses (Δη, Δω)
(`FitErrors3DOrient`). Any port must keep them distinct rather than unify them.

**Initial simplex — the asymmetry that probably matters most.** PF sets
`config.step_sizes` explicitly (`RunFit`, and `MIDAS_Math.c:28` forwards it to
`nlopt_set_initial_step`). **The FF refiner never sets `step_sizes` at all** —
`grep step_size FF_HEDM/src/*.c` hits only the PF file. So FF falls back to
nlopt's default initial step, which for a bounded problem is derived from the
BOUND RANGE. FF's position bounds are seed ± `Rsample` (1800 µm on
shade_LSHR), so the position simplex starts spanning **hundreds of µm**.

That is a coarse, quasi-global search, and it is the sharpest structural
contrast with python's L-BFGS, which takes local gradient steps from the seed.
It fits the established picture — both codes reach genuine stationary points
(§7j) and C reaches a better one — and it is the most likely single reason.
Untested as a cause; stated as the leading candidate, not a conclusion.

**Common to both**: every stage runs Nelder-Mead **twice, warm-started**, with
`ftol_rel = xtol_rel = 1e-5`.

### 7m. The port target: `FitUnified.c`, and a validated NM

`c_src/FitUnified.c` is the SHIPPED c-omp refiner and is already unified across
FF and PF behind one `isFF` flag, so it — not c-orig — is what to port. Its
recipe, per grain/voxel, every stage **two warm-started NM calls**,
`ftol_rel = xtol_rel = 1e-5`:

| stage | dim | objective | bounds | initial simplex |
|---|---|---|---|---|
| 1 orient | 3 | (Δη, Δω) | Euler0 ± MargOme2 | **0.05°** explicit |
| 2 posIni *(FF only)* | 3 | 2-D (Δy, Δz) | center ± Rsample/2, clamped | **none → default from bounds** |
| 3 strain | 6 | 2-D (Δy, Δz) | ±MargABC/ABG % | **0.001 Å / 0.01°** explicit |
| 4 posSec *(FF only)* | 3 | 2-D (Δy, Δz) | as stage 2, `maxeval` 5000 | **none → default from bounds** |

Re-match between stages (FF only). `obj_9D`/`obj_12D`/`obj_3D` are dead code.

**The number that matters.** The position stages pass NO explicit step, so NLopt's
default kicks in: `(ub-lb)·0.25`. With `Rsample` = 1800 µm the position bounds
are center ± 900 µm, so the **initial position simplex is ~450 µm wide**. That
is a coarse, quasi-global search over position — against L-BFGS taking local
gradient steps from the seed. Orientation and strain, by contrast, get tight
explicit simplices (0.05°, 0.001 Å). So C searches position globally and
everything else locally, and python searches everything locally.

**NM ported and validated bit-for-bit.**
`midas_fit_grain/solvers/nlopt_nm.py` is a line-for-line port of
`c_src/nelder_mead.c` (itself an exact reimplementation of NLopt `nldrmd.c`).
Checked against a C driver built on the vendored source over 7 cases —
sphere/rosen/beale unbounded, rosen bounded with default and explicit steps,
rosen started ON a bound, and a wide FF-like position box:

**every case returns a bit-identical `x` and the same return code**, including
the bound-pinned case exiting FTOL at `x = (-2.2344970703125, 5)`. The `f`
values differ only in the 14th digit, from numpy-vs-C summation inside the test
function rather than the solver.

The non-textbook behaviours all carried over: Richardson & Kuester pinning with
the coincidence test that TERMINATES rather than sliding along a bound face,
the default-initial-step heuristic, the initial-simplex out-of-bounds fallback
with its 0.1 rule, centroid excluding the worst vertex, NLopt's `relstop` plus
the L1 x-test, and returning the best point ever evaluated rather than a final
vertex.

### 7n. RESULT — the ported recipe reproduces the C refiner

`~/Desktop/analysis/shade_LSHR/c_recipe.py`: FitUnified's four stages driven by
the ported NM, per grain, on the shade_LSHR seeds. 200 grains, reference
**c-omp**:

| | median \|ΔPos\| vs c-omp | p95 |
|---|---|---|
| **ported C recipe** | **1.324 µm** | 6.21 |
| shipped py-f64 (lbfgs/full3d/iterative) | 39.906 µm | 126.93 |
| *c-orig vs c-omp, full set* | *2.81 µm* | *—* |

**The port agrees with c-omp roughly twice as closely as c-orig does** — it is
inside the envelope two accepted C implementations already span — and is
**30× closer than the shipped python refiner**.

That closes the investigation opened in §7b. The python position deficit was
never precision, batching, spot selection, the objective definition, iteration
budget or convergence.

**And it was not the optimizer either — RETRACTING §7m's leading candidate.**
I attributed the win to Nelder-Mead's coarse ~450 µm initial simplex on the
position stages searching quasi-globally. Holding stages, objectives and bounds
fixed and swapping ONLY the search, 100 grains vs c-omp:

| `stage_solver` | median \|ΔPos\| | p95 |
|---|---|---|
| `nm` | 1.679 µm | 8.01 |
| `lbfgs` | **1.553 µm** | 11.31 |

L-BFGS — a purely local gradient method — reaches the same answer, so the
solver is not the cause.

**The third arm settles it: it is the OBJECTIVES, not the staging.** Same
stages, same bounds, same re-matching, same solver — only the objective
changes:

| arm | median \|ΔPos\| | p95 |
|---|---|---|
| lbfgs + C objectives | **1.553 µm** | 11.31 |
| lbfgs + `full3d` | **38.569 µm** | 114.51 |
| nm + C objectives | 1.679 µm | 8.01 |
| *shipped py-f64* | *~42 µm* | |

The C's exact recipe under `full3d` still fails at 38.6 µm — indistinguishable
from the shipped path. But the story did NOT end there, and §7o corrects it.


Consequence: a **fully differentiable** path to the C's answer exists
(`stage_solver="lbfgs"`), so this mode need not be a permanent exception to the
differentiability rule.

Concretely, the shipped python path differs from C in four ways at once, all
now known to matter:
1. **local gradient descent** instead of a coarse-simplex derivative-free search;
2. **one objective for everything** (`full3d`, which carries Δω) instead of
   (Δη, Δω) for orientation and 2-D (Δy, Δz) for position and strain;
3. **no over-parameterise-and-discard / stage-specific bounds**;
4. **one shared optimizer per block** rather than per grain — measured
   irrelevant on its own (§7j), but it is what forced the batched design.

**Still to do**: integrate as a first-class solver/mode in `midas_fit_grain`
(the prototype calls `refine_block`'s pieces directly); validate the PF path
(stages 1 and 3 only) against the c-omp PF refiner; and address speed — the
prototype is per-grain and scalar, where the batched torch path was its main
advantage.

**Port implication.** The unit to port is not "an optimizer" but a *recipe*:
per-grain Nelder-Mead, stage-specific parameter subsets, stage-specific
objectives, over-parameterise-and-discard on FF stages 1-2, bound-derived
coarse initial simplex on FF, explicit small simplex on PF, two warm-started
calls per stage, re-match between stages. `midas_fit_grain` already has a
`nelder_mead` solver and an `iterative` mode; what is missing is this
structure.

### 7e. The python refiner does not refine the LATTICE (2026-08-04)

Found while chasing 7b, and **more consequential than the thing it was found
chasing**: per-grain strain is the main scientific output of FF-HEDM, and the
python path is not measuring it.

Refined lattice on the real FCC-parent dataset, first 2000 seeds, 1441 refined by every code
(`~/latspread.py`, `~/latdetail.py` on copland):

| arm | median strain µε | robust sd µε | deviatoric sd(a−b) µε | robust sd(angles)° |
|---|---|---|---|---|
| c-orig | 194.1 | 651.2 | 771.6 | 0.0515 |
| c-omp | 185.9 | 645.6 | 799.1 | 0.0442 |
| py:iterative | 3.3 | 13.6 | 20.0 | 0.00001 |
| py:all_at_once | 0.0 | 0.1 | 0.1 | 0.00000 |

Sample rows make it plain — python emits `5.1645002 5.1644996 5.1645004,
90.0000000 90.0000000 90.0000000` where C emits `5.1659358 5.1617696
5.1710111, 89.9944850 89.8809261 90.0096256`.

**Not a symmetry constraint.** a, b and c are all distinct in python's output
too (1441/1441 grains), and the angles are free — they just never move. So
this is not a cubic-lattice simplification; it is the lattice block failing to
refine, the same family of defect as the fp32 frozen position (§3c) and the
frozen position under `internal_angle`/`angular` (§7c).

**Established against known truth, with a positive control.** In the bridge at
1.6 px, truth deviatoric strain is 1222 µε median:

| arm | recovered deviatoric µε | error vs truth µε |
|---|---|---|
| truth | 1222 | — |
| c-orig | 1497.2 | 735.3 |
| c-omp | 1568.4 | 779.1 |
| py:iterative | 529.9 | 757.5 |
| py:all_at_once | 16.4 | **1209.4** |

Both C codes recover the strain, so the test works. `all_at_once` recovers
1.3 % of it and its error equals the full truth magnitude — the output carries
**zero information** about strain. `iterative` recovers 43 % — heavily shrunk,
though its *error* (757 µε) is comparable to C's (735 µε), so it is
under-refined rather than uninformative.

**On real data both modes are effectively frozen**: py:iterative's deviatoric
spread is 20 µε against C's 770 µε, i.e. 2.6 %, far worse than the 43 % it
manages in simulation. Why simulation is kinder here is not established.

**Which mode you get:** `FitAllAtOnce` defaults to 0 (`config.py:86`) →
`mode="iterative"` (`config.py:225`), and the FCC parent's paramstest does not set it. So
the default path is the *less* broken of the two — but on real data it is
still returning ~3 % of the strain. **Treat every strain number from the python
FF refiner as invalid until this is fixed.**

**Mechanism, confirmed by intervention.** Per-block gradient norms at a
realistic seed (150 µm, 0.1°) in the the real FCC-parent dataset geometry:

```
|g| position 2.98    |g| euler 8.51e5    |g| lattice 7.03e4
pos_scale = max(euler, lattice)/pos = 2.85e5
after rescale:  pos 8.51e5   euler 8.51e5   lattice 7.03e4   ->  12.1x down
```

`_equilibrated_pos_scale` lifts position to match the **largest** other block
(`refine_block.py:299`), which by construction leaves the **smallest** block
starved — here the lattice, by 12.1×. One shared L-BFGS step length then
advances it ~12× less per step.

**The package's synthetic fixture shows the OPPOSITE** — there |g| lattice is
the largest block (1.7e6 vs euler 9.4e5), so the joint fit is well conditioned
and no unit test can see this. Any regression test must use FF-scale geometry
(Lsd ~1.7e6 µm), not the fixture.

Rescaling the block confirms the causal link, and shows the cure trades:

| `lattice_scale` | recovered strain µε (err) | position median µm @ 0.0 / 0.05 / 0.2 / 0.5 / 1.6 px |
|---|---|---|
| 1.0 (off) | 16.4 (1209.4) | 38.7 / 26.4 / 42.1 / 21.0 / 55.4 |
| `"auto"` | **680.0 (664.6)** | 106.1 / 60.2 / 75.5 / 71.4 / **63.9** |

Strain recovery goes 1.3 % → 56 % of truth, with an error (665 µε) that beats
`iterative` (748) and nears c-orig (735) — so the under-refinement really is
block conditioning. But position degrades at every noise level, badly at low
noise. **Left default OFF** (`lattice_scale=1.0`, opt-in) because a single
scalar gradient equalizer is the wrong cure: gradient norm is not curvature,
and the lattice block is internally heterogeneous (Å alongside degrees), so no
one factor can equilibrate both halves. A per-component scale, or
reparameterizing to the dimensionless strain tensor, is the shape a real fix
would have.

`iterative` is **bit-identical** with the knob on or off — its lattice phase
optimizes that block alone, and a converged single-block L-BFGS phase is
scale-invariant in its answer. So this only ever touches the joint fit, and
the default mode is untouched.

**Still unexplained:** `iterative` recovers 43 % of truth strain in simulation
but only 2.6 % on the real FCC-parent dataset (20 µε vs C's 770). Same mode, same code — so
something about real spot lists suppresses it further. Not chased.

**It does NOT explain the position deficit of §7b.** Pre-registered reading,
fixed before looking: if python's per-grain position error rises with that
grain's truth strain and C's does not, the frozen lattice is the mechanism.
Spearman ρ over 200 simulated grains at 1.6 px: py:all_at_once **+0.069**,
py:iterative **+0.057**, c-orig −0.017, c-omp −0.045. Low- vs high-strain
tercile medians move 53.6 → 60.9 µm (python) against 71.0 → 62.6 µm (c-orig) —
no meaningful trend either way. **Null.** A real defect, but not this
mechanism. §7b stays open.

### 7f. Grain-correlated azimuthal noise — the §7 pre-registered arm, NULL

The pre-registration specified a grain-correlated systematic arm that had never
been run. Implemented as a per-grain rotation of the whole spot pattern about
the beam centre, drawn from N(0, σ_az) — which is the systematic actually
measured (Au3: coherent azimuthal +139 µm over 2236 spots ≈ 0.05°; Ce dhcp
0.026–0.040°) — on top of the 1.6 px white noise. Robust Z σ, µm:

| σ_az ° | c-orig | c-omp | py all_at_once | py iterative |
|---|---|---|---|---|
| 0 | 33.21 | 33.26 | 47.95 | 26.48 |
| 0.01 | 30.01 | 29.51 | 27.06 | 25.70 |
| 0.03 | 31.32 | 30.84 | 33.12 | 25.67 |
| 0.05 | 31.39 | 31.75 | 23.13 | 23.57 |

**Nobody degrades.** Not "all four degrade together" — nothing degrades at all.

The reason is clear in hindsight and worth recording so the arm is not re-run:
**a rigid rotation of a grain's pattern about the beam is degenerate with a
rotation of the grain**, so the fit absorbs it exactly into orientation and the
residual vanishes. This arm tested a systematic the model is built to swallow.
It also means the measured Au3 azimuthal systematic is *harmless* to grain
position, which is worth knowing on its own.

The version that could bite is a systematic **not** representable by position,
orientation or lattice — e.g. the ring-to-ring *oscillating radial* residual
seen in Ce dhcp (§7 intro), since a monotonic radial error is just hydrostatic
strain but an oscillating one is not. Not yet run.

### 7o. The deficit is MULTI-FACTORIAL — four attributions, all falsified

Four single-cause claims were made and each was killed by its own control:
the **solver** (lbfgs 1.553 vs nm 1.679 µm — no), the **staging** (C's staging
under `full3d` = 38.569 µm — no), the **objective components** (shipped path
with `phase_losses` = 38.434 vs 38.464 — no), and the **reduction over spots**
(see below — partly).

The reduction is **asymmetric**, and that asymmetry is the real finding:

| direction | result |
|---|---|
| impose Σ(r²) on the working recipe | 5.375 → **38.434 µm** (reproduces the shipped number to 3 dp) |
| remove Σ(r²) from the shipped path | 38.464 → **30.356 µm** (21 % only) |

So **Σ(r²) is sufficient to destroy, and its removal is necessary but not
sufficient.** That is why every earlier control returned null: while
sum-of-squares is present it **masks everything else** — objective components,
staging, `pos_scale` across a 285,000× range, batching, iteration budget — the
answer is ~38 µm regardless.

Mechanism: least squares weights a spot by its error **squared**, so a few
badly-matched spots dominate and pin the grain in the wrong basin however well
the rest is configured. The C accumulates **Σ‖rᵢ‖**, per-spot distances, which
is robust to them.

**Do not try to fix the shipped path with one change.** The configuration
measured to work is the whole recipe: per-stage objectives + per-stage bounds +
sum-of-norms + per-grain-independent optimisation.

| best configurations | median \|ΔPos\| vs c-omp | s/grain |
|---|---|---|
| **LM-batched + IRLS, cuda/f64** | **1.528 µm** | **0.0373** |
| c_recipe per-grain, lbfgs | 1.553 µm | 0.2869 |
| c-orig (for scale) | 2.813 µm | 0.0360 |
| c-omp | ref | 0.0162 |

`config.reduction` (`"sumsq"` default / `"sumnorm"`) and `config.phase_losses`
are now available on the shipped path for anyone wanting to explore further.

### 7p. The python INDEXER reproduces the C exactly — and its dtype default (2026-08-04)

2000 seeds, shade_LSHR layer 1, same inputs, CPU (so only dtype varies).
Seeds solved: **1946 / 2000 in all three arms**, identical.

| arm | vs C `IndexerOMP` | wall |
|---|---|---|
| **python fp64** | **0.0000 µm, 0.00000°** | 1274 s |
| python fp32 | 0.0089 µm, 0.00003° | 1254 s |
| C `IndexerOMP` | ref | **88 s** |

`midas-index` at float64 is an **exact** reproduction of `IndexerOMP` on this
data — not "close", zero to the printed precision in both position and
symmetry-aware misorientation. Worth knowing: the indexing stage is not a
source of any FF discrepancy.

fp32 vs fp64 **directly** (1946 common seeds):

| | median | p95 | max |
|---|---|---|---|
| \|ΔPos\| | 0.00894 µm | 0.0228 | **101.9** |
| misorientation | 0.000028° | 0.000090 | **0.100** |

plus **completeness differs on 6 seeds** (0.3 %). So fp32 is essentially exact
for the typical seed with a small tail where a few land in a different
solution. Nothing like the `lm_batched` case (229.96 µm MEDIAN) — and an
earlier move to disable indexer fp32 by analogy with that result was correctly
stopped by the user and reverted. The analogy had no force: `lm_batched`'s
problem is its finite-difference Jacobian, and the indexer does not use one.

**The defensible change is not a ban.** fp32 buys 1.6 % here, so the capability
is nearly worthless and nearly harmless. What is worth fixing is that
`midas_index/device.py` defaults dtype **by device** — float64 on cpu,
**float32 on cuda/mps** — so the same input indexes in different arithmetic
depending on which machine picks up the job, and on GPU you get the tail above
without asking for it. Defaulting to float64 everywhere removes the surprise
and leaves `--dtype float32` for anyone who wants it. NOT DONE — awaiting a
decision.

**Speed is the indexer's real gap: 14× slower than the C on CPU** (1274 s vs
88 s). Larger than the refiner's remaining 4.8×, and on a full layer it
dominates — the earlier all-python chain spent 2420 s in indexing on GPU
against the C's 425 s. If python-stack throughput matters, this is where the
work is.
