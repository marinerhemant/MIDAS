# FF-HEDM Lab Notebook — `pokharel_jul26` / `Au3_cubes_ff_000008`

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
| 9 | Spot list degrades after detection; brightness (SNR) is the discriminator, **not** ω multiplicity — 45.9 % of credible spots are single-frame and 8 indexed ones reach SNR 2511. `RingThresh 10` is close to what both criteria recommend. Gap: MIDAS computes no per-spot SNR | RESOLVED — `MinPeakSNR` not yet implemented | §6b, §6c |
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
DiffPos, DiffOme and DiffAngle. See Handbook Handbook §8a.

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
`~/Desktop/analysis/pokharel_jul26_calib/spot_{noise_audit,noise_null,snr_all,frames_fixed}.py`
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
| ω sign (`aero` is CW, negate every ω) | all **7297** `pokharel_jul26` FF par rows; rule shared with NF | 2 |
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
