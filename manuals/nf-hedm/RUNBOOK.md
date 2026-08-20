# NF-HEDM Runbook — operational state

> Part of the **NF-HEDM doc set**. The spine is [`README.md`](README.md).
>
> **This is the volatile document.** The handbook is procedure and changes slowly; the
> notebook only grows. This file describes *right now*. **Update §R3 before you finish.**

---

## R1. Where it runs

conda is not on the non-interactive ssh PATH, so call by full path:

| | |
|---|---|
| shared env | `/home/beams12/S1IDUSER/opt/envs/midas/bin/python` |
| install host | **chiltepin** — the only host with internet; shared home makes it visible everywhere |
| GPU prefix | `CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE` |
| GPU choice | by **utilisation**, not free memory |
| long jobs | `setsid`/`nohup` + redirect, or SIGHUP kills them |
| plotting | **`matplotlib` is not in the shared env** — reduce remotely, write an `.npz`, plot locally (§5a) |
| seed cache | `export MIDAS_NF_SEED_DIR=<checkout>/NF_HEDM/seedOrientations` — without it the run dies with `SeedCacheNotFound` *after* writing `hkls.csv`, so it looks like it started fine (§8a) |
| outputs | the beamtime's own `analysis/` tree — **never `/tmp`** |

Hosts: chiltepin (driver dead, has internet), copland (2× A6000, 96 cores), alleppey
(4× H100), sentosa (2× H200 + 2× RTX PRO 6000), chutoro (2× A6000, no internet).

**Run the §1 floor gate and paste its output before anything else.** The gate exists
because `SumFrames` inverted its unit convention (§8j) and a mixed resolve raises no
error — the reduction and the fit agree on a frame count derived from the same wrong key.

---

## R2. What healthy looks like

**There is no single number for "healthy".** A runbook that publishes one threshold
produces false alarms on the heavy measurements and silence on the broken ones. Every row
carries the conditions it was measured under; outside those it is not a specification.

### R2a. Beam centre — `bt_1id_jul26`, 95.0000 keV, px 1.48 µm

`DetZBeamPos` images 251–285, DetZ 7/9/11/13 mm, 9 conditions (§6h).

| DetZ (mm) | ybc | zbc |
|---|---|---|
| 7 | 997.00 | 38.31 |
| 9 | 1014.01 | 41.83 |
| 11 | 1029.68 | 44.13 |
| 13 | 1043.94 | 48.80 |

```
ybc(DetZ) = 942.91 + 0.007825 · DetZ[µm]     beta_y/p = +0.007825 px/µm
zbc(DetZ) =  26.38 + 0.001689 · DetZ[µm]     beta_z/p = +0.001689 px/µm
```

Sample on the rotation axis to 0.5 µm; sample width 41.8 µm (ω = 0/180), 47.5 µm (ω = 90).

**β is per-beamtime.** Borrowing it from another was wrong by **62× in y** (hard rule 12).
The beam stripe also moved **57 px = 31 µm** between two campaigns on the same detector,
so re-measure `zbc` every campaign (§6d).

### R2b. Ranges that are NOT thresholds

| quantity | observed | condition |
|---|---|---|
| confidence at a **wrong** geometry | **1.0000** | `ty` seeds 2° apart all reach it — confidence is a *plateau*, not an acceptance criterion (hard rule 14) |
| confidence with `BoxSize` unset | 0.949153 vs 1.000000 | one Au voxel; looks like a small geometry error and is not (§7d) |
| re-seeding a refinement with its own output | tilts drift **~1°/pass**, confidence stays high | `TiltsTol` is relative to the seed (hard rule 15) |
| neighbour vs random misorientation | 0.23° / 78 % < 5° **vs** 40.98° / 4.5 % | the test that *does* separate a real orientation field from a wrong plateau — maxC and median are blind to it |
| C-vs-python per-voxel orientation | median < 0.5°, ≥ 90 % of voxels < 0.5° | 30-voxel stratified sample of the bundled synthetic Au, **not** a dataset-level result (§11) |
| triangulated `Lsd` on a wide sample | 211 µm off before refinement, 6.8 µm after | it is a **seed**, not the answer (§6i-ter) |
| σ_MAD after NLM | 2.965 → 0.282 counts (10.5×) | so `BlanketSubtraction 2` is ~7.1σ, far more conservative than it looks (§8k) |
| optimal threshold across 14 configurations | **~3.5σ however it was reached** | which is why `BlanketSigma` transfers and an absolute count does not (§8k) |

---

## R3. Current pick-up point

> **Every session updates this before it ends.** A stale pick-up point is worse than none.

**Last updated: 2026-08-11.**

**State.** The doc set was split out of the single-file handbook today. NF now has the
`DIAGNOSIS.md` and `RUNBOOK.md` it never had, so `beamreport-doc-lint` passes.

The NF packages are released and self-consistent **in this repository**:
`midas-nf-pipeline 0.6.1`, `midas-nf-preprocess 0.6.0`, `midas-nf-fitorientation 0.8.0`,
`midas-hkls 0.7.0`. `midas-nf-pipeline 0.6.1` is the first release whose own metadata
floors the two siblings past the `SumFrames` change — below it a resolve can still mix raw
and post-sum readers.

> **This is a statement about the tree, not about any machine you will run on.** Measured
> 2026-08-12, the shared env reached from `copland` was on `0.4.0 / 0.5.0 / 0.6.0 / 0.7.0`
> — three of four a full release behind these numbers. Two independent sessions read the
> sentence above as "you are clear to run" and were wrong. **Run the phase-0 floor gate on
> the host you are actually using. Nothing in this file substitutes for it.**

**Open, not blocking:**

1. **The `SpotsInfo.bin` chain has no C-parity evidence** (§11). `hex_grid`, `diffr_spots`,
   `process_images`, `seed_orientations` and `tomo_filter` carry unit tests over synthetic
   data only. A reconstruction is not wrong because of this, but a report that does not say
   so is overclaiming.
2. **20-ID HT-HEDM: the ω sign is undetermined**, so every orientation map from that
   beamline is mirror-ambiguous and must say so. The two code blockers that used to sit
   here — no HDF5 support, and a whole layer loaded into RAM — are **closed**; set
   `extOrig h5` and the reduction runs, streaming by default (§3h, §10f). The sign is not
   a code problem and needs a determination at the beamline.
3. **Geometry A vs B is a preference, not a measurement** (§11 could-not-verify 11). Do
   not report A as "the verified geometry".
4. **`DIAGNOSIS.md` has four entries.** Three is a working start; it grows the day someone
   works out what a strange plot meant.

**Nothing is mid-run.** No jobs on any host belonging to this thread.
