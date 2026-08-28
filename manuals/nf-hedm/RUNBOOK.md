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

### R1a. Shared env — measured 2026-08-28, and it is CURRENT

Earlier notes in this doc set and in the campaign checkpoints say the shared env is stale.
**It was; it is not any more.** Measured on chiltepin, 2026-08-28:

| package | env | tree | |
|---|---|---|---|
| `midas-nf-preprocess` | 0.7.2 | 0.7.2 | ✅ |
| `midas-nf-fitorientation` | 0.9.2 | 0.9.2 | ✅ |
| `midas-nf-pipeline` | 0.6.5 | 0.6.6 † | — |
| `midas-hkls` | 0.9.0 | 0.9.0 | ✅ |

† the tree is one patch ahead only because of the floor bump made the same day (§R3).

**Metadata is not enough on this env and never was** — files were once patched in place on
a non-editable install, so version strings agreed with each other while the code disagreed
with both. Two checks were therefore run, and both must be part of any future re-check:

- `importlib.metadata.version()` **and** `__version__` agree for all four — no drift.
- The capability itself is present: `process_images.io` exports `is_hdf5`,
  `Hdf5FrameSource`, `check_pixel_scale`, `open_source`;
  `median.streaming_temporal_median` exists; and `process_images.params` binds all six
  20-ID keys (`extOrig`, `DataLoc`, `PixelScale`, `StreamFrames`, `MedianFrames`,
  `MedianRowBlock`).

**Consequences.** The reinstall that the campaign checkpoints list as an open item is
**done**. The `sys.path.insert(...)` overlays in the `nfdev_jul26` / `au0802` analysis
scripts, and the in-place `fit_multipoint.py` patch, are **no longer needed** — that
patch's content is committed. Do not reintroduce either.

> **Check the capability, not the number.** A grep for `"h5"` in `dir(io)` returns nothing
> here and means nothing — the symbols are spelled `is_hdf5` / `Hdf5FrameSource`, which do
> not contain that substring. The first pass of this very check reported a false negative
> for exactly that reason.

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

### R2c. 20-ID-D HT-HEDM — `NF_Au_cube_0802`, 63.314 keV, px 0.548 µm

The commissioning reference. Every number below is *this* campaign under *these*
conditions; the entries marked **not anchored** are limits of the campaign, not of the
method. Full derivation in Lab Notebook §8i.

| quantity | value | how it was established |
|---|---|---|
| `Lsd` | **6138.7 / 8138.7 / 10138.7 µm** | triangulation, ΔD = 2000 µm (operator); 3 pairs pass the gates, y-z splits 25/58/54 µm, nulls die **13.75×** (ω-shuffled) and **8.46×** (position-scrambled). Needs `r_min = 1200 px` |
| reproducibility of that `Lsd` | **1.0 µm** vs the July campaign | independent scan, same beamline |
| `ybc` | 2653.11 / 2650.01 / 2647.97 | on-axis cube's stationary shadow |
| `zbc` | 62.42 / 62.68 / 63.22 | `find_stripe`, 113–115 frames, IQR **0.02–0.08 px** |
| calibration | **maxC 1.000000, refinement +0.0000000000** | geometry exact *before* refinement |
| — and it is **not** a plateau | 100 % of 1071 C ≥ 0.9 voxels within 1°, median 0.707° | the neighbour-vs-random test of R2b, run in the direction that could have failed |
| external check | cube 2 at r = **497.0 µm** vs absorption shadow **499.8 ± 2 µm** | unrelated physics; 2.8 µm |
| SS316L distance offset | δ = **−837.9 µm** vs the Au campaign's **−837.7** | 0.2 µm, two beamtimes two weeks apart |
| encoding | **unscaled**, max 4092, gap 4 | `np.unique`. The July Au scan on the *same detector* is ×64 |
| ω | `aero`, **negated** — `OmegaStart 180`, `OmegaStep -0.25`, 1440 frames | instrument scientist, 2026-08-28 (§2a) |
| **nominal detector positions** | **not anchored** | only ΔD = 2000 µm was supplied; any δ quoted against a motor scale floats |
| **tilts** | **unconstrained** | two good SS316L refinements disagree on the sign of `ty` (−0.343° vs +0.098°). Au_0802's zeros are "the refinement left them alone", not a measurement |

**Do not read the `Lsd` triple as a specification.** It is what this geometry was; the
transferable results are the **1.0 µm** reproducibility and the **0.2 µm** δ agreement,
because those are the ones with a second measurement behind them.

---

## R3. Current pick-up point

> **Every session updates this before it ends.** A stale pick-up point is worse than none.

**Last updated: 2026-08-28.**

**State — 20-ID-D is fully in scope, end to end.** The last gate closed today. In order of
when they fell:

- **HDF5 reader + streaming median** (2026-08-19) — `extOrig h5` reads the 20-ID layout
  directly and a layer never has to fit in RAM (§3h, §10f).
- **ω sign** (2026-08-28) — `aero`, negated, from the instrument scientist (§2a). The
  mirror-ambiguity label on 20-ID maps is **retired**, and the completed reconstructions
  already used that convention, so nothing needs re-running.
- **Beam energy 63.314 keV** (2026-08-28) — confirmed for `nfdev_jul26` and
  `bt_20id_jul26b` by the instrument scientist. It was previously flagged as inferred from a
  filename. *The procedure gap is still open for a new beamtime* (§3h): the Bluesky log
  records the foil-wheel table but never which foil was selected.
- **Shared env** (measured 2026-08-28) — current, no drift, capability verified (§R1a).
  The reinstall that the checkpoints list as open is done.

Tree versions: `midas-nf-preprocess 0.7.2`, `midas-nf-fitorientation 0.9.2`,
`midas-hkls 0.9.0`, `midas-nf-pipeline 0.6.6`, `midas-suite 0.10.3`.

> **This is a statement about the tree, not about any machine you will run on**, and that
> distinction has burned two independent sessions who read it as "you are clear to run".
> §R1a happens to agree with it today. **Run the phase-0 floor gate on the host you are
> actually using anyway. Nothing in this file substitutes for it.**

**Uncommitted, and it needs a release to take effect:** `midas-nf-pipeline` 0.6.6 and
`midas-suite` 0.10.3 raise the `midas-nf-preprocess` floor from `>=0.6.0` to `>=0.7.0`.
The HDF5 reader first shipped in 0.7.0, so **until these are published, a fresh
`pip install midas-nf-pipeline` can still resolve an env in which the documented 20-ID
route does not exist — and the floor gate, which reads floors out of the tree, passes it.**

**Open, not blocking:**

1. **The `SpotsInfo.bin` chain has no C-parity evidence** (§11). `hex_grid`, `diffr_spots`,
   `process_images`, `seed_orientations` and `tomo_filter` carry unit tests over synthetic
   data only. A reconstruction is not wrong because of this, but a report that does not say
   so is overclaiming.
2. **Matched-filter spot detection is validated at DETECTION level only.** It finds
   5.8–8.6× more spots than a raw threshold and ~10× more than NLM at an equal *measured*
   false-positive budget, on three datasets — but **more blobs are not better indexing**,
   and it has never been carried through a full reconstruction. It is deliberately not the
   default. The open test is one head-to-head recon, matched filter vs NLM, same geometry.
   Its false-positive counts are also a **lower bound** (the negated-residual null
   under-counts for Poisson data): the *ranking* is fair, the absolute rate is not.
3. **Geometry A vs B is a preference, not a measurement** (§11 could-not-verify 11). Do
   not report A as "the verified geometry".
4. **20-ID tilts are unconstrained and the motor scale is unanchored** (§R2c). Neither is
   a code problem; both want a beamline measurement.
5. **SS316L wants a re-run at `Rsample ~900`.** At 600 the plate is clipped — 27 % of the
   top-edge voxels still index. Layer 2's reduction is staged and was never launched.

**Nothing is mid-run.** No jobs on any host belonging to this thread.

> **Access, changed 2026-08-19 and still true.** `s1iduser` is **no longer in**
> `bt20idjul26b-20id-962940` or `nfdevjul26-20id-0a26b1`, so neither 20-ID beamtime is
> readable from that account. Use **`hsharma@chutoro`** — in both groups, and chutoro
> mounts `/gdata` where alleppey does not. Backwards from the usual assumption:
> `~/nfdev_recon/` on chutoro is **gone** (including its 13 GB `SpotsInfo.bin`) while
> `/scratch/s1iduser/au0802_recon/` survives. `/scratch` is scratch — copy out anything
> that must persist.
