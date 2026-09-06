# Runbook — where it runs, what healthy looks like, and where we are

**The only volatile file in the set.** Every session updates the pick-up point before it
finishes. A stale pick-up point is worse than none.

**Last updated 2026-09-02.**

---

## Where it runs

| | |
|---|---|
| Package | `~/opt/MIDAS/packages/midas_defect` (editable) |
| Local env | `/Users/hsharma/miniconda3/envs/midas_env/bin/python` — **full path**, conda is not on a non-interactive PATH |
| Remote env | `/home/beams12/S1IDUSER/opt/envs/midas/bin/python` — same env on every beamline host |
| GPU prefix | `CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE` |
| Bulk data | `/scratch` on the compute host, **never** `/home/beams*` |

`KMP_DUPLICATE_LIB_OK=TRUE` is set at import time by `midas_defect/__init__.py` because
`geometry.pixel_to_qlab` reuses `midas_transforms.fit_setup.transform.apply_tilt_distortion`, which links a second
OpenMP runtime alongside torch's. Do not remove it.

## Tests

```bash
cd packages/midas_defect
pytest tests -q                                  # 558 pass, 13 skip
MIDAS_DEFECT_REAL_DATA=1 pytest tests -q         # 564 pass, 7 skip (571 collected)
```

The real-data fixtures live at `tests/fixtures/demk_g1592_9r.npz` (committed, 204 KB) and a
larger voxel NPZ that must be fetched from `/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/`.
Six tests skip without the env var; one skips without `parameters_final.txt` from copland;
six skip on MPS for float64/complex128 reasons and pass on CPU.

## Healthy ranges, with their conditions

**Never a single threshold.** One number produces false alarms on the heavy measurements and
silence on the broken ones. A row that cannot state its conditions is not ready to be a row.

| quantity | healthy | conditions it was measured under |
|---|---|---|
| masked fraction | 10–15 % | Pilatus 2M, vendor-negative gaps + low-count cut + 2 px grow. A tiled detector with more gaps runs higher; check the breakdown, not the total. |
| 3-D blobs kept / labelled | ~1/3 | threshold 200 counts, `min_vol` 10 real voxels, 36 frames. Scales with threshold; report both numbers, never the ratio alone. **200 is a default** — the demk cloud bottoms out at 30 with 71.5 % of voxels below 200, so read the floor off your own data (`ENVELOPE.md` §3). |
| blobs split by watershed | ~9 % | `split_ratio` 3.0 (scale-free). A much higher rate means the ratio is too low and streaks are shattering. |
| spot list, single crystal + gasket | 400–650 sub-peaks | 36 frames, 1679×1475, after `n_frames >= 2`. A DAC with a strong gasket ring can put >100 of these on one ring — separate them, see `ingest.flag_powder`. |
| median indexing residual | 1–2 px | flat detector model, no tilt, ~50 reflections. Above ~3 px, decompose it before touching the cell (`DIAGNOSIS.md` 11). |
| completeness MISSED | **0** | this is not a range. Non-zero MISSED means the solver skipped reflections that are present. |
| satellite ω width | σ ≈ 0.6° | 1° ω step. At a coarser step the compactness test cannot be made at all. |
| forbidden-shell occupancy | empty | across all indexed grains, by 3-D vector distance. Anything non-empty is either a real polytype or a decontamination failure — check which. |

## Timing, measured

| step | cost | on |
|---|---|---|
| `build_mask` | 0.6 s | 36 × 1679 × 1475, M1 Max |
| `subtract_background` (8 sectors) | 11 s | same |
| `find_blobs_3d` | 36 s | same, threshold 200 |
| `beam_centre_from_pairs` | ~20 s | 440 spots, ±40 px search at 1 px, 200 null draws |
| full suite | 96 s | CPU |
| full suite + real data | 102 s | CPU |

Ingest is embarrassingly parallel over raster points; a 900-point map is ~15 min on a
96-core node.

## Current pick-up point

**2026-09-02.** Two preregistered runs on the reference sample, both negative, and the doc set
updated to match. Nothing is running.

* **The 18.7 % fault-rod budget fraction is withdrawn** (`LAB_NOTEBOOK.md` R12) and nothing
  replaces it: a rod fraction is **not obtainable** on this sample (`ENVELOPE.md` §1a). Its
  replacement was itself withdrawn (R13), and the directional test that would have rescued it
  came back **void**, not null (R14).
* **New:** `bragg_diffuse.check_orientation_convention` + 5 tests, after finding that two demk
  voxel products need opposite conventions and `polytype/ladder.py` stated one universally.
  Suite 564 pass / 7 skip.
* **Manuscript consequence:** `dev/paper/demk_9r_ffhedm.tex` §3.5 and Fig. 5 must lose the
  18.7 % and gain no replacement; `build_demk_9r_figures.py:392` still hardcodes the old
  budget. The draft is out with the co-author with §3.5 flagged as open.

Next actions, in the order I would take them:

1. **Fix §3.5 and Fig. 5** — the only change the manuscript needs from this work.
2. **O6** — decide whether §4, §1a and §8 are one limit, and if so write the single phase-0
   precondition that would have caught all three.
3. **O4 / E12** — fold the ingest validation into the suite as a committed fixture; the four
   newest modules still have no real-data regression.
4. **O2** — pin the contrast factor against an independent method so densities can go absolute.
