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
pytest tests -q                                  # 925 pass, 14 skip
MIDAS_DEFECT_REAL_DATA=1 pytest tests -q         # 932 pass, 7 skip (946 collected)
```

The real-data fixtures live at `tests/fixtures/demk_g1592_9r.npz` (committed, 204 KB) and a
larger voxel NPZ that must be fetched from `/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/`.
Seven tests skip without the env var; one skips without `parameters_final.txt` from copland;
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

**2026-09-10.** A test run of the solve-cell doc set on two delivered La3Ni2O7 positions, and a fold-in of
the nickelate project's remaining local-only work. Nothing is running. Uncommitted.

* **P12 — a package bug, fixed.** `indexing.index_from_cloud` converged its cell through
  `refine_to_convergence` WITHOUT the crystal's space group (default 139, I-centring), and
  `rows.index_from_pairs` / `index_by_grid` dropped an in-scope `space_group_number` at three `match_mask`
  calls. On S5 (Fmmm) the fix took `index_from_cloud` from 9 to 11 INDEXED and the median residual from
  2.05 to 0.61 px; 2604 (sg 139) is byte-identical. `test_space_group_forwarded.py` fails on any new
  in-package call that drops one.
* **The composition is now in the package:** `domains.find_domains` (the per-position multi-domain driver),
  `completeness.targeted_recovery` (predicted-site extraction against a same-ring null),
  `selfcal.selfcalibrate_from_crystals` (detector tilts from indexed domains), `rod_profile.diffuse_to_bragg`
  and `centred_L_nodes` (the (h,k) rod test); from 2026-09-09/10, `geometry.detector_angle_maps` and
  `honesty.decoy_test` / `feature_in_raw`.
* **The driver port found a misalignment** in how the local driver called `omega_smear_duplicates` once two
  earlier domains existed — verified ESTABLISHED (four lenses): replaying the real call at five 2604 positions it
  missed 142 duplicates and changed 15 of 26 accept/reject decisions. `find_domains` builds the arrays per domain.
  **Re-run done (provisional):** aligning the check removes 364 of 1278 domains across the raster (pair-seeded
  546 → 305); domains found both ways keep their cells. Re-derive any count of third-or-later or pair-seeded
  domains before reviving it.
* The 2026-09-07 next action 1 is settled: `refine_to_convergence` has in-package callers
  (`index_from_cloud`, `find_domains`).
* `test_manual_calling_contract.py` now executes the PONI → live frames → angle maps → ω-mapping contract,
  with a dead frame in the scene so mixing live and raw frame indices fails.

**2026-09-07.** The doc set gained an indexing chain, two hard limits, and a
calling contract. Nothing is running. Suite **925 pass / 14 skip** (932 with
`MIDAS_DEFECT_REAL_DATA=1`, 946 collected).

* **`rows.py` is documented for the first time** (`phase-2-index.md`): lattice-row
  and pair seeding for weak domains, `search_null`, `refine_to_convergence`. 1389
  lines that previously appeared in no manual.
* **`ENVELOPE.md` §13/§14 and `DIAGNOSIS.md` 18/19 are new, and they are the
  expensive lessons.** A per-group difference cannot be validated by reproduction
  when the groups ARE orientations — a planted identical cell reproduced a real
  ordering at Kendall 0.73. A re-analysis sharing 95 % of its input is not a
  replication. And an acceptance gate of the form `|x/x_seed - 1| < tol`
  manufactures the correlation it would be read as proving (+0.128 ± 0.042 from a
  zero-effect null); `test_seed_referenced_gate_manufactures_correlation` pins it.
* **`LAB_NOTEBOOK.md` R15** retracts a six-way per-grain `c` ordering AND its
  two-level fallback, on four independent reviews.
* **`phase-1-ingest.md` now ends with a calling contract** — exact signatures plus
  a trap table — written after six of those traps were hit in one afternoon by
  the author of the rest of the doc set. `detect_powder_rings` without
  `azimuth_deg` is the costly one: it discards ~half the real reflections.

Next actions, in the order I would take them:

1. **Decide `refine_to_convergence`'s fate** — it has zero in-package callers and
   no test that can fail against the historical bug. Either something calls it or
   it goes private.
2. **Fold a second real-data anchor into the suite.** The chain now has three
   (Cu-Al, a DAC ingest, a tetragonal DAC raster) and only the first is committed.

**Superseded 2026-09-02 entry.**

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
