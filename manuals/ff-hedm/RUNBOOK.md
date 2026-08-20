# FF-HEDM Runbook — operational state

> Part of the **FF-HEDM doc set**. The spine is [`README.md`](README.md).
>
> **This is the volatile document.** The handbook is procedure and changes slowly; the
> notebook is evidence and only ever grows. This file describes *right now* — where things
> run, what a healthy number looks like on this instrument, and where the last session
> stopped. **Update §R3 before you finish.**

---

## R1. Where it runs

Full paths, because conda is not on the non-interactive ssh PATH:

| | |
|---|---|
| shared env | `/home/beams12/S1IDUSER/opt/envs/midas/bin/python` |
| install host | **chiltepin** — the only host with internet; shared home makes it visible everywhere |
| GPU prefix | `CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE` |
| GPU choice | by **utilisation**, not free memory |
| long jobs | `setsid`/`nohup` + redirect, or SIGHUP kills them |
| outputs | the beamtime's own `analysis/` tree — **never `/tmp`** |

Hosts: chiltepin (GPU driver dead, has internet), copland (2× A6000, 96 cores), alleppey
(4× H100), sentosa (2× H200 + 2× RTX PRO 6000), chutoro (2× A6000, no internet).

**Before any run, pass the §0 install gate and paste its output.** Every number below is
conditional on it.

---

## R2. What healthy looks like

**There is no single number for "healthy".** A runbook that publishes one threshold
produces false alarms on the heavy measurements and silence on the broken ones. Each row
below carries the conditions it was measured under; outside those, it is not a
specification.

### R2a. Reference geometry — `bt_1id_jul26`, GE5 (ADEPT), 95.0 keV

Established 2026-07-30. Detector 2048², 200 µm, monolithic. Single layer.

| quantity | value | how |
|---|---|---|
| energy | 95.0 keV (λ 0.130510 Å) | `HEM/Energy`, Emon, spec log, beamline confirmation |
| `DetZ` readback | 1485.00 mm | `instrument/DMS/DetZ` — **not** `Lsd` |
| `Lsd` | ≈ 1666 mm | CeO₂ fit; `analysis/ceo2_calib_ge5/summary_all.json` |
| `BC` | ≈ (1018.7, 1076.5) px | CeO₂ fit |
| tilts | ty ≈ 0.0–0.07°, tz ≈ 0.90–0.94° | CeO₂ fit, 0/180 spread |
| calibrant strain | ≈ 19 µε mean, 13 µε median | v2, residual map discarded |
| `Lsd` / `BC` repeatability | 0.013 % / 0.01 px | independent 180° repeat (§5f) |

### R2b. Reference reconstruction — Au cubes, file 8

| quantity | value | how |
|---|---|---|
| sweep | 1441 logged frames, **1440 used** | par field 21 + `SkipFrame 1` |
| ω (MIDAS) | `OmegaStart 180.00`, `OmegaStep -0.25` | §2 + §3e |
| `RingThresh` | 10 / 20 / 20 / 10 / 10 (rings 1–5) | measured, §6b — **not** the template's 60 |
| spots | 2076 binned rows, **229 credible** | Lab Notebook §4d |
| grains | **2** (parent + Σ3 twin), confidence 1.000 | Lab Notebook §4d |
| grain radius | **114.62 / 99.97 µm** | Lab Notebook §3a — C-cross-checked. **21 µm is the pre-fix buggy value**; reproduced 114.6 on 2026-08-12 |
| lattice | a = 4.07976 Å | |
| residuals | DiffPos ≈ 200 µm, DiffOme ≈ 0.05°, DiffAngle ≈ 0.08° | §8a, columns ≥ 0.5.7 |
| indexed fraction | 8.9 % — **and the recon is COMPLETE** | §11; ~98 % of the list is noise |

### R2c. Ranges that are *not* thresholds

| quantity | observed | condition |
|---|---|---|
| grain position accuracy | **~100 µm, no better** | candidates within a cluster disagree by 50–280 µm at completeness 1.0 (Lab Notebook §2d) |
| C-vs-python refiner position | median 12–14 µm, max 85 µm | 20 seeds refined by all six implementations (Lab Notebook §7) |
| c-orig vs c-omp position | up to 60 µm | the two C codes disagree with *each other* — neither is ground truth |
| orientation, any implementation | worst-case misorientation 0.155° | Lab Notebook §7 |
| lattice, any implementation | worst Δa 2.7e-3 Å (6.6e-4 relative) | Lab Notebook §7 |
| peak search runtime | 55 s (0.3 s if resumed) | 1440 frames, 5 rings — **0.3 s means it skipped** (§7) |

**A grain count in the thousands on a calibration sample is not a peak-search problem
until you have ruled out the plumbing** — check the §0 floors first (§8b item 1).

---

### R2d. Reference geometry and reconstruction — 20-ID Varex, `bt_20id_jul26b`

Established 2026-08-15 on ti7al (`ti7al_att6_e0p1_BH100_OL25_scans13`, layer 1).
Detector 2880², 150 µm, monolithic. Ti-7Al, hexagonal, SG 194.

| quantity | value | how |
|---|---|---|
| energy | 63.000 keV (λ 0.19680026389101524 Å) | **asserted, not measured** — these files carry no energy metadata; the GUI seed says 0.1958 Å |
| `Lsd` | 895 409 – 895 424 µm | CeO2 `Ceria_0805_…aero_0_002468`, `--mode ff` |
| `BC` | ≈ (1450.84, 1467.46) px | same; auto-seeded, never guessed |
| tilts | ty ≈ −0.36 to −0.38°, tz ≈ 0.52 to 0.55°, tx = 0 from powder | `--mode ff` |
| **`RhoD`** | **309 538 µm** (corner 2063.6 px × 150) | rule 15 / §6d. The template's `2000000` indexes **0 seeds** |
| calibrant strain | 58 – 61 µε post-residual, 88 – 91 µε in-loop | `--mode ff`, gate 100 µε |
| calibration repeatability | **±15 µm in `Lsd`, ±3 µε** — *not* bit-reproducible | three identical runs: 895409.47 / 895409.47 / 895423.84 |
| sweep | 1442 raw frames, **1441 used** | `SkipFrame 1` |
| ω (MIDAS) | `OmegaStart 180`, `OmegaStep -0.25` | already negated in the file |
| `RingThresh` | 75/75/75/70/70/70 (rings 1–6), `OverAllRingToIndex 3` | ring 3 carries ~3× ring 1's spots; seeding from ring 1 cost 35 grains |
| rings in `hkls.csv` | **46** | at the correct `RhoD`; 745 at the wrong one |
| grains | **208**, or **226** after §5h | `--refine tx,Wedge` re-run |
| `tx` / `Wedge` | −0.0786° / −0.0072° | §5h; profile contrast 3.9 in `tx` |
| grain-Z scatter | 153 µm → **76 µm** after §5h | against 29 µm for a uniform 100 µm beam slab |
| completeness | median 0.580 → 0.630 | still rising at the 0.4 cut, so the list is truncated |
| runtime | ~4 min for one layer | 32 CPU, CUDA, binning on CPU |

Other samples in the same beamtime, for scale: nf709 (cubic) 8060 grains and
only 70 rings even at the wrong `RhoD`; ruby (SG 167) just 6 grains, with a much
larger `tx` ≈ −0.26° that profiles cleanly (contrast 91).

## R3. Current pick-up point

> **Every session updates this section before it ends.** If it is stale, the next session
> re-derives what you already knew.

**Last updated: 2026-08-11.**

**State.** The doc set was split out of the single-file handbook today (this file is part
of that). The FF pipeline itself is in a released, self-consistent state **in this
repository**, which is not the same as on the machine you will run on:

> Measured 2026-08-12 before the environment was upgraded, the shared env reached from
> `copland` was on `midas-fit-grain 0.6.0` against a floor of 0.7.0, and a session was
> correctly stopped by the §0 gate. **Run the gate on the host you are using.** The list
> below describes the tree.


- All FF-path packages released and on PyPI with correct floors — `midas-pipeline 0.8.2`,
  `midas-ff-pipeline 0.4.3`, `midas-process-grains 0.7.1`, `midas-fit-grain 0.7.0`,
  `midas-zipper 0.1.5`, `midas-suite 0.7.3`.
- `midas_env` on the Mac passes the §0 gate: no package below floor, no metadata drift.
- Three checkers run in the pre-commit hook: `scrub_check`, `doc_citation_check`, and the
  cosmetic-commit-aware `pypi_audit`.

**Open, not blocking:**

1. **NF and Laue have not been split** into doc sets. FF is the template; port after this
   shape survives one real handover.
2. **`RUNBOOK.md` for NF does not exist** — NF has no operational-state document at all.
3. **`DIAGNOSIS.md` exists only for FF.** NF and Laue have no diagnosis reference, so
   `beamreport` can produce a descriptive report for them but not a diagnostic one.
4. **beamreport is pre-release** — the contract is written and enforced, the report
   builder is not finished. The FF adapter (`utils/midas_ff_report_beamreport.py`) is
   written against the contract.
5. **`pypi_audit` class B still lists three packages** — two are `__version__`-only syncs,
   one is `midas-parsl-configs`. None ship a behaviour change; left unreleased on purpose.

**Nothing is mid-run.** No jobs on any host belonging to this thread.
