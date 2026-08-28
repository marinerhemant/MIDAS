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

**20-ID raw data is group-restricted, and the grant can disappear under you.** The DM
tree `/gdata/dm/20ID/HT_HEDM/<cycle>/<proposal>` is `drwxr-x---+` with a per-proposal
group. On `nfdev_jul26` the `s1iduser` ACL entry that had worked all campaign vanished
on 2026-08-19, and every stage touching raw data had to move to `s20hedm`:

```bash
ssh chiltepin "ssh s20hedm@copland.xray.aps.anl.gov '<cmd>'"   # nested, NOT ProxyJump
```

`ProxyJump` only tunnels TCP and cannot authenticate as that account — do not retry it.
Three follow-on traps, all hit:

* the second account could not write `results/`, owned by the first. Both are in `xfdop`,
  so `chmod g+w results/ diag/` fixes it;
* a **stale root-owned logfile blocks the redirect** and the job then silently never
  starts. `rm` the log as its owner before relaunching;
* kill by **PID**, never `pkill -f` — these are shared accounts with ~16 users.

Also note the shared env must be on `PATH`, not just invoked by full path: the pipeline's
subprocess stages (`peakfit_torch`, `midas_indexer`, `midas_fitgrain`) are invoked by
**bare name**.

```bash
export PATH=/home/beams12/S1IDUSER/opt/envs/midas/bin:$PATH
```

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

### R2d. Reference geometry and reconstruction — 20-ID-D Varex, `bt_20id_jul26b`

> **The station is 20-ID-D.** Confirmed with the instrument scientist 2026-08-28. NF runs
> at **D**; FF and PF run at both **D** and **E**, and **everything reconstructed through
> this doc set to date is D** — both R2d and R2e. Say which branch a result came from:
> one of these campaigns was filed as "20-ID-E" for nine days on the strength of the
> beamline number alone, which is rule 13 in its purest form. The detector folder name
> `varexD` is the D branch's Varex and is a useful tell, but confirm rather than infer.

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
| ω (MIDAS) | `OmegaStart 180`, `OmegaStep -0.25` | **§2b, re-derived — not read from the file.** This row said "already negated in the file" until 2026-08-28; nothing in a `.vrx.h5` records the sign |
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

### R2e. Second 20-ID-D reference — `nfdev_jul26`, Varex, CeO2 0723 @ detY 900 mm

Established 2026-08-17 → 08-19. **Same station, same detector, same cycle as R2d — and
four of its conventions differ.** That is the point of carrying both: R2d is not a
template, and neither is this.

| quantity | value | how |
|---|---|---|
| `Lsd` | 899 916.02 µm | CeO2 `Ceria_0723_…`, `--mode ff`, `aero_180` of the 0/180 pair |
| `BC` | (1450.988, 1467.344) px | same — auto-seeded, never guessed |
| tilts | ty −0.2934°, tz 0.4935° | same |
| `RhoD` | 309 544.2 µm | resolved by `--mode ff`; `ps_au.txt`'s `2000000` is fatal (rule 15) |
| λ | 0.19582430 Å (63.314 keV) | **user-supplied, never measured** — no energy field exists in a `.vrx.h5` (§3b-2) |
| calibrant strain | **60.4 µε** (med 34.1, trim-5 % 40.9) | gate 100 µε — PASS |
| 0/180 repeat | `Lsd` 46 µm (0.005 %), BC 0.034/0.004 px | §5f |
| validation | §5b 11/11 rings, worst ratio dev 0.00317, innermost = (111); §5d worst cos η 0.541 px, rms 0.467 px | mandatory overlay, rule 5 |
| cross-beamtime BC | agrees with R2d's fit to **0.18 / 0.12 px** | independent check that both geometries are real |
| **`darkLoc`** | **per scan** — Au `/exchange/dark`, alumina `/exchange/bright`, CeO2 `/exchange/dark` | §3d. **This is where R2d's "the Varex dark is in `/exchange/bright`" was refuted** |
| ω / mirror | `OmegaStart 180.0`, `OmegaStep −0.25`, `ImTransOpt 2`, y as-is | §2b, three independent arguments — not inherited |
| `SkipFrame` | 1 → 1441 of 1442 frames | re-established on pixels: frame 0 is 0.01 % non-zero vs 2.6–3.7 % |
| `tx` / `Wedge` | −0.1585 / −0.0126 pass 1; composed **−0.2458**; extrapolated ≈ −0.267 | §5h — compose, and the last value is an extrapolation |
| Au reconstruction | 5 grains at confidence 1.000, `DiffPos` 237.4 µm | the two gold cubes; `MinNrPx 4` |
| alumina reconstruction | 1729 grains, `DiffPos` med 655.7 µm — **a mixture, see below** | `HPcat_aluminaRod_att5_e0p1_box100`, 8 rings, `Width 1200` |

**Two numbers from this run that must not be quoted flat:**

* **alumina `DiffPos` 655.7 µm is a mixture statistic**, not a fit quality — the 2.1 % of
  grains at `Confidence` ≥ 0.95 sit at 57.1 µm (DIAGNOSIS *A population `DiffPos` that
  will not come down*).
* **Only the near-axis core is reconstructable.** 1 mm rod, 100 µm beam: zero grains under
  `DiffPos` 150 µm beyond r = 100 µm (DIAGNOSIS `split.illumination_radial`, rule 17).

**Ring selection was by structure factor, not by radius** — the strongest corundum ring is
**8th by radius**, so "the first N rings" would have missed it. Rings used: 8 (2,−1,6) at
100 %, 5 (2,−1,3) 90 %, 2 (1,0,4) 82 %, 13 (3,0,0) 65 %, 1 (1,0,−2) 50 %, 7 (2,0,−4) 48 %,
12 (3,−1,4) 42 %, 3 (2,−1,0) 39 %; `RingThresh 30` on all eight. **Adding more rings was
tried and is exhausted, not untried:** the only two remaining rings clearing intensity,
yield and radial isolation (19 and 29) added 6441 genuine spots and +9 % spots/grain, and
moved the population `DiffPos` by **0.2 µm** (655.7 → 655.5) on identical seeds.

**This layer also carries the documented 0.25° ω zero-point offset** (§3e) — fine for
everything except absolute orientations quoted against an external measurement.

## R3. Current pick-up point

> **Every session updates this section before it ends.** If it is stale, the next session
> re-derives what you already knew.

**Last updated: 2026-08-28.**

**State.** The 20-ID Varex campaign (`bt_20id_jul26b` ti7al, then `nfdev_jul26` Au +
alumina, 2026-08-14 → 08-19) is **closed into the doc set as of today**. Its first half
had already landed; the second half — the ω-sign/mirror procedure (§2b), the per-scan
dark (§3d), the `DiffPos` mixture and the internal-angle censoring (DIAGNOSIS
`resid.population_mixture`), the `Width` band-overlap lever (§6b), and the second 20-ID
reference (§R2e) — landed today, together with three corrections listed under
*Retracted* below.

**Two things are documented as accepted rather than fixed**, both by decision:

1. **The 1.0° internal-angle cap in the matcher stays hardcoded.** It is a design
   constant, not a parameter, and the earlier plan to expose it is dropped. What matters
   downstream is that statistics derived from the per-spot internal angle are *censored*
   on some samples — that check is now in DIAGNOSIS.
2. **The 20-ID 0.25° ω zero-point offset stays.** Documented in §3e with its cost
   (≤ 2.2 µm position at r = 500 µm; a 0.25° rigid rotation of every orientation) and the
   one case that must state it — absolute orientations compared against a measurement
   outside this pipeline. The attribution is still open; the symptom is measured.

**Retracted today**, all three of which this doc set had been asserting:

* "On the 20-ID Varex the dark is in `/exchange/bright`" — one scan's answer promoted to
  a station property. It is per scan; one beamtime held all three cases (§3d).
* "20-ID ω is already negated in the file" — it is not determinable from the file at all
  (§2b).
* "Do not use `Confidence` to find the good grains" — true of a saturated run, false of a
  live one, and the discriminator is now the entry (DIAGNOSIS).

The FF pipeline itself is in a released, self-consistent state **in this repository**,
which is not the same as on the machine you will run on:

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

**Nothing is mid-run.** No jobs on any host belonging to this thread. The `nfdev_jul26`
result trees (~2 GB each) are on `copland:/home/s20a/nfdev_jul26_ff_hs/results/` —
`au_m0267` and `alumina_fix` are the good ones — with `Grains.csv` and
`processgrains_diagnostics.h5` copied back to
`$ANALYSIS/nfdev_jul26_20id_ff/report_*/`. Every 20-ID number in §R2e and in the
DIAGNOSIS mixture entry re-derives from those local copies via
`scripts/verify_for_docs.py` in that folder.

**Left undone from that campaign, deliberately or not:**

1. `uncensored_residual.py` is **broken** — only 9 % of its predictions find a candidate,
   likely a ring-number or ω-window issue. Do not cite anything it produced.
2. A third `grain-tx` pass on `au_m0267` would convert the extrapolated `tx ≈ −0.267`
   into a converged value (§R2e).
3. The campaign's diagnostic scripts are **not upstreamed** — they live in that analysis
   folder, hardcoded to its paths. DIAGNOSIS now prescribes two tests
   (`resid.population_mixture`, `split.illumination_radial`) that have no shipped tool
   behind them, which sits badly with hard rule 14. Generalising `resid_vs_conf.py` and
   `illumination_test.py` into `utils/` is the obvious next increment.
4. Layers 2–13 of ti7al, the other beam-height/overlap sets, and nf709 were never run.
