# Envelope — what has actually been exercised

**Owner:** Hemant Sharma. **Last reviewed 2026-08-21.**

## Tiers — which limits can move, and which cannot

The tier decides what a report may say: a *configured* limit can be proposed as a
change; an *intrinsic* one must be reported as unobtainable rather than tuned at.

| tier | meaning | what falls here |
|---|---|---|
| **Fixed** | the detector and beamline for this cycle | pixel size, panel geometry and module gaps, sentinel conventions per detector class, and the "Numbers that are detector-specific, not constants" sections below — these are properties of the hardware, and a report may not suggest changing them |
| **Configured** | chosen per run, changeable next time | `RhoD` and the distortion order, ring selection, integration binning (`RBinSize`, `EtaBinSize`), `SubPixelLevel`, the calibrant used. **`RhoD` is the distortion normalisation — changing it without refitting `p0..p14` silently corrupts the distortion**, so it is configured but not free |
| **Intrinsic** | no parameter recovers it | the λ–Lsd degeneracy (ring positions alone cannot separate wavelength from distance — see `DIAGNOSIS.md`, `degeneracy.lambda_lsd`), and anything in "Not exercised — stop and ask": an unexercised detector class is not a limit you can tune past, it is one you must go and measure |


The spine reads as procedure. This records how much of it has been *run*, so an
untested path does not get promoted to a recommendation. Written 2026-08-19.

## Exercised end-to-end, on real data

| path | evidence |
|---|---|
| calibrate from scratch, 48-panel Pilatus, CeO2 | five runs, panel parameterisations varied; 66.1 µε held-out; §4 Lab Notebook |
| ring-overlay verification | contours on crests vs 2.6 px inside for the bad block |
| integrate **one file**, v2, `--device cpu` and `cuda` | both run; outputs agree **exactly** (max abs 0) |
| `--v1-out` binaries | correct sizes; 1800×2×8 and 1800×360×8 |
| integrate **one folder**, `midas-integrate-v2-batch --image-glob` | 2 frames → 2 CSVs |
| **stream**, `V2FrameServer` on 60439 | real ceria frame over the C wire protocol; 0.2 s/frame; 3 frames identical |
| v1 one-shot CPU/GPU, v1 server, v1 batch pipeline | run; C parity 2.2e-07 |
| v2 vs v1-with-panels geometry | 9.1e-13 px after the panel fix |

## Not exercised — stop and ask rather than improvise

| path | why it is not covered |
|---|---|
| **one experiment** (several distances or detectors) | never run. `--lsd-offsets` linked-distance mode exists in `midas-calibrate-v2` but was not used here |
| batch on GPU | `midas-integrate-v2-batch` has no `--device`; only the one-shot and server take one |
| batch → v1 binaries | batch writes CSV/HDF5 only; loop the one-shot if the chain needs `lineout.bin` |
| `--mode soft` with a mask | the soft geometry takes no mask argument |
| `midas-integrate-v2-write-map` | never run |
| `--mode soft` with any downstream consumer | differentiable path; no `--v1-out` |
| calibrants other than CeO2 | `LaB6`, `Si`, `Al2O3` are accepted by `make_seed`; untested here |
| non-Pilatus geometry | the procedure should transfer; every *number* in the spine is from one detector |
| the expansion gauge in a real fit | unit-tested only; measured post-hoc, never inside a refinement |

## Numbers that are detector-specific, not constants

Do not carry these to another dataset without re-measuring:

- `per_row_max_entries = 40000` — derived from 1475 × 1679 at `RBinSize 0.5`.
- panel bound ±2 px — from this detector's modules reaching 1.28 / 1.54 px.
- the aliasing verdict in Lab Notebook §4 — **negative on this frame**; the
  published result is positive on a different detector.
- 66.1 µε held-out — a quality *gate* is <100 µε; 66 is not a target.


## Now exercised: single-panel, end to end (2026-08-19)

A second context-free run reduced a **monolithic 2880 x 2880, 150 um, 900 mm**
CeO2 frame — a different detector class from the one these docs were written on.
Single-panel path (no panel block), calibrated from scratch: **held-out strain
4.88 ue**, 16 rings landing at mean -0.021 px / RMS 0.031 px from ideal, ring
overlay on the crests. It built its own mask (none supplied) and measured its
effect. That closes the largest gap in this envelope — but it is one dataset.

## Found by handing this doc set to a context-free model (2026-08-19)

A model with no project context reduced the reference dataset from these docs
alone, on ANL Argo. It reached a calibration passing the gate (62.85 µε
held-out) and produced integrated patterns, and it caught halt condition H1
independently. It also found three defects these docs had missed:

1. **No `--mask` on the v2 one-shot** — 202 550 masked pixels entered the
   profile. Now fixed and required by §5a.
2. **A written paramstest inherited the template's `PanelShiftsFile`**, pointing
   the new geometry at the previous calibration's shifts. Fixed.
3. **`RBinSize`/`EtaBinSize` were not emitted**, so the file needed hand-patching
   before it could be integrated with. Fixed.

It also could not satisfy the §1 install gate from the released env and had to
override `PYTHONPATH` at the canonical tree. **That is now fixed**: the release
landed and the gate — rewritten to probe behaviour rather than version strings
after two of four numeric floors turned out to be wrong in opposite directions —
passes 5/5 on a stock install.

A **second** run, on the single-panel dataset above, found two more:

4. **A template's `SubPixelLevel 5` was copied into the written paramstest**, so
   the calibration output violated the pipeline's own hard rule 1 and had to be
   sed-patched before integrating. Now clamped to 1 with a warning.
5. **`ImTransOpt` was not mentioned anywhere in the doc set**, despite being
   load-bearing and uncheckable after the fact. Now §2.

Five defects across two runs. The rate has not flattened; expect more.


## Now exercised: mixed calibrant, off-panel beam centre (2026-08-19/20)

A third detector class: the **4-panel GE Hydra** at 1-ID (2048², 200 µm,
`tx` 300/30/120/210°), **CeO2 + LaB6 in one exposure** at 80.802 keV, ~2.73 m,
beam centre beyond a panel corner so every ring is a partial arc over 66–73° of
azimuth. Lab Notebook §6bis–§11.

| path | evidence |
|---|---|
| multi-phase ring table, both calibrants, all four panels | 39–42 usable rings/panel; exact hkl degeneracies merged (14 rows on ge1) |
| blend exclusion at 12 px | costs 6–7 rings of ~40, consistently on all four |
| H1 caught on real data | published block 91 mm off for these frames; four panels agree to SD 0.11 mm, ring residual 14 → 0.13 px |
| azimuth-coverage and RhoD gates | both `fail` on the shipped settings, correctly |
| distortion-block selector | `full` / `radial` / `none` compared on one frame; only `none` converged |
| per-ring quality filter | converged in 2 iterations vs a wandering loop without it |
| per-phase residual reporting | 45.6 / 69.0 µε on ge1 |
| per-phase sample position, `mode="same_detector"` | −71.8 ± 34.4 µm, ge1 only |

## Not exercised on that detector — do not promote

| path | why |
|---|---|
| **held-out strain gate (§4 / H3)** | every strain quoted for the Hydra is **full-set**, not held-out. The §4 gate is specifically held-out < 100 µε *and* a small held-out/full gap; that split was never run |
| refits on ge2 / ge3 / ge4 | the census, distance, azimuth and powder-quality numbers cover all four; **every strain and every fitted geometry is ge1 only**. ge3/ge4 have 7/15 railed coefficients against ge1's 3/15, so do not assume they land together |
| per-phase sample position beyond ge1 | one panel, 2.1σ, and degenerate with the lattice constant |
| integration on the Hydra | this work stopped at calibration; nothing was integrated |
| the multi-phase path on a beam-centre-on-panel detector | the whole mixed-calibrant record is from one narrow-wedge geometry, where the harmonics were unidentifiable anyway. What multi-phase does when the azimuth is *not* the binding limit is untested |

**A procedural violation worth recording.** The knob-isolation runs in Lab
Notebook §8 and §10 seeded the beam centre and distance from the published block
in order to hold everything but one knob fixed. That is halt condition **H2**.
The scope is bounded: the 91 mm result and the four-panel agreement come from
`make_seed` and from ring-radius ratios, neither of which used the prior block —
only the strain comparison inherited it. Recorded rather than quietly excused,
because H2 exists precisely because such seeding is easy to justify locally.

## Numbers that are Hydra-specific, not constants

- 66–73° azimuth, and everything that follows from it (rule 11's verdict that
  even `"radial"` diverges) — that is this beam-centre geometry, not a general result.
- 12 px blend cut — from this ring spacing at this distance and energy.
- `MinEtaBinsPerRing` thresholds — absolute counts that scale with `EtaBinSize`.
- LaB6 3.5–4.5× grainier than CeO2 — these two powder lots.

---

## Adversarial eval — the halt conditions, tested (2026-08-19)

The first three runs all used *healthy* data, so the doc set's detection
machinery was almost entirely unexercised. This run was given the single-panel
frame with **two faults planted silently** and no hint that anything was wrong.

| planted fault | caught? | how |
|---|---|---|
| `ImTransOpt 2` → `0` (frame mirrored) | **yes** | tested every candidate transform: 0.163 px RMS for the right one vs 1.091 / 1.634; corroborated by `make_seed` returning the mirrored `BC_z` |
| `Wavelength` +1 % (0.19582 → 0.197778 Å) | **yes** | cross-checked against the filename's 63 keV and identified the λ–`Lsd` degeneracy as the reason it cannot be resolved from this data |

**2 of 2** — the first run where detection, rather than procedure, was measured.

The `ImTransOpt` check was caught *because of* §2, which was written after the
previous run. The wavelength one was **not**: the doc set never mentioned the
λ–`Lsd` degeneracy, and that run recognised it from general knowledge. A weaker
run would have absorbed the 1 % into `Lsd`, passed the strain gate, and reported
a confidently wrong distance. Now hard rule 9, with a DIAGNOSIS entry.

Still untested as *planted faults*: H2 (seeded from a bad block), H4
(unresolvable shifts file), H5 (dLsd free), H7 (non-powder frame). Each needs its
own run.

H1, H9, H10 and H11 have since been met on **real** data rather than planted
ones — a published block 91 mm wrong for its frames, both identifiability gates
failing on the shipped settings, and a loop oscillating between 84 and 4692 µε
(Lab Notebook §6bis–§9). Real-data hits are weaker evidence than a planted fault,
because nobody withheld the answer; they do show the conditions fire outside a
test harness.

H8 (calibrants disagreeing at the floor) is **not** yet met either way: on the one
frame measured, the phases differ 1.51× at 45.6 / 69.0 µε, which is above the
threshold — but the absolute residual is not demonstrably at the floor, so the
gate's precondition is unproven.

## Detector classes: frames read vs calibration run (2026-08-22)

**Reading a frame is not calibrating one** — this table exists so the second is
never inferred from the first. It is now filled in from an archive-scale run:
**252 exposures from 57 beamtimes, 2016–2026**, calibrated end to end with no
human in the loop, each checked back against its own raw frame
(`calibration_table.csv`, ring check = ≥ 60 % of ideal rings matched **and**
scatter ≤ 0.30 px). "Verified" below means that check passed.

| detector | attempted | usable | verified | median ring scatter | span |
|---|---|---|---|---|---|
| GE 2048², single panel (`ge5`) | 84 | 75 | **67** | 0.078 px | 2019–2026, 31 beamtimes |
| Pilatus 2M tiled, `-1`/`-2` sentinels | 34 | 33 | **32** | 0.145 px | 2022–2026, 12 beamtimes |
| GE single panel (`ge3` standalone) | 24 | 21 | 13 | 0.200 px | 2016–2026, 24 beamtimes |
| Varex 2880², single panel (`varexC`) | 15 | 13 | 11 | 0.128 px | 2024–2026, 7 beamtimes |
| GE Hydra quad (`ge1`,`ge2`,`ge4`), BC off panel | 52 | 46 | 17 | ~0.21 px | 2021–2026 |
| Dexela, single panel | 3 | 3 | 3 | 0.072 px | 2019, 1 beamtime |
| **EIGER2 CdTe 16M**, `2**32-1` sentinel | 6 | 6 | **0** | 1.078 px | 2026, 2 beamtimes |
| pixirad (SAXS) | 34 | **0** | 0 | — | halted by the scope gate, correctly |

**Read the quad row as geometry, not quality.** Split by class at identical
median 100 % ring match: single panel 0.091 px scatter and 91 % verified, GE quad
0.286 px and 45 %. The 0.30 px threshold sits on the quad median and halves that
class by construction; it is a single-panel number and should be per-class.

**The EIGER is still the gap that matters** — and it is now a measured gap rather
than an unknown one. Six exposures calibrate, converge, and match 100 % of their
ideal rings, but at **1.078 px** median scatter, so none verifies. Established:
frames read (after the `hdf5plugin` import fix), rings sharp and plentiful,
sentinel 7.102 % mapping the module layout, mask round-trips through `--mask`.
Not established:

- **pixel size.** The file records 62 µm; the measured module geometry says
  EIGER2 16M, i.e. 75 µm. Decided as 75 on the geometry, *not confirmed by the
  beamline*. A 21 % error in `Lsd` rides on it.
- **beam centre off the panel**, so rule 11 and the azimuth gate apply.
- **no dark**, and the bundled `data_dark` / `data_white` are all zeros.

## Also not exercised

| path | why it is not covered |
|---|---|
| an EIGER geometry good enough to quote | 6 calibrate and 0 verify (above). The geometry is recoverable, not yet to spec |
| a per-geometry-class ring-scatter threshold | the need is measured (quad 0.286 px vs single 0.091 px); the replacement threshold is not yet chosen or tested |
| a distortion field reused across beamtimes | tested and **INCONCLUSIVE** — one calibration pins the ge5 field only to ~0.25 px RMS against a 0.39 px field. Lab Notebook §16 |
| recovering λ from the fit | **REFUTED**, do not retry. Lab Notebook §15, rule 9 |
