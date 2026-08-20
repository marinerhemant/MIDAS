# Envelope — what has actually been exercised

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

Still untested: H2 (seeded from a bad block), H4 (unresolvable shifts file),
H5 (dLsd free), H7 (non-powder frame). Each needs its own planted-fault run.
