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
| single-panel detector end-to-end | the code path is the multi-panel one minus the panel block; not run start to finish |
| non-Pilatus geometry | the procedure should transfer; every *number* in the spine is from one detector |
| the expansion gauge in a real fit | unit-tested only; measured post-hoc, never inside a refinement |

## Numbers that are detector-specific, not constants

Do not carry these to another dataset without re-measuring:

- `per_row_max_entries = 40000` — derived from 1475 × 1679 at `RBinSize 0.5`.
- panel bound ±2 px — from this detector's modules reaching 1.28 / 1.54 px.
- the aliasing verdict in Lab Notebook §4 — **negative on this frame**; the
  published result is positive on a different detector.
- 66.1 µε held-out — a quality *gate* is <100 µε; 66 is not a target.


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
override `PYTHONPATH` at the canonical tree. **Until the queued release lands,
the gate cannot pass on a stock install** — a real novice would stop there.
