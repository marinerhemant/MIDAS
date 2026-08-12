# pf-HEDM runbook — operational state

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).
>
> **This is the volatile document.** The handbook is procedure and changes slowly; the
> notebook only grows. This file describes *right now*. **Update §R3 before you finish.**

## R1. Where it runs

Full paths, because conda is not on the non-interactive ssh PATH.

| | |
|---|---|
| environment | the **shared MIDAS env** — call its python by full path (`.../envs/midas/bin/python`); conda is not on the ssh PATH |
| install host | the one host with internet + a shared home — install there, it is visible everywhere |
| GPU hosts | multi-GPU compute nodes; pick a GPU by **utilisation**, not free memory. GPU prefix `CUDA_DEVICE_ORDER=PCI_BUS_ID KMP_DUPLICATE_LIB_OK=TRUE` |
| long jobs | `setsid`/`nohup` + redirect to a log, or SIGHUP kills them on hangup. **Arm a monitor with an error/death branch on every relaunch** — a job that dies at launch (e.g. a missing `cd`) sits dead unnoticed otherwise |
| outputs | the run's own `LayerNr_<N>/` tree — **never `/tmp`** |

**Before any run, pass the §0 install gate and paste its output.** Every number below is
conditional on it, and on the c-omp→pf-odf bridge being present.

## R2. What healthy looks like

**There is no single number for "healthy".** Each row carries the conditions it was measured
under; outside those it is not a specification.

| quantity | value | measured on / conditions |
|---|---|---|
| seeded scanning indexing | tens of minutes | FF-seeded c-omp indexer, ~10⁷-spot layer; **unseeded is days / no output** |
| c-omp PF refine | ~1–2 min / layer | c-omp refiner, ~67 k voxels; python refine is multi-day (GIL-bound) |
| find_grains (well-behaved map) | minutes | GPU dedup enabled; **hours / never on a high-spread or cracked map** — fast-path it |
| per-voxel strain fit | seconds–minutes / grain | GPU, adam ~60 steps; big grains need `chunk_size_g` |
| full-layer strain extraction | I/O-bound, hours | reading the raw frames dominates; zarr ~1.6–2× faster than raw h5 |
| c-omp vs python refine agreement | <0.2° miso, <0.01 Å lattice | cross-check on a few voxels |

### R2c. Ranges that are NOT thresholds

- **Per-voxel strain magnitude** is provisional on a signal-limited (high-attenuation) scan —
  do not quote it to more precision than the SNR supports. Report the *pattern* (Moran's I,
  localisation) instead.
- **Completeness** on a heavily-attenuated scan can sit at a median well below 1 and still be
  the best obtainable — it is not a defect threshold.
- **Grain-segmentation grain count** depends on the misorientation tolerance and the
  segmentation path (`find_grains` vs fast-path); it is not a fixed property of the sample.

## R3. Current pick-up point

> **Every session updates this before it ends.** A stale pick-up point is worse than none.

**Last updated: 2026-08-12.**

**State.** Doc set created from the reference campaign (Handbook + phases 0–5 + parameters +
diagnosis + envelope + notebook). The c-omp→pf-odf bridge (`midas_fit_grain.scan_seed` +
`fitbest_adapter`, wired in `midas_pipeline.stages.refinement`) and pf-odf opt-in dark
subtraction + zarr frame reader are in the packages. Verified end-to-end on the reference
layer.

**Open, not blocking:**
1. Cross-modal (tomo) in-plane registration convention (flip + rotation-centre) — not yet
   recorded; only Z is registered (notebook §5).
2. Illumination-gated extraction remains unvalidated / not shipped (notebook §5).

**Mid-run:** nothing.
