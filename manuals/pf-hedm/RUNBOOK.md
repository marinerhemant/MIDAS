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

## R4. Running a MULTI-LAYER campaign

Everything above (and every phase file) describes **one layer**. A campaign of
tens of layers has its own failure modes, and they are not the layer's.

### R4a. Profile PER STAGE — the pipeline is not one workload

The single most expensive wrong turn available here is to measure one stage,
conclude something about "the pipeline", and re-plan the whole campaign on it.
Measured on a 64-core host, the halves of a PF layer want **opposite** resources:

| half | stages | bound by | uses |
|---|---|---|---|
| **prep** | `zip_convert → peakfit → transforms → binning` | **disk** | ~4 of 64 cores, whatever you give it |
| **index** | `indexing → refinement → find_grains` | **CPU** | saturates 63.5 of 64 |

Seeing peakfit peg 4 cores and concluding "the pipeline underuses the box", then
running 3 whole layers at 20 cores each, was measured **worse than serial** — it
starved the only stage that could use cores and still left the machine ~48 %
idle. Indexing timings on one layer, varying only `numProcs` (byte-identical
output every time): **64 → 27.9 min, 32 → 40.9 min, ~20 → 51.6–68.1 min.**

### R4b. Overlap the halves, do not split the cores

Because the halves are bound by different resources **and take about the same
wall time**, run them as a two-stage software pipeline: `prep(N+1)` concurrent
with `index(N)`, exactly one of each live. Wall time per layer becomes
`max(prep, index)` (~28 min) instead of the sum (~50–55 min), and indexing still
gets the whole box.

Both halves already exist in the CLI — no new code path:

```bash
# prep half
midas-pipeline run ... --skip indexing --skip refinement --skip find_grains \
                       --skip voxel_cleanup --skip sinogen ... 
# index half (same layer dir)
midas-pipeline run ... --resume from --from indexing
```

Prep hands over `Data.bin`, `Spots.bin`, `nData.bin`, `hkls.csv`,
`paramstest.txt`. **The index half never reads the zips**, so free them at the
end of *prep*: only one layer's ~26 GB is then ever live, and they rebuild from
raw in ~100 s if you need them back.

### R4c. Campaign invariants — check these before you launch, not after

| invariant | why | how to check |
|---|---|---|
| **One binary for the whole campaign** | a mid-run reinstall silently splits the output format (e.g. 27→33 col `OrientPosFit`, 39→45 col `Result_OrientPos_voxel`); `summary.csv` still looks uniform | record the binary mtime/version at launch; never install into the env a campaign is running from |
| **The env is verified FUNCTIONALLY, not by version** | a bumped version can ship stale code | call each new API and check the answer before launching |
| **One pinned reference cell per SAMPLE** | different samples can be at genuinely different states (measured: two NMC811 cathodes discharged at c/a 4.94, two delithiated at 5.07 — a 10 000–15 000 µε difference) | phase-2 §2.5; a global cell would inject more error than it removes |
| **Resume is by artefact, not by flag** | a half-finished layer dir makes `transforms` cache off stale `InputAllExtraInfoFittingAll*.csv` and index the previous attempt's spots, exit 0 | treat "has `Results/`" as done, "has binned inputs" as prepped, **anything else: wipe the dir** before re-running |
| **Disk budget is per concurrent layer, not per campaign** | zips are ~2 GB/scan | free them at end of prep (R4b) and the campaign runs flat |

### R4d. Scouting the reference cell is cheap — it needs no indexing

Pinning a sample's cell (phase-2 §2.5) uses `Ttheta` + `RingNumber` from
`InputAllExtraInfoFittingAll*.csv`, which the **prep half alone** produces. So
scout one layer per sample with the index stages skipped — minutes, not the full
layer cost — pin each cell, then launch the campaign. Do not launch a long
campaign on an unpinned cell and plan to "fix the strain later": the reference is
baked into every strain number *and* it changes completeness (measured
0.618 → 0.833).
