# Phase 2 — Configure: params, the zip trap, rings, the FF seed

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).

## 2.1 The parameter file

Start from the calibration `paramstest` (phase 1). PF-specific keys are in
[`PF_PARAMETERS.md`](PF_PARAMETERS.md). The ones that most often go wrong:

- `RingNumbers` — which rings to index/refine on. For **strain**, prefer the bright,
  fully-on-detector rings; the rings used for indexing are not always the best for strain
  (a low-order bright ring beats a corner-clipped high-order one).
- `Hbeam` / `BeamThickness` — the **true per-layer beam** (hard rule).
- `BeamSize` — the in-plane beam width; sets the voxel/scan overlap tolerance.
- `SpaceGroup`, `LatticeParameter` — the phase.
- `OmegaRange` — one or more valid ω spans; **gaps** (blocked ranges) are normal and reduce
  the reflections per voxel (envelope §2).

## 2.2 THE ZIP TRAP — analysis params are baked into the zarr

**This is the highest-value trap in pf-HEDM.** `zip_convert` writes the raw frames into a
per-scan zarr (`*.MIDAS.zip`) **together with the analysis parameters in force at
zip-time**. Peakfit (and other stages) then read the parameters **from the zip's stored
`analysis_parameters`, not from your live `paramstest`.**

Consequence: editing `paramstest` after the zips exist changes **nothing**. A wrong
`MaxNPeaks` (peak cap per spot), a wrong threshold, a wrong ring set — all silently persist
from whatever was in force when the zips were written.

**To change any zip-baked parameter you must regenerate the zips.** Confirm the value that
actually reached a zip:

```python
import zarr
z = zarr.open("<one *.MIDAS.zip>", "r")
print(dict(z["analysis_parameters"].attrs) if "analysis_parameters" in z else "check the zip layout")
```

Adopt the **authoritative** parameter file verbatim (do not paste keys from a different
layer's file — a stray `MaxNPeaks 8` from a copy-paste caps every spot at 8 peaks), then
**delete and regenerate all zips**. Verify one regenerated zip before running the full set.

## 2.3 Peakfit on GPU

Scanning peakfit is the compute floor — one core per scan is days. Run it on GPU:

- `--device cuda`, `dtype float32`, shard across the available GPUs.
- Watch for the **dense-frame failure mode**: extremely populated frames (a box-beam FF or a
  noisy scan) can overflow the Triton Jacobian's address arithmetic on GPU. This is fixed in
  current `midas-peakfit`; on an older version the symptom is a CUDA *illegal memory access*
  mid-flush. If you hit it, update the package or fall back to `--device cpu` with the
  thread count set to the core count (a standalone CPU peakfit must set its own
  `OMP/MKL/OPENBLAS_NUM_THREADS`, not inherit `=1` from a GPU-worker script).

An intensity floor on the merged spot list (drop noise-level spots) both speeds indexing and
avoids OOM; record the cut you used.

## 2.4 Obtain the FF orientation seed — you will need it

**Unseeded scanning indexing is intractable** (halt condition). A scanning layer produces
tens of millions of merged spots; combing the full orientation grid blind runs for days and
often writes no output. The tractable path is to seed indexing with far-field orientations.

1. From the matched FF layer (phase 1.4), get the far-field `Grains.csv` (the standard
   `ProcessGrains` output: `%ID O11..O33 X Y Z a b c … Confidence …`).
2. Point the seeding stage at it: `SeedingConfig.mode = "ff"`,
   `grains_file = <FF Grains.csv>`. The seeding stage converts it to
   `UniqueOrientations.csv` (14-col: grainID + 4 pad + 9 OM), which the per-voxel scanning
   indexer uses as per-voxel candidate seeds instead of the full grid.
3. The **c-omp** indexer additionally needs a `GrainsFile <path>` line in its paramstest to
   take the seeded path (`isGrainsInput=1`); `midas_pipeline` wires this. Without it the C
   binary silently combs the full grid — indistinguishable from unseeded, and just as slow.

> ⚠️ `Grains.csv` (ID + 9 OM in cols 1–9) is **not** `UniqueOrientations.csv` (OM in cols
> 5–13). The seeding handoff keys header detection off the OM column names and accepts both
> `GrainID` and `ID` spellings.

Seeded, a full scanning layer indexes in tens of minutes instead of days.

When params are authoritative, zips regenerated, and the FF seed staged, go to
[`phase-3-run.md`](phase-3-run.md).
