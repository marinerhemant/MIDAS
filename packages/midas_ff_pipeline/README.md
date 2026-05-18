# midas-ff-pipeline

End-to-end pure-Python FF-HEDM workflow. Drives all the
``midas-*`` sibling packages so a single command — or a single
notebook — takes raw detector zarr inputs through to ``Grains.csv``
with no C binaries on the path.

Status: **0.2.x** — dataset-density-aware GPU `group_size` resolver
and multi-process CPU sharding (`--cpu-shards`) for the indexer stage;
real-data shakedown on Park22 + Ti-7Al. Single-detector path is fully
exercised; 1-N multi-detector pinwheel layout is supported via the
existing ``midas_diffract.HEDMForwardModel(multi_mode="panel")``
backend. For the unified PF + FF orchestrator see
[`midas-pipeline`](../midas_pipeline/) — `midas-ff-pipeline` and
`midas-pipeline run --scan-mode ff` invoke the same kernels.

## What it does

For each FF-HEDM layer:

| Stage | Sibling package |
|---|---|
| HKL list generation | ``midas-hkls`` |
| Peak finding (per detector) | ``midas-peakfit`` (`peakfit_torch`) |
| Overlap merge | ``midas-peakfit`` (`midas-merge-peaks`) |
| Calc radius | ``midas-peakfit`` (`midas-calc-radius`) |
| Detector transforms | ``midas-transforms`` (`midas-fit-setup`) |
| Cross-detector merge | ``midas-ff-pipeline`` (this package) |
| Binning | ``midas-transforms`` (`midas-bin-data`) |
| Indexing | ``midas-index`` |
| Grain refinement | ``midas-fit-grain`` |
| Grain consolidation + strain | ``midas-process-grains`` |

The cross-detector merge is the only piece this package implements
itself. It concatenates per-detector spot lists and emits a side-car
``Spots_det.bin`` mapping each obs spot to its source detector
(``det_id``), keeping the main ``Spots.bin`` byte-compatible with the
C ``IndexerOMP`` so any cross-pipeline parity test still works.

## Quick start

### CLI

```bash
midas-ff-pipeline run \
    --params /path/to/Parameters.txt \
    --result /path/to/run_dir \
    --layers 1-1 \
    --device cuda --dtype float64
```

### Library API

```python
from midas_ff_pipeline import Pipeline, PipelineConfig

pipe = Pipeline(
    config=PipelineConfig(
        result_dir="/path/to/run_dir",
        params_file="/path/to/Parameters.txt",
        n_cpus=16,
        device="cuda",
        dtype="float64",
    ),
)
pipe.run()
print(pipe.layer_result.n_grains)
```

### Notebook

See ``notebooks/01_smoke_walkthrough.ipynb`` for the simplest
end-to-end demo (synthetic 50-grain Au, single detector).
``02_stage_diagnostics.ipynb`` adds per-stage plots between cells.
``03_multi_detector_demo.ipynb`` runs the four-detector pinwheel
layout against a synthetic ground truth.

## Multi-detector

```python
from midas_ff_pipeline import Pipeline, PipelineConfig, DetectorConfig

detectors = DetectorConfig.load_many("detectors.json")
pipe = Pipeline(
    config=PipelineConfig(
        result_dir="/path/to/multi_det_run",
        params_file="/path/to/Parameters_4det.txt",
        n_cpus=16, device="cuda", dtype="float64",
    ),
    detectors=detectors,
)
pipe.run()
```

`detectors.json`:

```json
[
  {"det_id": 1, "zarr_path": "det1.MIDAS.zip", "lsd": 1000000.0,
   "y_bc": 1024.0, "z_bc": 1024.0, "tx": 0.0, "ty": 0.0, "tz": 0.0,
   "p_distortion": [0,0,0,0,0,0,0,0,0,0,0]},
  {"det_id": 2, "zarr_path": "det2.MIDAS.zip", "lsd": 1005000.0, ...},
  {"det_id": 3, "zarr_path": "det3.MIDAS.zip", "lsd": 1010000.0, ...},
  {"det_id": 4, "zarr_path": "det4.MIDAS.zip", "lsd": 1015000.0, ...}
]
```

If `detectors.json` is absent the pipeline falls back to
``DetParams 1``/``DetParams 2``/... rows in `Parameters.txt`
(same convention as `FitMultipleGrains.c`).

## Resume

Each layer's directory holds a `midas_state.h5` provenance ledger.
``--resume auto`` skips stages whose recorded input/output hashes
match the current files. ``--resume from:indexing`` deletes the
indexing-stage outputs and resumes from there. ``midas-ff-pipeline status``
prints the per-stage state for a given run dir.

## Relationship to `ff_MIDAS.py`

`FF_HEDM/workflows/ff_MIDAS.py` is the historical workflow that
shells out to a mix of C binaries and torch tools. `midas-ff-pipeline`
is its pure-Python successor: same stages, no C binaries, multi-detector
aware out of the box. `ff_MIDAS.py` is left untouched for users still
on the mixed path.
