# `midas_transforms` Notebooks

Hands-on, runnable examples for the
[`midas_transforms`](../README.md) package — the pure-Python /
PyTorch port of the four FF-HEDM transforms that sit between
peak-fitting and indexing (`MergeOverlappingPeaksAllZarr`,
`CalcRadiusAllZarr`, `FitSetupZarr`, `SaveBinData`). Every notebook
builds its own **synthetic** data and runs the real code paths on
**CPU**. No network, no external data, no GPU required.

## Prerequisites

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
```

## Notebooks

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **01** | [Per-stage walkthrough](01_per_stage.ipynb) | ~10 s | Run the four stages one at a time — `merge_overlapping_peaks` → `calc_radius` → `fit_setup` → `bin_data` — threading each stage's in-memory array into the next, on a synthetic FF dataset (3 rings × 4 η × 4 ω frames). Ends with the four indexer-ready `*.bin` files. |
| **02** | [`Pipeline.from_zarr` end-to-end](02_pipeline_from_zarr.ipynb) | ~10 s | Build a synthetic `*.MIDAS.zip` Zarr + `AllPeaks_PS.bin` peak blob, then run all four stages as one chained `Pipeline.from_zarr(...).run()` (intermediates stay on-device) and `.dump()` the indexer handoff files. |

## Running them

```bash
cd /Users/hsharma/opt/MIDAS/packages/midas_transforms/notebooks
jupyter lab 01_per_stage.ipynb
# or batch-execute both
for nb in 0*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

## Device portability

Everything runs on CPU here (this machine has no GPU). On a CUDA / MPS
box, pass `device="cuda"` (or `"mps"`) to the stage functions or to
`Pipeline.from_zarr` — identical code path, no `.cu` sources.

## How the notebook source is organised

The `.ipynb` files are **build artefacts**, not source.
[`_build.py`](_build.py) is the source of truth (both notebooks share
an embedded synthetic-data helper). Rebuild after editing:

```bash
python _build.py               # rebuild all
python _build.py 01_per_stage  # rebuild one
```

## See also

- Per-stage CLIs: `midas-merge-peaks`, `midas-calc-radius`,
  `midas-fit-setup`, `midas-bin-data`.
- End-to-end CLI: `midas-transforms pipeline scan.zip --out-dir …`.
- Downstream: [`midas-index`](../../midas_index/) consumes the
  `Spots.bin` / `Data.bin` / `nData.bin` these notebooks produce.
