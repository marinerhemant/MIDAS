# `midas_zipper` Notebooks

Hands-on, runnable examples for the
[`midas_zipper`](../README.md) package — standalone Zarr-zip
generation for MIDAS FF/PF workflows. The notebook synthesizes its
own raw data and runs the real production code paths. No network, no
external data.

## Prerequisites

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
```

## Notebooks

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **01** | [Raw frames → `*.MIDAS.zip`](01_raw_to_zarr.ipynb) | ~5 s | Synthesize a multi-frame TIFF stack + a minimal param file → `generate_ff_zip()` → inspect the Zarr keys (`exchange/data`, `analysis/process/analysis_parameters`) inside the resulting `.MIDAS.zip` → patch a metadata key in place with `midas-update-zarr`. |

## Running them

```bash
cd /Users/hsharma/opt/MIDAS/packages/midas_zipper/notebooks
jupyter lab 01_raw_to_zarr.ipynb
# or batch-execute
jupyter nbconvert --to notebook --execute --inplace 01_raw_to_zarr.ipynb
```

## How the notebook source is organised

The `.ipynb` files are **build artefacts**, not source.
[`_build.py`](_build.py) is the source of truth. Rebuild after
editing:

```bash
python _build.py                 # rebuild all
python _build.py 01_raw_to_zarr  # rebuild one
```

## See also

- CLI equivalents: `midas-ff-zip -resultFolder … -paramFN … -LayerNr …`
  and `midas-update-zarr -fn … -keyToUpdate … -updatedValue …`.
- HDF5, GE, and CBF inputs are auto-detected from the data file
  extension; this notebook uses the TIFF path.
