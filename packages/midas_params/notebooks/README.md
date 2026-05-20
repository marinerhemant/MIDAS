# `midas_params` Notebooks

Hands-on, end-to-end runnable examples for the
[`midas_params`](../README.md) package — the parameter-file registry,
validator, and wizard for the MIDAS FF/NF/PF/RI pipelines. Every
notebook builds its own **synthetic** sample files and runs the
actual production code paths. No network, no external data.

## Prerequisites

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
```

## Notebooks

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **01** | [Quickstart](01_quickstart.ipynb) | ~2 s | The full tour: write a param file → `validate` + `format_report` → catch a broken file with cross-field rules → inspect the registry → enumerate Bragg `rings` → `discover` from filename + calibration file → build a file with the non-interactive `wizard`. |

## Running them

```bash
cd /Users/hsharma/opt/MIDAS/packages/midas_params/notebooks
jupyter lab 01_quickstart.ipynb
# or batch-execute
jupyter nbconvert --to notebook --execute --inplace 01_quickstart.ipynb
```

## How the notebook source is organised

The `.ipynb` files are **build artefacts**, not source.
[`_build.py`](_build.py) is the source of truth — each notebook is a
list of `(kind, source)` cells. Rebuild after editing:

```bash
python _build.py                # rebuild all
python _build.py 01_quickstart  # rebuild one
```

## See also

- The same operations from the CLI: `midas-params
  validate|inspect|rings|wizard|diagnose`.
- [manuals/FF_Parameters_Reference.md](../../../manuals/FF_Parameters_Reference.md)
