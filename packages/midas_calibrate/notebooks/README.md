# `midas_calibrate` Notebooks

Hands-on examples for the [`midas_calibrate`](../README.md) package —
the production native-Python/Torch reference engine for MIDAS detector
geometry calibration (the drop-in replacement for the C chain
`AutoCalibrateZarr → CalibrantIntegratorOMP → CalibrationCore`).

Both notebooks are **fully self-contained**: they render a synthetic
CeO₂ ring image with the same forward model the package's own
end-to-end test (`tests/test_e2e_synthetic.py`) uses. No data files, no
network, CPU only.

## Prerequisites

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
```

## The notebooks

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **00** | [Getting Started](00_getting_started.ipynb) | ~40 s | Render a synthetic CeO₂ image → run the full E↔M `autocalibrate` engine → inspect per-iteration convergence → verify recovery vs truth → write a C-compatible refined paramstest |
| **01** | [v1 vs v2 Comparison](01_v1_vs_v2_comparison.ipynb) | ~40 s | The same synthetic image through `midas_calibrate.autocalibrate` (v1) and `midas_calibrate_v2.autocalibrate_pv`; compare strain, geometry recovery, and runtime; when to reach for which engine |

## Running them

```bash
cd /Users/hsharma/opt/MIDAS/packages/midas_calibrate/notebooks
jupyter lab 00_getting_started.ipynb
```

Or batch-execute:

```bash
for nb in *.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb" \
        --ExecutePreprocessor.timeout=600
done
```

## How the notebook source is organised

The `.ipynb` files are **build artefacts**, not source.
[`_build.py`](_build.py) is the source of truth — each notebook is a
list of `(kind, source)` cells maintained as Python strings, which
makes them diffable and reviewable.

```bash
python _build.py                    # rebuild everything
python _build.py 00_getting_started  # rebuild one
```
