# midas-pf-odf notebooks

Runnable, CPU-only, synthetic notebooks for `midas-pf-odf` (Phase 1).

| Notebook | What it shows |
|---|---|
| [`01_peakshape_inversion.ipynb`](01_peakshape_inversion.ipynb) | Plant a 3-grain microstructure, simulate per-grain peak patches, recover per-voxel `(R, ε)` with `fit_multi_grain`, and validate per voxel with `recovery_metrics` (misorientation RMS, ε RMSE). Then compares peak-shape inversion against the centroid baseline (`fit_grain_centroid_baseline`) on a strained single grain. |

## Running

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
export KMP_DUPLICATE_LIB_OK=TRUE
cd packages/midas_pf_odf/notebooks
python _build.py
jupyter nbconvert --to notebook --execute --inplace 01_peakshape_inversion.ipynb
```

The notebook reuses the package's own test scaffolding
(`tests/conftest.py`) for the pf-HEDM forward model + scan config, so it
mirrors `tests/test_multi_grain.py` and `tests/test_centroid_vs_peakshape.py`.

The notebooks are generated from `_build.py` (cells as `(kind, source)`
tuples) — edit that file, not the `.ipynb` JSON, then rerun `python _build.py`.

Requires `midas-diffract` (installed in `midas_env`). No GPU, no network,
no real data. Phase 2 (per-voxel ODF) builds on the same forward — see
`dev/RESTART.md`.

The `dev/notebooks/*.py` study scripts (single-grain 50×50 convergence,
shape-vs-centroid 2D) are separate exploratory scripts retained as-is.
