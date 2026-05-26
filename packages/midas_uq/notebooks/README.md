# midas-uq notebooks

Runnable, CPU-only, synthetic notebooks for `midas-uq`.

| Notebook | What it shows |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Plant one synthetic FCC grain, then run all four UQ diagnostics on it: `half_half` (K-split reproducibility), `jackknife` (per-spot influence), `laplace_covariance` (Hessian baseline), `rfree_gap` (overfitting). Also demonstrates `ff`/`pf`/`nf` mode dispatch. |

## Running

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
export KMP_DUPLICATE_LIB_OK=TRUE
cd packages/midas_uq/notebooks
python _build.py                     # (re)generate .ipynb from _build.py
jupyter nbconvert --to notebook --execute --inplace 01_quickstart.ipynb
```

The notebooks are generated from `_build.py` (cells as `(kind, source)`
tuples) — edit that file, not the `.ipynb` JSON, then rerun `python _build.py`.

Requires `midas-diffract` and `midas-hkls` (installed in `midas_env`).
No GPU, no network, no real data.
