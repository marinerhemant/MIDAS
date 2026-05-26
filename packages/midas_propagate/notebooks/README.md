# midas-propagate notebooks

Runnable, CPU-only, synthetic notebooks for `midas-propagate`.

> **Status.** The package README labels `midas-propagate` a scaffold, but
> its three core numerical modules (`joint_nll`, `schur`, `propagate`) are
> implemented and unit-tested (`tests/` — 19 passing). The notebook below
> exercises them end to end on synthetic data. The remaining work
> (reading a real MIDAS dataset and a measured `Σ_cc` from
> `midas-calibrate-v2`) is tracked in `dev/paper/SKETCH.md`.

| Notebook | What it shows |
|---|---|
| [`01_calibration_aware_covariance.ipynb`](01_calibration_aware_covariance.ipynb) | The paper-1 chain "calibration σ → grain σ → stress σ": (1) `joint_nll.per_grain_hessian_blocks` for the `H_gg`/`H_gc` Hessian blocks of one synthetic FCC grain; (2) `schur.per_grain_schur_marginal` for the calibration-marginalised per-grain covariance (provably wider than the frozen one); (3) `propagate.per_grain_stress_with_cov` for per-grain Cauchy stress + its PSD Voigt covariance via the delta method. |

## Running

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
export KMP_DUPLICATE_LIB_OK=TRUE
cd packages/midas_propagate/notebooks
python _build.py
jupyter nbconvert --to notebook --execute --inplace 01_calibration_aware_covariance.ipynb
```

The notebook mirrors `tests/test_joint_nll.py`, `tests/test_schur.py`,
and `tests/test_propagate.py`.

The notebooks are generated from `_build.py` (cells as `(kind, source)`
tuples) — edit that file, not the `.ipynb` JSON, then rerun `python _build.py`.

Requires `midas-diffract`, `midas-hkls`, and `midas-stress` (installed in
`midas_env`). No GPU, no network, no real data.
