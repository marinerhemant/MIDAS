# `midas_peakfit` notebooks

Runnable, CPU-only tutorials. They are generated from `_build.py` (the source
of truth) — edit that, not the `.ipynb` JSON, then rebuild:

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
cd packages/midas_peakfit/notebooks
python _build.py                       # rebuild all
jupyter nbconvert --to notebook --execute --inplace 01_batched_lm_peakfit.ipynb
```

| Notebook | What it covers |
|----------|----------------|
| `01_batched_lm_peakfit.ipynb` | Standalone batched Levenberg-Marquardt peak fitting. Builds a tiny synthetic `*.MIDAS.zip` (schema from `tests/conftest.py`), runs `midas_peakfit.orchestrator.run` on CPU/fp64, shows the `device`/`dtype` knobs, and reads back the `AllPeaks_PS.bin` / `AllPeaks_PX.bin` outputs. |

All notebooks run on CPU with self-generated synthetic data — no GPU, no
network, no real datasets.
