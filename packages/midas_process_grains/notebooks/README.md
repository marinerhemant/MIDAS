# `midas_process_grains` notebooks

Runnable, CPU-only tutorials. They are generated from `_build.py` (the source
of truth) — edit that, not the `.ipynb` JSON, then rebuild:

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
cd packages/midas_process_grains/notebooks
python _build.py                       # rebuild all
jupyter nbconvert --to notebook --execute --inplace 01_ff_grain_consolidation.ipynb
```

| Notebook | What it covers |
|----------|----------------|
| `01_ff_grain_consolidation.ipynb` | FF grain consolidation in `c_parity` mode. Builds a tiny synthetic run directory (binary schemas from `tests/conftest.py`), runs `run_c_parity_pipeline_from_disk`, and explains the output columns of `Grains.csv` (47), `SpotMatrix.csv` (12), and `GrainIDsKey.csv`. |

All notebooks run on CPU with self-generated synthetic data — no GPU, no
network, no real datasets.
