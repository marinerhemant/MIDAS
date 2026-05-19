# `midas_fit_grain` notebooks

Runnable, CPU-only tutorials. They are generated from `_build.py` (the source
of truth) — edit that, not the `.ipynb` JSON, then rebuild:

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
cd packages/midas_fit_grain/notebooks
python _build.py                       # rebuild all
jupyter nbconvert --to notebook --execute --inplace 01_single_grain_refinement.ipynb
```

| Notebook | What it covers |
|----------|----------------|
| `01_single_grain_refinement.ipynb` | Single-grain refinement quickstart. Builds a synthetic single-grain fixture (reusing `tests/_synthetic.py`) and walks every **solver** (`lbfgs`/`adam`), **loss** (`pixel`/`angular`/`internal_angle`), and **mode** (`iterative`/`all_at_once`). |

All notebooks run on CPU in float64 with self-generated synthetic data — no
GPU, no network, no real datasets.
