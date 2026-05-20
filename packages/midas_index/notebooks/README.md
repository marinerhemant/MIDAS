# `midas_index` Notebooks

Hands-on, runnable examples for the
[`midas_index`](../README.md) package — the pure-Python / PyTorch
FF/PF-HEDM indexer (drop-in for C `IndexerOMP` / `IndexerScanningOMP`).
Every notebook runs the real code paths on **CPU** with the small
synthetic dataset shipped in the package's tests, or with
self-generated in-memory data. No network, no external data, no GPU
required.

## Prerequisites

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
```

The `c-omp` backend is an optional accelerator (bundled `midas_indexer`
binary). The notebooks probe for it with `backend_c.available()` but
run the portable **Python** backend regardless.

## Notebooks

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **01** | [FF indexing](01_ff_indexing.ipynb) | ~5 s | `Indexer.from_param_file(...).load_observations(...).run()` on a small synthetic FF dataset (5 Cu grains, 4 rings); read out per-seed orientation, position, and match statistics. |
| **02** | [Soft beam attribution](02_soft_attribution.ipynb) | ~5 s | The three attribution kernels (`hard_window_fn`, `soft_top_hat_fn`, `soft_gaussian_fn`), their autograd-differentiability, and how the per-match weights flow into `compare_spots` scoring (the values written to `IndexBest_weights_all.bin`). |
| **03** | [Scanning / PF indexing](03_scanning_pf.ipynb) | ~5 s | `Indexer.run_scanning(scan_positions, out_path=…)` over a synthetic 3×3 voxel grid; read the consolidated `IndexBest_all.bin` back with `read_index_best_all`. |

## Running them

```bash
cd /Users/hsharma/opt/MIDAS/packages/midas_index/notebooks
jupyter lab 01_ff_indexing.ipynb
# or batch-execute all three
for nb in 0*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

## Device portability

Everything runs on CPU here (this machine has no GPU). On a CUDA / MPS
box, pass `device="cuda"` (or `"mps"`) to `Indexer.from_param_file` /
`Indexer(...)` — identical code path, no `.cu` sources.

## How the notebook source is organised

The `.ipynb` files are **build artefacts**, not source.
[`_build.py`](_build.py) is the source of truth. Rebuild after
editing:

```bash
python _build.py                 # rebuild all
python _build.py 01_ff_indexing  # rebuild one
```

## See also

- CLI: `midas-index paramstest.txt 0 1 1000 8`.
- Pin device/dtype via `MIDAS_INDEX_DEVICE` / `MIDAS_INDEX_DTYPE`.
