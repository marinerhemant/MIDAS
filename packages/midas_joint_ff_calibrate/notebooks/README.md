# `midas_joint_ff_calibrate` Notebooks

Hands-on examples for the [`midas_joint_ff_calibrate`](../README.md)
package — **joint powder + FF-HEDM differentiable calibration**.  These
companion the submitted IUCrJ paper.

Every notebook is **fully self-contained**: it reuses the package's own
validated synthetic generators (the source of paper figure 1, in
`runners/run_synthetic_pilatus_joint.py`) at a **reduced problem size**
(4×4 panels, 24 grains, 6 rings) so each runs in seconds on a CPU. No
Zarr files, no real data, no subprocesses, no network. The forward
generators, residual closures, and Fisher diagnostic are the exact
production code paths — only the problem size is shrunk.

## Prerequisites

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
```

## The notebooks

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **00** | [Getting Started: the joint spec](00_getting_started.ipynb) | ~5 s | `build_joint_spec` — combine a powder spec with HEDM grain nuisance blocks; the **canonical parameter naming** (geometry / per-panel / grain blocks); refined-vs-frozen split; pack/unpack round-trip |
| **01** | [Alternating Driver](01_alternating_driver.ipynb) | ~10 s | `AlternatingDriver` — the recommended outer-loop alternation (pass A: geometry + grain orient/pos; pass B: grain strain) on synthetic Pilatus; per-panel shift recovery vs truth |
| **02** | [Full-Joint Driver + Laplace](02_full_joint_laplace.ipynb) | ~10 s | `FullJointDriver(compute_laplace=True)` — refine everything at once + Laplace covariance at MAP; **per-panel σ** map; uncovered panels saturate at the prior |
| **03** | [Fisher Block-Rank Diagnostic](03_fisher_block_rank.ipynb) | ~5 s | The paper headline: powder-only data Fisher is **rank-deficient** on the per-panel block; HEDM evidence makes the joint problem **full-rank**; the nullspace directions |
| **04** | [(L_sd, λ) Gauge Breaking](04_lsd_wavelength_gauge.ipynb) | ~5 s | The 2×2 Fisher on `(Lsd, Wavelength)`: HEDM-only is a near-gauge (cond ~10³⁰⁰); a powder calibrant with known d-spacing breaks it; joint is tightest |

## Recommended reading order

`00 → 03 → 01 → 02 → 04`. Notebook 03 (the rank diagnostic) is the
conceptual core — it explains *why* the joint fit is needed; 01/02 then
*do* the joint fit; 04 covers the wavelength-gauge corollary.

## Running them

```bash
cd /Users/hsharma/opt/MIDAS/packages/midas_joint_ff_calibrate/notebooks
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
list of `(kind, source)` cells maintained as Python strings. A shared
`PREAMBLE` string sets up the shrunk synthetic problem and the
`build_problem` / `build_spec` / `make_closures` helpers used across all
five notebooks.

```bash
python _build.py                    # rebuild everything
python _build.py 03_fisher_block_rank  # rebuild one
```
