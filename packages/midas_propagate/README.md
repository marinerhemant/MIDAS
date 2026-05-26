# midas-propagate

Calibration-aware uncertainty propagation and joint re-refinement
for end-to-end HEDM grain analysis.

**Status:** scaffold. See [`dev/paper/SKETCH.md`](dev/paper/SKETCH.md)
for the paper roadmap. No working code yet.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_propagate/notebooks).

## What this package will do

1. Compose the existing differentiable losses across the four
   pipeline stages — calibration (`midas-calibrate-v2`), indexing
   (`midas-index`), per-grain refinement (`midas-fit-grain`), and
   elastic inversion (`midas-stress`) — into a single joint NLL.

2. Compute a joint MAP estimate over any subset of:
   - detector calibration (Lsd, tilts, beam center, distortion)
   - per-grain orientation, lattice strain, position
   - global nuisance parameters (wavelength, beam profile)
   - optionally per-grain elastic stress

3. Assemble the block-structured Hessian at MAP and return
   **calibration-aware per-grain covariance** via Schur-complement
   marginalization — no full-matrix inverse needed.

4. Propagate per-grain covariance through the elastic inversion to
   per-grain stress error bars via the delta method.

## Why

Production HEDM tools (HEXRD, MIDAS, ImageD11, FABLE) report per-grain
σ at the converged grain state with detector calibration **held
fixed**. Downstream Bayesian crystal-plasticity work (Greeley 2026,
Iyer 2025) explicitly assumes an HEDM σ no current tool can derive.
This package closes that loop.

## Companion packages

- `midas-diffract` — differentiable HEDM forward model
- `midas-uq` — single-grain holdout / jackknife / Laplace UQ
- `midas-joint-ff-calibrate` — multi-grain + multi-detector joint refinement
- `midas-calibrate-v2` — instrument calibration
- `midas-stress` — single-crystal elastic inversion
