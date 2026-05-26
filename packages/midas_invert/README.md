# midas-invert

Domain-agnostic **differentiable-inversion primitives** for MIDAS, shared across
HEDM, Laue microdiffraction, pf-/grain-ODF, and 2D/ultrafast diffraction. None of
it knows about diffraction — you supply the forward model and the loss closure.

- **`optimize`** — `fit` (Adam / L-BFGS), `relative_l2_loss` (scale-robust),
  `cosine_loss` (scale-invariant shape loss; avoids the argmax kink of
  peak-normalisation — right for rocking curves, fringe profiles, spectra).
- **`uq`** — `laplace_uncertainty` (Hessian-at-optimum covariance & std-devs).
- **`design`** — `fisher_information`, `rank_measurements`, `next_best_measurement`
  (which delay / reflection / energy best constrains a target — experiment design).
- **`mixture`** — `mixture_deconvolution` (recover non-negative softmax weights
  over a component grid: thickness/grain-size distributions, ODF on a grid, a
  spectrum over an energy grid).
- **`surrogate`** — `ParameterMLP`, `train_surrogate` (amortised inference; the
  differentiable forward is the data generator).

All torch-differentiable, CPU / CUDA / MPS. Extracted from `midas_2d` so HEDM,
Laue, and 2D draw on one implementation (see `../HEDM_LAUE_TRANSFER_PLAN.md`).

```bash
pip install -e . --no-deps
KMP_DUPLICATE_LIB_OK=TRUE pytest
```
