"""midas-calibrate-v2 — fully differentiable detector calibration.

Coexists with midas-calibrate v1.  v1 stays as the C-backed reference
implementation; v2 is the research / advanced engine with multi-image,
Bayesian, NN-residual, and joint-forward-cake capabilities.

Primary entry points (subject to v0 churn):

- :func:`midas_calibrate_v2.pipelines.single.autocalibrate` — drop-in
  replacement for v1's autocalibrate.
- :func:`midas_calibrate_v2.pipelines.multi.autocalibrate_multi` — joint
  calibration over multiple images / distances.
- :func:`midas_calibrate_v2.pipelines.bayesian.autocalibrate_bayesian` —
  MAP + Laplace, VI, or NUTS posteriors.
- :func:`midas_calibrate_v2.pipelines.nn_residual.autocalibrate_nn` — train a
  conv NN ΔR residual on top of the analytical model.
- :func:`midas_calibrate_v2.pipelines.joint_cake.autocalibrate_joint` — joint
  forward-cake engine.
"""
from __future__ import annotations

__version__ = "0.3.1"

# One-shot fully-automated entry point: image + wavelength -> calibration.
from .pipelines.auto import calibrate, AutoCalibrationResult, CALIBRANTS

__all__ = ["__version__", "calibrate", "AutoCalibrationResult", "CALIBRANTS"]
