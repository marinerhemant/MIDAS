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

# Register the HDF5 filter plugins (bitshuffle, LZ4, zstd) that Eiger / Dectris
# and ESRF files are written with.  hdf5plugin ships the plugin binaries but
# only registers them with the HDF5 library when it is imported — declaring it
# as a dependency is not enough.  Without this, reading such a dataset fails
# with "can't open directory (/usr/local/lib/plugin)".
try:  # pragma: no cover - environment-dependent
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # HDF5 files with no plugin filter still read fine
    pass

__version__ = "0.10.0"

# One-shot fully-automated entry point: image + wavelength -> calibration.
from .pipelines.auto import calibrate, AutoCalibrationResult, CALIBRANTS

__all__ = ["__version__", "calibrate", "AutoCalibrationResult", "CALIBRANTS"]
