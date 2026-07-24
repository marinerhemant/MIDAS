"""midas_joint_ff_calibrate — joint powder + FF-HEDM differentiable
calibration.

Pure-Python, fully autograd-traced.  Composes three existing differentiable
packages over a single ``midas_peakfit.ParameterSpec``:

  - ``midas_calibrate_v2.loss.pseudo_strain`` for the powder-calibrant residual
  - ``midas_fit_grain.spec_residual.hedm_spot_residual`` for the HEDM residual
  - ``midas_peakfit`` for spec / pack / LM / Laplace / TPSpline / Σ=0 gauge

The package's job is *only* to wire these together — it adds:

  - :func:`build_joint_spec` — combines a powder spec with HEDM grain
    nuisance blocks under canonical names.
  - :func:`joint_residual` — concatenates per-modality residuals with
    user-chosen weights, plus the gauge / prior rows.
  - :class:`pipelines.alternating.AlternatingDriver` — the recommended
    default per the implementation plan: outer-loop alternation between
    (geometry + grain orientation/position) and (grain strain) passes.
  - :class:`pipelines.full_joint.FullJointDriver` — full joint refinement
    of every parameter at once with Laplace covariance at MAP.
  - :func:`pipelines.identifiability.fisher_block_rank` — quantifies how
    much HEDM evidence breaks the per-panel rank-1 degeneracy of the
    powder image (the headline figure of the paper).

The user picks which subset of the spec to refine — anything from a single
panel block to the entire 260-DOF Pilatus geometry plus N_g × 12 grain
parameters.  No baked-in assumption that only panel shifts are interesting.
"""

__version__ = "0.1.6"

from midas_joint_ff_calibrate.spec import build_joint_spec
from midas_joint_ff_calibrate.loss import joint_residual

__all__ = [
    "__version__",
    "build_joint_spec",
    "joint_residual",
]
