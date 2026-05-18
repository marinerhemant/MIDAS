"""User-facing entry points.

Each pipeline composes the forward model + a loss + an inference backend.
"""
from .single import autocalibrate
from .single_pv import autocalibrate_pv
from .single_pv_2d import autocalibrate_pv_2d
from .multi import autocalibrate_multi
from .bayesian import autocalibrate_bayesian
from .nn_residual import autocalibrate_nn
from .joint_cake import autocalibrate_joint
from .four_stage import autocalibrate_four_stage
from .downstream import sensitivity_diagnostic, joint_with_downstream
from .robust import autocalibrate_robust, RobustCalibrationDiagnostics
from .bic_search import select_basis_bic, BasisFit
from .first_time import first_time_calibrate, FirstTimeResult
from .fisher_prune import auto_select_basis, PruneReport
from . import diagnostics

__all__ = [
    "autocalibrate", "autocalibrate_pv", "autocalibrate_pv_2d",
    "autocalibrate_multi",
    "autocalibrate_bayesian", "autocalibrate_nn", "autocalibrate_joint",
    "autocalibrate_four_stage",
    "sensitivity_diagnostic", "joint_with_downstream",
    "autocalibrate_robust", "RobustCalibrationDiagnostics",
    "select_basis_bic", "BasisFit",
    "first_time_calibrate", "FirstTimeResult",
    "auto_select_basis", "PruneReport",
    "diagnostics",
]
