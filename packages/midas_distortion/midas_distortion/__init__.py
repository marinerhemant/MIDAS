"""midas_distortion — canonical MIDAS radial-distortion model (single source).

Shared by midas_calibrate_v2 (calibration), midas_peakfit (numpy spot geometry)
and midas_transforms (torch spot geometry) so all three evaluate one definition
of the distortion layout + the v1↔v2 coefficient mapping.
"""
from .core import (
    HarmonicTerm,
    P_COEF_NAMES,
    PHASE_NAMES,
    ISO_NAMES,
    AMP_NAMES,
    v1_term_layout,
    v2_term_layout,
    extended_term_layout,
    extended_p_coef_names,
    V1_TO_V2_DISTORTION,
    V2_TO_V1_DISTORTION,
    V2_TO_V1_PNAME,
    v1_to_v2_coeffs,
    v2_to_v1_coeffs,
    v2_coeffs_from_named,
    distortion_factor,
    apply_distortion,
)
from .rhod import (
    detector_max_corner_dist_um,
    check_rho_d_um,
    resolve_rho_d_um,
    resolve_rho_d_um_warn,
)

__version__ = "0.2.0"

__all__ = [
    "HarmonicTerm",
    "P_COEF_NAMES",
    "PHASE_NAMES",
    "ISO_NAMES",
    "AMP_NAMES",
    "v1_term_layout",
    "v2_term_layout",
    "extended_term_layout",
    "extended_p_coef_names",
    "V1_TO_V2_DISTORTION",
    "V2_TO_V1_DISTORTION",
    "V2_TO_V1_PNAME",
    "v1_to_v2_coeffs",
    "v2_to_v1_coeffs",
    "v2_coeffs_from_named",
    "distortion_factor",
    "apply_distortion",
    "detector_max_corner_dist_um",
    "check_rho_d_um",
    "resolve_rho_d_um",
    "resolve_rho_d_um_warn",
]
