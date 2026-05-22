"""Geometry sanity checks — re-export of the shared RhoD-units guard.

The RhoD unit logic now lives in the single-source :mod:`midas_distortion.rhod`
so calibrate-v2, integrate-v2 and integrate all share one definition. This
module re-exports it for backward compatibility with existing
``from ..forward.sanity import resolve_rho_d_um`` imports.
"""
from __future__ import annotations

from midas_distortion.rhod import (
    detector_max_corner_dist_um,
    check_rho_d_um,
    resolve_rho_d_um,
    resolve_rho_d_um_warn,
)

__all__ = [
    "detector_max_corner_dist_um",
    "check_rho_d_um",
    "resolve_rho_d_um",
    "resolve_rho_d_um_warn",
]
