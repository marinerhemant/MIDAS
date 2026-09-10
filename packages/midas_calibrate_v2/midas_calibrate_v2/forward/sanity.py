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



def resolve_v1_rho_d_um(v1, *, verbose: bool = False, label: str = "calibrate"):
    """Resolve ``v1.RhoD`` to micrometres IN PLACE and return ``(rho_d_um, how)``.

    The distortion polynomial is evaluated at ``rho = R_um / RhoD``, so RhoD must
    be in micrometres. ``CalibrationParams.RhoD`` defaults to 0.0 ("unset"), and
    three pipelines fell back to the PIXEL-valued ``MaxRingRad`` without
    converting it. That inflates rho by the pixel pitch, 150x on a 150 um Varex,
    so every distortion coefficient is fitted in a basis no other code shares.
    Measured on a real 2880x2880 / 150 um frame with RhoD unset:
    ``autocalibrate_pv`` fed rho = 150.000 at the rim, where
    ``autocalibrate_frozen_point`` fed 1.000.

    Delegates to :func:`midas_distortion.rhod.resolve_rho_d_um`, which detects a
    pixel-valued RhoD and defaults to the beam-centre-to-farthest-corner distance
    when none is usable. This is the resolution ``pipelines.single`` has always
    done; it now lives in one place so the pipelines cannot drift apart again.
    """
    rho_d_um, how = resolve_rho_d_um(
        v1.RhoD if v1.RhoD > 0 else v1.MaxRingRad,
        NrPixelsY=int(v1.NrPixelsY), NrPixelsZ=int(v1.NrPixelsZ),
        BC_y=float(v1.BC_y), BC_z=float(v1.BC_z),
        pxY=float(v1.pxY),
        pxZ=float(v1.pxZ if v1.pxZ > 0 else v1.pxY),
    )
    if verbose:
        print(f"[{label}] RhoD resolved to {rho_d_um:.1f} µm ({how})")
    v1.RhoD = rho_d_um
    return rho_d_um, how


__all__ = [
    "resolve_v1_rho_d_um",
    "detector_max_corner_dist_um",
    "check_rho_d_um",
    "resolve_rho_d_um",
    "resolve_rho_d_um_warn",
]
