"""RhoD unit self-consistency — the single source of truth for all MIDAS packages.

``RhoD`` (the radial-distortion normalisation radius) appears only inside
``ρ = R / RhoD`` (see :func:`midas_distortion.core.apply_distortion`). It MUST
therefore be in the **same units as R**, i.e. **micrometres**. Different MIDAS
workflows have historically stored it in µm *or* in pixels, and a wrong-unit
value silently corrupts the distortion stage:

* RhoD too small (pixels passed as µm) → ρ ≈ pixel-pitch× too large → the
  distortion polynomial explodes → pixels scatter into the wrong radial bins →
  powder rings wash out (the failure mode that cost real debugging time).
* RhoD too large → ρ ≈ 0 → distortion silently nulled.

These helpers remove the ambiguity once and for all. ``resolve_rho_d_um``
auto-detects the units and returns micrometres; ``check_rho_d_um`` validates a
value that is *supposed* to already be in µm. Both take detector context
(pixel count, beam centre, pixel pitch) because the sane scale for RhoD is the
beam-centre-to-farthest-corner distance in µm.

This module is intentionally dependency-free (pure ``math``) so every package
— ``midas_calibrate_v2``, ``midas_integrate_v2``, ``midas_integrate`` — can
call exactly the same logic.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional


def detector_max_corner_dist_um(
    NrPixelsY: int, NrPixelsZ: int,
    BC_y: float, BC_z: float,
    pxY: float, pxZ: Optional[float] = None,
) -> float:
    """Max distance from the beam centre to any detector corner, in µm —
    the natural upper-bound scale for ``RhoD``.
    """
    if pxZ is None:
        pxZ = pxY
    if NrPixelsY <= 0 or NrPixelsZ <= 0:
        raise ValueError(
            f"NrPixelsY/NrPixelsZ must be positive; got {NrPixelsY}x{NrPixelsZ}"
        )
    corners_px = [
        (0.0, 0.0),
        (NrPixelsY - 1.0, 0.0),
        (0.0, NrPixelsZ - 1.0),
        (NrPixelsY - 1.0, NrPixelsZ - 1.0),
    ]
    return max(
        math.sqrt(((y - BC_y) * pxY) ** 2 + ((z - BC_z) * pxZ) ** 2)
        for (y, z) in corners_px
    )


def check_rho_d_um(
    RhoD_um: float,
    NrPixelsY: int, NrPixelsZ: int,
    BC_y: float, BC_z: float,
    pxY: float, pxZ: Optional[float] = None,
    *,
    low_factor: float = 0.5,
    high_factor: float = 5.0,
    strict: bool = True,
) -> Optional[str]:
    """Raise (or return a warning) if ``RhoD_um`` looks like a unit mistake.

    The healthy range is roughly the detector-corner distance from the beam
    centre (in µm). Outside ``[low_factor·dmax, high_factor·dmax]`` we treat
    the value as a likely unit mix-up — most often RhoD passed in **pixels**,
    which is ``dmax / px`` ≈ 1/px-pitch of the expected µm magnitude.

    Returns ``None`` if healthy. With ``strict=False`` and an out-of-range
    value, returns the diagnostic string; with ``strict=True`` raises
    ``ValueError``.
    """
    if RhoD_um <= 0:
        msg = f"RhoD must be positive; got {RhoD_um}"
        if strict:
            raise ValueError(msg)
        return msg
    dmax = detector_max_corner_dist_um(NrPixelsY, NrPixelsZ, BC_y, BC_z, pxY, pxZ)
    lo, hi = low_factor * dmax, high_factor * dmax
    if RhoD_um < lo or RhoD_um > hi:
        pxY_eff = float(pxY)
        as_px_hint = ""
        if RhoD_um < lo and pxY_eff > 0 and 0.3 * dmax < RhoD_um * pxY_eff < 3 * dmax:
            as_px_hint = (f"  HINT: RhoD * px = {RhoD_um * pxY_eff:.1f} µm "
                          f"looks reasonable — did you pass RhoD in pixels?")
        msg = (
            f"RhoD = {RhoD_um:.4g} µm is outside the sane range "
            f"[{lo:.1f}, {hi:.1f}] µm for a {NrPixelsY}×{NrPixelsZ} detector at "
            f"px=({pxY},{pxZ if pxZ is not None else pxY}), BC=({BC_y},{BC_z}) "
            f"(max corner distance = {dmax:.1f} µm).{as_px_hint}"
        )
        if strict:
            raise ValueError(msg)
        return msg
    return None


def resolve_rho_d_um(
    rho_d,
    NrPixelsY: int, NrPixelsZ: int,
    BC_y: float, BC_z: float,
    pxY: float, pxZ: Optional[float] = None,
    *,
    low_factor: float = 0.5,
    high_factor: float = 5.0,
) -> "tuple[float, str]":
    """Resolve a ``RhoD`` of ambiguous units to micrometres.

    Tries ``{rho_d, rho_d·px, rho_d/px}`` and returns the first that falls in
    the sane window ``[low_factor·dmax, high_factor·dmax]`` (preferring the
    as-is interpretation), where ``dmax`` is the beam-centre-to-farthest-corner
    distance in µm. When no candidate is sane (or ``rho_d`` is missing /
    non-positive) it falls back to ``dmax`` — the natural default RhoD.

    Returns ``(rho_d_um, how)`` where ``how`` documents the chosen branch.
    """
    dmax = detector_max_corner_dist_um(NrPixelsY, NrPixelsZ, BC_y, BC_z, pxY, pxZ)
    lo, hi = low_factor * dmax, high_factor * dmax
    px = float(pxY) if (pxZ is None or pxZ <= 0) else 0.5 * (float(pxY) + float(pxZ))
    try:
        val = float(rho_d)
    except (TypeError, ValueError):
        val = 0.0
    if val > 0 and px > 0:
        for how, cand in (("as-is (µm)", val),
                          ("×px (was pixels)", val * px),
                          ("÷px", val / px)):
            if lo <= cand <= hi:
                return cand, how
    return dmax, "default = BC-to-farthest-edge"


def resolve_rho_d_um_warn(
    rho_d,
    NrPixelsY: int, NrPixelsZ: int,
    BC_y: float, BC_z: float,
    pxY: float, pxZ: Optional[float] = None,
    *,
    where: str = "",
    rel_tol: float = 1e-3,
    **kwargs,
) -> float:
    """Like :func:`resolve_rho_d_um` but emit a ``RuntimeWarning`` when the
    value had to be corrected (units were not already µm), and return just the
    resolved µm value. Intended for ``validate()``/build hooks so a wrong-unit
    RhoD is fixed *and* surfaced rather than silently mangling the distortion.
    """
    resolved, how = resolve_rho_d_um(
        rho_d, NrPixelsY, NrPixelsZ, BC_y, BC_z, pxY, pxZ, **kwargs)
    try:
        orig = float(rho_d)
    except (TypeError, ValueError):
        orig = 0.0
    if orig <= 0 or abs(resolved - orig) > rel_tol * max(resolved, 1.0):
        loc = f" in {where}" if where else ""
        warnings.warn(
            f"RhoD{loc} resolved to {resolved:.1f} µm ({how}); supplied value "
            f"was {orig:.4g}. RhoD must be in micrometres (ρ = R_µm / RhoD); a "
            f"pixel-valued RhoD silently corrupts the distortion. Auto-corrected.",
            RuntimeWarning, stacklevel=3,
        )
    return resolved


__all__ = [
    "detector_max_corner_dist_um",
    "check_rho_d_um",
    "resolve_rho_d_um",
    "resolve_rho_d_um_warn",
]
