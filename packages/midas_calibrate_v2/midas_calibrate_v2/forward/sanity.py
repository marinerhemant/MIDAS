"""Geometry sanity checks — catch unit-mistakes early.

The most common mistake is feeding ``RhoD`` in **pixels** to a path that
expects **micrometres** (or vice-versa).  Because ``RhoD`` only appears
inside the distortion normalisation ``ρ = R/RhoD``, a wrong-unit value
either explodes the distortion (RhoD too small) or quietly nulls it out
(RhoD too large) — both produce calibrations that look superficially fine
but are systematically wrong.  These helpers refuse to silently accept
nonsense values.
"""
from __future__ import annotations

import math
from typing import Optional


def detector_max_corner_dist_um(
    NrPixelsY: int, NrPixelsZ: int,
    BC_y: float, BC_z: float,
    pxY: float, pxZ: Optional[float] = None,
) -> float:
    """Maximum distance from the beam centre to any of the four detector
    corners, in micrometres — the natural upper-bound scale for ``RhoD``.
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

    The healthy range is roughly the detector-corner distance from the
    beam centre (in µm).  Outside the window
    ``[low_factor·dmax, high_factor·dmax]`` we treat the value as a likely
    unit mix-up — most often someone passed RhoD in **pixels**, which is
    `dmax / pxY` ≈ 1/200 of the expected µm magnitude on a 200-µm detector.

    Parameters
    ----------
    RhoD_um : the value being validated, in micrometres.
    NrPixelsY, NrPixelsZ, BC_y, BC_z, pxY, pxZ : detector geometry.
    low_factor, high_factor : tolerance window around ``dmax``.
    strict : when ``True`` (default) raise ``ValueError``; when ``False``
        return the warning message instead (caller logs it).

    Returns
    -------
    None if RhoD looks healthy.  When ``strict=False`` and the value is out
    of range, returns the diagnostic string.

    Raises
    ------
    ValueError if ``strict=True`` and RhoD is outside the window.
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
        # Best-guess diagnosis: pixels passed as µm?
        as_px_hint = ""
        if RhoD_um < lo:
            implied_px = RhoD_um / pxY_eff if pxY_eff > 0 else float("nan")
            if 0.3 * dmax < RhoD_um * pxY_eff < 3 * dmax:
                as_px_hint = (f"  HINT: RhoD * pxY = {RhoD_um * pxY_eff:.1f} µm "
                              f"looks reasonable — did you pass RhoD in pixels?")
        msg = (
            f"RhoD = {RhoD_um:.4g} µm is outside the sane range "
            f"[{lo:.1f}, {hi:.1f}] µm for a "
            f"{NrPixelsY}×{NrPixelsZ} detector at "
            f"px=({pxY},{pxZ if pxZ is not None else pxY}), "
            f"BC=({BC_y},{BC_z}) (max corner distance "
            f"= {dmax:.1f} µm).{as_px_hint}"
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
) -> tuple[float, str]:
    """Resolve a ``RhoD`` of ambiguous units to micrometres.

    ``RhoD`` only appears in the distortion normalisation ``ρ = R_um / RhoD``,
    so it must be in µm. Different MIDAS workflows have stored it in µm or in
    pixels, which silently breaks the distortion stage. This helper removes
    the ambiguity: given a value of unknown units, it tries
    ``{rho_d, rho_d·px, rho_d/px}`` and returns the first that falls in the
    physically sane window ``[low_factor·dmax, high_factor·dmax]`` (preferring
    the as-is interpretation), where ``dmax`` is the beam-centre-to-farthest-
    corner distance in µm.

    When no candidate is sane (or ``rho_d`` is missing / non-positive) it falls
    back to ``dmax`` itself — the natural default RhoD (BC to farthest detector
    edge), which is what the automated / from-scratch path should use.

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


__all__ = ["detector_max_corner_dist_um", "check_rho_d_um", "resolve_rho_d_um"]
