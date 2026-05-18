"""Direct geometry-fit refinement of seed (BC + Lsd + tilts) against arc pixels.

Port of v1's ``AutoCalibrateZarr.auto_guess_tilted`` — runs L-BFGS-B over
(BC_y, BC_z, Lsd, ty, tz) minimizing the sum of squared "distance to
nearest known ring radius" residuals across detected arc pixels.

Used as a follow-up after :func:`seed_from_image` when the initial
chord-bisector + multi-hypothesis Lsd seed isn't tight enough — typically
needed for short-Lsd multi-panel detectors (e.g. Pilatus at 657 mm
where ring 0 sits at R ~ 15 px and is BC-sensitive).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _pixel_to_R_np(y_px, z_px, bc_y, bc_z, lsd, ty_deg, tz_deg, px):
    """MIDAS forward model: pixel → R (px), distortion-free, tx=0."""
    tyr = np.deg2rad(ty_deg)
    tzr = np.deg2rad(tz_deg)
    cy, sy = np.cos(tyr), np.sin(tyr)
    cz, sz = np.cos(tzr), np.sin(tzr)
    TRs = np.array([
        [cy * cz, -cy * sz, sy],
        [sz, cz, 0.0],
        [-sy * cz, sy * sz, cy],
    ])
    Yc = (-np.asarray(y_px, dtype=np.float64) + bc_y) * px
    Zc = (np.asarray(z_px, dtype=np.float64) - bc_z) * px
    A = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    B = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    C = TRs[2, 1] * Yc + TRs[2, 2] * Zc
    R_um = (lsd / (lsd + A)) * np.sqrt(B * B + C * C)
    return R_um / px


def refine_seed_geometry(
    arc_coords: np.ndarray,
    sim_rads_px: np.ndarray,
    *,
    bc_init: Tuple[float, float],
    lsd_init: float,
    px: float,
    max_tilt_deg: float = 5.0,
    max_pts: int = 10_000,
    skip_threshold_px2: float = 5.0,
    min_ring_px: float = 0.0,            # drop sim rings below this radius
    min_arc_R_px: float = 0.0,            # drop arc pixels below this radius
) -> dict:
    """Refine (BC_y, BC_z, Lsd, ty, tz) by direct geometry fit on arc pixels.

    Parameters
    ----------
    arc_coords : (N, 2) array of (y_px, z_px) pixel coordinates.
    sim_rads_px : known ring radii in pixels.
    bc_init, lsd_init : seed values (e.g. from chord-bisector / multi-hyp Lsd).
    px : pixel size (μm).
    max_tilt_deg : tilt bound for the optimiser.
    max_pts : random subsample if too many arc pixels (speed).
    skip_threshold_px2 : skip the L-BFGS step if initial residual is already
        below this (avoids over-fitting noise on a clean seed).

    Returns
    -------
    dict ``{bc_y, bc_z, lsd, ty, tz, residual}``.
    """
    from scipy.optimize import minimize

    y_arc = arc_coords[:, 0]
    z_arc = arc_coords[:, 1]

    # Drop sim rings below ``min_ring_px`` — small rings have huge BC
    # leverage at short Lsd (Pilatus failure mode).  Drop arc pixels at
    # those radii too (using the seed BC).
    if min_ring_px > 0:
        sim_rads_px = sim_rads_px[sim_rads_px >= min_ring_px]
    if min_arc_R_px > 0:
        R_arc = np.sqrt((y_arc - bc_init[0]) ** 2 + (z_arc - bc_init[1]) ** 2)
        keep_arc = R_arc >= min_arc_R_px
        y_arc = y_arc[keep_arc]
        z_arc = z_arc[keep_arc]
    if len(y_arc) > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_arc), max_pts, replace=False)
        y_arc = y_arc[idx]
        z_arc = z_arc[idx]

    def objective(params):
        bc_y, bc_z, lsd, ty, tz = params
        R = _pixel_to_R_np(y_arc, z_arc, bc_y, bc_z, lsd, ty, tz, px)
        diffs = np.abs(R[:, None] - sim_rads_px[None, :])
        min_diffs = np.min(diffs, axis=1)
        return float(np.sum(min_diffs * min_diffs))

    x0 = np.array([bc_init[0], bc_init[1], lsd_init, 0.0, 0.0])
    bounds = [
        (bc_init[0] - 200.0, bc_init[0] + 200.0),
        (bc_init[1] - 200.0, bc_init[1] + 200.0),
        (lsd_init * 0.85, lsd_init * 1.15),
        (-max_tilt_deg, max_tilt_deg),
        (-max_tilt_deg, max_tilt_deg),
    ]
    initial_obj = objective(x0)
    initial_resid = initial_obj / len(y_arc)
    if initial_resid < skip_threshold_px2:
        return {
            "bc_y": float(bc_init[0]), "bc_z": float(bc_init[1]),
            "lsd": float(lsd_init), "ty": 0.0, "tz": 0.0,
            "residual": float(initial_resid),
        }
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-12})
    bc_y, bc_z, lsd, ty, tz = result.x
    residual = result.fun / len(y_arc)
    at_bound = (abs(ty) >= max_tilt_deg - 0.01 or abs(tz) >= max_tilt_deg - 0.01
                or lsd <= lsd_init * 0.86 or lsd >= lsd_init * 1.14)
    if at_bound or residual > initial_resid:
        return {
            "bc_y": float(bc_init[0]), "bc_z": float(bc_init[1]),
            "lsd": float(lsd_init), "ty": 0.0, "tz": 0.0,
            "residual": float(initial_resid),
        }
    return {
        "bc_y": float(bc_y), "bc_z": float(bc_z),
        "lsd": float(lsd), "ty": float(ty), "tz": float(tz),
        "residual": float(residual),
    }


__all__ = ["refine_seed_geometry"]
