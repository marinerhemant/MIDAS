"""Algebraic circle fits for arc-based BC estimation.

The chord-bisector method in :mod:`from_image` fails on multi-panel
detectors because each "arc region" is a per-panel fragment of the true
ring; the chord drawn on a single fragment doesn't bisect through the
ring center.

Circle fits (Pratt-method algebraic) recover the center directly from
any arc points, robust to fragment topology.  Combined with median-of-
arc-centers, this gives a clean BC even on Pilatus 8×6 panel grids.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def pratt_circle_fit(coords: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Pratt algebraic circle fit.  Returns (cy, cz, radius) or None.

    For each point (y, z), the equation y²+z²-2ay-2bz+(a²+b²-r²)=0 is linear
    in unknowns (a, b, c=a²+b²-r²).  We solve in least-squares form.

    More robust than direct LSQ because it handles small-arc geometry where
    the matrix is ill-conditioned.

    Parameters
    ----------
    coords : (N, 2) array of (y, z) pixel coordinates.
    """
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 3:
        return None
    y = coords[:, 0].astype(np.float64)
    z = coords[:, 1].astype(np.float64)
    yc = y.mean()
    zc = z.mean()
    u = y - yc
    v = z - zc
    Suu = (u * u).sum()
    Svv = (v * v).sum()
    Suv = (u * v).sum()
    Suuu = (u * u * u).sum()
    Svvv = (v * v * v).sum()
    Suvv = (u * v * v).sum()
    Svuu = (v * u * u).sum()
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
    try:
        uc, vc = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None
    cy = uc + yc
    cz = vc + zc
    r2 = uc * uc + vc * vc + (Suu + Svv) / max(len(coords), 1)
    if r2 < 0:
        return None
    return float(cy), float(cz), float(np.sqrt(r2))


def fit_arcs_for_bc(
    coords_list,
    *,
    min_arc_pts: int = 50,
    radius_dedup_tol_px: float = 5.0,
    max_radius_cv: float = 0.05,        # post-fit residual CV; 0.05 ≈ 5% scatter
    seed_bc: Optional[Tuple[float, float]] = None,
    radius_outlier_kappa: float = 5.0,
) -> dict:
    """Fit a circle to each arc region; aggregate centers into a BC estimate.

    Steps:
      1. Drop arcs with fewer than ``min_arc_pts``.
      2. Per-arc Pratt circle fit.  Drop arcs whose fit residual is too high.
      3. Group arcs by their fitted radius (within ``radius_dedup_tol_px``).
      4. For each ring group, median the (cy, cz) across arcs.
      5. Across rings, sigma-clip outliers, return median of remaining centers.

    Returns ``{"bc_y", "bc_z", "ring_radii_px", "n_arcs", "n_rings"}``.
    """
    arc_results = []
    for coords in coords_list:
        if len(coords) < min_arc_pts:
            continue
        fit = pratt_circle_fit(coords)
        if fit is None:
            continue
        cy, cz, r = fit
        # Sanity-check: reject fits where the residual std is very large.
        rs = np.linalg.norm(coords - np.array([cy, cz]), axis=1)
        if r <= 0 or rs.std() / max(r, 1.0) > max_radius_cv:
            continue
        arc_results.append((cy, cz, r))
    if not arc_results:
        return {"bc_y": float("nan"), "bc_z": float("nan"),
                "ring_radii_px": np.array([]), "n_arcs": 0, "n_rings": 0}
    arr = np.array(arc_results)            # [N_arcs, 3]
    # Sigma-clip outliers (e.g. spurious panel-gap arcs).
    centers = arr[:, :2]
    if seed_bc is not None:
        ref = np.asarray(seed_bc, dtype=np.float64)
    else:
        ref = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - ref, axis=1)
    mad = np.median(np.abs(distances - np.median(distances)))
    cutoff = np.median(distances) + radius_outlier_kappa * (1.4826 * mad + 1e-6)
    keep = distances <= cutoff
    if keep.sum() == 0:
        keep = np.ones(len(arr), dtype=bool)
    arr_keep = arr[keep]
    bc_y = float(np.median(arr_keep[:, 0]))
    bc_z = float(np.median(arr_keep[:, 1]))
    # Group radii.
    radii = np.sort(arr_keep[:, 2])
    grouped = []
    for r in radii:
        if not grouped or abs(r - grouped[-1]) > radius_dedup_tol_px:
            grouped.append(float(r))
    return {
        "bc_y": bc_y, "bc_z": bc_z,
        "ring_radii_px": np.array(grouped, dtype=np.float64),
        "n_arcs": int(keep.sum()),
        "n_rings": len(grouped),
    }


__all__ = ["pratt_circle_fit", "fit_arcs_for_bc"]
