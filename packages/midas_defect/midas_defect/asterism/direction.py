"""Per-grain edge fraction from the asterism tensor.

The edge fraction is

    f_edge = | a_hat . g_hat |^2

where :math:`a_hat` is the principal eigenvector (largest eigenvalue) of the
asterism second-moment tensor :math:`M^g` and :math:`g_hat` is the per-grain
intensity-weighted mean q-direction. A pure edge-dislocation broadens
*along* the diffraction vector; pure screw broadens *perpendicular* to it.

    f_edge ~ 1 -> pure edge
    f_edge ~ 0 -> pure screw
    f_edge ~ 1/3 -> isotropic mixture
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def edge_fraction_per_grain(
    M_per_grain: NDArray[np.floating],
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    grain_of_voxel: NDArray[np.intp],
    asterism_mask: NDArray[np.bool_],
    n_grains: int,
) -> NDArray[np.floating]:
    """Edge fraction f_edge per grain in [0, 1]."""
    M = np.asarray(M_per_grain, dtype=float)
    qs = np.asarray(qs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    g = np.asarray(grain_of_voxel, dtype=int)
    mask = np.asarray(asterism_mask, dtype=bool)

    sel = mask & (g >= 0) & (g < n_grains) & np.isfinite(vals) & (vals > 0)
    g_sel = g[sel]
    qs_sel = qs[sel]
    w = vals[sel]

    mean_q = np.full((n_grains, 3), np.nan, dtype=float)
    weight_sum = np.zeros(n_grains, dtype=float)
    weighted_q = np.zeros((n_grains, 3), dtype=float)
    np.add.at(weight_sum, g_sel, w)
    np.add.at(weighted_q, g_sel, w[:, None] * qs_sel)
    valid = weight_sum > 0
    mean_q[valid] = weighted_q[valid] / weight_sum[valid, None]

    out = np.full(n_grains, np.nan, dtype=float)
    for gi in range(n_grains):
        if not np.isfinite(M[gi]).all() or not np.isfinite(mean_q[gi]).all():
            continue
        eigvals, eigvecs = np.linalg.eigh(M[gi])
        a_hat = eigvecs[:, -1]
        g_norm = np.linalg.norm(mean_q[gi])
        if g_norm < 1e-15:
            continue
        g_hat = mean_q[gi] / g_norm
        out[gi] = float(np.dot(a_hat, g_hat) ** 2)
    return out


__all__ = ["edge_fraction_per_grain"]
