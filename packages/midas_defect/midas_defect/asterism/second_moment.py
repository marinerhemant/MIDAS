"""Per-grain q-space second-moment tensor of asterism voxels.

For each grain :math:`g` the intensity-weighted q-space second moment is

    M^g_{ij} = sum_k w_k (q_k - q_B_k)_i (q_k - q_B_k)_j / sum_k w_k

over voxels :math:`k` assigned to grain :math:`g` and flagged as asterism
intensity (i.e. not within the Bragg-core selection). :math:`q_B_k` is the
nearest predicted Bragg position for that voxel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def per_grain_asterism_tensor(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    grain_of_voxel: NDArray[np.intp],
    P_all_nearest: NDArray[np.floating],
    asterism_mask: NDArray[np.bool_],
    n_grains: int,
    min_voxels_per_grain: int = 10,
) -> NDArray[np.floating]:
    """Per-grain intensity-weighted second-moment tensor in q-space.

    Parameters
    ----------
    qs : (n_voxels, 3)
        Voxel q-positions.
    vals : (n_voxels,)
        Voxel intensities (weights).
    grain_of_voxel : (n_voxels,)
        Pre-computed grain assignment per voxel; entries with index >= n_grains
        are ignored.
    P_all_nearest : (n_voxels, 3)
        Nearest predicted Bragg q for each voxel.
    asterism_mask : (n_voxels,)
        True for voxels participating in the asterism tail.
    n_grains : number of grains
    min_voxels_per_grain : NaN-out grains with fewer than this many voxels in mask.

    Returns
    -------
    M_per_grain : (n_grains, 3, 3)
        Symmetric per-grain second-moment tensor; NaN where insufficient voxels.
    """
    qs = np.asarray(qs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    g = np.asarray(grain_of_voxel, dtype=int)
    Pn = np.asarray(P_all_nearest, dtype=float)
    mask = np.asarray(asterism_mask, dtype=bool)

    sel = mask & (g >= 0) & (g < n_grains) & np.isfinite(vals) & (vals > 0)
    if not sel.any():
        return np.full((n_grains, 3, 3), np.nan, dtype=float)

    dq = qs[sel] - Pn[sel]
    w = vals[sel]
    gi = g[sel]

    M = np.full((n_grains, 3, 3), np.nan, dtype=float)
    counts = np.bincount(gi, minlength=n_grains)
    weight_sum = np.zeros(n_grains, dtype=float)
    outer_sum = np.zeros((n_grains, 3, 3), dtype=float)
    np.add.at(weight_sum, gi, w)
    # Vectorised outer-product accumulator.
    outer = np.einsum("ki,kj->kij", dq, dq)
    np.add.at(outer_sum, gi, w[:, None, None] * outer)

    valid = (counts >= min_voxels_per_grain) & (weight_sum > 0)
    M[valid] = outer_sum[valid] / weight_sum[valid, None, None]
    return M


__all__ = ["per_grain_asterism_tensor"]
