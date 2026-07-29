"""Local spatial gradient of a per-grain scalar via mean kNN difference."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stress_spatial_gradient_per_grain(
    sigma_vM: NDArray[np.floating],
    pos: NDArray[np.floating],
    k_NN: int = 5,
) -> NDArray[np.floating]:
    """Per-grain ``<|s_self - s_NN|>`` over the ``k_NN`` nearest neighbours."""
    from scipy.spatial import cKDTree

    sigma = np.asarray(sigma_vM, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if sigma.shape[0] != pos.shape[0]:
        raise ValueError(f"length mismatch: sigma {sigma.shape[0]} vs pos {pos.shape[0]}")

    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k_NN + 1)
    nbrs = idx[:, 1:]  # exclude self

    diffs = np.abs(sigma[nbrs] - sigma[:, None])
    return diffs.mean(axis=1)


__all__ = ["stress_spatial_gradient_per_grain"]
