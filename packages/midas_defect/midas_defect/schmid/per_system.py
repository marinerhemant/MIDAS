"""Schmid factor per grain and per slip system; spatial agreement diagnostic."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def schmid_factor_per_system(
    OM: NDArray[np.floating],
    sigma_axis_sample: NDArray[np.floating],
    slip_systems: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Full Schmid-factor matrix.

    Returns
    -------
    schmid : (n_grains, n_sys)  |(n.a)(b.a)| per grain per system.
    """
    OM = np.asarray(OM, dtype=float)
    axis = np.asarray(sigma_axis_sample, dtype=float)
    axis = axis / np.linalg.norm(axis)
    sys_ = np.asarray(slip_systems, dtype=float)
    n_sample = np.einsum("gij,kj->gki", OM, sys_[:, 0, :])
    b_sample = np.einsum("gij,kj->gki", OM, sys_[:, 1, :])
    return np.abs(
        np.einsum("gki,i->gk", n_sample, axis) * np.einsum("gkj,j->gk", b_sample, axis)
    )


def spatial_active_system_agreement(
    pos: NDArray[np.floating],
    active_system: NDArray[np.intp],
    k_NN: int = 5,
) -> dict:
    """Fraction of each grain's k-NN that share its max-Schmid system.

    Parameters
    ----------
    pos : (n_grains, 3) coordinates in any consistent units
    active_system : (n_grains,) per-grain max-Schmid system index
    k_NN : number of nearest neighbours (excluding self)

    Returns
    -------
    dict with
        NN_agreement_per_grain : (n_grains,) fraction in [0, 1]
        NN_agreement_population : (mean, p16, p84) summary
    """
    from scipy.spatial import cKDTree  # local import: keep top-level light

    pos = np.asarray(pos, dtype=float)
    active = np.asarray(active_system, dtype=int)
    if pos.shape[0] != active.shape[0]:
        raise ValueError(
            f"pos {pos.shape[0]} vs active_system {active.shape[0]} length mismatch"
        )

    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k_NN + 1)
    if k_NN + 1 == 1:
        raise ValueError("k_NN must be >= 1")
    nbrs = idx[:, 1:]  # drop self
    agreement = (active[nbrs] == active[:, None]).mean(axis=1)
    return {
        "NN_agreement_per_grain": agreement,
        "NN_agreement_population": (
            float(agreement.mean()),
            float(np.percentile(agreement, 16)),
            float(np.percentile(agreement, 84)),
        ),
    }


__all__ = ["schmid_factor_per_system", "spatial_active_system_agreement"]
