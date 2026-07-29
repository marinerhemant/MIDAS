"""Schmid factor per grain for a uniaxial loading axis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def schmid_factor_per_grain(
    OM: NDArray[np.floating],
    sigma_axis_sample: NDArray[np.floating],
    slip_systems: NDArray[np.floating],
    return_active_system: bool = False,
) -> NDArray[np.floating] | tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Maximum Schmid factor per grain.

    Schmid factor S = | (n_sample . a) (b_sample . a) | for loading axis ``a``,
    where (n_sample, b_sample) = OM (n_crystal, b_crystal) rotates the slip
    system from the crystal frame into the sample frame.

    Parameters
    ----------
    OM : (n_grains, 3, 3)
        Orientation matrices; ``v_sample = OM @ v_crystal``.
    sigma_axis_sample : (3,)
        Sample-frame loading axis. Normalised internally.
    slip_systems : (n_sys, 2, 3)
        Crystal-frame (n_hat, b_hat) per system.
    return_active_system : bool
        If True, also return the per-grain index of the maximum-Schmid system.

    Returns
    -------
    schmid_max : (n_grains,)
    active_system : (n_grains,)  (only if requested)
    """
    OM = np.asarray(OM, dtype=float)
    axis = np.asarray(sigma_axis_sample, dtype=float)
    axis = axis / np.linalg.norm(axis)
    sys_ = np.asarray(slip_systems, dtype=float)
    n_crystal = sys_[:, 0, :]
    b_crystal = sys_[:, 1, :]

    n_sample = np.einsum("gij,kj->gki", OM, n_crystal)
    b_sample = np.einsum("gij,kj->gki", OM, b_crystal)
    schmid = np.abs(
        np.einsum("gki,i->gk", n_sample, axis) * np.einsum("gkj,j->gk", b_sample, axis)
    )

    active = np.argmax(schmid, axis=1)
    g = np.arange(OM.shape[0])
    schmid_max = schmid[g, active]
    if return_active_system:
        return schmid_max, active
    return schmid_max


__all__ = ["schmid_factor_per_grain"]
