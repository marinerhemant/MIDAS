"""Resolved-shear-stress projections.

The resolved shear stress on slip system (n_hat, b_hat) is the projection

    tau = | b_hat . sigma . n_hat |

which is also the Schmid factor times the axial stress in the uniaxial case.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def resolved_shear_stress(
    sigma_crystal: NDArray[np.floating],
    n_hat: NDArray[np.floating],
    b_hat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """RSS on one (n, b) system for a batch of crystal-frame stresses.

    Parameters
    ----------
    sigma_crystal : (..., 3, 3)
    n_hat, b_hat : (3,) unit vectors in the crystal frame

    Returns
    -------
    tau : (...,)
        | b . sigma . n |
    """
    s = np.asarray(sigma_crystal, dtype=float)
    return np.abs(np.einsum("i,...ij,j->...", b_hat, s, n_hat))


def max_resolved_shear_per_grain(
    sigma_crystal: NDArray[np.floating],
    slip_systems: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.intp]]:
    """For each grain, the max RSS across the supplied slip systems.

    Parameters
    ----------
    sigma_crystal : (n_grains, 3, 3)
    slip_systems  : (n_sys, 2, 3)  -- system[i, 0] = n_hat, system[i, 1] = b_hat

    Returns
    -------
    tau_max : (n_grains,)
    active_system : (n_grains,)
        Index into ``slip_systems`` of the maximum-RSS system per grain.
    """
    s = np.asarray(sigma_crystal, dtype=float)
    sys_ = np.asarray(slip_systems, dtype=float)
    n = sys_[:, 0, :]
    b = sys_[:, 1, :]
    # tau[g, k] = | b_k . s_g . n_k |
    tau = np.abs(np.einsum("ki,gij,kj->gk", b, s, n))
    active = np.argmax(tau, axis=1)
    return tau[np.arange(tau.shape[0]), active], active


__all__ = ["max_resolved_shear_per_grain", "resolved_shear_stress"]
