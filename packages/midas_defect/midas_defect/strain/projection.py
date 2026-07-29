"""Project a strain tensor onto a slip / twin system.

For an active slip (or twin shear) plane with sample-frame normal ``n_hat``
and shear direction ``b_hat``, the projected shear is

    gamma = b_hat . eps . n_hat.

Sample-frame projection: callers pass the system axes already rotated by
the orientation matrix of the relevant grain.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def project_onto_system(
    eps: NDArray[np.floating],
    n_hat_sample: NDArray[np.floating],
    b_hat_sample: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Per-grain projected shear gamma = b . eps . n.

    Parameters
    ----------
    eps : (n_grains, 3, 3)
    n_hat_sample, b_hat_sample : (n_grains, 3) sample-frame unit vectors

    Returns
    -------
    gamma : (n_grains,)
    """
    e = np.asarray(eps, dtype=float)
    n = np.asarray(n_hat_sample, dtype=float)
    b = np.asarray(b_hat_sample, dtype=float)
    return np.einsum("gi,gij,gj->g", b, e, n)


def _max_schmid_system_per_grain(
    OM: NDArray[np.floating],
    sigma_axis_sample: NDArray[np.floating],
    slip_systems: NDArray[np.floating],
) -> tuple[NDArray[np.intp], NDArray[np.floating], NDArray[np.floating]]:
    """Helper: find the max-Schmid system per grain and return its sample-frame axes."""
    OM = np.asarray(OM, dtype=float)
    axis = np.asarray(sigma_axis_sample, dtype=float)
    axis = axis / np.linalg.norm(axis)
    sys_ = np.asarray(slip_systems, dtype=float)
    n_crystal = sys_[:, 0, :]
    b_crystal = sys_[:, 1, :]
    # Rotate every system into every grain's sample frame.
    n_sample = np.einsum("gij,kj->gki", OM, n_crystal)
    b_sample = np.einsum("gij,kj->gki", OM, b_crystal)
    schmid = np.abs(np.einsum("gki,i,gkj,j->gk", n_sample, axis, b_sample, axis))
    active = np.argmax(schmid, axis=1)
    g = np.arange(OM.shape[0])
    return active, n_sample[g, active, :], b_sample[g, active, :]


def twin_shear_projection_per_pair(
    eps: NDArray[np.floating],
    OM: NDArray[np.floating],
    pairs: NDArray[np.intp],
    twin_systems_crystal: NDArray[np.floating],
    sigma_axis_sample: NDArray[np.floating],
) -> dict:
    """Per-matched-pair twin-shear projection of the inter-grain strain jump.

    For each matched (matrix, twin) pair the active twin system is selected
    by maximum Schmid factor in the **matrix** grain. The shear projection
    is then evaluated on the inter-grain strain difference

        delta_eps = eps_matrix - eps_twin

    on ``(b_active, n_active)`` (the strain-difference shear) and on an
    orthogonal "control" pair so the active-system signal can be calibrated
    against the in-pair noise floor.

    Parameters
    ----------
    eps : (n_grains, 3, 3) sample-frame strain
    OM  : (n_grains, 3, 3) orientation matrices
    pairs : (n_pairs, 2) matrix index, twin index
    twin_systems_crystal : (n_sys, 2, 3) crystal-frame (n, b) per system
    sigma_axis_sample : (3,) sample-frame loading axis

    Returns
    -------
    dict with
        dEps_twin_shear   (n_pairs,)  b . (eps_m - eps_t) . n  on active system
        dEps_orthogonal   (n_pairs,)  same on an orthogonal control direction
        active_system     (n_pairs,)  index of the max-Schmid system in matrix
    """
    e = np.asarray(eps, dtype=float)
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must be (n_pairs, 2); got {pairs.shape}")

    matrix_idx = pairs[:, 0]
    twin_idx = pairs[:, 1]

    active, n_sample, b_sample = _max_schmid_system_per_grain(
        np.asarray(OM, dtype=float)[matrix_idx], sigma_axis_sample, twin_systems_crystal
    )

    delta = e[matrix_idx] - e[twin_idx]
    gamma_active = np.einsum("pi,pij,pj->p", b_sample, delta, n_sample)

    # Orthogonal control: orthonormalise an arbitrary in-plane direction
    # against (n, b), evaluate on that pair as a noise-floor reference.
    arbitrary = np.tile([0.0, 0.0, 1.0], (b_sample.shape[0], 1))
    swap = np.abs(np.einsum("pi,pi->p", arbitrary, b_sample)) > 0.95
    arbitrary[swap] = np.array([1.0, 0.0, 0.0])
    perp = arbitrary - np.einsum("pi,pi->p", arbitrary, b_sample)[:, None] * b_sample
    perp -= np.einsum("pi,pi->p", perp, n_sample)[:, None] * n_sample
    perp_norm = np.linalg.norm(perp, axis=1, keepdims=True)
    perp = perp / np.where(perp_norm > 1e-9, perp_norm, 1.0)
    gamma_orthogonal = np.einsum("pi,pij,pj->p", perp, delta, n_sample)

    return {
        "dEps_twin_shear": gamma_active,
        "dEps_orthogonal": gamma_orthogonal,
        "active_system": active,
    }


__all__ = ["project_onto_system", "twin_shear_projection_per_pair"]
