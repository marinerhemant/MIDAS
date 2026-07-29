"""Per-grain spectral diagnostics of the asterism tensor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def asterism_anisotropy_per_grain(
    M_per_grain: NDArray[np.floating],
    mean_q_per_grain: NDArray[np.floating] | None = None,
) -> dict:
    """Per-grain eigenvalues, anisotropy ratios, and radial / azimuthal splits.

    Parameters
    ----------
    M_per_grain : (n_grains, 3, 3)
        Per-grain asterism second-moment tensor.
    mean_q_per_grain : (n_grains, 3) or None
        Per-grain mean q-direction. Needed only for the radial / azimuthal
        split; if None, those entries are NaN.

    Returns
    -------
    dict with
        eigvals_sorted        (n_grains, 3) ascending
        anisotropy_max_min    (n_grains,) lambda_max / max(lambda_min, eps)
        anisotropy_max_mid    (n_grains,)
        radial_eigenvalue     (n_grains,)  M projected along q_hat (one scalar)
        azimuthal_eigenvalue  (n_grains,)  mean of two perpendicular projections
    """
    M = np.asarray(M_per_grain, dtype=float)
    n = M.shape[0]
    eigvals = np.full((n, 3), np.nan)
    eigvecs = np.full((n, 3, 3), np.nan)
    valid = np.array([np.isfinite(M[i]).all() for i in range(n)])
    for i in np.where(valid)[0]:
        ev, vec = np.linalg.eigh(M[i])
        eigvals[i] = ev
        eigvecs[i] = vec

    eps_safe = 1e-30
    abs_eig = np.abs(eigvals)
    anisotropy_max_min = abs_eig[:, -1] / np.maximum(abs_eig[:, 0], eps_safe)
    anisotropy_max_mid = abs_eig[:, -1] / np.maximum(abs_eig[:, 1], eps_safe)

    radial = np.full(n, np.nan)
    azimuthal = np.full(n, np.nan)
    if mean_q_per_grain is not None:
        mq = np.asarray(mean_q_per_grain, dtype=float)
        for i in np.where(valid)[0]:
            g_norm = np.linalg.norm(mq[i])
            if g_norm < 1e-15:
                continue
            qhat = mq[i] / g_norm
            # Build an orthonormal basis (qhat, e1, e2).
            tmp = np.array([1.0, 0.0, 0.0]) if abs(qhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e1 = tmp - np.dot(tmp, qhat) * qhat
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(qhat, e1)
            radial[i] = float(qhat @ M[i] @ qhat)
            azimuthal[i] = 0.5 * float(e1 @ M[i] @ e1 + e2 @ M[i] @ e2)

    return {
        "eigvals_sorted": eigvals,
        "anisotropy_max_min": anisotropy_max_min,
        "anisotropy_max_mid": anisotropy_max_mid,
        "radial_eigenvalue": radial,
        "azimuthal_eigenvalue": azimuthal,
    }


__all__ = ["asterism_anisotropy_per_grain"]
