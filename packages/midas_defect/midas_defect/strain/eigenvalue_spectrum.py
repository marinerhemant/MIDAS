"""Per-grain eigenvalue spectrum of the strain tensor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def per_grain_eigenvalues(eps: NDArray[np.floating]) -> dict:
    """Eigenvalue decomposition of (n_grains, 3, 3) symmetric strain.

    Returns
    -------
    dict with
        eigvals             (n_grains, 3) ascending
        eigvecs             (n_grains, 3, 3) columns are eigenvectors
        anisotropy_max_min  (n_grains,) |lambda_max| / max(|lambda_min|, eps)
        anisotropy_max_mid  (n_grains,) |lambda_max| / max(|lambda_mid|, eps)
        lode_parameter      (n_grains,) mu = (2 l2 - l1 - l3) / (l1 - l3); NaN
                            for degenerate spectra
    """
    e = np.asarray(eps, dtype=float)
    if e.ndim != 3 or e.shape[1:] != (3, 3):
        raise ValueError(f"eps must be (n_grains, 3, 3); got {e.shape}")
    sym = 0.5 * (e + np.swapaxes(e, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(sym)  # ascending

    abs_eig = np.abs(eigvals)
    eps_safe = 1e-30
    anisotropy_max_min = abs_eig[:, -1] / np.maximum(abs_eig[:, 0], eps_safe)
    anisotropy_max_mid = abs_eig[:, -1] / np.maximum(abs_eig[:, 1], eps_safe)

    l1, l2, l3 = eigvals[:, -1], eigvals[:, 1], eigvals[:, 0]
    denom = l1 - l3
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(np.abs(denom) > 1e-12, (2.0 * l2 - l1 - l3) / denom, np.nan)

    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "anisotropy_max_min": anisotropy_max_min,
        "anisotropy_max_mid": anisotropy_max_mid,
        "lode_parameter": mu,
    }


__all__ = ["per_grain_eigenvalues"]
