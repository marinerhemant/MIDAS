"""Scalar invariants of a stress tensor.

All functions accept ``(..., 3, 3)`` stress tensors and broadcast over the
leading axes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def von_mises(sigma: NDArray[np.floating]) -> NDArray[np.floating]:
    """von-Mises (equivalent) stress.

    sqrt(3/2 s_dev_ij s_dev_ij)  with  s_dev = sigma - tr(sigma)/3 * I.
    """
    s = np.asarray(sigma, dtype=float)
    tr = np.einsum("...ii->...", s) / 3.0
    dev = s - tr[..., None, None] * np.eye(3)
    return np.sqrt(1.5 * np.einsum("...ij,...ij->...", dev, dev))


def hydrostatic(sigma: NDArray[np.floating]) -> NDArray[np.floating]:
    """Hydrostatic stress: tr(sigma) / 3."""
    s = np.asarray(sigma, dtype=float)
    return np.einsum("...ii->...", s) / 3.0


def max_shear(sigma: NDArray[np.floating]) -> NDArray[np.floating]:
    """Maximum shear: (lambda_max - lambda_min) / 2 from principal stresses."""
    s = np.asarray(sigma, dtype=float)
    # 0.5 (sigma + sigma^T) is a no-op for symmetric input but defensive.
    sym = 0.5 * (s + np.swapaxes(s, -1, -2))
    eigvals = np.linalg.eigvalsh(sym)
    return 0.5 * (eigvals[..., -1] - eigvals[..., 0])


def lode_parameter(sigma: NDArray[np.floating]) -> NDArray[np.floating]:
    """Lode parameter mu = (2 lambda_2 - lambda_1 - lambda_3) / (lambda_1 - lambda_3).

    Returns NaN where lambda_1 == lambda_3 (pure hydrostatic).
    """
    s = np.asarray(sigma, dtype=float)
    sym = 0.5 * (s + np.swapaxes(s, -1, -2))
    eigvals = np.linalg.eigvalsh(sym)  # ascending
    l1, l2, l3 = eigvals[..., -1], eigvals[..., 1], eigvals[..., 0]
    denom = l1 - l3
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (2.0 * l2 - l1 - l3) / denom
    return np.where(np.abs(denom) > 1e-12, out, np.nan)


def triaxiality(sigma: NDArray[np.floating]) -> NDArray[np.floating]:
    """Stress triaxiality T = sigma_H / sigma_vM.

    Returns +/- inf for pure hydrostatic stress (sigma_vM == 0).
    """
    H = hydrostatic(sigma)
    vM = von_mises(sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(vM > 0, H / vM, np.inf * np.sign(H))
    return T


__all__ = [
    "hydrostatic",
    "lode_parameter",
    "max_shear",
    "triaxiality",
    "von_mises",
]
