"""Tensor <-> Voigt-vector conversions for stress and strain.

The Voigt convention used here is the canonical 6-vector ordering
    (11, 22, 33, 23, 13, 12)
with the standard factor-of-2 on the off-diagonal *strain* entries (so that
the stiffness matrix C times the strain Voigt-vector gives stress).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stress_tensor_to_voigt(sigma: NDArray[np.floating]) -> NDArray[np.floating]:
    """(..., 3, 3) symmetric stress -> (..., 6) Voigt."""
    s = np.asarray(sigma, dtype=float)
    return np.stack(
        [s[..., 0, 0], s[..., 1, 1], s[..., 2, 2], s[..., 1, 2], s[..., 0, 2], s[..., 0, 1]],
        axis=-1,
    )


def voigt_to_stress_tensor(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """(..., 6) stress Voigt -> (..., 3, 3) symmetric stress."""
    v = np.asarray(v, dtype=float)
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=float)
    out[..., 0, 0] = v[..., 0]
    out[..., 1, 1] = v[..., 1]
    out[..., 2, 2] = v[..., 2]
    out[..., 1, 2] = out[..., 2, 1] = v[..., 3]
    out[..., 0, 2] = out[..., 2, 0] = v[..., 4]
    out[..., 0, 1] = out[..., 1, 0] = v[..., 5]
    return out


def strain_tensor_to_voigt(eps: NDArray[np.floating]) -> NDArray[np.floating]:
    """(..., 3, 3) symmetric strain -> (..., 6) Voigt, with 2 x off-diagonals.

    Note the factor 2 on shear strains: e_4 = 2 e_23, e_5 = 2 e_13, e_6 = 2 e_12.
    """
    e = np.asarray(eps, dtype=float)
    return np.stack(
        [
            e[..., 0, 0],
            e[..., 1, 1],
            e[..., 2, 2],
            2.0 * e[..., 1, 2],
            2.0 * e[..., 0, 2],
            2.0 * e[..., 0, 1],
        ],
        axis=-1,
    )


def voigt_to_strain_tensor(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """(..., 6) strain Voigt -> (..., 3, 3) symmetric strain."""
    v = np.asarray(v, dtype=float)
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=float)
    out[..., 0, 0] = v[..., 0]
    out[..., 1, 1] = v[..., 1]
    out[..., 2, 2] = v[..., 2]
    out[..., 1, 2] = out[..., 2, 1] = 0.5 * v[..., 3]
    out[..., 0, 2] = out[..., 2, 0] = 0.5 * v[..., 4]
    out[..., 0, 1] = out[..., 1, 0] = 0.5 * v[..., 5]
    return out


__all__ = [
    "strain_tensor_to_voigt",
    "stress_tensor_to_voigt",
    "voigt_to_strain_tensor",
    "voigt_to_stress_tensor",
]
