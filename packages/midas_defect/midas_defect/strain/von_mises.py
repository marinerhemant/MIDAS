"""von-Mises (equivalent) strain and deviatoric / hydrostatic decomposition."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def von_mises_strain(eps: NDArray[np.floating]) -> NDArray[np.floating]:
    """Equivalent strain: sqrt(2/3 * eps_dev_ij eps_dev_ij)."""
    e = np.asarray(eps, dtype=float)
    tr = np.einsum("...ii->...", e) / 3.0
    dev = e - tr[..., None, None] * np.eye(3)
    return np.sqrt((2.0 / 3.0) * np.einsum("...ij,...ij->...", dev, dev))


def deviatoric_hydrostatic_decomposition(
    eps: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return (deviatoric strain tensor, hydrostatic scalar)."""
    e = np.asarray(eps, dtype=float)
    tr = np.einsum("...ii->...", e) / 3.0
    dev = e - tr[..., None, None] * np.eye(3)
    return dev, tr


__all__ = ["deviatoric_hydrostatic_decomposition", "von_mises_strain"]
