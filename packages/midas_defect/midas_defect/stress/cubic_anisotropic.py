"""Cubic-elasticity stress from per-grain strain.

Units convention
----------------
Inputs are dimensionless strain and stiffness constants in GPa. Output stress
is in **Pa** (factor 1e9 applied internally) so downstream invariants and
energy expressions are SI-clean.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .voigt import strain_tensor_to_voigt, voigt_to_stress_tensor


def cubic_stiffness_voigt(c11: float, c12: float, c44: float) -> NDArray[np.floating]:
    """6x6 cubic stiffness in Voigt notation (units same as ``c11``)."""
    C = np.zeros((6, 6), dtype=float)
    C[0, 0] = C[1, 1] = C[2, 2] = c11
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = c12
    C[3, 3] = C[4, 4] = C[5, 5] = c44
    return C


def per_grain_stress_cubic(
    OM: NDArray[np.floating],
    eps_sample: NDArray[np.floating],
    c11: float,
    c12: float,
    c44: float,
) -> NDArray[np.floating]:
    """Per-grain stress tensor in sample frame from cubic elasticity.

    Parameters
    ----------
    OM : (n_grains, 3, 3)
        Sample-frame orientation matrix per grain. ``OM`` rotates a vector
        from the *crystal* frame to the *sample* frame:
            v_sample = OM @ v_crystal.
    eps_sample : (n_grains, 3, 3)
        Elastic strain tensor per grain in the **sample frame**.
    c11, c12, c44 : float
        Cubic elastic constants in GPa.

    Returns
    -------
    sigma_sample : (n_grains, 3, 3)
        Stress tensor per grain in the sample frame, in **Pa**.

    Notes
    -----
        eps_crystal = OM^T eps_sample OM
        sigma_crystal = C : eps_crystal           (Voigt-vector product)
        sigma_sample  = OM sigma_crystal OM^T
    """
    OM = np.asarray(OM, dtype=float)
    eps_sample = np.asarray(eps_sample, dtype=float)
    if OM.ndim != 3 or OM.shape[1:] != (3, 3):
        raise ValueError(f"OM must be (n_grains, 3, 3); got {OM.shape}")
    if eps_sample.shape != OM.shape:
        raise ValueError(
            f"eps_sample {eps_sample.shape} must match OM {OM.shape}"
        )

    eps_crystal = np.einsum("gki,gkl,glj->gij", OM, eps_sample, OM)
    eps_voigt = strain_tensor_to_voigt(eps_crystal)
    C = cubic_stiffness_voigt(c11, c12, c44)
    sigma_voigt = eps_voigt @ C.T
    sigma_crystal = voigt_to_stress_tensor(sigma_voigt)
    sigma_sample = np.einsum("gij,gjk,glk->gil", OM, sigma_crystal, OM)
    return sigma_sample * 1.0e9  # GPa -> Pa


__all__ = ["cubic_stiffness_voigt", "per_grain_stress_cubic"]
