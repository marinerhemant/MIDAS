"""Per-grain elastic strain-energy density.

Two flavors of constitutive law:
    * **Isotropic Lamé**: U = 1/2 λ tr(ε)² + μ ε_ij ε_ij. Closed-form;
      checked analytically against pure tensile / pure shear inputs.
    * **Cubic anisotropic**: U = 1/2 σ:ε with σ obtained from the cubic
      stiffness via :mod:`midas_defect.stress.cubic_anisotropic`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..stress.cubic_anisotropic import per_grain_stress_cubic


def elastic_energy_density_isotropic(
    eps: NDArray[np.floating],
    lambda_GPa: float,
    mu_GPa: float,
) -> NDArray[np.floating]:
    """U = 1/2 lambda tr(eps)^2 + mu eps:eps  (in Pa = J/m^3).

    Inputs are dimensionless strain and Lamé constants in GPa; output is
    converted to SI (Pa) so it composes with other invariants.
    """
    e = np.asarray(eps, dtype=float)
    tr = np.einsum("...ii->...", e)
    inner = np.einsum("...ij,...ij->...", e, e)
    U_GPa = 0.5 * lambda_GPa * tr * tr + mu_GPa * inner
    return U_GPa * 1.0e9


def elastic_energy_density_cubic(
    OM: NDArray[np.floating],
    eps_sample: NDArray[np.floating],
    c11: float,
    c12: float,
    c44: float,
) -> NDArray[np.floating]:
    """U = 1/2 sigma : eps using cubic stiffness; per-grain, in Pa.

    ``eps_sample`` is the elastic strain in the **sample frame**.
    """
    sigma = per_grain_stress_cubic(OM, eps_sample, c11, c12, c44)  # Pa
    return 0.5 * np.einsum("gij,gij->g", sigma, np.asarray(eps_sample, dtype=float))


__all__ = ["elastic_energy_density_cubic", "elastic_energy_density_isotropic"]
