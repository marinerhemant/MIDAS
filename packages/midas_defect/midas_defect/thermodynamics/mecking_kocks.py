"""Mecking-Kocks dislocation-evolution constitutive law.

Phenomenological one-parameter family for the dislocation density rate:

    d rho / d eps = k_1 sqrt(rho) - k_2 rho

with saturation density ``rho_sat = (k_1 / k_2)^2`` reached when the storage
and recovery rates balance. References: Mecking & Kocks 1981 *Acta Metall.*
29:1865; Kocks & Mecking 2003 *Prog. Mater. Sci.* 48:171-273.

Two operating modes
    * **variant_specific_k2**: given per-variant saturated rho, recover the
      ratio k_1/k_2 per variant; assuming a common k_1 across variants yields
      the variant-to-variant k_2 ratio, the most defensible statement of
      asymmetry.
    * **mk_evolve**: forward integrate rho(eps) given (k_1, k_2, rho_0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def variant_specific_k2(
    rho_sat_per_variant: dict[int | str, float],
    k1_literature: float | None = None,
) -> dict:
    """Extract k_1/k_2 ratios per variant from saturated dislocation densities.

    The MK saturation gives ``sqrt(rho_sat) = k_1 / k_2``. With a common k_1
    across variants (the conservative assumption when the slip-system family
    is shared), the per-variant ``k_2`` is k_1 / sqrt(rho_sat).

    Parameters
    ----------
    rho_sat_per_variant : {label: rho_sat (m^-2)}
    k1_literature : if given, returns absolute k_2 per variant via k_2 = k_1 / sqrt(rho_sat).

    Returns
    -------
    dict with
        k_ratio_per_variant   {label: sqrt(rho_sat)}  = k_1/k_2
        k2_ratio_pairs        {(i, j): k_2_i / k_2_j}
        k2_absolute           {label: k_2}  (only if k1_literature is given)
        lower_bound_note      str: caveat on the asymmetry interpretation
    """
    k_ratio = {lbl: float(np.sqrt(max(rho, 0.0))) for lbl, rho in rho_sat_per_variant.items()}
    labels = sorted(k_ratio.keys(), key=str)
    pairs: dict[tuple, float] = {}
    for i in labels:
        for j in labels:
            if i == j:
                continue
            if k_ratio[j] > 0:
                # k_2 = k_1 / k_ratio => k_2_i / k_2_j = k_ratio_j / k_ratio_i
                pairs[(i, j)] = float(k_ratio[j] / k_ratio[i]) if k_ratio[i] > 0 else float("inf")
            else:
                pairs[(i, j)] = float("inf")

    out: dict = {
        "k_ratio_per_variant": k_ratio,
        "k2_ratio_pairs": pairs,
        "lower_bound_note": (
            "The per-variant k_2 ratio assumes a common k_1 across variants. "
            "If k_1 itself differs (e.g. via Schmid-factor asymmetry in the "
            "storage cross-section) the recovered k_2 ratio is a lower bound "
            "on the true asymmetry."
        ),
    }
    if k1_literature is not None:
        out["k2_absolute"] = {
            lbl: (k1_literature / k_ratio[lbl]) if k_ratio[lbl] > 0 else float("inf")
            for lbl in labels
        }
    return out


def mk_evolve(
    eps: NDArray[np.floating],
    k1: float,
    k2: float,
    rho_init: float = 1.0e10,
    M: float = 3.06,
    alpha: float = 0.30,
) -> NDArray[np.floating]:
    """Analytic Mecking-Kocks rho(eps) integration.

    The MK ODE ``d rho / d eps = k_1 sqrt(rho) - k_2 rho`` has the closed-form
    solution

        rho(eps) = (k_1/k_2)^2 * (1 - C exp(-k_2 eps / 2))^2

    with C = 1 - sqrt(rho_init) / (k_1 / k_2).

    Parameters
    ----------
    eps : (n,) strain trajectory (dimensionless)
    k_1, k_2 : MK parameters (units consistent with rho in m^-2)
    rho_init : initial dislocation density (m^-2)
    M, alpha : Taylor factor + Taylor coupling; passed through for caller
        bookkeeping and unused in this function (kept for API compatibility
        with the broader thermodynamics workflow).

    Returns
    -------
    rho : (n,) dislocation density along the trajectory (m^-2).
    """
    eps = np.asarray(eps, dtype=float)
    if k2 <= 0:
        raise ValueError(f"k2 must be positive; got {k2}")
    sat = k1 / k2
    sqrt_rho0 = float(np.sqrt(max(rho_init, 0.0)))
    C = 1.0 - sqrt_rho0 / sat
    return (sat * (1.0 - C * np.exp(-0.5 * k2 * eps))) ** 2


__all__ = ["mk_evolve", "variant_specific_k2"]
