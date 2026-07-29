"""Thermodynamic energy-balance closure: matrix-twin elastic gap vs twin-boundary energy.

If matrix and twin variants partition the elastic strain energy unequally, the
difference must (in principle) be paid for by another energy reservoir. The
canonical candidate is the twin-boundary energy:

    U_TB / V = 2 * gamma_TB / L_lamella     (J / m^3)

(factor of 2 because each lamella has two parallel boundaries per unit
length). The closure ratio

    R = dU_measured / U_TB_predicted

is unity when the elastic gap is exactly accounted for by the boundary
energy. Values near 1 indicate thermodynamic self-consistency; deviations
flag missing reservoirs (dislocation cores, stacking faults, residual phase
fraction) or constitutive-law mismatch.
"""

from __future__ import annotations


def twin_boundary_energy_density(gamma_TB: float, L_lamella: float) -> float:
    """U_TB / V  (J/m^3) given boundary energy gamma_TB (J/m^2) and lamella length L (m)."""
    if L_lamella <= 0:
        raise ValueError(f"L_lamella must be positive; got {L_lamella}")
    return 2.0 * gamma_TB / L_lamella


def energy_balance_closure(
    U_matrix: float,
    U_twin: float,
    gamma_TB: float,
    L_lamella: float,
) -> dict:
    """Compare elastic-energy partition to predicted twin-boundary cost.

    Returns
    -------
    dict with
        dU_measured       |U_matrix - U_twin|, J/m^3
        U_TB_predicted    2 gamma_TB / L_lamella, J/m^3
        closure_ratio     dU_measured / U_TB_predicted
        within_unity_pct  |1 - closure_ratio| * 100
    """
    dU = abs(U_matrix - U_twin)
    U_TB = twin_boundary_energy_density(gamma_TB, L_lamella)
    ratio = dU / U_TB if U_TB > 0 else float("inf")
    return {
        "dU_measured": dU,
        "U_TB_predicted": U_TB,
        "closure_ratio": ratio,
        "within_unity_pct": abs(1.0 - ratio) * 100.0,
    }


__all__ = ["energy_balance_closure", "twin_boundary_energy_density"]
