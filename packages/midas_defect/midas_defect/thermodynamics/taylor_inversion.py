"""Taylor-inverse dislocation density estimate from flow stress.

Taylor's classical relation between flow stress and dislocation density is

    sigma = M * alpha * mu * b * sqrt(rho)

so the implied total dislocation density at a given flow stress is

    rho_total = (sigma / (M * alpha * mu * b))^2.

Useful as a sanity check against per-grain line-profile densities -- if the
LP-visible fraction is far below unity, that argues for a sub-resolved
fraction (dipoles, low-angle network) that the LP probe is blind to.
"""

from __future__ import annotations


def taylor_implied_total_rho(
    sigma_flow: float,
    M: float = 3.06,
    alpha: float = 0.30,
    mu: float = 47.0e9,
    burgers: float = 2.57e-10,
) -> float:
    """rho_total = (sigma / (M alpha mu b))^2  (m^-2).

    Defaults are Cu-like: M=3.06 (random FCC texture Taylor factor),
    alpha=0.30 (literature mid-range for FCC), mu=47 GPa, b=2.57 A.
    """
    denom = M * alpha * mu * burgers
    if denom <= 0:
        raise ValueError("M alpha mu b must be positive")
    return (sigma_flow / denom) ** 2


def wh_visible_fraction(
    rho_WH: float,
    sigma_flow_literature: float,
    **taylor_params,
) -> float:
    """Fraction of the Taylor-implied total density that the WH analysis sees."""
    rho_total = taylor_implied_total_rho(sigma_flow_literature, **taylor_params)
    if rho_total <= 0:
        return float("nan")
    return float(rho_WH / rho_total)


__all__ = ["taylor_implied_total_rho", "wh_visible_fraction"]
