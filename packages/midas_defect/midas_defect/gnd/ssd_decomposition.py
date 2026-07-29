"""SSD = total - GND decomposition of dislocation density."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ssd_gnd_decomposition(
    rho_total_per_grain: NDArray[np.floating],
    rho_GND_per_grain: NDArray[np.floating],
) -> dict:
    """Split a total per-grain dislocation density into GND and SSD components.

    ``rho_total`` typically comes from a line-broadening fit (modified
    Williamson-Hall). ``rho_GND`` is the geometrically necessary part from
    intergranular lattice curvature. The statistically stored density is the
    residual: ``rho_SSD = max(0, rho_total - rho_GND)``.

    Returns
    -------
    dict with
        rho_SSD_per_grain     (n_grains,) clipped at 0
        ssd_fraction_per_grain (n_grains,) rho_SSD / rho_total; NaN where total ~ 0
        gnd_fraction_per_grain (n_grains,) rho_GND / rho_total; NaN where total ~ 0
    """
    total = np.asarray(rho_total_per_grain, dtype=float)
    gnd = np.asarray(rho_GND_per_grain, dtype=float)
    if total.shape != gnd.shape:
        raise ValueError(f"shape mismatch: total {total.shape} vs gnd {gnd.shape}")
    ssd = np.clip(total - gnd, a_min=0.0, a_max=None)
    with np.errstate(invalid="ignore", divide="ignore"):
        ssd_frac = np.where(np.abs(total) > 0, ssd / total, np.nan)
        gnd_frac = np.where(np.abs(total) > 0, gnd / total, np.nan)
    return {
        "rho_SSD_per_grain": ssd,
        "ssd_fraction_per_grain": ssd_frac,
        "gnd_fraction_per_grain": gnd_frac,
    }


__all__ = ["ssd_gnd_decomposition"]
