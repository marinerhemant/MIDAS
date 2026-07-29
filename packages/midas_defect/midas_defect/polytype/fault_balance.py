"""Periodic (sharp 9R satellite) vs aperiodic (continuous relrod) intensity balance.

A grain can carry BOTH a discrete period-3 modulation (sharp n*G/3 satellites = 9R)
and a continuous stacking-fault relrod (aperiodic faults smearing intensity along
<111> between the reflections). Their ratio says how far the faulting has organized
into periodic 9R vs how much is still random:

    periodicity_fraction = I_thirds / (I_thirds + I_gaps)

where ``I_thirds`` is the intensity in tubes at the forbidden-gap satellites
(n = +-1, 2, 4, 5; n%3 != 0) and ``I_gaps`` is the intensity at the half-integer
control positions (n = +-0.5, 1.5, 2.5, ...) where a *pure* periodic 9R has ~zero
but a continuous relrod has signal. periodicity_fraction -> 1 means well-ordered 9R;
-> 0 means a relrod with little periodic order ("faulting caught in the act").

This complements ``satellite_radial_excess`` (which gives a yes/no 9R-vs-relrod
*verdict* at G/3, 2G/3) by returning a continuous *balance* over the whole ladder.
It is an aggregate, geometry-honest quantity -- NOT per-variant (see
:mod:`midas_defect.attribution`). Intensities are uncorrected proxies (structure
factors of satellites vs fundamentals differ), so use ratios, not absolute volumes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["periodic_aperiodic_balance"]


def periodic_aperiodic_balance(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    axis_dir: NDArray[np.floating],
    G_magnitude: float,
    *,
    n_max: int = 6,
    tube_parallel: float = 0.10,
    perp_max: float = 0.20,
) -> dict:
    """Periodic-vs-aperiodic fault intensity balance along ``axis_dir``.

    Returns a dict with
        periodicity_fraction      I_thirds / (I_thirds + I_gaps)
        I_satellites              sum over forbidden-gap thirds (n%3 != 0)
        I_relrod_gaps             sum over half-integer control positions
        I_fundamentals            sum over n in +-3, +-6
        satellite_to_fundamental  I_satellites / I_fundamentals
        relrod_to_fundamental     I_relrod_gaps / I_fundamentals
        per_position              {n: (I, n_voxels)} for every probed position
    """
    qs = np.asarray(qs, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    axis = np.asarray(axis_dir, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    g3 = G_magnitude / 3.0

    proj = qs @ axis
    perp = np.linalg.norm(qs - proj[:, None] * axis, axis=1)
    good = (vals > 0) & np.isfinite(vals)

    def tube(q_along: float):
        sel = good & (np.abs(proj - q_along) < tube_parallel) & (perp < perp_max)
        return float(vals[sel].sum()), int(sel.sum())

    thirds_n = [n for n in range(-n_max, n_max + 1) if n != 0 and (n % 3) != 0]
    fund_n = [n for n in range(-n_max, n_max + 1) if n != 0 and (n % 3) == 0]
    gap_half = [n + 0.5 for n in range(-n_max, n_max)]  # +-0.5, 1.5, ...

    per_position = {}
    I_thirds = 0.0
    for n in thirds_n:
        I, nv = tube(n * g3); per_position[n] = (I, nv); I_thirds += I
    I_fund = 0.0
    for n in fund_n:
        I, nv = tube(n * g3); per_position[n] = (I, nv); I_fund += I
    I_gaps = 0.0
    for h in gap_half:
        I, nv = tube(h * g3); per_position[h] = (I, nv); I_gaps += I

    denom = I_thirds + I_gaps
    return {
        "periodicity_fraction": (I_thirds / denom) if denom > 0 else float("nan"),
        "I_satellites": I_thirds,
        "I_relrod_gaps": I_gaps,
        "I_fundamentals": I_fund,
        "satellite_to_fundamental": (I_thirds / I_fund) if I_fund > 0 else float("nan"),
        "relrod_to_fundamental": (I_gaps / I_fund) if I_fund > 0 else float("nan"),
        "per_position": per_position,
    }
