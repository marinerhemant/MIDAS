"""Volume-weighted per-variant energy aggregation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def volume_weighted_energy_per_variant(
    U_per_grain: NDArray[np.floating],
    grain_radii: NDArray[np.floating],
    variant_labels: NDArray[np.intp],
) -> dict:
    """Volume-weighted ``<U/V>`` per variant.

    Uses spherical-equivalent volumes :math:`V = (4/3) \\pi r^3` and a NaN-aware
    weighted mean (failed-fit grains drop from both numerator and denominator).

    Returns
    -------
    dict with
        U_mean_per_variant   {label: weighted mean U (Pa)}
        V_total_per_variant  {label: total volume sum}
        ratio                ratio of every (label_i / label_j) entry, in a
                             nested dict so multi-variant cases compose
    """
    U = np.asarray(U_per_grain, dtype=float)
    r = np.asarray(grain_radii, dtype=float)
    V = (4.0 / 3.0) * np.pi * r**3
    var = np.asarray(variant_labels, dtype=int)

    labels = sorted(set(int(x) for x in np.unique(var)))
    U_mean: dict[int, float] = {}
    V_total: dict[int, float] = {}
    for k in labels:
        mask = (var == k) & np.isfinite(U) & np.isfinite(V)
        v = V[mask]
        u = U[mask]
        if v.sum() <= 0:
            U_mean[k] = float("nan")
            V_total[k] = 0.0
            continue
        U_mean[k] = float((u * v).sum() / v.sum())
        V_total[k] = float(v.sum())

    ratios: dict[tuple[int, int], float] = {}
    for i in labels:
        for j in labels:
            if i == j:
                continue
            denom = U_mean[j]
            ratios[(i, j)] = (U_mean[i] / denom) if denom not in (0.0, float("nan")) and np.isfinite(denom) else float("nan")

    return {
        "U_mean_per_variant": U_mean,
        "V_total_per_variant": V_total,
        "ratio": ratios,
    }


__all__ = ["volume_weighted_energy_per_variant"]
