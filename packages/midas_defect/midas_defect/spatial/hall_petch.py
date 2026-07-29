"""Hall-Petch slope from per-grain stress vs grain-size scatter.

    sigma = sigma_0 + k / sqrt(d)

with ``d = 2 r``. Per-variant fits allow matrix vs twin contrast.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hall_petch_slope(
    sigma_per_grain: NDArray[np.floating],
    grain_radius_per_grain: NDArray[np.floating],
    variant_labels: NDArray[np.intp] | None = None,
) -> dict:
    """Per-variant least-squares fit of sigma = sigma_0 + k / sqrt(d).

    Parameters
    ----------
    sigma_per_grain : (n_grains,) e.g. von-Mises stress in MPa
    grain_radius_per_grain : (n_grains,) radius in um
    variant_labels : optional; if None, fits all grains as one group.

    Returns
    -------
    dict with
        k_HP_per_variant      {label: slope (MPa * um^{1/2})}
        sigma_0_per_variant   {label: intercept}
        R_squared             {label: coefficient of determination}
        n_per_variant         {label: number of grains used}
    """
    s = np.asarray(sigma_per_grain, dtype=float)
    r = np.asarray(grain_radius_per_grain, dtype=float)
    if s.shape[0] != r.shape[0]:
        raise ValueError(f"length mismatch: sigma {s.shape[0]} vs r {r.shape[0]}")

    if variant_labels is None:
        var = np.zeros(s.shape[0], dtype=int)
    else:
        var = np.asarray(variant_labels, dtype=int)

    out_k: dict[int, float] = {}
    out_s0: dict[int, float] = {}
    out_r2: dict[int, float] = {}
    out_n: dict[int, int] = {}

    for lbl in sorted(set(int(x) for x in np.unique(var))):
        mask = (var == lbl) & np.isfinite(s) & np.isfinite(r) & (r > 0)
        if int(mask.sum()) < 3:
            out_k[lbl] = float("nan")
            out_s0[lbl] = float("nan")
            out_r2[lbl] = float("nan")
            out_n[lbl] = int(mask.sum())
            continue
        x = 1.0 / np.sqrt(2.0 * r[mask])  # 1 / sqrt(d)
        y = s[mask]
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        s0, k = float(coef[0]), float(coef[1])
        ss_res = float(((y - A @ coef) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out_k[lbl] = k
        out_s0[lbl] = s0
        out_r2[lbl] = r2
        out_n[lbl] = int(mask.sum())

    return {
        "k_HP_per_variant": out_k,
        "sigma_0_per_variant": out_s0,
        "R_squared": out_r2,
        "n_per_variant": out_n,
    }


__all__ = ["hall_petch_slope"]
