"""Per-grain stacking-fault probability from peak-shift Warren analysis.

For each grain, fit

    log(a_apparent_hkl) = log(a_0) + alpha * xi_hkl

where ``a_apparent_hkl = lambda / (2 d_hkl sin theta) approx (2 pi |G_meas|) / |G_pred|^{-1}``.
The slope ``alpha`` is the FCC stacking-fault probability. Phase-specific ``xi``
tables come from Warren, *X-Ray Diffraction*, Ch. 13.

For brevity the default table here covers the most common FCC reflections only;
callers can pass a custom ``xi_table`` dict for BCC / HCP or extended sets.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Warren 1969, Ch. 13: Stacking-fault peak-shift coefficients xi_hkl for FCC.
# Sign convention: a_apparent = a_0 (1 + alpha xi_hkl).
# Source: Warren Table 13.1 (rewritten for cubic systems).
WARREN_XI_FCC: dict[tuple[int, int, int], float] = {
    (1, 1, 1):  0.00,
    (2, 0, 0): +0.30,
    (2, 2, 0):  0.00,
    (3, 1, 1): -0.075,
    (2, 2, 2):  0.00,
    (4, 0, 0): +0.30,
    (3, 3, 1): +0.038,
    (4, 2, 0): -0.075,
    (4, 2, 2):  0.00,
}


def _canonicalize_hkl(hkl: tuple[int, int, int]) -> tuple[int, int, int]:
    """Sort the absolute values descending; SF coefficients depend on family, not sign."""
    return tuple(sorted([abs(int(x)) for x in hkl], reverse=True))


def warren_alpha_per_grain(
    per_grain_reflections: list[dict],
    hkls: NDArray[np.intp],
    xi_table: dict[tuple[int, int, int], float] | None = None,
    min_refl_per_grain: int = 3,
) -> dict:
    """Stacking-fault probability per grain via Warren-Velterop linear fit.

    Parameters
    ----------
    per_grain_reflections : list of dicts (see :func:`collect_per_grain_reflections`).
    hkls : (n_hkls, 3) crystal-frame Miller indices.
    xi_table : dict mapping canonical (h, k, l) to xi. Defaults to FCC.
    min_refl_per_grain : NaN-out grains with fewer usable reflections.

    Returns
    -------
    dict with
        alpha_per_grain (n_grains,) stacking-fault probability per grain
        R_squared       (n_grains,)
    """
    hkls = np.asarray(hkls, dtype=int)
    if xi_table is None:
        xi_table = WARREN_XI_FCC

    xi_lookup = np.array(
        [xi_table.get(_canonicalize_hkl(tuple(hkls[i].tolist())), np.nan) for i in range(hkls.shape[0])]
    )

    n_grains = len(per_grain_reflections)
    alpha = np.full(n_grains, np.nan)
    R2 = np.full(n_grains, np.nan)

    for gi, entry in enumerate(per_grain_reflections):
        idx = entry["refl_indices"]
        centroid = entry["centroid"]
        G_mag = entry["G_magnitude"]
        if idx.size < min_refl_per_grain:
            continue
        xi = xi_lookup[idx]
        # a_apparent proxy: 2 pi / G_meas; use centroid as G_meas (it is already
        # the radial centroid along q_target, so equal to |G_meas|).
        with np.errstate(divide="ignore", invalid="ignore"):
            a_app = 2.0 * np.pi / centroid
            ln_a = np.log(a_app)
        finite = np.isfinite(xi) & np.isfinite(ln_a)
        if int(finite.sum()) < min_refl_per_grain:
            continue
        x = xi[finite]
        y = ln_a[finite]
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        alpha[gi] = float(coef[1])
        ss_res = float(((y - A @ coef) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        R2[gi] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha_per_grain": alpha,
        "R_squared": R2,
    }


__all__ = ["WARREN_XI_FCC", "warren_alpha_per_grain"]
