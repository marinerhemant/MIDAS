"""Per-grain Debye-Waller B-factor via Wilson plot.

For each grain, fit

    log(I_hkl / (|F_hkl|^2 * LP_hkl * m_hkl)) = log(scale) - 2 B sin^2(theta) / lambda^2
                                              = log(scale) - 2 B (|G|/(4 pi))^2 * |G|^2 / |G|^2
                                              = log(scale) - 2 B |G|^2 / (16 pi^2)

so

    y = log(I_hkl / corrections)
    x = |G|^2 / (8 pi^2)         # = 2 sin^2 theta / lambda^2
    slope = - B

The per-grain B (in Å^2) drops out as ``- slope``. The Lorentz-polarisation
factor ``LP_hkl`` and multiplicity ``m_hkl`` are pulled out of the user-
supplied correction callable; the default simply uses ``LP = 1`` and ``m = 1``
which is the right thing for raw single-grain profiles (each (h, k, l) appears
once per grain, and the LP for a single-crystal rocking is geometry-dependent
so callers should supply their own when they need absolute B).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def per_grain_B_factor(
    per_grain_reflections: list[dict],
    hkls: NDArray[np.intp],
    structure_factor_squared: Callable[[tuple[int, int, int]], float],
    lp_factor_fn: Callable[[tuple[int, int, int], float], float] | None = None,
    min_refl_per_grain: int = 4,
) -> dict:
    """Per-grain Debye-Waller B via Wilson-plot linear fit.

    Parameters
    ----------
    per_grain_reflections
        Output of :func:`midas_defect.line_profile.collect_per_grain_reflections`.
    hkls : (n_hkls, 3) int
        Crystal-frame Miller indices indexed by ``refl_indices`` of each entry.
    structure_factor_squared : callable
        ``hkl -> |F_hkl|^2``. The user supplies this via the chosen
        :class:`midas_hkls.Crystal` or a closed form.
    lp_factor_fn : optional callable
        ``hkl, G_mag -> LP_hkl``. Defaults to 1 everywhere.
    min_refl_per_grain : NaN-out grains with fewer reflections.

    Returns
    -------
    dict with
        B_per_grain     (n_grains,) Angstrom^2  (== -slope, the Debye-Waller B)
        scale_per_grain (n_grains,)
        R_squared       (n_grains,)
    """
    hkls = np.asarray(hkls, dtype=int)
    n_grains = len(per_grain_reflections)
    B = np.full(n_grains, np.nan)
    scale = np.full(n_grains, np.nan)
    R2 = np.full(n_grains, np.nan)

    # Precompute |F|^2 per (h,k,l).
    F2 = np.array([structure_factor_squared(tuple(hkls[i].tolist())) for i in range(hkls.shape[0])])

    for gi, entry in enumerate(per_grain_reflections):
        idx = entry["refl_indices"]
        I = entry["intensity"]
        G = entry["G_magnitude"]
        if idx.size < min_refl_per_grain:
            continue
        finite = np.isfinite(I) & (I > 0) & np.isfinite(G) & (G > 0)
        if int(finite.sum()) < min_refl_per_grain:
            continue
        I_f = I[finite]
        G_f = G[finite]
        hk_f = hkls[idx[finite]]
        F2_f = F2[idx[finite]]
        lp_f = (
            np.ones_like(G_f)
            if lp_factor_fn is None
            else np.array([lp_factor_fn(tuple(hk_f[k].tolist()), float(G_f[k])) for k in range(hk_f.shape[0])])
        )
        # Wilson plot variables.
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(I_f / (F2_f * lp_f))
            x = G_f * G_f / (8.0 * np.pi**2)
        clean = np.isfinite(y) & np.isfinite(x)
        if int(clean.sum()) < min_refl_per_grain:
            continue
        A = np.column_stack([np.ones(clean.sum()), x[clean]])
        coef, *_ = np.linalg.lstsq(A, y[clean], rcond=None)
        scale[gi] = float(np.exp(coef[0]))
        B[gi] = float(-coef[1])
        ss_res = float(((y[clean] - A @ coef) ** 2).sum())
        ss_tot = float(((y[clean] - y[clean].mean()) ** 2).sum())
        R2[gi] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "B_per_grain": B,
        "scale_per_grain": scale,
        "R_squared": R2,
    }


__all__ = ["per_grain_B_factor"]
