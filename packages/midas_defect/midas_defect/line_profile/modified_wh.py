"""Per-grain modified Williamson-Hall fit with anisotropic contrast factor.

For each grain, fit

    beta_hkl = beta_0 + alpha |G| * sqrt(C(q_U))

across the grain's reflections, with the dislocation contrast factor in the
cubic-invariant form (Ungár, 1996)

    C(q_U) = max(1 - q_U H^2, eps)
    H^2    = (h^2 k^2 + k^2 l^2 + h^2 l^2) / (h^2 + k^2 + l^2)^2

The Burgers magnitude is then absorbed via

    rho = (alpha / sqrt(K))^2 * (1 / b^2)

with ``K = pi alpha_Wilkens^2 / 2 ~ 1.5`` (Wilkens 1970, "F prefactor"
typically 0.30 - 0.50 for cubic Cu).

This module is a thin wrapper that loops over the grains, calling a robust
per-grain LS fit. For populations with poor reflection coverage per grain we
fall back to plain Williamson-Hall (no contrast factor).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _cubic_H_squared(h: int, k: int, l: int) -> float:
    s = h * h + k * k + l * l
    if s == 0:
        return 0.0
    return (h * h * k * k + k * k * l * l + h * h * l * l) / (s * s)


def modified_wh_per_grain(
    per_grain_reflections: list[dict],
    hkls: NDArray[np.intp],
    burgers_length: float,
    q_U_grid: NDArray[np.floating] | None = None,
    F_prefactor: float = 0.30,
    min_refl_per_grain: int = 4,
    q_U_r2_min: float = 0.5,
    F_prefactor_band: tuple = (0.30, 0.50),
) -> dict:
    """Per-grain modified-WH dislocation density via 1-D scan over q_U.

    Honesty notes (AUDIT_2026-06-23.md):
      * **Absolute rho is uncertain by ~(F_hi/F_lo)^2** because the Wilkens prefactor F
        is only known to ~0.30-0.50 (rho ∝ 1/F^2). Report rho as a RATIO between
        populations (F cancels) or quote ``rho_band_per_grain``; do NOT headline an
        absolute per-grain rho.
      * **per-grain q_U is suppressed (NaN) when the fit R^2 < ``q_U_r2_min``** — the
        q_U vs R^2 surface is near-flat on few noisy reflections, so an ill-conditioned
        q_U ("screw fraction") is not a measurement. The retracted demk "q_U R^2~0.02"
        would auto-null here.

    Parameters
    ----------
    per_grain_reflections
        Output of :func:`collect_per_grain_reflections`.
    hkls : (n_hkls, 3) int
        Crystal-frame Miller indices indexed by ``refl_indices`` inside each
        per-grain dict entry.
    burgers_length : Burgers vector magnitude in metres.
    q_U_grid : 1-D grid for the q_U scan; default spans -0.5 .. 3.5 in steps of 0.05.
    F_prefactor : Wilkens prefactor relating slope to sqrt(rho).
    min_refl_per_grain : grains with fewer reflections get NaN.
    q_U_r2_min : suppress (NaN) per-grain q_U whose best fit R^2 is below this.
    F_prefactor_band : (F_lo, F_hi) used to report the rho systematic band.

    Returns
    -------
    dict with
        rho_per_grain      (n_grains,) m^-2 (at ``F_prefactor``; F-band-uncertain)
        rho_band_per_grain (n_grains, 2) m^-2 at (F_hi, F_lo) -> [rho_low, rho_high]
        q_U_per_grain      (n_grains,) best-fit q_U, NaN where R^2 < q_U_r2_min
        D_coherent         (n_grains,) coherent domain size = 2*pi/intercept (Angstrom-like)
        R_squared          (n_grains,)
        n_q_U_suppressed   int, how many grains had q_U nulled for low R^2
    """
    hkls = np.asarray(hkls, dtype=int)
    if q_U_grid is None:
        q_U_grid = np.arange(-0.5, 3.5 + 1e-9, 0.05)
    F_lo, F_hi = float(min(F_prefactor_band)), float(max(F_prefactor_band))

    n_grains = len(per_grain_reflections)
    rho = np.full(n_grains, np.nan)
    rho_band = np.full((n_grains, 2), np.nan)
    q_U = np.full(n_grains, np.nan)
    D = np.full(n_grains, np.nan)
    R2 = np.full(n_grains, np.nan)
    n_suppressed = 0

    # Precompute H^2 per hkl
    H2 = np.array([_cubic_H_squared(*hkls[i]) for i in range(hkls.shape[0])])

    for gi, entry in enumerate(per_grain_reflections):
        if entry["refl_indices"].size < min_refl_per_grain:
            continue
        refl_idx = entry["refl_indices"]
        G = entry["G_magnitude"]
        beta = entry["fwhm"]
        finite = np.isfinite(G) & np.isfinite(beta) & (G > 0) & (beta > 0)
        if int(finite.sum()) < min_refl_per_grain:
            continue
        G = G[finite]
        beta = beta[finite]
        H2_g = H2[refl_idx[finite]]

        # Search over q_U for the best linear fit.
        best_r2 = -np.inf
        best_q = float("nan")
        best_slope = float("nan")
        best_intercept = float("nan")
        for qu in q_U_grid:
            C = np.clip(1.0 - qu * H2_g, 1e-6, None)
            x = G * np.sqrt(C)
            A = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(A, beta, rcond=None)
            res = beta - A @ coef
            ss_res = float((res * res).sum())
            ss_tot = float(((beta - beta.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            if r2 > best_r2:
                best_r2 = r2
                best_q = float(qu)
                best_intercept = float(coef[0])
                best_slope = float(coef[1])

        # Convert slope to rho via simplified Ungar-Borbely (1996):
        # rho = (slope / (pi * F * b))^2  (b in metres)
        if best_slope > 0:
            rho[gi] = (best_slope / (np.pi * F_prefactor * burgers_length)) ** 2
            # systematic band from the F-prefactor uncertainty (rho ∝ 1/F^2):
            rho_band[gi, 0] = (best_slope / (np.pi * F_hi * burgers_length)) ** 2  # low
            rho_band[gi, 1] = (best_slope / (np.pi * F_lo * burgers_length)) ** 2  # high
        # q_U is only a measurement if the fit is well-conditioned
        if best_r2 >= q_U_r2_min:
            q_U[gi] = best_q
        else:
            n_suppressed += 1
        if best_intercept > 0:
            D[gi] = 2.0 * np.pi / best_intercept
        R2[gi] = best_r2

    return {
        "rho_per_grain": rho,
        "rho_band_per_grain": rho_band,
        "q_U_per_grain": q_U,
        "D_coherent": D,
        "R_squared": R2,
        "n_q_U_suppressed": int(n_suppressed),
    }


__all__ = ["modified_wh_per_grain"]
