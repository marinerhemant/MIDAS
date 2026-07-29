"""Classify the period-3 modulation behind the on-axis ladder: displacement vs composition.

An ideal 9R has NO on-axis ``n*G/3`` satellites (they are structurally extinct --
see :func:`~midas_defect.polytype.cell_index.structure_factor_intensity`). The
observed ladder therefore requires a **period-3 modulation**, and its *type* is
diagnosable from how the satellite intensity varies with order ``n``:

* **displacement / interlayer-spacing relaxation** (faults pucker the lattice):
  the first-order sideband amplitude grows ~ ``l = 3n``, so intensity **rises
  ~ n^2** at small amplitude and turns over (Bessel) at larger amplitude.
* **composition / Suzuki segregation** (chemical ordering, e.g. Al on every 3rd
  plane): the sideband intensity is **flat** in order (n-independent).

So a ladder that *rises* with order is displacement-dominated; a *flat* ladder is
composition-dominated. This module fits both forward models (using the exact finite-
cell structure factor) to a measured single-variant ladder and returns which one
wins, the best-fit amplitude, and the residuals -- the productized form of the
``v1_ladder_modtype.py`` analysis (demk V1: rise with order -> displacement, with a
minor composition wrinkle at the lowest order).

IMPORTANT -- use a SINGLE variant's ladder. Mixing the two doublet variants (which
populate only the low orders) biases the low-order intensities up and corrupts the
order trend. Correct the input intensities for ``f(s)^2`` / Debye-Waller and use a
matched integration box before calling this. Pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .cell_index import NINE_R_SEQUENCE, structure_factor_intensity

__all__ = ["ModulationFit", "classify_modulation"]


@dataclass
class ModulationFit:
    """Result of classifying the on-axis-ladder modulation.

    verdict : "displacement" | "composition" | "mixed"
    displacement_amplitude : best-fit spacing modulation (units of d_111).
    comp_amplitude : best-fit composition modulation (dimensionless, in (-1,1)).
    displacement_residual, comp_residual : normalized least-squares residual
        fraction of each model (0 = perfect, 1 = no better than zero).
    residual_ratio : comp_residual / displacement_residual (>1 favours displacement).
    order_rise_exponent : slope of log(intensity) vs log(order) -- a descriptive
        signature (~2 for small-amplitude displacement, ~0 for composition).
    orders, intensities, disp_model, comp_model : the data and best-fit curves.
    """

    verdict: str
    displacement_amplitude: float
    comp_amplitude: float
    displacement_residual: float
    comp_residual: float
    residual_ratio: float
    order_rise_exponent: float
    orders: NDArray[np.floating]
    intensities: NDArray[np.floating]
    disp_model: NDArray[np.floating]
    comp_model: NDArray[np.floating]
    metadata: dict = field(default_factory=dict)


def _fit_scaled(model: NDArray[np.floating], data: NDArray[np.floating]) -> tuple[float, float]:
    """Best linear scale A minimizing ||data - A*model||^2; return (A, residual_frac)."""
    denom = float(np.dot(model, model))
    if denom <= 0:
        return 0.0, 1.0
    A = float(np.dot(model, data) / denom)
    resid = data - A * model
    ss = float(np.dot(data, data))
    frac = float(np.dot(resid, resid) / ss) if ss > 0 else 1.0
    return A, frac


def classify_modulation(
    orders: NDArray[np.floating],
    intensities: NDArray[np.floating],
    *,
    sequence: str = NINE_R_SEQUENCE,
    delta_grid: NDArray[np.floating] | None = None,
    comp_grid: NDArray[np.floating] | None = None,
    decisive_ratio: float = 3.0,
) -> ModulationFit:
    """Classify a single-variant on-axis ladder as displacement- or composition-type.

    Parameters
    ----------
    orders : (K,)
        Satellite orders ``n`` (e.g. ``[1, 2, 4, 5]``); the reflection is (0, 0, 3n).
        Integer-triple orders (n=3,6,... the fundamentals) should be excluded.
    intensities : (K,)
        Measured satellite intensity per order, already corrected for ``f(s)^2`` /
        Debye-Waller and integrated with a matched box.
    sequence : str
        Stacking sequence for the forward structure factor (default 9R).
    delta_grid, comp_grid : arrays, optional
        Amplitude grids to scan for the displacement (units of d_111) and
        composition (dimensionless) models. Sensible defaults are used if None.
    decisive_ratio : float
        The residual ratio (worse/better) beyond which the verdict is called;
        otherwise "mixed".

    Returns
    -------
    ModulationFit
    """
    orders = np.asarray(orders, dtype=np.float64).ravel()
    intensities = np.asarray(intensities, dtype=np.float64).ravel()
    if orders.shape != intensities.shape:
        raise ValueError("orders and intensities must have the same shape")
    if orders.size < 2:
        raise ValueError("need at least 2 orders to classify a trend")
    if np.any(orders % 3 == 0):
        raise ValueError("exclude integer-triple orders (n=3,6,...): those are FCC "
                         "fundamentals, not satellites")

    if delta_grid is None:
        delta_grid = np.linspace(0.005, 0.20, 40)
    if comp_grid is None:
        comp_grid = np.linspace(0.01, 0.60, 40)
    delta_grid = np.asarray(delta_grid, dtype=np.float64)
    comp_grid = np.asarray(comp_grid, dtype=np.float64)

    hkls = np.stack([np.zeros_like(orders), np.zeros_like(orders), 3.0 * orders], axis=1)

    # --- displacement model: scan spacing modulation, fit a free scale each ---
    best = {"resid": np.inf, "amp": 0.0, "A": 0.0, "model": None}
    for d in delta_grid:
        m = structure_factor_intensity(hkls, sequence=sequence, spacing_modulation=float(d))
        m = np.atleast_1d(m).astype(np.float64)
        A, frac = _fit_scaled(m, intensities)
        if frac < best["resid"]:
            best = {"resid": frac, "amp": float(d), "A": A, "model": A * m}
    disp = best

    # --- composition model: scan comp modulation, fit a free scale each ---
    bestc = {"resid": np.inf, "amp": 0.0, "A": 0.0, "model": None}
    for cmod in comp_grid:
        m = structure_factor_intensity(hkls, sequence=sequence, comp_modulation=float(cmod))
        m = np.atleast_1d(m).astype(np.float64)
        A, frac = _fit_scaled(m, intensities)
        if frac < bestc["resid"]:
            bestc = {"resid": frac, "amp": float(cmod), "A": A, "model": A * m}
    comp = bestc

    # descriptive log-log slope (needs positive intensities)
    pos = intensities > 0
    if pos.sum() >= 2:
        slope = float(np.polyfit(np.log(orders[pos]), np.log(intensities[pos]), 1)[0])
    else:
        slope = float("nan")

    dr = disp["resid"]
    cr = comp["resid"]
    ratio = float(cr / dr) if dr > 0 else float("inf")
    if ratio >= decisive_ratio:
        verdict = "displacement"
    elif ratio <= 1.0 / decisive_ratio:
        verdict = "composition"
    else:
        verdict = "mixed"

    return ModulationFit(
        verdict=verdict,
        displacement_amplitude=disp["amp"],
        comp_amplitude=comp["amp"],
        displacement_residual=dr,
        comp_residual=cr,
        residual_ratio=ratio,
        order_rise_exponent=slope,
        orders=orders,
        intensities=intensities,
        disp_model=disp["model"] if disp["model"] is not None else np.zeros_like(orders),
        comp_model=comp["model"] if comp["model"] is not None else np.zeros_like(orders),
        metadata={
            "displacement_scale": disp["A"],
            "comp_scale": comp["A"],
            "note": ("use a SINGLE variant's ladder, f(s)^2/DW-corrected; the two "
                     "doublet variants bias the low orders up"),
        },
    )
