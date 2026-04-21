"""Peak fitting utilities for integrated profiles.

The primary path is MIDAS's own GSAS-II Thompson-Cox-Hastings pseudo-Voigt
fitter, which MIDASIntegrator runs when ``DoPeakFit 1`` is set in the
zarr bundle — output lands in ``<stem>.caked_peaks.h5``. Use
:func:`load_peaks_h5` to read it.

For standalone post-integration fitting (notebooks, batch processing that
didn't set DoPeakFit upfront), :func:`fit_peaks_1d` does a scipy-based
pseudo-Voigt fit around expected peak locations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

__all__ = ["fit_peaks_1d", "load_peaks_h5", "pseudo_voigt"]


def pseudo_voigt(
    x: np.ndarray,
    amp: float, center: float, fwhm: float, mixing: float, bg: float,
) -> np.ndarray:
    """Thompson-Cox-Hastings pseudo-Voigt: η·Lorentzian + (1-η)·Gaussian + bg."""
    sigma = fwhm / 2.354820045
    gamma = fwhm / 2.0
    gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return amp * (mixing * lorentz + (1.0 - mixing) * gauss) + bg


def fit_peaks_1d(
    r: np.ndarray,
    intensity: np.ndarray,
    initial_centers: Iterable[float],
    *,
    window: float = 5.0,
    max_fwhm: float = 10.0,
) -> list[dict]:
    """Fit pseudo-Voigt peaks at expected R centres.

    Parameters
    ----------
    r, intensity : 1-D ndarrays
        Radial profile (same length). ``r`` is typically in pixels or 2θ.
    initial_centers : iterable of float
        Starting-point peak positions in the same units as ``r``.
    window : float, default 5.0
        Half-width in ``r``-units of the fitting window around each centre.
    max_fwhm : float, default 10.0
        Upper bound on FWHM to reject runaway fits.

    Returns
    -------
    list of dict with keys: ``center, fwhm, amplitude, mixing, background, chi2``.
    Peaks whose fit diverged are emitted with ``center == NaN``.
    """
    from scipy.optimize import curve_fit

    r = np.asarray(r, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    out: list[dict] = []

    for c0 in initial_centers:
        mask = np.abs(r - c0) <= window
        if mask.sum() < 10:
            out.append(_failed_fit(c0))
            continue
        r_w, i_w = r[mask], intensity[mask]
        amp0 = float(i_w.max() - i_w.min())
        bg0 = float(np.median(i_w[:3]))
        p0 = [amp0, float(c0), 1.0, 0.5, bg0]
        try:
            popt, _ = curve_fit(
                pseudo_voigt, r_w, i_w, p0=p0, maxfev=500,
                bounds=(
                    [0.0, c0 - window, 0.01, 0.0, -np.inf],
                    [np.inf, c0 + window, max_fwhm, 1.0, np.inf],
                ),
            )
            residual = i_w - pseudo_voigt(r_w, *popt)
            chi2 = float(np.sum(residual ** 2))
            out.append({
                "center": float(popt[1]),
                "fwhm": float(popt[2]),
                "amplitude": float(popt[0]),
                "mixing": float(popt[3]),
                "background": float(popt[4]),
                "chi2": chi2,
            })
        except (RuntimeError, ValueError):
            out.append(_failed_fit(c0))
    return out


def _failed_fit(center0: float) -> dict:
    nan = float("nan")
    return {
        "center": nan, "fwhm": nan, "amplitude": nan,
        "mixing": nan, "background": nan, "chi2": nan,
        "initial_center": float(center0),
    }


def load_peaks_h5(path: Union[str, Path]) -> list[dict]:
    """Read MIDAS's ``<stem>.caked_peaks.h5`` into row dicts.

    Schema: ``PeakFits/<frame>/<peak_id>`` each containing the GSAS-II
    pseudo-Voigt fit (area, center, sig, gam, FWHM, eta, chi2).
    """
    import h5py

    rows: list[dict] = []
    p = Path(path)
    with h5py.File(p, "r") as f:
        if "PeakFits" not in f:
            return []
        for frame in sorted(f["PeakFits"]):
            for peak_id in sorted(f[f"PeakFits/{frame}"]):
                grp = f[f"PeakFits/{frame}/{peak_id}"]
                rows.append({
                    "frame": frame,
                    "peak_id": peak_id,
                    **{k: float(grp[k][()]) for k in grp.keys()
                       if grp[k].shape == ()},
                })
    return rows
