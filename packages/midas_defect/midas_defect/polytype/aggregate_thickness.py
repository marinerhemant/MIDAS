"""Geometry-honest AGGREGATE 9R coherence length along the measured satellite axis.

Replaces ``polytype.lamella_thickness.per_grain_lamella_thickness``, which assigned a
per-grain L by projecting onto each grain's OM@[1,1,1] and reading a radial FWHM. For a
satellite that lives on the shared twin plane that projection angle drives the result:
on demk, corr(theta, L)=-0.62 and L swung 50->22 A with theta alone, manufacturing a
parent/twin "thickness" gap (AUDIT_2026-06-23.md). There is no valid per-grain or
per-variant L from FF for a shared-plane feature.

What IS valid: ONE aggregate coherence length measured along the satellite's ACTUAL
sample-frame direction (found from the data), with a real peak fit and the Scherrer
shape factor. The radial width is largely insensitive to the angular (tangential)
mosaic, so radial-FWHM -> L is a reasonable aggregate domain size; any residual
broadening only makes the reported L a lower bound (flagged).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["find_satellite_axis", "aggregate_lamella_thickness"]


def find_satellite_axis(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    candidate_dirs: NDArray[np.floating],
    G_sat: float,
    *,
    tol_deg: float = 3.0,
    half: float = 0.05,
) -> NDArray[np.floating]:
    """Return the candidate direction carrying the most satellite-shell intensity.

    candidate_dirs : (M,3) directions to test (e.g. all grain <111>, sample frame).
    G_sat : satellite |q| (e.g. G(111)/3). Picks the dir with max Sum(I) within
    ``tol_deg`` of it in the |q|=G_sat +/- half shell. This is n_sat, the empirical
    satellite location — NOT assumed from any grain's orientation.
    """
    qs = np.asarray(qs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    D = np.asarray(candidate_dirs, dtype=float).copy()
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    D[D[:, 2] < 0] *= -1.0
    qmag = np.linalg.norm(qs, axis=1)
    sel = (np.abs(qmag - G_sat) < half) & (vals > 0)
    qh = qs[sel] / qmag[sel, None]
    w = vals[sel]
    ct = math.cos(math.radians(tol_deg))
    best_dir, best_I = D[0], -1.0
    for d in D:
        I = w[np.abs(qh @ d) >= ct].sum()
        if I > best_I:
            best_I, best_dir = I, d
    return best_dir


def aggregate_lamella_thickness(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    axis: NDArray[np.floating],
    G_sat: float,
    *,
    K: float = 0.94,
    tube_perp: float = 0.10,
    half: float = 0.20,
    n_bins: int = 80,
) -> dict:
    """Aggregate 9R coherence length L = K*2pi/FWHM from the radial satellite peak.

    Profiles intensity along ``axis`` (the measured n_sat) within a perpendicular tube,
    fits a Gaussian+constant to the radial peak near ``G_sat``, and inverts Scherrer.

    Returns
    -------
    dict with
        L_angstrom        K*2pi/FWHM   (aggregate; one number, NOT per-variant)
        fwhm              fitted radial FWHM (1/A)
        r2                Gaussian-fit R^2 (refuse to trust if low)
        n_voxels          voxels in the tube
        is_lower_bound    True (residual instrument/mosaic broadening only shrinks L)
        note              caveat string
    """
    qs = np.asarray(qs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    proj = qs @ axis
    perp = np.linalg.norm(qs - proj[:, None] * axis, axis=1)
    m = (np.abs(proj - G_sat) < half) & (perp < tube_perp) & (vals > 0)
    nvox = int(m.sum())
    out = dict(L_angstrom=float("nan"), fwhm=float("nan"), r2=float("nan"),
               n_voxels=nvox, is_lower_bound=True,
               note="aggregate along measured n_sat; radial FWHM ~ insensitive to "
                    "angular mosaic; residual broadening => L is a lower bound; "
                    "NOT a per-grain or per-variant quantity (shared-plane feature).")
    if nvox < 50:
        return out

    edges = np.linspace(G_sat - half, G_sat + half, n_bins + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    y, _ = np.histogram(proj[m], bins=edges, weights=vals[m])
    if y.max() <= 0:
        return out

    # Gaussian + constant fit
    def gauss(x, A, mu, sig, b):
        return A * np.exp(-0.5 * ((x - mu) / sig) ** 2) + b

    try:
        from scipy.optimize import curve_fit
        p0 = [y.max() - np.median(y), G_sat, 0.03, float(np.median(y))]
        popt, _ = curve_fit(gauss, c, y, p0=p0, maxfev=10000,
                            bounds=([0, G_sat - half, 1e-3, 0],
                                    [np.inf, G_sat + half, half, np.inf]))
        A, mu, sig, b = popt
        resid = y - gauss(c, *popt)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
        fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sig
    except Exception:
        # moment fallback (background = min)
        yb = y - y.min()
        if yb.sum() <= 0:
            return out
        mu = float((c * yb).sum() / yb.sum())
        sig = float(math.sqrt(max(((c - mu) ** 2 * yb).sum() / yb.sum(), 1e-12)))
        fwhm = 2.3548 * sig
        r2 = float("nan")

    if fwhm > 0:
        out["L_angstrom"] = float(K * 2.0 * math.pi / fwhm)
        out["fwhm"] = float(fwhm)
        out["r2"] = float(r2)
    return out
