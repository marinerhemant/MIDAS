"""Texture-safe polytype satellite excess + discrete-vs-relrod discriminator.

Replaces ``polytype.satellite_intensity.polytype_satellite_enhancement``, whose
"enhancement" (sum along axis / mean sum over random directions) was dominated by
voxel count, not intensity, and inflated the demk satellite by ~140x (700-1600x vs a
real ~5x; AUDIT_2026-06-23.md).

Geometry-honest definition: the radial **excess**

    e(q) = <I>_on-axis(q) / <I>_off-axis(q)

is a per-voxel MEAN-intensity ratio (count-normalised) between voxels within
``tube_deg`` of any <hkl> axis and voxels >= ``off_deg`` from every <hkl> axis, at the
SAME |q|. Background is taken at the same |q| (not a fixed empty radius), so detector
sampling density and texture cancel.

Built-in discriminator (the null the old metric lacked): a real periodic 9R has e
PEAKED at the thirds (G/3, 2G/3) and DIPPING to ~background at the half-integers
(G/6, G/2, 5G/6). A continuous ISF <111>* relrod rises monotonically inward from the
(111) Bragg. The verdict is returned alongside the numbers.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["satellite_radial_excess"]


def _max_abs_dot(qhat: NDArray[np.floating], dirs: NDArray[np.floating],
                 chunk: int = 1000) -> NDArray[np.floating]:
    """Max |qhat . d| over all axis directions d, chunked over directions."""
    m = np.zeros(qhat.shape[0], dtype=np.float64)
    for c0 in range(0, dirs.shape[0], chunk):
        m = np.maximum(m, np.abs(qhat @ dirs[c0:c0 + chunk].T).max(axis=1))
    return m


def satellite_radial_excess(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    axis_dirs: NDArray[np.floating],
    G_magnitude: float,
    *,
    tube_deg: float = 2.5,
    off_deg: float = 10.0,
    dq: float = 0.02,
    q_lo: float = 0.2,
    q_hi: float | None = None,
) -> dict:
    """Radial excess e(q)=<I>_on-axis/<I>_off-axis along the <hkl> axes, with controls.

    Parameters
    ----------
    qs, vals : (N,3), (N,)  sample-frame q-vectors and per-voxel intensity.
    axis_dirs : (M,3) the on-axis directions (e.g. all grain <111>, sample frame).
        Folded to a hemisphere internally; magnitudes ignored.
    G_magnitude : |G| of the parent Bragg (so satellites are at G/3, 2G/3).
    tube_deg : on-axis angular tolerance. off_deg : background must be this far off-axis.
    dq : radial bin width (1/A). q_lo, q_hi : radial range (q_hi defaults to G+0.2).

    Returns
    -------
    dict with
        q (n_bins,), excess (n_bins,), mean_on, mean_off, n_on, n_off
        at_positions : {label: e} for G/6,G/3,G/2,2G/3,5G/6,(111)
        verdict : "9R-periodic" | "ISF-relrod" | "ambiguous"
        monotone_inward : bool  (True => relrod signature)
    """
    qs = np.asarray(qs, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    D = np.asarray(axis_dirs, dtype=np.float64).copy()
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    D[D[:, 2] < 0] *= -1.0
    if q_hi is None:
        q_hi = G_magnitude + 0.2

    qmag = np.linalg.norm(qs, axis=1)
    sel = (qmag > q_lo) & (qmag < q_hi) & (vals > 0) & np.isfinite(vals)
    q = qs[sel]
    qm = qmag[sel]
    w = vals[sel]
    qhat = q / qm[:, None]
    ang = np.degrees(np.arccos(np.clip(_max_abs_dot(qhat, D), 0.0, 1.0)))
    on = ang <= tube_deg
    off = ang >= off_deg

    edges = np.arange(q_lo, q_hi + dq, dq)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    nb = len(ctr)

    def _binmean(mask):
        bi = np.clip(np.digitize(qm[mask], edges) - 1, 0, nb - 1)
        Isum = np.bincount(bi, weights=w[mask], minlength=nb)
        cnt = np.bincount(bi, minlength=nb).astype(np.float64)
        mean = np.where(cnt > 0, Isum / np.maximum(cnt, 1.0), np.nan)
        return mean, cnt

    mean_on, n_on = _binmean(on)
    mean_off, n_off = _binmean(off)
    with np.errstate(invalid="ignore", divide="ignore"):
        excess = mean_on / mean_off

    def at(qq):
        return float(excess[int(np.clip(np.argmin(np.abs(ctr - qq)), 0, nb - 1))])

    G = G_magnitude
    pos = {
        "G/6": at(G / 6), "G/3*": at(G / 3), "G/2": at(G / 2),
        "2G/3*": at(2 * G / 3), "5G/6": at(5 * G / 6), "(111)": at(G),
    }
    eG3, e2G3 = pos["G/3*"], pos["2G/3*"]
    eG6, eG2, e5G6 = pos["G/6"], pos["G/2"], pos["5G/6"]
    # discrete 9R: thirds exceed their flanking controls
    discrete = (e2G3 > e5G6) and (e2G3 > eG2) and (eG3 > eG2) and (eG3 > eG6)
    monotone_inward = (e5G6 >= e2G3 >= eG2 >= eG3 >= eG6)
    verdict = ("9R-periodic" if discrete
               else "ISF-relrod" if monotone_inward else "ambiguous")

    return {
        "q": ctr, "excess": excess, "mean_on": mean_on, "mean_off": mean_off,
        "n_on": n_on, "n_off": n_off, "at_positions": pos,
        "verdict": verdict, "monotone_inward": bool(monotone_inward),
    }
