"""Orientation and geometry refinement on the PAIR model of a two-component twin (PREREGISTER A27).

A blended pair of twin components is ONE blob whose centroid is the intensity-weighted mean of the two components' positions, ``(1 - w) p1 + w p2``,
not the position of the first component. Refining U on the first component's position (what :func:`midas_sx.orient.index_and_refine` does) leaves
a bias of ``w (p2 - p1)`` per blended blob. Here the blob is modelled as ``(1 - w) p1(U, h) + w p2(U Rz(+90), T h)`` with ``w = m`` for a blend, 0 for a
resolved first component and 1 for a resolved second component; the weight ``m`` is a global parameter of the fit (bounded to [0, 0.5]) and assumes that
``|F(h)|^2 = |F(T h)|^2`` for the blended pair (the structure is unknown at this stage).

What ``m`` is NOT: on simulated data it comes out at 0.06-0.08 for a true intensity fraction of 0.30. Blob positions come from the core above half the maximum,
not from an intensity moment, and the two components of a real blend are separated mostly in omega, so the centroid shift per blend is far smaller than
``m (p2 - p1)``. Read ``m`` as a nuisance blend shift; the gain in orientation (0.049 -> 0.0055 deg at b/a 1.011, 0.029 -> 0.021 deg at 1.03,
PREREGISTER A27a) comes from modelling the resolved components and a free blend shift, not from a physical intensity weighting.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from midas_defect.geometry import Geometry
from midas_hkls import Lattice

from .geometry import Scan
from .orient import OrientationResult, _unpack, predict_positions
from .predict import TWIN_90_C, twin_targets

__all__ = ["RZ90", "refine_orientation_pairs", "index_and_refine_twin"]

RZ90 = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()


def _pair_positions(U, geom, lattice, scan, hkl, partner, hint, w2):
    n = len(hkl)
    fr = np.full(n, np.nan); ro = np.full(n, np.nan); co = np.full(n, np.nan); ok = np.zeros(n, bool)
    one = w2 < 1.0                                           # any first-component weight
    two = w2 > 0.0
    if one.any():
        a = predict_positions(U, lattice, hkl[one], hint[one], geom, scan)
        f1, r1, c1, k1 = (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool))
        f1[one], r1[one], c1[one], k1[one] = a
    else:
        f1 = r1 = c1 = np.full(n, np.nan); k1 = np.zeros(n, bool)
    if two.any():
        b = predict_positions(U @ RZ90, lattice, partner[two], hint[two], geom, scan)
        f2, r2, c2, k2 = (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool))
        f2[two], r2[two], c2[two], k2[two] = b
    else:
        f2 = r2 = c2 = np.full(n, np.nan); k2 = np.zeros(n, bool)
    only1, only2, both = w2 <= 0.0, w2 >= 1.0, (w2 > 0.0) & (w2 < 1.0)
    fr = np.where(only1, f1, np.where(only2, f2, (1 - w2) * f1 + w2 * f2))
    ro = np.where(only1, r1, np.where(only2, r2, (1 - w2) * r1 + w2 * r2))
    co = np.where(only1, c1, np.where(only2, c2, (1 - w2) * c1 + w2 * c2))
    ok = np.where(only1, k1, np.where(only2, k2, k1 & k2))
    return fr, ro, co, ok


def refine_orientation_pairs(U0, geom0: Geometry, lattice: Lattice, scan: Scan, hkl, partner, kind, fr_obs, ro_obs, co_obs, *, m0: float = 0.3,
                             sigma_px: float = 0.25, sigma_frame: float = 0.3, refine_geometry: bool = True, f_scale: float = 3.0, max_nfev: int = 100):
    """One joint fit of U, the geometry and the blend weight m on the pair model. ``kind``: 0 = first component alone, 1 = blend, 2 = second alone.

    Returns ``(U, geom, m, info)`` with the same info keys as :func:`midas_sx.orient.refine_orientation` (robust rms, outliers, success)."""
    hkl, partner, kind = np.asarray(hkl, int), np.asarray(partner, int), np.asarray(kind, int)
    n = len(hkl)
    if n < 12:
        raise ValueError(f"only {n} assigned blobs; need >= 12")
    hint = scan.omega_of(np.asarray(fr_obs, float))
    has_blend = bool((kind == 1).any())
    npar = 6 if refine_geometry else 3
    npar_all = npar + (1 if has_blend else 0)

    def resid(p):
        pf = np.zeros(6); pf[:npar] = p[:npar]
        U, g = _unpack(pf, U0, geom0)
        m = float(p[npar]) if has_blend else m0
        w2 = np.where(kind == 1, m, np.where(kind == 2, 1.0, 0.0))
        fr, ro, co, ok = _pair_positions(U, g, lattice, scan, hkl, partner, hint, w2)
        r = np.concatenate([(ro - ro_obs) / sigma_px, (co - co_obs) / sigma_px, (fr - fr_obs) / sigma_frame])
        r[~np.tile(ok, 3)] = 10.0 * f_scale
        return r

    x0 = np.zeros(npar_all)
    lo = np.full(npar_all, -np.inf); hi = np.full(npar_all, np.inf)
    if has_blend:
        x0[npar] = m0; lo[npar], hi[npar] = 0.0, 0.5
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=f_scale, method="trf", bounds=(lo, hi), x_scale=1.0, diff_step=1e-5, max_nfev=max_nfev)
    pf = np.zeros(6); pf[:npar] = sol.x[:npar]
    U, g = _unpack(pf, U0, geom0)
    m = float(sol.x[npar]) if has_blend else m0
    r = resid(sol.x)
    rp, rf = r[:2 * n] * sigma_px, r[2 * n:] * sigma_frame
    rob_px, rob_fr = float(1.4826 * np.median(np.abs(rp))), float(1.4826 * np.median(np.abs(rf)))
    n_out = int(np.sum((np.abs(r[:n]) * sigma_px > 4 * max(rob_px, 1e-9)) | (np.abs(r[n:2 * n]) * sigma_px > 4 * max(rob_px, 1e-9))
                       | (np.abs(r[2 * n:]) * sigma_frame > 4 * max(rob_fr, 1e-9))))
    return U, g, m, dict(rms_px=rob_px, rms_frame=rob_fr, rms_px_all=float(np.sqrt(np.mean(rp ** 2))), rms_frame_all=float(np.sqrt(np.mean(rf ** 2))),
                         n_outliers=n_out, success=bool(sol.success), nfev=int(sol.nfev), status=int(sol.status))


def index_and_refine_twin(spots, lattice: Lattice, U0: np.ndarray, geom0: Geometry, scan: Scan, hkl_pool: np.ndarray, *, tols: Sequence[float] = (1.5, 1.0, 0.7),
                          frame_tol: float = 1.5, m0: float = 0.3, d_res_px: float = 6.0, d_res_frames: float = 3.0, refine_geometry: bool = True,
                          twin_matrix=TWIN_90_C, sigma_px: float = 0.25, sigma_frame: float = 0.3, min_blobs: int = 20) -> OrientationResult:
    """Match blobs to the PREDICTED targets of both components, refine on the pair model, tighten the match radius, repeat.

    Intended to follow :func:`midas_sx.orient.index_and_refine` (``U0``, ``geom0`` = its result): the start must already be within a pixel or two.
    ``hkl_pool`` are domain-1 indices (e.g. :func:`midas_sx.predict.enumerate_hkl`). The fitted blend weight is returned as ``result.rounds[-1]['m']``."""
    fr = spots["frame"].to_numpy(float); ro = spots["row"].to_numpy(float); co = spots["col"].to_numpy(float)
    U, geom = np.asarray(U0, float), geom0
    rounds: List[dict] = []
    m = m0
    info = {}
    for tol in tols:
        tg = twin_targets(U, U @ RZ90, lattice, geom, scan, hkl_pool, twin_matrix=twin_matrix, d_res_px=d_res_px, d_res_frames=d_res_frames)
        # position used for matching: the model's own centroid of each target
        w2 = np.where(tg.kind == "blend", m, np.where(tg.kind == "c2", 1.0, 0.0))
        mr = np.where(tg.kind == "c2", tg.row2, np.where(tg.kind == "blend", (1 - w2) * tg.row1 + w2 * tg.row2, tg.row1))
        mc = np.where(tg.kind == "c2", tg.col2, np.where(tg.kind == "blend", (1 - w2) * tg.col1 + w2 * tg.col2, tg.col1))
        mf = np.where(tg.kind == "c2", tg.frame2, np.where(tg.kind == "blend", (1 - w2) * tg.frame1 + w2 * tg.frame2, tg.frame1))
        ok = np.isfinite(mr) & np.isfinite(mc) & np.isfinite(mf)
        tree = cKDTree(np.c_[mr[ok], mc[ok]])
        d, j = tree.query(np.c_[ro, co], distance_upper_bound=tol)
        idx_t = np.flatnonzero(ok)
        take = np.isfinite(d)
        take &= np.abs(mf[ok][np.minimum(j, ok.sum() - 1)] - fr) < frame_tol
        # one blob per target: keep the nearest
        best = {}
        for b in np.flatnonzero(take):
            t = idx_t[j[b]]
            if t not in best or d[b] < d[best[t]]:
                best[t] = b
        pairs = [(t, b) for t, b in best.items()]
        if len(pairs) < min_blobs:
            rounds.append(dict(tol=tol, n_used=len(pairs), m=m))
            return OrientationResult(U, geom, np.full(3, np.inf), float("nan"), float("nan"), len(pairs), len(fr), False,
                                     f"only {len(pairs)} blobs matched at {tol} px", 0, rounds)
        ti = np.array([t for t, _ in pairs]); bi = np.array([b for _, b in pairs])
        kind = np.where(tg.kind.to_numpy()[ti] == "blend", 1, np.where(tg.kind.to_numpy()[ti] == "c2", 2, 0))
        hkl = tg[["h", "k", "l"]].to_numpy(int)[ti]; part = tg[["p0", "p1", "p2"]].to_numpy(int)[ti]
        U, geom, m, info = refine_orientation_pairs(U, geom, lattice, scan, hkl, part, kind, fr[bi], ro[bi], co[bi], m0=m, sigma_px=sigma_px,
                                                    sigma_frame=sigma_frame, refine_geometry=refine_geometry)
        rounds.append(dict(tol=tol, n_used=len(pairs), n_blend=int((kind == 1).sum()), n_c1=int((kind == 0).sum()), n_c2=int((kind == 2).sum()),
                           rms_px=info["rms_px"], rms_frame=info["rms_frame"], m=m, nfev=info["nfev"]))
    ok_ = info["success"] and info["rms_px"] <= 2.0 and info["rms_frame"] <= 1.0
    return OrientationResult(U, geom, np.full(3, np.nan), info["rms_px"], info["rms_frame"], len(pairs), len(fr), bool(ok_), "ok" if ok_ else "residuals above the gate",
                             info["nfev"], rounds, rms_px_all=info["rms_px_all"], rms_frame_all=info["rms_frame_all"], n_outliers=info["n_outliers"])
