"""Duplicate spots and relative alignment of overlapping detector panels.

When two panels overlap, one diffraction spot is recorded twice. This module
finds those pairs and, from the same pairs, measures how well the two panels
agree. Both come from one observation: a reflection seen on panel A and on
panel B is the SAME ray, so after each panel's own geometry has turned pixels
into lab coordinates the two records must carry the same (ring, eta, 2theta,
omega). Where they do not, the panels are misaligned, and the scatter of the
differences is the uncertainty of the panel geometry.

What is reported, per panel pair (a, b):

``n_pairs``
    mutual-nearest matches inside the data-driven window.
``d_eta, d_tth, d_ome`` (median, robust sigma)
    differences b - a in direction space. A median away from zero is a
    systematic offset (beam centre, tilt or Lsd of one panel); the sigma is the
    measurement scatter.
``translation_um``, ``scale``, ``rotation_mrad`` (+ standard errors)
    the 2-D similarity transform that maps panel b's lab positions onto panel
    a's, fitted to the pairs. Only the RELATIVE misalignment is observable: it
    says the two panels disagree, not which one is wrong.

Matching is done in lab position space through the fitted transform, not by a
constant (eta, 2theta) offset: a translation of one panel shifts eta and 2theta
by amounts that depend on where on the ring the spot is, so a fixed offset
cannot follow it. The window is not assumed either: it starts generous, the
transform is fitted to what it finds, and the window is reset to ``n_sigma``
robust sigmas of the residual (and likewise for omega), iterating until stable,
so the tolerance reported is the one the data support.

Pure numpy; no scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# InputAll.csv columns (0-based).
C_Y, C_Z, C_OME, C_RAD, C_ID, C_RING, C_ETA, C_TTH = range(8)


def _wrap(d: np.ndarray, period: float = 360.0) -> np.ndarray:
    return (d + period / 2.0) % period - period / 2.0


def _robust(x: np.ndarray) -> Tuple[float, float]:
    """(median, 1.4826 * MAD)."""
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(np.median(x))
    return m, float(1.4826 * np.median(np.abs(x - m)))


@dataclass
class PairStats:
    """Everything measured for one ordered panel pair (a, b)."""

    det_a: int
    det_b: int
    n_pairs: int = 0
    n_a: int = 0
    n_b: int = 0
    # (median, robust sigma) of b - a
    d_eta: Tuple[float, float] = (float("nan"), float("nan"))
    d_tth: Tuple[float, float] = (float("nan"), float("nan"))
    d_ome: Tuple[float, float] = (float("nan"), float("nan"))
    # window actually used for the final match (n_sigma robust sigmas)
    tol_pos_um: float = float("nan")
    tol_ome: float = float("nan")
    # similarity transform b -> a on the plane of a: p_a ~ t + (1+s) R(rho) p_b
    translation_um: Tuple[float, float] = (float("nan"), float("nan"))
    translation_se_um: Tuple[float, float] = (float("nan"), float("nan"))
    scale: float = float("nan")
    scale_se: float = float("nan")
    rotation_mrad: float = float("nan")
    rotation_se_mrad: float = float("nan")
    position_sigma_um: float = float("nan")
    condition: float = float("nan")
    idx_a: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64),
                              repr=False)
    idx_b: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64),
                              repr=False)


def _apply(p: np.ndarray, th: Tuple[float, float, float, float]) -> np.ndarray:
    """Map panel-b positions onto panel a: p + t + s*p + rho*(-z, y)."""
    ty, tz, sc, rho = th
    out = p.copy()
    out[:, 0] = p[:, 0] + ty + sc * p[:, 0] - rho * p[:, 1]
    out[:, 1] = p[:, 1] + tz + sc * p[:, 1] + rho * p[:, 0]
    return out


def _best_partner(pa: np.ndarray, oa: np.ndarray, pb: np.ndarray, ob: np.ndarray,
                  rad_um: float, ome_tol: float) -> np.ndarray:
    """For each a-row, the index of its nearest b-row inside the window, or -1.

    ``pb`` is already mapped into a's frame. Rows must share the ring. Nearest
    = smallest sqrt((|dp|/rad)^2 + (dome/ome_tol)^2).
    """
    out = np.full(pa.shape[0], -1, dtype=np.int64)
    if pa.shape[0] == 0 or pb.shape[0] == 0:
        return out
    order = np.argsort(ob)
    bo = ob[order]
    for i in range(pa.shape[0]):
        lo = np.searchsorted(bo, oa[i] - ome_tol, side="left")
        hi = np.searchsorted(bo, oa[i] + ome_tol, side="right")
        cand = np.arange(lo, hi)
        if oa[i] - ome_tol < -180.0:       # omega wraps at +-180
            cand = np.concatenate([cand, np.arange(
                np.searchsorted(bo, oa[i] - ome_tol + 360.0, side="left"), len(bo))])
        if oa[i] + ome_tol > 180.0:
            cand = np.concatenate([cand, np.arange(
                0, np.searchsorted(bo, oa[i] + ome_tol - 360.0, side="right"))])
        if cand.size == 0:
            continue
        j = order[cand]
        dp = np.hypot(pb[j, 0] - pa[i, 0], pb[j, 1] - pa[i, 1])
        do = np.abs(_wrap(ob[j] - oa[i]))
        ok = (dp <= rad_um) & (do <= ome_tol)
        if not ok.any():
            continue
        cost = np.where(ok, (dp / rad_um) ** 2 + (do / ome_tol) ** 2, np.inf)
        out[i] = j[int(np.argmin(cost))]
    return out


def _mutual_pairs(a: np.ndarray, b: np.ndarray, th: Tuple[float, float, float, float],
                  lsd_ratio: float, rad_um: float,
                  ome_tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mutual-nearest (index in a, index in b) pairs, ring by ring."""
    ia: List[int] = []
    ib: List[int] = []
    pb_all = _apply(b[:, [C_Y, C_Z]] * lsd_ratio, th)
    pa_all = a[:, [C_Y, C_Z]]
    for r in np.intersect1d(np.unique(a[:, C_RING]), np.unique(b[:, C_RING])):
        ra = np.flatnonzero(a[:, C_RING] == r)
        rb = np.flatnonzero(b[:, C_RING] == r)
        fwd = _best_partner(pa_all[ra], a[ra, C_OME], pb_all[rb], b[rb, C_OME],
                            rad_um, ome_tol)
        rev = _best_partner(pb_all[rb], b[rb, C_OME], pa_all[ra], a[ra, C_OME],
                            rad_um, ome_tol)
        for ii, jj in enumerate(fwd):
            if jj >= 0 and rev[jj] == ii:
                ia.append(int(ra[ii]))
                ib.append(int(rb[jj]))
    return np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64)


def _similarity_fit(pa: np.ndarray, pb: np.ndarray) -> Dict[str, float]:
    """Least-squares  pa ~ t + s*pb + rho*J*pb  (small s, rho), with errors.

    Model: d = pa - pb = t + s*pb + rho*(J pb), J = [[0,-1],[1,0]]. Linear in
    (ty, tz, s, rho), so ordinary least squares with an analytic covariance.
    """
    n = pa.shape[0]
    nan = float("nan")
    if n < 6:
        return dict(ty=nan, tz=nan, s=nan, rho=nan, ty_se=nan, tz_se=nan, s_se=nan,
                    rho_se=nan, sigma=nan, cond=nan)
    d = (pa - pb).reshape(-1)                      # (2n,) y0 z0 y1 z1 ...
    A = np.zeros((2 * n, 4))
    A[0::2, 0] = 1.0                               # ty
    A[1::2, 1] = 1.0                               # tz
    A[0::2, 2] = pb[:, 0]                          # s * y
    A[1::2, 2] = pb[:, 1]
    A[0::2, 3] = -pb[:, 1]                         # rho * (-z, y)
    A[1::2, 3] = pb[:, 0]
    # Column scaling so the condition number reflects geometry, not units.
    sc = np.linalg.norm(A, axis=0)
    sc[sc == 0] = 1.0
    As = A / sc
    cond = float(np.linalg.cond(As))
    sol, *_ = np.linalg.lstsq(As, d, rcond=None)
    sol = sol / sc
    res = d - A @ sol
    dof = max(2 * n - 4, 1)
    sigma = float(np.sqrt(res @ res / dof))
    try:
        cov = sigma ** 2 * np.linalg.inv(As.T @ As)
        cov = cov / np.outer(sc, sc)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(4, nan)
    return dict(ty=float(sol[0]), tz=float(sol[1]), s=float(sol[2]), rho=float(sol[3]),
                ty_se=float(se[0]), tz_se=float(se[1]), s_se=float(se[2]),
                rho_se=float(se[3]), sigma=sigma, cond=cond)


def analyse_pair(a: np.ndarray, b: np.ndarray, det_a: int, det_b: int, *,
                 lsd_a: float, lsd_b: float, omega_step_deg: Optional[float] = None,
                 n_sigma: float = 4.0, start_radius_um: float = 5000.0,
                 pos_floor_um: float = 5.0) -> PairStats:
    """Duplicates and relative alignment of two panels' spot tables.

    ``a`` and ``b`` are ``(n, >=8)`` arrays in InputAll.csv column order.
    ``start_radius_um`` is the generous first window; the final window is
    ``n_sigma`` robust sigmas of the fitted residual, never below
    ``pos_floor_um``.
    """
    st = PairStats(det_a=det_a, det_b=det_b, n_a=a.shape[0], n_b=b.shape[0])
    step = abs(omega_step_deg) if omega_step_deg else 0.5
    ome_floor = step / np.sqrt(12.0)
    ratio = lsd_a / lsd_b
    th = (0.0, 0.0, 0.0, 0.0)
    rad, ome_tol = start_radius_um, 1.5 * step
    ia = ib = np.zeros(0, dtype=np.int64)
    fit: Dict[str, float] = {}
    for _ in range(8):
        ia, ib = _mutual_pairs(a, b, th, ratio, rad, ome_tol)
        if ia.size < 6:
            return st
        pa = a[ia][:, [C_Y, C_Z]]
        pb = b[ib][:, [C_Y, C_Z]] * ratio
        # fit, drop the worst residuals once (a chance match must not drag it)
        fit = _similarity_fit(pa, pb)
        th_new = (fit["ty"], fit["tz"], fit["s"], fit["rho"])
        # Scatter per COMPONENT (y and z): the magnitude of a 2-D Gaussian is
        # Rayleigh, whose MAD-sigma is ~0.67 of the component sigma and would
        # clip real pairs.
        d2 = pa - _apply(pb, th_new)
        res = np.hypot(d2[:, 0], d2[:, 1])
        _, sres = _robust(np.concatenate([d2[:, 0], d2[:, 1]]))
        keep = res <= max(3.0 * np.sqrt(2.0) * sres, pos_floor_um) \
            if np.isfinite(sres) else np.ones_like(res, bool)
        if keep.sum() >= 6 and keep.sum() < res.size:
            fit = _similarity_fit(pa[keep], pb[keep])
            th_new = (fit["ty"], fit["tz"], fit["s"], fit["rho"])
            d2 = pa - _apply(pb, th_new)
            _, sres = _robust(np.concatenate([d2[keep, 0], d2[keep, 1]]))
        do = np.abs(_wrap(b[ib, C_OME] - a[ia, C_OME]))
        _, so = _robust(do)
        rad_new = min(rad, n_sigma * max(sres, pos_floor_um)) if np.isfinite(sres) else rad
        ome_new = min(1.5 * step, n_sigma * max(so if np.isfinite(so) else 0.0, ome_floor))
        moved = max(abs(th_new[0] - th[0]), abs(th_new[1] - th[1])) > 0.5 * pos_floor_um
        th = th_new
        if not moved and abs(rad_new - rad) < 0.05 * rad and abs(ome_new - ome_tol) < 1e-9:
            rad, ome_tol = rad_new, ome_new
            break
        rad, ome_tol = rad_new, ome_new
    ia, ib = _mutual_pairs(a, b, th, ratio, rad, ome_tol)
    if ia.size < 6:
        return st
    pa = a[ia][:, [C_Y, C_Z]]
    pb = b[ib][:, [C_Y, C_Z]] * ratio
    fit = _similarity_fit(pa, pb)
    st.n_pairs = int(ia.size)
    st.idx_a, st.idx_b = ia, ib
    st.d_eta = _robust(_wrap(b[ib, C_ETA] - a[ia, C_ETA]))
    st.d_tth = _robust(b[ib, C_TTH] - a[ia, C_TTH])
    st.d_ome = _robust(_wrap(b[ib, C_OME] - a[ia, C_OME]))
    st.tol_pos_um, st.tol_ome = float(rad), float(ome_tol)
    st.translation_um = (fit["ty"], fit["tz"])
    st.translation_se_um = (fit["ty_se"], fit["tz_se"])
    st.scale, st.scale_se = fit["s"], fit["s_se"]
    st.rotation_mrad, st.rotation_se_mrad = 1e3 * fit["rho"], 1e3 * fit["rho_se"]
    st.position_sigma_um, st.condition = fit["sigma"], fit["cond"]
    return st


def analyse_panels(spots: Sequence[np.ndarray], det_ids: Sequence[int],
                   lsd: Sequence[float], *, omega_step_deg: Optional[float] = None,
                   n_sigma: float = 4.0) -> List[PairStats]:
    """Every panel pair (a < b) that shares at least one duplicate."""
    out: List[PairStats] = []
    n = len(spots)
    for i in range(n):
        for j in range(i + 1, n):
            st = analyse_pair(spots[i], spots[j], int(det_ids[i]), int(det_ids[j]),
                              lsd_a=float(lsd[i]), lsd_b=float(lsd[j]),
                              omega_step_deg=omega_step_deg, n_sigma=n_sigma)
            if st.n_pairs > 0:
                out.append(st)
    return out


def duplicate_masks(spots: Sequence[np.ndarray], pairs: Sequence[PairStats],
                    det_ids: Sequence[int], *,
                    prefer: Optional[Sequence[np.ndarray]] = None) -> List[np.ndarray]:
    """Boolean ``drop`` mask per panel: for each duplicate pair, one record.

    ``prefer[k]`` is a per-row quality score for panel k (higher wins; e.g. the
    distance of the spot from its panel's edge, since an edge spot is cut off).
    Without it the earlier panel wins, which is deterministic and says nothing
    about quality. A spot that is a duplicate of two different partners is
    dropped at most once.
    """
    pos = {int(d): k for k, d in enumerate(det_ids)}
    drop = [np.zeros(s.shape[0], dtype=bool) for s in spots]
    for st in pairs:
        ka, kb = pos[st.det_a], pos[st.det_b]
        for ra, rb in zip(st.idx_a, st.idx_b):
            if drop[ka][ra] or drop[kb][rb]:
                continue
            if prefer is not None and prefer[kb][rb] > prefer[ka][ra]:
                drop[ka][ra] = True
            else:
                drop[kb][rb] = True
    return drop
