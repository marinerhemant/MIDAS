"""Detector tilts, rotation-axis wedge, beam centre and distance from Friedel pairs, with no orientation and no cell.

For a reflection hkl seen at omega_1 and its Friedel mate -h-k-l seen at omega_2 (ANY crossing of each, so also the two crossings that sit only
2 asin(lambda |c| / rho) >= 2 theta_B apart in omega, not just the ones 180 deg apart), the sample-frame vectors recovered from the two spots,

    q_s,i = R_z(-omega_i) R_y(W) (s_i - s0)          (units of 1/k; the package convention q_lab = R_y(-W) R_z(omega) q_s)

satisfy ``q_s,1 + q_s,2 = 0`` exactly, because the first is the crystal-fixed vector of hkl and the second that of -hkl. The relation contains
neither U nor the cell. Only the geometry that maps a pixel to ``s`` (tx, ty, tz, centre, Lsd) and the wedge enter, so the pair residual is a
U-free estimator of them. A common shift of every omega leaves the sum at zero: the omega zero is a gauge and is not fitted.

The estimator pairs spots through their hkl LABELS (which come from an orientation fit); a wrong label only makes an outlier, and no value
of U enters the residual.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from midas_defect.geometry import Geometry, qlab_to_qsample

from .geometry import spot_directions

__all__ = ["FRIEDEL_PARAMS", "labelled_pairs", "pair_residual", "fit_pairs", "FriedelFit", "with_params", "fit_multirun", "MultirunFit"]

FRIEDEL_PARAMS = ("tx", "ty", "tz", "wedge", "bcz", "bcy", "lsd")        # deg, deg, deg, deg, px, px, mm
_SCALE = {"tx": 0.1, "ty": 0.1, "tz": 0.1, "wedge": 0.1, "bcz": 1.0, "bcy": 1.0, "lsd": 1.0}
_FIELD = {"tx": "tx_deg", "ty": "ty_deg", "tz": "tz_deg", "wedge": "wedge_deg", "bcz": "bcz_px", "bcy": "bcy_px", "lsd": "lsd_um"}


def with_params(geom: Geometry, values: dict) -> Geometry:
    """``geom`` with the named parameters (units of :data:`FRIEDEL_PARAMS`) replaced."""
    kw = {_FIELD[k]: (v * 1000.0 if k == "lsd" else float(v)) for k, v in values.items()}
    return replace(geom, **kw)


def get_param(geom: Geometry, name: str) -> float:
    v = getattr(geom, _FIELD[name])
    return v / 1000.0 if name == "lsd" else float(v)


def labelled_pairs(obs: pd.DataFrame, *, max_hkl_residual: float = 0.1, omega_col: str = "omega_meas") -> pd.DataFrame:
    """Every (crossing of hkl, crossing of -hkl) pair inside one table of observations (one run, one crystal).

    ``obs`` needs ``h k l row col`` and the MEASURED omega column. Columns returned: ``row1 col1 om1 row2 col2 om2 group h k l`` with ``group`` the id of
    the unordered pair {hkl, -hkl} (the bootstrap resamples groups, since one spot can sit in several pairs).
    """
    o = obs
    if "hkl_residual" in o:
        o = o[o.hkl_residual <= max_hkl_residual]
    o = o.reset_index(drop=True)
    idx = {}
    for i, key in enumerate(zip(o.h.to_numpy(int), o.k.to_numpy(int), o.l.to_numpy(int))):
        idx.setdefault(key, []).append(i)
    rows = []
    for g, (key, ii) in enumerate(sorted(k for k in idx.items() if k[0] > (-k[0][0], -k[0][1], -k[0][2]))):
        mate = (-key[0], -key[1], -key[2])
        if mate not in idx or not any(key):
            continue
        for i in ii:
            for j in idx[mate]:
                rows.append((o.row[i], o.col[i], o[omega_col][i], o.row[j], o.col[j], o[omega_col][j], g, key[0], key[1], key[2]))
    return pd.DataFrame(rows, columns=["row1", "col1", "om1", "row2", "col2", "om2", "group", "h", "k", "l"])


def _qs(row, col, om_deg, geom: Geometry) -> np.ndarray:
    import torch
    s, _, _ = spot_directions(row, col, geom)
    q = s.copy()
    q[:, 0] -= 1.0                                                   # s - s0, s0 = +X
    qs = qlab_to_qsample(torch.as_tensor(q, dtype=torch.float64), torch.as_tensor(np.radians(np.asarray(om_deg, float)), dtype=torch.float64),
                         float(np.radians(geom.wedge_deg)))
    return qs.detach().cpu().numpy()


def pair_residual(geom: Geometry, pairs: pd.DataFrame, *, normalise: bool = True) -> np.ndarray:
    """(N, 3) ``q_s,1 + q_s,2`` for each pair, divided by the pair's mean |q| (dimensionless, about the angle in radians between q_s,1 and -q_s,2).

    The division is not cosmetic. The raw sum is driven to zero by ANY geometry that shrinks every q (Lsd -> infinity puts every s at s0), a degenerate
    minimum that fits scrambled mates as well as true ones; relative to |q| a shrink gains nothing (shrink placebo, test_scrambled_mates_do_not_recover_it).
    """
    a = _qs(pairs.row1.to_numpy(float), pairs.col1.to_numpy(float), pairs.om1.to_numpy(float), geom)
    b = _qs(pairs.row2.to_numpy(float), pairs.col2.to_numpy(float), pairs.om2.to_numpy(float), geom)
    r = a + b
    if normalise:
        r = r / np.maximum(0.5 * (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)), 1e-9)[:, None]
    return r


@dataclass
class FriedelFit:
    names: Tuple[str, ...]
    start: np.ndarray
    values: np.ndarray
    sigma_formal: np.ndarray           # from the Jacobian and the robust residual scale (pairs treated as independent)
    sigma_boot: Optional[np.ndarray]   # bootstrap over hkl groups (NaN entries when not run)
    boot: Optional[np.ndarray]         # (n_boot, n_params)
    singular_values: np.ndarray        # of the column-scaled Jacobian (units: 0.1 deg, 1 px, 1 mm)
    weakest_direction: np.ndarray      # right singular vector of the smallest singular value, in scaled parameters
    n_pairs: int
    n_groups: int
    rms_start: float
    rms_final: float
    scale: float                       # robust residual sd (1/k units) used for sigma_formal
    success: bool
    n_trimmed: int = 0                 # pairs removed as gross outliers (wrong labels, mismatched spots) before the final fit

    def table(self) -> pd.DataFrame:
        d = dict(param=self.names, start=self.start, value=self.values, sigma_formal=self.sigma_formal)
        if self.sigma_boot is not None:
            d["sigma_boot"] = self.sigma_boot
        return pd.DataFrame(d)


def _free_vector(geom: Geometry, names: Sequence[str]) -> np.ndarray:
    return np.array([get_param(geom, n) for n in names], float)


def fit_pairs(pairs: pd.DataFrame, geom0: Geometry, *, free: Sequence[str] = FRIEDEL_PARAMS, n_boot: int = 0, seed: int = 0,
              loss: str = "soft_l1", max_nfev: int = 200, trim: Optional[float] = 5.0) -> FriedelFit:
    """Fit the geometry ``free`` (names from :data:`FRIEDEL_PARAMS`) to the pair residuals, starting at ``geom0``.

    Robust least squares (the soft-L1 knee is set from the first pass: three robust sd); pairs whose residual norm exceeds ``trim`` x sqrt(3) robust sd
    after a pass are dropped (a wrong hkl label gives a residual of order 1, the noise is of order 1e-3) and the fit is repeated on the rest; ``trim=None``
    keeps every pair. ``sigma_formal`` is
    ``scale * sqrt(diag((J^T J)^-1))`` (pseudo-inverse when the scaled Jacobian is rank deficient); ``sigma_boot`` resamples hkl groups.
    """
    free = tuple(free)
    sc = np.array([_SCALE[n] for n in free])
    x0 = _free_vector(geom0, free)

    def resid(x, pr):
        return pair_residual(with_params(geom0, dict(zip(free, x * sc))), pr).ravel()

    def jac(x, pr, h=1e-3):
        cols = []
        for i in range(len(x)):
            d = np.zeros_like(x); d[i] = h
            cols.append((resid(x + d, pr) - resid(x - d, pr)) / (2 * h))
        return np.array(cols).T

    def solve(pr, xs, f_scale):
        return least_squares(resid, xs, jac=jac, args=(pr,), loss=loss, f_scale=f_scale, max_nfev=max_nfev, x_scale=1.0)

    xs = x0 / sc
    r0 = resid(xs, pairs)
    s1 = solve(pairs, xs, 3e-2)
    n_trim = 0
    knee = 3e-2
    for _ in range(2):                                         # pass 1 at a loose knee, trim, pass 2 at the measured knee, trim again
        r1 = s1.fun
        knee = 3.0 * 1.4826 * float(np.median(np.abs(r1 - np.median(r1)))) + 1e-9
        if trim is not None:
            bad = np.linalg.norm(r1.reshape(-1, 3), axis=1) > trim * np.sqrt(3.0) * (knee / 3.0)
            if bad.any() and (~bad).sum() >= 4 * len(free):
                n_trim += int(bad.sum()); pairs = pairs[~bad].reset_index(drop=True)
        s1 = solve(pairs, s1.x, knee)
    s2 = s1
    r2 = s2.fun
    scale = 1.4826 * float(np.median(np.abs(r2 - np.median(r2)))) + 1e-12
    J = jac(s2.x, pairs)
    u, sv, vt = np.linalg.svd(J, full_matrices=False)
    cov = (vt.T * (1.0 / np.maximum(sv, 1e-12 * sv.max()) ** 2)) @ vt * scale ** 2
    sig = np.sqrt(np.diag(cov)) * sc
    boot = None
    sb = None
    if n_boot:
        rng = np.random.default_rng(seed)
        groups = pairs.group.to_numpy()
        ug = np.unique(groups)
        by = {g: np.flatnonzero(groups == g) for g in ug}
        boot = np.full((n_boot, len(free)), np.nan)
        for b in range(n_boot):
            pick = rng.choice(ug, size=len(ug), replace=True)
            pr = pairs.iloc[np.concatenate([by[g] for g in pick])].reset_index(drop=True)
            try:
                boot[b] = solve(pr, s2.x, knee).x * sc
            except Exception:                                      # a degenerate resample: leave NaN
                pass
        sb = np.nanstd(boot, axis=0)
    return FriedelFit(names=free, start=x0, values=s2.x * sc, sigma_formal=sig, sigma_boot=sb, boot=boot, singular_values=sv,
                      weakest_direction=vt[-1], n_pairs=len(pairs), n_groups=int(pairs.group.nunique()),
                      rms_start=float(np.sqrt(np.mean(r0 ** 2))), rms_final=float(np.sqrt(np.mean(r2 ** 2))), scale=scale, success=bool(s2.success), n_trimmed=n_trim)


# ----------------------------------------------------------------------------------------------------------------------------------------------
# several runs of ONE crystal at different phi (shared detector geometry, unknown rotation of the sample frame per run)
# ----------------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class MultirunFit:
    names: Tuple[str, ...]
    values: np.ndarray
    sigma_formal: np.ndarray
    sigma_boot: Optional[np.ndarray]
    rotvecs: np.ndarray                # (n_runs - 1, 3) rotation vectors (rad) of P_k, run 0 is the reference
    rms_final: float
    scale: float
    n_groups: int
    n_groups_trimmed: int
    singular_values: np.ndarray
    n_cross_run_groups: int
    per_run: Tuple[str, ...] = ()
    dev: Optional[np.ndarray] = None            # (n_runs - 1, len(per_run)): run k's value minus run 0's, in the units of FRIEDEL_PARAMS
    sigma_dev: Optional[np.ndarray] = None      # bootstrap sd of dev (formal when no bootstrap)

    def table(self) -> pd.DataFrame:
        d = dict(param=self.names, value=self.values, sigma_formal=self.sigma_formal)
        if self.sigma_boot is not None:
            d["sigma_boot"] = self.sigma_boot
        return pd.DataFrame(d)


def _signed_groups(obs_runs: Sequence[pd.DataFrame], max_hkl_residual: float, omega_col: str) -> pd.DataFrame:
    tabs = []
    for k, o in enumerate(obs_runs):
        if "hkl_residual" in o:
            o = o[o.hkl_residual <= max_hkl_residual]
        t = o[["h", "k", "l", "row", "col", omega_col]].rename(columns={omega_col: "om"}).copy()
        t["run"] = k
        tabs.append(t)
    S = pd.concat(tabs, ignore_index=True)
    hk = S[["h", "k", "l"]].to_numpy(int)
    S = S[np.any(hk != 0, axis=1)].reset_index(drop=True)
    hk = S[["h", "k", "l"]].to_numpy(int)
    canon_pos = np.array([tuple(a) > tuple(-a) for a in hk])
    canon = np.where(canon_pos[:, None], hk, -hk)
    S["sign"] = np.where(canon_pos, 1.0, -1.0)
    S["gkey"] = [tuple(c) for c in canon]
    has_both = S.groupby("gkey").sign.transform(lambda x: bool((x > 0).any() and (x < 0).any()))
    S = S[has_both].reset_index(drop=True)
    S["gid"] = S.gkey.astype("category").cat.codes
    return S


def _group_arrays(S: pd.DataFrame) -> dict:
    u, inv = np.unique(S.gid.to_numpy(), return_inverse=True)
    return dict(inv=inv, ng=len(u), sign=S.sign.to_numpy(), run=S.run.to_numpy(), row=S.row.to_numpy(float), col=S.col.to_numpy(float), om=S.om.to_numpy(float))


def _multirun_residual(D: dict, geom0: Geometry, free: Sequence[str], sc: np.ndarray, n_runs: int, ids: bool = False, per_run: Sequence[str] = ()):
    from scipy.spatial.transform import Rotation as Rot
    nfree = len(free)
    n_dev0 = nfree + 3 * (n_runs - 1)
    masks = [D["run"] == k for k in range(n_runs)]

    def f(x):
        base = dict(zip(free, x[:nfree] * sc))
        q = np.empty((len(D["row"]), 3))
        for k in range(n_runs):
            vals = dict(base)
            if k > 0:
                for j, name in enumerate(per_run):
                    vals[name] = base[name] + x[n_dev0 + (k - 1) * len(per_run) + j] * _SCALE[name]
            m = masks[k]
            q[m] = _qs(D["row"][m], D["col"][m], D["om"][m], with_params(geom0, vals))
            if k > 0:
                q[m] = q[m] @ Rot.from_rotvec(x[nfree + 3 * (k - 1): nfree + 3 * k] * 0.01).as_matrix()      # row vectors: (P^T q)^T
        xq = q * D["sign"][:, None]
        pos = D["sign"] > 0
        Sp = np.zeros((D["ng"], 3)); Sm = np.zeros((D["ng"], 3)); Np = np.zeros(D["ng"]); Nm = np.zeros(D["ng"]); mq = np.zeros(D["ng"])
        np.add.at(Sp, D["inv"][pos], xq[pos]); np.add.at(Np, D["inv"][pos], 1.0)
        np.add.at(Sm, D["inv"][~pos], xq[~pos]); np.add.at(Nm, D["inv"][~pos], 1.0)
        np.add.at(mq, D["inv"], np.linalg.norm(q, axis=1))
        cnt = np.bincount(D["inv"], minlength=D["ng"])
        ok = (Np > 0) & (Nm > 0)
        r = (Sp[ok] / Np[ok, None] - Sm[ok] / Nm[ok, None]) * np.sqrt(Np[ok] * Nm[ok] / (Np[ok] + Nm[ok]))[:, None] / (mq[ok] / cnt[ok])[:, None]
        return (r.ravel(), np.flatnonzero(ok)) if ids else r.ravel()
    return f


def fit_multirun(obs_runs: Sequence[pd.DataFrame], geom0: Geometry, *, free: Sequence[str] = FRIEDEL_PARAMS, n_boot: int = 0, seed: int = 0,
                 max_hkl_residual: float = 0.1, omega_col: str = "omega_meas", trim: Optional[float] = 5.0, max_nfev: int = 200,
                 per_run: Sequence[str] = ()) -> MultirunFit:
    """Detector geometry shared by several runs of ONE crystal, from Friedel pairs that may span runs.

    Run k's sample frame differs from run 0's by an unknown rotation P_k (the phi stage): ``q_s,k(h) = P_k q_s,0(h)``, so for h seen in run a and -h in run b,
    ``P_a^T q_s,a(h) + P_b^T q_s,b(-h) = 0``. Unknowns: the geometry ``free`` plus a rotation vector per extra run. The P_k start is a Procrustes alignment of
    the runs' signed vectors at the START geometry (no U anywhere). Per hkl group {h, -h} with both signs somewhere, the residual is the difference of the
    sign-folded means of the two signs, weighted by sqrt(n+ n-/(n+ + n-)) and divided by the mean |q| (shrink-proof, see :func:`pair_residual`);
    groups with a gross residual are dropped (``trim``). Bootstrap resamples hkl groups. ``per_run`` names geometry parameters that may differ between runs
    (a free deviation of run k from run 0 each; ``dev`` / ``sigma_dev`` in the result), the test of whether the runs share one geometry.
    """
    from scipy.spatial.transform import Rotation as Rot
    free = tuple(free); sc = np.array([_SCALE[n] for n in free]); nfree = len(free); n_runs = len(obs_runs)
    S = _signed_groups(obs_runs, max_hkl_residual, omega_col)
    # P_k start: Procrustes of the folded vectors shared with run 0
    q = _qs(S.row.to_numpy(float), S.col.to_numpy(float), S.om.to_numpy(float), geom0) * S.sign.to_numpy()[:, None]
    ref = {g: q[list(ix)].mean(axis=0) for g, ix in S[S.run == 0].groupby("gid").groups.items()}
    rot0 = []
    for k in range(1, n_runs):
        a, b = [], []
        for g, ix in S[S.run == k].groupby("gid").groups.items():
            if g in ref:
                a.append(q[list(ix)].mean(axis=0)); b.append(ref[g])
        if len(a) < 3:
            raise ValueError(f"run {k} shares {len(a)} hkl groups with run 0: cannot align the sample frames")
        R, _ = Rot.align_vectors(np.array(b), np.array(a))
        rot0.append(-R.as_rotvec() / 0.01)                         # q @ P = R a  =>  P = R^T
    per_run = tuple(per_run)
    x0 = np.concatenate([_free_vector(geom0, free) / sc] + rot0 + [np.zeros(len(per_run) * (n_runs - 1))])

    def solve(Sub, xs, knee=3e-2):
        Sub = Sub.copy()
        for it in range(3):
            D = _group_arrays(Sub); ff = _multirun_residual(D, geom0, free, sc, n_runs, False, per_run)

            def jac(x, h=1e-3, ff=ff):
                cols = []
                for i in range(len(x)):
                    dd = np.zeros_like(x); dd[i] = h
                    cols.append((ff(x + dd) - ff(x - dd)) / (2 * h))
                return np.array(cols).T
            sol = least_squares(ff, xs, jac=jac, loss="soft_l1", f_scale=knee, max_nfev=max_nfev)
            r = sol.fun
            kn = 3.0 * 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-9
            xs, knee = sol.x, kn
            if it == 2 or trim is None:
                if trim is None and it == 0:
                    sol = least_squares(ff, xs, jac=jac, loss="soft_l1", f_scale=kn, max_nfev=max_nfev)
                break
            _, ok_idx = _multirun_residual(D, geom0, free, sc, n_runs, True, per_run)(xs)
            bad = np.linalg.norm(r.reshape(-1, 3), axis=1) > trim * np.sqrt(3.0) * (kn / 3.0)
            if bad.any() and (~bad).sum() >= 4 * len(xs):
                drop = np.unique(Sub.gid.to_numpy())[ok_idx[bad]]
                Sub = Sub[~Sub.gid.isin(drop)].reset_index(drop=True)
        return sol, jac, Sub

    sol, jac, Sk = solve(S, x0)
    r = sol.fun
    scale = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-12
    J = jac(sol.x)
    _, sv, vt = np.linalg.svd(J, full_matrices=False)
    cov = (vt.T * (1.0 / np.maximum(sv, 1e-12 * sv.max()) ** 2)) @ vt * scale ** 2
    sigma = np.sqrt(np.diag(cov))
    sb = None
    if n_boot:
        rng = np.random.default_rng(seed); ug = np.unique(Sk.gid.to_numpy()); boot = []
        by = {g: Sk[Sk.gid == g] for g in ug}
        for b in range(n_boot):
            pick = rng.choice(ug, size=len(ug), replace=True)
            parts = []
            for j, g in enumerate(pick):
                t = by[g].copy(); t["gid"] = j; parts.append(t)
            try:
                boot.append(solve(pd.concat(parts, ignore_index=True), sol.x)[0].x)
            except Exception:
                pass
        sb = np.nanstd(np.array(boot), axis=0)
    cross = 0
    for _, x in S.groupby("gid"):
        cross += int(any(a != b for a in x[x.sign > 0].run for b in x[x.sign < 0].run))
    rv = np.array([sol.x[nfree + 3 * k: nfree + 3 * (k + 1)] * 0.01 for k in range(n_runs - 1)])
    n_dev0 = nfree + 3 * (n_runs - 1)
    dev = sdev = None
    if per_run:
        scd = np.array([_SCALE[n] for n in per_run])
        dev = sol.x[n_dev0:].reshape(n_runs - 1, len(per_run)) * scd
        src = (sb if sb is not None else sigma)[n_dev0:].reshape(n_runs - 1, len(per_run)) * scd
        sdev = src
    return MultirunFit(names=free, values=sol.x[:nfree] * sc, sigma_formal=sigma[:nfree] * sc, sigma_boot=None if sb is None else sb[:nfree] * sc, rotvecs=rv,
                       rms_final=float(np.sqrt(np.mean(r ** 2))), scale=scale, n_groups=int(Sk.gid.nunique()), n_groups_trimmed=int(S.gid.nunique() - Sk.gid.nunique()),
                       singular_values=sv, n_cross_run_groups=cross, per_run=per_run, dev=dev, sigma_dev=sdev)
