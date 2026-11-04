"""Symmetry merge of F² observations and the SHELX .hkl export.

Equivalents come from ``midas_hkls.SpaceGroup`` (Laue class, Friedel pairs included: no anomalous
dispersion is modelled, so Friedel mates are merged). The merged sigma is the larger of the internal
(propagated) and external (scatter) estimates, the usual guard against an over-optimistic error from a
handful of agreeing observations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from midas_hkls import SpaceGroup

__all__ = ["merge_observations", "r_int", "write_hkl", "read_hkl", "scale_groups", "scale_smooth", "estimate_error_model_g", "systematic_floor"]


def _robust_group(x: np.ndarray, s: np.ndarray, g: float, reject_sigma: float):
    """One unique reflection, robust estimator: error model sigma'^2 = s^2 + (g c)^2 with c = max(median(x), 0) (a location that does not follow any single observation),
    weights 1/sigma'^2, worst-first leave-one-out rejection while more than two observations remain: z_i = (x_i - m_{-i}) / sqrt(sigma'_i^2 + 1/sum_{j != i} w_j), reject the
    largest |z_i| if it exceeds ``reject_sigma``. With n = 2 nothing is rejected (which of two disagreeing observations is bad cannot be told); the merged sigma then carries the
    external scatter. Returns (keep mask, sigma' of all observations)."""
    c = max(float(np.median(x)), 0.0)
    sp = np.sqrt(s ** 2 + (g * c) ** 2)
    keep = np.ones(len(x), bool)
    while keep.sum() > 2:
        idx = np.flatnonzero(keep)
        w = 1.0 / sp[idx] ** 2
        z = np.empty(len(idx))
        for a, i in enumerate(idx):
            wo = np.delete(w, a)
            m_o = float((np.delete(w, a) * np.delete(x[idx], a)).sum() / wo.sum())
            z[a] = (x[i] - m_o) / np.sqrt(sp[i] ** 2 + 1.0 / wo.sum())
        a = int(np.argmax(np.abs(z)))
        if abs(z[a]) <= reject_sigma:
            break
        keep[idx[a]] = False
    return keep, sp


def estimate_error_model_g(obs: pd.DataFrame, space_group: SpaceGroup, *, value: str = "F2", sigma: str = "sigF2") -> float:
    """Relative error-model term g of sigma'^2 = sigma^2 + (g c)^2, estimated from the data (SADABS/XDS style): over reflections with n >= 2 and a positive median, the
    normalised leave-one-out deviation z_i = (x_i - m_{-i}) / sqrt(s_i^2 + 1/sum_{j != i} 1/s_j^2 + g^2 c^2 (1 + 1/(n-1))) must have a robust sd (1.4826 MAD) of one; g is the root
    (bisection on [0, 1]); 0 when the Poisson-only sigmas already give a robust sd <= 1."""
    rep = np.array([space_group.asu_representative(int(h), int(k), int(l)) for h, k, l in zip(obs["h"], obs["k"], obs["l"])])
    key = pd.factorize(pd.Series([tuple(r) for r in rep]))[0]
    x_all, s_all = obs[value].to_numpy(float), np.maximum(obs[sigma].to_numpy(float), 1e-12)
    dev, var0, c2f = [], [], []
    for u in np.unique(key):
        m = key == u
        n = int(m.sum())
        if n < 2:
            continue
        x, s = x_all[m], s_all[m]
        c = float(np.median(x))
        if c <= 0:
            continue
        w = 1.0 / s ** 2
        for a in range(n):
            wo = np.delete(w, a)
            m_o = float((wo * np.delete(x, a)).sum() / wo.sum())
            dev.append(x[a] - m_o)
            var0.append(s[a] ** 2 + 1.0 / wo.sum())
            c2f.append(c * c * (1.0 + 1.0 / (n - 1)))
    if not dev:
        return 0.0
    dev, var0, c2f = np.array(dev), np.array(var0), np.array(c2f)

    def rsd(g):
        z = dev / np.sqrt(var0 + g * g * c2f)
        return 1.4826 * float(np.median(np.abs(z - np.median(z))))
    if rsd(0.0) <= 1.0:
        return 0.0
    lo, hi = 0.0, 1.0
    if rsd(hi) > 1.0:
        return hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rsd(mid) > 1.0 else (lo, mid)
    return 0.5 * (lo + hi)


def merge_observations(obs: pd.DataFrame, space_group: SpaceGroup, *, value: str = "F2",
                       sigma: str = "sigF2", robust: bool = False, error_g: float = None, reject_sigma: float = 4.0) -> pd.DataFrame:
    """Merge to the unique set. Columns: h k l n F2 sigF2 sig_int sig_ext R_int_refl (one row per unique hkl).

    Default (``robust=False``): the plain 1/sigma^2 weighted mean. ``robust=True`` (opt-in): error model sigma'^2 = sigma^2 + (g c)^2 (c = median of the equivalents, ``error_g`` estimated from
    the data by :func:`estimate_error_model_g` unless given) so that the weights do not follow an observation's own intensity (the Poisson-only 1/sigma^2 mean is biased low and is
    dominated by a failed integration with a small sigma), plus worst-first leave-one-out rejection of observations more than ``reject_sigma`` from their equivalents (never with
    n = 2). Then ``n`` counts the retained observations and the columns ``n_rej`` and ``g`` are added."""
    if obs.empty:
        return pd.DataFrame(columns=["h", "k", "l", "n", value, sigma])
    rep = [space_group.asu_representative(int(h), int(k), int(l))
           for h, k, l in zip(obs["h"], obs["k"], obs["l"])]
    d = obs.copy()
    d[["uh", "uk", "ul"]] = np.array(rep)
    if robust and error_g is None:
        error_g = estimate_error_model_g(obs, space_group, value=value, sigma=sigma)
    out = []
    for (uh, uk, ul), grp in d.groupby(["uh", "uk", "ul"]):
        if robust:
            x0 = grp[value].to_numpy(float)
            s0 = np.maximum(grp[sigma].to_numpy(float), 1e-12)
            keep, sp = _robust_group(x0, s0, error_g, reject_sigma)
            x, sx = x0[keep], sp[keep]
            w = 1.0 / sx ** 2
            mean = float((w * x).sum() / w.sum())
            sig_int = float(1.0 / np.sqrt(w.sum()))
            n = int(keep.sum())
            sig_ext = float(np.sqrt((w * (x - mean) ** 2).sum() / w.sum() / (n - 1))) if n > 1 else sig_int
            out.append(dict(h=uh, k=uk, l=ul, n=n, **{value: mean, sigma: max(sig_int, sig_ext)},
                            sig_int=sig_int, sig_ext=sig_ext,
                            R_int_refl=float(np.abs(x - mean).sum() / np.abs(x).sum()) if n > 1 else np.nan,
                            n_rej=int((~keep).sum()), g=float(error_g)))
            continue
        g = grp
        x = g[value].to_numpy(float)
        sx = np.maximum(g[sigma].to_numpy(float), 1e-12)
        w = 1.0 / sx ** 2
        mean = float((w * x).sum() / w.sum())
        sig_int = float(1.0 / np.sqrt(w.sum()))
        n = len(g)
        sig_ext = float(np.sqrt((w * (x - mean) ** 2).sum() / w.sum() / (n - 1))) if n > 1 else sig_int
        out.append(dict(h=uh, k=uk, l=ul, n=n, **{value: mean, sigma: max(sig_int, sig_ext)},
                        sig_int=sig_int, sig_ext=sig_ext,
                        R_int_refl=float(np.abs(x - mean).sum() / np.abs(x).sum()) if n > 1 else np.nan))
    return pd.DataFrame(out)


def scale_groups(obs: pd.DataFrame, space_group: SpaceGroup, *, group: str = "run", n_iter: int = 50, tol: float = 1e-8, value: str = "F2",
                 sigma: str = "sigF2") -> tuple:
    """Relative scale of each observation group (a run, a frame batch): the multiplicative k_g minimising sum w (k_g F2 - <F2>_hkl)^2 over symmetry-equivalent observations,
    iterated with the merged mean; k is normalised to a weighted mean of 1. Returns ``(obs_scaled, table)``; F2 and sigF2 of ``obs_scaled`` are multiplied by k_g. Only
    reflections seen in at least two observations constrain the scales; a group that shares no reflection with the others keeps k = 1 and is reported with ``n_shared = 0``."""
    d = obs.copy()
    rep = np.array([space_group.asu_representative(int(h), int(k), int(l)) for h, k, l in zip(d["h"], d["k"], d["l"])])
    d["_key"] = [tuple(r) for r in rep]
    groups = sorted(d[group].unique())
    k = {g: 1.0 for g in groups}
    x = d[value].to_numpy(float); sx = np.maximum(d[sigma].to_numpy(float), 1e-12); gi = d[group].to_numpy()
    keys = pd.factorize(d["_key"])[0]
    n_per = np.bincount(keys)
    shared = n_per[keys] >= 2
    for _ in range(n_iter):
        kv = np.array([k[g] for g in gi])
        w = 1.0 / (kv * sx) ** 2
        xs = kv * x
        mean = np.bincount(keys, w * xs) / np.maximum(np.bincount(keys, w), 1e-300)
        new = {}
        for g in groups:
            m = (gi == g) & shared
            if m.sum() == 0:
                new[g] = 1.0
                continue
            ww = 1.0 / sx[m] ** 2
            new[g] = float((ww * x[m] * mean[keys[m]]).sum() / (ww * x[m] ** 2).sum())
        norm = np.exp(np.mean([np.log(new[g]) for g in groups if ((gi == g) & shared).any()] or [0.0]))
        new = {g: (v / norm if ((gi == g) & shared).any() else 1.0) for g, v in new.items()}
        if max(abs(new[g] - k[g]) for g in groups) < tol:
            k = new
            break
        k = new
    kv = np.array([k[g] for g in gi])
    out = obs.copy()
    out[value] = x * kv
    out[sigma] = sx * kv
    table = pd.DataFrame(dict(group=groups, scale=[k[g] for g in groups], n_obs=[int((gi == g).sum()) for g in groups], n_shared=[int(((gi == g) & shared).sum()) for g in groups]))
    return out, table


def scale_smooth(obs: pd.DataFrame, space_group: SpaceGroup, *, group: str = "run", knot: str = "frame", n_knots: int = 8, min_obs_per_knot: int = 20,
                 rel_floor: float = 0.1, n_iter: int = 40, tol: float = 1e-7, fit_mask=None, value: str = "F2", sigma: str = "sigF2") -> tuple:
    """Smooth multiplicative scale k_g(t) of each group (a run) as a piecewise-linear function of ``knot`` (default the frame index), fitted from symmetry-equivalent observations.

    Model: log F2_i = log m_u(i) - log k_g(i)(t_i) + noise, with m_u the true value of the unique reflection u. Minimises sum w_i (log F2_i + log k_i - log m_u)^2 over observations of reflections seen at
    least twice (F2 > 0), by alternating the per-reflection mean of log (F2 k) with a linear least-squares for the knot values; w_i = 1 / (sigma_i/F2_i)^2 + rel_floor^2) so that a systematic floor (``rel_floor``,
    relative) keeps strong reflections from dominating. Gauge: the observation-mean of log k is 0. ``n_knots`` knots span the range of ``knot`` of each group; a group with fewer than
    ``min_obs_per_knot`` shared observations per knot gets fewer knots (down to one: a plain group scale). ``fit_mask`` (bool per row of ``obs``) restricts which observations CONSTRAIN the fit
    (cross-validation); the scale is applied to every row. Returns ``(obs_scaled, table)``; ``obs_scaled`` has F2 and sigF2 multiplied by k and a column ``k_smooth``; ``table`` has per group
    n_obs, n_shared, n_knots and the max-min of log k over the group's observations.

    The scale absorbs anything smooth in the knot variable that multiplies a whole group of observations: beam and sample-volume drift, crystal centring, absorption by a symmetric environment.
    It cannot be told apart from a real intensity variation with the same dependence, so it needs redundancy (every group's range must be covered by reflections also seen elsewhere)."""
    d = obs.copy()
    rep = np.array([space_group.asu_representative(int(h), int(k), int(l)) for h, k, l in zip(d["h"], d["k"], d["l"])])
    keys = pd.factorize(pd.Series([tuple(r) for r in rep], index=d.index))[0]
    x = d[value].to_numpy(float); sx = np.maximum(d[sigma].to_numpy(float), 1e-12); t = d[knot].to_numpy(float); gi = d[group].to_numpy()
    groups = sorted(pd.unique(gi))
    use = np.ones(len(d), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    ok = use & (x > 0)
    n_per = np.bincount(keys[ok], minlength=keys.max() + 1)
    shared = ok & (n_per[keys] >= 2)
    if not shared.any():                                     # nothing constrains a scale: identity, reported as one knot per group
        out = obs.copy(); out['k_smooth'] = 1.0
        return out, pd.DataFrame(dict(group=groups, n_obs=[int((gi == g).sum()) for g in groups], n_shared=0, n_knots=1, log_k_range=0.0))
    # basis: per group a piecewise-linear hat basis over its own knot range
    cols, offs, rows_b, vals_b, nk_g = [], 0, [], [], {}
    col_of = np.zeros((len(d), 2), int); wt_of = np.zeros((len(d), 2))
    for g in groups:
        m = gi == g
        nsh = int((m & shared).sum())
        nk = int(max(1, min(n_knots, nsh // max(min_obs_per_knot, 1))))
        nk_g[g] = nk
        if nk == 1:
            col_of[m, 0] = offs; col_of[m, 1] = offs; wt_of[m, 0] = 1.0; wt_of[m, 1] = 0.0
        else:
            lo, hi = float(np.min(t[m])), float(np.max(t[m]))
            hi = hi if hi > lo else lo + 1.0
            u = np.clip((t[m] - lo) / (hi - lo), 0.0, 1.0) * (nk - 1)
            j = np.minimum(np.floor(u).astype(int), nk - 2); f = u - j
            col_of[m, 0] = offs + j; col_of[m, 1] = offs + j + 1; wt_of[m, 0] = 1.0 - f; wt_of[m, 1] = f
        offs += nk
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    n = len(d)
    Bfull = sparse.csr_matrix((np.r_[wt_of[:, 0], wt_of[:, 1]], (np.r_[np.arange(n), np.arange(n)], np.r_[col_of[:, 0], col_of[:, 1]])), shape=(n, offs))
    lf = np.full(n, np.nan); lf[x > 0] = np.log(x[x > 0])
    w = np.zeros(n); w[shared] = 1.0 / ((sx[shared] / x[shared]) ** 2 + rel_floor ** 2)
    sw = np.sqrt(w)
    B = Bfull[shared]
    coef = np.zeros(offs)
    for _ in range(n_iter):
        logk = Bfull @ coef                                  # log k: the scaled value is F2 * k, so log F2 + log k is the corrected log
        r = np.where(shared, lf + logk, 0.0)
        mu = np.bincount(keys[shared], (w * r)[shared], minlength=keys.max() + 1) / np.maximum(np.bincount(keys[shared], w[shared], minlength=keys.max() + 1), 1e-300)
        target = mu[keys[shared]] - lf[shared]               # log k that would make this observation equal the reflection mean
        A = sparse.diags(sw[shared]) @ B
        new = lsqr(A, sw[shared] * target, atol=1e-12, btol=1e-12, iter_lim=2000)[0]
        new = new - float(np.average(Bfull[shared] @ new, weights=w[shared]))          # gauge: weighted mean of log k is zero
        done = np.max(np.abs(new - coef)) < tol
        coef = new
        if done:
            break
    logk = Bfull @ coef
    kv = np.exp(logk)
    out = obs.copy()
    out[value] = x * kv
    out[sigma] = sx * kv
    out["k_smooth"] = kv
    table = pd.DataFrame(dict(group=groups, n_obs=[int((gi == g).sum()) for g in groups], n_shared=[int(((gi == g) & shared).sum()) for g in groups],
                              n_knots=[nk_g[g] for g in groups], log_k_range=[float(np.ptp(logk[gi == g])) for g in groups]))
    return out, table


def systematic_floor(obs: pd.DataFrame, space_group: SpaceGroup, *, by=None, value: str = "F2", sigma: str = "sigF2", min_obs: int = 30) -> pd.DataFrame:
    """Disagreement of symmetry-equivalent observations beyond their counting statistics: the systematic floor ``f`` (relative, in F2).

    For every reflection seen at least twice (F2 > 0) the log deviation of each observation from the mean of the OTHERS (leave-one-out, so a pair gives an unbiased deviation) is
    ``dev``; its robust sd (1.4826 MAD) squared estimates ``2 f^2 + (sigma/F2)^2 (1 + 1/(n-1))``, so ``f = sqrt(max(0, robust_sd^2 - median counting^2) / 2)``. ``counting`` is the
    median of the counting term ``sqrt((sigma/F2)^2 (1 + 1/(n-1)))`` alone (what the Poisson sigmas alone predict). Reflections whose leave-one-out mean is not positive are dropped.

    ``by``: None (one row, group ``"all"``), the name of a column of ``obs``, or an array/Series (aligned with ``obs.index``) of group labels, e.g. ``pd.cut(obs.L, bins)``. The leave-one-out
    deviation is always formed over ALL equivalents; ``by`` only groups the observations afterwards. Groups with fewer than ``min_obs`` usable observations get NaN.
    Returns a DataFrame indexed by group with columns ``n`` (usable observations), ``robust_sd`` (of the log deviation), ``counting`` and ``floor``.

    It sees only what differs between equivalents: an error that is the same in every equivalent (a wrong cell in the Lorentz factor, a common absorption) is invisible to it."""
    cols = ["n", "robust_sd", "counting", "floor"]
    if obs.empty:
        return pd.DataFrame(columns=cols)
    d = obs.reset_index(drop=True).copy()
    cache: dict = {}
    keys = []
    for h, k, l in zip(d["h"].to_numpy(int), d["k"].to_numpy(int), d["l"].to_numpy(int)):
        t = (int(h), int(k), int(l))
        if t not in cache:
            cache[t] = tuple(int(v) for v in space_group.asu_representative(*t))
        keys.append(cache[t])
    d["_u"] = pd.factorize(pd.Series(keys))[0]
    if by is None:
        d["_g"] = "all"
    elif isinstance(by, str):
        d["_g"] = obs[by].to_numpy()
    else:
        arr = by.to_numpy() if hasattr(by, "to_numpy") else np.asarray(by)
        if len(arr) != len(obs):
            raise ValueError("systematic_floor: `by` must have one label per row of obs")
        d["_g"] = arr if not hasattr(by, "cat") else pd.Series(arr, dtype=by.dtype)
    x = d[value].to_numpy(float); sx = d[sigma].to_numpy(float)
    ok = x > 0
    g = d[ok].groupby("_u")[value]
    d.loc[ok, "_n"] = g.transform("size").to_numpy(); d.loc[ok, "_s"] = g.transform("sum").to_numpy()
    d = d[ok & (d["_n"] >= 2)].copy()
    d["loo"] = (d["_s"] - d[value]) / (d["_n"] - 1)
    d = d[d["loo"] > 0].copy()
    d["dev"] = np.log(d[value] / d["loo"])
    d["ps"] = (d[sigma] / d[value]) ** 2 * (1.0 + 1.0 / (d["_n"] - 1))
    rows = {}
    for lab, sub in d.groupby("_g", observed=True):
        if len(sub) < min_obs:
            rows[lab] = (len(sub), np.nan, np.nan, np.nan)
            continue
        v = (1.4826 * float(np.median(np.abs(sub["dev"] - np.median(sub["dev"]))))) ** 2
        c = float(np.median(sub["ps"]))
        rows[lab] = (len(sub), float(np.sqrt(v)), float(np.sqrt(c)), float(np.sqrt(max(0.0, v - c) / 2.0)))
    out = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    out.index.name = "group"
    return out


def r_int(obs: pd.DataFrame, space_group: SpaceGroup, *, value: str = "F2", merge_kwargs: dict = None) -> float:
    """Σ|x - <x>| / Σ x over reflections observed at least twice; NaN when none are. ``merge_kwargs`` go to :func:`merge_observations` (the mean it measures against)."""
    m = merge_observations(obs, space_group, value=value, **(merge_kwargs or {}))
    if m.empty or (m["n"] < 2).all():
        return float("nan")
    rep = np.array([space_group.asu_representative(int(h), int(k), int(l))
                    for h, k, l in zip(obs["h"], obs["k"], obs["l"])])
    key = pd.Series(list(map(tuple, rep)), index=obs.index)
    mean = {(r.h, r.k, r.l): getattr(r, value) for r in m.itertuples()}
    cnt = {(r.h, r.k, r.l): r.n for r in m.itertuples()}
    num = den = 0.0
    for idx, kk in key.items():
        if cnt[kk] < 2:
            continue
        num += abs(obs.at[idx, value] - mean[kk])
        den += abs(obs.at[idx, value])
    return num / den if den > 0 else float("nan")


def write_hkl(path: Union[str, Path], merged: pd.DataFrame, *, value: str = "F2", sigma: str = "sigF2",
              max_value: float = 9999.0) -> float:
    """Write a SHELX HKLF 4 file (``3I4, 2F8.2, I4``) with the terminating 0 0 0 record.

    Values are scaled so the largest is ``max_value`` (the F8.2 field overflows near 1e5); the factor is
    returned and must be kept: refinement against this file is then on the scaled F², which a free
    overall scale absorbs. Negative F² are written as they are (SHELXL wants them).
    """
    m = merged
    if (m[["h", "k", "l"]].abs() >= 1000).any().any():
        raise ValueError("|h|,|k|,|l| must be < 1000 for the I4 fields")
    scale = max_value / float(m[value].abs().max())
    with open(path, "w") as fh:
        for r in m.itertuples():
            fh.write("%4d%4d%4d%8.2f%8.2f%4d\n" % (r.h, r.k, r.l, getattr(r, value) * scale,
                                                     getattr(r, sigma) * scale, 1))
        fh.write("%4d%4d%4d%8.2f%8.2f%4d\n" % (0, 0, 0, 0.0, 0.0, 0))
    return scale


def read_hkl(path: Union[str, Path]) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        h, k, l = int(line[0:4]), int(line[4:8]), int(line[8:12])
        if (h, k, l) == (0, 0, 0):
            break
        rows.append(dict(h=h, k=k, l=l, F2=float(line[12:20]), sigF2=float(line[20:28]), batch=int(line[28:32])))
    return pd.DataFrame(rows)
