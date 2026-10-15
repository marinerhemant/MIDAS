"""v3 obpitch response for :mod:`midas_dfxm.forward_maps`: a global flat-top profile, per-pixel centre.

:mod:`midas_dfxm.forward_maps` gives every pixel a Gaussian in obpitch whose centre is the centre of mass of 1.85
points per FWHM and whose width is a half-max. On the Mg-4Al ID03 scan that fails in a specific way: the observed
obpitch response is a flat top with steep edges (it falls ~100x between one and two motor steps off the peak), which
is what a response dominated by the objective's angular acceptance looks like (``manuals/dfxm/ENVELOPE.md`` section 6).
A Gaussian therefore over-predicts the planes two steps off the peak and under-predicts the peak itself.

This module keeps the mu and chi model of ``forward_maps`` and replaces only the obpitch factor:

    y = b + A * P((ob_k - c) / step; w, s) * g(mu_i; mu0, s_mu) * g(chi_j; chi0, s_chi)

with ``P`` an erf-edged box of width ``w`` and edge softness ``s`` (both in motor steps, global), ``c`` and ``A`` per
pixel. ``c`` is NOT a centre of mass: it is the best of a grid of candidate centres by least squares against the
training planes' per-pixel totals, which removes the bias and quantisation of the moment estimate. ``(w, s)`` are chosen
by nested leave-one-training-plane-out error, so the held-out plane never influences them and the in-sample centre
fit cannot make the profile look sharper than it is out of sample.

The same limits apply as in ``forward_maps``: a fidelity check of the maps as a summary of the stack, not a measurement
of physical accuracy, strain, depth or absolute orientation. The profile is learned from this scan; it is not an
independent measurement of the objective's acceptance. Registration: ``examples/id03_forward_loo/PREREGISTER_v3.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.special import erf

from . import forward_maps as fm

__all__ = ["box_profile", "ProfileFit", "fit_centres", "choose_profile", "amplitudes_for_centres",
           "predict_plane_profile", "score_plane_profile", "leave_one_plane_out_profile",
           "synthetic_cube_profile", "W_GRID", "S_GRID", "CAND_LO", "CAND_HI", "CAND_STEP"]

W_GRID = (1.2, 1.6, 2.0, 2.5, 3.0)       # box widths, motor steps
S_GRID = (0.2, 0.5, 0.9)                 # edge softness, motor steps
CAND_LO, CAND_HI, CAND_STEP = -1.0, 8.0, 0.1   # candidate centres, steps from plane 0


def box_profile(d, w: float, s: float):
    """Unit-height flat top of width ``w`` with erf edges of softness ``s`` (both in steps)."""
    r = np.sqrt(2.0) * s
    return 0.5 * (erf((d + 0.5 * w) / r) - erf((d - 0.5 * w) / r))


@dataclass
class ProfileFit:
    """Per-pixel centre ``c`` (steps from plane 0) and amplitude, with the global profile that made them."""

    c: np.ndarray
    A: np.ndarray
    w: float
    s: float
    at_edge: float            # fraction of pixels whose centre sits on the candidate-grid limit


def _cand():
    return np.arange(CAND_LO, CAND_HI + 1e-9, CAND_STEP)


def fit_centres(sig: np.ndarray, S: np.ndarray, pos: np.ndarray, w: float, s: float, *,
                chunk: int = 20_000):
    """Per-pixel best centre and amplitude of ``sig_j = A S P(pos_j - c)`` over candidate centres.

    ``sig`` ``(n, P)`` the training planes' totals above baseline, ``S`` ``(P,)`` the mu*chi sum of the Gaussian
    factors, ``pos`` ``(n,)`` the planes' positions in steps. Chunked over pixels. Returns ``(c, A)``.
    """
    cand = _cand()
    n, P = sig.shape
    g = box_profile(pos[:, None, None] - cand[None, None, :], w, s)         # (n, 1, C)
    gg = (g ** 2).sum(0)[0]                                                  # (C,)
    c_out = np.empty(P); A_out = np.empty(P)
    for a in range(0, P, chunk):
        b = min(a + chunk, P)
        sg = sig[:, a:b]; Sc = S[a:b]
        num = (sg[:, :, None] * g).sum(0) * Sc[:, None]                      # (p, C)
        den = gg[None, :] * (Sc[:, None] ** 2)
        A = np.maximum(num / np.maximum(den, 1e-12), 0.0)
        sse = ((sg[:, :, None] - A[None] * g * Sc[None, :, None]) ** 2).sum(0)
        ic = np.argmin(sse, 1)
        c_out[a:b] = cand[ic]; A_out[a:b] = A[np.arange(b - a), ic]
    return c_out, A_out


def amplitudes_for_centres(sig: np.ndarray, S: np.ndarray, pos: np.ndarray, c: np.ndarray, w: float, s: float):
    """Clipped closed-form amplitude when the centres are given (used by the permutation null)."""
    g = box_profile(pos[:, None] - c[None], w, s)
    return np.maximum((sig * g).sum(0) * S / np.maximum(((g * S) ** 2).sum(0), 1e-12), 0.0)


def choose_profile(sig: np.ndarray, S: np.ndarray, pos: np.ndarray, *, w_grid=W_GRID, s_grid=S_GRID):
    """Global ``(w, s)`` minimising the nested leave-one-training-plane-out squared error.

    Each training plane ``j`` is predicted from centres and amplitudes fitted on the other training planes, so the
    centre fit is out-of-sample exactly as it is for the held-out plane. Returns ``(w, s, table)``.
    """
    n = sig.shape[0]
    table = {}
    for w in w_grid:
        for s in s_grid:
            sse = 0.0
            for j in range(n):
                rest = [i for i in range(n) if i != j]
                c, A = fit_centres(sig[rest], S, pos[rest], w, s)
                sse += float(((sig[j] - A * box_profile(pos[j] - c, w, s) * S) ** 2).sum())
            table[(w, s)] = sse
    (w, s) = min(table, key=table.get)
    return w, s, table


def predict_plane_profile(p: fm.SeparableParams, fit: ProfileFit, pos_k: float, b: np.ndarray) -> np.ndarray:
    """Predicted counts of the held-out plane at position ``pos_k``: ``(n_mu, n_chi, P)``."""
    g = box_profile(pos_k - fit.c, fit.w, fit.s)
    G_mu = fm._gauss(p.mu_g[:, None], p.mu0[None], p.smu[None])
    G_chi = fm._gauss(p.chi_g[:, None], p.chi0[None], p.schi[None])
    return b[None, None] + (fit.A * g)[None, None] * G_mu[:, None, :] * G_chi[None, :, :]


def score_plane_profile(obs: np.ndarray, p: fm.SeparableParams, fit: ProfileFit, pos_k: float, b, *,
                        chunk: int = 40_000):
    """Per-pixel SSE and SST against ``obs`` ``(n_mu, n_chi, P)``, streamed over pixel chunks."""
    P = obs.shape[-1]
    sse = np.empty(P); sst = np.empty(P)
    for a in range(0, P, chunk):
        e = min(a + chunk, P)
        q = p.take(slice(a, e))
        f = replace(fit, c=fit.c[a:e], A=fit.A[a:e])
        pr = predict_plane_profile(q, f, pos_k, b[a:e])
        y = obs[:, :, a:e].astype(np.float64)
        sse[a:e] = ((y - pr) ** 2).sum((0, 1))
        sst[a:e] = ((y - y.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
    return sse, sst


def _positions(ob_g, idx):
    ob_g = np.asarray(ob_g, float)
    step = float(np.median(np.abs(np.diff(np.sort(ob_g)))))
    return (ob_g[list(idx)] - ob_g[0]) / step if not np.isscalar(idx) else (ob_g[idx] - ob_g[0]) / step


def leave_one_plane_out_profile(cube, mu_g, chi_g, ob_g, folds, masks, *, nested_sub: int = 40_000,
                                seed: int = 0, keep: bool = False, log=lambda *a: None):
    """v3 leave-one-plane-out: fit on every plane but ``k``, score plane ``k``.

    ``masks[k]`` is the (P,) bool of pixels scored in fold k. ``(w, s)`` come from the nested search on a random
    ``nested_sub`` subsample of that mask. Returns ``{k: dict(sse, sst, r2, w, s, at_edge)}`` and, with ``keep``,
    the fold's ``params``, ``fit`` and training totals under ``_params`` / ``_fit`` / ``_sig`` / ``_S``.
    """
    rng = np.random.default_rng(seed)
    n_ob = cube.shape[2]; n_mu, n_chi = cube.shape[0], cube.shape[1]
    res = {}
    for k in folds:
        train = [j for j in range(n_ob) if j != k]
        p = fm.fit_separable_params(cube, mu_g, chi_g, ob_g, train)
        S = fm._gauss(p.mu_g[:, None], p.mu0[None], p.smu[None]).sum(0) * \
            fm._gauss(p.chi_g[:, None], p.chi0[None], p.schi[None]).sum(0)
        sig = p.mob_train - p.base_ob[None]
        pos_tr = _positions(ob_g, train); pos_k = float(_positions(ob_g, k))
        idx = np.flatnonzero(masks[k])
        sub = np.sort(rng.choice(idx, size=min(nested_sub, len(idx)), replace=False))
        w, s, _ = choose_profile(sig[:, sub], S[sub], pos_tr)
        c, A = fit_centres(sig, S, pos_tr, w, s)
        edge = float(np.mean((c <= CAND_LO + 1e-9) | (c >= CAND_HI - 1e-9)))
        fit = ProfileFit(c=c, A=A, w=w, s=s, at_edge=edge)
        b = p.base_ob / (n_mu * n_chi)
        sse, sst = score_plane_profile(cube[:, :, k, :], p, fit, pos_k, b)
        res[k] = dict(sse=sse, sst=sst, r2=fm.pooled_r2(sse, sst, masks[k]), w=w, s=s, at_edge=edge)
        log(f"fold {k}: w {w} s {s} R2 {res[k]['r2']:.3f}")
        if keep:
            res[k].update(_params=p, _fit=fit, _sig=sig, _S=S, _pos_tr=pos_tr, _pos_k=pos_k, _b=b)
    return res


def synthetic_cube_profile(p: fm.SeparableParams, fit: ProfileFit, ob_g, b, *, noise_var0: float,
                           gain: float, rng, gaussian_fwhm: Optional[float] = None) -> np.ndarray:
    """Cube from the v3 model with Gaussian noise ``var = noise_var0 + gain * signal``.

    ``p`` carries the mu/chi parameters, ``fit`` the per-pixel ``c, A`` and the profile. With ``gaussian_fwhm``
    (in steps) the obpitch response is a Gaussian of that FWHM about the same ``c``: the misspecified control.
    """
    n_mu, n_chi, P = len(p.mu_g), len(p.chi_g), p.n_pixels
    pos = _positions(ob_g, range(len(ob_g)))
    G_mu = fm._gauss(p.mu_g[:, None], p.mu0[None], p.smu[None])
    G_chi = fm._gauss(p.chi_g[:, None], p.chi0[None], p.schi[None])
    out = np.empty((n_mu, n_chi, len(ob_g), P), np.float32)
    for k in range(len(ob_g)):
        d = pos[k] - fit.c
        g = box_profile(d, fit.w, fit.s) if gaussian_fwhm is None else np.exp(-np.log(2) * (2 * d / gaussian_fwhm) ** 2)
        sig = (fit.A * g)[None, None] * G_mu[:, None, :] * G_chi[None, :, :]
        mean = b[None, None] + sig
        out[:, :, k, :] = mean + np.sqrt(noise_var0 + gain * np.maximum(sig, 0.0)) * rng.standard_normal(mean.shape)
    return out
