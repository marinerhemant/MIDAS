"""Forward-predict the raw frames of an ID03 mu x chi x obpitch scan from its inverted maps.

The inversion (:mod:`midas_dfxm.multiplane`) reduces 8 obpitch planes of a (mu, chi) mesh to
per-pixel centres and widths in mu, chi and obpitch. This module goes the other way: it
rebuilds the **frames** from those maps and scores the rebuild against the raw counts of a plane
the maps were NOT fitted on (leave-one-plane-out). It is the empirical forward half of the
ID03 hand-off notebook; the registrations it was run against are
the ``PREREGISTER*.md`` files in ``examples/id03_forward_loo/``.

Model (separable, per detector pixel p, plane k, frame (mu_i, chi_j))::

    y = b_p + A_p * g(ob_k; ob0_p, s_ob_p) * g(mu_i; mu0_p, s_mu_p) * g(chi_j; chi0_p, s_chi_p)

``g`` is a unit-peak Gaussian. Centres are the baseline-removed centres of mass of the marginals of
the TRAINING planes (:func:`~midas_dfxm.multiplane.angular_moments`); sigmas are the half-max
widths of the same marginals (``widths="halfmax"``, the default; the baseline-removed second
moment, ``widths="moment"``, is kept as ``smu_mom`` etc. because it overstates a peak that spans
~1 grid step, see :func:`halfmax_sigma`). ``b`` and ``A``
are then a closed-form least-squares fit to the training planes' obpitch totals. No mu-chi
covariance, no per-pixel free fit: deliberately the simplest model that uses only what the maps
carry.

What this is, and is not
------------------------
A **sufficiency / fidelity** check of the maps as a summary of the stack. It is not a validation of
physical accuracy (README rule 5): the maps were made from the same sample, optic and background.
Leaving a plane out removes the direct leak only. Nothing here says anything about strain (one
reflection gives delta d / d, and >90 % of the obpitch width is objective acceptance), depth
(unresolved), or absolute orientation.

Data layout
-----------
Everything works on a ``cube`` of shape ``(n_mu, n_chi, n_ob, P)``: per-frame-pedestal-subtracted,
**unclipped** counts for ``P`` selected pixels. Pixels never interact, so a pixel subsample is a
faithful stand-in for the whole grain.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from .multiplane import angular_moments, derive_grain_mask

__all__ = [
    "SIGMA_FLOOR_STEPS", "halfmax_sigma", "SeparableParams", "fit_separable_params", "amplitude_and_baseline",
    "predict_plane", "score_plane", "pooled_r2", "uniform_params", "moment_width_params",
    "ablate_params",
    "complete_blocks", "block_donors", "pixel_donors", "swap_bundle", "permutation_null",
    "leave_one_plane_out", "synthetic_cube", "load_planes_cube", "read_noise_floor",
    "gain_from_residuals",
]

# A Gaussian narrower than half a motor step cannot be told from a delta function on this grid.
# Moments of a peak that fits inside one bin give sigma = 0, which would predict nothing but the
# one bin. The floor is reported (fraction of pixels floored), never hidden.
SIGMA_FLOOR_STEPS = 0.5


def _gauss(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def halfmax_sigma(Mc: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian sigma from the half-maximum width of each column of ``Mc`` ``(n, P)``.

    ``Mc`` is baseline-removed and non-negative. Per column: take the peak bin, walk out each side
    while the profile stays >= half the peak height (CONTIGUOUS crossings, README rule 12: a global
    outermost crossing lets one noise spike set the width), and interpolate each crossing linearly.
    FWHM = left + right half-width; sigma = FWHM / 2.3548. A side that runs off the grid
    (truncated) takes the other side's half-width; if both do, the whole span is used.
    """
    g = np.asarray(grid, float)
    n, P = Mc.shape
    ipk = np.argmax(Mc, 0)
    ypk = Mc[ipk, np.arange(P)]
    half = 0.5 * ypk
    cols = np.arange(P)
    xpk = g[ipk]
    hw = {}
    for side, sgn in (("left", -1), ("right", 1)):
        alive = np.ones(P, bool)                    # still >= half, walking outward
        last = ipk.copy()                           # last bin still >= half
        found = np.zeros(P, bool)
        x_cross = np.full(P, np.nan)
        for d in range(1, n):
            idx = ipk + sgn * d
            inside = (idx >= 0) & (idx < n)
            j = np.clip(idx, 0, n - 1)
            y = Mc[j, cols]
            dead_now = alive & inside & (y < half)
            if dead_now.any():
                yl = Mc[last, cols]
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = (yl - half) / (yl - y)      # fraction of the way from last to j
                    xc = g[last] + f * (g[j] - g[last])
                x_cross = np.where(dead_now, xc, x_cross)
                found |= dead_now
            alive &= inside & (y >= half)
            last = np.where(alive, j, last)
        hw[side] = np.where(found, np.abs(x_cross - xpk), np.nan)
    left, right = hw["left"], hw["right"]
    left = np.where(np.isnan(left), right, left)
    right = np.where(np.isnan(right), left, right)
    span = g.max() - g.min()
    fwhm = np.where(np.isnan(left), span, left + right)
    return fwhm * _FWHM_TO_SIGMA


@dataclass
class SeparableParams:
    """Per-pixel model parameters from the training planes (all ``(P,)`` unless noted)."""

    mu0: np.ndarray
    chi0: np.ndarray
    ob0: np.ndarray
    smu: np.ndarray
    schi: np.ndarray
    sob: np.ndarray
    mob_train: np.ndarray        # (n_train, P) per-plane totals of the training planes
    base_ob: np.ndarray          # (P,) per-plane-total angular baseline
    ob_train: np.ndarray         # (n_train,) obpitch of the training planes
    mu_g: np.ndarray             # (n_mu,)
    chi_g: np.ndarray            # (n_chi,)
    floored: dict = field(default_factory=dict)   # axis -> fraction of pixels at the floor
    smu_mom: Optional[np.ndarray] = None   # the moment-width alternatives, same floor
    schi_mom: Optional[np.ndarray] = None
    sob_mom: Optional[np.ndarray] = None

    def take(self, idx) -> "SeparableParams":
        """The same model restricted to pixels ``idx`` (index array or bool mask)."""
        sel = lambda a: a[..., idx] if a.ndim == 2 else a[idx]
        return replace(
            self, mu0=self.mu0[idx], chi0=self.chi0[idx], ob0=self.ob0[idx], smu=self.smu[idx],
            schi=self.schi[idx], sob=self.sob[idx], mob_train=sel(self.mob_train),
            base_ob=self.base_ob[idx],
            smu_mom=None if self.smu_mom is None else self.smu_mom[idx],
            schi_mom=None if self.schi_mom is None else self.schi_mom[idx],
            sob_mom=None if self.sob_mom is None else self.sob_mom[idx])

    @property
    def n_pixels(self) -> int:
        return int(self.mu0.size)


def fit_separable_params(cube: np.ndarray, mu_g, chi_g, ob_g, train, *,
                         base_frac_ang: float = 0.30, base_frac_ob: float = 0.25,
                         widths: str = "halfmax") -> SeparableParams:
    """Centres and widths from the marginals of the planes in ``train`` only.

    ``cube`` is ``(n_mu, n_chi, n_ob, P)``; ``train`` indexes its third axis. The foregone clip
    of :func:`~midas_dfxm.multiplane.accumulate_marginals` is deliberate: this is scored against
    unclipped counts. ``widths`` is ``"halfmax"`` (default) or ``"moment"``; both are stored.
    """
    if widths not in ("halfmax", "moment"):
        raise ValueError("widths must be 'halfmax' or 'moment'")
    mu_g = np.asarray(mu_g, float); chi_g = np.asarray(chi_g, float); ob_g = np.asarray(ob_g, float)
    train = list(train)
    n_mu, n_chi, _, P = cube.shape
    Mmu = np.zeros((n_mu, P)); Mchi = np.zeros((n_chi, P)); Mob = np.zeros((len(train), P))
    for t, k in enumerate(train):
        plane = cube[:, :, k, :]
        Mmu += plane.sum(1, dtype=np.float64)
        Mchi += plane.sum(0, dtype=np.float64)
        Mob[t] = plane.sum((0, 1), dtype=np.float64)
    mu_m = angular_moments(Mmu[:, :, None], mu_g, base_frac=base_frac_ang)
    chi_m = angular_moments(Mchi[:, :, None], chi_g, base_frac=base_frac_ang)
    ob_m = angular_moments(Mob[:, :, None], ob_g[train], base_frac=base_frac_ob)
    out, floored = {}, {}
    for name, m, M, g in (("mu", mu_m, Mmu, mu_g), ("chi", chi_m, Mchi, chi_g),
                          ("ob", ob_m, Mob, ob_g[train])):
        step = float(np.median(np.abs(np.diff(np.sort(g)))))
        floor = SIGMA_FLOOR_STEPS * step
        s_mom = np.maximum(m["sig"][:, 0], floor)
        Mc = np.maximum(M - m["base"][:, 0][None], 0.0)
        s_hm = np.maximum(halfmax_sigma(Mc, g), floor)
        chosen = s_hm if widths == "halfmax" else s_mom
        floored[name] = float(np.mean(chosen <= floor * (1 + 1e-9)))
        out[name] = (m["com"][:, 0], chosen, s_mom)
    return SeparableParams(
        mu0=out["mu"][0], chi0=out["chi"][0], ob0=out["ob"][0],
        smu=out["mu"][1], schi=out["chi"][1], sob=out["ob"][1],
        mob_train=Mob, base_ob=ob_m["base"][:, 0], ob_train=ob_g[train],
        mu_g=mu_g, chi_g=chi_g, floored=floored,
        smu_mom=out["mu"][2], schi_mom=out["chi"][2], sob_mom=out["ob"][2])


def amplitude_and_baseline(p: SeparableParams):
    """Closed-form ``A`` and per-frame ``b`` for the (possibly altered) parameters ``p``.

    The training planes' totals obey ``T_k - base = A g_ob(k) S_mu S_chi`` with ``S`` the sum of
    the Gaussian over the motor grid, so ``A`` is a one-parameter least squares per pixel, clipped
    at zero. ``b`` is the angular baseline spread over the frames of one plane.
    """
    g_ob = _gauss(p.ob_train[:, None], p.ob0[None], p.sob[None])               # (n_tr, P)
    S_mu = _gauss(p.mu_g[:, None], p.mu0[None], p.smu[None]).sum(0)
    S_chi = _gauss(p.chi_g[:, None], p.chi0[None], p.schi[None]).sum(0)
    R = S_mu * S_chi
    sig = p.mob_train - p.base_ob[None]
    A = (sig * g_ob).sum(0) * R / np.maximum(((g_ob * R) ** 2).sum(0), 1e-12)
    A = np.maximum(A, 0.0)
    b = p.base_ob / (len(p.mu_g) * len(p.chi_g))
    return A, b


def predict_plane(p: SeparableParams, A, b, ob_k: float) -> np.ndarray:
    """Predicted counts of plane ``ob_k``: ``(n_mu, n_chi, P)``."""
    g_ob = _gauss(ob_k, p.ob0, p.sob)
    G_mu = _gauss(p.mu_g[:, None], p.mu0[None], p.smu[None])
    G_chi = _gauss(p.chi_g[:, None], p.chi0[None], p.schi[None])
    return b[None, None] + (A * g_ob)[None, None] * G_mu[:, None, :] * G_chi[None, :, :]


def score_plane(obs: np.ndarray, p: SeparableParams, ob_k: float, *, chunk: int = 40_000,
                A=None, b=None, return_pred: bool = False):
    """Per-pixel SSE and SST of the prediction against ``obs`` ``(n_mu, n_chi, P)``.

    SST is about each pixel's own mean over its frames, so pixel-to-pixel brightness structure
    cannot inflate the score. Streams over pixel chunks in float64.
    """
    if A is None:
        A, b = amplitude_and_baseline(p)
    P = obs.shape[-1]
    sse = np.empty(P); sst = np.empty(P)
    pred_all = np.empty(obs.shape, np.float32) if return_pred else None
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        q = p.take(slice(s, e))
        pr = predict_plane(q, A[s:e], b[s:e], ob_k)
        y = obs[:, :, s:e].astype(np.float64)
        sse[s:e] = ((y - pr) ** 2).sum((0, 1))
        sst[s:e] = ((y - y.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
        if return_pred:
            pred_all[:, :, s:e] = pr
    return (sse, sst, pred_all) if return_pred else (sse, sst)


def pooled_r2(sse, sst, mask=None) -> float:
    """``1 - sum(SSE) / sum(SST)`` over ``mask`` (default: every pixel)."""
    if mask is not None:
        sse, sst = sse[mask], sst[mask]
    return float(1.0 - sse.sum() / max(sst.sum(), 1e-12))


# ---------------------------------------------------------------- comparison models

def uniform_params(p: SeparableParams, mask) -> SeparableParams:
    """N0: every pixel gets the grain-median centres and widths. A and b stay per pixel."""
    med = lambda a: np.full_like(a, np.median(a[mask]))
    return replace(p, mu0=med(p.mu0), chi0=med(p.chi0), ob0=med(p.ob0), smu=med(p.smu),
                   schi=med(p.schi), sob=med(p.sob))


def moment_width_params(p: SeparableParams, mask=None) -> SeparableParams:
    """The registered-v1 model: the same centres with the baseline-removed second-moment widths."""
    return replace(p, smu=p.smu_mom, schi=p.schi_mom, sob=p.sob_mom)


def ablate_params(p: SeparableParams, mask, which) -> SeparableParams:
    """Replace the named fields (any of ``smu schi sob mu0 chi0 ob0``) by their grain median."""
    kw = {n: np.full_like(getattr(p, n), np.median(getattr(p, n)[mask])) for n in which}
    return replace(p, **kw)


# ---------------------------------------------------------------- permutation nulls

def complete_blocks(mask: np.ndarray, block: int = 64):
    """Top-left ``(r0, c0)`` of every ``block x block`` square lying wholly inside ``mask``."""
    H, W = mask.shape
    out = []
    for r0 in range(0, H - block + 1, block):
        for c0 in range(0, W - block + 1, block):
            if mask[r0:r0 + block, c0:c0 + block].all():
                out.append((r0, c0))
    return out


def _pool_index(rc, blocks, block):
    """``(n_blocks, block, block)`` table giving the pool pixel index at each block offset."""
    where = {(int(r), int(c)): i for i, (r, c) in enumerate(map(tuple, rc))}
    tab = np.empty((len(blocks), block, block), np.int64)
    for b, (r0, c0) in enumerate(blocks):
        for dr in range(block):
            for dc in range(block):
                tab[b, dr, dc] = where[(r0 + dr, c0 + dc)]
    return tab


def block_donors(rc_pool, blocks, block, rng, *, idx_table=None):
    """Donor pool index for every pool pixel under one random permutation of the blocks.

    Pixels outside the listed blocks keep themselves (they are never scored by the null).
    """
    tab = idx_table if idx_table is not None else _pool_index(rc_pool, blocks, block)
    perm = rng.permutation(len(blocks))
    donor = np.arange(len(rc_pool))
    for b, pb in enumerate(perm):
        donor[tab[b].ravel()] = tab[pb].ravel()
    return donor


def pixel_donors(n, rng):
    return rng.permutation(n)


def swap_bundle(own: SeparableParams, pool: SeparableParams, donor_idx) -> SeparableParams:
    """``own``'s data (training totals, baseline) with the orientation bundle of ``pool[donor]``."""
    return replace(own, mu0=pool.mu0[donor_idx], chi0=pool.chi0[donor_idx],
                   ob0=pool.ob0[donor_idx], smu=pool.smu[donor_idx],
                   schi=pool.schi[donor_idx], sob=pool.sob[donor_idx])


def permutation_null(obs, own: SeparableParams, pool: SeparableParams, ob_k, donor_fn,
                     n_perm: int, rng, *, score_idx_in_pool) -> dict:
    """Null distribution of pooled R^2 when each scored pixel wears another pixel's bundle.

    ``obs`` ``(n_mu, n_chi, S)`` and ``own`` are the S scored pixels; ``pool`` holds every
    candidate donor; ``donor_fn(rng)`` returns a donor index per POOL pixel;
    ``score_idx_in_pool`` maps scored pixel -> its pool index. ``A`` is refitted for the donated
    bundle on the scored pixel's own training totals, so the null has the same amplitude freedom
    as the real model.
    """
    r2 = np.empty(n_perm)
    for i in range(n_perm):
        donor = donor_fn(rng)[score_idx_in_pool]
        q = swap_bundle(own, pool, donor)
        sse, sst = score_plane(obs, q, ob_k)
        r2[i] = pooled_r2(sse, sst)
    return dict(r2=r2)


# ---------------------------------------------------------------- leave-one-plane-out

def leave_one_plane_out(cube, mu_g, chi_g, ob_g, folds, masks, variants: Optional[dict] = None,
                        *, base_frac_ang=0.30, base_frac_ob=0.25, keep_params: bool = False):
    """Fit on every plane but ``k`` and score plane ``k``, for each ``k`` in ``folds``.

    ``masks[k]`` is the (P,) bool mask of the pixels scored in fold k (the grain mask derived
    WITHOUT plane k). ``variants`` maps a name to ``f(params, mask) -> params``; ``"full"`` (the
    unmodified maps) is always scored. Returns ``{fold: {variant: dict(sse, sst, r2)}}`` plus,
    when ``keep_params``, the fold's :class:`SeparableParams` under ``"_params"``.
    """
    n_ob = cube.shape[2]
    variants = dict(variants or {})
    variants.setdefault("full", lambda p, m: p)
    res = {}
    for k in folds:
        train = [j for j in range(n_ob) if j != k]
        p = fit_separable_params(cube, mu_g, chi_g, ob_g, train,
                                 base_frac_ang=base_frac_ang, base_frac_ob=base_frac_ob)
        obs = cube[:, :, k, :]
        res[k] = {}
        for name, f in variants.items():
            q = f(p, masks[k])
            A, b = amplitude_and_baseline(q)
            sse, sst = score_plane(obs, q, ob_g[k], A=A, b=b)
            res[k][name] = dict(sse=sse, sst=sst, r2=pooled_r2(sse, sst, masks[k]))
        if keep_params:
            res[k]["_params"] = p
    return res


# ---------------------------------------------------------------- synthetic controls

def synthetic_cube(p: SeparableParams, ob_all, *, noise_var0: float, gain: float, rng,
                   second_peak: Optional[tuple] = None) -> np.ndarray:
    """A cube drawn from a model (no pedestal, already 'subtracted'), with Gaussian noise.

    ``p`` must have been fitted on ALL planes (``ob_train == ob_all``). Noise variance per value
    is ``noise_var0 + gain * max(signal, 0)``. ``second_peak=(offset_in_sigma, rel_height[, axis])``
    adds a second component, offset along ``axis`` (``"mu"`` default, or ``"chi"``) by that many
    sigmas of the axis, making the truth NOT a single Gaussian (the misspecified control). Choose
    the axis the scan can RESOLVE: mu at 1.16 points per FWHM cannot separate two peaks 3 sigma
    (~1.5 steps) apart, chi at 3.35 can.
    """
    A, b = amplitude_and_baseline(p)
    n_mu, n_chi, P = len(p.mu_g), len(p.chi_g), p.n_pixels
    cube = np.empty((n_mu, n_chi, len(ob_all), P), np.float32)
    for k, ob in enumerate(ob_all):
        mu_part = _gauss(p.mu_g[:, None], p.mu0[None], p.smu[None])
        chi_part = _gauss(p.chi_g[:, None], p.chi0[None], p.schi[None])
        if second_peak is not None:
            off, h = second_peak[:2]
            axis = second_peak[2] if len(second_peak) > 2 else "mu"
            if axis == "mu":
                mu_part = mu_part + h * _gauss(p.mu_g[:, None], (p.mu0 + off * p.smu)[None],
                                               p.smu[None])
            elif axis == "chi":
                chi_part = chi_part + h * _gauss(p.chi_g[:, None], (p.chi0 + off * p.schi)[None],
                                                 p.schi[None])
            else:
                raise ValueError("second_peak axis must be 'mu' or 'chi'")
        sig = (A * _gauss(ob, p.ob0, p.sob))[None, None] * mu_part[:, None] * chi_part[None]
        mean = b[None, None] + sig
        sd = np.sqrt(noise_var0 + gain * np.maximum(sig, 0.0))
        cube[:, :, k, :] = mean + sd * rng.standard_normal(mean.shape)
    return cube


# ---------------------------------------------------------------- noise read from the data

def read_noise_floor(far_plane: np.ndarray, mask=None) -> float:
    """Median over pixels of the variance across one far-wing plane's frames: ``sigma_0^2``.

    ``far_plane`` is ``(n_mu, n_chi, P)`` of the plane farthest from the grain's obpitch peak.
    It still carries a little signal, so this slightly OVERSTATES the read noise.
    """
    v = far_plane.reshape(-1, far_plane.shape[-1]).var(0)
    return float(np.median(v if mask is None else v[mask]))


def gain_from_residuals(pred: np.ndarray, base, resid: np.ndarray, *, n_bins: int = 10):
    """Upper-bound gain: slope of residual variance against predicted signal, by decile.

    Misfit inflates the residual variance, so this is an UPPER bound on the photon-transfer
    slope, and its intercept an upper bound on ``sigma_0^2``. Returns ``(gain, intercept)``.
    """
    sig = (pred - base[None, None]).ravel()
    r2 = (resid ** 2).ravel()
    order = np.argsort(sig)
    chunks = np.array_split(order, n_bins)
    x = np.array([sig[c].mean() for c in chunks]); y = np.array([r2[c].mean() for c in chunks])
    g, c0 = np.polyfit(x, y, 1)
    return float(max(g, 0.0)), float(c0)


# ---------------------------------------------------------------- reading the real planes

def load_planes_cube(h5_path: str, *, planes=None, pedestal_percentile: float = 5.0,
                     background_percentile: float = 5.0, threshold_frac: float = 0.10,
                     round_decimals: int = 3, require_consistent_order: bool = True,
                     roi=None, log=print):
    """Read every plane of an ID03 master file into the unclipped pixel cube.

    Two passes over the file. Pass 1: per-plane frame-order check, per-frame pedestal (5th
    percentile of the whole crop), total image. Then the grain mask of each fold (derived from the
    OTHER planes' totals, so a held-out plane never sees its own mask) and their union. Pass 2:
    rebuild the frames and keep only union-mask pixels.

    Returns a dict: ``cube (n_mu, n_chi, n_ob, P)`` float32, ``mu_g chi_g ob_g``, ``rc (P, 2)``
    pixel (row, col), ``union`` and ``fold_masks`` (``(n_ob, H, W)`` bool), ``totals (n_ob, H, W)``,
    ``order_checks`` and the per-plane pedestal arrays. With an ROI, ``scans`` holds the cropped
    ``RockingScan`` of every plane (``None`` for a full read). ``roi`` is ``(row0, row1, col0, col1)``
    and is for smoke tests: the per-frame pedestal is then taken over the ROI, not the full crop.
    """
    from .io_id03 import list_id03_planes, load_id03_scan
    from .rocking import check_frame_order

    names = list(planes) if planes is not None else list_id03_planes(h5_path)
    ped, totals, ob_g, checks = [], [], [], []
    cache = []                      # with an ROI the frames are small: keep them, read each plane once
    mu_g = chi_g = None
    for n in names:
        sc = load_id03_scan(h5_path, plane=n, roi=roi)
        chk = check_frame_order(sc)
        checks.append(chk)
        log(f"plane {n}: {chk.message.splitlines()[0]}")
        if require_consistent_order and not chk.consistent:
            raise RuntimeError(f"plane {n}: frame order NOT CONSISTENT -- {chk.message}")
        fr = sc.frames
        pd = np.percentile(fr.reshape(fr.shape[0], -1), pedestal_percentile, axis=1)
        ped.append(pd)
        totals.append((fr - pd[:, None, None]).sum(0, dtype=np.float64))
        ob_g.append(float(sc.meta["obpitch"]))
        mu_r = np.round(sc.motors["mu"], round_decimals); chi_r = np.round(sc.motors["chi"], round_decimals)
        if mu_g is None:
            mu_g = np.sort(np.unique(mu_r)); chi_g = np.sort(np.unique(chi_r))
        if roi is not None:
            cache.append(sc)
        del fr
        if roi is None:
            del sc
    totals = np.stack(totals); ob_g = np.asarray(ob_g)
    n_ob = len(names)
    fold_masks = np.stack([
        derive_grain_mask(totals[[j for j in range(n_ob) if j != k]].sum(0),
                          background_percentile=background_percentile,
                          threshold_frac=threshold_frac) for k in range(n_ob)])
    union = fold_masks.any(0)
    rr, cc = np.nonzero(union)
    P = rr.size
    log(f"grain pixels in the union mask: {P}")
    cube = np.empty((len(mu_g), len(chi_g), n_ob, P), np.float32)
    for k, n in enumerate(names):
        sc = cache[k] if roi is not None else load_id03_scan(h5_path, plane=n)
        mu_r = np.round(sc.motors["mu"], round_decimals); chi_r = np.round(sc.motors["chi"], round_decimals)
        i = np.searchsorted(mu_g, mu_r); j = np.searchsorted(chi_g, chi_r)
        if not (np.allclose(mu_g[i], mu_r) and np.allclose(chi_g[j], chi_r)):
            raise RuntimeError(f"plane {n}: a frame's (mu, chi) is not on the grid")
        cell = np.zeros((len(mu_g), len(chi_g)), int)
        np.add.at(cell, (i, j), 1)
        if not (cell == 1).all():
            raise RuntimeError(f"plane {n}: the mesh is not one frame per (mu, chi) cell")
        sel = sc.frames[:, rr, cc] - ped[k][:, None]
        cube[i, j, k, :] = sel
        del sc
    return dict(cube=cube, mu_g=mu_g, chi_g=chi_g, ob_g=ob_g, rc=np.stack([rr, cc], 1),
                union=union, fold_masks=fold_masks, totals=totals, order_checks=checks,
                pedestals=ped, planes=names, scans=cache if roi is not None else None)
