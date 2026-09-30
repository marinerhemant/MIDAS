"""Per-pixel rocking-curve reduction on a scan GRID of any dimension: 1-D rocks and theta-2theta
scans, 2-D meshes (theta x chi, ...), 3-D (theta x chi x 2theta) -- the ``support`` estimator.

Why a second estimator. :func:`midas_dfxm.rocking.reduce_rocking` (``window="peak"``) sizes each
pixel's window as ``peak_halfwidth x (points above half max)`` either side of the maximum. Where a
peak is a large fraction of the scan (datasetJ S173: ~15 of 41 points above half max) that window is
wider than the scan, and three things follow: the pixel is flagged ``truncated`` although its
curve returns to the floor inside the scan; no frames remain outside the window, so the baseline
falls back to a percentile of ALL frames (a flank value, biased high); and the off-peak noise is
undefined, so SNR is NaN and the pixel is not ``lit``. The brightest, broadest pixels vanish from
the map. The mesh path (``tilt2d``) has no width and a percentile baseline. Nothing here depends
on the window being narrower than the scan.

Per pixel, on the grid of measured points (missing cells allowed):

noise
    ``sigma`` = 1.4826 x MAD of the second differences along every grid axis with >= 3 points,
    divided by sqrt(6): the per-frame noise of a smooth curve, defined for every pixel whatever
    its peak width (the few frames at a sharp peak are outvoted by the median).
floor
    the mean of the 3-deep band at the scan end FARTHEST from the pixel's (smoothed) maximum, per
    axis: provisional, and the fallback baseline. Where the peak's tail never reaches the floor
    inside the scan this reads the tail level there -- the floor is then not observable, and
    ``end_fraction`` says so.
support
    the connected region around the smoothed maximum where the smoothed signal above the
    baseline is >= max(``frac`` x peak, ``nsig`` x sigma_smoothed), dilated by ``pad``
    grid steps (default 3: a weak peak's support stops early in its own noise; pad 1 lost 6 % of
    a planted weak peak's intensity, pad 3 2.5 %).
    The edge is set by NOISE (``frac=0`` default): a peak-fraction edge (5 %) left 1-5 % tails in
    the "outside" frames and biased the baseline +15 counts on a planted 1000-count peak. Two
    passes: support from the baseline, baseline from outside the support.
baseline
    median of the measured cells outside the support if there are >= ``min_out`` of them, else
    the floor (``baseline_ok`` False). Never a percentile of the whole curve.
centre, spread
    first moment of (I - baseline) over the support, per axis; covariance from the clipped
    signal; ``fwhm`` per axis = Gaussian-equivalent 2.3548 x the RMS width over the support (the
    half-max of a noisy curve is biased low -- the maximum is pushed up by noise: planted 12 mdeg
    read 7.9); ``fwhm_halfmax`` = contiguous half-max of the smoothed support-restricted marginal,
    kept for high-SNR comparison with reduce_rocking. Measured on planted Gaussians (2026-09-27):
    FWHM 2 / 4 / 11.8 / 15 mdeg at 300 counts read 2.01 / 3.92 / 11.22 / 14.30 (Gaussian-equivalent:
    the support ends at the noise and trims broad tails, -5 %) vs 3.17 / 4.56 / 11.60 / 14.99
    (half-max of the smoothed curve: the 3-point smoothing broadens a 2-point peak); a broad LOW
    peak (S996-like, per-frame SNR ~3) 11.37 vs 7.85.
truncated
    the support touches a grid face with smoothed signal still above
    max(``end_frac`` x peak, 3 sigma_smoothed): the range cuts the peak. ``cut`` keeps which
    axis/side; ``end_fraction`` the largest face level over the peak.
shape
    ``main_share`` = positive signal in the support / positive signal in every cell above the
    support threshold (1 = one connected peak; < 1 = more above-threshold signal elsewhere);
    ``shape_resid`` = RMS(s - A g) / RMS(s) over the support, g the Gaussian with the measured
    centre and covariance (0 = Gaussian; tails, asymmetry and double peaks raise it).
lit
    the NEIGHBOURHOOD SNR (same estimator on the ``support_smooth`` box average) >= ``lit_snr``,
    the pixel's OWN SNR >= ``lit_snr_own`` (3: else the box half-width of empty pixels around
    every grain is lit), and >= ``lit_neighbours`` lit of 8 neighbours. ``snr`` is the pixel's own. Values and error
    bars are always the pixel's own; the neighbourhood only decides where the grain is and which
    frames hold its peak.
error bar
    split-half: each repeat half (or even/odd points if no repeats) reduced on the SAME support,
    ``sigma = SD(A - B) / 2`` per axis (5x5 local, as in reduce_rocking).
"""
import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .rocking import (RockingScan, THETA_NAMES, TWO_THETA_NAMES, TILT_NAMES, FIXED_TOL_DEG,
                      _canon, _finite, _fwhm, _box_rms_half)

__all__ = ["grid_axes", "stack_scans", "reduce_support", "SupportMaps", "support_curve", "support_sensitivity"]

_OTHER_TILTS = tuple(n for n in TILT_NAMES if n not in THETA_NAMES)


# --------------------------------------------------------------------------- grid
def _cluster(v, rel=0.3):
    """Integer level per value: sorted values split where the gap exceeds rel x median step."""
    v = np.asarray(v, float)
    u = np.sort(np.unique(np.round(v, 9)))
    if u.size == 1:
        return np.zeros(v.size, int), u
    d = np.diff(u)
    step = np.median(d[d > 1e-9]) if (d > 1e-9).any() else 0.0
    cut = np.flatnonzero(d > rel * step) if step > 0 else np.arange(u.size - 1)
    edges = np.concatenate([[-np.inf], 0.5 * (u[cut] + u[cut + 1]), [np.inf]])
    lev = np.searchsorted(edges, v, side="right") - 1
    centres = np.array([v[lev == i].mean() for i in range(len(edges) - 1)])
    return lev, centres


def grid_axes(motors: dict, *, fixed_tol_deg: float = FIXED_TOL_DEG):
    """The scan's grid axes from what MOVES in its motor table.

    Returns a list of ``(name, values (M,), kind)`` with kind ``"strain"`` (2theta/2, the d-spacing
    axis) or ``"tilt"``. A moving 2theta is one axis at 2theta/2; theta that tracks it (theta -
    2theta/2 constant to within a third of a step) is absorbed, theta that moves independently
    becomes a tilt axis ``theta - 2theta/2``. Every other moving tilt (chi, phi, mu, omega) is an
    axis. Filenames are never used.
    """
    def key(names):
        for k in motors:
            if _canon(k) in names:
                return k
        return None

    def moves(k):
        f = _finite(motors[k]) if k is not None else np.array([])
        return f.size > 1 and float(np.ptp(f)) > fixed_tol_deg

    tth, th = key(TWO_THETA_NAMES), key(THETA_NAMES)
    axes = []
    if tth is not None and moves(tth):
        half = 0.5 * np.asarray(motors[tth], float)
        axes.append((f"{tth}/2", half, "strain"))
        if th is not None and moves(th):
            res = np.asarray(motors[th], float) - half
            _, cs = _cluster(half)
            step = float(np.median(np.diff(cs))) if cs.size > 1 else 0.0
            if float(np.ptp(res)) > max(fixed_tol_deg, step / 3.0):
                axes.append((f"{th}-{tth}/2", res, "tilt"))
    elif th is not None and moves(th):
        axes.append((th, np.asarray(motors[th], float), "tilt"))
    for k in motors:
        if _canon(k) in _OTHER_TILTS and moves(k):
            axes.append((k, np.asarray(motors[k], float), "tilt"))
    if not axes:
        raise ValueError("no rocking angle moves in this motor table")
    return axes


def _build_grid(axes):
    levs, coords = zip(*[_cluster(v) for _, v, _ in axes])
    shape = tuple(len(c) for c in coords)
    flat = np.ravel_multi_index(tuple(levs), shape)
    if len(np.unique(flat)) != len(flat):
        raise ValueError(f"two scan points map to one grid cell (grid {shape}): repeated angle "
                         "settings must be averaged first, or an axis is mis-clustered")
    if np.prod(shape) > 1.5 * len(flat) + 2:
        raise ValueError(f"the points fill only {len(flat)} of {int(np.prod(shape))} cells of a "
                         f"{shape} grid: the scan is not on a grid (drifting angles?)")
    return shape, flat, [np.asarray(c) for c in coords]


def stack_scans(scans, *, source: Optional[str] = None) -> RockingScan:
    """Stack several scans of ONE sample position into one mesh (e.g. theta rocks at several chi,
    or theta-2theta scans of the four twin peaks). Frames, motors and repeat halves are
    concatenated; the result's ``scan_type`` is ``"mesh"`` and only :func:`reduce_support`
    should reduce it. Frames must share the ROI; repeat counts must agree."""
    scans = list(scans)
    if len({s.frames.shape[1:] for s in scans}) != 1:
        raise ValueError("scans have different frame shapes (ROI)")
    if len({s.n_repeats for s in scans}) != 1:
        raise ValueError("scans have different repeat counts")
    keys = set.intersection(*[set(s.motors) for s in scans])
    motors = {k: np.concatenate([np.asarray(s.motors[k], float) for s in scans]) for k in keys}
    halves = None
    if all(s.halves is not None for s in scans):
        halves = tuple(np.concatenate([s.halves[i] for s in scans]) for i in (0, 1))
    tth = [s.two_theta_deg for s in scans if s.two_theta_deg is not None]
    return RockingScan(frames=np.concatenate([s.frames for s in scans]), motors=motors,
                       scan_type="mesh", axes=tuple(a for a, _, _ in grid_axes(motors)),
                       two_theta_deg=float(np.median(tth)) if tth else None, halves=halves,
                       n_repeats=scans[0].n_repeats, dark_level=scans[0].dark_level,
                       source=source or " + ".join(s.source for s in scans),
                       notes=[f"stacked from {len(scans)} scans"],
                       meta={"stacked": [s.source for s in scans]})


# --------------------------------------------------------------------------- core
def _shift(a, ax, step):
    """a shifted by `step` along grid axis `ax` (axis of a), NaN padded."""
    out = np.full_like(a, np.nan)
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if step > 0:
        src[ax] = slice(None, -step); dst[ax] = slice(step, None)
    else:
        src[ax] = slice(-step, None); dst[ax] = slice(None, step)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _smooth(D, shape):
    out = D
    for ax, n in enumerate(shape):
        if n >= 3:
            with np.errstate(invalid="ignore"):
                st = np.stack([_shift(out, ax, 1), out, _shift(out, ax, -1)])
                cnt = np.isfinite(st).sum(0)
                out = np.where(np.isfinite(out), np.nansum(st, 0) / np.maximum(cnt, 1), np.nan)
    return out


def _noise(D, shape):
    d2 = []
    for ax, n in enumerate(shape):
        if n >= 3:
            a = D
            v = _shift(a, ax, 1) - 2 * a + _shift(a, ax, -1)
            v = v.reshape(-1, v.shape[-1])
            v = v[np.isfinite(v).all(1)]
            d2.append(v)
    if not d2:
        return np.full(D.shape[-1], np.nan)
    d2 = np.concatenate(d2)
    if d2.shape[0] < 5:
        return np.full(D.shape[-1], np.nan)
    med = np.median(d2, 0)
    return 1.4826 * np.median(np.abs(d2 - med), 0) / math.sqrt(6.0)


def _floor(sm, n=3):
    v = sm.reshape(-1, sm.shape[-1])
    v = np.where(np.isfinite(v), v, np.inf)
    lo = np.sort(v, 0)[:n]
    lo = np.where(np.isfinite(lo), lo, np.nan)
    return np.nanmean(lo, 0)


def _ends_floor(D, shape, peak_idx, k=3):
    """Floor from the scan END FARTHEST from the pixel's peak, per axis: the mean over a k-deep
    band of measured cells at that end, averaged over axes with >= 2k points. The side is chosen
    by geometry (where the peak is), not by the data values, so there is no selection bias --
    choosing the LOWEST end (a min over noisy means) needed a correction that over-shot by +5
    counts whenever the two ends differed (planted broad peak). Returns (floor, cells averaged)."""
    N = D.shape[-1]
    acc = np.zeros(N); cnt = np.zeros(N); used = 0
    for ax, n in enumerate(shape):
        if n < 2 * k:
            continue
        used += 1
        far_hi = peak_idx[:, ax] < (n - 1) / 2.0                 # peak in the low half: use the high end
        for sl, pick in ((slice(0, k), ~far_hi), (slice(n - k, n), far_hi)):
            idx = [slice(None)] * D.ndim
            idx[ax] = sl
            band = D[tuple(idx)].reshape(-1, N)
            with np.errstate(all="ignore"):
                m = np.nanmean(band, 0)
            nb = np.isfinite(band).sum(0)
            acc += np.where(pick, np.nan_to_num(m) * nb, 0.0)
            cnt += np.where(pick, nb, 0)
    if used == 0:
        return _floor(D), np.full(N, 3.0)
    with np.errstate(all="ignore"):
        return np.where(cnt > 0, acc / cnt, _floor(D)), np.maximum(cnt, 1)


def _dilate(reg, shape, measured):
    grow = reg.copy()
    for ax, n in enumerate(shape):
        if n < 2:
            continue
        a = [slice(None)] * reg.ndim; b = [slice(None)] * reg.ndim
        a[ax] = slice(1, None); b[ax] = slice(None, -1)
        grow[tuple(a)] |= reg[tuple(b)]
        grow[tuple(b)] |= reg[tuple(a)]
    return grow & measured


def _support(sm_s, thr, shape, measured):
    """Connected region of (sm_s >= thr) containing the argmax, per pixel."""
    N = sm_s.shape[-1]
    flat = np.where(np.isfinite(sm_s), sm_s, -np.inf).reshape(-1, N)
    k = flat.argmax(0)
    above = (np.nan_to_num(sm_s, nan=-np.inf) >= thr)
    if len(shape) == 1:                                   # fast exact path: a contiguous run
        M = shape[0]
        idx = np.arange(M)[:, None]
        below = ~above
        left = np.where(below & (idx < k[None]), idx, -1).max(0) + 1
        right = np.where(below & (idx > k[None]), idx, M).min(0) - 1
        reg = (idx >= left[None]) & (idx <= right[None])
        return reg, k
    seed = np.zeros(flat.shape, bool)
    seed[k, np.arange(N)] = True
    reg = seed.reshape(sm_s.shape) & above
    for _ in range(int(sum(shape))):
        grow = _dilate(reg, shape, measured[..., None] if measured.ndim < reg.ndim else measured) & above
        if (grow == reg).all():
            break
        reg = grow
    return reg, k


def _faces(shape):
    for ax, n in enumerate(shape):
        for side, i in ((0, 0), (1, n - 1)):
            sl = [slice(None)] * len(shape)
            sl[ax] = i
            yield ax, side, tuple(sl)


def support_curve(D, shape, coords, *, frac=0.0, nsig=2.0, pad=3, min_out=6, end_frac=0.10,
                  halves=None, Dw=None):
    """Reduce per-pixel curves on a grid. ``D`` is ``(*shape, N)`` with NaN in unmeasured cells;
    ``coords`` one 1-D array of grid coordinates (deg) per axis. Returns a dict of (N,) arrays
    (``centre`` and ``fwhm`` are (N, d), ``cov`` (N, d, d), ``cut`` (N, d, 2)).

    ``Dw`` (same shape): curves that DECIDE the support, e.g. a spatial k x k average of the
    pixel's neighbourhood. The support, its noise threshold, the cut test and ``end_fraction``
    come from ``Dw``; the baseline (median of the pixel's OWN frames outside that support),
    centre, covariance, width, intensity, SNR and shape come from ``D``. Without it a pixel whose
    per-frame SNR is ~1 (a broad, low curve: datasetJ S996) seeds its support on a noise spike and
    keeps a fraction of its own counts."""
    D = np.asarray(D, float)
    Dw = D if Dw is None else np.asarray(Dw, float)
    N = D.shape[-1]
    d = len(shape)
    measured = np.isfinite(D[..., 0])
    meas_n = measured[..., None]
    nsm = math.sqrt(3.0 ** sum(1 for n in shape if n >= 3))
    sig = _noise(D, shape)                                   # the pixel's own per-frame noise
    sigw = sig if Dw is D else _noise(Dw, shape)
    sigw_sm = sigw / nsm
    sm0w = _smooth(Dw, shape)
    kw = np.nanargmax(np.where(np.isfinite(sm0w), sm0w, -np.inf).reshape(-1, N), 0)
    peak_idx = np.stack(np.unravel_index(kw, shape), 1)      # (N, d) grid index of the smoothed max
    b0w, n_floor = _ends_floor(Dw, shape, peak_idx)
    bw = b0w.copy()
    for _ in range(2):                                       # support <- baseline <- support, on Dw
        smw = sm0w - bw
        pkw = np.nanmax(smw.reshape(-1, N), 0)
        thr = np.maximum(frac * pkw, nsig * sigw_sm)
        reg, k = _support(smw, thr, shape, meas_n)
        W = reg
        for _p in range(pad):
            W = _dilate(W, shape, meas_n)
        n_out = (meas_n & ~W).reshape(-1, N).sum(0)
        with np.errstate(all="ignore"):
            med_w = np.nanmedian(np.where(W | ~meas_n, np.nan, Dw).reshape(-1, N), 0)
        bw = np.where(n_out >= min_out, med_w, b0w)
    smw = sm0w - bw
    pkw = np.nanmax(smw.reshape(-1, N), 0)
    thr = np.maximum(frac * pkw, nsig * sigw_sm)
    reg, k = _support(smw, thr, shape, meas_n)
    W = reg
    for _p in range(pad):
        W = _dilate(W, shape, meas_n)
    # the pixel's own baseline on that support
    n_out = (meas_n & ~W).reshape(-1, N).sum(0)
    ok = n_out >= min_out
    sm0 = sm0w if Dw is D else _smooth(D, shape)
    with np.errstate(all="ignore"):
        med_out = np.nanmedian(np.where(W | ~meas_n, np.nan, D).reshape(-1, N), 0)
    b = np.where(ok, med_out, _ends_floor(D, shape, peak_idx)[0])
    s = D - b
    sm = sm0 - b
    pk = np.nanmax(sm.reshape(-1, N), 0)
    sw = np.where(W & meas_n, s, 0.0)
    S = sw.reshape(-1, N).sum(0)
    grids = np.meshgrid(*coords, indexing="ij")
    cen = np.full((N, d), np.nan)
    with np.errstate(all="ignore"):
        for a in range(d):
            cen[:, a] = (sw * grids[a][..., None]).reshape(-1, N).sum(0) / np.where(S > 0, S, np.nan)
    # a first moment outside the scanned range has no meaning (sum of signal ~ 0 over the support,
    # e.g. a single noisy frame): planted by a single-repeat reduction, it gave 2e12 mdeg
    for a in range(d):
        lo, hi = float(np.min(coords[a])), float(np.max(coords[a]))
        cen[:, a] = np.where((cen[:, a] >= lo) & (cen[:, a] <= hi), cen[:, a], np.nan)
    swp = np.clip(sw, 0.0, None)
    Sp = swp.reshape(-1, N).sum(0)
    cov = np.full((N, d, d), np.nan)
    with np.errstate(all="ignore"):
        for a in range(d):
            for c in range(a, d):
                da = grids[a][..., None] - cen[:, a]
                dc = grids[c][..., None] - cen[:, c]
                v = (swp * da * dc).reshape(-1, N).sum(0) / np.where(Sp > 0, Sp, np.nan)
                cov[:, a, c] = cov[:, c, a] = v
    # marginal half-max width per axis
    # on the pixel's grid-smoothed signal: on the raw curve the half-max crossings sit around the
    # tallest noise spike when the per-frame SNR is ~1 (planted: 0.65 mdeg reported for 12 mdeg)
    fw = np.full((N, d), np.nan)
    swm = np.clip(np.where(W & meas_n, sm, 0.0), 0.0, None)
    for a in range(d):
        other = tuple(i for i in range(d) if i != a)
        marg = swm.sum(axis=other) if other else swm               # (shape[a], N)
        if shape[a] >= 2:
            kk = marg.argmax(0)
            w_, _, _ = _fwhm(marg, np.asarray(coords[a], float), kk)
            fw[:, a] = w_
    # RMS width over the UNPADDED support only: the pad frames (there for the intensity) carry
    # noise at a large lever arm and inflated a 2 mdeg planted width to 2.7
    swr = np.clip(np.where(reg & meas_n, s, 0.0), 0.0, None)
    Sr = swr.reshape(-1, N).sum(0)
    rms_w = np.full((N, d), np.nan)
    with np.errstate(all="ignore"):
        for a in range(d):
            ca = (swr * grids[a][..., None]).reshape(-1, N).sum(0) / np.where(Sr > 0, Sr, np.nan)
            va = (swr * (grids[a][..., None] - ca) ** 2).reshape(-1, N).sum(0) / np.where(Sr > 0, Sr, np.nan)
            rms_w[:, a] = np.where(np.isfinite(va), np.sqrt(np.clip(va, 0, None)),
                                   np.sqrt(np.clip(cov[:, a, a], 0, None)))   # empty core: padded
    # truncation: support touches a face with signal still above the cut level
    cut_thr = np.maximum(end_frac * pkw, 3.0 * sigw_sm)
    cut = np.zeros((N, d, 2), bool)
    endf = np.zeros(N)
    for ax, side, sl in _faces(shape):
        if shape[ax] < 2:
            continue
        face_s = np.where(reg[sl], smw[sl], -np.inf).reshape(-1, N).max(0)
        cut[:, ax, side] = face_s > cut_thr
        with np.errstate(all="ignore"):
            endf = np.fmax(endf, np.where(pkw > 0, face_s / pkw, np.nan))
    endf = np.where(np.isfinite(endf), np.clip(endf, 0, None), np.nan)
    truncated = cut.reshape(N, -1).any(1)
    # shape: connected share and Gaussian residual
    all_above = (np.nan_to_num(smw, nan=-np.inf) >= thr) & meas_n
    tot_above = np.where(all_above, np.clip(s, 0, None), 0.0).reshape(-1, N).sum(0)
    in_reg = np.where(reg, np.clip(s, 0, None), 0.0).reshape(-1, N).sum(0)
    with np.errstate(all="ignore"):
        main_share = np.where(tot_above > 0, in_reg / tot_above, np.nan)
    steps = [float(np.median(np.diff(c))) if len(c) > 1 else 1.0 for c in coords]
    reg_cov = cov + np.einsum("a,ab->ab", np.array([(st ** 2) / 12.0 for st in steps]), np.eye(d))[None]
    with np.errstate(all="ignore"):
        try:
            P = np.linalg.inv(np.where(np.isfinite(reg_cov), reg_cov, np.eye(d)[None]))
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(np.nan_to_num(reg_cov))
        q = np.zeros(D.shape)
        for a in range(d):
            for c in range(d):
                da = grids[a][..., None] - cen[:, a]
                dc = grids[c][..., None] - cen[:, c]
                q = q + da * P[:, a, c] * dc
        g = np.where(W & meas_n, np.exp(-0.5 * q), 0.0)
        A = (g * sw).reshape(-1, N).sum(0) / np.where((g * g).reshape(-1, N).sum(0) > 0,
                                                       (g * g).reshape(-1, N).sum(0), np.nan)
        r = np.where(W & meas_n, s - A * g, 0.0)
        shape_resid = np.sqrt((r * r).reshape(-1, N).sum(0) / np.where((sw * sw).reshape(-1, N).sum(0) > 0,
                                                                         (sw * sw).reshape(-1, N).sum(0), np.nan))
        n_sup = (W & meas_n).reshape(-1, N).sum(0)
        # the baseline is subtracted n times, so its own error enters n^2 times: median of n_out
        # frames var ~ pi/(2 n_out) sigma^2; the end floor: Monte Carlo variance. Leaving it out
        # overstated SNR ~3x on floor-baseline pixels and lit pure noise next to a grain.
        v_b = np.where(ok, np.pi / (2.0 * np.maximum(n_out, 1)), 1.0 / n_floor)
        n1 = np.maximum(n_sup, 1)
        snr = np.where(sig > 0, S / (sig * np.sqrt(n1 + n1 ** 2 * v_b)), np.nan)
        # the same on the support-deciding curves: whether the neighbourhood holds a peak here
        if Dw is D:
            snr_w = snr
        else:
            Sw_ = np.where(W & meas_n, Dw - bw, 0.0).reshape(-1, N).sum(0)
            okw = n_out >= min_out
            v_bw = np.where(okw, np.pi / (2.0 * np.maximum(n_out, 1)), 1.0 / n_floor)
            snr_w = np.where(sigw > 0, Sw_ / (sigw * np.sqrt(n1 + n1 ** 2 * v_bw)), np.nan)
    out = dict(centre=cen, cov=cov, fwhm=2.3548 * rms_w, fwhm_halfmax=fw, rms_width=rms_w, snr_w=snr_w, intensity=S, baseline=b, baseline_ok=ok,
               truncated=truncated, cut=cut, end_fraction=endf, snr=snr, sigma_frame=sig,
               n_support=n_sup, main_share=main_share, shape_resid=shape_resid, peak=pk,
               argmax=k)
    if halves is not None:
        cs = []
        for J in halves:
            J = np.asarray(J, float)
            bj0 = _ends_floor(J, shape, peak_idx)[0]
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(W | ~meas_n, np.nan, J).reshape(-1, N), 0)
            bj = np.where(ok, med, bj0)
            sj = np.where(W & np.isfinite(J), J - bj, 0.0)          # halves may leave cells empty
            Sj = sj.reshape(-1, N).sum(0)
            cj = np.full((N, d), np.nan)
            with np.errstate(all="ignore"):
                for a in range(d):
                    cj[:, a] = (sj * grids[a][..., None]).reshape(-1, N).sum(0) / np.where(Sj > 0, Sj, np.nan)
                lo, hi = float(np.min(coords[a])), float(np.max(coords[a]))
                cj[:, a] = np.where((cj[:, a] >= lo) & (cj[:, a] <= hi), cj[:, a], np.nan)
            cs.append(cj)
        out["diff"] = cs[0] - cs[1]
    return out


# --------------------------------------------------------------------------- maps
@dataclass
class SupportMaps:
    """Per-pixel maps from :func:`reduce_support`; images are ``(H, W)`` or ``(H, W, d)``.

    ``value[..., a]`` is the centre along axis ``a`` relative to ``reference_deg[a]``, in
    ``units[a]`` (mdeg for a tilt axis, microstrain = -cot(theta_B) x Delta(2theta/2) for the
    strain axis: relative Delta d/d from one reflection, not an elastic strain).
    """
    axes: tuple
    kinds: tuple
    units: tuple
    scales: tuple
    grid_shape: tuple
    grid_coords: list
    value: np.ndarray
    centre_deg: np.ndarray
    reference_deg: np.ndarray
    cov_deg2: np.ndarray
    fwhm_mdeg: np.ndarray
    intensity: np.ndarray
    snr: np.ndarray
    sigma_frame: np.ndarray
    baseline: np.ndarray
    baseline_ok: np.ndarray
    lit: np.ndarray
    truncated: np.ndarray
    cut: np.ndarray
    end_fraction: np.ndarray
    n_support: np.ndarray
    main_share: np.ndarray
    shape_resid: np.ndarray
    sigma: Optional[np.ndarray]
    sigma_global: Optional[list]
    split: Optional[str]
    pedestal_share: float
    settings: dict
    notes: list = field(default_factory=list)
    fwhm_halfmax_mdeg: Optional[np.ndarray] = None   # contiguous half-max of the smoothed marginal
    snr_neighbourhood: Optional[np.ndarray] = None   # SNR of the support_smooth box (decides lit)

    def summary(self) -> str:
        lit = self.lit
        good = lit & ~self.truncated
        L = [f"support map: grid {self.grid_shape} over axes {list(self.axes)}; {int(lit.sum())} lit px "
             f"({lit.mean():.1%} of the frame); {int((lit & self.truncated).sum())} of them cut by the "
             "scan range (curve above the cut level on a grid face)"]
        for a, (ax, u) in enumerate(zip(self.axes, self.units)):
            if good.any():
                q = np.nanpercentile(self.value[good, a], [2, 25, 50, 75, 98])
                f = np.nanpercentile(self.fwhm_mdeg[good, a], [25, 50, 75])
                L.append(f"axis {ax:12s} value median {q[2]:.3g} {u}, IQR [{q[1]:.3g}, {q[3]:.3g}], "
                         f"p2-p98 [{q[0]:.3g}, {q[4]:.3g}]; marginal FWHM median {f[1]:.3g} mdeg "
                         f"(IQR {f[0]:.3g}-{f[2]:.3g})"
                         + (f"; split-half sigma {self.sigma_global[a]:.3g} {u}" if self.sigma_global else ""))
        if lit.any():
            L.append(f"shape       main_share median {np.nanmedian(self.main_share[lit]):.3f} (1 = one "
                     f"connected peak); shape_resid median {np.nanmedian(self.shape_resid[lit]):.3f} "
                     "(0 = Gaussian)")
            L.append(f"baseline    {1 - self.baseline_ok[lit].mean():.1%} of lit px had < "
                     f"{self.settings['min_out']} cells outside the support (floor used)")
        L.append(f"pedestal    {self.pedestal_share:.3f} of the recorded counts in lit pixels was baseline")
        L.append("settings    " + ", ".join(f"{k}={self.settings[k]}" for k in
                                             ("frac", "nsig", "pad", "min_out", "end_frac", "lit_snr")))
        L += [f"note        {n}" for n in self.notes]
        return "\n".join(L)

    def save(self, path: str) -> None:
        arr = {k: getattr(self, k) for k in ("value", "centre_deg", "cov_deg2", "fwhm_mdeg", "intensity",
                                              "snr", "sigma_frame", "baseline", "baseline_ok", "lit",
                                              "truncated", "cut", "end_fraction", "n_support",
                                              "main_share", "shape_resid", "fwhm_halfmax_mdeg",
                                              "snr_neighbourhood")}
        if self.sigma is not None:
            arr["sigma"] = self.sigma
        meta = dict(axes=list(self.axes), kinds=list(self.kinds), units=list(self.units),
                    scales=list(self.scales), grid_shape=list(self.grid_shape),
                    grid_coords=[list(map(float, c)) for c in self.grid_coords],
                    reference_deg=np.asarray(self.reference_deg).tolist(), sigma_global=self.sigma_global,
                    split=self.split, pedestal_share=self.pedestal_share, settings=self.settings,
                    notes=self.notes)
        np.savez_compressed(path, **{k: np.asarray(v, np.float32) if np.asarray(v).dtype.kind == "f" else v
                                     for k, v in arr.items()}, meta=json.dumps(meta, default=str))


def reduce_support(scan: RockingScan, *, roi=None, frac: float = 0.0, nsig: float = 2.0, pad: int = 3,
                   min_out: int = 6, end_frac: float = 0.10, lit_snr: float = 10.0,
                   lit_neighbours: int = 3, support_smooth: int = 5, lit_snr_own: float = 3.0,
                   split_half: bool = True, chunk_rows: int = 32, reference=None,
                   backend: str = "numpy", device=None) -> SupportMaps:
    """Reduce any rocking scan -- 1-D, mesh, or stacked -- with the support estimator (module
    docstring). ``roi=None`` reduces the whole frame; the lit mask decides what is signal.

    ``backend="torch"`` runs the identical estimator (:mod:`midas_dfxm.support_torch`, float64) on
    ``device`` (e.g. ``"cuda"``); use a larger ``chunk_rows`` there (128-256)."""
    axes = grid_axes(scan.motors)
    shape, flat, coords = _build_grid(axes)
    d = len(shape)
    F = scan.frames if roi is None else scan.frames[:, roi[0]:roi[1], roi[2]:roi[3]]
    M, H, Wd = F.shape
    if M != len(flat):
        raise ValueError(f"{M} frames vs {len(flat)} motor points")
    halves = None
    if split_half:
        if scan.halves is not None:
            halves = [h if roi is None else h[:, roi[0]:roi[1], roi[2]:roi[3]] for h in scan.halves]
            split = "repeat parity"
        else:
            split = "point parity"
    else:
        split = None
    kinds = tuple(k for _, _, k in axes)
    names = tuple(n for n, _, _ in axes)
    scales, units = [], []
    for k in kinds:
        if k == "strain":
            if scan.two_theta_deg is None:
                raise ValueError("strain axis without a 2theta reading")
            cot = 1.0 / math.tan(math.radians(scan.two_theta_deg / 2.0))
            scales.append(-cot * math.pi / 180.0 * 1e6); units.append("microstrain")
        else:
            scales.append(1000.0); units.append("mdeg")
    G = int(np.prod(shape))
    keys = ("intensity", "baseline", "baseline_ok", "truncated", "end_fraction", "snr", "snr_w", "sigma_frame",
            "n_support", "main_share", "shape_resid")
    res = {k: np.empty(H * Wd) for k in keys}
    res["baseline_ok"] = np.zeros(H * Wd, bool); res["truncated"] = np.zeros(H * Wd, bool)
    cen = np.empty((H * Wd, d)); cov = np.empty((H * Wd, d, d)); fw = np.empty((H * Wd, d)); fwh = np.empty((H * Wd, d))
    cut = np.zeros((H * Wd, d, 2), bool); diff = np.full((H * Wd, d), np.nan)
    total = np.empty(H * Wd)
    odd_even = None
    if split == "point parity":
        odd_even = (np.arange(M) % 2 == 0)
    hk = support_smooth // 2 if support_smooth and support_smooth > 1 else 0
    if hk:
        from scipy import ndimage
    if backend == "torch":
        import torch
        from .support_torch import support_curve_torch, box_filter_rows
        tdev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tdt = torch.float64
        flat_t = torch.as_tensor(flat, device=tdev)
    elif backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'torch'")
    for r0 in range(0, H, chunk_rows):
        if backend == "torch":
            r1 = min(r0 + chunk_rows, H)
            I_t = torch.as_tensor(np.ascontiguousarray(F[:, r0:r1]), device=tdev).to(tdt).reshape(M, -1)
            N = I_t.shape[1]
            def grid_t(vals, rows_idx=flat_t):
                g = torch.full((G, N), float("nan"), device=tdev, dtype=tdt); g[rows_idx] = vals
                return g.reshape(shape + (N,))
            D_t = grid_t(I_t)
            Dw_t = None
            if hk:
                a0, a1 = max(r0 - hk, 0), min(r1 + hk, H)
                slab = torch.as_tensor(np.ascontiguousarray(F[:, a0:a1]), device=tdev).to(tdt)
                Dw_t = grid_t(box_filter_rows(slab, support_smooth)[:, r0 - a0:r0 - a0 + (r1 - r0)].reshape(M, -1))
            hv_t = None
            if halves is not None:
                hv_t = [grid_t(torch.as_tensor(np.ascontiguousarray(h[:, r0:r1]), device=tdev).to(tdt).reshape(M, -1))
                        for h in halves]
            elif odd_even is not None:
                hv_t = []
                for sel_ in (odd_even, ~odd_even):
                    sel_t = torch.as_tensor(sel_, device=tdev)
                    hv_t.append(grid_t(I_t[sel_t], flat_t[sel_t]))
            o = support_curve_torch(D_t, shape, coords, frac=frac, nsig=nsig, pad=pad, min_out=min_out,
                                    end_frac=end_frac, halves=hv_t, Dw=Dw_t)
            sl = slice(r0 * Wd, r1 * Wd)
            for k in keys:
                res[k][sl] = o[k]
            cen[sl] = o["centre"]; cov[sl] = o["cov"]; fw[sl] = o["fwhm"]; fwh[sl] = o["fwhm_halfmax"]; cut[sl] = o["cut"]
            total[sl] = I_t.sum(0).cpu().numpy()
            if "diff" in o:
                diff[sl] = o["diff"]
            continue
        r1 = min(r0 + chunk_rows, H)
        I = F[:, r0:r1].reshape(M, -1).astype(np.float64)
        N = I.shape[1]
        D = np.full((G, N), np.nan); D[flat] = I
        D = D.reshape(shape + (N,))
        Dw = None
        if hk:
            a0, a1 = max(r0 - hk, 0), min(r1 + hk, H)       # halo rows: no seams between chunks
            slab = ndimage.uniform_filter(F[:, a0:a1].astype(np.float64), size=(1, support_smooth, support_smooth),
                                          mode="nearest")[:, r0 - a0:r0 - a0 + (r1 - r0)]
            Dw = np.full((G, N), np.nan); Dw[flat] = slab.reshape(M, -1)
            Dw = Dw.reshape(shape + (N,))
        hv = None
        if halves is not None:
            hv = []
            for h in halves:
                Dh = np.full((G, N), np.nan); Dh[flat] = h[:, r0:r1].reshape(M, -1)
                hv.append(Dh.reshape(shape + (N,)))
        elif odd_even is not None:
            hv = []
            for sel in (odd_even, ~odd_even):
                Dh = np.full((G, N), np.nan); Dh[flat[sel]] = I[sel]
                hv.append(Dh.reshape(shape + (N,)))
        o = support_curve(D, shape, coords, frac=frac, nsig=nsig, pad=pad, min_out=min_out, end_frac=end_frac,
                          halves=hv, Dw=Dw)
        sl = slice(r0 * Wd, r1 * Wd)
        for k in keys:
            res[k][sl] = o[k]
        cen[sl] = o["centre"]; cov[sl] = o["cov"]; fw[sl] = o["fwhm"]; fwh[sl] = o["fwhm_halfmax"]; cut[sl] = o["cut"]
        total[sl] = I.sum(0)
        if "diff" in o:
            diff[sl] = o["diff"]
    im = lambda a: a.reshape((H, Wd) + a.shape[1:])
    notes = list(scan.notes)
    # lit: does the NEIGHBOURHOOD (support_smooth box) hold a peak at SNR >= lit_snr? Each pixel
    # still reports its own value and error; a pixel-only SNR cut blanked real, weak, broad
    # pixels inside the grain (datasetJ S996 holes: planted analogue 36 % dropped at true SNR ~11).
    lit = (np.nan_to_num(res["snr_w"], nan=0.0) >= lit_snr) & (np.nan_to_num(res["snr"], nan=0.0) >= lit_snr_own)
    lit &= np.isfinite(cen).all(1)
    lit = im(lit)
    if lit_neighbours:
        # real DFXM signal spans many pixels; a pure-noise pixel passes SNR >= 10 at ~0.1-0.2 %
        # (planted flat Poisson), scattered. Keep lit pixels with >= lit_neighbours lit of 8.
        from scipy import ndimage
        nb = ndimage.convolve(lit.astype(np.int16), np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int16),
                              mode="constant")
        n_iso = int((lit & (nb < lit_neighbours)).sum())
        lit = lit & (nb >= lit_neighbours)
        notes.append(f"lit: SNR >= {lit_snr:g} and >= {lit_neighbours} of 8 lit neighbours "
                     f"({n_iso} isolated SNR-passing px dropped)")
    trunc = im(res["truncated"])
    good = lit & ~trunc
    refpx = good if good.any() else lit
    if not good.any():
        notes.append("WARNING: no lit pixel inside the scan range; reference over all lit pixels")
    C = im(cen)
    ref = (np.nanmedian(C[refpx], 0) if refpx.any() else np.zeros(d)) if reference is None \
        else np.asarray(reference, float).reshape(d)
    value = (C - ref) * np.asarray(scales)
    sigma = sigma_global = None
    if split is not None:
        Dd = im(diff) * np.abs(np.asarray(scales))
        sigma = np.stack([_box_rms_half(np.where(lit, Dd[..., a], np.nan)) for a in range(d)], -1)
        sigma_global = [float(1.4826 * np.nanmedian(np.abs(Dd[good, a] - np.nanmedian(Dd[good, a]))) / 2.0)
                        if good.any() else None for a in range(d)]
        if split == "point parity":
            notes.append("error bar from even vs odd scan points: includes sampling error, "
                         "conservative below ~3 points per FWHM")
    L = lit.reshape(-1)
    share = float(M * res["baseline"][L].sum() / total[L].sum()) if L.any() and total[L].sum() > 0 else float("nan")
    import midas_dfxm as _dx
    settings = dict(frac=frac, nsig=nsig, pad=pad, min_out=min_out, end_frac=end_frac, lit_snr=lit_snr,
                    lit_neighbours=lit_neighbours, support_smooth=support_smooth, lit_snr_own=lit_snr_own, roi=roi,
                    split=split, chunk_rows=chunk_rows, source=scan.source, backend=backend,
                    midas_dfxm=getattr(_dx, "__version__", "?"), estimator="support")
    return SupportMaps(axes=names, kinds=kinds, units=tuple(units), scales=tuple(scales),
                       grid_shape=shape, grid_coords=coords, value=value, centre_deg=C,
                       reference_deg=ref, cov_deg2=im(cov), fwhm_mdeg=im(fw) * 1000.0,
                       intensity=im(res["intensity"]), snr=im(res["snr"]),
                       sigma_frame=im(res["sigma_frame"]), baseline=im(res["baseline"]),
                       baseline_ok=im(res["baseline_ok"]), lit=lit, truncated=trunc, cut=im(cut),
                       end_fraction=im(res["end_fraction"]), n_support=im(res["n_support"]),
                       main_share=im(res["main_share"]), shape_resid=im(res["shape_resid"]),
                       sigma=sigma, sigma_global=sigma_global, split=split, pedestal_share=share,
                       settings=settings, notes=notes, fwhm_halfmax_mdeg=im(fwh) * 1000.0,
                       snr_neighbourhood=im(res["snr_w"]))


def _block_stats(ref, other, sel, block=8):
    """8x8-block RMS and slope of (other - ref), each referenced to its own median on ``sel``
    (the definition of :func:`midas_dfxm.baseline_sensitivity`), plus the difference map."""
    both = sel & np.isfinite(ref) & np.isfinite(other)
    d = np.full(ref.shape, np.nan)
    if both.sum() < 3:
        return d, float("nan"), float("nan"), float("nan")
    a = np.where(both, ref - np.median(ref[both]), 0.0)
    c = np.where(both, other - np.median(other[both]), 0.0)
    d[both] = (c - a)[both]
    dn = d[both]
    spread = float(1.4826 * np.median(np.abs(dn - np.median(dn))))
    H, W = both.shape
    h, w = H // block, W // block
    rms = slope = float("nan")
    if h and w:
        shp = (h, block, w, block)
        full = both[:h * block, :w * block].reshape(shp).all(axis=(1, 3))
        if full.sum() >= 3:
            ba = a[:h * block, :w * block].reshape(shp).mean(axis=(1, 3))[full]; ba = ba - np.median(ba)
            bb = c[:h * block, :w * block].reshape(shp).mean(axis=(1, 3))[full]; bb = bb - np.median(bb)
            rms = float(np.sqrt(np.mean((bb - ba) ** 2)))
            slope = float(ba @ bb / (ba @ ba)) if ba @ ba > 0 else float("nan")
    return d, rms, slope, spread


def support_sensitivity(scan: RockingScan, *, variants=None, base: Optional[SupportMaps] = None,
                        block: int = 8, **kwargs):
    """How far defensible support/baseline choices move the centre map, per axis.

    Default variants: support edge ``nsig`` 1.5 and 3 (default 2), padding 1 and 5 (default 3),
    ``min_out`` 3 and 12 (default 6: how many cells outside the support the median baseline
    needs). Each map is referenced to its own median over the pixels lit and uncut in the BASE
    reduction. Returns ``(rows, maps)``: per variant and axis the 8x8-block RMS and slope (the
    systematic to quote beside the split-half sigma), the pixel spread (an upper bound), and the
    difference maps ``(H, W, d)`` in the axis units.
    """
    if base is None:
        base = reduce_support(scan, split_half=False, **kwargs)
    if variants is None:
        variants = [dict(nsig=1.5), dict(nsig=3.0), dict(pad=1), dict(pad=5), dict(min_out=3),
                    dict(min_out=12)]
    sel = base.lit & ~base.truncated
    rows, maps = [], []
    for var in variants:
        m = reduce_support(scan, split_half=False, **{**kwargs, **var})
        dd = np.full(base.value.shape, np.nan)
        per = []
        for a in range(base.value.shape[-1]):
            d, rms, slope, spread = _block_stats(base.value[..., a], m.value[..., a], sel, block)
            dd[..., a] = d
            per.append(dict(axis=base.axes[a], unit=base.units[a], block_rms=rms, block_slope=slope,
                            pixel_spread=spread))
        rows.append(dict(variant=var, axes=per,
                         lit_change=float((m.lit != base.lit).mean()),
                         intensity_ratio_median=float(np.nanmedian(m.intensity[sel] / base.intensity[sel]))
                         if sel.any() else float("nan")))
        maps.append(dd)
    return rows, maps
