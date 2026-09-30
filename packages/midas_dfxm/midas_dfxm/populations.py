"""Per-pixel orientation populations in DFXM rocking curves, and a null-calibrated two-population test.

A single Gaussian per pixel is not just a simplification when a pixel's rocking curve holds two
separated populations (two orientations along the beam path, or a boundary seen in projection); its
centroid lands between them and its width reports the separation. This module decomposes each pixel's
curve into up to ``MAXP`` populations, model-free, and tests the two-population reading against
single-population nulls built from the scan's OWN per-pixel centroids, intensities, widths and noise.

Method (validated on datasetJ S168, 6-ID-C, Dec-2025; four-lens adversarial verification, claim b1daf157cb9f):

* the DECISION curve of a pixel is the 5 x 5 neighbourhood mean of the above-baseline curve; it decides
  where the populations and the valleys between them are: unimodal (isotonic up/down) regression,
  the most significant positive residual bump, recursive valley split while the bump z exceeds z*;
* z* is the 99th percentile of the first-test z on synthetic single-peaked curves of the scan's own
  single-population width (four shapes incl. an asymmetric sharp-rise/long-tail and a Lorentzian);
  calibrated twice (provisional width -> measured single-population width);
* each population's intensity, centre and rms width come from the pixel's OWN curve, the centre and
  width as moments over the population's window (the run above 2 sigma around its maximum, padded);
* significance per population: windowed own-curve intensity / (sigma_own sqrt(window length)).

Known limits (measured, S168):

* the angle-band split of the image-wide centre histogram (``band_edges``) splits even ONE smooth
  population into two bands -- a band split is NOT evidence of two populations; the per-pixel test is.
* the two-population statistic needs a SHAPE test: a fixed-band occupancy test (intensity above and
  below a cut) has no power (single-peak nulls score 0.6-0.8).
* where a second population sits in the image is brightness-confounded: dim pixels miss it. Do not
  read location statistics (distance to an edge, "high-dominated" pixels) without a matched null.
* ~12 scan points per population FWHM is at the floor for resolving curve shape.
* the 5 x 5 decision curve mixes neighbours, so false pairs concentrate within ~2 px of a SHARP orientation step
  across the edge (+-100 mdeg step: 7-16 % of pixels within 2.5 px flagged; 0.1-0.4 % of a whole phantom image;
  4-px stripes 4.6 %): a pair next to a sharp boundary is not evidence of two populations along the beam.
* a common per-frame flux jitter (this module does not normalise flux) creates false pairs: on single-population
  phantoms the false-pair rate rose from ~1e-5 (none) to a median ~0.3 % (mean 1.1 %, range 4e-5 - 12 %, 40 draws)
  at 2.5 % jitter, median 0.7 % at 3.5 %, 4 % at 5 % (heavy-tailed; 10 draws each). Do not read a pair fraction below ~10 % without the scan's flux stability.
* the separation between the populations is detector-dependent (S168: 157 / 179 / 197 mdeg median with
  three detectors): quote it with the method.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

from .support import _noise

# numba's default OpenMP threading layer and torch's libomp in one process segfault (macOS, S168);
# the workqueue layer is always available. Only a default: an explicit user setting wins.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
from numba import njit, prange  # noqa: E402

MAXP = 4

__all__ = ["MAXP", "decide", "measure", "calibrate_zstar", "null_curves", "robust_sigma",
           "pixel_populations", "confirmed_populations", "band_edges", "two_population", "single_population_null",
           "separation_bootstrap"]


# ----------------------------------------------------------------------------------------- kernels
@njit(cache=True)
def _pava_inc(y):
    n = y.shape[0]
    val = np.empty(n); wt = np.empty(n); cnt = np.empty(n, np.int64)
    k = 0
    for i in range(n):
        val[k] = y[i]; wt[k] = 1.0; cnt[k] = 1
        while k > 0 and val[k - 1] > val[k]:
            w = wt[k - 1] + wt[k]
            val[k - 1] = (val[k - 1] * wt[k - 1] + val[k] * wt[k]) / w
            wt[k - 1] = w; cnt[k - 1] += cnt[k]
            k -= 1
        k += 1
    out = np.empty(n); j = 0
    for b in range(k):
        for _ in range(cnt[b]):
            out[j] = val[b]; j += 1
    return out


@njit(cache=True)
def _unimodal(y, search):
    n = y.shape[0]
    sm = np.empty(n)
    for i in range(n):
        lo = max(i - 1, 0); hi = min(i + 1, n - 1)
        sm[i] = y[lo:hi + 1].mean()
    k0 = int(np.argmax(sm))
    best = 1e300; bu = np.zeros(n); bm = k0
    for m in range(max(k0 - search, 0), min(k0 + search, n - 1) + 1):
        up = _pava_inc(y[:m + 1])
        rev = y[m:][::-1].copy()
        dn = _pava_inc(rev)[::-1]
        u = np.empty(n)
        u[:m] = up[:m]
        u[m] = max(up[m], dn[0])
        u[m + 1:] = dn[1:]
        r = 0.0
        for i in range(n):
            r += (y[i] - u[i]) ** 2
        if r < best:
            best = r; bu = u; bm = m
    return bu, bm


@njit(cache=True)
def _best_bump(y, u, sig):
    n = y.shape[0]
    bz = 0.0; ba = -1; bb = -1
    i = 0
    while i < n:
        if y[i] - u[i] > 0:
            j = i; s = 0.0
            while j < n and y[j] - u[j] > 0:
                s += y[j] - u[j]; j += 1
            z = s / (sig * np.sqrt(j - i))
            if z > bz:
                bz = z; ba = i; bb = j - 1
            i = j
        else:
            i += 1
    return bz, ba, bb


@njit(cache=True)
def _decide_one(y, sig, zstar, search):
    """Recursive valley split of one curve. Returns (n, starts[MAXP], ends[MAXP], first_z)."""
    n = y.shape[0]
    st = np.full(MAXP, -1, np.int64); en = np.full(MAXP, -1, np.int64)
    qa = np.empty(MAXP * 4, np.int64); qb = np.empty(MAXP * 4, np.int64)
    nq = 1; qa[0] = 0; qb[0] = n - 1
    nout = 0; first_z = 0.0; first = True
    sm = np.empty(n)
    for i in range(n):
        lo = max(i - 1, 0); hi = min(i + 1, n - 1)
        sm[i] = y[lo:hi + 1].mean()
    while nq > 0:
        nq -= 1
        a = qa[nq]; b = qb[nq]
        split = False
        if b - a + 1 >= 7 and nout + nq + 2 <= MAXP:
            seg = y[a:b + 1]
            u, m = _unimodal(seg, search)
            z, ra, rb = _best_bump(seg, u, sig)
            if first:
                first_z = z; first = False
            if z > zstar:
                pk = ra + int(np.argmax(seg[ra:rb + 1]))
                lo = min(m, pk); hi = max(m, pk)
                if hi - lo >= 2:
                    v = lo + 1 + int(np.argmin(sm[a + lo + 1:a + hi]))
                    qa[nq] = a; qb[nq] = a + v - 1; nq += 1
                    qa[nq] = a + v; qb[nq] = b; nq += 1
                    split = True
        elif first:
            first = False
        if not split:
            st[nout] = a; en[nout] = b; nout += 1
    # segments leave the stack in LIFO order: sort by angle so population index = angle order
    o = np.argsort(st[:nout])
    st2 = np.full(MAXP, -1, np.int64); en2 = np.full(MAXP, -1, np.int64)
    for i in range(nout):
        st2[i] = st[o[i]]; en2[i] = en[o[i]]
    return nout, st2, en2, first_z


@njit(parallel=True, cache=True)
def decide(Yd, sig, zstar, search=6):
    """Segment each decision curve (rows of ``Yd``, noise ``sig`` per row) into <= MAXP populations.
    Returns (nseg, starts, ends, first_z); starts/ends are frame indices, -1 = unused."""
    N = Yd.shape[0]
    nseg = np.zeros(N, np.int64); S = np.full((N, MAXP), -1, np.int64); E = np.full((N, MAXP), -1, np.int64)
    fz = np.zeros(N)
    for i in prange(N):
        if sig[i] > 0 and np.isfinite(sig[i]):
            nn, st, en, z = _decide_one(Yd[i], sig[i], zstar, search)
            nseg[i] = nn; S[i] = st; E[i] = en; fz[i] = z
    return nseg, S, E, fz


def _windows(Yd, sig, S, E, nsig=2.0, pad=3):
    """Per population, the contiguous run around its maximum (3-pt smoothed decision curve) above
    nsig x the smoothed noise, padded, clipped to the population's segment."""
    N, M = Yd.shape
    sm = Yd.copy(); sm[:, 1:-1] = (Yd[:, :-2] + Yd[:, 1:-1] + Yd[:, 2:]) / 3.0
    thr = nsig * sig / np.sqrt(3.0)
    idx = np.arange(M)[None, :]
    A = np.full(S.shape, -1, np.int64); B = np.full(S.shape, -1, np.int64)
    for p in range(S.shape[1]):
        a = S[:, p]; b = E[:, p]; ok = a >= 0
        inseg = (idx >= a[:, None]) & (idx <= b[:, None]) & ok[:, None]
        v = np.where(inseg, sm, -np.inf)
        k = np.argmax(v, 1)
        below = (sm < thr[:, None]) | ~inseg
        left = np.where(below & (idx < k[:, None]), idx, -1).max(1) + 1
        right = np.where(below & (idx > k[:, None]), idx, M).min(1) - 1
        A[:, p] = np.where(ok, np.maximum(left - pad, a), -1)
        B[:, p] = np.where(ok, np.minimum(right + pad, b), -1)
    return A, B


def measure(Yo, x, nseg, S, E, Yd=None, sig=None):
    """Per population from the pixel's own curve ``Yo``: intensity (sum over the segment), centre and rms
    width (moments over the population's window; a moment over a whole segment is dominated by far-frame
    noise). Returns (I, C, W), each (N, MAXP), NaN where absent."""
    N, M = Yo.shape
    Aw, Bw = (S, E) if Yd is None else _windows(Yd, sig, S, E)
    I = np.full((N, MAXP), np.nan); C = np.full((N, MAXP), np.nan); Wd = np.full((N, MAXP), np.nan)
    idx = np.arange(M)[None, :]
    for p in range(MAXP):
        ok = S[:, p] >= 0
        seg = (idx >= S[:, p][:, None]) & (idx <= E[:, p][:, None]) & ok[:, None]
        win = (idx >= Aw[:, p][:, None]) & (idx <= Bw[:, p][:, None]) & ok[:, None]
        tot = np.where(seg, Yo, 0.0).sum(1)
        s = np.where(win, Yo, 0.0); sp = np.clip(s, 0, None)
        ts = s.sum(1); tp = sp.sum(1)
        with np.errstate(all="ignore"):
            c = (s * x[None]).sum(1) / np.where(ts > 0, ts, np.nan)
            cp = (sp * x[None]).sum(1) / np.where(tp > 0, tp, np.nan)
            w = np.sqrt((sp * (x[None] - cp[:, None]) ** 2).sum(1) / np.where(tp > 0, tp, np.nan))
        lo = x[np.clip(Aw[:, p], 0, M - 1)]; hi = x[np.clip(Bw[:, p], 0, M - 1)]
        valid = ok & (c >= lo) & (c <= hi)
        I[:, p] = np.where(ok, tot, np.nan)
        C[:, p] = np.where(valid, c, np.nan)
        Wd[:, p] = np.where(valid, w, np.nan)
    return I, C, Wd


def null_curves(x, fwhm, n=4000, seed=0, amp=(10, 300)):
    """Single-peaked curves in units of sigma (= 1): widths 0.5-2 x ``fwhm``, four shapes (Gaussian,
    broad Gaussian, sharp-rise/long-tail, Lorentzian), centre in the middle 60 % of the scan."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); L = x[-1] - x[0]
    out = np.empty((n, len(x)))
    for k in range(n):
        c = x[0] + rng.uniform(0.2, 0.8) * L
        w = fwhm * rng.uniform(0.5, 2.0) / 2.3548
        A = rng.uniform(*amp); t = x - c; shape = k % 4
        if shape == 0:
            f = np.exp(-0.5 * (t / w) ** 2)
        elif shape == 1:
            f = np.exp(-0.5 * (t / (2.5 * w)) ** 2)
        elif shape == 2:
            f = np.where(t < 0, np.exp(-0.5 * (t / (0.5 * w)) ** 2), np.exp(-t / (2.0 * w)))
        else:
            f = 1.0 / (1 + (t / w) ** 2)
        out[k] = A * f + rng.normal(0, 1, len(x))
    return out


def calibrate_zstar(x, fwhm, q=99.0, seed=0):
    """z* = the q-th percentile of the first-test bump z on single-peaked null curves (1 % false split
    per curve at q = 99). Returns (z*, the null z values)."""
    Y = null_curves(x, fwhm, seed=seed)
    _, _, _, fz = decide(Y, np.ones(len(Y)), 1e9)
    return float(np.percentile(fz, q)), fz


def robust_sigma(Y):
    """Per-curve noise (rows of ``Y``) from second differences along the angle axis, MAD-scaled:
    ``midas_dfxm.support._noise`` (the estimator the support reducer uses)."""
    Y = np.asarray(Y, float)
    return _noise(Y.T, (Y.shape[1],))


# ----------------------------------------------------------------------------------------- per pixel
def pixel_populations(frames, x, baseline, lit, sigma_own, box=5, fwhm0=0.035):
    """Populations of every lit pixel.

    frames : (M, H, W) sorted by angle ``x`` (M,); baseline (H, W); lit (H, W) bool;
    sigma_own : (n_lit,) per-pixel noise of the (repeat-averaged) frames.
    Returns dict: n, S, E (segments), I, C, W (per population, (n_lit, MAXP)), snr (population z),
    zstar, fwhm_single (median FWHM of one-population pixels, deg), zstar_pass1.
    """
    x = np.asarray(x, float)
    Fc = (np.asarray(frames, np.float32) - np.asarray(baseline, np.float32)[None])
    Yo = Fc[:, lit].T.astype(np.float64)
    Yd = ndimage.uniform_filter(Fc, size=(1, box, box), mode="nearest")[:, lit].T.astype(np.float64)
    sig_d = robust_sigma(Yd)
    z1, _ = calibrate_zstar(x, fwhm0)
    n1, S1, E1, _ = decide(Yd, sig_d, z1)
    _, _, W1 = measure(Yo, x, n1, S1, E1, Yd, sig_d)
    single = (n1 == 1) & np.isfinite(W1[:, 0])
    fw = float(np.median(2.3548 * W1[single, 0])) if single.any() else fwhm0
    z2, _ = calibrate_zstar(x, fw)
    n, S, E, _ = decide(Yd, sig_d, z2)
    I, C, W = measure(Yo, x, n, S, E, Yd, sig_d)
    A, B = _windows(Yd, sig_d, S, E)
    nwin = np.where(S >= 0, B - A + 1, 0).astype(float)
    sig_own = np.asarray(sigma_own, float)
    with np.errstate(all="ignore"):
        snr = np.where(S >= 0, I / (sig_own[:, None] * np.sqrt(np.maximum(nwin, 1))), np.nan)
        # peak SNR of the pixel's OWN curve: max of its 3-point-smoothed above-baseline curve / its noise
        peak_snr = ndimage.uniform_filter1d(Yo, 3, axis=1, mode="nearest").max(1) / sig_own
    return dict(n=n, S=S, E=E, I=I, C=C, W=W, snr=snr, peak_snr=peak_snr, zstar=z2, fwhm_single=fw, zstar_pass1=z1)


def confirmed_populations(P, snr_min=5.0, zmin=5.0):
    """(N, MAXP) bool: population is significant (z >= zmin) AND its pixel's peak SNR >= snr_min.

    Noise makes a LONE population where there is none: on real signal-free S168 pixels (467,241 px, 700 x 700
    window away from the grain) a population with z >= 5 appears in 1.7 % (peak SNR 1-1.5), 8.4 % (1.5-2),
    22.8 % (2-3) and 47.8 % (3-5) of pixels; two or more populations appear in 0.00 / 0.00 / 0.03 / 0.23 %.
    A lone population in a pixel with peak SNR < ~5 is therefore unconfirmed; pairs need no such gate
    (false pair rate ~3e-5: 160 of 5,858,557 single-population phantom pixels, at most 6.7e-5 in any SNR bin,
    spatially clustered; 14 of 467,241 real signal-free pixels). Source: PREREGISTER_S168_
    populations_headtohead.md, stages A / A' / A''."""
    ok = np.isfinite(P["C"]) & (np.nan_to_num(P["snr"]) >= zmin)
    return ok & (np.asarray(P["peak_snr"]) >= snr_min)[:, None]


def band_edges(C, I, x, smooth_bins=3, valid=None):
    """Minima of the smoothed, intensity-weighted histogram of population centres between its major
    maxima. DESCRIPTIVE ONLY: a single smoothly varying population also splits into bands here."""
    x = np.asarray(x, float)
    step = float(np.median(np.diff(x)))
    edges = np.append(x - step / 2, x[-1] + step / 2)
    ok = np.isfinite(C) & np.isfinite(I) & (I > 0)
    h, _ = np.histogram(C[ok], bins=edges, weights=I[ok])
    h = h.astype(float)
    if valid is not None:
        # an excluded frame leaves an empty bin at its angle, where a valley finder would put the edge
        v = np.asarray(valid, bool)
        if (~v).any() and v.sum() >= 2:
            h[~v] = np.interp(np.flatnonzero(~v), np.flatnonzero(v), h[v])
    hs = ndimage.uniform_filter1d(h, smooth_bins, mode="nearest")
    mx = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.05 * hs.max()]
    keep = []
    for i in mx:
        if keep and hs[keep[-1]:i + 1].min() > 0.6 * min(hs[keep[-1]], hs[i]):
            if hs[i] > hs[keep[-1]]:
                keep[-1] = i
            continue
        keep.append(i)
    cuts = [float(x[keep[j] + int(np.argmin(hs[keep[j]:keep[j + 1] + 1]))]) for j in range(len(keep) - 1)]
    return cuts, x[keep], hs


def two_population(P, edge, half_width=0.015, zmin=5.0, weights=None):
    """Pixels holding a significant population below ``edge - half_width`` AND one above
    ``edge + half_width`` (both z >= zmin). T = weighted fraction of such pixels (default weights: the
    pixel's total population intensity). Also per pixel: low/high intensity, intensity-weighted centres,
    and the high share I_high / (I_low + I_high)."""
    C, I, z = P["C"], P["I"], np.nan_to_num(P["snr"])
    ok = np.isfinite(C) & np.isfinite(I) & (I > 0) & (z >= zmin)
    lo = ok & (C < edge - half_width); hi = ok & (C > edge + half_width)
    Ilo = np.where(lo, I, 0).sum(1); Ihi = np.where(hi, I, 0).sum(1)
    with np.errstate(all="ignore"):
        Clo = np.where(lo, I * np.nan_to_num(C), 0).sum(1) / Ilo
        Chi = np.where(hi, I * np.nan_to_num(C), 0).sum(1) / Ihi
        share = Ihi / (Ilo + Ihi)
    both = lo.any(1) & hi.any(1)
    w = np.where(np.isfinite(I) & (I > 0), I, 0).sum(1) if weights is None else np.asarray(weights, float)
    T = float(w[both].sum() / max(w.sum(), 1e-300))
    hc, hI = C[hi], I[hi]
    high_centre = float((hc * hI).sum() / hI.sum()) if hI.size else float("nan")
    return dict(T=T, both=both, I_low=Ilo, I_high=Ihi, C_low=Clo, C_high=Chi, share_high=share,
                high_centre=high_centre, n_both=int(both.sum()))


def single_population_null(frames, x, baseline, lit, sigma_frame, fwhm, *, kind="gauss", seed=0,
                           gain=1.2, n_repeats=2, flux=None, sigma_own=None, **two_pop_kwargs):
    """Rebuild the scan as ONE population per pixel -- the pixel's own first-moment centroid and
    integrated intensity, width ``fwhm`` ('gauss'), the pixel's own rms width ('own_width'), or an
    asymmetric sharp-rise/long-tail profile ('asym') -- with noise var = sigma_frame^2 + gain * signal /
    n_repeats (optionally multiplied frame-wise by ``flux``), run it through ``pixel_populations`` and
    ``two_population``. Returns the null's two_population dict (T should be ~0)."""
    x = np.asarray(x, float)
    Fc = (np.asarray(frames, np.float64) - np.asarray(baseline, np.float64)[None])[:, lit].T
    sp = np.asarray(sigma_frame, float)
    sp = sp[lit] if sp.ndim == 2 else sp
    w = np.where(Fc > 2 * sp[:, None], Fc, 0.0)
    Isum = np.clip(Fc.sum(1), 0, None)
    with np.errstate(all="ignore"):
        cen = (w * x).sum(1) / w.sum(1)
        rms = np.sqrt((w * (x - cen[:, None]) ** 2).sum(1) / w.sum(1))
    good = np.isfinite(cen) & (w.sum(1) > 0)
    cen = np.where(good, cen, np.median(cen[good])); rms = np.where(good & (rms > 0.004), rms, 0.004)
    u = x[None, :] - cen[:, None]
    if kind == "gauss":
        prof = np.exp(-0.5 * (u / (fwhm / 2.3548)) ** 2)
    elif kind == "own_width":
        prof = np.exp(-0.5 * (u / rms[:, None]) ** 2)
    elif kind == "asym":
        prof = np.where(u < 0, np.exp(-0.5 * (u / 0.008) ** 2), np.exp(-u / 0.020))
    else:
        raise ValueError(f"unknown null kind {kind!r}")
    prof = prof / np.maximum(prof.sum(1, keepdims=True), 1e-12)
    sig_c = prof * Isum[:, None]
    if flux is not None:
        sig_c = sig_c * np.asarray(flux, float)[None, :]
    rng = np.random.default_rng(seed)
    y = sig_c + rng.standard_normal(sig_c.shape) * np.sqrt(sp[:, None] ** 2 + gain * np.clip(sig_c, 0, None) / n_repeats)
    b = np.asarray(baseline, np.float32)
    Fn = np.repeat(b[None], len(x), 0)
    Fn[:, lit] = (b[lit][None, :] + y.T).astype(np.float32)
    P = pixel_populations(Fn, x, b, lit, sp if sigma_own is None else sigma_own)
    return two_population(P, weights=Isum, **two_pop_kwargs)


def separation_bootstrap(tp, lit, block=150, n_boot=2000, seed=0):
    """Median (C_high - C_low) over pixels holding both populations, in mdeg, with a spatial block
    bootstrap 90 % CI (blocks of ``block`` px resampled with replacement)."""
    rows, cols = np.nonzero(lit)
    both = tp["both"]
    s = 1000 * (tp["C_high"] - tp["C_low"])[both]
    bid = (rows[both] // block) * 100000 + cols[both] // block
    _, inv = np.unique(bid, return_inverse=True)
    groups = [s[inv == k] for k in range(inv.max() + 1)] if s.size else []
    rng = np.random.default_rng(seed); meds = []
    for _ in range(n_boot if groups else 0):
        pick = rng.integers(0, len(groups), len(groups))
        meds.append(np.median(np.concatenate([groups[k] for k in pick])))
    return dict(n_px=int(both.sum()), n_blocks=len(groups),
                median_mdeg=float(np.median(s)) if s.size else float("nan"),
                p5_mdeg=float(np.percentile(s, 5)) if s.size else float("nan"),
                p95_mdeg=float(np.percentile(s, 95)) if s.size else float("nan"),
                ci90_mdeg=[float(np.percentile(meds, 5)), float(np.percentile(meds, 95))] if meds else [float("nan")] * 2)
