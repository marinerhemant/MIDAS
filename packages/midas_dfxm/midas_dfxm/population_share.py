"""Threshold-free two-template population share for DFXM rocking curves.

For every lit pixel: the fraction of the pixel's rocking-curve intensity that belongs to a HIGH-angle
population versus a LOW-angle population, WITHOUT a detection threshold. ``populations.two_population``
gives a detector-based share I_high / (I_low + I_high) that is binary in a dim transition zone (a
population is either detected or absent); this estimator instead fits every pixel's curve with a
non-negative combination of unit-area Gaussian templates and reads the share off the amplitudes.

Method (datasetJ S168, 6-ID-C, Dec-2025; PREREGISTER_S168_boundary_templates.md; PROVISIONAL until verified):

* ``centre_maps``: per-pixel LOW and HIGH population centres come from ``populations.two_population``
  (C_low, C_high). Each map is smoothed by normalised Gaussian smoothing over the pixels that HAVE that
  population (``sigma_low`` px), holes are filled from a wider smoothing (``sigma_hole`` px), then from the
  nearest filled pixel. Every lit pixel therefore gets both centres.
* ``template_share``: per pixel, a bank of unit-area Gaussian templates on each side -- centres
  c + offsets, widths width_mult * fwhm / 2.3548 (15 templates per side by default). The pixel's
  above-baseline curve is fitted by non-negative least squares over the whole bank (both sides at once);
  share = sum of HIGH-side amplitudes / sum of all amplitudes, NaN where the total is 0. A bank instead of
  one template per side is what makes the estimator robust to a wrong centre or width; a single exact
  template per side gave decile medians 0.13-0.24 on the reference's uniform-0.30 phantom (not re-measured here).
* SPEED: the reference used one ``scipy.optimize.nnls`` call per pixel in a Python loop (~10 min for
  490k pixels). Here each pixel's template matrix is built inside a numba ``prange`` kernel and solved by
  Lawson-Hanson active-set NNLS (the algorithm scipy uses; passive-set least squares by ``lstsq``).
  Cyclic coordinate descent on the Gram matrix was tried first and REJECTED: the bank's condition number
  is ~1e12 (templates 10 mdeg apart, ~1/12 of a FWHM), and even 50,000 sweeps left 737 of 6000 pixels more
  than 1e-4 from scipy's share (max 0.02); Lawson-Hanson matches scipy to 1e-13.

Validation (measured when this module was written; the tests re-assert the criteria):

* equivalence with scipy nnls on 5000 random synthetic pixels: max |share difference| 1.3e-13, median
  1.7e-16 (6000 pixels: 8e-14 / 2e-16);
* full-pipeline phantoms (pixel_populations -> two_population -> centre_maps -> template_share), 160 x 160
  grain, 61 angles, single-population FWHM 0.124 deg, noise variance 16 + 0.6 signal, brightness gradient
  x40, centres LOW 15.70 / HIGH 15.89 deg + smooth random field (sd 10 mdeg), widths x U(0.8, 1.25):
  T3 uniform share 0.30: every brightness-decile median in 0.258-0.268; T1 ramp (30 deg, 40 px): recovered
  phi 30.0 deg, w 39.2 px, plateaus 0.00 / 1.00; T2 step: fitted w 1.7 px;
* timing, real S168 700 x 700 crop (h2h/ctx_cache.npz, 489,895 lit px x 61 frames, 15 templates per side,
  10 cores, Apple silicon, workqueue layer): template_share 9.7 s (centre_maps 1.1 s, pixel_populations
  22 s); the scipy loop it replaces takes ~10 min.

Known limits:

* centre maps are smoothed and hole-filled, so where a population is ABSENT its centre is an
  interpolation and the separation between the two templates there is meaningless. A share in such a
  region is the share of a template that was never observed.
* shares are biased LOW by about 0.01-0.05 on a uniform 0.30 phantom (the bank absorbs some HIGH
  intensity into LOW templates that overlap it: the two populations are ~0.19 deg apart, ~1.5 FWHM).
* the share is intensity-weighted (amplitude), not pixel-count weighted; noise-only pixels give a share
  of noise fits, so restrict to pixels with real signal before reading location statistics.
* the estimator does not know the populations are Gaussian; a strongly asymmetric population is fitted
  by several bank templates, which can leak across the LOW/HIGH split if its tail crosses the midpoint.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

# numba's default OpenMP threading layer and torch's libomp in one process segfault (macOS);
# the workqueue layer is always available. Only a default: an explicit user setting wins.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
from numba import njit, prange  # noqa: E402

__all__ = ["centre_maps", "template_share", "nnls_share"]

_FWHM_TO_SIGMA = 2.3548


# ---------------------------------------------------------------------------------------- centre maps
def _smooth_fill(v, present, lit, s_low, s_hole):
    """Normalised Gaussian smoothing (``s_low`` px) of the pixels that have a value, holes from ``s_hole``
    px, then the nearest filled value. ``v`` and ``present`` are per lit pixel; returns per lit pixel."""
    shape = lit.shape
    val = np.zeros(shape); val[lit] = np.where(present, np.nan_to_num(v), 0.0)
    w = np.zeros(shape); w[lit] = present.astype(float)
    out = None
    for s in (s_low, s_hole):
        num = ndimage.gaussian_filter(val, s)
        den = ndimage.gaussian_filter(w, s)
        est = np.where(den > 1e-3, num / np.maximum(den, 1e-12), np.nan)
        out = est if out is None else np.where(np.isfinite(out), out, est)
    bad = ~np.isfinite(out)
    if bad.any():
        if bad.all():
            raise ValueError("no lit pixel holds this population: cannot build a centre map")
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        out = out[idx[0], idx[1]]
    return out[lit]


def centre_maps(P, tp, lit, sigma_low=15.0, sigma_hole=60.0):
    """Smoothed, hole-filled LOW and HIGH population-centre maps, per lit pixel.

    P : ``populations.pixel_populations`` result (kept for the caller's provenance; the centres come from
    ``tp``); tp : ``populations.two_population`` result (uses I_low, I_high, C_low, C_high);
    lit : (H, W) bool. Returns (c_low, c_high), each (n_lit,) in the units of ``x`` (degrees).
    """
    lit = np.asarray(lit, bool)
    cl, ch = np.asarray(tp["C_low"], float), np.asarray(tp["C_high"], float)
    lo_p = (np.asarray(tp["I_low"]) > 0) & np.isfinite(cl)
    hi_p = (np.asarray(tp["I_high"]) > 0) & np.isfinite(ch)
    return (_smooth_fill(cl, lo_p, lit, sigma_low, sigma_hole),
            _smooth_fill(ch, hi_p, lit, sigma_low, sigma_hole))


# --------------------------------------------------------------------------------------- NNLS kernel
@njit(cache=True)
def _lawson_hanson(A, y, x, P, w, tol, max_outer):
    """Lawson-Hanson active-set NNLS of ``A x ~ y`` (A is (M, K)); solution left in ``x``.
    Same algorithm as ``scipy.optimize.nnls``: the passive-set least squares are solved with
    ``lstsq`` (not normal equations -- the template bank has condition number ~1e12)."""
    M, K = A.shape
    x[:] = 0.0; P[:] = False
    r = y.copy()
    for j in range(K):
        s = 0.0
        for m in range(M):
            s += A[m, j] * r[m]
        w[j] = s
    for _ in range(max_outer):
        jb = -1; wb = tol
        for j in range(K):
            if (not P[j]) and w[j] > wb:
                wb = w[j]; jb = j
        if jb < 0:
            break
        P[jb] = True
        for _inner in range(3 * K):
            npas = 0
            for j in range(K):
                if P[j]:
                    npas += 1
            Ap = np.empty((M, npas)); idx = np.empty(npas, np.int64)
            c = 0
            for j in range(K):
                if P[j]:
                    idx[c] = j
                    for m in range(M):
                        Ap[m, c] = A[m, j]
                    c += 1
            sp = np.linalg.lstsq(Ap, y)[0]
            alpha = 1e300; neg = False
            for c in range(npas):
                if sp[c] <= 0.0:
                    neg = True
                    d = x[idx[c]] - sp[c]
                    if d > 0.0:
                        al = x[idx[c]] / d
                        if al < alpha:
                            alpha = al
            if not neg:
                for j in range(K):
                    x[j] = 0.0
                for c in range(npas):
                    x[idx[c]] = sp[c]
                break
            if alpha > 1.0:
                alpha = 1.0
            for j in range(K):
                if P[j]:
                    c = 0
                    for cc in range(npas):
                        if idx[cc] == j:
                            c = cc
                    x[j] += alpha * (sp[c] - x[j])
            for j in range(K):
                if P[j] and x[j] <= tol:
                    x[j] = 0.0; P[j] = False
        for m in range(M):
            s = y[m]
            for j in range(K):
                s -= A[m, j] * x[j]
            r[m] = s
        for j in range(K):
            s = 0.0
            for m in range(M):
                s += A[m, j] * r[m]
            w[j] = s


@njit(parallel=True, cache=True)
def _bank_share(Y, x, cl, ch, offs, wmul, s0, tol):
    """Per pixel: build the 2 * len(offs) * len(wmul) unit-area Gaussian templates (columns of A; the
    first half is the LOW side), solve NNLS exactly, return HIGH amplitude / total amplitude."""
    N, M = Y.shape
    no = offs.shape[0]; nw = wmul.shape[0]
    K = 2 * no * nw
    out = np.full(N, np.nan)
    for i in prange(N):
        c0 = cl[i]; c1 = ch[i]
        if not (np.isfinite(c0) and np.isfinite(c1)):
            continue
        A = np.empty((M, K))
        k = 0
        for side in range(2):
            c = c0 if side == 0 else c1
            for a in range(no):
                for b in range(nw):
                    sg = s0 * wmul[b]; ctr = c + offs[a]
                    t = 0.0
                    for m in range(M):
                        v = np.exp(-0.5 * ((x[m] - ctr) / sg) ** 2)
                        A[m, k] = v; t += v
                    if t > 0.0:
                        for m in range(M):
                            A[m, k] /= t
                    else:
                        for m in range(M):
                            A[m, k] = 0.0
                    k += 1
        amp = np.zeros(K); P = np.zeros(K, np.bool_); w = np.empty(K)
        _lawson_hanson(A, Y[i], amp, P, w, tol, 3 * K)
        tot = 0.0; hi = 0.0
        for j in range(K):
            tot += amp[j]
            if j >= K // 2:
                hi += amp[j]
        if tot > 0.0:
            out[i] = hi / tot
    return out


def nnls_share(Y, x, c_low, c_high, fwhm, offsets=(-0.02, -0.01, 0.0, 0.01, 0.02),
               width_mult=(0.7, 1.0, 1.4), tol=None):
    """Bank-NNLS HIGH share for curves ``Y`` (n, M) already baseline-subtracted, at angles ``x`` (M,)."""
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if tol is None:                     # scipy.optimize.nnls' scale: 10 max(M, K) eps ||A||_1, unit-area columns
        tol = 10.0 * max(x.shape[0], 2 * len(offsets) * len(width_mult)) * np.finfo(float).eps
    if Y.ndim != 2 or Y.shape[1] != x.shape[0]:
        raise ValueError("Y must be (n_pixels, len(x))")
    cl = np.ascontiguousarray(c_low, dtype=np.float64); ch = np.ascontiguousarray(c_high, dtype=np.float64)
    if cl.shape != (Y.shape[0],) or ch.shape != cl.shape:
        raise ValueError("c_low and c_high must be (n_pixels,)")
    return _bank_share(Y, x, cl, ch, np.asarray(offsets, np.float64), np.asarray(width_mult, np.float64),
                       float(fwhm) / _FWHM_TO_SIGMA, float(tol))


def template_share(frames, x, baseline, lit, c_low, c_high, fwhm,
                   offsets=(-0.02, -0.01, 0.0, 0.01, 0.02), width_mult=(0.7, 1.0, 1.4)):
    """Per-lit-pixel HIGH-population share in [0, 1] (NaN where the fitted total amplitude is 0).

    frames : (M, H, W) float32 rocking-curve frames at angles x (M,) degrees; baseline : (H, W);
    lit : (H, W) bool; c_low, c_high : (n_lit,) template centres (degrees) from ``centre_maps``;
    fwhm : single-population FWHM, degrees (``pixel_populations(...)["fwhm_single"]``).
    Bank: centres c + offsets, widths width_mult * fwhm / 2.3548, on each side.
    """
    frames = np.asarray(frames)
    lit = np.asarray(lit, bool)
    if frames.ndim != 3 or frames.shape[1:] != lit.shape:
        raise ValueError("frames must be (M, H, W) matching lit (H, W)")
    Y = (frames - np.asarray(baseline, np.float32)[None])[:, lit].T
    return nnls_share(Y, x, c_low, c_high, fwhm, offsets, width_mult)
