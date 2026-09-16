"""Width of a rod TRANSVERSE to itself on the detector — the complement to
:mod:`midas_defect.rod_profile`, which measures intensity ALONG a rod.

A diffuse rod (or a Bragg node sitting on one) has some extent perpendicular to its
own direction, set by a mix of the instrument's angular resolution, the crystal's
mosaic spread, and — the physically interesting part — any lateral coherence length
of whatever feature is broadening it. This module measures that transverse width by
fitting a Gaussian directly to background-subtracted pixels, rather than by reading
it off a q-space projection.

Two design choices matter and are pinned by tests:

- **Pixels are addressed by their exact centres**, not sampled along a parametrised
  curve and rounded to the nearest pixel — that rounding creates a staircase that
  biases a narrow width. A Gaussian fitted to pixel-AREA-integrated samples returns
  ``sigma**2 + 1/12`` (the variance of a unit pixel projected onto any direction is
  1/12), not the true ``sigma**2``. Fitting the node and the feature being compared
  to it the same way makes that term cancel in an excess (``sigma_a**2 - sigma_b**2``);
  it does NOT cancel in a ratio.
- **Raw counts, not background-subtracted ones, decide censoring.** A pixel is
  dropped if its raw count exceeds a threshold in ANY frame being summed, because a
  clipped or rate-limited core reports a flat top whose width means nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "FWHM_PER_SIGMA",
    "TransverseFit",
    "rod_axes",
    "offsets",
    "collect",
    "fit_transverse",
    "interp_reference",
    "excess",
    "weighted_line_fit",
    "quadratic_term_pvalue",
    "inject_rocking",
]

FWHM_PER_SIGMA = 2.354820045


@dataclass
class TransverseFit:
    """One transverse-width measurement. ``sigma`` includes the pixel-integration
    ``1/12`` term (see module docstring); subtract it in quadrature before comparing
    to a physical width. ``secondary``/``t2`` mark a second, BIC-justified Gaussian
    fitted alongside the primary one (a neighbouring feature that would otherwise
    bias the primary width if left unmodelled)."""
    ok: bool
    reason: str = ""
    sigma: float = float("nan")          # px, includes the 1/12 pixel term
    sigma_err: float = float("nan")
    t0: float = float("nan")
    amp: float = float("nan")
    contrast: float = float("nan")
    n_pix: int = 0
    n_censored: int = 0
    secondary: bool = False
    t2: float = float("nan")


def rod_axes(tangent_rc: Tuple[float, float]):
    """Unit tangent (row, col) and the perpendicular obtained by rotating it +90 deg."""
    tr, tc = float(tangent_rc[0]), float(tangent_rc[1])
    n = math.hypot(tr, tc)
    if n == 0.0:
        raise ValueError("zero tangent")
    tr, tc = tr / n, tc / n
    return (tr, tc), (-tc, tr)


def offsets(rows, cols, r0: float, c0: float, tangent_rc) -> Tuple[np.ndarray, np.ndarray]:
    """(s, t): along-rod and perpendicular offsets of pixel centres from (r0, c0)."""
    (tr, tc), (pr, pc) = rod_axes(tangent_rc)
    dr = np.asarray(rows, float) - r0
    dc = np.asarray(cols, float) - c0
    return dr * tr + dc * tc, dr * pr + dc * pc


def collect(stack: np.ndarray, raw: np.ndarray, mask: np.ndarray, r0: float, c0: float,
            tangent_rc, k_center: int, W: int, *, s_max: float = 2.0, t_max: float = 12.0,
            censor: Optional[float] = None):
    """Pixels for one measurement: frame-summed values and their transverse (t) offsets.

    Returns ``(dict, "")`` or ``(None, reason)``. ``stack`` is background-subtracted;
    ``raw`` holds the raw counts, used only for censoring. Both are
    ``(n_frames, rows, cols)``. Frames ``k_center - W .. k_center + W`` are summed.
    """
    n_f, n_r, n_c = stack.shape
    k0, k1 = k_center - W, k_center + W
    if k0 < 0 or k1 >= n_f:
        return None, "frame window off stack edge"
    R = int(math.ceil(max(s_max, t_max))) + 2
    ri, ci = int(round(r0)), int(round(c0))
    rr, cc = np.mgrid[ri - R:ri + R + 1, ci - R:ci + R + 1]
    inside = (rr >= 0) & (rr < n_r) & (cc >= 0) & (cc < n_c)
    rr, cc = rr[inside], cc[inside]
    s, t = offsets(rr, cc, r0, c0, tangent_rc)
    sel = (np.abs(s) <= s_max) & (np.abs(t) <= t_max) & ~mask[rr, cc]
    rr, cc, t = rr[sel], cc[sel], t[sel]
    if rr.size == 0:
        return None, "no unmasked pixels in window"
    vals = stack[k0:k1 + 1, rr, cc].sum(axis=0).astype(float)
    n_cens = 0
    if censor is not None:
        cens = (raw[k0:k1 + 1, rr, cc] > censor).any(axis=0)
        n_cens = int(cens.sum())
        vals, t = vals[~cens], t[~cens]
    return dict(t=t.astype(float), v=vals, n_censored=n_cens, n_window=int(sel.sum())), ""


def _g1(t, a, t0, sg, b0, b1):
    return a * np.exp(-0.5 * ((t - t0) / sg) ** 2) + b0 + b1 * t


def _g2(t, a, t0, sg, b0, b1, a2, t2, sg2):
    return _g1(t, a, t0, sg, b0, b1) + a2 * np.exp(-0.5 * ((t - t2) / sg2) ** 2)


def _bic(rss: float, n: int, k: int) -> float:
    return n * math.log(max(rss, 1e-300) / n) + k * math.log(n)


def fit_transverse(t: np.ndarray, v: np.ndarray, *, t_max: float = 12.0, min_contrast: float = 5.0,
                   bic_margin: float = 10.0, min_side: int = 4, max_rel_err: float = 0.3,
                   n_censored: int = 0) -> TransverseFit:
    """Gaussian + linear background in ``t``, with an optional second Gaussian.

    The single-component fit is always tried first. A second Gaussian is added
    only if it improves the Bayesian Information Criterion by more than
    ``bic_margin`` over the first — the margin exists so a marginal, noise-driven
    improvement cannot silently change what "the width" means. The result is
    refused (``ok=False``) rather than trusted when: the contrast (amplitude over
    the noise estimated from ``|t|>=8``) is below ``min_contrast``; a fitted second
    component sits within 3 px of the primary (unresolved crowding, not two
    separable features); either side of the peak has fewer than ``min_side`` pixels
    within 3 sigma; or the fitted sigma's own relative error exceeds ``max_rel_err``.
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    n = int(t.size)
    if n < 20:
        return TransverseFit(False, f"{n} pixels (<20)", n_pix=n, n_censored=n_censored)
    edge = np.abs(t) >= 8.0
    if edge.sum() < 6:
        return TransverseFit(False, "too few |t|>=8 pixels for a noise estimate", n_pix=n,
                             n_censored=n_censored)
    b_edge = float(np.median(v[edge]))
    noise = float(1.4826 * np.median(np.abs(v[edge] - b_edge)))
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(v[edge])) or 1.0
    core = np.abs(t) <= 3.0
    a0 = float(np.max(v[core]) - b_edge) if core.any() else float(np.max(v) - b_edge)
    a0 = max(a0, noise)

    lb = [0.0, -2.0, 0.2, -np.inf, -np.inf]
    ub = [np.inf, 2.0, 8.0, np.inf, np.inf]
    try:
        p1, c1 = curve_fit(_g1, t, v, p0=[a0, 0.0, 1.5, b_edge, 0.0], bounds=(lb, ub), maxfev=20000)
    except Exception as exc:                                    # noqa: BLE001 - reported, not hidden
        return TransverseFit(False, f"fit failed: {exc}", n_pix=n, n_censored=n_censored)
    rss1 = float(np.sum((v - _g1(t, *p1)) ** 2))
    p, c, sec = p1, c1, False

    resid = v - _g1(t, *p1)
    far = np.abs(t - p1[1]) >= 2.0
    if far.any():
        j = int(np.argmax(np.where(far, resid, -np.inf)))
        seed1 = [p1[0], float(np.clip(p1[1], -1.99, 1.99)), float(np.clip(p1[2], 0.21, 7.99)), p1[3], p1[4]]
        p0b = seed1 + [max(float(resid[j]), noise), float(np.clip(t[j], -t_max + 0.01, t_max - 0.01)), 1.5]
        try:
            p2, c2 = curve_fit(_g2, t, v, p0=p0b, bounds=(lb + [0.0, -t_max, 0.2], ub + [np.inf, t_max, 8.0]),
                               maxfev=20000)
            rss2 = float(np.sum((v - _g2(t, *p2)) ** 2))
            if _bic(rss2, n, 8) < _bic(rss1, n, 5) - bic_margin:
                p, c, sec = p2, c2, True
        except Exception:                                       # noqa: BLE001 - a failed 2-comp fit keeps 1-comp
            pass

    a, t0, sg = float(p[0]), float(p[1]), abs(float(p[2]))
    err = np.sqrt(np.diag(c)) if np.all(np.isfinite(c)) else np.full(len(p), np.nan)
    fit = TransverseFit(True, "", sigma=sg, sigma_err=float(err[2]), t0=t0, amp=a, contrast=a / noise,
                        n_pix=n, n_censored=n_censored, secondary=sec,
                        t2=float(p[6]) if sec else float("nan"))
    if fit.contrast < min_contrast:
        fit.ok, fit.reason = False, f"contrast {fit.contrast:.1f} < {min_contrast}"
    elif sec and abs(fit.t2 - t0) < 3.0:
        fit.ok, fit.reason = False, "unresolved crowding (second component within 3 px)"
    else:
        left = int(np.sum((t < t0) & (t >= t0 - 3 * sg)))
        right = int(np.sum((t > t0) & (t <= t0 + 3 * sg)))
        if left < min_side or right < min_side:
            fit.ok, fit.reason = False, f"one-sided coverage ({left} left, {right} right)"
        elif not np.isfinite(fit.sigma_err) or fit.sigma_err / sg > max_rel_err:
            fit.ok, fit.reason = False, "sigma poorly constrained"
    return fit


def interp_reference(sig_lo: float, err_lo: float, r_lo: float, sig_hi: float, err_hi: float,
                     r_hi: float, r_mid: float):
    """``sigma_ref**2`` linear in detector radius between two flanking reference
    measurements, with its propagated error. Returns ``(var, var_err, weight)``."""
    w = 0.5 if r_hi == r_lo else (r_mid - r_lo) / (r_hi - r_lo)
    var = (1 - w) * sig_lo ** 2 + w * sig_hi ** 2
    var_err = math.hypot((1 - w) * 2 * sig_lo * err_lo, w * 2 * sig_hi * err_hi)
    return var, var_err, w


def excess(sig_gap: float, err_gap: float, ref_var: float, ref_var_err: float):
    """``excess**2``, its error, and the excess FWHM (0 when ``excess**2 <= 0``) of a
    measured sigma over a reference variance -- e.g. a diffuse feature's width over a
    matched instrumental/resolution reference."""
    ex2 = sig_gap ** 2 - ref_var
    ex2_err = math.hypot(2 * sig_gap * err_gap, ref_var_err)
    fwhm = FWHM_PER_SIGMA * math.sqrt(ex2) if ex2 > 0 else 0.0
    return ex2, ex2_err, fwhm


def weighted_line_fit(x, y, var, *, quadratic: bool = False, sys_rel: float = 0.05):
    """Weighted least squares ``y = a + b*x (+ c*x**2)`` -- e.g. width-squared vs |q|.

    ``var`` is each point's variance of y; ``sys_rel*|y|`` is added in quadrature so
    one very precise point cannot pin the line by itself. ``x`` is rescaled
    internally (avoids an ill-conditioned ``x**2`` when x spans several orders of
    magnitude) and parameters are returned in the original units. The covariance is
    scaled by ``max(1, chi2_red)``, so an over-dispersed fit does not under-report
    its own parameter uncertainty.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.asarray(var, float) + (sys_rel * np.abs(y)) ** 2
    scale = float(np.max(np.abs(x))) or 1.0
    xs = x / scale
    cols = [np.ones_like(xs), xs] + ([xs * xs] if quadratic else [])
    X = np.column_stack(cols)
    w = 1.0 / v
    cov_s = np.linalg.inv((X.T * w) @ X)
    p_s = cov_s @ ((X.T * w) @ y)
    r = y - X @ p_s
    chi2 = float(np.sum(w * r * r))
    dof = int(len(y) - X.shape[1])
    chi2_red = chi2 / dof if dof > 0 else float("nan")
    s = max(1.0, chi2_red) if np.isfinite(chi2_red) else 1.0
    unit = np.array([1.0, 1.0 / scale] + ([1.0 / scale ** 2] if quadratic else []))
    return dict(params=p_s * unit, cov=cov_s * s * np.outer(unit, unit), chi2=chi2, dof=dof,
                chi2_red=chi2_red, n=int(len(y)))


def quadratic_term_pvalue(x, y, var, *, sys_rel: float = 0.05) -> float:
    """F-test p-value for adding an ``x**2`` term to :func:`weighted_line_fit`
    (``nan`` if too few points to test)."""
    from scipy import stats as _stats
    lin = weighted_line_fit(x, y, var, sys_rel=sys_rel)
    quad = weighted_line_fit(x, y, var, quadratic=True, sys_rel=sys_rel)
    if quad["dof"] <= 0 or quad["chi2"] <= 0:
        return float("nan")
    F = (lin["chi2"] - quad["chi2"]) / (quad["chi2"] / quad["dof"])
    return float(_stats.f.sf(max(F, 0.0), 1, quad["dof"]))


def _render_patch(r0: float, c0: float, tangent_rc, sigma_t: float, amp: float, sigma_s: float, *,
                  R: int = 20, supersample: int = 10, t_shift: float = 0.0):
    """Pixel-area-integrated ``amp * exp(-(t-t_shift)**2/2sigma_t**2) * exp(-s**2/2sigma_s**2)``.

    Pixel (i, j) covers ``[i-0.5, i+0.5] x [j-0.5, j+0.5]``, matching the
    pixel-centre convention of :func:`offsets`. Returns ``(rows, cols, values)``,
    2-D arrays. Internal to :func:`inject_rocking`.
    """
    ri, ci = int(round(r0)), int(round(c0))
    RR, CC = np.meshgrid(np.arange(ri - R, ri + R + 1), np.arange(ci - R, ci + R + 1), indexing="ij")
    sub = (np.arange(supersample) + 0.5) / supersample - 0.5
    acc = np.zeros(RR.shape)
    for du in sub:
        for dv in sub:
            s, t = offsets(RR + du, CC + dv, r0, c0, tangent_rc)
            acc += np.exp(-0.5 * ((t - t_shift) / sigma_t) ** 2 - 0.5 * (s / sigma_s) ** 2)
    return RR, CC, acc * (amp / supersample ** 2)


def inject_rocking(stack: np.ndarray, raw: np.ndarray, r0: float, c0: float, tangent_rc, k_center: int,
                   sigma_t: float, summed_peak: float, sigma_s: float, *,
                   weights: Sequence[float] = (0.05, 0.25, 0.40, 0.25, 0.05),
                   motion_px_per_frame: float = 1.2, rng: Optional[np.random.Generator] = None,
                   poisson: bool = True) -> None:
    """Add a rocking, moving, pixel-integrated profile to BOTH ``stack`` and ``raw``,
    in place -- for synthetic recovery tests and for demonstrating what a planted
    feature looks like to :func:`collect`/:func:`fit_transverse`.

    Frame ``k_center + o`` gets weight ``weights[o + len(weights)//2]`` of
    ``summed_peak`` and is shifted by ``motion_px_per_frame * o`` in ``t`` --
    a rod segment rocks through several frames while also sweeping across the
    detector, unlike a stationary Bragg node. Poisson noise (normal approximation)
    is added when ``poisson=True``.
    """
    rng = rng or np.random.default_rng(0)
    half = len(weights) // 2
    n_f, n_r, n_c = stack.shape
    for o, w in zip(range(-half, half + 1), weights):
        k = k_center + o
        if not (0 <= k < n_f) or w == 0:
            continue
        RR, CC, patch = _render_patch(r0, c0, tangent_rc, sigma_t, summed_peak * w, sigma_s,
                                      t_shift=motion_px_per_frame * o)
        ok = (RR >= 0) & (RR < n_r) & (CC >= 0) & (CC < n_c)
        add = patch[ok]
        if poisson:
            add = add + rng.normal(0.0, np.sqrt(np.clip(add, 0, None)))
        stack[k, RR[ok], CC[ok]] += add
        raw[k, RR[ok], CC[ok]] += add
