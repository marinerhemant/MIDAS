"""Fixed-support integration of rotation-series spots.

Why not the blob label sum
--------------------------
``midas_defect.ingest.find_blobs_3d`` reports ``integrated``, the sum over the voxels that exceed an
ABSOLUTE threshold. For two identically shaped spots whose amplitudes differ by a factor k that sum
differs by k**|s| with |s| between 1 and 5, because the threshold sits at a different place on each
profile (known-limits KL-106). A weak reflection is therefore under-counted by more than its amplitude
ratio, which is exactly a fake high-angle fall-off and is read as a Debye-Waller factor. This module is
the replacement:

* the SUPPORT is the same rule for every reflection: a box of ``k`` profile sigmas around the spot
  centre, with the ω half-width scaled by the reflection's own Lorentz factor (the rocking width grows
  with L) and the in-plane half-width fixed;
* the profile sigmas are MEASURED from the strong spots (:func:`fit_profile_model`), never tuned on
  the weak ones;
* ``k`` is swept to convergence on the strong spots (:func:`choose_support_scale`);
* the background is estimated OUTSIDE the box, on the raw counts, from a shell, with a robust
  outlier cut (a percentile trim would bias a Poisson mean low and, multiplied by the box volume, cost a
  weak spot tens of percent).

The sum is over raw counts minus the shell background, so the variance is honest Poisson:
``var(I) = sum(raw in box) + n_box**2 * bg / n_shell``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import poisson

__all__ = ["ProfileModel", "IntegrationResult", "integrate_box", "BACKGROUNDS", "DEFAULT_BACKGROUND", "fit_profile_model",
           "choose_support_scale", "measure_extents", "choose_margin", "build_profile"]


BACKGROUNDS = ("mean", "quadratic")
DEFAULT_BACKGROUND = "mean"       # PREREGISTER A36: flips to "quadratic" only if T1, T2 and T3 are all met


@dataclass(frozen=True)
class ProfileModel:
    """Rule for the support of a reflection with Lorentz factor L.

    Two rules live here. The v1 rule assumes a Gaussian: half-width ``k`` sigmas with ``sigma_frames = sf0 + sf1 L``.
    The v2 rule (PREREGISTER A11) is MEASURED: the half-extent holding 99.5 % of a strong spot's signal, regressed on L
    (``ext_f0 + ext_f1 L`` frames, ``ext_p`` pixels), plus an integer ``margin``. v2 is used whenever ``ext_f0`` is set.
    v1 under-covers high-L spots, whose omega profile is the flat-topped shell crossing (width proportional to L) and not a
    Gaussian: measured losses of 5 to 17 % at L = 11 to 62 on noise-free frames.
    """
    sigma_px: float            # in-plane sigma, pixels (v1)
    sf0: float                 # sigma_frames = sf0 + sf1 * L (v1)
    sf1: float
    k: float = 3.5             # half-width in sigmas (v1)
    shell_px: int = 3          # background shell thickness, pixels
    shell_frames: int = 2      # and frames
    ext_f0: Optional[float] = None     # v2: half-extent in frames = ext_f0 + ext_f1 * L
    ext_f1: float = 0.0
    ext_p: Optional[float] = None      # v2: half-extent in pixels
    margin: int = 0                    # v2: extra frames and pixels added to both
    background: str = DEFAULT_BACKGROUND   # "mean" (shell mean, assumes a flat background) or "quadratic" (polynomial fitted to the shell, summed over the box)

    def half_widths(self, L: float) -> Tuple[int, int]:
        """(frames, pixels) half-widths of the support for a reflection with Lorentz factor L."""
        if self.ext_f0 is not None:
            hf = int(np.ceil(self.ext_f0 + self.ext_f1 * float(L))) + self.margin
            hp = int(np.ceil(self.ext_p)) + self.margin
            return (max(1, hf), max(2, hp))
        sf = self.sf0 + self.sf1 * float(L)
        return (max(1, int(np.ceil(self.k * sf))), max(2, int(np.ceil(self.k * self.sigma_px))))

    def with_k(self, k: float) -> "ProfileModel":
        return replace(self, k=float(k))

    def with_margin(self, m: int) -> "ProfileModel":
        return replace(self, margin=int(m))


@dataclass
class IntegrationResult:
    intensity: float           # counts, background removed
    sigma: float               # counts, Poisson
    background: float          # counts per voxel
    n_box: int                 # voxels in the support
    n_shell: int
    peak: float                # max raw voxel in the support
    clipped_frames: bool       # support extends beyond the scan
    clipped_detector: bool     # support extends beyond the detector
    n_masked: int              # masked voxels inside the support
    ok: bool                   # background estimable and support non-empty
    background_kind: str = "mean"      # estimator actually used (a "quadratic" request falls back to "mean" when the shell cannot support the fit)


def _robust_mean(v: np.ndarray) -> float:
    """Mean of ``v`` after removing high outliers (neighbouring spots): > median + 4.5 sqrt(median+1)."""
    if v.size == 0:
        return float("nan")
    med = float(np.median(v))
    keep = v <= med + 4.5 * np.sqrt(med + 1.0)
    return float(v[keep].mean())


_N_TERMS = 7                      # 1, f, r, c, r^2, c^2, r c
_MIN_SHELL_PER_TERM = 5
_CLIP_TAIL = 3.17e-5               # one-sided Poisson tail mass of the clip (= 4 Gaussian sigma): voxels above the (1 - tail) quantile of Poisson(fit) are dropped
_PASSES = 4                       # one unweighted fit, then three re-weighted fits with one-sided clipping


def _basis(F: np.ndarray, R: np.ndarray, C: np.ndarray, centre: Tuple[int, int, int], half: Tuple[int, int]) -> np.ndarray:
    """(n, 7) design matrix: quadratic in the detector plane, linear in the frame index; coordinates are offsets from the box centre in half-widths."""
    f = (F - centre[0]) / max(half[0], 1)
    r = (R - centre[1]) / max(half[1], 1)
    c = (C - centre[2]) / max(half[1], 1)
    return np.stack([np.ones_like(r), f, r, c, r * r, c * c, r * c], axis=-1)


def _fit_shell(shell_xyz: Tuple[np.ndarray, np.ndarray, np.ndarray], y: np.ndarray, box_xyz: Tuple[np.ndarray, np.ndarray, np.ndarray],
               centre: Tuple[int, int, int], half: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """Polynomial fitted to the shell voxels (Poisson weights 1/fit, iterated; shell voxels above the 4-sigma-equivalent Poisson quantile dropped, so a neighbouring spot is
    kept out), summed over the box voxels. Returns ``(background counts in the box, variance of that sum)`` or None when the shell cannot support the fit
    (fewer than 5 voxels per term survive, or a singular normal matrix)."""
    X = _basis(*shell_xyz, centre, half)
    Xb = _basis(*box_xyz, centre, half).sum(axis=0)
    keep = np.ones(y.size, bool)
    w = np.ones(y.size)
    for _ in range(_PASSES):
        if keep.sum() < _MIN_SHELL_PER_TERM * _N_TERMS:
            return None
        Xk, wk = X[keep], w[keep]
        A = Xk.T @ (wk[:, None] * Xk)
        if not np.isfinite(np.linalg.cond(A)) or np.linalg.cond(A) > 1e12:
            return None
        beta = np.linalg.solve(A, Xk.T @ (wk * y[keep]))
        fit = X @ beta
        w = 1.0 / np.maximum(fit, 1.0)             # NOT 1/y: that weights low draws up and biases the fit by -1 count per voxel
        keep = y <= poisson.isf(_CLIP_TAIL, np.maximum(fit, 1e-3))      # the Poisson quantile, not fit + 4 sqrt(fit): at 1 count/voxel the Gaussian rule clips 6e-4 of the voxels and biases the background low
    if keep.sum() < _MIN_SHELL_PER_TERM * _N_TERMS:
        return None
    Xk, wk = X[keep], w[keep]
    Ainv = np.linalg.inv(Xk.T @ (wk[:, None] * Xk))
    total, var = float(Xb @ beta), float(Xb @ Ainv @ Xb)
    return (total, var) if np.isfinite(total) and np.isfinite(var) and var >= 0 else None


def integrate_box(raw: np.ndarray, centre: Tuple[float, float, float], half: Tuple[int, int],
                  *, mask: Optional[np.ndarray] = None, shell: Tuple[int, int] = (2, 3),
                  background: str = DEFAULT_BACKGROUND) -> IntegrationResult:
    """Integrate one spot in a box around ``centre = (frame, row, col)`` (fractional array indices).

    ``half = (frames, pixels)`` half-widths. ``mask`` is the (rows, cols) bool detector mask (True = bad).
    ``background``: ``"mean"`` subtracts n_box times the (outlier-trimmed) mean of the shell, exact for a flat background; ``"quadratic"`` subtracts the
    sum over the box of a polynomial fitted to the shell (see :func:`_fit_shell`), exact for any background that is quadratic in the detector plane and
    linear in omega. A smooth curved background makes the shell mean exceed the box mean, and the error grows with the box volume
    (PREREGISTER A36). A clipped box, or a shell too small for the fit, falls back to ``"mean"`` (``IntegrationResult.background_kind`` says which ran).
    """
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {BACKGROUNDS}, got {background!r}")
    nf, nr, nc = raw.shape
    hf, hp = half
    sf, sp = shell
    cf, cr, cc = (int(np.rint(centre[0])), int(np.rint(centre[1])), int(np.rint(centre[2])))
    f0, f1 = cf - hf, cf + hf + 1
    r0, r1 = cr - hp, cr + hp + 1
    c0, c1 = cc - hp, cc + hp + 1
    clipped_f = f0 < 0 or f1 > nf
    clipped_d = r0 < 0 or r1 > nr or c0 < 0 or c1 > nc
    F0, F1 = max(f0 - sf, 0), min(f1 + sf, nf)
    R0, R1 = max(r0 - sp, 0), min(r1 + sp, nr)
    C0, C1 = max(c0 - sp, 0), min(c1 + sp, nc)
    if F1 <= F0 or R1 <= R0 or C1 <= C0:
        return IntegrationResult(float("nan"), float("nan"), float("nan"), 0, 0, float("nan"),
                                 clipped_f, clipped_d, 0, False)
    outer = raw[F0:F1, R0:R1, C0:C1].astype(np.float64)
    inner = np.zeros(outer.shape, dtype=bool)
    inner[max(f0, 0) - F0:min(f1, nf) - F0, max(r0, 0) - R0:min(r1, nr) - R0,
          max(c0, 0) - C0:min(c1, nc) - C0] = True
    bad = np.zeros(outer.shape, dtype=bool)
    if mask is not None:
        bad[:] = np.asarray(mask, bool)[R0:R1, C0:C1][None, :, :]
    shell_v = outer[~inner & ~bad]
    box_v = outer[inner & ~bad]
    n_masked = int((inner & bad).sum())
    if shell_v.size < 20 or box_v.size == 0:
        return IntegrationResult(float("nan"), float("nan"), float("nan"), int(box_v.size),
                                 int(shell_v.size), float("nan"), clipped_f, clipped_d, n_masked, False)
    n_box = int(box_v.size)
    kind, fitted = "mean", None
    if background == "quadratic" and not (clipped_f or clipped_d):
        ff, rr, cc_ = np.indices(outer.shape).astype(np.float64)
        ff += F0; rr += R0; cc_ += C0
        sel, bsel = ~inner & ~bad, inner & ~bad
        fitted = _fit_shell((ff[sel], rr[sel], cc_[sel]), shell_v, (ff[bsel], rr[bsel], cc_[bsel]), (cf, cr, cc), (hf, hp))
        if fitted is not None:
            kind = "quadratic"
    if fitted is not None:
        bg_sum, var_bg = fitted
        bg = bg_sum / n_box
        intensity = float(box_v.sum() - bg_sum)
        var = float(box_v.sum()) + var_bg
    else:
        bg = _robust_mean(shell_v)
        intensity = float(box_v.sum() - n_box * bg)
        var = float(box_v.sum()) + (n_box ** 2) * max(bg, 0.0) / shell_v.size
    return IntegrationResult(intensity, float(np.sqrt(max(var, 1e-12))), bg, n_box, int(shell_v.size),
                             float(box_v.max()), clipped_f, clipped_d, n_masked, True, kind)


def _gauss_widths(win: np.ndarray, bg: float) -> Tuple[float, float, float]:
    """Gaussian-equivalent sigmas (frame, row, col) of a spot, from a Poisson-weighted least-squares fit.

    Not second moments: over a window much larger than the spot the r**2 weighting makes a moment
    dominated by background noise at the window edge (measured: +-15-35 % per spot at 4e4 counts), while
    the fit weights by the profile itself.
    """
    from scipy.optimize import least_squares
    f, r, c = np.indices(win.shape, dtype=float)
    y = win.astype(float).ravel()
    wgt = 1.0 / np.sqrt(np.maximum(y, 1.0))
    pk = np.unravel_index(int(np.argmax(win)), win.shape)

    def model(p):
        A, f0, r0, c0, sf, sr, sc, b = p
        return (A * np.exp(-0.5 * (((f - f0) / sf) ** 2 + ((r - r0) / sr) ** 2 + ((c - c0) / sc) ** 2)) + b).ravel()

    p0 = [max(float(win[pk]) - bg, 1.0), pk[0], pk[1], pk[2], 1.0, 1.0, 1.0, bg]
    lo = [0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0]
    hi = [np.inf, win.shape[0], win.shape[1], win.shape[2], 8.0, 8.0, 8.0, np.inf]
    sol = least_squares(lambda p: (model(p) - y) * wgt, p0, bounds=(lo, hi))
    return float(sol.x[4]), float(sol.x[5]), float(sol.x[6])


def fit_profile_model(raw: np.ndarray, centres: np.ndarray, L: np.ndarray, *,
                      mask: Optional[np.ndarray] = None, window: Tuple[int, int] = (8, 8),
                      min_peak_over_bg: float = 20.0, max_L: float = 8.0,
                      k: float = 3.5, shell_px: int = 3, shell_frames: int = 2,
                      background: str = DEFAULT_BACKGROUND) -> Tuple[ProfileModel, dict]:
    """Measure the profile model from STRONG spots only.

    ``centres`` (N, 3) = (frame, row, col); ``L`` (N,) the Lorentz factors. Strong = peak voxel above
    ``min_peak_over_bg`` times the shell background and L < ``max_L`` (wider spots overrun the window).
    The widths are Gaussian-fit sigmas (:func:`_gauss_widths`), not moments; ``k`` is then chosen
    separately by :func:`choose_support_scale`.
    Returns ``(model, info)``; ``info`` carries the per-spot widths for plotting and for refusing to
    proceed when too few strong spots exist.
    """
    wf, wp = window
    rows = []
    for (cf, cr, cc), Li in zip(centres, L):
        if not np.isfinite(Li) or Li > max_L:
            continue
        res = integrate_box(raw, (cf, cr, cc), (wf, wp), mask=mask, shell=(shell_frames, shell_px), background=background)
        if not res.ok or res.clipped_frames or res.clipped_detector or res.n_masked:
            continue
        if res.peak < res.background * (1.0 + min_peak_over_bg):
            continue
        f0, r0, c0 = int(np.rint(cf)), int(np.rint(cr)), int(np.rint(cc))
        win = raw[f0 - wf:f0 + wf + 1, r0 - wp:r0 + wp + 1, c0 - wp:c0 + wp + 1].astype(float)
        sf, sr, sc = _gauss_widths(win, res.background)
        rows.append((Li, sf, 0.5 * (sr + sc)))
    if len(rows) < 5:
        raise RuntimeError(f"only {len(rows)} strong, unclipped, low-L spots to measure the profile from; "
                           "need >= 5 (collect stronger data or relax min_peak_over_bg)")
    a = np.array(rows)
    A = np.vstack([np.ones(len(a)), a[:, 0]]).T
    coef, *_ = np.linalg.lstsq(A, a[:, 1], rcond=None)
    sf0, sf1 = float(max(coef[0], 0.1)), float(max(coef[1], 0.0))
    model = ProfileModel(sigma_px=float(np.median(a[:, 2])), sf0=sf0, sf1=sf1, k=k,
                         shell_px=shell_px, shell_frames=shell_frames, background=background)
    return model, dict(n_strong=len(a), L=a[:, 0], sigma_frames=a[:, 1], sigma_px=a[:, 2])


def choose_support_scale(raw: np.ndarray, centres: np.ndarray, L: np.ndarray, model: ProfileModel, *,
                         mask: Optional[np.ndarray] = None, ks: Sequence[float] = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0),
                         plateau: float = 0.995, min_peak_over_bg: float = 20.0) -> Tuple[ProfileModel, dict]:
    """Sweep ``k`` on the strong spots and pick the smallest on the plateau.

    For each k the strong spots are re-integrated; the criterion is that the MEDIAN of I(k)/I(k_max)
    reaches ``plateau``. If it never does, the largest k is returned and ``info['converged']`` is False,
    which callers must report.
    """
    ks = sorted(float(x) for x in ks)
    table = []
    for k in ks:
        m = model.with_k(k)
        vals = []
        for (cf, cr, cc), Li in zip(centres, L):
            if not np.isfinite(Li) or Li > 8.0:
                continue
            ref = integrate_box(raw, (cf, cr, cc), model.with_k(ks[-1]).half_widths(Li), mask=mask,
                                shell=(model.shell_frames, model.shell_px), background=model.background)
            if (not ref.ok or ref.clipped_frames or ref.clipped_detector or ref.n_masked
                    or ref.peak < ref.background * (1.0 + min_peak_over_bg)):
                continue
            r = integrate_box(raw, (cf, cr, cc), m.half_widths(Li), mask=mask,
                              shell=(model.shell_frames, model.shell_px), background=model.background)
            if r.ok and ref.intensity > 0:
                vals.append(r.intensity / ref.intensity)
        table.append((k, float(np.median(vals)) if vals else float("nan"),
                      float(np.percentile(vals, 10)) if vals else float("nan"), len(vals)))
    chosen, converged = ks[-1], False
    for k, med, _, n in table:
        if n >= 5 and med >= plateau:
            chosen, converged = k, True
            break
    return model.with_k(chosen), dict(table=table, converged=converged, chosen=chosen)


def _fit_flat_top(p: np.ndarray, var: np.ndarray) -> Tuple[np.ndarray, float]:
    """Weighted fit of ``A * [erf((x-x0+h)/(sqrt2 s)) - erf((x-x0-h)/(sqrt2 s))] / 2`` to a frame profile.

    A boxcar of half-width h convolved with a Gaussian of width s: a Gaussian when h -> 0 (narrow spots) and a flat top when
    h >> s (the wide shell crossings of high-L spots). Returns ``(params A,x0,h,s, chi2)``.
    """
    from scipy.optimize import least_squares
    from scipy.special import erf
    n = len(p)
    x = np.arange(n, dtype=float)
    tot = float(p.sum())
    c0 = float((p * x).sum() / tot) if tot > 0 else n / 2.0
    spread = float(np.sqrt(max((p * (x - c0) ** 2).sum() / max(tot, 1e-9), 0.25)))
    w = 1.0 / np.sqrt(np.maximum(var, 1.0))

    def model(q, xx=x):
        A, x0, h, sg = q
        return A * 0.5 * (erf((xx - x0 + h) / (np.sqrt(2) * sg)) - erf((xx - x0 - h) / (np.sqrt(2) * sg)))

    best = None
    lo, hi = np.array([0.0, 0.0, 0.0, 0.25]), np.array([np.inf, n - 1.0, n / 2.0, n / 2.0])
    for h0, s0 in ((max(spread * 1.3 - 0.5, 0.0), 0.6), (0.0, max(spread, 0.5)), (spread * 1.7, 1.0)):
        q0 = [max(float(p.max()), 1.0), c0, h0, s0]
        # the start must lie strictly inside the bounds: on a profile with negative background-subtracted bins (a near-zero background, e.g. a photon-counting detector) the centroid c0 can
        # fall outside [0, n-1] and the width start above n/2 (scipy: 'Initial guess is outside of provided bounds', which aborted the whole profile measurement on the L-alanine frames)
        q0 = np.minimum(np.maximum(q0, lo + 1e-6), hi - 1e-6)
        sol = least_squares(lambda q: (model(q) - p) * w, q0, bounds=(lo, hi))
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x, float(2 * best.cost)


def _model_half_extent(q: np.ndarray, centre: float, frac: float, span: float) -> float:
    """Smallest h such that the fitted profile holds ``frac`` of its area within ``centre +- h`` (fine grid)."""
    from scipy.special import erf
    A, x0, h, sg = q
    xx = np.arange(-span - 30.0, span + 30.0 + 1, 0.05) + centre
    m = A * 0.5 * (erf((xx - x0 + h) / (np.sqrt(2) * sg)) - erf((xx - x0 - h) / (np.sqrt(2) * sg)))
    total = m.sum()
    if total <= 0:
        return 0.0
    cum = np.cumsum(m)
    cum_at = lambda t: np.interp(t, xx, cum)
    for hh in np.arange(0.0, span + 30.0, 0.05):
        if cum_at(centre + hh) - cum_at(centre - hh - 0.05) >= frac * total:
            return float(hh)
    return float(span + 30.0)


def measure_extents(raw: np.ndarray, centres: np.ndarray, L: np.ndarray, *, mask: Optional[np.ndarray] = None,
                    frac: float = 0.995, max_L: float = 80.0, min_snr: float = 15.0, win_p: int = 8, core_px: int = 3,
                    shell: Tuple[int, int] = (2, 3), grow: float = 1.6, max_tries: int = 4,
                    background: str = DEFAULT_BACKGROUND) -> Tuple[ProfileModel, dict]:
    """Measure the half-extent holding ``frac`` of each strong spot's signal and regress it on L (PREREGISTER A11).

    The omega profile (summed over the in-plane core, background removed) is fitted with a flat-top model
    (:func:`_fit_flat_top`) and the extent is read from the FITTED profile, measured from the integration centre (the rounded blob
    centroid, so any offset of a skewed profile is inside the extent). A fit is used rather than the raw cumulative because 0.5 %
    of a weak spot is below its own noise. The window starts at ``6 + 0.15 L`` frames and is grown by ``grow`` until the
    extent uses less than 80 % of it. Strong means core S/N >= ``min_snr`` (not a peak-over-background test, which drops the wide,
    thin, high-L spots this exists for). The in-plane extent is 3.3 Gaussian sigmas (99.5 % of a 2-D Gaussian) of the fitted image.
    Returns ``(model, info)`` with the v2 fields set and ``margin = 0``.
    """
    from scipy.optimize import least_squares
    rows = []
    for (cf, cr, cc), Li in zip(centres, L):
        if not np.isfinite(Li) or Li > max_L:
            continue
        wf = int(np.ceil(6 + 0.15 * Li))
        res = None
        for _ in range(max_tries):
            r = integrate_box(raw, (cf, cr, cc), (wf, win_p), mask=mask, shell=shell, background=background)
            if (not r.ok or r.clipped_frames or r.clipped_detector or r.n_masked):
                res = None
                break
            f0, r0, c0 = int(np.rint(cf)), int(np.rint(cr)), int(np.rint(cc))
            sub = raw[f0 - wf:f0 + wf + 1, r0 - win_p:r0 + win_p + 1, c0 - win_p:c0 + win_p + 1].astype(float) - r.background
            core = sub[:, win_p - core_px:win_p + core_px + 1, win_p - core_px:win_p + core_px + 1]
            prof = core.sum(axis=(1, 2))
            var = (2 * core_px + 1) ** 2 * max(r.background, 1.0) + np.maximum(prof, 0.0)
            snr = float(prof.sum() / np.sqrt(var.sum()))
            if snr < min_snr:
                res = None
                break
            q, chi2 = _fit_flat_top(prof, var)
            h99 = _model_half_extent(q, float(wf), frac, float(wf))
            if h99 < 0.8 * wf:
                res = (r, sub, q, h99)
                break
            wf = int(np.ceil(grow * wf))
        if res is None:
            continue
        r, sub, q, h99 = res
        hf = int(np.ceil(h99))
        # in-plane: isotropic Gaussian fit to the frame-summed image over the frames the spot occupies
        img = sub[max(wf - hf, 0):wf + hf + 1].sum(axis=0)
        yy, xx = np.indices(img.shape, dtype=float)

        def g2(t):
            return t[0] * np.exp(-0.5 * (((yy - t[1]) / t[3]) ** 2 + ((xx - t[2]) / t[3]) ** 2))
        t0 = [max(float(img.max()), 1.0), win_p, win_p, 1.0]
        sol = least_squares(lambda t: (g2(t) - img).ravel(), t0, bounds=([0, 0, 0, 0.3], [np.inf, 2 * win_p, 2 * win_p, 6.0]))
        sig_p = float(sol.x[3])
        off = float(max(abs(sol.x[1] - win_p), abs(sol.x[2] - win_p)))
        rows.append((float(Li), h99, 3.3 * sig_p + off, r.intensity, snr))
    if len(rows) < 8:
        raise RuntimeError(f"only {len(rows)} strong, unclipped spots to measure extents from; need >= 8")
    a = np.array(rows)
    A = np.vstack([np.ones(len(a)), a[:, 0]]).T
    coef, *_ = np.linalg.lstsq(A, a[:, 1], rcond=None)
    ext_f0, ext_f1 = float(max(coef[0], 1.0)), float(max(coef[1], 0.0))
    ext_p = float(np.percentile(a[:, 2], 90))
    model = ProfileModel(sigma_px=float(ext_p / 3.3), sf0=ext_f0 / 3.0, sf1=ext_f1 / 3.0, k=3.0, ext_f0=ext_f0,
                         ext_f1=ext_f1, ext_p=ext_p, margin=0, shell_frames=shell[0], shell_px=shell[1], background=background)
    return model, dict(n_strong=len(a), L=a[:, 0], half_frames=a[:, 1], half_px=a[:, 2],
                       n_high_L=int((a[:, 0] > 10).sum()), L_max=float(a[:, 0].max()))


def choose_margin(raw: np.ndarray, centres: np.ndarray, L: np.ndarray, model: ProfileModel, *,
                  mask: Optional[np.ndarray] = None, margins: Sequence[int] = (0, 1, 2, 3, 4, 6),
                  plateau: float = 0.995, min_snr: float = 15.0, max_L: float = 80.0) -> Tuple[ProfileModel, dict]:
    """Smallest integer margin on the plateau: median of I(m)/I(m_max) over strong spots (ALL L, unlike the v1 sweep) >= ``plateau``."""
    ms = sorted(int(m) for m in margins)
    table = []
    ref_model = model.with_margin(ms[-1])
    ref = {}
    for i, ((cf, cr, cc), Li) in enumerate(zip(centres, L)):
        if not np.isfinite(Li) or Li > max_L:
            continue
        r = integrate_box(raw, (cf, cr, cc), ref_model.half_widths(Li), mask=mask, shell=(model.shell_frames, model.shell_px), background=model.background)
        if r.ok and not (r.clipped_frames or r.clipped_detector or r.n_masked) and r.intensity >= min_snr * r.sigma:
            ref[i] = r.intensity
    for m in ms:
        mm = model.with_margin(m)
        vals = []
        for i, I_ref in ref.items():
            cf, cr, cc = centres[i]
            r = integrate_box(raw, (cf, cr, cc), mm.half_widths(L[i]), mask=mask, shell=(model.shell_frames, model.shell_px), background=model.background)
            if r.ok:
                vals.append(r.intensity / I_ref)
        table.append((m, float(np.median(vals)) if vals else float("nan"),
                      float(np.percentile(vals, 10)) if vals else float("nan"), len(vals)))
    chosen, converged = ms[-1], False
    for m, med, _, n in table[:-1]:                           # the reference margin (the largest) compares with itself: it cannot certify a plateau
        if n >= 5 and med >= plateau:
            chosen, converged = m, True
            break
    return model.with_margin(chosen), dict(table=table, converged=converged, chosen=chosen)


def build_profile(raw: np.ndarray, centres: np.ndarray, L: np.ndarray, *, mask: Optional[np.ndarray] = None,
                  background: str = DEFAULT_BACKGROUND) -> Tuple[ProfileModel, dict]:
    """v2 support for a reduction: measured extents, then the plateau margin. Returns ``(model, info)``; the model carries ``background``."""
    model, ext = measure_extents(raw, centres, L, mask=mask, background=background)
    model, marg = choose_margin(raw, centres, L, model, mask=mask)
    if not marg["converged"]:
        import warnings
        warnings.warn(f"build_profile: the integrated intensity of the strong spots is still growing with the support margin (median I(m)/I(m={marg['chosen']}) = "
                      f"{[(m, round(v, 3)) for m, v, _, _ in marg['table']]}): broad tails, a neighbouring signal or a curved background. The largest margin is used; "
                      "the box is large and the weak reflections carry its noise.", RuntimeWarning, stacklevel=2)
    return model, dict(extents=ext, margin=marg)
