"""Rocking-curve centres near the ends of the scanned range. Version 3.

**v1 and v2 both failed their own preregistered benchmarks**
($ANALYSIS/dfxm_datasetG/edge_fix/, claims `0a9c1932172d` REFUTED, next round
INCONCLUSIVE). v1 forced a *symmetric* window around the peak, discarding real far-side data and
producing a -17 mdeg bias. v2 replaced that with an unclipped moment over the *entire* recorded
curve, on the reasoning that a zero-mean baseline error summed over any number of points still
averages toward zero -- true in principle, but real Mg-4Al curves sit on a documented angular
background that is not perfectly flat across the whole scan (LAB_NOTEBOOK §11), and summing over
more points made the centroid *more* exposed to that shape, not less: -2 to -4 mdeg residual bias
on real frames, worse than simply using the package's own existing `reduce_rocking(window="peak")`
(+0.2 to +0.6 mdeg on the same real-data test). Building v2 also surfaced a real second bug -- the
shared baseline routine's percentile fallback is systematically biased -- but the fix for it, as
first scoped, changed the answer for 11.7% of *all* pixels on real data, not just edge cases
(differing pixels had a median margin of 10 out of 12 available, nowhere near an edge), which is
outside what a single synthetic test validated.

**v3 stops trying to out-invent the centre estimator.** It delegates the centre entirely to
`midas_dfxm.reduce_rocking(window="peak")`, the package's own, already-tested reducer, which
already avoids v1's specific mistake (its window narrows to the array bound on one side without
discarding good data on the other) and measured the smallest bias of everything tried. The
percentile-fallback bug found while building v2 is documented (`LAB_NOTEBOOK.md`) but not patched
here, since a fix broad enough to be safe needs owning `rocking.py` itself, not working around it
from outside.

**What v3 actually adds, on top of that centre:**

(a) a bound ``[lower, upper]`` on the complete curve's centroid where the recorded curve is cut,
    from "nothing beyond the end" to "the cut flank runs ``flank_ratio`` times as far as the
    recorded flank, at no more than its last recorded level" -- ``flank_ratio`` calibrated from the
    dataset's own well-recorded pixels (:func:`calibrate_flank_ratio`).
(c) a lineshape extrapolation with :func:`midas_dfxm.fit_rocking_curve` (one pseudo-Voigt),
    accepted only where it describes the curve (single-peaked, low residual, not railed) **and**
    the observed maximum is not on the very end frame -- v3 extrapolates a missing flank, never a
    missing peak.

**The bound is a measured diagnostic, not a reliable interval -- do not report it as one.** Three
designs, re-run on two real datasets each time
($ANALYSIS/dfxm_datasetG/edge_fix/VERDICT_edge_centres_final.md), never reliably
covered the true centroid at the 90% target this campaign set for it: Mg-4Al pooled coverage on
planted cuts was 0.60 even at the widest calibrated ratio (30x); a second, independent dataset
with broader, multi-featured curves ranged 0.86-0.93 across three scans and could not be
calibrated at all on a fourth (zero well-recorded pixels available to calibrate from). ``lower``/``upper`` are still returned, for
inspection of scale, but **use ``truncated_low``/``truncated_high`` (reliable: a direct read of
whether real signal sits above noise at the boundary) plus ``fit_ok`` (reliable when it fires:
83-96% of accepted fits land within a quarter of the peak's FWHM, degrading as the missing
fraction grows) as the two numbers actually worth reporting per pixel** -- see
:attr:`EdgeCentres.status` for the categorisation this recommends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["EdgeCentres", "edge_centres", "edge_centres_scan", "calibrate_flank_ratio"]


@dataclass
class EdgeCentres:
    """Per-pixel output of :func:`edge_centres` (arrays are ``(N,)``, or ``(H, W)`` from a scan)."""

    centre: np.ndarray        # deg, == reduce_rocking(window="peak").centre_deg exactly
    lower: np.ndarray         # deg, bound on the complete curve's centroid (-inf: open)
    upper: np.ndarray         # deg (+inf: open)
    truncated_low: np.ndarray   # curve above noise at the first point, in the peak's own run
    truncated_high: np.ndarray  # same at the last point
    truncated_pkg: np.ndarray   # reduce_rocking's own flag: the peak window hit an array bound
    margin: np.ndarray        # frames between the (raw-corrected) maximum and the nearer end
    tail_low: np.ndarray      # signal at the first point / peak height
    tail_high: np.ndarray     # signal at the last point / peak height
    peak: np.ndarray          # smoothed peak height (counts above baseline)
    baseline: np.ndarray      # counts per point, from reduce_rocking(window="peak")
    noise: np.ndarray         # per-point noise, 1.4826 x MAD of the frames outside the peak
    single_peaked: np.ndarray
    flank_ratio: float
    bound_calibration: dict
    fit_centre: Optional[np.ndarray] = None   # deg, NaN where not fitted
    fit_width: Optional[np.ndarray] = None    # deg FWHM
    fit_resid: Optional[np.ndarray] = None    # residual RMS / peak height
    fit_ok: Optional[np.ndarray] = None
    settings: dict = field(default_factory=dict)

    @property
    def truncated(self) -> np.ndarray:
        return self.truncated_low | self.truncated_high

    @property
    def bound_width(self) -> np.ndarray:
        """``upper - lower`` in deg (``inf`` where a side is open, 0 where nothing is cut)."""
        return self.upper - self.lower

    @property
    def status(self) -> np.ndarray:
        """Per pixel, the recommended reading (module docstring): ``"ok"`` (not truncated, use
        ``centre``), ``"recovered"`` (truncated, ``fit_ok`` -- use ``fit_centre``, with the cut
        fraction/residual as its own caveat), or ``"undetermined"`` (truncated, no accepted fit --
        do not report a numeric centre; ``lower``/``upper`` remain available to inspect, not to
        quote as a reliable interval)."""
        out = np.full(self.centre.shape, "ok", dtype=object)
        t = self.truncated
        ok_fit = self.fit_ok if self.fit_ok is not None else np.zeros_like(t)
        out[t] = "undetermined"
        out[t & ok_fit] = "recovered"
        return out


def _peak_reduce(I, x, *, peak_halfwidth, baseline_percentile, min_baseline_points):
    """centre and baseline from the package's own reduce_rocking(window='peak'); the single
    source of truth for both, per the module docstring."""
    from .rocking import RockingScan, reduce_rocking

    N = I.shape[1]
    scan = RockingScan.from_arrays(I[:, None, :].astype(np.float32), {"mu": np.asarray(x, float)})
    m = reduce_rocking(scan, window="peak", peak_halfwidth=peak_halfwidth,
                       baseline_percentile=baseline_percentile,
                       min_baseline_points=min_baseline_points, split_half=False,
                       lit=np.ones((1, N), bool))
    return m.centre_deg[0].astype(np.float64), m.baseline[0].astype(np.float64), m.truncated[0]


def _full_moment(s, x):
    """Unclipped first moment over every row -- used only for the reference centroid of an
    UNCROPPED, well-recorded pixel (calibration, A2-style validation), never as the reported
    centre (module docstring: that is what v2 got wrong)."""
    S = s.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (s * x[:, None]).sum(0) / S
    return np.where(S > 0, c, np.nan), S


def _locate_from_baseline(I, bl, peak_halfwidth):
    """The baseline-subtracted curve, its (raw-corrected) maximum index and the peak-window
    half-width used to find frames outside the peak for the noise estimate and the truncation
    check -- never to restrict a centroid. ``bl`` is always the package's own baseline."""
    from .rocking import _peak_window, _smooth3

    M, N = I.shape
    s = I - bl[None]
    ks, _n, h = _peak_window(s, peak_halfwidth)
    cols = np.arange(N)
    k = ks.copy()
    e0, e1 = ks == 0, ks == M - 1
    k[e0] = np.where(s[1, e0] > s[0, e0], 1, 0)
    k[e1] = np.where(s[M - 2, e1] > s[M - 1, e1], M - 2, M - 1)
    sm = _smooth3(s)
    peak = np.maximum(sm[ks, cols], s[k, cols])
    return s, k, h, peak


def _noise(s, k, h):
    from .rocking import _nanmedian0

    M, N = s.shape
    idx = np.arange(M)[:, None]
    off = np.abs(idx - k[None]) > h[None]
    sig = 1.4826 * _nanmedian0(np.where(off, np.abs(s), np.nan))
    bad = ~np.isfinite(sig) | (off.sum(0) < 3)
    if bad.any():
        glob = np.nanmedian(sig[~bad]) if (~bad).any() else np.nan
        sig = np.where(bad, glob, sig)
    return sig


def _run_and_bound(s, x, k, tail_frac, z, sig, peak, flank_ratio, centre):
    """Truncation flags and the bound on the complete curve's centroid, at a given flank_ratio."""
    from .rocking import _contiguous_run, _smooth3

    M, N = s.shape
    idx = np.arange(M)[:, None]
    cols = np.arange(N)
    sm = _smooth3(s)
    peak_row = np.where(np.arange(M)[:, None] == k[None], np.maximum(sm, s), sm)
    l1, r1 = _contiguous_run(peak_row, k, tail_frac)
    thr = z * sig / np.sqrt(2.0)
    lo_end = s[:2].mean(0)
    hi_end = s[-2:].mean(0)
    trunc_lo = (l1 <= 0) & (lo_end > thr) & (peak > z * sig)
    trunc_hi = (r1 >= M - 1) & (hi_end > thr) & (peak > z * sig)

    step = float(np.median(np.diff(x)))
    run = (idx >= l1[None]) & (idx <= r1[None])
    c_obs, S_obs = _full_moment(np.where(run, s, 0.0), x)
    a_lo = np.maximum(np.maximum(sm[0], s[0]), 0.0)
    a_hi = np.maximum(np.maximum(sm[-1], s[-1]), 0.0)
    L_lo = np.maximum(flank_ratio * (r1 - k) - k, 1.0)
    L_hi = np.maximum(flank_ratio * (k - l1) - (M - 1 - k), 1.0)
    m_lo = np.where(trunc_lo, a_lo * L_lo, 0.0)
    m_hi = np.where(trunc_hi, a_hi * L_hi, 0.0)
    x_lo = x[0] - step * (L_lo + 1.0) / 2.0
    x_hi = x[-1] + step * (L_hi + 1.0) / 2.0
    combos = []
    for tl in (0.0, 1.0):
        for th in (0.0, 1.0):
            num = S_obs * c_obs + tl * m_lo * x_lo + th * m_hi * x_hi
            den = S_obs + tl * m_lo + th * m_hi
            with np.errstate(invalid="ignore", divide="ignore"):
                combos.append(np.where(den > 0, num / den, np.nan))
    combos = np.stack(combos)
    with np.errstate(invalid="ignore"):
        lower = np.where(np.all(np.isnan(combos), 0), np.nan, np.nanmin(combos, 0))
        upper = np.where(np.all(np.isnan(combos), 0), np.nan, np.nanmax(combos, 0))
    lower = np.where(trunc_lo & (k == 0), -np.inf, lower)
    upper = np.where(trunc_hi & (k == M - 1), np.inf, upper)
    return trunc_lo, trunc_hi, lower, upper, l1, r1


def calibrate_flank_ratio(I, x, *, peak_halfwidth=2.0, baseline_percentile=25.0,
                          min_baseline_points=6, z=3.0, tail_frac=0.10, target_coverage=0.90,
                          candidates=None, max_drop=6, min_closed=200, test_frac=0.5, seed=0):
    """Choose ``flank_ratio`` from this dataset's own well-recorded ("closed") pixels.

    Crops each closed pixel's curve by 1..``max_drop`` points from a random end (never past its
    own peak) and checks whether the bound at each candidate ratio contains the *uncropped*
    pixel's own full-curve centroid. Picks the smallest ratio reaching ``target_coverage`` on a
    calibration half, and reports coverage on a held-out test half. Falls back to the smallest
    candidate with a note if fewer than ``min_closed`` pixels qualify. If no candidate reaches the
    target, returns the largest candidate and the report shows the coverage it actually achieved --
    read that number, not just the fact that calibration "ran".
    """
    if candidates is None:
        candidates = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 20.0, 30.0])
    I = np.asarray(I, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    M, N = I.shape
    centre, bl, trunc_pkg = _peak_reduce(I, x, peak_halfwidth=peak_halfwidth,
                                         baseline_percentile=baseline_percentile,
                                         min_baseline_points=min_baseline_points)
    s, k, h, peak = _locate_from_baseline(I, bl, peak_halfwidth)
    sig = _noise(s, k, h)
    full_centre, _ = _full_moment(s, x)
    trunc_lo, trunc_hi, lo0, up0, _l1, _r1 = _run_and_bound(s, x, k, tail_frac, z, sig, peak, 1.0, centre)
    margin = np.minimum(k, M - 1 - k)
    closed = (~trunc_lo & ~trunc_hi & ~trunc_pkg & (peak > 10 * sig)
             & (margin >= max(max_drop + 2, M // 4)) & np.isfinite(centre))
    idx = np.flatnonzero(closed)
    report = {"n_closed": int(idx.size), "candidates": candidates.tolist()}
    if idx.size < min_closed:
        report.update(calibrated=False, n_calib=0, n_test=0, coverage_calib=None, coverage_test=None)
        return float(candidates[0]), report

    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    n_calib = max(1, int(round(idx.size * (1 - test_frac))))
    calib_idx, test_idx = idx[:n_calib], idx[n_calib:]

    def coverage(sub, ratio):
        rows_ref, rows_cov = [], []
        for q in range(1, max_drop + 1):
            for side in ("low", "high"):
                sel = sub[margin[sub] > q]
                if sel.size == 0:
                    continue
                if side == "low":
                    Ic, xc = I[q:][:, sel], x[q:]
                else:
                    Ic, xc = I[:-q][:, sel], x[:-q]
                cc, blc, _tp = _peak_reduce(Ic, xc, peak_halfwidth=peak_halfwidth,
                                            baseline_percentile=baseline_percentile,
                                            min_baseline_points=min_baseline_points)
                sc, kc, hc, pkc = _locate_from_baseline(Ic, blc, peak_halfwidth)
                sigc = _noise(sc, kc, hc)
                tl, th_, lo, up, _l1, _r1 = _run_and_bound(sc, xc, kc, tail_frac, z, sigc, pkc, ratio, cc)
                flagged = tl if side == "low" else th_
                fin = flagged & np.isfinite(lo) & np.isfinite(up)
                ref = full_centre[sel]
                rows_ref.append(ref[fin]); rows_cov.append((lo[fin] <= ref[fin]) & (ref[fin] <= up[fin]))
        cov = np.concatenate(rows_cov) if rows_cov else np.array([])
        return float(cov.mean()) if cov.size else None

    best = float(candidates[-1])
    cov_calib_best = None
    for ratio in candidates:
        c = coverage(calib_idx, float(ratio))
        if c is not None and c >= target_coverage:
            best, cov_calib_best = float(ratio), c
            break
        cov_calib_best = c
    cov_test = coverage(test_idx, best) if test_idx.size else None
    report.update(calibrated=True, n_calib=int(calib_idx.size), n_test=int(test_idx.size),
                  coverage_calib=cov_calib_best, coverage_test=cov_test, chosen=best,
                  target_reached=bool(cov_calib_best is not None and cov_calib_best >= target_coverage))
    return best, report


def edge_centres(I, x, *, peak_halfwidth: float = 2.0, baseline_percentile: float = 25.0,
                 min_baseline_points: int = 6, z: float = 3.0, tail_frac: float = 0.10,
                 flank_ratio="auto", target_coverage: float = 0.90,
                 fit: str = "truncated", fit_margin: int = 2, fit_min_margin: int = 1,
                 fit_eta: bool = False, resid_max: float = 0.10, chunk: int = 20000) -> EdgeCentres:
    """Edge-safe rocking-curve centres, a truncation bound, and a fitted extrapolation.

    ``centre`` is exactly ``midas_dfxm.reduce_rocking(window="peak")``'s own centroid -- see the
    module docstring for why that, rather than a new estimator, is what this module recommends.

    Parameters
    ----------
    I : (M, N) array
        Counts per rocking point (rows, increasing ``x``) per pixel (columns).
    x : (M,) array
        Rocking coordinate in deg, increasing and (near-)uniformly stepped.
    flank_ratio : float or "auto"
        Assumption behind the bound: the cut flank runs at most this many times as far as the
        recorded flank, at no more than its last recorded level. ``"auto"`` (default) calibrates
        it from this dataset's own well-recorded pixels via :func:`calibrate_flank_ratio` --
        **read the returned coverage, do not assume it hit the target** (module docstring).
    fit : {"none", "truncated", "all"}
        Which pixels get the lineshape extrapolation: none; those cut or within ``fit_margin``
        frames of an end (default); or every pixel above noise. Accepted only when, beyond the
        usual checks, the observed maximum sits at least ``fit_min_margin`` frames inside the
        recorded range -- extrapolating a missing flank, never a missing peak.
    resid_max : float
        Largest residual RMS, as a fraction of the peak height, for a fit to be accepted.
    """
    I = np.asarray(I, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if I.ndim != 2 or I.shape[0] != x.size:
        raise ValueError(f"I must be (M, N) with M = len(x); got {I.shape} and {x.size}")
    M, N = I.shape
    if M < 5:
        raise ValueError("need at least 5 rocking points")
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("x must be strictly increasing (order the frames first)")
    if fit not in ("none", "truncated", "all"):
        raise ValueError("fit must be 'none', 'truncated' or 'all'")

    centre, bl, trunc_pkg = _peak_reduce(I, x, peak_halfwidth=peak_halfwidth,
                                         baseline_percentile=baseline_percentile,
                                         min_baseline_points=min_baseline_points)
    s, k, h, peak = _locate_from_baseline(I, bl, peak_halfwidth)
    sig = _noise(s, k, h)
    margin = np.minimum(k, M - 1 - k)

    calib_report = {"calibrated": False}
    if flank_ratio == "auto":
        flank_ratio, calib_report = calibrate_flank_ratio(
            I, x, peak_halfwidth=peak_halfwidth, baseline_percentile=baseline_percentile,
            min_baseline_points=min_baseline_points, z=z, tail_frac=tail_frac,
            target_coverage=target_coverage)
    flank_ratio = float(flank_ratio)

    trunc_lo, trunc_hi, lower, upper, l1, r1 = _run_and_bound(
        s, x, k, tail_frac, z, sig, peak, flank_ratio, centre)
    with np.errstate(invalid="ignore", divide="ignore"):
        tail_lo = np.where(peak > 0, s[0] / peak, np.nan)
        tail_hi = np.where(peak > 0, s[-1] / peak, np.nan)

    from .rocking import _shape_stats
    share, n_feat = _shape_stats(s)
    single = (np.nan_to_num(share, nan=0.0) >= 0.45) & (n_feat == 1)

    out = EdgeCentres(centre=centre, lower=lower, upper=upper, truncated_low=trunc_lo,
                      truncated_high=trunc_hi, truncated_pkg=trunc_pkg, margin=margin,
                      tail_low=tail_lo, tail_high=tail_hi, peak=peak, baseline=bl, noise=sig,
                      single_peaked=single, flank_ratio=flank_ratio, bound_calibration=calib_report,
                      settings=dict(peak_halfwidth=peak_halfwidth, baseline_percentile=baseline_percentile,
                                    min_baseline_points=min_baseline_points, z=z, tail_frac=tail_frac,
                                    target_coverage=target_coverage, fit=fit, fit_margin=fit_margin,
                                    fit_min_margin=fit_min_margin, fit_eta=fit_eta, resid_max=resid_max))

    if fit != "none":
        import torch
        from .mosaicity_fit import fit_rocking_curve

        above = np.isfinite(centre) & (peak > z * sig) & (margin >= fit_min_margin)
        sel = above if fit == "all" else above & (trunc_lo | trunc_hi | (margin <= fit_margin))
        fc = np.full(N, np.nan); fw = np.full(N, np.nan); fr = np.full(N, np.nan)
        railed = np.zeros(N, bool)
        rng = float(x[-1] - x[0])
        c_lo, c_hi = x[0] - 0.1 * rng, x[-1] + 0.1 * rng
        step = float(np.median(dx))
        where = np.flatnonzero(sel)
        theta = torch.as_tensor(x, dtype=torch.float64)
        for a in range(0, where.size, chunk):
            j = where[a:a + chunk]
            data = torch.as_tensor(s[:, j].T.copy(), dtype=torch.float64)
            sg = torch.as_tensor(np.maximum(sig[j], 1e-9), dtype=torch.float64)
            r = fit_rocking_curve(data, theta, n_components=1, sigma=sg, fit_eta=fit_eta)
            cj = r["center"][:, 0].numpy(); wj = r["width"][:, 0].numpy()
            res = (data - r["model"]).numpy()
            fc[j] = cj; fw[j] = wj
            fr[j] = np.sqrt((res ** 2).mean(1)) / np.maximum(peak[j], 1e-9)
            railed[j] = ((np.abs(cj - c_lo) < 0.01 * rng) | (np.abs(cj - c_hi) < 0.01 * rng)
                         | (wj <= 1.01 * step) | (wj >= 0.99 * rng))
        out.fit_centre = fc
        out.fit_width = fw
        out.fit_resid = fr
        out.fit_ok = sel & single & ~railed & (fr <= resid_max)
    return out


def edge_centres_scan(scan, *, roi=None, **kwargs) -> EdgeCentres:
    """:func:`edge_centres` on a 1-D :class:`midas_dfxm.RockingScan`; arrays come back ``(H, W)``."""
    from .rocking import _crop, _order_1d

    if scan.scan_type not in ("tilt", "strain"):
        raise ValueError("edge_centres_scan needs a 1-D rocking scan (tilt or strain)")
    F = _crop(scan.frames, roi)
    M, H, W = F.shape
    order = _order_1d(scan)
    x = np.asarray(scan.coordinate, dtype=float)[order]
    res = edge_centres(F[order].reshape(M, -1).astype(np.float64), x, **kwargs)
    for name in ("centre", "lower", "upper", "truncated_low", "truncated_high", "truncated_pkg",
                 "margin", "tail_low", "tail_high", "peak", "baseline", "noise", "single_peaked",
                 "fit_centre", "fit_width", "fit_resid", "fit_ok"):
        v = getattr(res, name)
        if v is not None:
            setattr(res, name, np.asarray(v).reshape(H, W))
    return res
