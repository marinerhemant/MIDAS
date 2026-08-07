"""Data-driven grain-quality gate — with the right to refuse.

Users need to know which grains to trust. The obvious answer, a threshold on
``DiffPos`` or ``Confidence``, cannot be shipped: those are properties of a
detector and a geometry, not of the method. Measured on shade_LSHR layer 1, the
EBSD-optimal ``DiffPos`` cut is **195.4 µm** for the classical C chain and
**222.8 µm** for the python chain — 14% apart on *the same raw data at the same
beamline*. A constant baked into the package would be an artefact of whichever
dataset it was tuned on.

What does transfer is the *procedure*: grain quality on a good dataset is
bimodal — a tight population of well-fitted grains and a separate tail of
failures — so the cut can be read off each dataset's own distribution. On the
same two runs, the antimode of the log₁₀ histogram picks 166 µm and 219 µm
respectively and reaches the EBSD-optimal purity (0.993 and 0.981) without ever
being shown the EBSD.

The load-bearing caveat, and the reason this module exists rather than a
one-liner: **the procedure is only valid when the distribution really is
bimodal.** On a dataset whose quality degrades smoothly there is no valley, and
an antimode taken anyway is a number invented from noise. So the gate tests for
separation first and returns ``threshold=None`` when it is not there, leaving
the caller to keep every grain rather than cut at a fiction.

This mirrors :func:`midas_process_grains.compute.adaptive.derive_misori_tol`,
which already derives the *misorientation* tolerance from an antimode; that one
clamps to a floor/ceiling, which is right for a tolerance that must exist and
wrong for a gate that may not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

__all__ = ["QualityGate", "adaptive_quality_threshold"]


@dataclass
class QualityGate:
    """Outcome of a gate attempt.

    ``threshold`` is ``None`` when the distribution gave no defensible place to
    cut; ``reason`` then says why, and callers must keep everything.
    """

    threshold: Optional[float]
    lower_is_better: bool
    method: str
    reason: str
    n_total: int
    n_kept: Optional[int]
    bimodality: float          # ΔBIC, 1-component minus 2-component; >0 favours 2
    ashman_d: float            # sqrt(2)|dmu|/sqrt(sd1^2+sd2^2); >2 = resolvable
    valley_depth: float        # diagnostic only; 0 when the modes differ in height
    modes: Tuple[float, float] | None   # the two mode centres, in transformed units

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Boolean keep-mask. All-True when the gate declined."""
        v = np.asarray(values, dtype=float)
        if self.threshold is None:
            return np.ones(v.shape[0], dtype=bool)
        return v <= self.threshold if self.lower_is_better else v >= self.threshold


def _fit_gmm(x: np.ndarray, k: int, iters: int = 300, seed: int = 0):
    """Tiny 1-D EM. Returns (means, sds, weights, loglik)."""
    rng = np.random.default_rng(seed)
    if k == 1:
        mu = np.array([x.mean()])
        sd = np.array([max(x.std(), 1e-9)])
        w = np.array([1.0])
    else:
        q = np.percentile(x, np.linspace(100 / (k + 1), 100 * k / (k + 1), k))
        mu = q + rng.normal(0, 1e-6, k)
        sd = np.full(k, max(x.std(), 1e-9))
        w = np.full(k, 1.0 / k)
    xc = x.reshape(-1, 1)
    for _ in range(iters):
        p = w * np.exp(-0.5 * ((xc - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        tot = np.clip(p.sum(1, keepdims=True), 1e-300, None)
        r = p / tot
        n = np.clip(r.sum(0), 1e-12, None)
        mu = (r * xc).sum(0) / n
        sd = np.sqrt(np.clip((r * (xc - mu) ** 2).sum(0) / n, 1e-12, None))
        w = n / len(x)
    p = w * np.exp(-0.5 * ((xc - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    ll = float(np.log(np.clip(p.sum(1), 1e-300, None)).sum())
    return mu, sd, w, ll


def _posterior_crossing(mu, sd, w) -> float:
    """Where the two components have equal posterior. Falls back to the
    midpoint if the quadratic has no root inside the interval."""
    lo, hi = float(mu[0]), float(mu[1])
    g = np.linspace(lo, hi, 4000)
    p0 = w[0] * np.exp(-0.5 * ((g - mu[0]) / sd[0]) ** 2) / sd[0]
    p1 = w[1] * np.exp(-0.5 * ((g - mu[1]) / sd[1]) ** 2) / sd[1]
    d = p0 - p1
    cross = np.where(np.diff(np.sign(d)) != 0)[0]
    return float(g[cross[0]]) if len(cross) else 0.5 * (lo + hi)


def _bic(ll: float, k: int, n: int) -> float:
    """BIC for a k-component 1-D Gaussian mixture (3k-1 free parameters)."""
    return (3 * k - 1) * np.log(n) - 2 * ll


def adaptive_quality_threshold(
    values: np.ndarray,
    *,
    lower_is_better: bool = True,
    log_transform: bool = True,
    min_delta_bic: float = 10.0,
    min_ashman_d: float = 2.0,
    min_peak_prominence: float = 0.05,
    min_minority_frac: float = 0.01,
    smooth_bins: int = 160,
    smooth_sigma: float = 3.0,
) -> QualityGate:
    """Find a per-dataset quality cut, or decline to.

    Parameters
    ----------
    values
        Per-grain quality metric, e.g. ``DiffPos`` (µm) or ``Confidence``.
    lower_is_better
        True for error-like metrics (DiffPos), False for score-like ones.
    log_transform
        Work in log₁₀. Correct for positive, scale-free error metrics; turn it
        off for a bounded score such as Confidence.
    min_delta_bic
        How much better the 2-component fit must be before a cut is allowed.
        10 is the conventional "strong evidence" margin.
    min_ashman_d
        Required separation between the two fitted components,
        ``sqrt(2)|mu1-mu2| / sqrt(sd1^2+sd2^2)``. D > 2 is the standard bar for
        a resolvable mixture. This, not a histogram valley, is the guard against
        BIC preferring two components merely because one skewed population is
        badly fitted by a single Gaussian.
    min_minority_frac
        The smaller population must hold at least this fraction, else the
        "second mode" is a handful of outliers, not a population.
    min_peak_prominence
        A histogram maximum counts as a population only if its prominence
        reaches this fraction of the tallest peak. Rejects noise bumps.

    Returns
    -------
    QualityGate
        ``threshold=None`` means no defensible cut exists — keep everything.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.shape[0]
    nil = dict(lower_is_better=lower_is_better, n_total=n, n_kept=None,
               modes=None)
    if n < 50:
        return QualityGate(None, method="none", bimodality=0.0, ashman_d=0.0,
                           valley_depth=0.0,
                           reason=f"only {n} finite values; too few to gate", **nil)

    # A quantised metric makes spurious "modes". Grain radius derived from an
    # integer voxel count is the case that caught this: 4,328 grains took only
    # 324 distinct values, and the discreteness alone produced peaks the gate
    # happily cut on. A continuous quality metric never looks like this.
    n_distinct = int(np.unique(np.round(v, 9)).size)
    if n_distinct < max(50, n // 10):
        return QualityGate(
            None, method="none", bimodality=0.0, ashman_d=0.0, valley_depth=0.0,
            reason=(f"metric is quantised ({n_distinct} distinct values for "
                    f"{n} grains); its modes are discretisation, not populations"),
            **nil)

    x = np.log10(np.clip(v, 1e-12, None)) if log_transform else v.copy()
    if np.allclose(x, x[0]):
        return QualityGate(None, method="none", bimodality=0.0, ashman_d=0.0,
                           valley_depth=0.0, reason="metric is constant", **nil)

    # 1. is two populations actually a better description than one?
    _, _, _, ll1 = _fit_gmm(x, 1)
    mu2, sd2, w2, ll2 = _fit_gmm(x, 2)
    d_bic = _bic(ll1, 1, n) - _bic(ll2, 2, n)
    order = np.argsort(mu2)
    mu2, sd2, w2 = mu2[order], sd2[order], w2[order]
    minority = float(w2.min())

    # 2. is there a genuine second POPULATION, or just a skewed one?
    #
    # Neither a valley depth nor Ashman's D can answer this. Valley depth reads
    # 0.00 whenever the failure mode is much smaller than the good one (the
    # density never dips below the lower peak). Ashman's D is worse than
    # useless here: fitting two Gaussians to a plain log-normal always
    # "separates" them, and on this dataset a pure EBSD grain-size distribution
    # -- which contains no failures whatsoever -- scored D = 2.17 against real
    # data's 2.5, so a D-based gate cuts distributions that are merely skewed.
    #
    # The discriminating question is whether the smoothed density has a real
    # local minimum between two PROMINENT maxima. A skewed unimodal density is
    # monotonic past its peak and has none; a genuine second population makes
    # one. Prominence, not height, is what separates a real bump from noise.
    sep = float(abs(mu2[1] - mu2[0]))
    ashman_d = float(np.sqrt(2.0) * sep / np.sqrt(sd2[0] ** 2 + sd2[1] ** 2))

    h, e = np.histogram(x, bins=smooth_bins)
    hs = gaussian_filter1d(h.astype(float), smooth_sigma)
    centres = 0.5 * (e[:-1] + e[1:])
    pk, _ = find_peaks(hs, prominence=min_peak_prominence * hs.max())
    modes = (float(mu2[0]), float(mu2[1]))

    # The prominent-peak count decides WHETHER to cut. WHERE to cut comes from
    # the mixture's equal-posterior crossing, not from a histogram valley.
    #
    # Four defensible ways of "finding the valley" were measured on the same
    # shade_LSHR run and produced 160, 175, 219 and 425 µm -- the last at purity
    # 0.852, worse than not cutting at all. Valley-finding is too sensitive to
    # binning and to which pair of maxima you bracket to be shipped. The mixture
    # crossing is stable across both runs (131 µm and 160 µm, purity 0.996 and
    # 0.993, against supervised optima of 195 µm and 223 µm): it keeps fewer
    # grains than optimal, which is the right direction to err.
    thr = _posterior_crossing(mu2, sd2, w2)

    # reported as a diagnostic only; it reads ~0 whenever the failure mode is
    # much smaller than the good one, which is the usual case
    valley_depth = 0.0
    if len(pk) >= 2:
        top = pk[np.argsort(hs[pk])[-2:]]
        a, b = int(min(top)), int(max(top))
        j = a + int(np.argmin(hs[a:b + 1]))
        peak_h = min(hs[a], hs[b])
        valley_depth = float(1.0 - hs[j] / peak_h) if peak_h > 0 else 0.0

    reasons = []
    if d_bic < min_delta_bic:
        reasons.append(f"ΔBIC {d_bic:.1f} < {min_delta_bic}")

    if len(pk) < 2:
        reasons.append("density has fewer than two prominent maxima "
                       "(unimodal or merely skewed)")
    if ashman_d < min_ashman_d:
        reasons.append(f"Ashman D {ashman_d:.2f} < {min_ashman_d}")
    if minority < min_minority_frac:
        reasons.append(f"minority population {minority:.3%} < {min_minority_frac:.0%}")

    if reasons:
        return QualityGate(
            None, method="antimode", bimodality=float(d_bic),
            ashman_d=ashman_d, valley_depth=valley_depth,
            reason="declined: " + "; ".join(reasons) + " — keep all grains",
            **{**nil, "modes": modes})

    threshold = 10.0 ** thr if log_transform else thr
    keep = v <= threshold if lower_is_better else v >= threshold
    return QualityGate(
        threshold=float(threshold), lower_is_better=lower_is_better,
        method="antimode", reason="bimodal; cut at the histogram antimode",
        n_total=n, n_kept=int(keep.sum()), bimodality=float(d_bic),
        ashman_d=ashman_d, valley_depth=valley_depth, modes=modes)
