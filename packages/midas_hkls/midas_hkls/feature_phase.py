"""Does a set of spot d-spacings contain a population of candidate phase X?

Companion to :mod:`midas_hkls.phase_id` for sparse still-frame data: the input
is a list of *features* (one entry per distinct spot, repeats already merged),
most of which may be noise or unrelated. The question is not "which phase are
these d-values" but "do more of them sit on X's lines than chance allows".

What this module enforces (each rule answers a failure seen on real data):

* **Lines from the structure, basis absences included** -- reflections whose
  normalised |F|^2 is ~0 are dropped (:func:`allowed_d_lines`).
* **Count and families, never one line** -- the statistic is the number of
  features within tolerance of a line (M) *and* the number of distinct lines hit
  (L). One line under a free scale identifies nothing.
* **Line density is reported** -- ``chance_coverage`` is the fraction of the
  d-window within tolerance of some line at the best scale; a dense candidate
  matches anything.
* **The null is part of the test** -- surrogate feature sets are drawn from a
  caller-supplied sampler (e.g. valid detector pixels, matrix bands removed) and
  pushed through the identical max-over-scale statistic.
* **Negative controls must fail** -- :func:`feature_phase_test` refuses to call
  a pass if any control candidate passes.
* **Scale at the edge is flagged** -- a best scale on the window edge means the
  true optimum is outside; report the cell, not the name.
* **Look-elsewhere** -- :func:`cell_scan` compares the observed peak against the
  null distribution of the MAXIMUM over the whole scan.

A pass identifies a *cell and pattern*, not a chemistry: phases sharing a
structure type differ only in cell size, which a free scale partly absorbs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

__all__ = ["allowed_d_lines", "match_count", "pixel_weighted_sampler", "FeatureTestRow",
           "feature_phase_test", "cell_scan", "d_clusters", "family_count"]


def allowed_d_lines(crystal, d_min: float, d_max: float, f2_min: float = 1e-4) -> np.ndarray:
    """Distinct allowed d-spacings of ``crystal`` in [d_min, d_max], dropping
    reflections with normalised |F|^2 <= ``f2_min`` (basis absences)."""
    from .hkl_gen import generate_hkls
    from .structure_factor import f2_normalised
    refl = generate_hkls(crystal.space_group, crystal.lattice, d_min=d_min, d_max=d_max)
    if not refl:
        return np.array([])
    if crystal.atoms:
        w = f2_normalised(crystal.lattice, crystal.space_group, crystal.atoms,
                          [(r.h, r.k, r.l) for r in refl])
    else:
        w = np.ones(len(refl))
    d = np.array([r.d_spacing for r in refl])[np.asarray(w) > f2_min]
    return np.unique(np.round(d, 6))


def match_count(d_feat: np.ndarray, tol, d_lines: np.ndarray,
                scales: np.ndarray) -> tuple[int, int, float, bool]:
    """(M, L, s_best, at_edge): max over ``scales`` of features within relative
    ``tol`` (scalar or per feature) of ``s * d_line``.

    The count is flat over a plateau of scales; ``s_best`` is the centre of the
    plateau block holding the maximum, and ``at_edge`` is True when that block
    touches either end of ``scales`` -- the optimum may lie outside the window
    (a population just past the edge otherwise reports an interior s_best).
    """
    d_feat = np.asarray(d_feat, float)
    scales = np.asarray(scales, float)
    if d_feat.size == 0 or len(d_lines) == 0:
        return 0, 0, float("nan"), False
    tol = np.broadcast_to(np.asarray(tol, float), d_feat.shape)
    ll = np.log(np.sort(d_lines))
    lf = np.log(d_feat)
    lt = np.log1p(tol)
    counts, fams = [], []
    for s in scales:
        x = lf - np.log(s)
        j = np.clip(np.searchsorted(ll, x), 1, len(ll) - 1)
        nearest = np.where(np.abs(x - ll[j - 1]) < np.abs(x - ll[j]), j - 1, j)
        hit = np.abs(x - ll[nearest]) < lt
        counts.append(int(hit.sum()))
        fams.append(len(np.unique(nearest[hit])))
    counts = np.array(counts)
    M = int(counts.max())
    i0 = int(np.argmax(counts))
    i1 = i0
    while i1 + 1 < len(counts) and counts[i1 + 1] == M:
        i1 += 1
    at_edge = M > 0 and len(scales) > 1 and (i0 == 0 or i1 == len(scales) - 1)
    return M, int(fams[i0]), float(scales[(i0 + i1) // 2]), bool(at_edge)


def d_clusters(d_feat: np.ndarray, tol) -> tuple[np.ndarray, np.ndarray]:
    """Group features whose d agree within relative ``tol`` (single linkage on log d).

    Several crystallites of one phase -- or anything else that diffracts -- put several
    spots on the same ring. Those spots are one d-value, not independent evidence for a
    cell. Returns (median d per cluster, cluster label per feature)."""
    d = np.asarray(d_feat, float)
    if d.size == 0:
        return d, np.zeros(0, int)
    t = float(np.median(np.atleast_1d(tol)))
    order = np.argsort(d)
    ld = np.log(d[order])
    brk = np.r_[0, np.cumsum(np.diff(ld) > np.log1p(t))]
    labels = np.empty(d.size, int)
    labels[order] = brk
    reps = np.array([np.median(d[labels == k]) for k in range(int(brk[-1]) + 1)])
    return reps, labels


def family_count(d: np.ndarray, tol, d_lines: np.ndarray, scales: np.ndarray) -> tuple[int, float]:
    """(F, s): the most DISTINCT lines hit by ``d`` (within relative ``tol``) over ``scales``,
    and the centre of the scale plateau where that maximum holds. Counts lines, not
    d-values: two d-values either side of one line are one line."""
    d = np.asarray(d, float)
    scales = np.asarray(scales, float)
    if d.size == 0 or len(d_lines) == 0:
        return 0, float("nan")
    ll = np.log(np.sort(d_lines))
    lt = np.log1p(float(np.median(np.atleast_1d(tol))))
    f = []
    for s in scales:
        x = np.log(d) - np.log(s)
        j = np.clip(np.searchsorted(ll, x), 1, len(ll) - 1)
        nearest = np.where(np.abs(x - ll[j - 1]) < np.abs(x - ll[j]), j - 1, j)
        f.append(len(np.unique(nearest[np.abs(x - ll[nearest]) < lt])))
    f = np.array(f)
    i0 = int(np.argmax(f))
    i1 = i0
    while i1 + 1 < len(f) and f[i1 + 1] == f[i0]:
        i1 += 1
    return int(f[i0]), float(scales[(i0 + i1) // 2])


def pixel_weighted_sampler(tth_valid_deg: np.ndarray, lam: float, bin_deg: float = 0.01,
                           exclude: Optional[Callable[[np.ndarray], np.ndarray]] = None
                           ) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Sampler of surrogate d-values with the detector's 2-theta coverage.

    ``tth_valid_deg`` holds the scattering angle of every pixel where a feature
    could have been reported; ``exclude(d) -> bool`` removes d-values the
    selection could never have produced (e.g. inside matrix bands).
    """
    t = np.asarray(tth_valid_deg, float)
    t = t[np.isfinite(t)]
    h, e = np.histogram(t, bins=np.arange(t.min(), t.max() + bin_deg, bin_deg))
    cen, p = (e[:-1] + e[1:]) / 2, h / h.sum()

    def draw(n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty(0)
        while out.size < n:
            m = 4 * n + 100
            tt = rng.choice(cen, m, p=p) + rng.uniform(-bin_deg / 2, bin_deg / 2, m)
            dd = lam / (2 * np.sin(np.radians(tt / 2)))
            if exclude is not None:
                dd = dd[~exclude(dd)]
            out = np.r_[out, dd]
        return out[:n]
    return draw


@dataclass
class FeatureTestRow:
    name: str
    n_lines: int
    chance_coverage: float
    M: int
    L: int
    s_best: float
    at_edge: bool
    null_mean: float
    null_p99: float
    p: float                      # spot-count p (descriptive: several spots per ring inflate it)
    passed: bool
    control: bool = False
    K: int = 0                    # distinct d-values (clusters) among the features
    F: int = 0                    # distinct lines hit by the d-clusters (max over scale)
    s_family: float = float("nan")  # scale where F is reached (s_best is the spot-count scale)
    F_null_mean: float = float("nan")
    p_family: float = float("nan")   # the test: F against K independent surrogate d-values


def _coverage(d_lines, tol, s, d_min, d_max, n=20000):
    grid = np.exp(np.linspace(np.log(d_min), np.log(d_max), n))
    return match_count(grid, float(np.median(np.atleast_1d(tol))), d_lines, np.array([s]))[0] / n


def feature_phase_test(d_feat: np.ndarray, tol, candidates: Dict[str, np.ndarray],
                       sampler: Callable[[int, np.random.Generator], np.ndarray], *,
                       controls: Optional[Dict[str, np.ndarray]] = None,
                       scale_range=(1.0, 1.025), scale_step=0.0005, n_null=2000,
                       alpha=0.05, min_M=5, min_L=3, min_ratio=3.0, seed=0) -> dict:
    """Test each candidate (and each control) with the same statistics and null.

    ``candidates`` / ``controls`` map a label to its allowed d-lines. Features are first
    grouped into distinct d-values (:func:`d_clusters`): many spots on one ring are one
    piece of evidence. The TEST is family-level: F = distinct lines hit by the clusters
    (max over scale, at its own scale ``s_family``) against the same statistic on K independent surrogate d-values
    (K = number of clusters), p_family. The spot count M and its p are reported but do
    not decide: they reward several spots per ring. The alpha is Bonferroni-corrected
    over the candidates. A candidate passes when p_family < alpha, F >= min_L and
    M >= min_M, M >= min_ratio * null mean -- and the run is only readable
    (``valid``) if no control passes.
    """
    rng = np.random.default_rng(seed)
    d_feat = np.asarray(d_feat, float)
    scales = np.arange(scale_range[0], scale_range[1] + 1e-12, scale_step)
    nulls = [sampler(len(d_feat), rng) for _ in range(n_null)]
    reps, _ = d_clusters(d_feat, tol)
    tol_c = float(np.median(np.atleast_1d(tol)))
    nulls_c = [sampler(len(reps), rng) for _ in range(n_null)]
    a_corr = alpha / max(len(candidates), 1)
    d_min, d_max = (d_feat.min(), d_feat.max()) if d_feat.size else (1.0, 2.0)
    rows = []
    for is_ctrl, group in ((False, candidates), (True, controls or {})):
        for name, lines in group.items():
            M, L, s, edge = match_count(d_feat, tol, lines, scales)
            mn = np.array([match_count(x, tol, lines, scales)[0] for x in nulls])
            p = (1 + int((mn >= M).sum())) / (1 + n_null)
            F, s_f = family_count(reps, tol_c, lines, scales)
            fn = np.array([family_count(x, tol_c, lines, scales)[0] for x in nulls_c])
            pf = (1 + int((fn >= F).sum())) / (1 + n_null)
            ok = bool(pf < a_corr and F >= min_L and M >= min_M and M >= min_ratio * mn.mean())
            rows.append(FeatureTestRow(name, int(len(lines)),
                                       float(_coverage(lines, tol, s if np.isfinite(s) else 1.0, d_min, d_max)),
                                       M, L, s, edge, float(mn.mean()), float(np.percentile(mn, 99)),
                                       float(p), ok, is_ctrl, K=int(len(reps)), F=int(F), s_family=s_f,
                                       F_null_mean=float(fn.mean()), p_family=float(pf)))
    valid = not any(r.passed for r in rows if r.control)
    return dict(rows=rows, alpha_corrected=a_corr, valid=valid, n_features=int(d_feat.size),
                note=None if valid else "a negative control passed: the run is not readable")


def cell_scan(d_feat: np.ndarray, tol, d_lines_ref: np.ndarray, a_ref: float,
              a_grid: np.ndarray, sampler, n_null=400, seed=0) -> dict:
    """Match count vs cell size (lines rescaled by a/a_ref, no free scale), with the
    null distribution of the MAXIMUM over the whole grid (look-elsewhere)."""
    rng = np.random.default_rng(seed)
    one = np.array([1.0])
    obs = np.array([match_count(d_feat, tol, d_lines_ref * a / a_ref, one)[0] for a in a_grid])
    maxes = np.empty(n_null)
    for k in range(n_null):
        x = sampler(len(d_feat), rng)
        maxes[k] = max(match_count(x, tol, d_lines_ref * a / a_ref, one)[0] for a in a_grid)
    i = int(np.argmax(obs))
    return dict(a_grid=a_grid, observed=obs, a_best=float(a_grid[i]), M_best=int(obs[i]),
                null_max_mean=float(maxes.mean()), null_max_p99=float(np.percentile(maxes, 99)),
                p_look_elsewhere=float((1 + (maxes >= obs[i]).sum()) / (1 + n_null)))
