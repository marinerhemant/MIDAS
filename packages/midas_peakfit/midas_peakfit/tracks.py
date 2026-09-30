"""Link spot detections across frames into features, and test them on raw counts.

A stationary crystallite, a hot pixel or a detector column is detected again and
again. Those repeats are ONE feature: counting them separately makes any
statistic built on the detection list look far more significant than it is
(negative controls start to pass). This module

* merges detections within ``radius`` px into features (lifetime, n_det, medians),
* measures each feature on RAW counts in two windows (e.g. before / after an
  event), with a Poisson test on aperture counts against a local background -- a
  "present before, absent after" claim must hold on photons, not only on the
  absence of a thresholded detection,
* flags features that sit at the same detector pixel as features from unrelated
  data (detector-fixed artefacts).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import poisson

__all__ = ["merge_detections", "aperture_rates", "WindowTest", "window_test",
           "fixed_pixel_flags"]


def merge_detections(frame: np.ndarray, row: np.ndarray, col: np.ndarray,
                     radius: float = 3.0, min_det: int = 1,
                     values: dict | None = None) -> dict:
    """Merge detections closer than ``radius`` px (any frames) into features.

    Returns a dict of arrays, one entry per feature: ``row``, ``col`` (medians),
    ``first``, ``last`` (frame), ``n_det``, ``label`` (per detection), and the
    median of every array in ``values`` (e.g. ``{"d": d_spacing}``).
    Transitive merging is intended: a slowly drifting spot stays one feature.
    """
    frame, row, col = (np.asarray(x, float) for x in (frame, row, col))
    n = len(row)
    if n == 0:
        out = {k: np.array([]) for k in ("row", "col", "first", "last", "n_det")}
        out["label"] = np.array([], int)
        for k in (values or {}):
            out[k] = np.array([])
        return out
    pairs = cKDTree(np.c_[row, col]).query_pairs(radius, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) \
        else coo_matrix((n, n))
    nf, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab, minlength=nf)
    keep = np.flatnonzero(cnt >= min_det)
    order = np.argsort(lab, kind="stable")
    bounds = np.r_[0, np.cumsum(cnt)]
    out = {"row": [], "col": [], "first": [], "last": [], "n_det": []}
    vals = {k: [] for k in (values or {})}
    for k in keep:
        m = order[bounds[k]:bounds[k + 1]]
        out["row"].append(np.median(row[m])); out["col"].append(np.median(col[m]))
        out["first"].append(frame[m].min()); out["last"].append(frame[m].max())
        out["n_det"].append(len(m))
        for key, arr in (values or {}).items():
            vals[key].append(np.median(np.asarray(arr)[m]))
    res = {k: np.asarray(v, float) for k, v in out.items()}
    res["n_det"] = res["n_det"].astype(int)
    res.update({k: np.asarray(v, float) for k, v in vals.items()})
    remap = -np.ones(nf, int)
    remap[keep] = np.arange(len(keep))
    res["label"] = remap[lab]
    return res


def _disk(radius):
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx <= radius * radius).astype(np.float64)


def aperture_rates(sum_img: np.ndarray, ok: np.ndarray, bkg_img: np.ndarray,
                   row: np.ndarray, col: np.ndarray, radius: float = 2.5
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Aperture counts N and expected background mu at each (row, col).

    ``sum_img`` is the sum of raw frames over a window and ``bkg_img`` the matching
    summed background model (same number of frames). Invalid pixels are excluded
    from both.
    """
    k = _disk(radius)
    N = ndimage.convolve(np.where(ok, sum_img, 0.0), k, mode="constant")
    mu = ndimage.convolve(np.where(ok, bkg_img, 0.0), k, mode="constant")
    r = np.clip(np.rint(row).astype(int), 0, sum_img.shape[0] - 1)
    c = np.clip(np.rint(col).astype(int), 0, sum_img.shape[1] - 1)
    return N[r, c], mu[r, c]


@dataclass
class WindowTest:
    """Raw-count comparison of features between window A and window B."""
    net_rate_a: np.ndarray      # (N - mu) / n_frames, window A
    net_rate_b: np.ndarray
    p_present_a: np.ndarray     # P(counts >= N_a | background only), window A
    p_present_b: np.ndarray
    p_drop: np.ndarray          # P(B at least as low as observed | rate unchanged from A)
    present_a: np.ndarray       # bool
    absent_b: np.ndarray        # bool: not significant in B AND significantly below A


def window_test(sum_a, bkg_a, n_a, sum_b, bkg_b, n_b, ok, row, col, radius=2.5,
                alpha_present=1e-4, alpha_drop=1e-3) -> WindowTest:
    """Test each feature for "present in A" and "absent in B" on raw counts.

    present_a: aperture counts in A exceed background at ``alpha_present``.
    absent_b: counts in B are NOT significant at ``alpha_present`` AND lower than
    expected if the feature kept its A rate (Poisson, ``alpha_drop``).
    """
    Na, mua = aperture_rates(sum_a, ok, bkg_a, row, col, radius)
    Nb, mub = aperture_rates(sum_b, ok, bkg_b, row, col, radius)
    pa = poisson.sf(np.rint(Na) - 1, np.maximum(mua, 1e-9))
    pb = poisson.sf(np.rint(Nb) - 1, np.maximum(mub, 1e-9))
    net_a = np.maximum(Na - mua, 0.0) / max(n_a, 1)
    expected_b = mub + net_a * n_b
    p_drop = poisson.cdf(np.rint(Nb), np.maximum(expected_b, 1e-9))
    present_a = pa < alpha_present
    absent_b = (pb >= alpha_present) & (p_drop < alpha_drop)
    return WindowTest((Na - mua) / max(n_a, 1), (Nb - mub) / max(n_b, 1), pa, pb, p_drop,
                      present_a, absent_b)


def fixed_pixel_flags(row, col, other_rows, other_cols, radius: float = 3.0) -> np.ndarray:
    """True where a feature coincides (within ``radius`` px) with any position in an
    unrelated dataset (another sample position, a background run). A diffraction
    spot from a stationary crystallite cannot sit at the same pixel in unrelated data;
    a detector defect does."""
    if len(other_rows) == 0 or len(row) == 0:
        return np.zeros(len(row), bool)
    t = cKDTree(np.c_[other_rows, other_cols])
    d, _ = t.query(np.c_[row, col], k=1)
    return d <= radius
