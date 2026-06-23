"""Fully-automated calibrant-image seeding (port of AutoCalibrateZarr recipe).

Recovers ($\\BC$, $\\Lsd$) from a calibrant image with NO operator
input beyond the material identity and detector pixel size.  This is
a direct PyTorch-friendly port of the production-tested algorithm in
``utils/AutoCalibrateZarr.py`` (see the MIDAS repository), preserving
its parameter choices and numerical behaviour.

Pipeline:
    1.  Median-filter background subtraction (kernel 101, 5 iters).
    2.  Threshold = 100 · (1 + std(corr) // 100).
    3.  Connected-component labelling on the thresholded image.
    4.  Per-region chord-bisector → median across regions = $\\BC$.
        4× downsampling speeds labelling without losing thin arcs
        (block-max preserves arc continuity).
    5.  Per-region mean radial distance from $\\BC$ → ring radii (px).
    6.  Multi-hypothesis Lsd matching: try assigning det[k] → sim[j]
        for k in {0,1,2}; pick the (k, j) with the most consistent
        matches (within 5 % rel.\\ err.) across all detected rings.

Returns BC in MIDAS convention: BC = (BC_y, BC_z) where BC_y is the
column-axis position, BC_z is the row-axis position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class AutoSeedResult:
    BC_y: float
    BC_z: float
    Lsd_um: float
    n_arcs: int
    n_rings_matched: int
    detected_radii_px: np.ndarray
    threshold_used: float

    def __str__(self) -> str:
        return (f"AutoSeedResult(BC=({self.BC_y:.2f},{self.BC_z:.2f}), "
                f"Lsd={self.Lsd_um:.0f} µm, "
                f"{self.n_arcs} arcs, "
                f"{self.n_rings_matched} rings matched, "
                f"thr={self.threshold_used:.1f})")


# ----------------------------------------------------------------------
# Step 1: median filter (mirror AutoCalibrateZarr._safe_median_filter)
# ----------------------------------------------------------------------

# diplib's median filter is 10–30× faster on big images; use when available.
try:
    import diplib as _DIP_BG
    _HAVE_DIPLIB_BG = True
except ImportError:
    _DIP_BG = None
    _HAVE_DIPLIB_BG = False


def _safe_median_filter(data: np.ndarray, kernel_size: int = 101,
                          n_iters: int = 5) -> np.ndarray:
    """Iterated median-filter background estimator.

    Default: legacy scipy 1-D × 2 × n_iters scheme.  Set
    ``MIDAS_USE_DIPLIB=1`` to use diplib's faster 2-D median filter.
    """
    out = data.astype(np.float64)
    import os as _os
    if _HAVE_DIPLIB_BG and _os.environ.get("MIDAS_USE_DIPLIB", "") == "1":
        try:
            return np.asarray(_DIP_BG.MedianFilter(
                out, _DIP_BG.Kernel(int(kernel_size), "rectangular")))
        except Exception:
            pass
    for _ in range(n_iters):
        out = ndimage.median_filter(out, size=(1, kernel_size), mode="reflect")
        out = ndimage.median_filter(out, size=(kernel_size, 1), mode="reflect")
    return out


# ----------------------------------------------------------------------
# Step 4: chord-bisector per arc (mirror _find_center_point)
# ----------------------------------------------------------------------

def _find_center_point(coords: np.ndarray,
                       bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Recover the center of an arc by chord bisection.

    coords : (N, 2) array of (row, col) — pixels belonging to the arc.
    bbox   : (row_min, col_min, row_max, col_max).
    Returns (row, col) of the recovered center, or (-1, -1) on failure.
    """
    edge_indices = np.where(coords[:, 0] == bbox[0])[0]
    if len(edge_indices) == 0:
        return np.array([-1.0, -1.0])
    edgecoorda = coords[edge_indices[len(edge_indices) // 2]]
    distances = np.linalg.norm(coords - edgecoorda, axis=1)
    furthest_idx = int(np.argmax(distances))
    edgecoordb = coords[furthest_idx]
    arcLen = float(distances[furthest_idx]) / 2.0
    candidate_indices = np.where(np.abs(distances - arcLen) < 2.0)[0]
    if len(candidate_indices) == 0:
        return np.array([-1.0, -1.0])
    candidatea = coords[candidate_indices[len(candidate_indices) // 2]]
    candidateb = candidatea
    midpointa = (edgecoorda + candidatea) / 2.0
    midpointb = (edgecoordb + candidateb) / 2.0
    x1, y1 = edgecoorda; x2, y2 = candidatea
    x3, y3 = candidateb; x4, y4 = edgecoordb
    x5, y5 = midpointa; x6, y6 = midpointb
    if y4 == y3 or y2 == y1:
        return np.array([-1.0, -1.0])
    m1 = (x1 - x2) / (y2 - y1)
    m2 = (x3 - x4) / (y4 - y3)
    if abs(m1 - m2) < 1e-10:
        return np.array([-1.0, -1.0])
    x = (y6 - y5 + m1 * x5 - m2 * x6) / (m1 - m2)
    y = m1 * (x - x5) + y5
    return np.array([x, y])


# ----------------------------------------------------------------------
# Step 4: detect_beam_center_optimized
# ----------------------------------------------------------------------

def _detect_beam_center(thresh: np.ndarray, min_area: int) -> np.ndarray:
    """Median chord-bisector center across thresholded regions.
    Returns (row, col) which in MIDAS convention is (BC_z, BC_y).
    """
    scale = 4
    h, w = thresh.shape
    sh, sw = h // scale, w // scale
    ds = thresh[:sh*scale, :sw*scale].reshape(sh, scale, sw, scale).max(axis=(1, 3))
    ds_labels, ds_nlabels = ndimage.label(ds)
    if ds_nlabels == 0:
        return np.array([0.0, 0.0])

    from skimage import measure as _measure
    ds_props = _measure.regionprops(ds_labels)
    ds_minArea = max(1, min_area // (scale * scale))

    all_centers = []
    for rp in ds_props:
        if rp.area < ds_minArea:
            continue
        coords = rp.coords
        bbox = (rp.bbox[0], rp.bbox[1], rp.bbox[2], rp.bbox[3])
        c = _find_center_point(coords, bbox)
        if c[0] >= 0 and c[1] >= 0:
            all_centers.append(c)

    if not all_centers:
        return np.array([0.0, 0.0])
    centers_array = np.array(all_centers) * scale
    return np.array([np.median(centers_array[:, 0]),
                     np.median(centers_array[:, 1])])


# ----------------------------------------------------------------------
# Step 5: detect_ring_radii
# ----------------------------------------------------------------------

def _detect_ring_radii(props, bc_rowcol: np.ndarray,
                       min_area: int, dedup_px: float = 20.0) -> np.ndarray:
    """Mean radial distance per region, deduplicated by ``dedup_px``."""
    rads = []
    for rp in props:
        if rp.area < min_area:
            continue
        coords = rp.coords
        rad = float(np.mean(np.linalg.norm(
            np.transpose(coords) - bc_rowcol[:, None], axis=0)))
        toAdd = True
        for existing in rads:
            if abs(existing - rad) < dedup_px:
                toAdd = False; break
        if toAdd:
            rads.append(rad)
    return np.sort(np.array(rads, dtype=np.float64)) if rads else np.array([])


# ----------------------------------------------------------------------
# Step 6: multi-hypothesis Lsd matching
# ----------------------------------------------------------------------

def _estimate_lsd(rads: np.ndarray, sim_rads: np.ndarray,
                  *, first_ring: int = 1,
                  initial_lsd: float = 1_000_000.0,
                  max_ring: int = 0,
                  rel_tol: float = 0.05,
                  lsd_window: Optional[float] = None) -> Tuple[float, int]:
    """Multi-hypothesis ring-matching distance estimator.

    Try assigning detected ring k → simulated ring j for k in {0,1,2}.
    For each (k, j), compute trial Lsd and count how many other detected
    rings match a simulated ring within ``rel_tol``.  Pick (k, j) with
    most matches (ties → lowest std of per-match Lsd estimates).
    Returns (best_lsd, n_matches).

    ``lsd_window`` : when set, ``initial_lsd`` is treated as a trustworthy
    prior — only hypotheses whose recovered Lsd falls in
    ``[initial_lsd / lsd_window, initial_lsd * lsd_window]`` are considered,
    and ties are broken toward the hypothesis closest to ``initial_lsd``.
    This rescues weak data where only 2 rings are detected and the
    blind ratio matcher would otherwise lock onto a spurious assignment
    (e.g. the HYDRA off-panel GE panels: true 2455 mm vs spurious 770 mm).
    Leave ``None`` for the blind from-scratch case.
    """
    if len(rads) == 0:
        return initial_lsd, 0
    n_sim = len(sim_rads) if max_ring <= 0 else min(max_ring, len(sim_rads))
    use_prior = lsd_window is not None and lsd_window > 1.0
    lo_lsd = initial_lsd / lsd_window if use_prior else 0.0
    hi_lsd = initial_lsd * lsd_window if use_prior else np.inf
    best_lsd = initial_lsd
    # Scoring tuple (lex ordering, top-down):
    #   1. ``det0_match``   — does the innermost detected ring participate
    #      in the match set?  At a spurious half-Lsd basin the first detected
    #      ring often has NO sim ring within rel_tol; at the correct basin it
    #      maps to sim[0] or sim[1].  (Was the trigger for Pilatus2M_16IDB_2024
    #      flipping from -49 % to ≤ 1 % in blind seeding.)
    #   2. ``n_consistent`` — number of matches whose implied Lsd lies within
    #      ``consistency_tol`` of the hypothesis' median implied Lsd.
    #   3. ``lsd_median``   — prefer LARGER Lsd among ties.  Justification:
    #      a smaller-Lsd hypothesis packs more simulated rings inside the
    #      detected radial range, so it gets more "free" matches by accident.
    #      Without this bias the seed locks onto half-Lsd shadow basins on
    #      Pilatus Nov2025 (truth 343 mm, spurious 284 mm both n_consistent=5).
    #   4. ``tie``          — implied-Lsd RMS / median (minimised).
    #   5. ``matches``      — total within-rel_tol match count.
    consistency_tol = 0.02
    best_score = (-1, -1, -1.0, 1e9, -1)
    max_det_start = min(3, len(rads))
    rad0 = float(rads[0])  # innermost detected ring
    for det_start in range(max_det_start):
        for hyp in range(first_ring - 1, n_sim):
            trial_lsd = initial_lsd * rads[det_start] / sim_rads[hyp]
            scale = trial_lsd / initial_lsd
            trial_sim_px = sim_rads * scale
            matches = 0
            lsds_this = []
            det0_matched = False
            for k, det_rad in enumerate(rads):
                diffs = np.abs(trial_sim_px[:n_sim] - det_rad)
                best_j = int(np.argmin(diffs))
                if diffs[best_j] / max(det_rad, 1.0) < rel_tol:
                    matches += 1
                    lsds_this.append(initial_lsd * det_rad / sim_rads[best_j])
                    if k == 0:
                        det0_matched = True
            if not lsds_this:
                continue
            lsd_median = float(np.median(lsds_this))
            # Reject hypotheses outside the trusted window entirely.
            if use_prior and not (lo_lsd <= lsd_median <= hi_lsd):
                continue
            n_consistent = int(sum(1 for lsd_i in lsds_this
                                    if abs(lsd_i - lsd_median) / max(lsd_median, 1.0)
                                       < consistency_tol))
            std_lsd = (float(np.std(lsds_this)) if len(lsds_this) >= 2 else 0.0)
            if use_prior:
                tie = abs(lsd_median - initial_lsd) / initial_lsd
            else:
                tie = std_lsd / max(lsd_median, 1.0)
            score = (int(det0_matched), n_consistent, lsd_median, tie, matches)
            # Lex compare: det0_matched (max) > n_consistent (max) > lsd_median
            # (max) > tie (min) > matches (max).
            if (score[0] > best_score[0] or
                (score[0] == best_score[0] and score[1] > best_score[1]) or
                (score[0] == best_score[0] and score[1] == best_score[1]
                 and score[2] > best_score[2]) or
                (score[0] == best_score[0] and score[1] == best_score[1]
                 and score[2] == best_score[2] and score[3] < best_score[3]) or
                (score[0] == best_score[0] and score[1] == best_score[1]
                 and score[2] == best_score[2] and score[3] == best_score[3]
                 and score[4] > best_score[4])):
                best_score = score
                best_lsd = lsd_median
    return best_lsd, best_score[4] if best_score[4] >= 0 else 0


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def auto_seed_calibrant(
    image: np.ndarray,
    *,
    sim_radii_px: np.ndarray,
    initial_lsd_um: float = 1_000_000.0,
    min_area: int = 300,
    median_kernel: int = 101,
    median_iters: int = 5,
    threshold: Optional[float] = None,
    first_ring: int = 1,
    max_ring: int = 15,
    skip_median: bool = False,
    mask: Optional[np.ndarray] = None,
    lsd_window: Optional[float] = None,
) -> AutoSeedResult:
    """Fully-automated $\\BC$ + Lsd seed from a calibrant image.

    Parameters
    ----------
    image : 2-D float array
        Raw or dark-subtracted calibrant image (MIDAS convention:
        image.shape = (n_z, n_y), image[z_row, y_col]).
    sim_radii_px : np.ndarray
        Simulated ring radii in pixels at the ``initial_lsd_um`` seed.
        Caller computes from the calibrant's d-spacings, the X-ray
        wavelength, and a nominal $\\Lsd$.
    initial_lsd_um : float
        Nominal Lsd used to compute ``sim_radii_px``.  Only an
        anchor — the multi-hypothesis matcher recovers absolute Lsd.
    min_area : int
        Minimum connected-component area (pixels) to keep as a ring
        arc candidate.  Default 300 matches AutoCalibrateZarr.
    threshold : float or None
        Pixel-intensity cutoff after background subtraction.  If
        None, uses ``100 * (1 + std(corr) // 100)`` (AutoCalibrateZarr
        recipe).
    skip_median : bool
        If True, skip background subtraction (image is assumed
        pre-subtracted, e.g.\\ via dark frame).
    """
    # Step 1-2: background subtraction + threshold
    if skip_median:
        corr = image.astype(np.float64)
    else:
        bg = _safe_median_filter(image, kernel_size=median_kernel,
                                  n_iters=median_iters)
        corr = image.astype(np.float64) - bg

    from skimage import measure as _measure
    bool_mask = np.asarray(mask, dtype=bool) if mask is not None else None

    def _threshold_pass(thr_val: float):
        """Apply one threshold, label, area-filter, and report keepers."""
        corr_thr_ = np.where(corr < thr_val, 0.0, corr)
        if bool_mask is not None:
            corr_thr_ = np.where(bool_mask, 0.0, corr_thr_)
        bin_ = (corr_thr_ > 0).astype(np.uint8) * 255
        labels_, nlab_ = _measure.label(bin_, return_num=True)
        if nlab_ == 0:
            return bin_, labels_, nlab_, [], 0
        props_ = _measure.regionprops(labels_)
        keep_ = np.zeros(nlab_ + 1, dtype=bool)
        for rp in props_:
            if rp.area >= min_area:
                keep_[rp.label] = True
        bin_[~keep_[labels_]] = 0
        return bin_, labels_, nlab_, props_, int(np.sum(keep_))

    # Adaptive threshold: start from the user value (or the legacy AutoCalibrateZarr
    # formula `100·(1+std//100)`); if that yields zero arcs ≥ min_area, retry at
    # progressively lower thresholds.  This rescues weak-signal images (e.g.
    # PerkinElmer 13BMD CeO2 where rings sit ~200 above background but the
    # legacy formula floors at 100, demanding rings > 1000).
    auto_thr0 = 100.0 * (1.0 + np.std(corr) // 100.0)
    if threshold is not None:
        thr_candidates = [float(threshold)]
    else:
        # Probe the legacy threshold first, then back off in halves down to a
        # noise-floor (5× MAD) and a hard floor of 10.  This widens the
        # acceptance from "thresholds that just happen to match the 100 grid"
        # to "any threshold where rings survive the min_area cut".
        mad = float(np.median(np.abs(corr - np.median(corr))) * 1.4826)
        floor_mad = max(5.0 * mad, 10.0)
        thr_candidates = [auto_thr0, auto_thr0 / 2.0, auto_thr0 / 4.0,
                          auto_thr0 / 8.0, max(floor_mad, 30.0), floor_mad]
        # Dedup while preserving order; drop trivially-low (≤0) entries
        seen = set(); _tmp = []
        for t in thr_candidates:
            tr = round(float(t), 4)
            if tr > 0 and tr not in seen:
                seen.add(tr); _tmp.append(tr)
        thr_candidates = _tmp

    binary = None; labels = None; nlabels = 0; props = []; n_arcs = 0
    thr = thr_candidates[0] if thr_candidates else 0.0
    for thr_try in thr_candidates:
        binary, labels, nlabels, props, n_arcs = _threshold_pass(thr_try)
        if n_arcs >= 1:
            thr = thr_try
            break
    if nlabels == 0:
        raise RuntimeError(
            f"auto_seed_calibrant: no connected components above thresholds "
            f"{thr_candidates}; image may be empty or dark not subtracted"
        )

    # Step 4: BC via chord bisector (returns (row, col) = (BC_z, BC_y))
    bc_rowcol = _detect_beam_center(binary, min_area)
    if not (np.isfinite(bc_rowcol).all() and bc_rowcol[0] > 0 and bc_rowcol[1] > 0):
        raise RuntimeError(
            "auto_seed_calibrant: chord-bisector failed to recover BC; "
            "no ring arcs survived the area filter"
        )

    # Step 5: ring radii (uses the un-downsampled labels from above)
    rads = _detect_ring_radii(props, bc_rowcol, min_area)

    # Step 6: multi-hypothesis Lsd matching
    lsd, n_matched = _estimate_lsd(rads, sim_radii_px,
                                     first_ring=first_ring,
                                     initial_lsd=initial_lsd_um,
                                     max_ring=max_ring,
                                     lsd_window=lsd_window)

    # MIDAS convention: BC = (BC_y, BC_z) = (col, row)
    return AutoSeedResult(
        BC_y=float(bc_rowcol[1]), BC_z=float(bc_rowcol[0]),
        Lsd_um=float(lsd),
        n_arcs=n_arcs, n_rings_matched=int(n_matched),
        detected_radii_px=rads,
        threshold_used=float(thr),
    )


__all__ = ["AutoSeedResult", "auto_seed_calibrant"]
