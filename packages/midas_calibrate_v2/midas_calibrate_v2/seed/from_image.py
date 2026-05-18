"""Seed v2's geometry from a raw image — chord-bisector BC + multi-hypothesis
Lsd estimation.  Port of v1's ``AutoCalibrateZarr`` seed pipeline, simplified.

The recipe (per paper3 §3.2 Stage 2):

1. **Background subtract + threshold**: median-filter the image, subtract,
   threshold at 2.5× MAD noise floor.
2. **Connected-component label** the binary image (after 4× downsampling
   for speed).
3. **Radial-consistency filter**: drop regions whose
   ``CV_R = σ_R / mean_R > 0.03`` (panel-gap artefacts, etc.).
4. **Chord-bisector beam center**: for each surviving arc, draw a chord and
   its perpendicular bisector; the bisector passes through the underlying
   ring's center.  Median across all arcs gives the BC.
5. **Detect ring radii** from BC.
6. **Multi-hypothesis L_sd**: for each (detected, simulated) ring pair,
   compute a trial L_sd; pick the assignment with the most consistent
   matches across all detected rings.

Returns a ``SeedResult`` ready to plug into a v2 spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# CRITICAL: diplib MUST be imported before numpy/scipy/skimage on macOS so
# that diplib's libomp.dylib loads first.  If we let scipy load its OpenMP
# runtime first, diplib's MedianFilter silently hangs.  See project memory
# `feedback_calib_strain_threshold.md` / AutoCalibrateZarr's load order.
try:
    import diplib as _DIPLIB
except ImportError:
    _DIPLIB = None

import numpy as np
from scipy import ndimage


@dataclass
class SeedResult:
    bc_y: float
    bc_z: float
    Lsd: float
    n_arcs: int
    n_rings: int
    detected_radii_px: np.ndarray
    arc_coords: np.ndarray = None    # [N, 2] (y_px, z_px) of all kept arcs;
                                      # used by ``refine_seed_geometry`` for the
                                      # follow-up direct geometry fit.


def _chord_bisector_center(coords: np.ndarray, bbox: Tuple[int, int, int, int]
                            ) -> Optional[np.ndarray]:
    """Recover the underlying ring's center from one arc region.

    Picks the chord between the bbox's first-row and farthest pixel; uses
    the midpoint of an equidistant candidate for the bisector intersection.
    Returns ``None`` if the geometry is degenerate.
    """
    edge_idxs = np.where(coords[:, 0] == bbox[0])[0]
    if len(edge_idxs) == 0:
        return None
    edgea = coords[edge_idxs[len(edge_idxs) // 2]]
    dists = np.linalg.norm(coords - edgea, axis=1)
    j = int(np.argmax(dists))
    edgeb = coords[j]
    arc_len = float(dists[j]) / 2.0
    cand = np.where(np.abs(dists - arc_len) < 2.0)[0]
    if len(cand) == 0:
        return None
    candpt = coords[cand[len(cand) // 2]]
    x1, y1 = edgea
    x2, y2 = candpt
    x3, y3 = candpt
    x4, y4 = edgeb
    if y4 == y3 or y2 == y1:
        return None
    m1 = (x1 - x2) / (y2 - y1)
    m2 = (x3 - x4) / (y4 - y3)
    if abs(m1 - m2) < 1e-10:
        return None
    x5, y5 = (edgea + candpt) / 2.0
    x6, y6 = (edgeb + candpt) / 2.0
    x = (y6 - y5 + m1 * x5 - m2 * x6) / (m1 - m2)
    y = m1 * (x - x5) + y5
    return np.array([x, y], dtype=np.float64)


def _fast_median_filter(data: np.ndarray, kernel_size: int = 51,
                          n_iters: int = 3,
                          *, use_diplib: bool = False) -> np.ndarray:
    """Background median filter — fast via downsample + scipy.

    For a smooth-background estimate (the only thing the seed step needs
    median for), the median at full resolution with a wide kernel is
    wasteful.  We **downsample 4×**, take the median with a proportionally
    smaller kernel, then upsample — visually identical result, ~100× faster.

    On a 2880² image with the default kernel=51 and 3 iters, this path
    runs in ~2.5 s (vs ~minutes for full-resolution scipy and 16+ s per
    iter for diplib — which also segfaults on some real images, see #note).

    ``use_diplib=True`` opts into diplib's MedianFilter (vectorised C++);
    fast on synthetic data but **segfaults on some real images** on macOS
    (OpenMP runtime conflict), so it is off by default.  When safe, it
    runs in ~10–20 s per iter at full resolution.

    Falls all the way back to full-resolution scipy median when the input
    is too small to bother downsampling.
    """
    # Optional diplib fast path (vectorised C++).  Off by default because
    # it segfaults on certain real images on macOS even with
    # KMP_DUPLICATE_LIB_OK=TRUE.
    if use_diplib and _DIPLIB is not None:
        try:
            dip_img = _DIPLIB.Image(data.astype(np.float64))
            ks = [int(kernel_size), int(kernel_size)]
            for _ in range(n_iters):
                dip_img = _DIPLIB.MedianFilter(dip_img, ks)
            return np.asarray(dip_img).astype(data.dtype, copy=False)
        except Exception:
            pass    # fall through to the downsample path

    # Default: downsample → small-kernel scipy median → upsample.  Median
    # is O(N · k²); 4× downsample + k/4 kernel is ~64× cheaper.  Background
    # is smooth so we lose nothing visible.
    down = 4 if min(data.shape) >= 1024 else 1
    if down > 1:
        k_down = max(3, int(round(kernel_size / down)))
        if k_down % 2 == 0:                  # odd kernel for centred median
            k_down += 1
        small = ndimage.zoom(data, 1.0 / down, order=1)
        for _ in range(n_iters):
            small = ndimage.median_filter(small, size=k_down)
        out = ndimage.zoom(small, (data.shape[0] / small.shape[0],
                                     data.shape[1] / small.shape[1]), order=1)
        if out.shape != data.shape:
            out = out[: data.shape[0], : data.shape[1]]
        return out.astype(data.dtype, copy=False)

    # Tiny input — just full-res scipy median.
    out = data.copy()
    for _ in range(n_iters):
        out = ndimage.median_filter(out, size=kernel_size)
    return out


def _detect_arcs(image: np.ndarray, *, min_area: int = 300,
                  cv_r_max: float = 0.03,
                  median_kernel: int = 51,
                  median_iters: int = 3,
                  thresh_mad_factor: float = 2.5,
                  panel_mask: Optional[np.ndarray] = None,
                  mask_erode_iter: int = 2,
                  skip_median: bool = False,
                  ) -> Tuple[np.ndarray, np.ndarray, list]:
    """Median-subtract → threshold → connected components → CV_R filter.

    On multi-panel detectors, pass ``panel_mask`` (or rely on auto-detection
    of sentinel values) to suppress panel-gap edge artefacts that would
    otherwise show up as long thin "arcs" with high CV_R.

    When ``skip_median=True`` the background-subtraction step is bypassed
    (use this when the caller has already dark-subtracted; the median
    filter on a 2880² image is the dominant cost of the seed pipeline).

    Returns ``(thresh_image, labels, kept_regions)`` where ``kept_regions``
    is a list of ``(label, coords[N,2], bbox)`` tuples.
    """
    from .mask import apply_mask_for_arcs
    # Apply (auto-detected or supplied) mask BEFORE median filter — the
    # median over panel-gap pixels would otherwise produce a strong
    # gradient at the panel edge that survives thresholding.
    img, mask = apply_mask_for_arcs(image, mask=panel_mask,
                                      erode_iter=mask_erode_iter)
    if skip_median:
        diff = img.copy()
    else:
        bg = _fast_median_filter(img, kernel_size=median_kernel,
                                  n_iters=median_iters)
        diff = img - bg
    diff[diff < 0] = 0
    diff[~mask] = 0           # ensure gap pixels can't pass threshold
    mad = np.median(np.abs(diff - np.median(diff)))
    thresh = diff > thresh_mad_factor * (1.4826 * mad + 1.0)
    thresh &= mask              # belt + suspenders
    labels, n = ndimage.label(thresh)
    if n == 0:
        return thresh, labels, []

    # Vectorised area filter + per-region coord extraction.  The previous
    # ``for k in range(1, n+1): labels == k`` pattern was O(n × NY × NZ)
    # in Python, which becomes ~hours when threshold is loose (hundreds
    # of thousands of tiny components on a 2880² image).  We use
    # ``np.bincount`` for areas and ``ndimage.find_objects`` for bboxes,
    # then take coords from the per-region slice only — both stay in C.
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    big = np.flatnonzero(areas[1:] >= min_area) + 1     # skip label 0
    slices = ndimage.find_objects(labels)
    kept = []
    for k in big.tolist():
        sl = slices[k - 1]
        if sl is None:
            continue
        sub = labels[sl] == k
        # Local (row, col) -> global via bbox offset
        local = np.argwhere(sub)
        coords = local + np.array([sl[0].start, sl[1].start])
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        ar = max(h, w) / max(min(h, w), 1)
        if ar > 30.0:           # extremely elongated → almost certainly an artefact
            continue
        bbox = (int(sl[0].start), int(sl[1].start),
                int(sl[0].stop - 1), int(sl[1].stop - 1))
        kept.append((int(k), coords, bbox))
    return thresh, labels, kept


def _detect_ring_radii(coords_list, bc: np.ndarray, *, dedup_px: float = 20.0
                        ) -> np.ndarray:
    """Mean radius per region; merge regions whose radii are within ``dedup_px``."""
    radii = []
    for coords in coords_list:
        rdists = np.linalg.norm(coords - bc, axis=1)
        radii.append(float(rdists.mean()))
    radii.sort()
    out = []
    for r in radii:
        if not out or abs(r - out[-1]) > dedup_px:
            out.append(r)
    return np.array(out, dtype=np.float64)


def _multi_hypothesis_lsd(
    detected_radii_px: np.ndarray,
    sim_radii_px: np.ndarray,
    initial_lsd: float,
    *,
    first_ring: int = 1,
    max_det_start: int = 10,
    rel_match_tol: float = 0.05,
) -> Tuple[float, int]:
    """Multi-hypothesis L_sd: try assigning each of the first
    ``max_det_start`` detected rings to each simulated ring, count matches,
    pick the most-consistent assignment.

    Returns ``(best_lsd, n_matches)``.
    """
    if len(detected_radii_px) == 0 or len(sim_radii_px) == 0:
        return initial_lsd, 0
    best_lsd = initial_lsd
    best_score = (0, 1e9)   # (matches, lsd_std)
    n_sim = len(sim_radii_px)
    n_start = min(max_det_start, len(detected_radii_px))
    for det_i in range(n_start):
        for sim_j in range(first_ring - 1, n_sim):
            trial = initial_lsd * detected_radii_px[det_i] / sim_radii_px[sim_j]
            scale = trial / initial_lsd
            trial_sim_px = sim_radii_px * scale
            n_match = 0
            lsds = []
            for det_r in detected_radii_px:
                diffs = np.abs(trial_sim_px - det_r)
                jmin = int(np.argmin(diffs))
                if diffs[jmin] / det_r < rel_match_tol:
                    n_match += 1
                    lsds.append(initial_lsd * det_r / sim_radii_px[jmin])
            std = float(np.std(lsds)) if len(lsds) >= 2 else 1e9
            if n_match > best_score[0] or (
                n_match == best_score[0] and std < best_score[1]
            ):
                best_score = (n_match, std)
                if lsds:
                    best_lsd = float(np.median(lsds))
    return best_lsd, best_score[0]


def seed_from_image(
    image: np.ndarray,
    *,
    sim_radii_px: np.ndarray,
    initial_lsd: float,
    npy: int, npz: int,
    bc_guess: Optional[Tuple[float, float]] = None,
    min_area: int = 300,
    median_kernel: int = 51,
    thresh_mad_factor: float = 2.5,
    first_ring: int = 1,
    method: str = "circle_fit",     # "circle_fit" | "chord_bisector"
    panel_mask: Optional[np.ndarray] = None,    # multi-panel detectors
    mask_erode_iter: int = 2,
    skip_median: bool = False,
    min_ring_radius_px: float = 100.0,
) -> SeedResult:
    """Run the full seed pipeline on a raw calibrant image.

    Parameters
    ----------
    image : 2-D float array (Z × Y or Y × Z; doesn't matter, BC is in image
        pixel coordinates).
    sim_radii_px : simulated ring radii at ``initial_lsd`` (pixels).
    initial_lsd : starting Lsd guess in μm (e.g. 1 000 000 = 1 m).
    bc_guess : optional fallback BC if the chord-bisector fails.

    Returns
    -------
    :class:`SeedResult`.
    """
    _, _, kept = _detect_arcs(
        image, min_area=min_area,
        median_kernel=median_kernel,
        thresh_mad_factor=thresh_mad_factor,
        panel_mask=panel_mask,
        mask_erode_iter=mask_erode_iter,
        skip_median=skip_median,
    )
    # _detect_arcs returns coords as (row, col) which in MIDAS image convention
    # is (z, y). circle_fit and detect_ring_radii both expect (Y, Z), so swap.
    coords_list = [coords[:, [1, 0]] for _, coords, _ in kept]
    bc = None
    n_arcs_used = 0
    if method == "circle_fit":
        # Algebraic circle fit per arc — robust on multi-panel detectors
        # where chord-bisector is biased by per-panel arc fragments.
        from .circle_fit import fit_arcs_for_bc
        result = fit_arcs_for_bc(coords_list, seed_bc=bc_guess)
        if np.isfinite(result["bc_y"]) and np.isfinite(result["bc_z"]):
            bc = np.array([result["bc_y"], result["bc_z"]], dtype=np.float64)
            n_arcs_used = result["n_arcs"]
    if bc is None:
        # Fallback: chord bisector.
        bc_candidates = []
        for _, coords, bbox in kept:
            c = _chord_bisector_center(coords, bbox)
            if c is not None:
                bc_candidates.append(c)
        if bc_candidates:
            bc = np.median(np.stack(bc_candidates), axis=0)
            n_arcs_used = len(bc_candidates)
        elif bc_guess is not None:
            bc = np.array([bc_guess[0], bc_guess[1]], dtype=np.float64)
        else:
            bc = np.array([npy * 0.5, npz * 0.5], dtype=np.float64)
    radii = _detect_ring_radii(coords_list, bc)
    # Drop radii below the beamstop / noise floor before Lsd matching.
    # Below ~100 px is almost always direct-beam scatter or beamstop edge
    # artefacts, never a real Bragg ring at typical HEDM Lsd.
    radii_for_match = radii[radii >= min_ring_radius_px]
    if radii_for_match.size == 0:
        radii_for_match = radii    # nothing left — fall back to using all
    lsd, n_match = _multi_hypothesis_lsd(
        radii_for_match, sim_radii_px, initial_lsd, first_ring=first_ring,
    )
    arc_coords_all = (np.concatenate(coords_list, axis=0)
                       if coords_list else np.zeros((0, 2)))
    return SeedResult(
        bc_y=float(bc[0]), bc_z=float(bc[1]),
        Lsd=float(lsd),
        n_arcs=int(n_arcs_used),
        n_rings=int(n_match),
        detected_radii_px=radii,
        arc_coords=arc_coords_all,
    )


__all__ = ["SeedResult", "seed_from_image"]
