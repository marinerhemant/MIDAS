"""Robust, seed-free auto-calibration primitives.

These power the fully-automated path on hard real data (off-panel beam centre,
weak signal, unreliable dark) where gradient refinement and the existing
Hough/arc seed get stuck. All functions are pure numpy/scipy and independently
testable.

- :func:`median_background_subtract` — fast background removal that does NOT
  rely on a (possibly garbage) dark frame.
- :func:`detect_ring_radii_profile` — ring-peak radii from the azimuthal-mean
  radial profile of the corrected image.
- :func:`lsd_from_ring_ratios` — **seed-free Lsd**: identify which detected
  ring is which reflection from Lsd-independent radius ratios, then solve Lsd.
- :func:`bc_grid_search` — global beam-centre search at fixed Lsd, maximising
  bright-ring-pixel concentration at the predicted radii.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    median_filter = None
    find_peaks = None


def median_background_subtract(
    image: np.ndarray, *, downsample: int = 8, kernel: int = 13,
) -> np.ndarray:
    """Background-subtract via downsample → median filter → upsample.

    Robust replacement for dark subtraction when the dark is missing or
    unreliable. The median over a ~``downsample*kernel`` px window estimates the
    smooth background (air scatter, detector offset) without eroding thin rings,
    which survive as positive residuals.
    """
    img = np.asarray(image, dtype=np.float64)
    if median_filter is None:
        raise RuntimeError("scipy is required for median_background_subtract")
    small = img[::downsample, ::downsample]
    bg_small = median_filter(small, size=kernel)
    bg = np.repeat(np.repeat(bg_small, downsample, axis=0), downsample, axis=1)
    bg = bg[: img.shape[0], : img.shape[1]]
    return img - bg


def detect_ring_radii_profile(
    corr: np.ndarray, bc_y: float, bc_z: float, *,
    r_min: float = 100.0, r_max: Optional[float] = None,
    prominence_factor: float = 0.6, distance: int = 20, smooth: int = 5,
    clip_pct: Tuple[float, float] = (1.0, 99.5),
) -> np.ndarray:
    """Detected ring-peak radii (px) from the azimuthal-mean radial profile.

    ``corr`` is the (median-)background-subtracted image. Hot pixels are clipped
    before averaging so single bright pixels do not masquerade as rings. Returns
    the peak radii sorted ascending.
    """
    if find_peaks is None:
        raise RuntimeError("scipy is required for detect_ring_radii_profile")
    nz, ny = corr.shape
    zz, yy = np.mgrid[0:nz, 0:ny]
    rpix = np.sqrt((yy - bc_y) ** 2 + (zz - bc_z) ** 2)
    rmax = int(rpix.max()) + 1 if r_max is None else int(r_max) + 1
    lo, hi = np.percentile(corr, clip_pct[0]), np.percentile(corr, clip_pct[1])
    cc = np.clip(corr, lo, hi).ravel()
    ridx = rpix.astype(np.int64).ravel()
    num = np.bincount(ridx, weights=cc, minlength=rmax)
    cnt = np.bincount(ridx, minlength=rmax)
    prof = num / np.maximum(cnt, 1)
    if smooth > 1:
        prof = np.convolve(np.nan_to_num(prof), np.ones(smooth) / smooth, mode="same")
    rax = np.arange(len(prof))
    seg = (rax >= r_min) & (rax <= min(rmax - 1, r_max if r_max else rmax - 1))
    if seg.sum() < 5:
        return np.array([])
    prom = np.nanstd(prof[seg]) * prominence_factor
    pk, _ = find_peaks(prof[seg], prominence=prom, distance=distance)
    return np.sort(rax[seg][pk].astype(np.float64))


def _two_theta_to_lsd_um(R_px: float, two_theta_deg: float, px: float) -> float:
    return R_px * px / math.tan(math.radians(two_theta_deg))


def lsd_from_ring_ratios(
    detected_radii_px: Sequence[float],
    ring_two_theta_deg: Sequence[float],
    px: float, *,
    initial_lsd_um: float = 1_000_000.0, first_ring: int = 1,
    match_tol: float = 0.05, n_start: int = 3,
    lsd_hint_um: Optional[float] = None, hint_window: Tuple[float, float] = (0.5, 2.0),
) -> Tuple[Optional[float], int, int]:
    """Seed-free Lsd by multi-hypothesis ring matching (port of
    AutoCalibrateZarr's ``estimate_lsd``).

    Rather than assume the innermost detected ring is the first reflection
    (fragile — spurious inner detections, missing rings, dense patterns all
    break it), this tries each of the first ``n_start`` *detected* rings as
    *every* possible reflection, computes the implied trial Lsd, predicts all
    ring radii at that Lsd, and **scores the hypothesis by how many detected
    rings land within ``match_tol`` of a predicted ring** (ties broken by
    lowest Lsd scatter). The winning hypothesis returns the median per-ring Lsd.

    Returns ``(lsd_um, n_matches, n_detected)``; ``lsd_um`` is ``None`` if no
    rings were detected.
    """
    rads = np.sort(np.asarray(detected_radii_px, dtype=np.float64))
    tt = np.asarray(ring_two_theta_deg, dtype=np.float64)
    if rads.size == 0 or tt.size == 0:
        return None, 0, int(rads.size)
    sim_rads = initial_lsd_um * np.tan(np.radians(tt)) / px   # predicted radii @ nominal Lsd
    n_sim = len(sim_rads)
    lo_hint = hi_hint = None
    if lsd_hint_um is not None and lsd_hint_um > 0:
        lo_hint, hi_hint = hint_window[0] * lsd_hint_um, hint_window[1] * lsd_hint_um
    best_matches, best_std, best_lsd = 0, np.inf, float(initial_lsd_um)
    for det_start in range(min(n_start, len(rads))):
        for hyp in range(first_ring - 1, n_sim):
            if sim_rads[hyp] <= 0:
                continue
            trial_lsd = initial_lsd_um * rads[det_start] / sim_rads[hyp]
            if lo_hint is not None and not (lo_hint <= trial_lsd <= hi_hint):
                continue   # reject hypotheses inconsistent with the distance hint
            trial_sim_px = sim_rads * (trial_lsd / initial_lsd_um)
            lsds_this = []
            for det in rads:
                diffs = np.abs(trial_sim_px - det)
                j = int(np.argmin(diffs))
                if diffs[j] / det < match_tol:
                    lsds_this.append(initial_lsd_um * det / sim_rads[j])
            matches = len(lsds_this)
            std = float(np.std(lsds_this)) if matches >= 2 else np.inf
            med = float(np.median(lsds_this)) if lsds_this else trial_lsd
            if matches > best_matches or (matches == best_matches and std < best_std):
                best_matches, best_std, best_lsd = matches, std, med
    return best_lsd, best_matches, int(rads.size)


def predicted_ring_radii_px(
    ring_two_theta_deg: Sequence[float], lsd_um: float, px: float,
) -> np.ndarray:
    """Predicted ring radii (px) at a given Lsd."""
    return np.array([lsd_um * math.tan(math.radians(t)) / px
                     for t in ring_two_theta_deg])


def bc_grid_search(
    corr: np.ndarray, pred_radii_px: Sequence[float], bc0: Tuple[float, float], *,
    search_px: float = 60.0, coarse: float = 3.0, fine: float = 0.5,
    bright_pct: float = 99.7, tol: float = 3.0,
) -> Tuple[float, float, float, float]:
    """Global beam-centre search at fixed Lsd (encoded in ``pred_radii_px``).

    Bright (ring) pixels of the corrected image are extracted once; the BC that
    maximises the bright-pixel intensity falling within ``tol`` px of any
    predicted ring radius is returned (coarse grid then local refine). Robust on
    weak / off-panel data where the gradient optimiser cannot move the BC.

    Returns ``(bc_y, bc_z, score, frac_on_ring)`` where ``frac_on_ring`` is the
    fraction of bright pixels within ``tol`` px of a predicted ring at the best
    BC (a quality flag: ~0 means the BC/Lsd are wrong).
    """
    pred = np.asarray(pred_radii_px, dtype=np.float64)
    thr = np.percentile(corr, bright_pct)
    zz, yy = np.where(corr > thr)
    inten = corr[zz, yy].astype(np.float64)
    yy = yy.astype(np.float64); zz = zz.astype(np.float64)

    def score(bcy, bcz):
        r = np.sqrt((yy - bcy) ** 2 + (zz - bcz) ** 2)
        near = np.min(np.abs(r[:, None] - pred[None, :]), axis=1) < tol
        return float(inten[near].sum())

    by0, bz0 = bc0
    best = (score(by0, bz0), by0, bz0)
    for by in np.arange(by0 - search_px, by0 + search_px + 1e-6, coarse):
        for bz in np.arange(bz0 - search_px, bz0 + search_px + 1e-6, coarse):
            s = score(by, bz)
            if s > best[0]:
                best = (s, by, bz)
    _, byc, bzc = best
    for by in np.arange(byc - coarse, byc + coarse + 1e-6, fine):
        for bz in np.arange(bzc - coarse, bzc + coarse + 1e-6, fine):
            s = score(by, bz)
            if s > best[0]:
                best = (s, by, bz)
    s, by, bz = best
    r = np.sqrt((yy - by) ** 2 + (zz - bz) ** 2)
    frac = float((np.min(np.abs(r[:, None] - pred[None, :]), axis=1) < tol).mean())
    return float(by), float(bz), s, frac


def robust_multipanel_seed(
    images: Sequence[np.ndarray],
    approx_bcs: Sequence[Tuple[float, float]],
    ring_two_theta_deg: Sequence[float],
    px: float, *,
    r_min: float = 400.0, r_max: float = 1300.0,
    frac_min: float = 0.6, ratio_tol_max: float = 0.01,
    search_px: float = 60.0,
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Two-pass robust seed for a shared-distance multi-panel detector.

    Designed for arrangements like HYDRA where every panel sits at the *same*
    sample-to-detector distance but the per-panel beam centre is off-panel and
    poorly constrained alone (so a single-panel detect→Lsd→BC iterate is
    unstable — Lsd and BC are coupled and drift).

    **Pass 1** — per panel: median-background-subtract, detect ring radii at the
    supplied approximate BC, and estimate Lsd from ring-radius *ratios*
    (Lsd-independent reflection identification). Panels whose ring-fraction is
    high and whose ratio residual is small have a roughly-correct BC, so their
    Lsd is trustworthy; their median defines the **shared Lsd**.

    **Pass 2** — every panel: BC grid-search at the shared Lsd, which recovers a
    good beam centre even for panels whose Pass-1 BC was off (the bias that made
    their Pass-1 Lsd unreliable).

    Returns ``(shared_lsd_um, [(bc_y, bc_z, frac_on_ring), ...])`` — feed each
    panel's ``(shared_lsd, bc)`` as the seed for the gradient calibration.
    """
    tt = np.asarray(ring_two_theta_deg, dtype=np.float64)
    corrs = [median_background_subtract(im) for im in images]
    clean_lsds: List[float] = []
    for corr, (bcy, bcz) in zip(corrs, approx_bcs):
        radii = detect_ring_radii_profile(corr, bcy, bcz, r_min=r_min, r_max=r_max)
        lsd, nmatch, ndet = lsd_from_ring_ratios(radii, tt, px)
        if lsd is None:
            continue
        pred = predicted_ring_radii_px(tt, lsd, px)
        *_, frac = bc_grid_search(corr, pred, (bcy, bcz), search_px=search_px * 0.6)
        if nmatch >= max(3, int(0.5 * ndet)) and frac >= frac_min:
            clean_lsds.append(lsd)
    if not clean_lsds:
        raise RuntimeError(
            "robust_multipanel_seed: no panel produced a clean ratio-Lsd; "
            "check the calibrant/wavelength or the approximate beam centres"
        )
    shared_lsd = float(np.median(clean_lsds))
    pred = predicted_ring_radii_px(tt, shared_lsd, px)
    out: List[Tuple[float, float, float]] = []
    for corr, (bcy, bcz) in zip(corrs, approx_bcs):
        by, bz, _, frac = bc_grid_search(corr, pred, (bcy, bcz), search_px=search_px)
        out.append((by, bz, frac))
    return shared_lsd, out


__all__ = [
    "median_background_subtract",
    "detect_ring_radii_profile",
    "lsd_from_ring_ratios",
    "predicted_ring_radii_px",
    "bc_grid_search",
    "robust_multipanel_seed",
]
