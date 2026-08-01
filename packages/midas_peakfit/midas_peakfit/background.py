"""Local background estimation inside ring bands, and per-spot SNR.

Two related jobs:

1. **Per-spot SNR** (``region_snr`` / ``filter_regions_by_snr``) -- the
   physically meaningful quality criterion for a detected peak, and the reason
   most of this module exists. See ``filter_regions_by_snr`` for why intensity,
   fit residual and omega multiplicity are all *proxies* that fail.
2. **Optional background subtraction** (``BgSubtract 1``, default 0) -- removes
   an azimuthally varying background before thresholding, so ``RingThresh``
   becomes a height above local background rather than an absolute count.

A CORRECTION, recorded so it is not repeated: an earlier version of this
docstring justified (2) with "within a single band the background level spans
90-139 counts against a noise sigma of ~5, i.e. ~20 sigma". **That measurement
was wrong.** It used a band mask built from a plain radius-from-beam-centre
instead of the distortion-corrected ``Rt`` the production path uses after
``apply_image_transformations`` + ``transpose_square``. That naive mask shares
only 13.4 % of its pixels with the real band and straddles the ring's steep
radial edge, manufacturing the variation. Through the production geometry the
reference dataset's bands are **flat** (spread 0.4 sigma), and an absolute
per-ring threshold is adequate there. Background subtraction remains available
for detectors where the background genuinely does vary -- but do not cite
``Au3_cubes_ff_000008`` as its motivation.

Model
-----
The background is estimated per **(ring band, azimuthal sector)** cell rather
than with a large 2-D median filter. That matches how the background actually
varies (smooth in eta around a ring, discontinuous across bands), and it costs
one ``bincount``-style pass instead of a 101x101 median filter over 2048^2
pixels on every one of ~1440 frames.

Robust statistics throughout: the cell background is the **median** and the
noise is ``1.4826 * MAD``. Both are insensitive to the Bragg spots sitting in
the cell, which is essential -- the spots are the signal we are trying to keep,
and a mean/std estimator would let a bright spot inflate its own background.

Background subtraction is **opt-in**. With ``BgSubtract 0`` (the default) the
peak search behaves exactly as it did before, so existing reconstructions are
unchanged. The SNR filter is likewise off unless ``MinPeakSNR`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_N_SECTORS = 36          # 10-degree azimuthal cells
MIN_PIXELS_PER_CELL = 64        # below this a cell median is too noisy to trust


@dataclass
class BackgroundBins:
    """Static per-pixel cell assignment, computed once from geometry.

    ``labels`` is ``-1`` outside every ring band and otherwise
    ``ring_index * n_sectors + sector_index``. It depends only on the detector
    geometry and the ring radii, never on the frame, so it is built once and
    reused for the whole scan.
    """

    labels: np.ndarray            # int32 (N, N); -1 = outside all bands
    n_bins: int
    n_sectors: int
    counts: np.ndarray            # int64 (n_bins,) pixels per cell

    @property
    def in_band(self) -> np.ndarray:
        return self.labels >= 0

    def thin_cells(self, min_pixels: int = MIN_PIXELS_PER_CELL) -> np.ndarray:
        """Indices of cells with too few pixels for a trustworthy median."""
        return np.where((self.counts > 0) & (self.counts < min_pixels))[0]


def build_background_bins(
    Rt: np.ndarray,
    Eta: np.ndarray,
    ring_radii: np.ndarray,
    width_px: float,
    n_sectors: int = DEFAULT_N_SECTORS,
) -> BackgroundBins:
    """Assign every pixel to a (ring, azimuthal sector) cell.

    ``Rt`` and ``Eta`` must come from :func:`geometry.compute_rt_eta` so the
    bands line up exactly with the ones ``compute_good_coords`` thresholds --
    building them from a plain radius-from-beam-centre instead gets the bands
    wrong (no distortion correction, no ``transpose_square``) and silently
    estimates the background over the wrong pixels.

    Overlapping bands follow the same last-ring-wins rule as
    ``compute_good_coords``, so the background cells and the threshold map
    always agree about which ring a pixel belongs to.
    """
    if n_sectors < 1:
        raise ValueError(f"n_sectors must be >= 1, got {n_sectors}")
    if Rt.shape != Eta.shape:
        raise ValueError(f"Rt {Rt.shape} and Eta {Eta.shape} must match")

    labels = np.full(Rt.shape, -1, dtype=np.int32)
    # Eta is in (-180, +180] degrees -> sector index 0..n_sectors-1
    sector = np.floor((Eta + 180.0) / (360.0 / n_sectors)).astype(np.int32)
    np.clip(sector, 0, n_sectors - 1, out=sector)

    for r, rad in enumerate(ring_radii):
        in_band = (Rt > rad - width_px) & (Rt < rad + width_px)
        labels[in_band] = r * n_sectors + sector[in_band]

    n_bins = int(len(ring_radii)) * n_sectors
    flat = labels.ravel()
    counts = np.bincount(flat[flat >= 0], minlength=n_bins).astype(np.int64)
    return BackgroundBins(labels=labels, n_bins=n_bins,
                          n_sectors=n_sectors, counts=counts)


def estimate_cell_stats(
    img: np.ndarray, bins: BackgroundBins
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell (median, 1.4826*MAD) for one frame.

    Returns two ``(n_bins,)`` arrays. Empty cells get ``(0.0, 0.0)``.
    """
    flat_img = img.ravel()
    flat_lab = bins.labels.ravel()
    sel = flat_lab >= 0
    vals = flat_img[sel]
    labs = flat_lab[sel]

    med = np.zeros(bins.n_bins, dtype=np.float64)
    sig = np.zeros(bins.n_bins, dtype=np.float64)
    if vals.size == 0:
        return med, sig

    # Group by cell. argsort once, then slice contiguous runs -- far cheaper
    # than a boolean mask per cell when n_bins is in the hundreds.
    order = np.argsort(labs, kind="stable")
    labs_s, vals_s = labs[order], vals[order]
    edges = np.searchsorted(labs_s, np.arange(bins.n_bins + 1))
    for b in range(bins.n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        v = vals_s[lo:hi]
        m = float(np.median(v))
        med[b] = m
        sig[b] = float(np.median(np.abs(v - m))) * 1.4826
    return med, sig


def local_background(
    img: np.ndarray, bins: BackgroundBins, *, min_pixels: int = MIN_PIXELS_PER_CELL
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel background and noise-sigma maps for one frame.

    Cells with fewer than ``min_pixels`` pixels fall back to the median over
    the cells of the *same ring* (a thin cell is usually a band clipped by the
    detector edge, where a handful of pixels would give a wild median).
    Pixels outside every band get 0.
    """
    med, sig = estimate_cell_stats(img, bins)

    if bins.n_sectors > 0:
        n_rings = bins.n_bins // bins.n_sectors
        thin = set(bins.thin_cells(min_pixels).tolist())
        empty = set(np.where(bins.counts == 0)[0].tolist())
        bad = thin | empty
        if bad:
            for r in range(n_rings):
                lo, hi = r * bins.n_sectors, (r + 1) * bins.n_sectors
                good = [b for b in range(lo, hi)
                        if b not in bad and bins.counts[b] > 0]
                if not good:
                    continue
                fm = float(np.median(med[good]))
                fs = float(np.median(sig[good]))
                for b in range(lo, hi):
                    if b in bad:
                        med[b], sig[b] = fm, fs

    bg = np.zeros_like(img, dtype=np.float64)
    noise = np.zeros_like(img, dtype=np.float64)
    m = bins.in_band
    lab = bins.labels[m]
    bg[m] = med[lab]
    noise[m] = sig[lab]
    return bg, noise


def subtract_local_background(
    img: np.ndarray, bins: BackgroundBins, *, min_pixels: int = MIN_PIXELS_PER_CELL
) -> Tuple[np.ndarray, np.ndarray]:
    """``img - local_background(img)`` inside the bands, plus the sigma map.

    Pixels outside every ring band are left untouched: they are zeroed by the
    ``good_coords`` mask downstream anyway, and subtracting a background there
    would be meaningless.
    """
    bg, noise = local_background(img, bins, min_pixels=min_pixels)
    out = img.copy()
    m = bins.in_band
    out[m] = img[m] - bg[m]
    return out, noise


def bins_from_params(
    p, panels: List, ring_radii: Optional[np.ndarray],
    n_sectors: int = DEFAULT_N_SECTORS,
) -> Optional[BackgroundBins]:
    """Convenience constructor from a :class:`ZarrParams` + panel list.

    Returns ``None`` when background subtraction cannot apply -- no rings, no
    radii, or ``DoFullImage=1`` (which has no ring bands to bin by).
    """
    if ring_radii is None or len(getattr(p, "RingNrs", ())) == 0:
        return None
    if getattr(p, "DoFullImage", 0) == 1:
        return None
    from midas_peakfit.geometry import compute_rt_eta

    Rt, Eta = compute_rt_eta(p, panels)
    return build_background_bins(
        Rt, Eta, np.asarray(ring_radii, dtype=np.float64)[: p.nRingsThresh],
        float(p.Width), n_sectors=n_sectors,
    )


__all__ = [
    "BackgroundBins",
    "filter_regions_by_snr",
    "region_snr",
    "DEFAULT_N_SECTORS",
    "MIN_PIXELS_PER_CELL",
    "bins_from_params",
    "build_background_bins",
    "estimate_cell_stats",
    "local_background",
    "subtract_local_background",
]


# ─── Per-spot SNR (the physically meaningful quality criterion) ─────────────
def region_snr(
    region, corrected: np.ndarray, bins: BackgroundBins,
    med: np.ndarray, sig: np.ndarray,
) -> float:
    """Local SNR of one detected region against its own (ring, sector) cell.

    ``(peak - cell_median) / cell_sigma``, with the cell chosen by the region's
    centroid. The cell statistics are robust (median, 1.4826*MAD) over the
    thousands of pixels in a 10-degree arc of the band, so a handful of Bragg
    spots inside the cell cannot inflate the background against themselves --
    the failure that makes a small per-spot annulus over-optimistic.

    ``corrected`` MUST be the UNGATED frame (``correct_frame``, not
    ``preprocess_frame``): on a thresholded frame every sub-threshold pixel is
    0, so the background and its MAD collapse and every SNR is meaningless.

    Returns 0.0 when the region is outside all bands or its cell has no
    measurable noise.
    """
    if region.pixel_rows.size == 0:
        return 0.0
    r = int(round(float(region.pixel_rows.mean())))
    c = int(round(float(region.pixel_cols.mean())))
    if not (0 <= r < bins.labels.shape[0] and 0 <= c < bins.labels.shape[1]):
        return 0.0
    lab = int(bins.labels[r, c])
    if lab < 0 or lab >= sig.size or sig[lab] <= 0.0:
        return 0.0
    peak = float(corrected[region.pixel_rows, region.pixel_cols].max())
    return (peak - float(med[lab])) / float(sig[lab])


def filter_regions_by_snr(
    regions, corrected: np.ndarray, bins: Optional[BackgroundBins],
    min_snr: float,
) -> Tuple[list, list]:
    """Drop regions whose local SNR falls below ``min_snr``.

    Returns ``(kept, snrs_of_kept)``. ``min_snr <= 0`` disables the filter and
    every region is kept (with its SNR still computed and returned, so the
    number can be recorded even when nothing is being rejected).

    Why SNR and not a proxy:

    * **not raw intensity** -- ``MinIntegratedIntensity`` cannot separate a weak
      real spot on a quiet patch of detector from a noise excursion on a hot
      one, because it carries no noise estimate;
    * **not fit residual** -- ``FitRMSE`` is an *absolute* residual, so it grows
      with peak intensity; cutting on it discards the brightest, most certainly
      real spots first (58 % of indexed spots on the reference dataset);
    * **not omega multiplicity** -- ``NImgs`` encodes mosaicity, not reality. A
      small or undeformed grain can satisfy Bragg inside one frame; 45.9 % of
      credible spots on the reference dataset were single-frame, and 8 indexed
      spots reached SNR 2511 on a single frame.

    SNR asks the only question that matters -- is this peak above the local
    noise -- and makes no assumption about grain size, mosaicity or omega step.
    """
    if bins is None:
        return list(regions), [0.0] * len(regions)
    med, sig = estimate_cell_stats(corrected, bins)
    kept, snrs = [], []
    for reg in regions:
        s = region_snr(reg, corrected, bins, med, sig)
        if min_snr <= 0.0 or s >= min_snr:
            kept.append(reg)
            snrs.append(s)
    return kept, snrs
