"""Cosmic-ray / single-event-upset rejection for sweep-mode stacks.

A single bad pixel in a single frame creates a spike in the integrated
profile. With a sweep stack you have temporal redundancy: the same
pixel sees the same beam intensity across many frames, so an outlier
in the time series is almost certainly a cosmic ray (or SEU) rather
than real signal.

:func:`reject_cosmic_rays` does per-pixel sigma-clipping along the
stack axis and returns (a) the cleaned stack with outliers replaced
by the median or by NaN, and (b) the mask of detected outliers.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def reject_cosmic_rays(
    images: np.ndarray,
    *,
    n_sigma: float = 5.0,
    mode: str = "replace_with_median",
    use_mad: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel sigma-clip outlier rejection along the stack axis.

    Parameters
    ----------
    images :
        ``(N, NrPixelsZ, NrPixelsY)`` numpy array. N must be at least 3
        for the median + sigma estimate to be meaningful.
    n_sigma :
        Threshold above which a pixel is flagged as an outlier (positive
        and negative deviations both flagged).
    mode :
        - ``"replace_with_median"`` (default): replace each outlier with
          the per-pixel temporal median.
        - ``"replace_with_nan"``: replace with ``np.nan`` (downstream
          can mask).
        - ``"flag_only"``: leave images unchanged; just return the mask.
    use_mad :
        If True (default), estimate per-pixel σ via the median absolute
        deviation (robust to multiple outliers in the same time series).
        If False, use the standard deviation (faster but breaks if more
        than one outlier per pixel).

    Returns
    -------
    cleaned : np.ndarray
        ``(N, NrPixelsZ, NrPixelsY)`` cleaned (or unchanged for
        ``flag_only``) stack.
    outlier_mask : np.ndarray
        ``(N, NrPixelsZ, NrPixelsY)`` bool mask, True = outlier.
    """
    if images.ndim != 3 or images.shape[0] < 3:
        raise ValueError(
            f"images must be (N>=3, NZ, NY); got shape {images.shape}"
        )
    if mode not in ("replace_with_median", "replace_with_nan", "flag_only"):
        raise ValueError(f"unknown mode {mode!r}")

    images = images.astype(np.float64)
    median = np.median(images, axis=0)                   # (NZ, NY)
    if use_mad:
        # MAD-based σ: 1.4826 · median(|x - median|) is the unbiased
        # estimator for Gaussian σ. Robust to ≤ 50% outliers per pixel.
        mad = np.median(np.abs(images - median[None, :, :]), axis=0)
        sigma = 1.4826 * mad
        sigma[sigma == 0] = images.std(axis=0)[sigma == 0]   # fallback
    else:
        sigma = images.std(axis=0)
    sigma[sigma == 0] = 1e-30                             # avoid div-by-zero

    deviations = np.abs(images - median[None, :, :]) / sigma[None, :, :]
    outliers = deviations > n_sigma

    if mode == "flag_only":
        return images, outliers
    if mode == "replace_with_nan":
        cleaned = images.copy()
        cleaned[outliers] = np.nan
        return cleaned, outliers
    # replace_with_median (default)
    cleaned = images.copy()
    median_full = np.broadcast_to(median[None, :, :], images.shape)
    cleaned[outliers] = median_full[outliers]
    return cleaned, outliers


def reject_spatial_spikes(
    image: np.ndarray,
    *,
    n_sigma: float = 5.0,
    method: str = "laplacian",
    kernel_size: int = 3,
    mode: str = "replace_with_median",
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame spatial dezinger (sibling of :func:`reject_cosmic_rays`).

    The temporal dezinger needs the stack; per-frame work needs spatial
    statistics. ``method='laplacian'`` flags pixels whose Laplacian-of-
    Gaussian (LoG) response exceeds ``n_sigma · MAD`` of the LoG image
    — this isolates pixel-scale events from smooth peaks. ``method=
    'median'`` is the simpler classical filter: a pixel is an outlier
    if it deviates from its local median by more than ``n_sigma · MAD``
    of the local neighbourhood.

    Parameters
    ----------
    image :
        2-D detector image.
    n_sigma :
        Threshold in robust-σ (MAD-based).
    method :
        ``"laplacian"`` (LoG response) or ``"median"`` (local median
        deviation).
    kernel_size :
        Neighbourhood size for the median method (odd integer >= 3).
    mode :
        ``"replace_with_median"`` (default) or ``"replace_with_nan"`` or
        ``"flag_only"``.

    Returns
    -------
    cleaned, outlier_mask : both same shape as ``image``.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {image.shape}")
    if mode not in ("replace_with_median", "replace_with_nan", "flag_only"):
        raise ValueError(f"unknown mode {mode!r}")
    if method not in ("laplacian", "median"):
        raise ValueError(f"method must be 'laplacian' or 'median'")
    img = image.astype(np.float64)

    if method == "laplacian":
        # 5-point Laplacian as a cheap LoG approximation. Bigger kernels
        # are available via scipy if installed; we keep this dependency-
        # free for the foundation tier.
        pad = np.pad(img, 1, mode="edge")
        lap = (pad[:-2, 1:-1] + pad[2:, 1:-1]
                + pad[1:-1, :-2] + pad[1:-1, 2:]
                - 4.0 * pad[1:-1, 1:-1])
        med_lap = np.median(lap)
        mad_lap = np.median(np.abs(lap - med_lap))
        sigma = 1.4826 * mad_lap if mad_lap > 0 else lap.std() or 1.0
        outliers = np.abs(lap - med_lap) > n_sigma * sigma
        local_median = np.zeros_like(img)
        if mode == "replace_with_median":
            # Use 3x3 local median for replacement
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    local_median += np.roll(np.roll(img, di, axis=0), dj, axis=1)
            local_median /= 9.0
    else:  # method == "median"
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        half = kernel_size // 2
        # Stack of shifted views, take median across the stack
        stack = []
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                stack.append(
                    np.roll(np.roll(img, di, axis=0), dj, axis=1)
                )
        local_stack = np.stack(stack, axis=0)
        local_median = np.median(local_stack, axis=0)
        local_mad = np.median(np.abs(local_stack - local_median[None]), axis=0)
        sigma = 1.4826 * local_mad
        sigma = np.where(sigma > 0, sigma, np.maximum(local_stack.std(axis=0), 1.0))
        outliers = np.abs(img - local_median) > n_sigma * sigma

    if mode == "flag_only":
        return img, outliers
    if mode == "replace_with_nan":
        out = img.copy()
        out[outliers] = np.nan
        return out, outliers
    out = img.copy()
    out[outliers] = local_median[outliers]
    return out, outliers


def _azimuthal_clip_2d_local(
    image: np.ndarray,
    img_flat: np.ndarray,
    valid: np.ndarray,
    r_bin: np.ndarray,
    eta_bin: np.ndarray,
    *,
    n_r: int,
    n_eta: int,
    n_sigma: float,
    mode: str,
    radial_window: int,
    eta_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """2-D-local-MAD branch for tilted / curved-ring data.

    Builds the 2-D (η, R) integrated map, computes a local median + MAD
    in a small (η, R) window around each bin, and flags bins whose
    mean intensity exceeds local_median + n_sigma · 1.4826 · MAD. Then
    projects the flag back to pixels (every pixel landing in a flagged
    bin is flagged).

    Uses ``scipy.ndimage.generic_filter`` for the 2-D median. η wraps
    naturally (mode='wrap'); R reflects (mode='reflect') so edge bins
    don't get a one-sided window.
    """
    from scipy.ndimage import median_filter

    # Build mean-per-bin 2-D map from valid pixels.
    flat_bin_idx = eta_bin * n_r + r_bin
    sums = np.bincount(flat_bin_idx[valid], weights=img_flat[valid],
                        minlength=n_eta * n_r)
    counts = np.bincount(flat_bin_idx[valid], minlength=n_eta * n_r)
    mean_map = (sums / np.maximum(counts, 1)).reshape(n_eta, n_r)
    valid_map = (counts > 0).reshape(n_eta, n_r)

    # Local 2-D median over (η, R). η wraps, R reflects.
    size = (2 * eta_window + 1, 2 * radial_window + 1)
    # generic_filter with separate axis modes isn't supported; do a
    # cheap manual wrap in η by padding before filtering.
    pad_eta = eta_window
    padded = np.concatenate([
        mean_map[-pad_eta:, :],
        mean_map,
        mean_map[:pad_eta, :],
    ], axis=0)
    median_pad = median_filter(padded, size=size, mode="reflect")
    local_median = median_pad[pad_eta:pad_eta + n_eta, :]

    # MAD using same 2-D-local treatment of |x - median|.
    abs_dev_pad = np.abs(padded - median_pad)
    mad_pad = median_filter(abs_dev_pad, size=size, mode="reflect")
    local_mad = mad_pad[pad_eta:pad_eta + n_eta, :]
    local_sigma = 1.4826 * local_mad

    # Flag bins exceeding local-median + n_sigma · sigma (only positive
    # deviations, and only where the bin actually had pixels).
    bin_thresh = local_median + n_sigma * local_sigma
    bin_outlier = valid_map & (local_sigma > 0) & (mean_map > bin_thresh)

    # Project bin-flags back to pixels.
    flag_bin = bin_outlier.reshape(-1)
    pix_outlier = np.zeros_like(img_flat, dtype=bool)
    pix_outlier[valid] = flag_bin[flat_bin_idx[valid]]
    outliers = pix_outlier.reshape(image.shape)

    if mode == "flag_only":
        return img_flat.reshape(image.shape), outliers
    cleaned_flat = img_flat.copy()
    # Replacement value: the local median for that pixel's bin.
    if mode == "replace_with_nan":
        cleaned_flat[pix_outlier] = np.nan
    else:  # replace_with_median
        replacement = local_median.reshape(-1)[flat_bin_idx]
        cleaned_flat[pix_outlier] = replacement[pix_outlier]
    return cleaned_flat.reshape(image.shape), outliers


def azimuthal_sigma_clip(
    image: np.ndarray,
    geom,
    *,
    n_sigma: float = 5.0,
    mode: str = "replace_with_median",
    min_pixels_per_ring: int = 16,
    radial_window: int = 0,
    eta_window: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-ring (azimuthal) sigma-clip for parasitic single-crystal spots.

    A powder ring is approximately uniform in η at fixed R; a parasitic
    single-crystal Bragg spot from a sample-environment window, gasket,
    capillary, or stray grain is localised in η and spikes well above the
    azimuthal median of its ring. This function groups pixels by their
    radial bin (via :class:`HardBinGeometry`), computes a robust median +
    MAD across η in each ring, and flags pixels whose intensity exceeds
    ``median + n_sigma · 1.4826 · MAD``.

    Two operating modes:

    - **Per-ring (default, ``radial_window=0``)**: assumes circular,
      η-uniform rings (i.e. detector well-calibrated, no significant
      tilt). Computes one median + MAD per radial bin across all η.
      Fast (O(n_pixels)), but FALSELY FLAGS ring pixels when the rings
      are curved/elliptical in (η, R) space — e.g. when the detector
      has uncalibrated tilt (tx, ty, tz ≠ 0).

    - **2-D local (``radial_window > 0``)**: computes the median + MAD
      over a small local window in *both* η and R, on the integrated
      2-D map. Robust to tilted/elliptical rings, texture gradients in
      the ring, and any smooth large-scale variation. Slower (builds
      the full 2-D map + a 2-D median filter), but the only choice for
      real data that hasn't been geometry-refined.

    Unlike :func:`reject_cosmic_rays` (temporal) and
    :func:`reject_spatial_spikes` (pixel-scale neighbourhood), this is
    the principled approach when:

    - The spot is multi-pixel (a real Bragg spot at finite mosaicity),
      not pixel-scale (cosmic ray).
    - You have a single frame, not a sweep.
    - You want to remove parasitic-but-static features that *temporal*
      clipping cannot see.

    Parameters
    ----------
    image :
        ``(NrPixelsZ, NrPixelsY)`` 2-D detector image.
    geom :
        :class:`~midas_integrate_v2.binning.HardBinGeometry` built from
        the same spec as the image. Provides per-pixel
        ``(η_bin, r_bin)`` assignment.
    n_sigma :
        Robust-σ threshold (MAD-based). Defaults to 5.0; for very
        dilute parasitic crystals, 3.0 is reasonable; for noisy data,
        try 7.0.
    mode :
        ``"replace_with_median"`` (default) replaces flagged pixels with
        the per-ring median; ``"replace_with_nan"`` writes NaN;
        ``"flag_only"`` leaves the image untouched.
    min_pixels_per_ring :
        Rings with fewer than this many valid pixels are skipped (no
        clipping applied) — too few samples to robustly estimate median
        and MAD. Defaults to 16. Only used in the per-ring mode.
    radial_window :
        Half-width in radial bins for the 2-D local mode. ``0``
        (default) keeps the original per-ring algorithm. ``5`` to
        ``15`` are sensible defaults for tilted-detector data: large
        enough that the window covers the local ring background, small
        enough that nearby rings don't both fall in the same window.
    eta_window :
        Half-width in η bins for the 2-D local mode. Defaults to ``0``,
        which auto-selects ``max(1, n_eta // 36)`` (≈ 10° for the
        standard 1°-η-bin layout). Use a larger value if your data has
        sharp η features you want to *preserve* in the background; use
        a smaller one if spots are sharp in η.

    Returns
    -------
    cleaned : np.ndarray
        Cleaned image, same shape and dtype family (float64) as the
        input.
    outlier_mask : np.ndarray
        Bool mask, same shape as ``image``; True = flagged parasitic
        pixel.

    Notes
    -----
    Only *positive* deviations are flagged — parasitic Bragg spots are
    additive. Negative MAD-deviations (panel gaps, masked pixels)
    survive untouched.

    The clip is single-pass. For very dense parasitic clusters where
    the spot pixels bias the median itself, call twice (the second
    pass operates on the already-replaced image and converges).
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {image.shape}")
    if image.shape != (geom.n_pixels_z, geom.n_pixels_y):
        raise ValueError(
            f"image shape {image.shape} does not match geometry "
            f"({geom.n_pixels_z}, {geom.n_pixels_y})"
        )
    if mode not in ("replace_with_median", "replace_with_nan", "flag_only"):
        raise ValueError(f"unknown mode {mode!r}")

    img_flat = image.astype(np.float64).reshape(-1)
    flat_bin = geom.flat_bin.cpu().numpy()
    valid = geom.valid.cpu().numpy()
    n_r = int(geom.n_r)
    n_eta = int(geom.n_eta)

    r_bin = flat_bin % n_r                        # per-pixel radial bin
    eta_bin = flat_bin // n_r                     # per-pixel η bin
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return image.astype(np.float64), np.zeros_like(image, dtype=bool)

    if radial_window > 0:
        return _azimuthal_clip_2d_local(
            image, img_flat, valid, r_bin, eta_bin,
            n_r=n_r, n_eta=n_eta,
            n_sigma=n_sigma, mode=mode,
            radial_window=int(radial_window),
            eta_window=int(eta_window) or max(1, n_eta // 36),
        )

    # Group valid pixels by radial bin via sort.
    r_valid = r_bin[valid_idx]
    order = np.argsort(r_valid, kind="stable")
    pix_sorted = valid_idx[order]                 # pixel indices, grouped by r
    r_sorted = r_valid[order]
    # Group boundaries: edges[k] is the start of bin k in pix_sorted.
    edges = np.searchsorted(r_sorted, np.arange(n_r + 1))

    # Per-ring median and MAD over η-distributed valid pixels.
    median_r = np.zeros(n_r, dtype=np.float64)
    sigma_r = np.zeros(n_r, dtype=np.float64)
    enough = np.zeros(n_r, dtype=bool)
    for k in range(n_r):
        lo, hi = edges[k], edges[k + 1]
        if hi - lo < min_pixels_per_ring:
            continue
        vals = img_flat[pix_sorted[lo:hi]]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        median_r[k] = med
        sigma_r[k] = 1.4826 * mad if mad > 0 else 0.0
        enough[k] = True

    # Per-pixel threshold (only meaningful where `enough[r_bin[p]]`).
    # When sigma_r is zero (degenerate ring, all identical), don't flag.
    thresh = median_r[r_bin] + n_sigma * sigma_r[r_bin]
    ring_clippable = enough[r_bin] & (sigma_r[r_bin] > 0)
    outliers_flat = valid & ring_clippable & (img_flat > thresh)
    outliers = outliers_flat.reshape(image.shape)

    if mode == "flag_only":
        return img_flat.reshape(image.shape), outliers
    cleaned_flat = img_flat.copy()
    if mode == "replace_with_nan":
        cleaned_flat[outliers_flat] = np.nan
    else:  # replace_with_median
        cleaned_flat[outliers_flat] = median_r[r_bin[outliers_flat]]
    return cleaned_flat.reshape(image.shape), outliers


def azimuthal_sigma_clip_multi_panel(
    images,
    geoms,
    *,
    n_sigma: float = 5.0,
    mode: str = "replace_with_median",
    min_pixels_per_ring: int = 16,
    radial_window: int = 0,
    eta_window: int = 0,
):
    """Per-panel application of :func:`azimuthal_sigma_clip`.

    For tiled detectors (Pilatus, Eiger) or multi-detector layouts
    (APS 1-ID Hydra ge1..ge4) each panel has its own
    :class:`HardBinGeometry`. This helper loops over panels.

    Parameters
    ----------
    images :
        Sequence of 2-D images, one per panel.
    geoms :
        Sequence of :class:`HardBinGeometry`, one per panel, matching
        ``images`` in length and order.

    Returns
    -------
    cleaned : list of np.ndarray
    outlier_masks : list of np.ndarray (bool)
    """
    if len(images) != len(geoms):
        raise ValueError(
            f"images ({len(images)}) and geoms ({len(geoms)}) length mismatch"
        )
    cleaned = []
    masks = []
    for img, geom in zip(images, geoms):
        c, m = azimuthal_sigma_clip(
            np.asarray(img), geom,
            n_sigma=n_sigma, mode=mode,
            min_pixels_per_ring=min_pixels_per_ring,
            radial_window=radial_window,
            eta_window=eta_window,
        )
        cleaned.append(c)
        masks.append(m)
    return cleaned, masks


__all__ = [
    "reject_cosmic_rays",
    "reject_spatial_spikes",
    "azimuthal_sigma_clip",
    "azimuthal_sigma_clip_multi_panel",
]
