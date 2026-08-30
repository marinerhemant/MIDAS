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

import warnings
from typing import Optional, Tuple

import numpy as np


class ShallowStackWarning(UserWarning):
    """A robust estimator is being used on too few frames to support it.

    Its own category so a caller can silence or escalate it without touching
    every other warning — and so it is visible in a log rather than lost, which
    matters because the default rejection mode OVERWRITES the pixels it flags.
    """


#: Measured false-positive rate of the per-pixel MAD sigma-clip on a CLEAN
#: Poisson stack at ``n_sigma = 5``, versus stack depth N. Nominal (Gaussian,
#: exact sigma) is 5.7e-7, so every entry here is orders of magnitude high.
#:
#: The cause is NOT discreteness — the rates are the same at lambda = 3000 as at
#: lambda = 20 — and it is NOT bias, so rescaling does not fix it. It is the
#: small-sample VARIANCE of the MAD: at N = 5 the 5th percentile of
#: sigma_MAD/sigma_true is 0.210, i.e. 5 % of pixels get a sigma five times too
#: small, and ordinary noise there reads as > 5 sigma. Bias-correcting so the
#: MEDIAN of sigma_MAD matches truth only takes N=5 from 2.77 % to 1.49 %.
#: The 1.4826 factor is the asymptotic consistency constant; short stacks need a
#: different estimator, not a corrected one.
#:
#: Measured with 200k independent pixels per N (scratch: mad_fpr.py, mad_fix.py).
_MAD_FALSE_POSITIVE_RATE_AT_5_SIGMA = {
    5: 2.8e-2, 9: 8.0e-3, 15: 2.1e-3, 30: 2.8e-4, 60: 5.0e-5,
}

#: Below this stack depth the MAD estimator's false-positive rate at 5 sigma is
#: worse than ~1e-4 (about 175x nominal), which on a megapixel detector means
#: hundreds of good pixels silently replaced. Warn rather than fail: a shallow
#: stack is a legitimate thing to want to clean, just not with this estimator.
_MAD_MIN_RECOMMENDED_N = 30


def reject_cosmic_rays(
    images: np.ndarray,
    *,
    n_sigma: float = 5.0,
    mode: str = "replace_with_median",
    use_mad: bool = True,
    sigma_model: Optional[str] = None,
    gain: float = 1.0,
    warn_shallow_stack: bool = True,
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
        Legacy selector, kept for backward compatibility. True (default)
        means ``sigma_model="mad"``, False means ``sigma_model="std"``.
        Ignored when ``sigma_model`` is given explicitly.
    sigma_model :
        How the per-pixel σ is estimated. Overrides ``use_mad``.

        - ``"mad"`` — ``1.4826 · median|x − median|`` along the stack.
          Robust to several outliers per pixel, but see the warning below:
          it needs a DEEP stack.
        - ``"std"`` — plain standard deviation along the stack. Not robust
          (a real cosmic ray inflates σ and so partly hides itself), but its
          measured false-positive rate is 0.000 at every depth tested.
        - ``"poisson"`` — ``σ = sqrt(gain · max(median, 1))``, i.e. use the
          KNOWN photon-counting noise model instead of estimating σ from a
          handful of samples. Measured false-positive rate 7e-6 at N = 5,
          1e-6 at N = 30 — within about one order of nominal, and four
          orders better than ``"mad"``. **Correct only when the frames are
          raw counts** (no dark subtraction, normalisation or flat-fielding
          applied yet) and the detector is photon-counting; pass ``gain``
          for an integrating detector, and do not use it at all if read
          noise dominates.
    gain :
        ADU per photon, used only by ``sigma_model="poisson"``. 1.0 for a
        photon-counting detector (Pilatus, Eiger).
    warn_shallow_stack :
        Emit a warning when ``sigma_model="mad"`` is used on a stack too
        shallow for it. On by default because the failure is otherwise
        SILENT — the default ``mode="replace_with_median"`` overwrites the
        flagged pixels, so good data is replaced with no visible sign.

    .. warning::

       **The MAD estimator's false-positive rate is orders of magnitude above
       the nominal Gaussian rate on short stacks.** Measured on a clean
       Poisson stack at ``n_sigma=5`` (nominal 5.7e-7):

       ====  ==========  ==========  ==========
       N     ``"mad"``   ``"std"``   ``"poisson"``
       ====  ==========  ==========  ==========
       5     2.8e-2      0.0         7.0e-6
       9     8.0e-3      0.0         4.4e-6
       15    2.1e-3      0.0         2.7e-6
       30    2.8e-4      0.0         1.0e-6
       60    5.0e-5      0.0         1.3e-6
       ====  ==========  ==========  ==========

       So a 9-frame sweep cleaned at "5σ" has ~0.8 % of its pixels replaced
       by the temporal median — on a 2880² Varex that is roughly 66 000 good
       pixels, silently. The cause is the small-sample *variance* of the MAD,
       not its bias; see ``_MAD_FALSE_POSITIVE_RATE_AT_5_SIGMA``. Prefer
       ``sigma_model="poisson"`` for raw counts, or use a deep stack.

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

    if sigma_model is None:
        sigma_model = "mad" if use_mad else "std"
    if sigma_model not in ("mad", "std", "poisson"):
        raise ValueError(
            f"sigma_model must be 'mad', 'std' or 'poisson', got {sigma_model!r}")

    N = int(images.shape[0])
    if sigma_model == "mad" and warn_shallow_stack and N < _MAD_MIN_RECOMMENDED_N:
        known = sorted(_MAD_FALSE_POSITIVE_RATE_AT_5_SIGMA)
        nearest = min(known, key=lambda k: abs(k - N))
        rate = _MAD_FALSE_POSITIVE_RATE_AT_5_SIGMA[nearest]
        warnings.warn(
            f"reject_cosmic_rays: sigma_model='mad' on a stack of only {N} "
            f"frames. The MAD's small-sample variance makes its false-positive "
            f"rate about {rate:.1e} at 5 sigma (measured at N={nearest}; "
            f"nominal is 5.7e-07), so roughly {rate*100:.2f} % of GOOD pixels "
            f"will be flagged"
            + (" and overwritten with the temporal median"
               if mode == "replace_with_median" else "")
            + f". Use sigma_model='poisson' for raw counts, or a stack of at "
              f"least {_MAD_MIN_RECOMMENDED_N} frames.",
            ShallowStackWarning, stacklevel=2)

    images = images.astype(np.float64)
    median = np.median(images, axis=0)                   # (NZ, NY)
    if sigma_model == "mad":
        # MAD-based σ: 1.4826 · median(|x - median|) is the consistency
        # constant for Gaussian σ **asymptotically**. Robust to ≤ 50 %
        # outliers per pixel, but high-variance on short stacks — see the
        # warning in the docstring.
        mad = np.median(np.abs(images - median[None, :, :]), axis=0)
        sigma = 1.4826 * mad
        sigma[sigma == 0] = images.std(axis=0)[sigma == 0]   # fallback
    elif sigma_model == "poisson":
        # Do not estimate what is already known. For raw photon counts the
        # variance IS the mean, so σ needs no stack depth at all. Floor the
        # median at 1 count so an empty pixel gets σ = sqrt(gain) rather
        # than 0 (which would flag any nonzero reading as infinitely many σ).
        if gain <= 0:
            raise ValueError(f"gain must be positive, got {gain!r}")
        sigma = np.sqrt(float(gain) * np.maximum(median, 1.0))
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
