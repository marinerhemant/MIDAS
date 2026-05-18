"""Variance propagation through v2 integration kernels.

Every published 1D profile should carry per-bin uncertainty (σ) so
downstream peak-fitting / Rietveld / Bayesian analysis can weight the
bins correctly. This module provides ``integrate_*_with_variance``
helpers that return ``(mean, sigma)`` per bin instead of just the
intensity.

Three error models supported, selected by ``error_model``:

- ``'poisson'`` (default): each pixel's variance equals its (positive)
  intensity. This is the right model for shot-noise-limited counting
  detectors. ``variance_image=None`` triggers this.
- ``'azimuthal'``: per-bin σ is estimated from the sample variance of
  in-bin pixels, ``σ²_bin = Σ_i w_i² (I_i - μ_bin)² / (Σ_i w_i)²``.
  Mirrors pyFAI's ``error_model='azimuthal'``. Captures non-Poisson
  scatter (texture, spotty grains, beam jitter) which dominates real-
  data σ on FF/PF-HEDM frames.
- ``'hybrid'``: per-bin σ = max(σ_poisson, σ_azimuthal). pyFAI's
  ``error_model='hybrid'`` equivalent. Robust default when both
  shot-noise and azimuthal-spread contributions are expected.

In all three modes, an explicit ``variance_image`` (same shape as
``image``) overrides the Poisson default for the per-pixel σ².

Per-bin propagation for area-weighted integration
(``Σ_i w_i x_i / Σ_i w_i``):

    σ²_bin = Σ_i (w_i² · σ²_pixel_i) / (Σ_i w_i)²

where ``w_i`` is the per-pixel area weight (``frac`` for soft binning,
``area`` for polygon binning, 1.0 for hard binning).

Intensity-correction propagation
--------------------------------

If a per-pixel ``correction`` factor ``c_i`` is supplied, the function
treats the integrated quantity as ``I'_i = I_i / c_i`` (matching the
multiplicative correction convention used by
``corrections.intensity``). Both the per-pixel intensity and per-pixel
variance are scaled: ``σ'²_i = σ²_i / c_i²``. The returned ``(mean,
sigma)`` is therefore on the *corrected* intensity scale — consistent
with how the upstream pipeline reports ``mean``.

Empty-bin handling
------------------

Bins with zero accumulated weight (fully masked, off-detector, or
empty after pixel-bin geometry) are flagged with ``empty_bin_value``
(default ``NaN``) in *both* the mean and sigma outputs. Use
``torch.isnan(...)`` (or ``torch.isfinite(...)``) downstream to skip
them; the prior silent ``0.0`` behaviour is recoverable by passing
``empty_bin_value=0.0``.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .hard import HardBinGeometry
from .polygon import PolygonBinGeometry
from .subpixel import SubpixelBinGeometry
from .trans_opt import apply_trans_opt_forward, needs_trans_opt


_VALID_ERROR_MODELS = ("poisson", "azimuthal", "hybrid")


def _validate_error_model(error_model: str) -> str:
    if error_model not in _VALID_ERROR_MODELS:
        raise ValueError(
            f"error_model must be one of {_VALID_ERROR_MODELS}, got {error_model!r}"
        )
    return error_model


def _maybe_trans_opt(t: torch.Tensor, geom, apply_flag: bool) -> torch.Tensor:
    if apply_flag and geom.trans_opt and needs_trans_opt(geom.trans_opt):
        return apply_trans_opt_forward(
            t, geom.trans_opt,
            NrPixelsY=geom.n_pixels_y, NrPixelsZ=geom.n_pixels_z,
        )
    return t


def _prepare_pixel_arrays(
    image: torch.Tensor,
    geom,
    *,
    variance_image: Optional[torch.Tensor],
    correction: Optional[torch.Tensor],
    apply_trans_opt: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(I_eff_flat, var_eff_flat)`` in fp64, with trans_opt and
    correction already applied. Both are length ``n_pixels_y * n_pixels_z``."""
    if image.shape != (geom.n_pixels_z, geom.n_pixels_y):
        raise ValueError(
            f"image shape {tuple(image.shape)} does not match "
            f"geometry ({geom.n_pixels_z}, {geom.n_pixels_y})"
        )
    if variance_image is not None and variance_image.shape != image.shape:
        raise ValueError(
            f"variance_image shape {tuple(variance_image.shape)} does not "
            f"match image shape {tuple(image.shape)}"
        )
    if correction is not None and correction.shape != image.shape:
        raise ValueError(
            f"correction shape {tuple(correction.shape)} does not "
            f"match image shape {tuple(image.shape)}"
        )

    image_f = _maybe_trans_opt(image.to(torch.float64), geom, apply_trans_opt)

    if variance_image is None:
        var_f = image_f.clamp(min=0.0)
    else:
        var_f = _maybe_trans_opt(
            variance_image.to(torch.float64), geom, apply_trans_opt,
        )

    if correction is not None:
        corr_f = _maybe_trans_opt(
            correction.to(torch.float64), geom, apply_trans_opt,
        )
        image_f = image_f / corr_f
        var_f = var_f / (corr_f * corr_f)

    return image_f.reshape(-1), var_f.reshape(-1)


def _apply_empty_bin_fill(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    weight_sums: torch.Tensor,
    empty_bin_value: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Where ``weight_sums <= 0``, replace mean and sigma with
    ``empty_bin_value``. Skipped when the user explicitly chose 0.0
    (legacy behaviour)."""
    if empty_bin_value == 0.0 and not math.isnan(empty_bin_value):
        return mean, sigma
    valid = weight_sums > 0
    fill = torch.tensor(empty_bin_value, dtype=mean.dtype, device=mean.device)
    return torch.where(valid, mean, fill), torch.where(valid, sigma, fill)


def _azimuthal_var_weighted(
    img_flat: torch.Tensor,
    pix_idx: torch.Tensor,
    bin_idx: torch.Tensor,
    weights: torch.Tensor,
    mean_per_bin: torch.Tensor,
    weight_sums: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    """``σ²_bin = Σ w_i² (I_i - μ_bin)² / (Σ w_i)²`` — pyFAI's
    azimuthal estimator, weighted form for polygon / subpixel."""
    pix_I = img_flat[pix_idx]
    resid = pix_I - mean_per_bin[bin_idx]
    w2 = weights * weights
    accum = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    accum = accum.index_add(0, bin_idx, w2 * resid * resid)
    safe_w = weight_sums.clamp(min=1e-30)
    return accum / (safe_w * safe_w)


def integrate_hard_with_variance(
    image: torch.Tensor,
    geom: HardBinGeometry,
    *,
    variance_image: Optional[torch.Tensor] = None,
    correction: Optional[torch.Tensor] = None,
    apply_trans_opt: bool = True,
    error_model: str = "poisson",
    empty_bin_value: float = float("nan"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hard-bin integrate with per-bin variance propagation.

    Returns ``(mean, sigma)`` each shape ``(n_eta, n_r)``.

    Parameters
    ----------
    error_model :
        ``'poisson'`` (default), ``'azimuthal'``, or ``'hybrid'`` — see
        module docstring.
    correction :
        Per-pixel multiplicative correction ``c_i`` such that the
        corrected intensity is ``I_i / c_i``. Variance is propagated
        consistently (``σ²/c²``). ``None`` (default) skips correction.
    empty_bin_value :
        Fill value for bins with zero in-range pixels. Default ``NaN``;
        pass ``0.0`` for the legacy silent-zero behaviour.
    """
    _validate_error_model(error_model)
    img_flat, var_flat = _prepare_pixel_arrays(
        image, geom,
        variance_image=variance_image,
        correction=correction,
        apply_trans_opt=apply_trans_opt,
    )

    n_bins = geom.n_eta * geom.n_r
    valid = geom.valid
    bins_v = geom.flat_bin[valid]
    img_v = img_flat[valid]
    var_v = var_flat[valid]

    sums = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    var_sums = torch.zeros_like(sums)
    counts = torch.zeros_like(sums)
    sums = sums.index_add(0, bins_v, img_v)
    var_sums = var_sums.index_add(0, bins_v, var_v)
    counts = counts.index_add(0, bins_v, torch.ones_like(img_v))

    safe_n = counts.clamp(min=1.0)
    mean = sums / safe_n
    var_poisson = var_sums / (safe_n * safe_n)

    if error_model == "poisson":
        var_bin = var_poisson
    else:
        # Sample variance of in-bin pixels (hard bin, w=1 so w²=1):
        #   σ²_azim_bin = Σ (I_i - μ_bin)² / N²
        resid_sq_sums = torch.zeros_like(sums)
        resid = img_v - mean[bins_v]
        resid_sq_sums = resid_sq_sums.index_add(0, bins_v, resid * resid)
        var_azim = resid_sq_sums / (safe_n * safe_n)
        if error_model == "azimuthal":
            var_bin = var_azim
        else:  # hybrid
            var_bin = torch.maximum(var_poisson, var_azim)

    sigma = torch.sqrt(var_bin)
    mean, sigma = _apply_empty_bin_fill(mean, sigma, counts, empty_bin_value)
    return mean.reshape(geom.n_eta, geom.n_r), sigma.reshape(geom.n_eta, geom.n_r)


def integrate_subpixel_with_variance(
    image: torch.Tensor,
    geom: SubpixelBinGeometry,
    *,
    variance_image: Optional[torch.Tensor] = None,
    correction: Optional[torch.Tensor] = None,
    apply_trans_opt: bool = True,
    error_model: str = "poisson",
    empty_bin_value: float = float("nan"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K×K subpixel integrate with per-bin variance propagation.

    Each subpixel contributes ``image[pix] · (1/K²)`` to its bin and
    ``var_image[pix] · (1/K²)²`` to the bin variance accumulator. See
    :func:`integrate_hard_with_variance` for the ``error_model``,
    ``correction`` and ``empty_bin_value`` parameters.
    """
    _validate_error_model(error_model)
    img_flat, var_flat = _prepare_pixel_arrays(
        image, geom,
        variance_image=variance_image,
        correction=correction,
        apply_trans_opt=apply_trans_opt,
    )

    n_bins = geom.n_eta * geom.n_r
    inv_K2 = 1.0 / (geom.K * geom.K)

    sums = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    var_sums = torch.zeros_like(sums)
    weight_sums = torch.zeros_like(sums)

    contrib  = img_flat * inv_K2
    var_contrib = var_flat * (inv_K2 ** 2)
    one_w   = torch.full_like(img_flat, inv_K2)
    for k in range(geom.K * geom.K):
        v = geom.valid[k]
        idx = geom.flat_bin[k][v]
        sums = sums.index_add(0, idx, contrib[v])
        var_sums = var_sums.index_add(0, idx, var_contrib[v])
        weight_sums = weight_sums.index_add(0, idx, one_w[v])

    safe_w = weight_sums.clamp(min=1e-30)
    mean = sums / safe_w
    var_poisson = var_sums / (safe_w * safe_w)

    if error_model == "poisson":
        var_bin = var_poisson
    else:
        # Each subpixel contributes (1/K²)² · (I_pix - μ_bin)²
        # to the residual sum; subpixels within the same pixel share I_pix.
        resid_sq_sums = torch.zeros_like(sums)
        w2 = inv_K2 ** 2
        for k in range(geom.K * geom.K):
            v = geom.valid[k]
            idx = geom.flat_bin[k][v]
            resid = img_flat[v] - mean[idx]
            resid_sq_sums = resid_sq_sums.index_add(0, idx, w2 * resid * resid)
        var_azim = resid_sq_sums / (safe_w * safe_w)
        if error_model == "azimuthal":
            var_bin = var_azim
        else:  # hybrid
            var_bin = torch.maximum(var_poisson, var_azim)

    sigma = torch.sqrt(var_bin)
    mean, sigma = _apply_empty_bin_fill(mean, sigma, weight_sums, empty_bin_value)
    return mean.reshape(geom.n_eta, geom.n_r), sigma.reshape(geom.n_eta, geom.n_r)


def integrate_polygon_with_variance(
    image: torch.Tensor,
    geom: PolygonBinGeometry,
    *,
    variance_image: Optional[torch.Tensor] = None,
    correction: Optional[torch.Tensor] = None,
    apply_trans_opt: bool = True,
    error_model: str = "poisson",
    empty_bin_value: float = float("nan"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Polygon-area integrate with per-bin variance propagation.

    Per-bin mean = ``Σ area_i · I_i / Σ area_i``; per-bin σ uses the
    area² weights for correct propagation of independent pixel
    variances. See :func:`integrate_hard_with_variance` for the
    ``error_model``, ``correction`` and ``empty_bin_value`` parameters.
    """
    _validate_error_model(error_model)
    img_flat, var_flat = _prepare_pixel_arrays(
        image, geom,
        variance_image=variance_image,
        correction=correction,
        apply_trans_opt=apply_trans_opt,
    )

    n_bins = geom.n_eta * geom.n_r
    pix_I   = img_flat[geom.pix_idx]
    pix_var = var_flat[geom.pix_idx]
    w  = geom.area
    w2 = w * w

    sums = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    var_sums = torch.zeros_like(sums)
    weight_sums = torch.zeros_like(sums)
    sums = sums.index_add(0, geom.bin_idx, pix_I * w)
    var_sums = var_sums.index_add(0, geom.bin_idx, pix_var * w2)
    weight_sums = weight_sums.index_add(0, geom.bin_idx, w)

    safe_w = weight_sums.clamp(min=1e-30)
    mean = sums / safe_w
    var_poisson = var_sums / (safe_w * safe_w)

    if error_model == "poisson":
        var_bin = var_poisson
    else:
        var_azim = _azimuthal_var_weighted(
            img_flat, geom.pix_idx, geom.bin_idx, w,
            mean, weight_sums, n_bins,
        )
        if error_model == "azimuthal":
            var_bin = var_azim
        else:  # hybrid
            var_bin = torch.maximum(var_poisson, var_azim)

    sigma = torch.sqrt(var_bin)
    mean, sigma = _apply_empty_bin_fill(mean, sigma, weight_sums, empty_bin_value)
    return mean.reshape(geom.n_eta, geom.n_r), sigma.reshape(geom.n_eta, geom.n_r)


__all__ = [
    "integrate_hard_with_variance",
    "integrate_subpixel_with_variance",
    "integrate_polygon_with_variance",
]
