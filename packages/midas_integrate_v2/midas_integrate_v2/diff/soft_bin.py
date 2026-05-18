"""Soft-binning differentiable integration kernel.

The hard-binning forward in :mod:`midas_integrate_v2.kernels` is
bit-identical to v1 but produces zero gradient w.r.t. the parameters
that determine bin assignments — :func:`torch.floor` is non-differentiable
and the bin indices are integer-valued anyway.

This module provides a soft-binning alternative: each pixel's intensity
is distributed linearly between the two nearest bins along R (and
optionally along η for 2D). The weights are smooth in ``(R, η)``, so
gradient flows from ``Profile[b]`` back through ``R, η`` and onward to
any refinable field of the :class:`IntegrationSpec`.

The soft and hard kernels agree when every pixel sits at a bin centre;
they differ in how intensity is split between adjacent bins. For
calibration-style refinements (where the loss is on smoothly-varying
profile features) the difference is well below noise; for parity vs v1
on a pre-built ``Map.bin``, use the hard path.
"""
from __future__ import annotations

from typing import Tuple

import torch

from ..forward import eval_pixel_REta
from ..spec import IntegrationSpec


def soft_bin_indices_weights(
    R: torch.Tensor,
    *,
    R_min: float, R_bin_size: float, n_r: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear-interpolation soft binning.

    For each ``R``: the intensity is distributed between bin ``b0 = floor((R - R_min) / R_bin_size)``
    (weight ``1 - frac``) and bin ``b1 = b0 + 1`` (weight ``frac``).

    Out-of-range bins are clamped — the corresponding weight is multiplied
    by an in-range mask so out-of-range pixels contribute zero (and do
    not pull gradient through clamped bin indices).

    Returns ``(b0, b1, w0, w1)``, all shaped like ``R``.
    """
    rf = (R - R_min) / R_bin_size                        # fractional bin
    b0 = torch.floor(rf).to(torch.long)
    in_range = (b0 >= 0) & (b0 < n_r - 1)
    b0_clamped = b0.clamp(0, n_r - 1)
    b1_clamped = (b0 + 1).clamp(0, n_r - 1)
    frac = rf - b0.to(rf.dtype)                          # ∈ [0, 1)
    mask = in_range.to(rf.dtype)
    w0 = (1.0 - frac) * mask
    w1 = frac * mask
    return b0_clamped, b1_clamped, w0, w1


def integrate_diff(
    image: torch.Tensor,
    spec: IntegrationSpec,
    *,
    mode: str = "linear",
) -> torch.Tensor:
    """Soft-bin integrate ``image`` to a 2D ``(n_eta, n_r)`` array.

    Differentiable in every refinable field of ``spec`` (so the result
    can drive a loss whose gradient flows back to ``Lsd, BC_y/z, tilts,
    distortion, Parallax``).

    ``mode='linear'`` distributes each pixel between two adjacent R bins
    and two adjacent η bins (4 contributions per pixel total). This is
    the only mode in Phase 2; ``'gaussian'`` is reserved for a future
    smoothing kernel.

    Returns a tensor of shape ``(n_eta, n_r)`` with ``image.dtype`` and
    ``image.device``.
    """
    if mode != "linear":
        raise NotImplementedError(f"soft-binning mode {mode!r} not implemented")

    R, Eta = eval_pixel_REta(spec)              # (NZ, NY)
    R = R.reshape(-1)
    Eta = Eta.reshape(-1)
    img = image.to(dtype=R.dtype).reshape(-1)

    n_r, n_eta = spec.n_r_bins, spec.n_eta_bins
    rb0, rb1, rw0, rw1 = soft_bin_indices_weights(
        R, R_min=spec.RMin, R_bin_size=spec.RBinSize, n_r=n_r,
    )
    eb0, eb1, ew0, ew1 = soft_bin_indices_weights(
        Eta, R_min=spec.EtaMin, R_bin_size=spec.EtaBinSize, n_r=n_eta,
    )

    # Use index_add into a flat (n_eta * n_r) buffer; differentiable
    # because index_add propagates gradient to its source.
    flat = torch.zeros(n_eta * n_r, dtype=img.dtype, device=img.device)
    for ei, ew in ((eb0, ew0), (eb1, ew1)):
        for ri, rw in ((rb0, rw0), (rb1, rw1)):
            idx = ei * n_r + ri
            flat = flat.index_add(0, idx, img * ew * rw)
    return flat.reshape(n_eta, n_r)


def profile_1d_diff(
    int2d: torch.Tensor,
    spec: IntegrationSpec,
    *,
    mode: str = "mean",
) -> torch.Tensor:
    """Reduce a soft-binned ``(n_eta, n_r)`` to a 1D ``(n_r,)`` profile.

    ``mode='mean'`` averages over η bins that received any intensity
    (matching the spirit of v1's ``simple_mean`` reducer). Differentiable.
    """
    if mode != "mean":
        raise NotImplementedError(f"reducer {mode!r} not implemented")
    # Avoid `where(counts > 0, …)` because the bool branch can produce
    # spurious NaN gradients when the inactive branch hits a divide-by-zero
    # internally. Add a tiny floor instead — safe because counts are
    # non-negative integers in practice.
    counts = (int2d.detach() != 0).to(int2d.dtype).sum(dim=0)
    return int2d.sum(dim=0) / (counts + 1e-12)


__all__ = [
    "soft_bin_indices_weights",
    "integrate_diff",
    "profile_1d_diff",
]
