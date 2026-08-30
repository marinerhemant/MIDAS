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


def eta_is_full_circle(spec, *, tol: float = 1e-9) -> bool:
    """Does this spec's eta range close on itself?

    Only then may eta be soft-binned periodically: the outer half-bins at
    EtaMin and EtaMax are neighbours across the seam, so nothing should be
    dropped there. For a wedge (a partial eta range, e.g. a DAC gasket opening)
    the ends really are ends and wrapping would fold opposite sides of the
    detector onto each other.
    """
    span = float(spec.EtaMax) - float(spec.EtaMin)
    return abs(abs(span) - 360.0) <= tol


def soft_bin_indices_weights(
    R: torch.Tensor,
    *,
    R_min: float, R_bin_size: float, n_r: int,
    periodic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear-interpolation soft binning onto bin CENTRES.

    Each value is distributed between the two nearest bin centres, weighted by
    distance. The interpolation nodes are the bin centres
    ``R_min + R_bin_size·(k + 0.5)`` — the same axis
    :func:`midas_integrate_v2.io.v1_outputs.r_axis_from_spec` reports and the
    hard kernel bins to.

    Returns ``(b0, b1, w0, w1)``, all shaped like ``R``.

    .. warning::

       **Two defects fixed on 2026-08-29; both changed results.**

       *Half-bin offset.* The nodes used to be at ``(R - R_min)/R_bin_size``,
       i.e. at bin LOWER EDGES, while the reported R axis is bin centres. The
       weighted mean index of a value at R was then exactly
       ``(R - R_min)/R_bin_size``, so it was reported at ``R + 0.5·R_bin_size``.
       Measured against :func:`integrate_hard` on three sharp rings with
       ``RBinSize = 2`` px: the soft path put them at 61.02 / 121.01 / 181.01 px
       where hard put them at 60.05 / 120.05 / 180.05 — a systematic
       **+0.96 px**, half a bin, at every radius. That is a bias on the
       DIFFERENTIABLE path, i.e. the one anything refines geometry through.

       *Dropped last bin.* The in-range test was ``b0 < n_r - 1``, which zeroed
       BOTH weights for a value landing in the final bin instead of depositing
       its ``1 - frac`` share. With ``n_r = 10`` a uniform population lost
       **10.0 %** of its total weight (179 992 of 200 000), and the last bin
       received only the spill from its neighbour. ``w0`` and ``w1`` now carry
       independent masks, so only the part that genuinely falls outside the
       domain is dropped.

    Parameters
    ----------
    periodic :
        Treat the axis as a closed circle: the bin below the first centre IS
        the last bin, and the bin above the last centre IS the first. Set for
        **η** when the η range covers the full 360° — η is periodic, so the two
        outer half-bins are neighbours across the seam and nothing should be
        lost there. Leave False for R, and for a partial η range (a wedge),
        where the ends genuinely are ends.

        Added 2026-08-29. Without it the outer half-bins at ``EtaMin`` and
        ``EtaMax`` each dropped the share that belonged to their wrapped
        neighbour, i.e. a seam of missing intensity at ±180°.
    """
    # Nodes at bin CENTRES: subtracting 0.5 makes rf = k exactly when R sits at
    # the centre of bin k, so that value deposits wholly into bin k.
    rf = (R - R_min) / R_bin_size - 0.5
    b0 = torch.floor(rf).to(torch.long)
    b1 = b0 + 1
    frac = rf - b0.to(rf.dtype)                          # ∈ [0, 1)
    if periodic:
        # A closed circle: the neighbour below the first centre is the LAST
        # bin, and above the last centre is the FIRST. Nothing falls off an
        # end because there are no ends, so both weights are kept whole.
        return b0 % n_r, b1 % n_r, 1.0 - frac, frac
    # Independent masks: a value beside the first or last centre still deposits
    # the share that lands INSIDE the domain. Only the outward share is lost.
    m0 = ((b0 >= 0) & (b0 < n_r)).to(rf.dtype)
    m1 = ((b1 >= 0) & (b1 < n_r)).to(rf.dtype)
    b0_clamped = b0.clamp(0, n_r - 1)
    b1_clamped = b1.clamp(0, n_r - 1)
    w0 = (1.0 - frac) * m0
    w1 = frac * m1
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
        periodic=eta_is_full_circle(spec),
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
    "eta_is_full_circle",
    "soft_bin_indices_weights",
    "integrate_diff",
    "profile_1d_diff",
]
