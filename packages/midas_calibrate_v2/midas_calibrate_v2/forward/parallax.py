"""Differentiable parallax correction.

The v1 implementation contained an ``if parallax.abs().item() > 0:`` guard
that broke autograd when parallax was being refined and started at zero.
This module fixes that by always applying the parallax term — when the
amplitude is zero, ``sin(0) = 0`` produces a clean zero gradient without a
graph break.
"""
from __future__ import annotations

import torch


def parallax_correction(
    R_px: torch.Tensor,
    rad_um: torch.Tensor,
    Lsd: torch.Tensor,
    parallax: torch.Tensor,
    px: torch.Tensor,
) -> torch.Tensor:
    """Return ``R_px + parallax · sin(2θ) / px``.

    Parameters
    ----------
    R_px : tensor
        Predicted radius in pixels (after distortion).
    rad_um : tensor
        Pre-distortion projected radius (μm).  Used to compute 2θ via
        ``2θ = atan(rad_um / Lsd)``.
    Lsd, parallax, px : tensors
        Geometry + parallax depth, in μm.

    Notes
    -----
    Always applied; parallax = 0 produces no offset and a clean zero gradient
    (no Python ``if`` branch).
    """
    two_theta = torch.atan(rad_um / Lsd)
    return R_px + parallax * torch.sin(two_theta) / px


__all__ = ["parallax_correction"]
