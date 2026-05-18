"""Q-mode bin edges, differentiable in Lsd and wavelength.

Mirror of :func:`midas_integrate.geometry.build_q_bin_edges_in_R` but in
torch so the bin edges themselves can be refined or used in a
gradient-aware loss (e.g. when wavelength or Lsd is being learned).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def build_q_bin_edges_in_R(
    *,
    QMin: float, QMax: float, QBinSize: float,
    Lsd: torch.Tensor, px: torch.Tensor, wavelength_A: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Convert a uniform Q grid to R bin edges (pixel units).

    Uses ``Q = (4π/λ) sin(θ)`` and ``R = (Lsd / px) tan(2θ)``.
    Differentiable in ``Lsd, px, wavelength_A``.
    """
    dt = Lsd.dtype
    dev = Lsd.device
    n_r = int(math.ceil((QMax - QMin) / QBinSize))
    q_lo = QMin + QBinSize * torch.arange(n_r, dtype=dt, device=dev)
    q_hi = q_lo + QBinSize
    # 2θ = 2 arcsin(Qλ / 4π); clip the argument to [-1, 1] to avoid NaN
    # at the edge of physical Q range.
    arg_lo = q_lo * wavelength_A / (4.0 * math.pi)
    arg_hi = q_hi * wavelength_A / (4.0 * math.pi)
    two_theta_lo = 2.0 * torch.asin(torch.clamp(arg_lo, -1.0, 1.0))
    two_theta_hi = 2.0 * torch.asin(torch.clamp(arg_hi, -1.0, 1.0))
    r_lo = (Lsd / px) * torch.tan(two_theta_lo)
    r_hi = (Lsd / px) * torch.tan(two_theta_hi)
    return r_lo, r_hi, n_r


__all__ = ["build_q_bin_edges_in_R"]
