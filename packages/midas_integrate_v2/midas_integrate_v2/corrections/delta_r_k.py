"""Per-ring radial offsets ``δr_k`` as a refinable :class:`nn.Module`.

Calibrate-v2's F2 fix introduces one offset per ring with a zero-sum
gauge ``Σδr_k = 0``. v1 had no per-ring concept in the radial map and
emitted the offsets as a JSON sidecar for downstream peak-fit consumers.
v2 brings them inside the integration kernel: the per-pixel R is
nudged by ``δr_k`` of the ring the pixel was assigned to, before soft
binning. The offsets are refinable jointly with geometry and the spline
correction.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def assign_ring(R_px: torch.Tensor, ring_R_centres_px: torch.Tensor) -> torch.Tensor:
    """For each pixel, return the index of the nearest ring centre.

    Differentiable in neither input — this is a hard assignment. Use the
    output to gather ``δr_k`` and add to ``R_px``; the gradient w.r.t.
    ``δr_k`` flows through the gather, w.r.t. ``R_px`` is unchanged.
    """
    # |R - centre|, shape (n_pix, n_rings); argmin over rings.
    d = (R_px.detach().unsqueeze(-1) - ring_R_centres_px.detach().unsqueeze(0)).abs()
    return d.argmin(dim=-1)


def delta_r_k_from_R(
    R_px: torch.Tensor,
    delta_r_k: torch.Tensor,
    ring_R_centres_px: torch.Tensor,
    *,
    capture_radius_px: Optional[float] = None,
) -> torch.Tensor:
    """Look up the per-ring offset for each pixel and return ``R + δ``.

    If ``capture_radius_px`` is provided, pixels farther than this from
    any ring centre receive zero offset (so noise pixels in inter-ring
    gaps don't pull on δr_k during refinement).
    """
    idx = assign_ring(R_px, ring_R_centres_px)
    delta = delta_r_k[idx]
    if capture_radius_px is not None:
        nearest = ring_R_centres_px[idx]
        in_band = (R_px.detach() - nearest).abs() < capture_radius_px
        delta = delta * in_band.to(delta.dtype)
    return R_px + delta


class PerRingOffsets(nn.Module):
    """Refinable ``δr_k`` with a zero-sum projection.

    Parameters
    ----------
    n_rings :
        Number of rings.
    enforce_zero_sum :
        If True, the forward applies ``δ_k → δ_k - mean(δ)`` so the gauge
        is automatic regardless of the optimiser's step. Mirrors the
        zero-sum residual constraint used in calibrate-v2's M-step.
    init :
        Optional initial values; defaults to zeros.
    """

    def __init__(self, n_rings: int, *,
                 enforce_zero_sum: bool = True,
                 init: Optional[torch.Tensor] = None,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        self.n_rings = int(n_rings)
        self.enforce_zero_sum = bool(enforce_zero_sum)
        self.delta_raw = nn.Parameter(
            torch.zeros(self.n_rings, dtype=dtype) if init is None
            else init.to(dtype=dtype).clone(),
        )

    @property
    def delta(self) -> torch.Tensor:
        """The (gauge-projected) δr_k vector. Differentiable in delta_raw."""
        if self.enforce_zero_sum:
            return self.delta_raw - self.delta_raw.mean()
        return self.delta_raw

    def forward(
        self,
        R_px: torch.Tensor,
        ring_R_centres_px: torch.Tensor,
        *,
        capture_radius_px: Optional[float] = None,
    ) -> torch.Tensor:
        """Return ``R_px + δ_{nearest_ring(R_px)}``."""
        return delta_r_k_from_R(
            R_px, self.delta, ring_R_centres_px,
            capture_radius_px=capture_radius_px,
        )


__all__ = ["PerRingOffsets", "delta_r_k_from_R", "assign_ring"]
