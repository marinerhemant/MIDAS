"""Geometry-driven losses: exploit physical structure of calibrant rings.

Two recipes:

- :class:`EtaUniformityLoss` — a Debye-Scherrer ring is uniform along η
  for a true polycrystalline calibrant. Variance across η at each R bin
  is therefore a direct signal of geometric mis-calibration (a tilted
  detector, a mis-set BC, or distortion all break the uniformity in
  characteristic ways). The loss is the mean of per-bin variances,
  weighted by per-bin intensity so noise-only bins don't dominate.
- :class:`PeakPositionLoss` — observed peak centroid R must match the
  predicted ring R from ``R_ideal = (Lsd / px) · tan(2θ)`` with
  ``2θ = 2 arcsin(λ / 2d)``. This is the workhorse loss in
  ``midas-calibrate-v2``'s M-step; we expose a profile-side variant here
  so it composes with corrections (δr_k, spline) inside the integration
  pipeline.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn


class EtaUniformityLoss(nn.Module):
    """Sum of intensity-weighted per-R-bin variances of the integrated η profile.

    Specifically, for each R bin ``r``:
        ``v_r = Var_η(int2d[:, r])``
        ``w_r = Sum_η(int2d[:, r])``      (clamped to ≥ 0)
        ``loss = Σ_r w_r · v_r / Σ_r w_r``

    Bins with zero intensity contribute zero to both numerator and
    denominator (no division by zero). Differentiable in every
    upstream parameter that affects pixel→bin assignments.

    Parameters
    ----------
    r_indices :
        Optional iterable of R-bin indices to include. Defaults to
        every bin (``None`` ↔ slice ``[:]``).
    intensity_floor :
        Bins whose ``Sum_η`` is below this floor receive zero weight.
        Helps the loss ignore inter-ring gaps and saturation bands.
    """

    def __init__(
        self,
        r_indices: Optional[Iterable[int]] = None,
        intensity_floor: float = 0.0,
    ):
        super().__init__()
        self.r_indices = (None if r_indices is None
                          else torch.tensor(list(r_indices), dtype=torch.long))
        self.intensity_floor = float(intensity_floor)

    def forward(self, int2d: torch.Tensor) -> torch.Tensor:
        # Subset over R if requested
        if self.r_indices is not None:
            sub = int2d.index_select(dim=1, index=self.r_indices.to(int2d.device))
        else:
            sub = int2d
        n_eta = sub.shape[0]
        intensity = sub.sum(dim=0).clamp(min=0.0)
        # Per-bin mean and variance over η
        mean = sub.mean(dim=0)
        var = ((sub - mean.unsqueeze(0)) ** 2).sum(dim=0) / max(1, n_eta - 1)
        w = (intensity > self.intensity_floor).to(int2d.dtype) * intensity
        wsum = w.sum() + 1e-30
        return (w * var).sum() / wsum


class PeakPositionLoss(nn.Module):
    """Distance between observed centroid R and predicted ring R.

    For each ring (predicted ``R_pred``), the loss takes a window of R
    bins around it, computes the intensity-weighted centroid of the
    integrated profile inside the window, and compares to ``R_pred``.

    Forward signature:
        ``loss(int2d, spec, ring_R_centres_px, window_px=2.0) -> scalar``

    The loss is the mean of squared centroid-vs-prediction errors over
    rings. Differentiable in the spec parameters that affect bin
    assignments (Lsd, BC, tilts, distortion).
    """

    def forward(
        self,
        int2d: torch.Tensor,
        spec,                            # IntegrationSpec
        ring_R_centres_px: torch.Tensor,
        window_px: float = 2.0,
    ) -> torch.Tensor:
        # Per-bin R centres along the R axis (shape (n_r,))
        n_r = int2d.shape[1]
        r_centres = (
            torch.linspace(
                spec.RMin + spec.RBinSize / 2.0,
                spec.RMax - spec.RBinSize / 2.0,
                n_r,
                dtype=int2d.dtype, device=int2d.device,
            )
        )
        prof_1d = int2d.sum(dim=0)            # collapse η
        errs = []
        for k in range(int(ring_R_centres_px.shape[0])):
            R_pred = ring_R_centres_px[k]
            in_win = (r_centres - R_pred.detach()).abs() < window_px
            mask = in_win.to(prof_1d.dtype)
            wprof = prof_1d * mask
            wsum = wprof.sum() + 1e-12
            centroid = (wprof * r_centres).sum() / wsum
            errs.append((centroid - R_pred) ** 2)
        if not errs:
            return torch.tensor(0.0, dtype=int2d.dtype, device=int2d.device)
        return torch.stack(errs).mean()


__all__ = ["EtaUniformityLoss", "PeakPositionLoss"]
