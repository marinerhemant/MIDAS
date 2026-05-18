"""Profile-target losses: match an observed 1D profile to a reference.

The 2D-input variants reduce over η internally so the user just supplies
the reference 1D profile.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..diff.soft_bin import profile_1d_diff
from ..spec import IntegrationSpec


class ProfileMSELoss(nn.Module):
    """Mean-squared error between a soft-binned profile and a reference.

    Forward signature:
        ``loss(int2d, spec, reference_1d) -> scalar``

    Where ``reference_1d`` is a tensor of shape ``(n_r,)`` containing
    the target profile. ``int2d`` is the ``(n_eta, n_r)`` output of
    :func:`integrate_with_corrections` or :func:`integrate_diff`.
    """

    def forward(
        self,
        int2d: torch.Tensor,
        spec: IntegrationSpec,
        reference_1d: torch.Tensor,
    ) -> torch.Tensor:
        prof = profile_1d_diff(int2d, spec)
        if prof.shape != reference_1d.shape:
            raise ValueError(
                f"profile shape {tuple(prof.shape)} does not match "
                f"reference shape {tuple(reference_1d.shape)}"
            )
        return ((prof - reference_1d) ** 2).mean()


class ProfileWeightedMSELoss(nn.Module):
    """Per-bin-weighted MSE — useful when some R bins are noise-dominated.

    Weights of zero exclude that bin from both the loss and the gradient.
    Negative weights are clamped to zero (a no-op).
    """

    def forward(
        self,
        int2d: torch.Tensor,
        spec: IntegrationSpec,
        reference_1d: torch.Tensor,
        weights_1d: torch.Tensor,
    ) -> torch.Tensor:
        prof = profile_1d_diff(int2d, spec)
        if prof.shape != reference_1d.shape:
            raise ValueError(
                f"profile shape {tuple(prof.shape)} does not match "
                f"reference shape {tuple(reference_1d.shape)}"
            )
        if prof.shape != weights_1d.shape:
            raise ValueError(
                f"profile shape {tuple(prof.shape)} does not match "
                f"weights shape {tuple(weights_1d.shape)}"
            )
        w = weights_1d.clamp(min=0.0)
        wsum = w.sum() + 1e-30
        return (w * (prof - reference_1d) ** 2).sum() / wsum


__all__ = ["ProfileMSELoss", "ProfileWeightedMSELoss"]
