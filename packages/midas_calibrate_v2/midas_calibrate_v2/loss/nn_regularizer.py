"""Regularisation terms for the NN-augmented residual model.

Two contributions:
  - Weight decay (sum of squared weights).
  - Smoothness penalty: ∫|∇f(y, z)|² approximated via finite differences on a
    grid evaluation of the conv NN.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def weight_decay(model: nn.Module) -> torch.Tensor:
    """Sum of squared weights across all model parameters."""
    s = torch.zeros((), dtype=torch.float64)
    for p in model.parameters():
        if p.requires_grad:
            s = s + (p.float() ** 2).sum().double()
    return s


def smoothness_penalty(field: torch.Tensor) -> torch.Tensor:
    """Discrete |∇f|² over a [..., H, W] field.

    Uses central differences with reflect padding on the boundary.
    """
    if field.ndim < 2:
        raise ValueError(f"smoothness expects at least 2D field; got shape {field.shape}")
    f = field.double()
    dy = f[..., 1:, :] - f[..., :-1, :]
    dz = f[..., :, 1:] - f[..., :, :-1]
    return (dy ** 2).sum() + (dz ** 2).sum()


__all__ = ["weight_decay", "smoothness_penalty"]
