"""Differentiable per-pixel mask — auto-learn bad pixels via gradient descent.

A static mask (``binning/mask.py``) requires the user to know in advance
which pixels are bad: beam stop, dead pixels, module gaps, hot pixels.
For the latter two the user often *doesn't* know — hot pixels drift over
time, and dead-pixel maps from the manufacturer go stale.

The differentiable mask treats per-pixel inclusion weight as a
**learnable parameter**. Each pixel ``i`` has a logit
``raw_logit_i``; its weight is ``w_i = sigmoid(raw_logit_i) ∈ (0, 1)``.
At integrate time the image is multiplied element-wise by ``w``.

Training: jointly minimise (a) a data loss like
:class:`EtaUniformityLoss` (the geometric calibration signal) and
(b) a sparsity prior like ``λ · Σ (1 - w_i)²`` (encourages ``w_i = 1``
unless the data clearly says otherwise). After convergence:

- **Bad pixels** had a large effect on the data loss; the optimiser
  drove their ``w_i → 0`` to remove them.
- **Good pixels** stayed at ``w_i ≈ 1`` because the sparsity prior
  pinned them there.

A hard mask can then be extracted via ``mask.extract_hard_mask(threshold)``
and passed to any other binning geometry for production integration.

This is the MIDAS automation play — pyFAI / dxchange / DPDAK have no
analogue; their masks are user-supplied static arrays.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LearnableMask(nn.Module):
    """Per-pixel learnable inclusion weight in (0, 1) via sigmoid.

    Parameters
    ----------
    NrPixelsZ, NrPixelsY :
        Detector dimensions.
    init_weight :
        Initial inclusion weight for every pixel. Default 0.99 means
        "everything starts as kept; the optimiser only pushes pixels
        down when the data demands it". Lower init (e.g. 0.5) means
        every pixel starts uncertain; the optimiser learns both
        directions.
    static_mask :
        Optional bool tensor of pixels to FORCE to weight zero (e.g.
        beam stop you already know). These pixels are pinned at
        ``w = 0`` and don't participate in gradient updates.
    dtype :
        torch dtype of the learnable parameter (and the returned weights).
    """

    def __init__(
        self,
        NrPixelsZ: int,
        NrPixelsY: int,
        *,
        init_weight: float = 0.99,
        static_mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if not (0.0 < init_weight < 1.0):
            raise ValueError(
                f"init_weight must be in (0, 1), got {init_weight}"
            )
        init_logit = math.log(init_weight / (1.0 - init_weight))
        self.raw_logits = nn.Parameter(
            torch.full((NrPixelsZ, NrPixelsY), init_logit, dtype=dtype)
        )
        if static_mask is not None:
            if static_mask.shape != (NrPixelsZ, NrPixelsY):
                raise ValueError(
                    f"static_mask shape {tuple(static_mask.shape)} must be "
                    f"({NrPixelsZ}, {NrPixelsY})"
                )
            self.register_buffer("static_mask", static_mask.to(torch.bool))
        else:
            self.static_mask = None

    def forward(self) -> torch.Tensor:
        """Return per-pixel weights of shape ``(NrPixelsZ, NrPixelsY)``."""
        w = torch.sigmoid(self.raw_logits)
        if self.static_mask is not None:
            # Force statically-masked pixels to zero (no gradient through them)
            w = w * (~self.static_mask).to(w.dtype)
        return w

    def extract_hard_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Return a bool ndarray (True = masked) from the learned weights.

        Use this to convert the trained mask into a hard mask for
        production integration via any other binning geometry.
        """
        with torch.no_grad():
            return (self.forward().detach().cpu().numpy() < threshold)

    def n_low_weight_pixels(self, threshold: float = 0.5) -> int:
        """How many pixels currently have weight below ``threshold``."""
        return int(self.extract_hard_mask(threshold).sum())


def sparsity_prior(
    mask: LearnableMask,
    *,
    weight: float = 1.0,
    target: float = 1.0,
) -> torch.Tensor:
    """Quadratic prior pulling each weight toward ``target`` (default 1.0).

    ``loss_prior = weight · mean((mask - target)²)``

    With ``target=1.0`` and ``weight`` large, almost no pixel gets masked.
    With ``weight`` small, the data loss can drive bad pixels to zero
    cheaply. Tune ``weight`` so that the cost of masking a *good* pixel
    is just larger than the data-loss reduction it'd produce — then only
    truly-bad pixels are masked.
    """
    w = mask()
    return weight * ((w - target) ** 2).mean()


def smoothness_prior(
    mask: LearnableMask,
    *,
    weight: float = 1.0,
) -> torch.Tensor:
    """Total-variation prior — neighbouring pixels prefer similar weights.

    Encodes the physical observation that bad pixels often cluster
    (a dead row, an entire module gap). Use alongside the sparsity
    prior to suppress isolated false-positive masking on noise.
    """
    w = mask()
    dy = (w[:, 1:] - w[:, :-1]).abs().mean()
    dz = (w[1:, :] - w[:-1, :]).abs().mean()
    return weight * (dy + dz)


__all__ = ["LearnableMask", "sparsity_prior", "smoothness_prior"]
