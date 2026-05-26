"""Mixture deconvolution: recover non-negative component weights from a sum.

Domain-agnostic core for "recover a distribution over a discrete component grid"
problems -- thickness distributions, grain-size distributions, ODF on a grid,
spectra over an energy grid.  Given a basis matrix (one row per component) and
an observed signal that is a non-negative mixture of the rows, fit softmax
weights so the mixture matches the observation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["mixture_deconvolution"]


def mixture_deconvolution(observed, basis, *, loss="cosine", steps=800, lr=0.05,
                          entropy_weight=0.0):
    """Fit softmax weights ``w`` so ``w @ basis`` matches ``observed``.

    Parameters
    ----------
    observed : tensor (L,)
        The measured mixture.
    basis : tensor (K, L)
        One row per component (e.g. the pattern for each candidate thickness).
    loss : {"cosine", "rel_l2"}
        Scale-invariant shape loss, or scale-robust L2.
    entropy_weight : float
        Encourage a smoother (higher-entropy) distribution if > 0.

    Returns
    -------
    tensor (K,) : the recovered, normalised component weights.
    """
    import torch
    from .optimize import cosine_loss, fit, relative_l2_loss

    basis = torch.as_tensor(basis)
    observed = torch.as_tensor(observed, dtype=basis.dtype, device=basis.device)
    K = basis.shape[0]
    logits = torch.zeros(K, dtype=basis.dtype, device=basis.device, requires_grad=True)
    lf = cosine_loss if loss == "cosine" else relative_l2_loss

    def loss_fn():
        w = torch.softmax(logits, dim=0)
        pred = w @ basis
        out = lf(pred, observed)
        if entropy_weight > 0:
            ent = -(w * (w + 1e-12).log()).sum()
            out = out - entropy_weight * ent
        return out

    fit([logits], loss_fn, steps=steps, lr=lr)
    return torch.softmax(logits.detach(), dim=0)
