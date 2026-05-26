"""Laplace-approximation uncertainty (Hessian at the optimum).

Canonical shared implementation: at a converged MAP point the negative
log-likelihood is locally quadratic, so the posterior covariance is the inverse
Hessian.  Uses a pseudo-inverse (robust to rank-deficiency) and reports
eigen-diagnostics (condition number, effective rank) for identifiability
analysis.  ``laue_torch`` delegates to this (one shared implementation across
HEDM / Laue / 2D).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["laplace_uncertainty"]


def laplace_uncertainty(loss_fn, theta, *, noise_var=1.0, pinv_rtol=1e-9):
    """Laplace posterior for a flat parameter vector at ``theta``.

    ``loss_fn(theta)`` is the (per-data-point mean) loss; ``noise_var`` scales
    MSE-loss curvature to log-likelihood curvature under Gaussian noise
    (``H_ll = 0.5 (H + H^T) / noise_var``; a good plug-in choice is the converged
    loss value).  Covariance is ``pinv(H_ll)``.

    Returns dict with ``cov`` (P x P), ``std`` / ``sigma`` (P,), ``hessian``
    (the scaled, symmetrised H_ll), ``eigvals`` (ascending), ``cond_number``,
    and ``rank_eff`` (eigenvalues above the pinv floor).
    """
    import torch
    from torch.autograd.functional import hessian

    theta0 = torch.as_tensor(theta).detach().clone().double()
    H = hessian(lambda t: loss_fn(t), theta0)
    H_sym = 0.5 * (H + H.T) / float(noise_var)
    eigvals = torch.linalg.eigvalsh(H_sym)
    eps_floor = max(1e-30, float(eigvals.abs().max().item()) * pinv_rtol)
    rank_eff = int((eigvals > eps_floor).sum().item())
    cov = torch.linalg.pinv(H_sym, rtol=pinv_rtol)
    std = torch.sqrt(torch.clamp(torch.diag(cov), min=0.0))
    cond = float(eigvals.max() / max(eigvals.min().item(), eps_floor))
    return {"cov": cov, "std": std, "sigma": std, "hessian": H_sym,
            "eigvals": eigvals, "cond_number": cond, "rank_eff": rank_eff}
