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
    ``rank_eff`` (eigenvalues above the pinv floor), ``is_positive_definite``
    and ``n_negative_eigvals``.

    **A non-positive-definite Hessian is reported, not hidden.** At a saddle the
    Laplace approximation is not merely imprecise, it is invalid: there is no
    Gaussian posterior whose width could be quoted. Components with a
    non-positive posterior variance therefore get ``std = nan``, and
    ``is_positive_definite`` is False. Earlier versions clamped the variance at
    zero, which reported the WORST-conditioned direction as the most confident.
    Check ``is_positive_definite`` before reading ``std``.
    """
    import torch
    from torch.autograd.functional import hessian

    theta0 = torch.as_tensor(theta).detach().clone().double()
    H = hessian(lambda t: loss_fn(t), theta0)
    H_sym = 0.5 * (H + H.T) / float(noise_var)
    eigvals = torch.linalg.eigvalsh(H_sym)
    eps_floor = max(1e-30, float(eigvals.abs().max().item()) * pinv_rtol)
    rank_eff = int((eigvals > eps_floor).sum().item())
    n_negative = int((eigvals < -eps_floor).sum().item())
    cov = torch.linalg.pinv(H_sym, rtol=pinv_rtol)
    # NaN (not 0.0) for a variance that does not exist: it propagates into
    # anything built from it, which is the honest behaviour for a quantity the
    # approximation cannot supply.
    var = torch.diag(cov)
    ok = var > 0.0
    var_safe = torch.where(ok, var, torch.ones_like(var))   # safe INPUT to sqrt
    std = torch.where(ok, var_safe.sqrt(), torch.full_like(var, float("nan")))
    cond = float(eigvals.max() / max(eigvals.min().item(), eps_floor))
    return {"cov": cov, "std": std, "sigma": std, "hessian": H_sym,
            "eigvals": eigvals, "cond_number": cond, "rank_eff": rank_eff,
            "is_positive_definite": n_negative == 0,
            "n_negative_eigvals": n_negative}
