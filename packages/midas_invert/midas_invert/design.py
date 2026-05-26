"""Fisher-information experiment design (which measurement to take next).

Domain-agnostic: given a differentiable forward ``f(theta) -> per-measurement
predictions`` and a target scalar parameter, rank candidate measurements by the
Fisher information ``(d f / d theta)^2 / sigma^2``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["sensitivity", "fisher_information", "rank_measurements",
           "next_best_measurement", "jacobian", "information_matrix",
           "greedy_optimal_design"]


def sensitivity(forward_fn, theta):
    """``d forward_fn(theta) / d theta`` (per-measurement), via autograd."""
    import torch
    theta = torch.as_tensor(theta, dtype=torch.float64).clone().requires_grad_(True)
    pred = forward_fn(theta)
    if pred.numel() > 1:
        J = torch.zeros(pred.numel(), dtype=pred.dtype)
        for i in range(pred.numel()):
            ti = torch.as_tensor(theta.detach(), dtype=torch.float64).requires_grad_(True)
            J[i] = torch.autograd.grad(forward_fn(ti)[i], ti)[0]
        return J
    return torch.autograd.grad(pred.sum(), theta)[0].reshape(())


def fisher_information(forward_fn, theta, *, sigma=1.0):
    """Per-measurement Fisher information ``(d pred/d theta)^2 / sigma^2``."""
    import torch
    J = sensitivity(forward_fn, theta)
    sig = torch.as_tensor(sigma, dtype=J.dtype)
    return (J ** 2) / (sig ** 2)


def rank_measurements(forward_fn, theta, labels=None, *, sigma=1.0):
    """Measurement (index, info, label) tuples sorted by info, most first."""
    import torch
    fi = fisher_information(forward_fn, theta, sigma=sigma)
    order = torch.argsort(fi, descending=True)
    return [(int(i), float(fi[i]), (labels[i] if labels is not None else int(i)))
            for i in order]


def next_best_measurement(forward_fn, theta, labels=None, *, sigma=1.0):
    """The single most informative measurement about ``theta``."""
    return rank_measurements(forward_fn, theta, labels=labels, sigma=sigma)[0]


# ---------------------------------------------------- multi-parameter design

def jacobian(forward_fn, theta_vec):
    """Jacobian ``J[m, p] = d forward_fn(theta)[m] / d theta[p]`` (M x P).

    ``forward_fn(theta_vec) -> (M,)`` per-measurement predictions for a
    parameter VECTOR theta_vec (P,).
    """
    import torch
    from torch.autograd.functional import jacobian as _jac
    theta = torch.as_tensor(theta_vec, dtype=torch.float64)
    return _jac(lambda t: forward_fn(t), theta)              # (M, P)


def information_matrix(forward_fn, theta_vec, *, sigma=1.0, per_measurement=False):
    """Fisher information matrix ``F = sum_m J_m J_m^T / sigma^2`` (P x P).

    With ``per_measurement=True`` returns the per-measurement (M, P, P) stack
    (used by greedy design).
    """
    import torch
    J = jacobian(forward_fn, theta_vec)                      # (M, P)
    sig2 = float(sigma) ** 2
    fims = torch.einsum("mp,mq->mpq", J, J) / sig2           # (M, P, P)
    if per_measurement:
        return fims
    return fims.sum(dim=0)


def greedy_optimal_design(forward_fn, theta_vec, k, *, sigma=1.0,
                          prior_precision=1e-6, labels=None):
    """Greedily pick ``k`` measurements maximising D-optimality (log det of the
    accumulated Fisher information) for the parameter vector ``theta_vec``.

    Returns the list of selected ``(index, label)`` in selection order.
    """
    import torch
    fims = information_matrix(forward_fn, theta_vec, sigma=sigma, per_measurement=True)
    M, P, _ = fims.shape
    F = prior_precision * torch.eye(P, dtype=fims.dtype)
    chosen, remaining = [], set(range(M))
    for _ in range(min(k, M)):
        best_i, best_logdet = None, -float("inf")
        for i in remaining:
            ld = torch.linalg.slogdet(F + fims[i])[1].item()
            if ld > best_logdet:
                best_logdet, best_i = ld, i
        F = F + fims[best_i]
        remaining.discard(best_i)
        chosen.append((best_i, labels[best_i] if labels is not None else best_i))
    return chosen
