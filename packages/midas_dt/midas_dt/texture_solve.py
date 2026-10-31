"""Positivity-constrained solver for the node-basis texture operator (:mod:`midas_dt.texture_nodes`).

Minimises ``sum_i w_i rho(A c - y)_i + l1 * sum(c)`` over ``c >= 0`` by FISTA (Beck & Teboulle 2009): a gradient step on the data term
from the extrapolated point, followed by the proximal map of ``l1 * sum(c) + indicator(c >= 0)``, which is ``max(v - l1 / L, 0)``.
``rho`` is the Huber function (``delta`` in data units) or the squared loss.

**Why positivity and not a smoothness prior.** Diffraction sees only the even part of the ODF, so odd-``l`` components are a null space
(known-limits entry KL-006); a non-negative expansion on nodes removes it by construction, because the full function is
non-negative. That is the only regulariser that is always on. **The iteration count is the other**: with a bounded number of FISTA
steps from ``c = 0`` the solution is an early-stopped one, and the count is a model choice that must be reported, not tuned per
sample. ``l1`` is optional and off by default.

**Step size.** ``L`` is the largest eigenvalue of ``A^T W A`` by power iteration. ``A`` has non-negative entries, so ``A^T W A`` is a
non-negative matrix and the Perron vector is non-negative: starting from a positive vector the iteration cannot cancel, and the estimate
converges from below (hence the ``safety`` factor, 1.1 by default, as a Huber loss has curvature at most ``w``).

**What is recorded.** ``objective[k]`` is the full objective at the iterate ``c_k`` (``objective[0]`` at the start), exact -- the extrapolated
forward image is obtained by linearity from the iterates' images, so each iteration costs one forward and one adjoint.
``n_increases`` counts iterations that increased it: FISTA is not monotone, a few are normal, a systematic rise means ``L`` is too small.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SolveResult", "fista_nonneg", "estimate_lipschitz"]


def _torch():
    import torch
    return torch


@dataclass
class SolveResult:
    """Output of :func:`fista_nonneg`. ``coef`` is ``(n_vox, K)`` float32 numpy."""

    coef: np.ndarray
    objective: list = field(default_factory=list)
    lipschitz: float = float("nan")
    n_iter: int = 0
    loss: str = "huber"
    delta: float = float("nan")
    l1: float = 0.0
    rel_resid: float = float("nan")

    @property
    def n_increases(self) -> int:
        f = self.objective
        return int(sum(b > a for a, b in zip(f[:-1], f[1:])))

    @property
    def nonzero_fraction(self) -> float:
        return float((self.coef > 0).mean())


def estimate_lipschitz(op, weights=None, *, n_iter: int = 15, seed: int = 0) -> float:
    """Largest eigenvalue of ``A^T W A`` by power iteration from a positive random vector (see the module docstring)."""
    torch = _torch()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = (0.5 + torch.rand((op.n_vox, op.K), generator=gen)).to(dtype=op.dtype, device=op.device)
    w = None if weights is None else torch.as_tensor(weights, dtype=op.dtype, device=op.device)
    lam = 0.0
    for _ in range(int(n_iter)):
        x = x / torch.linalg.vector_norm(x)
        y = op.forward(x)
        if w is not None:
            y = y * w
        z = op.adjoint(y)
        lam = float(torch.linalg.vector_norm(z))
        x = z
        if lam == 0.0:
            return 0.0
    return lam


def _loss(r, w, loss: str, delta: float) -> float:
    torch = _torch()
    if loss == "l2":
        v = 0.5 * r * r
    else:
        a = r.abs()
        v = torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    if w is not None:
        v = v * w
    return float(v.sum(dtype=torch.float64))


def fista_nonneg(op, data, *, n_iter: int = 100, loss: str = "huber", delta: float = 30.0, weights=None, l1: float = 0.0,
                 lipschitz: float | None = None, x0=None, n_power: int = 15, safety: float = 1.1, seed: int = 0,
                 callback=None) -> SolveResult:
    """Non-negative FISTA for ``min_{c >= 0}  sum w rho(op.forward(c) - data) + l1 sum(c)``.

    Parameters
    ----------
    op
        Anything with ``forward``, ``adjoint``, ``n_vox``, ``K``, ``dtype``, ``device`` (a :class:`~midas_dt.texture_nodes.NodeOperator`).
    data
        Array of ``op.data_shape``, ring-normalised (:func:`~midas_dt.texture_nodes.ring_normalise`).
    loss, delta
        ``"huber"`` with threshold ``delta`` in data units, or ``"l2"``.
    weights
        Optional per-datum weights of the data shape (inverse variances, or a 0/1 mask: zero entries do not constrain the fit).
    l1
        Weight of ``sum(c)``; 0 turns it off.
    lipschitz
        Supply to skip the power iteration (the value used is returned either way); otherwise ``safety`` times the estimate.
    n_iter
        The number of FISTA steps: part of the model, report it.
    callback
        Optional ``callback(k, objective, coef_tensor)`` after each step.
    """
    torch = _torch()
    if loss not in ("huber", "l2"):
        raise ValueError("loss must be 'huber' or 'l2'")
    if l1 < 0 or n_iter < 0:
        raise ValueError("l1 and n_iter must be non-negative")
    y = torch.as_tensor(data, dtype=op.dtype, device=op.device)
    if tuple(y.shape) != tuple(op.data_shape):
        raise ValueError(f"data must be {tuple(op.data_shape)}, got {tuple(y.shape)}")
    w = None if weights is None else torch.as_tensor(weights, dtype=op.dtype, device=op.device)
    L = float(lipschitz) if lipschitz is not None else safety * estimate_lipschitz(op, w, n_iter=n_power, seed=seed)
    if not L > 0:
        raise ValueError("the operator has zero norm (no node produces any signal in the data bins)")

    if x0 is None:
        x = torch.zeros((op.n_vox, op.K), dtype=op.dtype, device=op.device)
        Ax = torch.zeros_like(y)
    else:
        x = torch.as_tensor(x0, dtype=op.dtype, device=op.device).clamp_min(0).clone()
        Ax = op.forward(x)
    z, Az = x.clone(), Ax.clone()
    t = 1.0
    d = float("inf") if loss == "l2" else float(delta)
    hist = [_loss(Ax - y, w, loss, d) + l1 * float(x.sum(dtype=torch.float64))]
    for k in range(int(n_iter)):
        r = Az - y
        psi = r if loss == "l2" else r.clamp(-d, d)
        if w is not None:
            psi = psi * w
        g = op.adjoint(psi)
        x_new = (z - g * (1.0 / L)).sub_(l1 / L).clamp_min_(0.0)
        Ax_new = op.forward(x_new)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        beta = (t - 1.0) / t_new
        z = x_new + beta * (x_new - x)
        Az = Ax_new + beta * (Ax_new - Ax)
        x, Ax, t = x_new, Ax_new, t_new
        hist.append(_loss(Ax - y, w, loss, d) + l1 * float(x.sum(dtype=torch.float64)))
        if callback is not None:
            callback(k, hist[-1], x)
    res = (Ax - y)
    ww = w if w is not None else 1.0
    num = float(torch.linalg.vector_norm((res * ww ** 0.5) if w is not None else res))
    den = float(torch.linalg.vector_norm((y * ww ** 0.5) if w is not None else y))
    return SolveResult(coef=x.cpu().numpy(), objective=hist, lipschitz=L, n_iter=int(n_iter), loss=loss,
                       delta=d, l1=float(l1), rel_resid=num / den if den > 0 else float("nan"))
