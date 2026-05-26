"""Transformation kinetics from an in-situ time/load series (HEDM).

Input is a *transformed-fraction* trajectory ``X(t)`` in [0, 1] extracted
upstream (e.g. product-phase volume/grain-count fraction from HEDM).

* :func:`fit_jmak` -- JMAK/Avrami isothermal model
  ``X(t) = 1 - exp(-(k t)^n)``, recovers ``k`` and ``n``.
* :func:`discover_rate_law` -- sparse regression (STLSQ) of ``dX/dt`` against a
  MONOMIAL library ``{1, X, X^2, X^3}`` to discover the rate-law form without
  assuming it: first-order ``dX/dt = k(1-X) = k - kX`` -> ``{"1": +k, "X": -k}``,
  autocatalytic ``dX/dt = kX(1-X) = kX - kX^2`` -> ``{"X": +k, "X2": -k}``.

(Monomial library is deliberately non-redundant; ``{1-X, X(1-X)}`` is linear in
the monomials, so a library containing both is collinear and splits the
coefficients ambiguously.)

Folded in from the standalone ``midas_kinetics`` package (consolidation pass).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["jmak_fraction", "fit_jmak", "rate_library", "discover_rate_law"]


def jmak_fraction(t, k, n):
    """JMAK transformed fraction ``X(t) = 1 - exp(-(k t)^n)``."""
    import torch
    t = torch.as_tensor(t)
    k = torch.as_tensor(k, dtype=t.dtype, device=t.device)
    n = torch.as_tensor(n, dtype=t.dtype, device=t.device)
    kt = torch.clamp(k * t, min=0.0)
    return 1.0 - torch.exp(-(kt ** n))


def fit_jmak(t, X_obs, *, init_k=0.1, init_n=1.0, steps=2000, lr=0.02):
    """Recover ``(k, n)`` from a transformed-fraction trajectory."""
    import torch
    from .optimize import fit, relative_l2_loss

    t = torch.as_tensor(t, dtype=torch.float64)
    X_obs = torch.as_tensor(X_obs, dtype=torch.float64)
    sp = torch.nn.functional.softplus
    raw_k = torch.tensor(math.log(math.expm1(init_k)), dtype=torch.float64, requires_grad=True)
    raw_n = torch.tensor(math.log(math.expm1(init_n)), dtype=torch.float64, requires_grad=True)

    def loss_fn():
        return relative_l2_loss(jmak_fraction(t, sp(raw_k), sp(raw_n)), X_obs)

    out = fit([raw_k, raw_n], loss_fn, steps=steps, lr=lr)
    return {"k": float(sp(raw_k).detach()), "n": float(sp(raw_n).detach()),
            "loss": out["loss"]}


_RATE_TERMS = ("1", "X", "X2", "X3")


def rate_library(X, terms=_RATE_TERMS):
    """Evaluate the monomial kinetics library at ``X``.  Returns (N, K)."""
    import torch
    X = torch.as_tensor(X)
    powers = {"1": 0, "X": 1, "X2": 2, "X3": 3}
    cols = []
    for name in terms:
        if name not in powers:
            raise ValueError(f"unknown rate term {name!r}")
        cols.append(X ** powers[name])
    return torch.stack(cols, dim=1)


def discover_rate_law(X_obs, t, *, terms=_RATE_TERMS, threshold=0.02):
    """Discover the rate law ``dX/dt = sum_j c_j X^j`` by STLSQ over a monomial
    library.  See module notes for the signatures of first-order vs autocatalytic."""
    import torch
    t = torch.as_tensor(t, dtype=torch.float64)
    X = torch.as_tensor(X_obs, dtype=torch.float64)
    dXdt = torch.gradient(X, spacing=(t,))[0]
    sl = slice(1, -1)
    Theta = rate_library(X[sl], terms)
    target = dXdt[sl]

    active = torch.ones(len(terms), dtype=torch.bool)
    coeffs = torch.zeros(len(terms), dtype=torch.float64)
    for _ in range(8):
        cols = torch.where(active)[0]
        if cols.numel() == 0:
            break
        sol = torch.linalg.lstsq(Theta[:, cols], target.unsqueeze(1)).solution.squeeze(1)
        coeffs = torch.zeros(len(terms), dtype=torch.float64)
        coeffs[cols] = sol
        new_active = active & (coeffs.abs() >= threshold)
        if bool((new_active == active).all()):
            break
        active = new_active
    return {name: float(coeffs[i]) for i, name in enumerate(terms)}
