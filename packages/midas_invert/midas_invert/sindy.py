"""Sparse-regression dynamics discovery (SINDy / STLSQ), domain-agnostic.

Given a 1-D state trajectory ``x(t)``, discover a second-order latent ODE
``x_ddot = sum_k c_k * term_k(x, x_dot)`` over a small candidate library by
sparse linear regression on finite-difference derivatives (sequential
thresholded least squares).  Well-posed (unlike integrate-and-match over a
single trajectory) and yields a sparse, interpretable equation.

Shared by midas_2d (coherent-phonon EOM) and midas_kinetics (in-situ HEDM laws).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["library_terms", "integrate_latent_ode", "discover_dynamics"]

_DEFAULT_TERMS = ("x", "v", "x3")


def library_terms(x, v, terms=_DEFAULT_TERMS):
    """Evaluate the candidate library at (x, v).  Returns a list of tensors."""
    import torch
    out = []
    for name in terms:
        if name == "x":
            out.append(x)
        elif name == "v":
            out.append(v)
        elif name == "x2":
            out.append(x * x)
        elif name == "x3":
            out.append(x * x * x)
        elif name == "1":
            out.append(torch.ones_like(x))
        else:
            raise ValueError(f"unknown library term {name!r}")
    return out


def integrate_latent_ode(coeffs, x0, v0, t, terms=_DEFAULT_TERMS):
    """RK4-integrate ``x_dot=v, v_dot=sum coeff*term(x,v)`` on grid ``t``.
    Differentiable in ``coeffs`` and the initial conditions."""
    import torch
    t = torch.as_tensor(t)
    coeffs = torch.as_tensor(coeffs, dtype=t.dtype, device=t.device)
    dt = (t[1] - t[0])

    def accel(x, v):
        return sum(c * term for c, term in zip(coeffs, library_terms(x, v, terms)))

    def deriv(x, v):
        return v, accel(x, v)

    x = torch.as_tensor(x0, dtype=t.dtype, device=t.device)
    v = torch.as_tensor(v0, dtype=t.dtype, device=t.device)
    xs = [x]
    for _ in range(t.numel() - 1):
        k1x, k1v = deriv(x, v)
        k2x, k2v = deriv(x + dt / 2 * k1x, v + dt / 2 * k1v)
        k3x, k3v = deriv(x + dt / 2 * k2x, v + dt / 2 * k2v)
        k4x, k4v = deriv(x + dt * k3x, v + dt * k3v)
        x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        xs.append(x)
    return torch.stack(xs)


def discover_dynamics(x_obs, t, *, terms=_DEFAULT_TERMS, threshold=0.05):
    """Discover the latent-ODE coefficients reproducing ``x_obs(t)`` by STLSQ.

    Returns dict mapping term name -> coefficient, plus ``omega`` and ``gamma``
    when ``x``/``v`` are in the library (``omega=sqrt(-coeff_x)``,
    ``gamma=-coeff_v``).
    """
    import torch
    t = torch.as_tensor(t)
    x = torch.as_tensor(x_obs, dtype=t.dtype, device=t.device)
    v = torch.gradient(x, spacing=(t,))[0]
    a = torch.gradient(v, spacing=(t,))[0]
    sl = slice(2, -2)
    Theta = torch.stack(library_terms(x[sl], v[sl], terms), dim=1)
    target = a[sl]

    active = torch.ones(len(terms), dtype=torch.bool)
    coeffs = torch.zeros(len(terms), dtype=t.dtype)
    for _ in range(8):
        cols = torch.where(active)[0]
        if cols.numel() == 0:
            break
        sol = torch.linalg.lstsq(Theta[:, cols], target.unsqueeze(1)).solution.squeeze(1)
        coeffs = torch.zeros(len(terms), dtype=t.dtype)
        coeffs[cols] = sol
        new_active = active & (coeffs.abs() >= threshold)
        if bool((new_active == active).all()):
            break
        active = new_active

    out = {name: float(coeffs[i]) for i, name in enumerate(terms)}
    if "x" in terms:
        out["omega"] = float((-out["x"]) ** 0.5) if out["x"] < 0 else float("nan")
    if "v" in terms:
        out["gamma"] = -out["v"]
    return out
