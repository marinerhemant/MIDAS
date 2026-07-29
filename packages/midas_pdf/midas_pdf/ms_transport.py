"""All-orders multiple scattering by differentiable radiative transfer (slab).

The scalar geometric-series resummation under-predicts the multiple-scattering
fraction at large optical depth because it uses the order-2/order-1 ratio as the
per-order multiplier, ignoring how the angular distribution redistributes with
each scatter. The rigorous fix is to solve the transport (radiative-transfer)
equation, whose solution contains *all* scattering orders simultaneously.

For a slab with a collimated normal beam the problem is azimuthally symmetric
about the beam axis, so the radiative-transfer equation reduces exactly to two
variables — optical depth ``tau`` and direction cosine ``mu = cos(theta)`` from
the beam:

    mu dI/dtau = -I + a * ∫ P(mu', mu) I(tau, mu') dmu' + S(tau, mu),
    S(tau, mu) = a * P(mu0=1, mu) * exp(-tau)        (first scattering of the beam)

with the azimuthally-averaged phase (redistribution) matrix ``P`` built from the
differential cross-section, single-scattering albedo ``a``, and vacuum boundary
conditions (no diffuse intensity entering either face). Discretising ``mu`` on
Gauss-Legendre ordinates and ``tau`` on a uniform grid turns this into a linear
system solved with ``torch.linalg.solve`` — **differentiable** in composition /
tau / albedo, and exact in the continuum limit (only angular/depth discretisation
error). Solving with the in-scattering term off gives the single-scattering
field, so the multiple-scattering fraction is ``beta = 1 - I_single/I_total``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .composition import Composition
from .cross_section import differential_cross_section

__all__ = ["phase_matrix", "slab_transport_ms"]

_FOUR_PI = 4.0 * float(np.pi)


def phase_matrix(
    composition: Composition,
    wavelength_A: float,
    mu_nodes: torch.Tensor,
    *,
    n_azimuth: int = 240,
) -> torch.Tensor:
    """Azimuthally-averaged redistribution matrix ``P[k, l]`` (scatter l -> k).

    ``cos chi = mu_k mu_l + sqrt(1-mu_k^2) sqrt(1-mu_l^2) cos(phi)`` averaged over
    the relative azimuth ``phi``. Columns are normalised so that
    ``sum_k w_k P[k, l] = 1`` (probability of scattering somewhere is 1; the
    albedo handles survival), which guarantees photon conservation. Differentiable
    in composition through the cross-section.
    """
    mu = mu_nodes
    K = mu.shape[0]
    s = torch.sqrt((1.0 - mu * mu).clamp(min=0.0))
    phi = torch.linspace(0.0, 2 * float(np.pi), n_azimuth, dtype=torch.float64)
    cphi = torch.cos(phi)
    # cos chi for every (k, l, phi)
    cos_chi = (mu[:, None, None] * mu[None, :, None]
               + s[:, None, None] * s[None, :, None] * cphi[None, None, :])
    cos_chi = cos_chi.clamp(-1.0, 1.0)
    chi = torch.arccos(cos_chi)
    Q = (_FOUR_PI / wavelength_A) * torch.sin(0.5 * chi)
    dsig = differential_cross_section(Q, composition, wavelength_A=wavelength_A)
    P = torch.trapz(dsig, phi, dim=2) / (2 * float(np.pi))      # (K, K), avg over phi
    return P


def _source_shape(
    composition: Composition, wavelength_A: float, mu_nodes: torch.Tensor,
) -> torch.Tensor:
    """First-scattering angular shape P(mu0=1, mu_k) = dσ/dΩ at chi = theta_k.
    (Absolute scale is irrelevant: it cancels in beta = 1 - I_single/I_total.)"""
    chi = torch.arccos(mu_nodes.clamp(-1.0, 1.0))              # mu0=1 -> cos chi = mu_k
    Q = (_FOUR_PI / wavelength_A) * torch.sin(0.5 * chi)
    return differential_cross_section(Q, composition, wavelength_A=wavelength_A)


def slab_transport_ms(
    composition: Composition,
    *,
    wavelength_A: float,
    tau: float,
    albedo: float,
    q_max: float = 20.0,
    n_mu: int = 32,
    n_tau: int = 100,
) -> dict:
    """All-orders multiple-scattering fraction for a slab by discrete ordinates.

    Returns dict with forward-transmission ``Q``, ``mu``, ``I_single``,
    ``I_total`` (diffuse, all orders), and ``beta`` ( = 1 - I_single/I_total ).
    Differentiable in composition / tau / albedo via ``torch.linalg.solve``.
    """
    a = torch.as_tensor(albedo, dtype=torch.float64)
    T = float(tau)

    # Gauss-Legendre ordinates on [-1, 1] (even count; skip mu = 0)
    nodes, weights = np.polynomial.legendre.leggauss(n_mu)
    mu = torch.as_tensor(nodes, dtype=torch.float64)
    w = torch.as_tensor(weights, dtype=torch.float64)
    K = n_mu

    P = phase_matrix(composition, wavelength_A, mu)            # (K, K), col-normalised
    colsum = (w[:, None] * P).sum(dim=0)                       # sum_k w_k P[k,l]
    P = P / colsum[None, :].clamp(min=1e-30)                  # normalise columns -> 1
    src_shape = _source_shape(composition, wavelength_A, mu)   # (K,)

    M = n_tau
    tau_grid = torch.linspace(0.0, T, M + 1, dtype=torch.float64)
    h = T / M

    N = K * (M + 1)

    def idx(k, m):
        return k * (M + 1) + m

    src = src_shape[:, None] * torch.exp(-tau_grid)[None, :]   # (K, M+1), shape only

    def build_and_solve(with_scatter: bool) -> torch.Tensor:
        A = torch.zeros((N, N), dtype=torch.float64)
        b = torch.zeros(N, dtype=torch.float64)
        for k in range(K):
            mk = float(mu[k])
            if mk > 0:                                         # downward beam: BC at top
                A[idx(k, 0), idx(k, 0)] = 1.0                  # I[k, 0] = 0
                interior = range(1, M + 1)
            else:                                              # upward: BC at bottom
                A[idx(k, M), idx(k, M)] = 1.0                  # I[k, M] = 0
                interior = range(0, M)
            for m in interior:
                r = idx(k, m)
                if mk > 0:                                     # backward (upwind) difference
                    A[r, idx(k, m)] += mk / h + 1.0
                    A[r, idx(k, m - 1)] += -mk / h
                else:                                          # forward difference
                    A[r, idx(k, m + 1)] += mk / h
                    A[r, idx(k, m)] += -mk / h + 1.0
                if with_scatter:                               # -a sum_l w_l P[k,l] I[l,m]
                    for l in range(K):
                        A[r, idx(l, m)] += -(a * w[l] * P[k, l])
                b[r] = a * src[k, m]
        x = torch.linalg.solve(A, b)
        return x.reshape(K, M + 1)

    I_tot = build_and_solve(with_scatter=True)
    I_sng = build_and_solve(with_scatter=False)

    # forward transmission: exit at tau = T with mu > 0
    fwd = mu > 0
    mu_f = mu[fwd]
    It = I_tot[fwd, M]
    Is = I_sng[fwd, M]
    theta = torch.arccos(mu_f.clamp(-1, 1))
    Q = (_FOUR_PI / wavelength_A) * torch.sin(0.5 * theta)
    beta = 1.0 - Is / It.clamp(min=1e-30)
    keep = Q <= q_max
    order = torch.argsort(Q[keep])
    return {
        "Q": Q[keep][order],
        "mu": mu_f[keep][order],
        "I_single": Is[keep][order],
        "I_total": It[keep][order],
        "beta": beta[keep][order].clamp(0.0, 1.0),
    }
