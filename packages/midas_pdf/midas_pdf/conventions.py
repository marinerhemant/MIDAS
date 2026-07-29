"""Real-space correlation-function family + reciprocal-space F(Q).

Total-scattering communities report several closely related functions and the
mapping between them trips people up. We provide all of them from the reduced
PDF ``G(r)`` (the quantity the sine FT returns) and the atomic number density
ρ₀, following Keen, *J. Appl. Cryst.* **34**, 172 (2001):

    F(Q)  = Q [S(Q) - 1]                      reduced structure function
    G(r)  = (2/π) ∫ F(Q) sin(Qr) dQ           reduced PDF  (Keen's D(r); Egami-Billinge G(r))
    g(r)  = 1 + G(r) / (4π r ρ₀)              pair distribution function
    T(r)  = G(r) + 4π r ρ₀ = 4π r ρ₀ g(r)     total correlation function
    R(r)  = r G(r) + 4π r² ρ₀ = 4π r² ρ₀ g(r) radial distribution function (RDF)

``R(r) dr`` is the mean number of atoms between r and r+dr — the coordination
number integrates directly from it. σ propagates linearly through every one of
these (they are linear in G at fixed r), so an input ``sigma_G`` is rescaled by
the same factor as G.

The unit/convention choice (which function to report, FZ vs Keen S(Q)) is the
first thing to settle with collaborators; having the whole family on tap makes
that a one-line switch rather than a re-derivation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

__all__ = [
    "structure_function_F",
    "pair_distribution_g",
    "total_correlation_T",
    "radial_distribution_R",
]

_FOUR_PI = 4.0 * float(np.pi)


def structure_function_F(
    q: torch.Tensor | np.ndarray,
    S: torch.Tensor | np.ndarray,
    *,
    sigma_S: Optional[torch.Tensor | np.ndarray] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reduced structure function ``F(Q) = Q [S(Q) - 1]`` (and σ_F = Q σ_S)."""
    q_t = torch.as_tensor(q, dtype=torch.float64)
    S_t = torch.as_tensor(S, dtype=torch.float64)
    F = q_t * (S_t - 1.0)
    if sigma_S is None:
        return F, None
    sig = torch.as_tensor(sigma_S, dtype=torch.float64)
    return F, (q_t.abs() * sig)


def _rescale_sigma(
    sigma_G: Optional[torch.Tensor], factor: torch.Tensor
) -> Optional[torch.Tensor]:
    if sigma_G is None:
        return None
    return torch.as_tensor(sigma_G, dtype=torch.float64).abs() * factor.abs()


def pair_distribution_g(
    r: torch.Tensor | np.ndarray,
    G: torch.Tensor | np.ndarray,
    *,
    number_density: float,
    sigma_G: Optional[torch.Tensor | np.ndarray] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pair distribution function ``g(r) = 1 + G(r) / (4π r ρ₀)``.

    ``g(r) → 1`` at large r; ``g(r) = 0`` below the closest interatomic
    distance. The 1/r factor makes σ blow up as r→0, which is physically
    honest (g is poorly determined near the origin). The r=0 point is set to
    g=0 (no self-pair) to avoid the singularity.
    """
    r_t = torch.as_tensor(r, dtype=torch.float64)
    G_t = torch.as_tensor(G, dtype=torch.float64)
    denom = _FOUR_PI * r_t * number_density
    safe = denom.abs() > 1e-30
    factor = torch.where(safe, 1.0 / torch.where(safe, denom, torch.ones_like(denom)),
                         torch.zeros_like(denom))
    g = torch.where(safe, 1.0 + G_t * factor, torch.zeros_like(G_t))
    return g, _rescale_sigma(
        None if sigma_G is None else torch.as_tensor(sigma_G, dtype=torch.float64),
        factor,
    )


def total_correlation_T(
    r: torch.Tensor | np.ndarray,
    G: torch.Tensor | np.ndarray,
    *,
    number_density: float,
    sigma_G: Optional[torch.Tensor | np.ndarray] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Total correlation function ``T(r) = G(r) + 4π r ρ₀``.

    A pure additive shift of G(r), so σ_T = σ_G (the offset is exact).
    """
    r_t = torch.as_tensor(r, dtype=torch.float64)
    G_t = torch.as_tensor(G, dtype=torch.float64)
    T = G_t + _FOUR_PI * r_t * number_density
    sig = None if sigma_G is None else torch.as_tensor(sigma_G, dtype=torch.float64).abs()
    return T, sig


def radial_distribution_R(
    r: torch.Tensor | np.ndarray,
    G: torch.Tensor | np.ndarray,
    *,
    number_density: float,
    sigma_G: Optional[torch.Tensor | np.ndarray] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Radial distribution function ``R(r) = r G(r) + 4π r² ρ₀``.

    ``∫ R(r) dr`` over a peak = coordination number. σ_R = r · σ_G.
    """
    r_t = torch.as_tensor(r, dtype=torch.float64)
    G_t = torch.as_tensor(G, dtype=torch.float64)
    R = r_t * G_t + _FOUR_PI * r_t * r_t * number_density
    return R, _rescale_sigma(
        None if sigma_G is None else torch.as_tensor(sigma_G, dtype=torch.float64),
        r_t,
    )
