"""Small-angle X-ray scattering form factors.

All form factors return |F(Q)|² (intensity, not amplitude), normalised so
that at Q → 0 the value equals V² (particle volume squared, Å⁶). This
matches the convention used in the SasView / IsGISAXS literature.

Every function is torch-differentiable in both Q and the shape parameters
so that gradient-based fitting through the SAXS+PDF joint pipeline works
end-to-end.

References:
  * Guinier & Fournet, *Small-Angle Scattering of X-Rays* (1955).
  * Pedersen, *Adv. Colloid Interface Sci.* 70, 171 (1997) — polydispersity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

_FOUR_PI = 4.0 * float(np.pi)


def _as_t(x, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

def sphere_form_factor_squared(
    q: torch.Tensor | np.ndarray,
    radius_A: float | torch.Tensor,
) -> torch.Tensor:
    """|F(Q)|² for a uniform-density sphere of radius R (Å).

    Closed form:
        F(Q) = 3 V · [ sin(QR) − QR cos(QR) ] / (QR)³
        where V = 4π R³ / 3.

    At Q → 0, |F|² → V². The stable Q→0 limit uses the Taylor expansion.
    """
    q_t = _as_t(q)
    R = _as_t(radius_A)
    V = _FOUR_PI * R ** 3 / 3.0
    x = q_t * R
    # Stable expansion for small x
    small = x.abs() < 1e-3
    # F = V · [3 (sin x − x cos x) / x³]
    # small-x: [3 (x − x³/6 − x cos x)/x³ ] → 1 as x→0
    val_small = 1.0 - x ** 2 / 10.0 + x ** 4 / 280.0        # Taylor to O(x⁴)
    x_safe = x.clamp(min=1e-6)
    val_general = 3.0 * (torch.sin(x_safe) - x_safe * torch.cos(x_safe)) / x_safe ** 3
    val = torch.where(small, val_small, val_general)
    F = V * val
    return F ** 2


# ---------------------------------------------------------------------------
# Ellipsoid (uniform, prolate/oblate)
# ---------------------------------------------------------------------------

def ellipsoid_form_factor_squared(
    q: torch.Tensor | np.ndarray,
    a_A: float | torch.Tensor,
    b_A: float | torch.Tensor,
    *,
    n_mu: int = 40,
) -> torch.Tensor:
    """|F(Q)|² for an axially-symmetric ellipsoid, radii a (rotation axis)
    and b (equatorial).

    Orientation-averaged form:

        <|F|²>(Q) = ∫₀¹ |F_sphere(Q · r(μ))|² dμ,     r(μ) = √(a² μ² + b² (1-μ²))

    Uses a fixed-node Gauss-Legendre quadrature of ``n_mu`` nodes over μ ∈ [0, 1].
    Reduces to :func:`sphere_form_factor_squared` when a = b.
    """
    q_t = _as_t(q)
    a = _as_t(a_A)
    b = _as_t(b_A)
    # Ellipsoid volume is FIXED (= 4π a b² / 3); orientation only changes
    # the argument of the shape function, not the volume prefactor.
    V = _FOUR_PI * a * b ** 2 / 3.0
    # Nodes / weights on [0, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_mu)
    mu_t = _as_t(0.5 * (nodes + 1.0))                                   # (n_mu,)
    w_t  = _as_t(0.5 * weights)                                          # (n_mu,)
    r_mu = torch.sqrt(a ** 2 * mu_t ** 2 + b ** 2 * (1.0 - mu_t ** 2))   # (n_mu,)
    # Argument x = Q · r_mu, shape (nQ, n_mu)
    x = q_t[:, None] * r_mu[None, :]
    small = x.abs() < 1e-3
    x_safe = x.clamp(min=1e-9)
    shape_small = 1.0 - x ** 2 / 10.0 + x ** 4 / 280.0                   # Taylor to O(x⁴)
    shape_general = 3.0 * (torch.sin(x_safe) - x_safe * torch.cos(x_safe)) / x_safe ** 3
    shape = torch.where(small, shape_small, shape_general)              # (nQ, n_mu)
    F2 = (V * shape) ** 2                                                # (nQ, n_mu)
    return (F2 * w_t[None, :]).sum(dim=-1)


# ---------------------------------------------------------------------------
# Cylinder (finite, axially-symmetric)
# ---------------------------------------------------------------------------

def cylinder_form_factor_squared(
    q: torch.Tensor | np.ndarray,
    radius_A: float | torch.Tensor,
    length_A: float | torch.Tensor,
    *,
    n_theta: int = 40,
) -> torch.Tensor:
    """|F(Q)|² for a solid finite cylinder, radius R and length L.

    Orientation-averaged:

        <|F|²>(Q) = ∫₀¹ [ V · sinc(Q L cos θ / 2)
                          · 2 J₁(Q R sin θ) / (Q R sin θ) ]² sin θ dθ

    where V = π R² L. Gauss-Legendre quadrature over cos θ ∈ [0, 1].
    """
    q_t = _as_t(q)
    R = _as_t(radius_A)
    L = _as_t(length_A)
    V = float(np.pi) * R ** 2 * L
    # Gauss-Legendre on [0, 1] for cos θ
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    ct = _as_t(0.5 * (nodes + 1.0))                                    # (n_theta,)
    w = _as_t(0.5 * weights)                                            # (n_theta,)
    st = torch.sqrt(1.0 - ct ** 2)

    # Broadcast (Q, θ)
    q_col = q_t[:, None]                                                # (nQ, 1)
    ct_row = ct[None, :]                                                # (1, n_theta)
    st_row = st[None, :]                                                # (1, n_theta)

    x_L = q_col * L * ct_row / 2.0                                      # (nQ, n_theta)
    # sinc(x) = sin(x)/x, stable at 0
    sinc = torch.where(x_L.abs() < 1e-6,
                        1.0 - x_L ** 2 / 6.0,
                        torch.sin(x_L) / x_L.clamp(min=1e-9))

    x_R = q_col * R * st_row                                            # (nQ, n_theta)
    # 2 J₁(x)/x — the "Airy-like" transverse factor. Use torch's Bessel J1.
    x_R_safe = x_R.clamp(min=1e-9)
    j1 = torch.special.bessel_j1(x_R_safe)
    airy = torch.where(x_R.abs() < 1e-6,
                        1.0 - x_R ** 2 / 8.0,
                        2.0 * j1 / x_R_safe)

    F = V * sinc * airy                                                 # (nQ, n_theta)
    return (F ** 2 * w[None, :]).sum(dim=-1)


# ---------------------------------------------------------------------------
# Percus-Yevick structure factor (hard spheres)
# ---------------------------------------------------------------------------

def percus_yevick_S(
    q: torch.Tensor | np.ndarray,
    radius_A: float | torch.Tensor,
    volume_fraction: float | torch.Tensor,
) -> torch.Tensor:
    """Percus-Yevick structure factor for hard spheres, closed form.

    Follows Kinning & Thomas (1984) / Pedersen (1997) §4. Diverges cleanly
    at φ → π/6 (close packing). Multiply the single-particle |F|² by this
    to get the "interacting-hard-spheres" I(Q).
    """
    q_t = _as_t(q)
    R = _as_t(radius_A)
    phi = _as_t(volume_fraction)
    if float(phi) < 0 or float(phi) > 0.5:
        raise ValueError(f"volume_fraction must be in (0, 0.5), got {phi}")

    # Standard Kotlarchyk-Chen 1983 (JCP 79:2461) formulation.
    # x = Q σ, σ = 2R.
    # S(Q) = 1 / (1 + 24 φ G(x)),
    # G(x) = α (sin x − x cos x)/x²
    #       + β [(2 − x²) cos x + 2 x sin x − 2]/x³
    #       + γ [(4 x³ − 24 x) sin x − (x⁴ − 12 x² + 24) cos x + 24]/x⁵
    x = 2.0 * q_t * R
    alpha = ((1.0 + 2.0 * phi) ** 2) / ((1.0 - phi) ** 4)
    beta  = -6.0 * phi * (1.0 + 0.5 * phi) ** 2 / ((1.0 - phi) ** 4)
    gamma = 0.5 * phi * ((1.0 + 2.0 * phi) ** 2) / ((1.0 - phi) ** 4)

    x_safe = x.clamp(min=1e-6)
    sx = torch.sin(x_safe); cx = torch.cos(x_safe)
    G = (alpha * (sx - x_safe * cx) / x_safe ** 2
         + beta * ((2.0 - x_safe ** 2) * cx + 2.0 * x_safe * sx - 2.0) / x_safe ** 3
         + gamma * ((4.0 * x_safe ** 3 - 24.0 * x_safe) * sx
                     - (x_safe ** 4 - 12.0 * x_safe ** 2 + 24.0) * cx + 24.0)
                    / x_safe ** 5)

    # Small-x Taylor limit (Wertheim): S(0) = (1-φ)⁴ / (1 + 2φ)²
    # Use the closed form aggressively: cancellations in G(x) at x < ~0.01
    # eat float64 precision. The Wertheim value is exact at Q → 0.
    S_at_zero = ((1.0 - phi) ** 4) / ((1.0 + 2.0 * phi) ** 2)
    small = x.abs() < 5e-2
    # S(Q) = 1 / (1 + 24 φ G(x)/x)
    S_full = 1.0 / (1.0 + 24.0 * phi * G / x_safe)
    return torch.where(small, S_at_zero.expand_as(S_full), S_full)


__all__ = [
    "sphere_form_factor_squared",
    "ellipsoid_form_factor_squared",
    "cylinder_form_factor_squared",
    "percus_yevick_S",
]
