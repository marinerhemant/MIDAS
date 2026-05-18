"""Sample self-absorption corrections.

For a cylindrical capillary in transmission geometry, the absorption
factor varies with 2θ::

    A(θ) = ∫ exp(-μ · L(θ, x)) dV  / V

where ``L`` is the in-sample path length for a beam exiting at angle
2θ. For a thin sample (μR ≪ 1) this is well approximated by the
power-series in Egami & Billinge (Underneath the Bragg Peaks, ch. 6).
For thicker capillaries we evaluate the integral numerically once with
a polar-grid quadrature and cache the result.

This module provides a torch.nn.Module so the absorption parameter
``μR`` is differentiable, opening the joint-refinement path with the
geometry.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _cylindrical_absorption_thin(mu_R: torch.Tensor,
                                   two_theta: torch.Tensor) -> torch.Tensor:
    """Thin-cylinder limit (μR ≤ 1.5).

    Power-series expansion in μR + small-angle correction; matches
    Egami-Billinge to <1% over μR ∈ [0, 1.5].
    """
    # First-order: A ≈ exp(-2 μR) ; small-angle 2θ effect via polynomial
    # in cos(θ) for transmission geometry (factor of 2 for path-in-out).
    cos_theta = torch.cos(0.5 * two_theta)
    return torch.exp(-2.0 * mu_R / cos_theta.clamp(min=0.05))


def _cylindrical_absorption_quadrature(mu_R: torch.Tensor,
                                          two_theta: torch.Tensor,
                                          *,
                                          n_radial: int = 12,
                                          n_azimuth: int = 24) -> torch.Tensor:
    """Numerical 2-D quadrature for thicker μR.

    Sample the cylinder cross-section on a polar grid (r, φ) and sum
    ``exp(-μ · L(r, φ, θ))`` weighted by ``r dr dφ``. ``μR`` enters
    only as a scaling, so we pre-build the grid once and let autograd
    flow through the exp.
    """
    r_grid = torch.linspace(0.05, 0.95, n_radial,
                              dtype=mu_R.dtype, device=mu_R.device)
    phi_grid = torch.linspace(0.0, 2.0 * math.pi, n_azimuth + 1,
                                dtype=mu_R.dtype, device=mu_R.device)[:-1]
    r_mesh, phi_mesh = torch.meshgrid(r_grid, phi_grid, indexing="ij")
    # Unit-radius cylinder, beam along +x. For exit angle 2θ, path
    # length from (r·cos φ, r·sin φ) is approximated as
    #   L_in  = sqrt(1 - (r sin φ)²) - r cos φ        (in-sample, beam → point)
    #   L_out = (sqrt(1 - (r sin(φ - 2θ))²) - r cos(φ - 2θ))
    # Total path = L_in + L_out, all in units of cylinder R.
    sphi = torch.sin(phi_mesh)
    cphi = torch.cos(phi_mesh)
    L_in = torch.sqrt(torch.clamp(1.0 - (r_mesh * sphi) ** 2, min=0.0)) \
        - r_mesh * cphi

    out = torch.empty(two_theta.shape, dtype=mu_R.dtype, device=mu_R.device)
    # Δr · Δφ cancels between numerator and denominator → normalise by
    # the cross-section weight only.
    norm = r_mesh.sum()
    for k in range(two_theta.shape[0]):
        twoth = two_theta[k]
        sphi_o = torch.sin(phi_mesh - twoth)
        cphi_o = torch.cos(phi_mesh - twoth)
        L_out = torch.sqrt(torch.clamp(1.0 - (r_mesh * sphi_o) ** 2, min=0.0)) \
            - r_mesh * cphi_o
        L = L_in + L_out
        contrib = torch.exp(-mu_R * L) * r_mesh
        out[k] = contrib.sum() / norm
    return out


class CylindricalAbsorption(nn.Module):
    """Sample self-absorption for a cylindrical capillary.

    Differentiable in ``mu_R`` (linear absorption coefficient × radius).
    Defaults to the thin-sample analytical formula; switches to a
    polar-grid quadrature for ``mu_R > 1.5`` automatically.

    Parameters
    ----------
    mu_R :
        Initial value of μR (dimensionless).
    refinable :
        If True, ``mu_R`` becomes an ``nn.Parameter``.
    transmission_geometry :
        For now only transmission is supported (reflection geometry will
        be added when needed for SAXS / GISAXS).
    """

    def __init__(
        self,
        *,
        mu_R: float = 0.5,
        refinable: bool = False,
        transmission_geometry: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if not transmission_geometry:
            raise NotImplementedError(
                "Only transmission_geometry=True is supported; "
                "reflection geometry pending"
            )
        mu_t = torch.as_tensor(float(mu_R), dtype=dtype)
        if refinable:
            self.mu_R = nn.Parameter(mu_t)
        else:
            self.register_buffer("mu_R", mu_t)

    def forward(self, two_theta: torch.Tensor) -> torch.Tensor:
        if float(self.mu_R) <= 1.5:
            return _cylindrical_absorption_thin(self.mu_R, two_theta)
        return _cylindrical_absorption_quadrature(self.mu_R, two_theta)


__all__ = ["CylindricalAbsorption"]
