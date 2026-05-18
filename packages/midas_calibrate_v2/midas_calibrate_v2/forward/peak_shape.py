"""Area-normalised pseudo-Voigt (Thompson-Cox-Hastings) peak model — torch.

Used by both the alternating engine (per-(ring, η-bin) peak fit replacing v1's
centroid extraction) and the joint forward-cake engine (M5.5).
"""
from __future__ import annotations

import math

import torch


_LN2 = math.log(2.0)
_INV_PI = 1.0 / math.pi


def gaussian(R: torch.Tensor, center: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Area-normalised Gaussian."""
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    z = (R - center) / sigma
    return coef * torch.exp(-0.5 * z * z)


def lorentzian(R: torch.Tensor, center: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Area-normalised Lorentzian (FWHM = 2γ)."""
    return (gamma * _INV_PI) / ((R - center) ** 2 + gamma ** 2)


def pseudo_voigt(
    R: torch.Tensor,
    center: torch.Tensor,
    sigma: torch.Tensor,
    gamma: torch.Tensor,
    eta_v: torch.Tensor,
    area: torch.Tensor,
) -> torch.Tensor:
    """Area-normalised pseudo-Voigt: A · [η L(R) + (1-η) G(R)].

    sigma, gamma : Gaussian / Lorentzian widths (σ, γ).
    eta_v ∈ [0, 1] : Lorentzian mixing fraction.
    """
    G = gaussian(R, center, sigma)
    L = lorentzian(R, center, gamma)
    return area * (eta_v * L + (1.0 - eta_v) * G)


def tch_pseudo_voigt(
    R: torch.Tensor,
    center: torch.Tensor,
    sigmaG: torch.Tensor,
    gammaL: torch.Tensor,
    area: torch.Tensor,
) -> torch.Tensor:
    """Thompson-Cox-Hastings pseudo-Voigt with internally derived eta_v.

    Computes the effective FWHM via TCH polynomial mixing of Gaussian and
    Lorentzian components, then evaluates the area-normalised pseudo-Voigt
    at that width.
    """
    fG = sigmaG * 2.0 * math.sqrt(2.0 * _LN2)
    fL = gammaL * 2.0
    f5 = (fG ** 5
          + 2.69269 * fG ** 4 * fL
          + 2.42843 * fG ** 3 * fL ** 2
          + 4.47163 * fG ** 2 * fL ** 3
          + 0.07842 * fG * fL ** 4
          + fL ** 5)
    f = f5 ** 0.2
    r = fL / f.clamp(min=1e-30)
    eta_v = (1.36603 * r - 0.47719 * r * r + 0.11116 * r * r * r).clamp(0.0, 1.0)
    sigma_eff = f / (2.0 * math.sqrt(2.0 * _LN2))
    gamma_eff = 0.5 * f
    return pseudo_voigt(R, center, sigma_eff, gamma_eff, eta_v, area)


__all__ = ["gaussian", "lorentzian", "pseudo_voigt", "tch_pseudo_voigt"]
