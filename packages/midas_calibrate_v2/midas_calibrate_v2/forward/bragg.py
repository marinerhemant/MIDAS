"""Bragg's law in torch — differentiable in wavelength and d-spacing."""
from __future__ import annotations

import torch


_DEG2RAD = 0.017453292519943295
_RAD2DEG = 57.29577951308232


def two_theta_from_d(d_spacing_A: torch.Tensor, wavelength_A: torch.Tensor) -> torch.Tensor:
    """2θ [deg] = 2 arcsin(λ / 2d)."""
    s = (wavelength_A / (2.0 * d_spacing_A)).clamp(min=-0.999999, max=0.999999)
    return 2.0 * torch.asin(s) * _RAD2DEG


def d_from_two_theta(two_theta_deg: torch.Tensor, wavelength_A: torch.Tensor) -> torch.Tensor:
    """d [Å] = λ / (2 sin θ)."""
    theta_rad = 0.5 * two_theta_deg * _DEG2RAD
    return wavelength_A / (2.0 * torch.sin(theta_rad))


def R_ideal_px(two_theta_deg: torch.Tensor, Lsd: torch.Tensor,
               px: torch.Tensor) -> torch.Tensor:
    """Ideal (geometric, undistorted) ring radius in pixels.

    R_ideal = Lsd · tan(2θ) / px.
    """
    return Lsd * torch.tan(two_theta_deg * _DEG2RAD) / px


__all__ = ["two_theta_from_d", "d_from_two_theta", "R_ideal_px"]
