"""Per-pixel intensity correction modules: polarization + solid-angle.

Mirrors the multiplicative factors v1 applies inside ``MapperCore.c``
(``corrected /= sa`` and ``corrected /= polFactor``), but as torch
``nn.Module``s so the corrections themselves are differentiable through
the geometry and through their own configurable parameters
(polarization fraction / plane angle).

Conventions match v1:

- Polarization factor:  ``1 - PF · sin²(2θ) · cos²(η - plane)``;
  ``corrected = raw / polFactor``.
- Solid-angle factor:   **exact tilt-aware form**
  ``Ω_pix / Ω_ref = Lsd² · (n̂·r) / |r|³``, where ``r`` is the lab-frame
  vector from sample to pixel and ``n̂`` is the lab-frame detector
  normal (``TRs · (1, 0, 0)``). For a perpendicular detector this
  reduces to ``cos³(2θ)``; for a tilted detector it captures the
  per-pixel incidence angle exactly. Mirrors v1's
  :func:`midas_integrate.geometry.solid_angle_factor`. **No
  approximation** — was previously the flat-detector ``cos³(2θ)`` form
  in v0.7-v0.8.1 (silently wrong on tilted detectors); fixed in v0.8.2.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


_DEG2RAD = math.pi / 180.0


def two_theta_from_R(R_px: torch.Tensor, *, Lsd: torch.Tensor,
                      px: torch.Tensor) -> torch.Tensor:
    """``2θ = atan(R · px / Lsd)``.  Differentiable."""
    return torch.atan(R_px * px / Lsd)


def polarization_factor(
    R_px: torch.Tensor,
    eta_deg: torch.Tensor,
    *,
    Lsd: torch.Tensor,
    px: torch.Tensor,
    pol_fraction: torch.Tensor,
    pol_plane_eta_deg: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel polarization correction factor.

    Matches v1: ``1 - PF · sin²(2θ) · cos²(η - plane)``.
    The integration kernel divides recorded counts by this factor.
    """
    two_theta = two_theta_from_R(R_px, Lsd=Lsd, px=px)
    s2t = torch.sin(two_theta)
    eta_offset_rad = (eta_deg - pol_plane_eta_deg) * _DEG2RAD
    ce = torch.cos(eta_offset_rad)
    return 1.0 - pol_fraction * s2t * s2t * ce * ce


def solid_angle_factor_flat(
    R_px: torch.Tensor,
    *,
    Lsd: torch.Tensor,
    px: torch.Tensor,
) -> torch.Tensor:
    """Flat-detector solid-angle factor ``cos³(2θ)``.

    **Approximate** — only correct for a perpendicular (zero-tilt)
    detector. Use :func:`solid_angle_factor_tilted` for any detector
    with a real tilt. Kept here for explicit opt-in / regression
    comparison only; the production :class:`SolidAngleCorrection`
    module uses the exact tilt-aware form.
    """
    two_theta = two_theta_from_R(R_px, Lsd=Lsd, px=px)
    c = torch.cos(two_theta)
    return c * c * c


def solid_angle_factor_tilted(
    Y_px: torch.Tensor,
    Z_px: torch.Tensor,
    *,
    Ycen: torch.Tensor,
    Zcen: torch.Tensor,
    TRs: torch.Tensor,                     # (3, 3) tilt matrix
    Lsd: torch.Tensor,
    pxY: torch.Tensor,
    pxZ: torch.Tensor,
) -> torch.Tensor:
    """**Exact** tilt-aware solid-angle factor for any detector pose.

    Returns ``Ω_pix / Ω_ref`` where:

        Ω_pix = A_pix · |n̂·r̂| / r²    (true solid angle of the pixel)
        Ω_ref = A_pix / Lsd²            (on-axis reference)

    so

        Ω_pix / Ω_ref = Lsd² · (n̂ · r) / |r|³

    with ``r`` = lab-frame vector sample → pixel and
    ``n̂`` = lab-frame detector normal = ``TRs · (1, 0, 0)``.

    Reduces to ``cos³(2θ)`` for a perpendicular detector
    (``TRs = I``); captures the local incidence angle exactly for
    any tilt.

    Bit-identical to :func:`midas_integrate.geometry.solid_angle_factor`
    (v1's reference implementation) at fp64.
    """
    Yc = (-Y_px + Ycen) * pxY
    Zc = ( Z_px - Zcen) * pxZ
    abcpr_x = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    abcpr_y = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    abcpr_z = TRs[2, 1] * Yc + TRs[2, 2] * Zc
    XYZ_x = Lsd + abcpr_x
    XYZ_y = abcpr_y
    XYZ_z = abcpr_z
    nx = TRs[0, 0]; ny = TRs[1, 0]; nz = TRs[2, 0]
    n_dot_r = nx * XYZ_x + ny * XYZ_y + nz * XYZ_z
    r_mag2  = XYZ_x * XYZ_x + XYZ_y * XYZ_y + XYZ_z * XYZ_z
    r3 = r_mag2 * torch.sqrt(r_mag2)
    return Lsd * Lsd * n_dot_r / r3


class PolarizationCorrection(nn.Module):
    """``nn.Module`` wrapper around :func:`polarization_factor`.

    Parameters
    ----------
    pol_fraction :
        Polarization fraction (0 = unpolarised, 1 = fully horizontally
        polarised). Default 0.99 matches v1's ``PolarizationFraction``.
    pol_plane_eta_deg :
        Azimuth of the polarization plane, deg. 0 = horizontal at η = 0
        (pyFAI convention).
    refinable :
        Whether the two parameters are :class:`nn.Parameter`. Defaults
        False — typical use freezes them at the calibrant-known values.
    """
    def __init__(self, *, pol_fraction: float = 0.99,
                 pol_plane_eta_deg: float = 0.0,
                 refinable: bool = False,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        pf = torch.tensor(pol_fraction, dtype=dtype)
        pe = torch.tensor(pol_plane_eta_deg, dtype=dtype)
        if refinable:
            self.pol_fraction = nn.Parameter(pf)
            self.pol_plane_eta_deg = nn.Parameter(pe)
        else:
            self.register_buffer("pol_fraction", pf)
            self.register_buffer("pol_plane_eta_deg", pe)

    def forward(
        self,
        R_px: torch.Tensor,
        eta_deg: torch.Tensor,
        *,
        Lsd: torch.Tensor,
        px: torch.Tensor,
    ) -> torch.Tensor:
        return polarization_factor(
            R_px, eta_deg,
            Lsd=Lsd, px=px,
            pol_fraction=self.pol_fraction,
            pol_plane_eta_deg=self.pol_plane_eta_deg,
        )


class SolidAngleCorrection(nn.Module):
    """Exact tilt-aware solid-angle factor.

    Bit-identical to v1's :func:`midas_integrate.geometry.solid_angle_factor`
    (which the MIDAS calibration paper uses). For a perpendicular
    detector this reduces to ``cos³(2θ)``; for any tilt the per-pixel
    incidence angle is exact.

    Forward signature uses positional ``(Y_px, Z_px)`` and named
    geometry kwargs (different from v0.8.1's
    ``forward(R_px, Lsd=, px=)``) — the exact form needs the full
    pixel coordinates and the tilt matrix, not just the radial R.
    """
    def forward(
        self,
        Y_px: torch.Tensor,
        Z_px: torch.Tensor,
        *,
        Ycen: torch.Tensor,
        Zcen: torch.Tensor,
        TRs: torch.Tensor,
        Lsd: torch.Tensor,
        pxY: torch.Tensor,
        pxZ: torch.Tensor,
    ) -> torch.Tensor:
        return solid_angle_factor_tilted(
            Y_px, Z_px,
            Ycen=Ycen, Zcen=Zcen, TRs=TRs, Lsd=Lsd,
            pxY=pxY, pxZ=pxZ,
        )


__all__ = [
    "two_theta_from_R",
    "polarization_factor",
    "solid_angle_factor_flat",
    "solid_angle_factor_tilted",
    "PolarizationCorrection",
    "SolidAngleCorrection",
]
