"""Differentiable detector forward model in pure torch.

Mirrors midas_integrate.geometry.pixel_to_REta but in torch ops so the result
is autograd-traced through to geometry parameters (Lsd, BC, tilts, p0..p14,
parallax, wavelength).  All units match MIDAS conventions:

  * Lsd in μm; BC in pixels; tilts in degrees (Rx · Ry · Rz order).
  * Eta in degrees, atan2(-Y', Z'), [-180, 180).
  * Distortion ΔR/RhoD = polynomial in (R/RhoD).

Inverse mapping (R, η) → (Y_pix, Z_pix) is a Newton-Raphson loop and is
provided here for the alternating engine; the joint engine forward-models
predicted-R(geometry) directly without inversion.
"""
from __future__ import annotations

from typing import Tuple

import torch


def build_tilt_matrix_torch(tx: torch.Tensor, ty: torch.Tensor, tz: torch.Tensor) -> torch.Tensor:
    """TRs = Rx(tx) · Ry(ty) · Rz(tz),  angles in degrees, [3,3] tensor.

    All inputs scalar torch tensors; output [3,3] tensor with grad flowing.
    """
    one = torch.ones((), dtype=tx.dtype, device=tx.device)
    zero = torch.zeros((), dtype=tx.dtype, device=tx.device)
    deg2rad = torch.tensor(0.017453292519943295, dtype=tx.dtype, device=tx.device)
    cx, sx = torch.cos(tx * deg2rad), torch.sin(tx * deg2rad)
    cy, sy = torch.cos(ty * deg2rad), torch.sin(ty * deg2rad)
    cz, sz = torch.cos(tz * deg2rad), torch.sin(tz * deg2rad)

    Rx = torch.stack([torch.stack([one, zero, zero]),
                      torch.stack([zero, cx, -sx]),
                      torch.stack([zero, sx, cx])])
    Ry = torch.stack([torch.stack([cy, zero, sy]),
                      torch.stack([zero, one, zero]),
                      torch.stack([-sy, zero, cy])])
    Rz = torch.stack([torch.stack([cz, -sz, zero]),
                      torch.stack([sz, cz, zero]),
                      torch.stack([zero, zero, one])])
    return Rx @ Ry @ Rz


def pixel_to_REta_torch(
    Y_pix: torch.Tensor,            # [...] pixel Y indices
    Z_pix: torch.Tensor,            # [...] pixel Z indices
    *,
    Lsd: torch.Tensor,
    BC_y: torch.Tensor,
    BC_z: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    tz: torch.Tensor,
    p_coeffs: torch.Tensor,         # [15] distortion coefficients p0..p14
    parallax: torch.Tensor,
    px: torch.Tensor,               # μm; mean of pxY/pxZ
    rho_d: torch.Tensor,            # px; distortion normalization radius
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable pixel → (R [px], Eta [deg]).

    Byte-for-byte port of midas_integrate.geometry.pixel_to_REta — vector layout
    ``(0, Yc, Zc)`` → tilt → add Lsd along x; then 2θ via Lsd/Xp projection;
    distortion applied as a *multiplicative scaling* on Rad (NOT ΔR).
    """
    deg2rad = torch.tensor(0.017453292519943295, dtype=Y_pix.dtype, device=Y_pix.device)
    rad2deg = torch.tensor(57.29577951308232, dtype=Y_pix.dtype, device=Y_pix.device)

    # Untilted physical coordinates (μm) — note the X-component is 0 before tilt.
    Yc = (-Y_pix + BC_y) * px
    Zc = (Z_pix - BC_z) * px

    TRs = build_tilt_matrix_torch(tx, ty, tz)
    # Apply tilt to (0, Yc, Zc): only columns 1 and 2 of TRs matter.
    abcpr_x = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    abcpr_y = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    abcpr_z = TRs[2, 1] * Yc + TRs[2, 2] * Zc

    XYZ_x = Lsd + abcpr_x
    XYZ_y = abcpr_y
    XYZ_z = abcpr_z
    safe_x = torch.where(XYZ_x.abs() < 1e-30, torch.full_like(XYZ_x, 1e-30), XYZ_x)
    rad_um = (Lsd / safe_x) * torch.sqrt(XYZ_y * XYZ_y + XYZ_z * XYZ_z)
    eta_tilted = rad2deg * torch.atan2(-XYZ_y, XYZ_z)

    # Radial distortion via the shared midas_distortion kernel (single source
    # of truth, identical to peakfit + calibrate-v2). p_coeffs is the legacy
    # v1 p0..p14 vector; shim it to v2 canonical order (a differentiable gather)
    # and evaluate. The kernel applies EtaT = 90 - eta internally.
    from midas_distortion import distortion_factor, v1_to_v2_coeffs

    R_norm = rad_um / rho_d if rho_d > 0 else torch.zeros_like(rad_um)
    dist = distortion_factor(R_norm, eta_tilted, v1_to_v2_coeffs(p_coeffs))
    Rt = rad_um * dist / px

    if isinstance(parallax, torch.Tensor):
        if parallax.abs().item() > 0:
            two_theta = torch.atan(rad_um / Lsd)
            Rt = Rt + parallax * torch.sin(two_theta) / px
    return Rt, eta_tilted


def predict_R_at_pixel(Y_pix: torch.Tensor, Z_pix: torch.Tensor,
                        params_vec: torch.Tensor, px: float, rho_d: float) -> torch.Tensor:
    """Compute R [px] given a packed geometry parameter vector.

    Layout of params_vec (length 23):
        [0]   Lsd
        [1,2] BC_y, BC_z
        [3,4] ty, tz                  (tx fixed at 0 for now)
        [5..19] p0..p14
        [20]  parallax
        [21]  wavelength             (unused in geometry; needed for predict_R_ideal)
        [22]  tx                     (fixed at the input value)
    """
    Lsd = params_vec[0]
    BC_y = params_vec[1]; BC_z = params_vec[2]
    ty = params_vec[3]; tz = params_vec[4]
    p = params_vec[5:20]
    parallax = params_vec[20]
    tx = params_vec[22]
    px_t = torch.as_tensor(px, dtype=params_vec.dtype, device=params_vec.device)
    rho_d_t = torch.as_tensor(rho_d, dtype=params_vec.dtype, device=params_vec.device)
    R, _eta = pixel_to_REta_torch(
        Y_pix, Z_pix, Lsd=Lsd, BC_y=BC_y, BC_z=BC_z,
        tx=tx, ty=ty, tz=tz, p_coeffs=p, parallax=parallax,
        px=px_t, rho_d=rho_d_t,
    )
    return R


def predict_R_ideal(two_theta_deg: torch.Tensor, params_vec: torch.Tensor, px: float) -> torch.Tensor:
    """R_ideal[px] = Lsd · tan(2θ) / px.  When wavelength is refined, two_theta_deg
    must be recomputed externally from d-spacing — this function expects the
    already-resolved 2θ values."""
    Lsd = params_vec[0]
    return Lsd * torch.tan(two_theta_deg * 0.017453292519943295) / px


def predict_two_theta_from_d(d_spacing_A: torch.Tensor, wavelength_A: torch.Tensor) -> torch.Tensor:
    """Bragg's law in torch:  2θ = 2 arcsin(λ / 2d)."""
    s = wavelength_A / (2.0 * d_spacing_A)
    s = s.clamp(min=-0.999999, max=0.999999)
    return 2.0 * torch.asin(s) * 57.29577951308232  # → deg
