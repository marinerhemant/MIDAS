"""Differentiable per-spot tilt + distortion + residual-correction kernel.

Wraps ``midas_calibrate.geometry_torch.pixel_to_REta_torch`` and adds the
optional residual-correction-map lookup. The C reference is
``FitSetupParamsAllZarr.c:200-245`` (problem_function) and
``FitSetupParamsAllZarr.c:357-400`` (CorrectTiltSpatialDistortion).

Every input that flows into the geometry — Lsd, ybc, zbc, tx/ty/tz, p0..p14,
residual map values — supports ``requires_grad=True``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


def _bilinear_residual_corr(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    corr_map: Optional[torch.Tensor],
) -> torch.Tensor:
    """Bilinear lookup of a residual-correction map at floating-point pixel
    coordinates.

    The map is a ``(NrPixelsZ, NrPixelsY)`` float64 tensor. Out-of-bounds
    pixels contribute zero correction. Matches the C
    ``dg_residual_corr_lookup`` semantics (which is bilinear per
    ``DetectorGeometry.h:34``).
    """
    if corr_map is None:
        return torch.zeros_like(Y_pix)

    # grid_sample expects (N, C, H_in, W_in) and a (N, H_out, W_out, 2)
    # normalised grid in [-1, 1].
    Hp, Wp = corr_map.shape
    yf = Y_pix.flatten()
    zf = Z_pix.flatten()

    # Normalise pixel indices to [-1, 1]; align_corners=True so 0 -> -1, N-1 -> +1.
    grid_x = 2.0 * yf / max(Wp - 1, 1) - 1.0   # Y_pix is column index
    grid_y = 2.0 * zf / max(Hp - 1, 1) - 1.0   # Z_pix is row index
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, -1, 2)
    img = corr_map.unsqueeze(0).unsqueeze(0).to(Y_pix.dtype)
    sampled = torch.nn.functional.grid_sample(
        img, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
    )
    return sampled.flatten().reshape_as(Y_pix)


def apply_tilt_distortion(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    *,
    Lsd: torch.Tensor,
    BC_y: torch.Tensor,
    BC_z: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    tz: torch.Tensor,
    p_coeffs: torch.Tensor,         # [15] — p0..p14
    px: torch.Tensor,               # µm
    rho_d: torch.Tensor,            # px
    parallax: Optional[torch.Tensor] = None,
    residual_corr_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute corrected ``(Y_lab_µm, Z_lab_µm)`` for each spot.

    Returns the lab-frame corrected coordinates (in µm) — the same as the C
    ``CorrectTiltSpatialDistortion`` outputs. All operations are autograd-traced.
    """
    from midas_calibrate.geometry_torch import pixel_to_REta_torch

    if parallax is None:
        parallax = torch.zeros((), dtype=Y_pix.dtype, device=Y_pix.device)

    R_corr_px, eta = pixel_to_REta_torch(
        Y_pix, Z_pix,
        Lsd=Lsd, BC_y=BC_y, BC_z=BC_z,
        tx=tx, ty=ty, tz=tz, p_coeffs=p_coeffs,
        parallax=parallax, px=px, rho_d=rho_d,
    )

    # Residual correction map adds a per-pixel ΔR (in px), separately from the
    # polynomial. The C code adds it to Rcorr (in µm) after multiplying by px.
    if residual_corr_map is not None:
        R_corr_px = R_corr_px + _bilinear_residual_corr(Y_pix, Z_pix, residual_corr_map)

    # Lab-frame coordinates in µm: Y = -R·sin(eta), Z = R·cos(eta).
    R_um = R_corr_px * px
    Y_lab = -R_um * torch.sin(eta * _DEG2RAD)
    Z_lab = R_um * torch.cos(eta * _DEG2RAD)
    return Y_lab, Z_lab


def calc_eta_angle_local(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """C ``CalcEtaAngleLocal`` — ``acos`` form with sign flip on ``y > 0``."""
    r = torch.sqrt(y * y + z * z).clamp(min=1e-30)
    alpha = _RAD2DEG * torch.acos(torch.clamp(z / r, min=-1.0, max=1.0))
    return torch.where(y > 0, -alpha, alpha)


def correct_wedge_no_op(
    y: torch.Tensor, z: torch.Tensor, Lsd: torch.Tensor, omega_ini: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wedge-free path: passthrough geometry.

    Used when ``|wedge| < 1e-10``. Returns
    ``(y_out, z_out, omega_out, eta_out, ttheta_out)``.
    """
    eta = calc_eta_angle_local(y, z)
    R = torch.sqrt(y * y + z * z)
    tth = _RAD2DEG * torch.atan(R / Lsd)
    return y, z, omega_ini, eta, tth


def correct_wedge_full(
    y: torch.Tensor, z: torch.Tensor, Lsd: torch.Tensor, omega_ini: torch.Tensor,
    wavelength: torch.Tensor, wedge: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full wedge correction — direct port of C ``CorrectWedge`` (FitSetupParamsAllZarr.c:404-545).

    Returns ``(y_out, z_out, omega_out, eta_out, ttheta_out)``. All tensors
    are the same shape as ``y``.
    """
    # Branch selection per spot. We compute the no-wedge result and the full
    # wedge result and select per-element.
    y_nw, z_nw, om_nw, eta_nw, tth_nw = correct_wedge_no_op(y, z, Lsd, omega_ini)

    eta = calc_eta_angle_local(y, z)
    ring_radius = torch.sqrt(y * y + z * z)
    tth = _RAD2DEG * torch.atan(ring_radius / Lsd)

    cos_ome = torch.cos(omega_ini * _DEG2RAD)
    sin_ome = torch.sin(omega_ini * _DEG2RAD)
    theta = tth / 2
    sin_th = torch.sin(theta * _DEG2RAD)
    cos_th = torch.cos(theta * _DEG2RAD)
    ds = 2 * sin_th / wavelength
    cos_w = torch.cos(wedge * _DEG2RAD)
    sin_w = torch.sin(wedge * _DEG2RAD)
    sin_eta = torch.sin(eta * _DEG2RAD)
    cos_eta = torch.cos(eta * _DEG2RAD)
    k1 = -ds * sin_th
    k2 = -ds * cos_th * sin_eta
    k3 = ds * cos_th * cos_eta
    # eta=±90 special-case from the C code — k3 = 0; k2 = ∓CosTheta.
    eta_eq_p90 = torch.isclose(eta, torch.tensor(90.0, dtype=eta.dtype, device=eta.device))
    eta_eq_m90 = torch.isclose(eta, torch.tensor(-90.0, dtype=eta.dtype, device=eta.device))
    k2 = torch.where(eta_eq_p90, -cos_th, k2)
    k3 = torch.where(eta_eq_p90, torch.zeros_like(k3), k3)
    k2 = torch.where(eta_eq_m90, cos_th, k2)
    k3 = torch.where(eta_eq_m90, torch.zeros_like(k3), k3)

    k1f = (k1 * cos_w) + (k3 * sin_w)
    k2f = k2
    k3f = (k3 * cos_w) - (k1 * sin_w)
    G1a = (k1f * cos_ome) + (k2f * sin_ome)
    G2a = (k2f * cos_ome) - (k1f * sin_ome)
    G3a = k3f
    LenGa = torch.sqrt(G1a * G1a + G2a * G2a + G3a * G3a).clamp(min=1e-30)
    g1 = G1a * ds / LenGa
    g2 = G2a * ds / LenGa
    g3 = G3a * ds / LenGa
    LenG = torch.sqrt(g1 * g1 + g2 * g2 + g3 * g3).clamp(min=1e-30)
    k1i = -(LenG * LenG * wavelength) / 2
    tth_new = 2 * _RAD2DEG * torch.asin(torch.clamp(wavelength * LenG / 2, min=-1.0, max=1.0))
    ring_radius_new = Lsd * torch.tan(tth_new * _DEG2RAD)

    # After wedge=0 (the C code re-zeroes wedge here for the in-frame solve):
    sin_w = torch.zeros_like(sin_w)
    cos_w = torch.ones_like(cos_w)
    A = (k1i + g3 * sin_w) / cos_w
    a_sin = g1 * g1 + g2 * g2
    b_sin = 2 * A * g2
    c_sin = A * A - g1 * g1
    a_cos = a_sin
    b_cos = -2 * A * g1
    c_cos = A * A - g2 * g2

    par_sin = b_sin * b_sin - 4 * a_sin * c_sin
    par_cos = b_cos * b_cos - 4 * a_cos * c_cos
    p_check_sin = par_sin < 0
    p_check_cos = par_cos < 0
    p_sin = torch.sqrt(torch.where(p_check_sin, torch.zeros_like(par_sin), par_sin))
    p_cos = torch.sqrt(torch.where(p_check_cos, torch.zeros_like(par_cos), par_cos))

    a_sin_safe = torch.where(a_sin.abs() < 1e-30, torch.full_like(a_sin, 1e-30), a_sin)
    a_cos_safe = torch.where(a_cos.abs() < 1e-30, torch.full_like(a_cos, 1e-30), a_cos)

    sin_om1 = (-b_sin - p_sin) / (2 * a_sin_safe)
    sin_om2 = (-b_sin + p_sin) / (2 * a_sin_safe)
    cos_om1 = (-b_cos - p_cos) / (2 * a_cos_safe)
    cos_om2 = (-b_cos + p_cos) / (2 * a_cos_safe)

    sin_om1 = sin_om1.clamp(min=-1.0, max=1.0)
    sin_om2 = sin_om2.clamp(min=-1.0, max=1.0)
    cos_om1 = cos_om1.clamp(min=-1.0, max=1.0)
    cos_om2 = cos_om2.clamp(min=-1.0, max=1.0)

    sin_om1 = torch.where(p_check_sin, torch.zeros_like(sin_om1), sin_om1)
    sin_om2 = torch.where(p_check_sin, torch.zeros_like(sin_om2), sin_om2)
    cos_om1 = torch.where(p_check_cos, torch.zeros_like(cos_om1), cos_om1)
    cos_om2 = torch.where(p_check_cos, torch.zeros_like(cos_om2), cos_om2)

    opt1 = torch.abs(sin_om1 * sin_om1 + cos_om1 * cos_om1 - 1)
    opt2 = torch.abs(sin_om1 * sin_om1 + cos_om2 * cos_om2 - 1)
    use_opt1 = opt1 < opt2
    omega1 = torch.where(use_opt1,
                          _RAD2DEG * torch.atan2(sin_om1, cos_om1),
                          _RAD2DEG * torch.atan2(sin_om1, cos_om2))
    omega2 = torch.where(use_opt1,
                          _RAD2DEG * torch.atan2(sin_om2, cos_om2),
                          _RAD2DEG * torch.atan2(sin_om2, cos_om1))

    diff1 = torch.abs(omega1 - omega_ini)
    diff2 = torch.abs(omega2 - omega_ini)
    diff1_360 = torch.abs(diff1 - 360) < 0.1
    diff2_360 = torch.abs(diff2 - 360) < 0.1
    diff1 = torch.where(diff1_360, torch.zeros_like(diff1), diff1)
    diff2 = torch.where(diff2_360, torch.zeros_like(diff2), diff2)
    omega1 = torch.where(diff1_360, -omega1, omega1)
    omega2 = torch.where(diff2_360, -omega2, omega2)

    omega_out = torch.where(diff1 < diff2, omega1, omega2)
    # Eta of the wedge-free G, R_z(omega_out) g. The C CorrectWedge
    # (FitSetupParamsAllZarr.c) computes k2N/k3N for exactly this and then
    # uses the OBSERVED (k2, k3) instead, leaving an eta error
    # ~ W tan(theta) / sin(eta) per spot; the refiner's CorrectForOme had the
    # same slip (grain positions ~440 um off at Wedge = 2 deg). Only reached
    # for |wedge| >= 1e-10 (the no-op branch below returns first otherwise).
    s_oo = torch.sin(omega_out * _DEG2RAD)
    c_oo = torch.cos(omega_out * _DEG2RAD)
    eta_solved = calc_eta_angle_local(g1 * s_oo + g2 * c_oo, g3)
    sin_eta_s = torch.sin(eta_solved * _DEG2RAD)
    cos_eta_s = torch.cos(eta_solved * _DEG2RAD)
    y_out = -ring_radius_new * sin_eta_s
    z_out = ring_radius_new * cos_eta_s
    eta_out = eta_solved
    tth_out = _RAD2DEG * torch.atan(ring_radius_new / Lsd)

    # Branch on wedge magnitude.
    no_wedge_mask = wedge.abs() < 1e-10
    if bool(no_wedge_mask):
        return y_nw, z_nw, om_nw, eta_nw, tth_nw

    return y_out, z_out, omega_out, eta_out, tth_out
