"""Pixel → (R, η) projection — fully differentiable, with optional panels.

This module is a clean superset of v1's ``geometry_torch.pixel_to_REta_torch``:

  * pxY, pxZ are tensors (refinable; v1 had them as Python floats).
  * tx is refinable (v1 fixed it at 0).
  * Per-panel rigid-body transform is applied before projection (v1 had no
    panel support in the differentiable path).
  * Parallax is always-on and graph-clean (v1 had a ``.item()`` break).
  * Distortion factor is delegated to :mod:`forward.distortion` so the basis
    is extensible.

Inputs are tensors carrying autograd if their parent Parameter is refined;
all internal arithmetic is torch and the output (R, η) is differentiable in
every refined input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .distortion import apply_distortion, v1_term_layout, HarmonicTerm
from .lattice import LATTICES, lattice_to_phys
from .panels import (
    PanelLayout, apply_panel_shifts,
    per_panel_lsd_offset, per_panel_p2_offset,
)
from .parallax import parallax_correction


_DEG2RAD = 0.017453292519943295
_RAD2DEG = 57.29577951308232


def build_tilt_matrix(tx: torch.Tensor, ty: torch.Tensor, tz: torch.Tensor) -> torch.Tensor:
    """Intrinsic Z-Y-X Euler rotation R = Rx(tx) Ry(ty) Rz(tz)."""
    one = torch.ones((), dtype=tx.dtype, device=tx.device)
    zero = torch.zeros((), dtype=tx.dtype, device=tx.device)
    cx, sx = torch.cos(tx * _DEG2RAD), torch.sin(tx * _DEG2RAD)
    cy, sy = torch.cos(ty * _DEG2RAD), torch.sin(ty * _DEG2RAD)
    cz, sz = torch.cos(tz * _DEG2RAD), torch.sin(tz * _DEG2RAD)
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


@dataclass
class ForwardOutputs:
    """Convenience container for the forward outputs at a pixel set."""

    R_px: torch.Tensor       # corrected radius (px)
    eta_deg: torch.Tensor    # azimuth (deg, [-180, 180))
    rad_um: torch.Tensor     # raw projected radius (μm), pre-distortion
    two_theta_rad: torch.Tensor


def pixel_to_REta(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    *,
    Lsd: torch.Tensor,
    BC_y: torch.Tensor,
    BC_z: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    tz: torch.Tensor,
    p_coeffs: torch.Tensor,         # [15] amplitudes/phases
    parallax: torch.Tensor,
    pxY: torch.Tensor,              # μm; refinable (cartesian)
    pxZ: Optional[torch.Tensor] = None,
    rho_d: torch.Tensor = None,     # px; distortion normalisation radius
    panel_layout: Optional[PanelLayout] = None,
    panel_idx: Optional[torch.Tensor] = None,
    delta_yz: Optional[torch.Tensor] = None,
    delta_theta: Optional[torch.Tensor] = None,
    delta_lsd_panel: Optional[torch.Tensor] = None,
    delta_p2_panel: Optional[torch.Tensor] = None,
    fix_panel_id: int = 0,
    harmonic_terms: Optional[list] = None,
    lattice: str = "cartesian",
    apothem: Optional[torch.Tensor] = None,
    orientation_deg: Optional[torch.Tensor] = None,
    residual_corr_map: Optional[torch.Tensor] = None,
) -> ForwardOutputs:
    """Differentiable pixel → (R, η).

    Parameters
    ----------
    Y_pix, Z_pix : tensors of pixel coordinates (any broadcastable shape).
    Lsd, BC_y, BC_z : geometry scalars.
    tx, ty, tz : Euler tilts (deg).
    p_coeffs : [15] tensor of distortion coefficients.
    parallax : depth correction (μm).
    pxY, pxZ : pixel size (μm).  When ``pxZ`` is ``None`` we use ``pxY`` for
        both axes (square pixels).
    rho_d : distortion normalisation radius (px).
    panel_layout, panel_idx, delta_yz, delta_theta, delta_lsd_panel,
    delta_p2_panel, fix_panel_id : optional multi-panel inputs.
    lattice : "cartesian" (default) or "hex_offset_y" (PIXIRAD-style
        staggered hex; pxY/pxZ are derived from ``apothem`` for the
        radial scale).
    apothem : hex apothem (μm), required when ``lattice='hex_offset_y'``.
    orientation_deg : optional in-plane rotation of lattice axes vs
        detector frame; only meaningful when ``lattice='hex_offset_y'``.

    Returns
    -------
    :class:`ForwardOutputs`
    """
    if lattice == "hex_offset_y":
        if apothem is None:
            raise ValueError("lattice='hex_offset_y' requires apothem (μm)")
        # Override pxY/pxZ to the hex-derived pitch so the radial μm→px
        # conversion below uses the correct scale.  Refinable through apothem.
        pxY_eff = 2.0 * apothem
        pxZ_eff = apothem * (3.0 ** 0.5)
    elif lattice == "cartesian":
        pxY_eff = pxY
        pxZ_eff = pxY if pxZ is None else pxZ
    else:
        raise ValueError(f"Unknown lattice {lattice!r}; supported: {LATTICES}")
    px_mean = 0.5 * (pxY_eff + pxZ_eff)

    # ---- Optional per-panel rigid body
    if panel_layout is not None:
        if panel_idx is None or delta_yz is None or delta_theta is None:
            raise ValueError(
                "panel_layout supplied without panel_idx / delta_yz / delta_theta"
            )
        Y_pix, Z_pix = apply_panel_shifts(
            Y_pix, Z_pix, panel_idx, panel_layout,
            delta_yz, delta_theta, fix_panel_id=fix_panel_id,
        )

    # ---- Untilted physical coords (μm) — X-component starts at 0 before tilt.
    Yc, Zc = lattice_to_phys(
        Y_pix, Z_pix,
        lattice=lattice,
        BC_y=BC_y, BC_z=BC_z,
        pxY=pxY_eff if lattice == "cartesian" else None,
        pxZ=pxZ_eff if lattice == "cartesian" else None,
        apothem=apothem,
        orientation_deg=orientation_deg,
    )

    TRs = build_tilt_matrix(tx, ty, tz)
    abcpr_x = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    abcpr_y = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    abcpr_z = TRs[2, 1] * Yc + TRs[2, 2] * Zc

    Lsd_eff = Lsd
    if delta_lsd_panel is not None and panel_idx is not None:
        Lsd_eff = Lsd + per_panel_lsd_offset(
            panel_idx, delta_lsd_panel, fix_panel_id=fix_panel_id,
        )

    XYZ_x = Lsd_eff + abcpr_x
    XYZ_y = abcpr_y
    XYZ_z = abcpr_z

    # Stable division: XYZ_x is positive in physical regimes; clamp tiny.
    safe_x = torch.where(
        XYZ_x.abs() < 1e-30,
        torch.full_like(XYZ_x, 1e-30),
        XYZ_x,
    )
    rad_um = (Lsd_eff / safe_x) * torch.sqrt(XYZ_y * XYZ_y + XYZ_z * XYZ_z)
    eta = _RAD2DEG * torch.atan2(-XYZ_y, XYZ_z)
    two_theta = torch.atan(rad_um / Lsd_eff)

    # ---- Distortion (extensible harmonic basis)
    if rho_d is None:
        # Default normalisation: R itself (unit ρ = 1 at the rim).
        # Caller should supply rho_d if they want a fixed reference.
        rho_d_t = torch.ones_like(rad_um)
    else:
        # apply_distortion expects rho_d in the same units as rad_um (µm),
        # but callers (run_estep_v1) pass it in pixels (RhoD / MaxRingRad
        # are stored in px). Convert to µm via the mean pixel pitch so the
        # normalised radius ρ = rad_um / rho_d_um is dimensionless and
        # matches v1 C's RhoD-in-µm convention.
        rho_d_t = rho_d * px_mean

    p_eff = p_coeffs
    if delta_p2_panel is not None and panel_idx is not None:
        # Inject per-panel p₂ offset by index 2 of a panel-broadcast copy.
        p2_off = per_panel_p2_offset(panel_idx, delta_p2_panel, fix_panel_id=fix_panel_id)
        # Build per-pixel p_eff by adding the panel offset to component 2 only.
        # We'll add the panel offset directly inside the distortion call by
        # constructing a per-pixel ρ² contribution.  For efficiency, evaluate
        # the standard distortion factor first (using bulk p₂), then add the
        # per-panel ρ² · δp₂ term.
        rad_corr = apply_distortion(rad_um, eta, p_eff, rho_d_t, terms=harmonic_terms)
        R_norm = rad_um / rho_d_t
        rad_corr = rad_corr + rad_um * p2_off * R_norm.pow(2)
    else:
        rad_corr = apply_distortion(rad_um, eta, p_eff, rho_d_t, terms=harmonic_terms)

    # Convert μm → px using mean pixel size for the radial scale (matches v1).
    R_px = rad_corr / px_mean

    # Re-project to the global Lsd plane.  Without this, R_px is at the
    # panel-local Lsd plane (Lsd_eff = Lsd + dLsd_panel) while the
    # downstream R_pred = R_ideal_px(2θ, Lsd, …) is at the global Lsd
    # plane.  Comparing the two introduces a systematic bias of
    # |dLsd_panel / Lsd| ≈ 100 μm / 657 mm ≈ 152 μϵ per panel — observable
    # as the 220 μϵ floor on Pilatus eval at v1's MAP.  This matches v1
    # C's ``Rt = Rt * (Lsd / panelLsd)`` rescale in
    # ``DetectorGeometry.c::dg_pixel_to_REta_corr`` line 135.
    if delta_lsd_panel is not None and panel_idx is not None:
        R_px = R_px * (Lsd / Lsd_eff)

    # ---- Parallax (always applied; sin(0) = 0 when parallax = 0)
    R_px = parallax_correction(R_px, rad_um, Lsd_eff, parallax, px_mean)

    # ---- Residual correction map (port of v1 C dg_residual_corr_lookup).
    # Empirical smooth ΔR(Y, Z) absorbed after harmonic distortion converges.
    # Differentiable in (Y_pix, Z_pix) and in the map values via grid_sample.
    if residual_corr_map is not None:
        from .residual_corr import residual_corr_lookup
        R_px = R_px + residual_corr_lookup(Y_pix, Z_pix, residual_corr_map)

    return ForwardOutputs(R_px=R_px, eta_deg=eta, rad_um=rad_um, two_theta_rad=two_theta)


__all__ = ["build_tilt_matrix", "ForwardOutputs", "pixel_to_REta"]
