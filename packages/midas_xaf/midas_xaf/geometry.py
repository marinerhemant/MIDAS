"""Geometry: mountings, omega wedges, exit-cone access, and the remount transform.

This module encodes what makes XAF-HEDM distinct from ordinary FF-HEDM:

1. **Narrow omega wedges.**  Each face opening only admits the incident beam
   over +/- ``wedge_half`` degrees around the four wedge centres, so an omega
   solution is accessible only inside one of those windows.
2. **Exit-cone (opening) cap on 2theta.**  The *diffracted* beam must clear the
   same ~15 deg opening, so a spot is accessible only if ``2theta <=
   opening_half`` (to first order -- the wedges are narrow and centred on the
   beam).  This is the constraint that ties strain sensitivity to opening
   angle and drives the 15 vs 20 deg cell decision.
3. **Two orthogonal-axis mountings.**  The second mounting is the sample rigidly
   rotated by ``remount_angle`` about ``remount_axis`` (default 90 deg about the
   beam axis), which swings the top/bottom faces onto the equator -- giving an
   orthogonal rotation axis in the crystal frame and filling the first
   mounting's missing cone.

Lab frame convention: beam ``+x``, rotation axis ``+z`` (vertical), transverse
``+y``.  Omega is rotation about ``+z``; wedge centres 0/90/180/270 correspond
to the four equatorial faces facing the beam.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from .config import XAFConfig


# --------------------------------------------------------------------------- #
#  Detector geometry                                                          #
# --------------------------------------------------------------------------- #
def build_hedm_geometry(cfg: XAFConfig):
    """Construct a :class:`midas_diffract.HEDMGeometry` from an XAFConfig.

    A single FF detector distance; omega spans a full turn so the forward model
    reports every reflection's omega and we gate to the wedges afterwards.
    """
    import midas_diffract as md

    Lsd = cfg.resolved_Lsd_um()
    # Full-turn omega so no reflection is dropped before wedge masking.
    n_frames = max(1, int(round(360.0 / abs(cfg.omega_step_deg))))
    return md.HEDMGeometry(
        Lsd=Lsd,
        y_BC=0.5 * cfg.n_pixels_y,
        z_BC=0.5 * cfg.n_pixels_z,
        px=cfg.px_um,
        omega_start=-180.0,
        omega_step=cfg.omega_step_deg,
        n_frames=n_frames,
        n_pixels_y=cfg.n_pixels_y,
        n_pixels_z=cfg.n_pixels_z,
        min_eta=cfg.min_eta_deg,
        wavelength=cfg.wavelength_A,
    )


# --------------------------------------------------------------------------- #
#  Remount (two orthogonal-axis mountings)                                    #
# --------------------------------------------------------------------------- #
def remount_matrix(cfg: XAFConfig) -> np.ndarray:
    """3x3 rigid-body rotation carrying mounting-1 orientations to mounting-2."""
    from midas_stress.orientation import axis_angle_to_orient_mat
    axis = list(cfg.remount_axis)
    return np.asarray(
        axis_angle_to_orient_mat(axis, cfg.remount_angle_deg),
        dtype=np.float64).reshape(3, 3)


def mounting_matrix(cfg: XAFConfig, mounting: int) -> np.ndarray:
    """Rigid transform applied to grain orientations for a given mounting.

    Mounting 0 is identity.  If ``cfg.remount_specs`` is given, mounting ``m``
    uses its explicit ``(axis, angle)`` remount from mounting 0 -- this is how
    3 *orthogonal* rotation axes are set up (R_x(90) for M2, R_y(90) for M3).
    Otherwise the single ``remount_axis`` is composed ``m`` times (the default
    two-mounting design: M1 is the 90° remount about the beam).
    """
    if mounting == 0:
        return np.eye(3)
    if cfg.remount_specs is not None:
        from midas_stress.orientation import axis_angle_to_orient_mat
        axis, angle = cfg.remount_specs[mounting - 1]
        return np.asarray(axis_angle_to_orient_mat(list(axis), angle),
                          dtype=np.float64).reshape(3, 3)
    R = np.eye(3)
    Rm = remount_matrix(cfg)
    for _ in range(mounting):
        R = Rm @ R
    return R


# --------------------------------------------------------------------------- #
#  Access masks                                                               #
# --------------------------------------------------------------------------- #
def wedge_mask(omega_rad: torch.Tensor, cfg: XAFConfig) -> torch.Tensor:
    """True where ``omega`` falls inside any of the four wedge windows."""
    deg = torch.rad2deg(omega_rad)
    centers = torch.tensor(cfg.wedge_centers_deg, dtype=deg.dtype, device=deg.device)
    # signed angular distance to each centre, wrapped to (-180, 180]
    d = deg.unsqueeze(-1) - centers  # (..., C)
    d = (d + 180.0) % 360.0 - 180.0
    nearest = d.abs().min(dim=-1).values
    return nearest <= cfg.wedge_half_deg


# The six cube-face outward normals in the *sample* frame.  Every face has an
# opening, so the diffracted beam may exit through whichever one it aligns with.
_SAMPLE_FACE_NORMALS = torch.tensor(
    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
     [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float64)


def scattered_dir(two_theta_rad: torch.Tensor, eta_rad: torch.Tensor) -> torch.Tensor:
    """Lab-frame unit vector of the diffracted beam for each spot.

    Convention taken from the forward model (forward.py:915-921, 1203):
    ``d = (cos2theta, -sin2theta*sin(eta), sin2theta*cos(eta))`` with beam +x,
    transverse +y, vertical (rotation-axis) +z.
    """
    s = torch.sin(two_theta_rad)
    return torch.stack([torch.cos(two_theta_rad),
                        -s * torch.sin(eta_rad),
                        s * torch.cos(eta_rad)], dim=-1)   # (..., 3)


def exit_aperture_mask(spot_desc, cfg: XAFConfig) -> torch.Tensor:
    """True where the diffracted beam clears a face opening at the spot's omega.

    The six sample-face normals rotate with the sample about +z by ``omega``.
    A spot exits iff its diffracted beam lies within ``opening_half`` of at
    least one rotated face normal.  This is the physically correct aperture:
    because the transmitting cone tilts with omega, the accessible detector
    region shifts and clips asymmetrically with rotation (the shadowing the
    cell body imposes on the exit path).  Reduces to ``2theta <= opening_half``
    at wedge centre.
    """
    if cfg.exit_model == "tth_cap":
        return torch.rad2deg(spot_desc.two_theta) <= cfg.opening_half_deg

    om = spot_desc.omega
    # Native dtype (float32 from the forward model) is ample for a geometric
    # gate, and avoids float64 -- which MPS does not support.
    d = scattered_dir(spot_desc.two_theta, spot_desc.eta)          # (...,K,M,3)
    co, so = torch.cos(om), torch.sin(om)
    normals = _SAMPLE_FACE_NORMALS.to(device=om.device, dtype=om.dtype)  # (6,3)
    # Rotate each sample normal about +z by omega -> lab normal per spot.
    cos_thresh = math.cos(math.radians(cfg.opening_half_deg))
    best = None
    for i in range(normals.shape[0]):
        nx, ny, nz = normals[i]
        lx = nx * co - ny * so
        ly = nx * so + ny * co
        lz = torch.full_like(co, float(nz))
        cosang = d[..., 0] * lx + d[..., 1] * ly + d[..., 2] * lz
        best = cosang if best is None else torch.maximum(best, cosang)
    return best >= cos_thresh


# Pilatus module tiling: 487x195 px modules, 7 px (h) / 17 px (v) gaps.
_PILATUS_MOD_H, _PILATUS_GAP_H = 487, 7
_PILATUS_MOD_V, _PILATUS_GAP_V = 195, 17


def detector_live_mask(y_pixel: torch.Tensor, z_pixel: torch.Tensor,
                       cfg: XAFConfig) -> torch.Tensor:
    """True where a pixel is on live silicon (not an inter-module gap/beamstop).

    For ``detector_type="pilatus2m"`` a coordinate is in a gap when its position
    within a (module+gap) period exceeds the module size.  The central beamstop
    removes a disk around the beam centre.
    """
    live = torch.ones_like(y_pixel, dtype=torch.bool)
    if cfg.detector_type == "pilatus2m":
        ph = _PILATUS_MOD_H + _PILATUS_GAP_H
        pv = _PILATUS_MOD_V + _PILATUS_GAP_V
        live = live & (torch.remainder(y_pixel, ph) < _PILATUS_MOD_H) \
                    & (torch.remainder(z_pixel, pv) < _PILATUS_MOD_V)
    elif cfg.detector_type not in ("none", ""):
        raise ValueError(f"unknown detector_type {cfg.detector_type!r}")
    if cfg.beamstop_radius_px > 0.0:
        yc = 0.5 * cfg.n_pixels_y
        zc = 0.5 * cfg.n_pixels_z
        r2 = (y_pixel - yc) ** 2 + (z_pixel - zc) ** 2
        live = live & (r2 > cfg.beamstop_radius_px ** 2)
    return live


def accessible_mask(spot_desc, cfg: XAFConfig) -> torch.Tensor:
    """Combine the forward model's ``valid`` mask with the incident-wedge,
    exit-aperture (shadowing), and detector dead-region gates.  Shaped like
    ``spot_desc.omega``."""
    valid = spot_desc.valid > 0.5
    mask = (valid
            & wedge_mask(spot_desc.omega, cfg)
            & exit_aperture_mask(spot_desc, cfg))
    if cfg.detector_type not in ("none", "") or cfg.beamstop_radius_px > 0.0:
        # y_pixel/z_pixel are (...,K,M) for single-distance FF (what XAF uses).
        yp, zp = spot_desc.y_pixel, spot_desc.z_pixel
        if yp.dim() == mask.dim():
            mask = mask & detector_live_mask(yp, zp, cfg)
    return mask


def geometry_summary(cfg: XAFConfig) -> dict:
    """Human-readable rundown of the resolved geometry (for logs / sweep rows)."""
    Lsd = cfg.resolved_Lsd_um()
    half_um = 0.5 * min(cfg.n_pixels_y, cfg.n_pixels_z) * cfg.px_um
    tth_at_edge = math.degrees(math.atan(half_um / Lsd))
    return {
        "energy_keV": cfg.energy_keV,
        "wavelength_A": cfg.wavelength_A,
        "opening_full_deg": cfg.opening_full_deg,
        "tth_max_deg": cfg.tth_max_deg,
        "wedge_half_deg": cfg.wedge_half_deg,
        "Lsd_mm": Lsd / 1000.0,
        "tth_at_detector_edge_deg": tth_at_edge,
        # If the detector edge is inside the exit cone, the detector -- not the
        # opening -- is the binding limit on 2theta; the sweep should flag this.
        "detector_limited": tth_at_edge < cfg.tth_max_deg,
    }
