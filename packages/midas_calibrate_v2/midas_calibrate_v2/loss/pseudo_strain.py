"""Per-spot pseudo-strain residual: 1 - R_obs / R_pred.

This is the v1 calibrant cost.  In v2 it operates on a parameter dict (output
of :func:`unpack_spec`), so refining pxY, pxZ, tx, panels, etc. is automatic.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..forward.geometry import pixel_to_REta
from ..forward.bragg import R_ideal_px, two_theta_from_d
from ..forward.distortion import build_p_coeffs


def pseudo_strain_residual(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    ring_two_theta_deg: torch.Tensor,    # [n_pts] expected 2θ per fitted point
    p: Dict[str, torch.Tensor],          # unpacked parameter dict
    *,
    rho_d: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    panel_layout=None, panel_idx=None,
    ring_idx: Optional[torch.Tensor] = None,
    ring_d_spacing_A: Optional[torch.Tensor] = None,
    lattice: str = "cartesian",
) -> torch.Tensor:
    """Return weighted residual r = w · (1 - R_obs / R_pred).

    The forward model uses ``p["pxY"]`` and ``p["pxZ"]`` if present, and falls
    back to ``p["pxY"]`` only.  Per-panel parameters (delta_yz, delta_theta,
    delta_lsd_panel, delta_p2_panel) are read if the keys are present.

    If ``p["delta_r_k"]`` is present and ``ring_idx`` is supplied, the
    predicted ring radius is shifted by ``delta_r_k[ring_idx]`` (per-ring
    radial offset, F2 in the basis-fixes table).  Pair with the
    ``Σ delta_r_k = 0`` gauge in :mod:`midas_calibrate_v2.loss.constraints`
    to break the gauge-redundancy with ``Lsd``.

    If ``ring_d_spacing_A`` is supplied AND ``"Wavelength"`` is in ``p``,
    the per-fit ring 2θ is recomputed inside the residual via Bragg's law
    ``2θ = 2 arcsin(λ / 2d)`` rather than using the passed-in
    ``ring_two_theta_deg`` as a constant.  This is required to refine
    ``Wavelength`` jointly with the geometry; without it, the autograd
    chain from ``λ`` to the residual is broken at the pre-computed 2θ.

    When ``lattice='hex_offset_y'`` (PIXIRAD-style honeycomb), the
    parameter dict must contain ``p["Apothem"]`` (μm) — the cell
    apothem.  ``pxY`` / ``pxZ`` are derived from it (``2a``, ``a√3``)
    so the radial μm→px scale stays self-consistent with the centroid
    map, and the gradient through Apothem propagates correctly.
    Optional ``p["LatticeOrientation"]`` (deg) rotates the lattice
    axes vs the detector frame.
    """
    apothem = None
    orientation_deg = None
    if lattice == "hex_offset_y":
        if "Apothem" not in p:
            raise KeyError(
                "lattice='hex_offset_y' requires p['Apothem'] (μm)"
            )
        apothem = p["Apothem"]
        sqrt3 = torch.as_tensor(3.0 ** 0.5,
                                 dtype=apothem.dtype, device=apothem.device)
        pxY = 2.0 * apothem
        pxZ = apothem * sqrt3
        orientation_deg = p.get("LatticeOrientation")
    else:
        pxY = p["pxY"]
        pxZ = p.get("pxZ", pxY)

    # If ring d-spacings + a refinable Wavelength are supplied, recompute
    # 2θ inside so the autograd chain through λ stays unbroken.
    if ring_d_spacing_A is not None and "Wavelength" in p:
        ring_two_theta_deg = two_theta_from_d(ring_d_spacing_A, p["Wavelength"])

    # Build the [15] coeffs vector from v2-named scalars.
    p_coeffs = build_p_coeffs(p, dtype=pxY.dtype, device=pxY.device)

    out = pixel_to_REta(
        Y_pix, Z_pix,
        Lsd=p["Lsd"], BC_y=p["BC_y"], BC_z=p["BC_z"],
        tx=p.get("tx", torch.zeros((), dtype=Y_pix.dtype, device=Y_pix.device)),
        ty=p["ty"], tz=p["tz"],
        p_coeffs=p_coeffs,
        parallax=p.get("Parallax", torch.zeros((), dtype=Y_pix.dtype, device=Y_pix.device)),
        pxY=pxY, pxZ=pxZ, rho_d=rho_d,
        panel_layout=panel_layout, panel_idx=panel_idx,
        delta_yz=p.get("panel_delta_yz"),
        delta_theta=p.get("panel_delta_theta"),
        delta_lsd_panel=p.get("panel_delta_lsd"),
        delta_p2_panel=p.get("panel_delta_p2"),
        lattice=lattice,
        apothem=apothem,
        orientation_deg=orientation_deg,
        residual_corr_map=p.get("residual_corr_map"),
    )
    px_mean = 0.5 * (pxY + pxZ)
    R_pred = R_ideal_px(ring_two_theta_deg, p["Lsd"], px_mean)
    if "delta_r_k" in p and ring_idx is not None:
        R_pred = R_pred + p["delta_r_k"][ring_idx]
    # Per-(panel, ring) radial offset: shift R_pred by δR[panel, ring].  Gap
    # pixels (panel_idx < 0) get no shift.  Requires both indices.
    if "panel_ring_delta_r" in p and panel_idx is not None and ring_idx is not None:
        prd = p["panel_ring_delta_r"]
        safe_pid = torch.where(panel_idx >= 0, panel_idx,
                                torch.zeros_like(panel_idx)).long()
        add = prd[safe_pid, ring_idx.long()]
        add = torch.where(panel_idx >= 0, add, torch.zeros_like(add))
        R_pred = R_pred + add
    r = 1.0 - out.R_px / R_pred
    if weights is not None:
        r = r * weights
    return r


def pseudo_strain_loss(*args, **kwargs) -> torch.Tensor:
    r = pseudo_strain_residual(*args, **kwargs)
    return 0.5 * (r * r).sum()


__all__ = ["pseudo_strain_residual", "pseudo_strain_loss"]
