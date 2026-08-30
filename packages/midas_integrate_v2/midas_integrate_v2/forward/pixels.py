"""Differentiable pixel → (R, η) for an :class:`IntegrationSpec`.

Thin shim over :func:`midas_calibrate_v2.forward.geometry.pixel_to_REta`
that takes the parameters from a v2 :class:`IntegrationSpec` instead of
loose tensors. Used by the soft-binning integrate path in Phase 2.

The full grid evaluator :func:`eval_pixel_REta` returns the per-pixel
``(R, η)`` for every detector pixel so the caller can soft-bin or
evaluate residual corrections in pixel space.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from midas_calibrate_v2.forward.geometry import pixel_to_REta as _v2_pixel_to_REta
from midas_calibrate_v2.forward.panels import PanelLayout, panel_idx_for_points

from ..spec import IntegrationSpec, DISTORTION_NAMES

class PanelShiftsMissingWarning(UserWarning):
    """``PanelShiftsFile`` is named in the spec but could not be read.

    The panel layout is still applied, with all shifts zero. Raised rather
    than passed over in silence: on a multi-panel detector this is the
    difference between using and discarding the panel calibration.
    """


#: Per-panel corrections, keyed by the panel-defining spec fields. Building
#: these re-reads PanelShiftsFile and re-derives the layout, and the subpixel
#: geometry calls into here K^2 times per build, so the cache is not optional.
_PANEL_CACHE: dict = {}

#: Loaded residual maps, keyed by (path, mtime, size, dtype, device).
#: The map is one float64 per detector pixel -- 66 MB at 2880 x 2880 -- and
#: ``eval_pixel_REta`` is called once per geometry build, so re-reading it from
#: disk each time is the difference between a build that is free and one that
#: is not.
_RESID_CACHE: dict = {}


def _load_residual_map(spec: IntegrationSpec) -> Optional[torch.Tensor]:
    """The empirical ΔR(Y, Z) map named by ``spec.ResidualCorrectionMap``.

    This is what the fitted analytic geometry could **not** absorb: measured
    per pixel, in pixels of radius, written by ``midas_calibrate_v2`` when
    calibrating with ``build_residual_corr=True``. v1 applies it in
    ``midas_integrate.detector_mapper``; until this function existed the v2
    binning path declared the field and silently ignored it, so every
    integration through v2 discarded a correction the calibration had already
    measured. On an 11-ID-C CeO2 calibrant that map has an rms of 0.068 px --
    negligible for intensity, but the same order as the azimuthal systematic
    that limits strain there.

    Returns ``None`` when no map is configured, which is the common case and
    leaves the geometry exactly as before.
    """
    path = str(getattr(spec, "ResidualCorrectionMap", "") or "")
    if not path:
        return None

    NY, NZ = int(spec.NrPixelsY), int(spec.NrPixelsZ)
    dev, dt = spec.device(), spec.dtype()
    try:
        st = os.stat(path)
    except OSError as exc:
        # Loudly: a configured-but-unreadable map means the caller believes a
        # correction is being applied that is not. Silence here would produce a
        # geometry that is wrong in exactly the way the map exists to fix.
        raise FileNotFoundError(
            f"ResidualCorrectionMap {path!r} is set but cannot be read: {exc}"
        ) from exc

    key = (path, st.st_mtime_ns, st.st_size, str(dt), str(dev))
    hit = _RESID_CACHE.get(key)
    if hit is not None:
        return hit

    flat = np.fromfile(path, dtype=np.float64)
    if flat.size != NY * NZ:
        raise ValueError(
            f"ResidualCorrectionMap {path!r} has {flat.size} values but this "
            f"detector is {NZ} x {NY} = {NY * NZ}. The map is one float64 per "
            f"pixel; a mismatch means it belongs to a different detector."
        )
    # v1 C stores it row-major as map[z * Ny + y], i.e. shape [Nz, Ny], which is
    # also what residual_corr_lookup expects. Reshaping the other way round
    # transposes the correction and is not detectable downstream.
    m = torch.from_numpy(flat.reshape(NZ, NY)).to(dtype=dt, device=dev)
    _RESID_CACHE[key] = m
    return m


def _build_p_coeffs_from_spec(spec: IntegrationSpec) -> torch.Tensor:
    """Stack the 15 distortion tensors in canonical order.

    Order matches :data:`DISTORTION_NAMES` and the v2 forward model's
    expectation (``iso_R2, iso_R4, iso_R6, a1, phi1, a2, phi2, …, a6, phi6``).
    """
    return torch.stack([getattr(spec, n) for n in DISTORTION_NAMES])


def _panel_inputs_from_spec(spec: IntegrationSpec):
    """Per-panel corrections for this spec, or ``None`` if it is single-panel.

    Returns ``(layout, delta_yz, delta_theta, delta_lsd, delta_p2)`` ready to
    hand to :func:`midas_calibrate_v2.forward.geometry.pixel_to_REta`.

    Panels are generated and their shift file parsed by **v1**
    (``build_panels_from_params``), which is what the C ``DetectorMapper``
    mirrors — so panel numbering (``id = iy * n_z + iz``) and the
    ``PanelShiftsFile`` column order are shared rather than reimplemented.
    """
    ny = int(getattr(spec, "NPanelsY", 0) or 0)
    nz = int(getattr(spec, "NPanelsZ", 0) or 0)
    if ny <= 0 or nz <= 0:
        return None

    sy = int(getattr(spec, "PanelSizeY", 0) or 0)
    sz = int(getattr(spec, "PanelSizeZ", 0) or 0)
    gy = tuple(int(g) for g in (getattr(spec, "PanelGapsY", None) or ()))
    gz = tuple(int(g) for g in (getattr(spec, "PanelGapsZ", None) or ()))
    shifts = str(getattr(spec, "PanelShiftsFile", "") or "")
    try:
        stamp = os.path.getmtime(shifts) if shifts else 0.0
    except OSError:
        stamp = 0.0
    key = (ny, nz, sy, sz, gy, gz, os.path.abspath(shifts) if shifts else "",
           stamp, str(spec.dtype()), str(spec.device()))
    hit = _PANEL_CACHE.get(key)
    if hit is not None:
        return hit

    # Lazy import: v1 is a runtime dependency of the binning path already,
    # and importing it at module load would make a cycle through compat.
    from midas_integrate.detector_mapper import build_panels_from_params
    from midas_integrate.panel import generate_panels
    from ..compat.to_v1 import v1_params_from_spec

    v1p = v1_params_from_spec(spec)
    try:
        panels = build_panels_from_params(v1p)
    except OSError:
        # PanelShiftsFile is produced by CalibrantPanelShiftsOMP, so it is
        # legitimately absent before calibration has been run (the shipped
        # Example/Calibration parameters.txt names one). Fall back to the
        # layout with zero shifts -- but say so, because the alternative is
        # integrating with a panel calibration silently discarded.
        warnings.warn(
            f"PanelShiftsFile {shifts!r} could not be read; integrating with "
            f"the {ny}x{nz} panel layout but ZERO panel shifts. Any panel "
            f"calibration in that file is not being applied.",
            PanelShiftsMissingWarning, stacklevel=3,
        )
        panels = generate_panels(
            n_panels_y=ny, n_panels_z=nz,
            panel_size_y=sy, panel_size_z=sz,
            gaps_y=list(gy), gaps_z=list(gz),
        )
    if not panels:
        _PANEL_CACHE[key] = None
        return None

    dt, dev = spec.dtype(), spec.device()
    layout = PanelLayout.regular(ny, nz, sy, sz, gap_y=gy or 0, gap_z=gz or 0)
    # v1 puts the rotation centre at (min + max) / 2 with max *inclusive*, i.e.
    # half a pixel below PanelLayout.regular's y_start + sy/2. dTheta is small
    # so the difference is ~1e-4 px, but parity with the C is the whole point.
    cy = torch.tensor([p.centerY for p in panels], dtype=dt, device=dev)
    cz = torch.tensor([p.centerZ for p in panels], dtype=dt, device=dev)
    layout.panel_centers_y = cy.reshape(ny, nz)
    layout.panel_centers_z = cz.reshape(ny, nz)
    layout.panel_index_mask = layout.panel_index_mask.to(dev)

    delta_yz = torch.tensor([[p.dY, p.dZ] for p in panels], dtype=dt, device=dev)
    delta_theta = torch.tensor([p.dTheta for p in panels], dtype=dt, device=dev)
    delta_lsd = torch.tensor([p.dLsd for p in panels], dtype=dt, device=dev)
    delta_p2 = torch.tensor([p.dP2 for p in panels], dtype=dt, device=dev)

    out = (layout, delta_yz, delta_theta, delta_lsd, delta_p2)
    _PANEL_CACHE[key] = out
    return out


def pixel_to_REta_from_spec(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    spec: IntegrationSpec,
    *,
    panel_idx: Optional[torch.Tensor] = None,
):
    """Differentiable pixel → (R_px, η_deg) via v2's torch geometry.

    Applies the per-panel rigid shift and the per-panel Lsd / p2 offsets when
    the spec describes a multi-panel detector, matching the C
    ``DetectorMapper``. Without this a 48-panel Pilatus integrates with its
    panel calibration silently discarded.

    ``panel_idx`` overrides the per-point panel lookup. Leave it ``None`` (the
    default) and each point is assigned to the panel it rounds into, which is
    what every per-pixel-centre caller wants.

    Supply it when the points are not independent samples but the CORNERS of
    one shape that must move rigidly -- the polygon kernel's pixel quads, for
    instance. ``panel_idx_for_points`` rounds each point separately, so a pixel
    straddling a panel boundary would otherwise have its corners shifted by
    different panels and be torn in two. Pass the index looked up at the pixel
    CENTRE, broadcast over that pixel's corners.
    """
    dt, dev = spec.dtype(), spec.device()
    lattice = getattr(spec, "lattice", "cartesian")
    apothem = (spec.Apothem.to(dt) if lattice == "hex_offset_y" else None)
    orientation = (spec.LatticeOrientation.to(dt)
                   if lattice == "hex_offset_y" else None)
    panel_kw = {}
    panels = _panel_inputs_from_spec(spec)
    if panels is not None:
        layout, delta_yz, delta_theta, delta_lsd, delta_p2 = panels
        if panel_idx is None:
            panel_idx = panel_idx_for_points(layout, Y_pix, Z_pix)
        panel_kw = dict(
            panel_layout=layout,
            panel_idx=panel_idx,
            delta_yz=delta_yz,
            delta_theta=delta_theta,
            delta_lsd_panel=delta_lsd,
            delta_p2_panel=delta_p2,
            # PanelShiftsFile holds already-refined shifts, so every panel's
            # value must be applied as stored. fix_panel_id >= 0 would zero the
            # reference panel's shift -- that gauge fix belongs to calibration,
            # and the C DetectorMapper does not read FixPanelID at all.
            fix_panel_id=-1,
        )

    return _v2_pixel_to_REta(
        Y_pix, Z_pix,
        Lsd=spec.Lsd, BC_y=spec.BC_y, BC_z=spec.BC_z,
        tx=spec.tx, ty=spec.ty, tz=spec.tz,
        p_coeffs=_build_p_coeffs_from_spec(spec),
        parallax=spec.Parallax,
        pxY=torch.as_tensor(spec.pxY, dtype=dt, device=dev),
        pxZ=torch.as_tensor(spec.pxZ, dtype=dt, device=dev),
        rho_d=torch.as_tensor(spec.RhoD, dtype=dt, device=dev),
        lattice=lattice,
        apothem=apothem,
        orientation_deg=orientation,
        residual_corr_map=_load_residual_map(spec),
        **panel_kw,
    )


def eval_pixel_REta(spec: IntegrationSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-pixel (R_px, η_deg) for the full detector grid.

    Output shapes are ``(NrPixelsZ, NrPixelsY)`` to match v1's image
    convention (z-outer, y-inner). Both tensors carry gradient when any
    refinable spec field has ``requires_grad=True``.
    """
    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    dev, dt = spec.device(), spec.dtype()
    ys = torch.arange(NY, dtype=dt, device=dev)
    zs = torch.arange(NZ, dtype=dt, device=dev)
    Z, Y = torch.meshgrid(zs, ys, indexing="ij")    # shape (NZ, NY)
    out = pixel_to_REta_from_spec(Y, Z, spec)
    return out.R_px, out.eta_deg


__all__ = ["pixel_to_REta_from_spec", "eval_pixel_REta",
           "_build_p_coeffs_from_spec", "_load_residual_map"]
