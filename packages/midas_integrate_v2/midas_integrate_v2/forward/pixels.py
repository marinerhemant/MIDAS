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
from typing import Optional, Tuple

import numpy as np
import torch

from midas_calibrate_v2.forward.geometry import pixel_to_REta as _v2_pixel_to_REta

from ..spec import IntegrationSpec, DISTORTION_NAMES

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


def pixel_to_REta_from_spec(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    spec: IntegrationSpec,
):
    """Differentiable pixel → (R_px, η_deg) via v2's torch geometry."""
    dt, dev = spec.dtype(), spec.device()
    lattice = getattr(spec, "lattice", "cartesian")
    apothem = (spec.Apothem.to(dt) if lattice == "hex_offset_y" else None)
    orientation = (spec.LatticeOrientation.to(dt)
                   if lattice == "hex_offset_y" else None)
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
