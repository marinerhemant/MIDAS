"""Differentiable pixel → (R, η) for an :class:`IntegrationSpec`.

Thin shim over :func:`midas_calibrate_v2.forward.geometry.pixel_to_REta`
that takes the parameters from a v2 :class:`IntegrationSpec` instead of
loose tensors. Used by the soft-binning integrate path in Phase 2.

The full grid evaluator :func:`eval_pixel_REta` returns the per-pixel
``(R, η)`` for every detector pixel so the caller can soft-bin or
evaluate residual corrections in pixel space.
"""
from __future__ import annotations

from typing import Tuple

import torch

from midas_calibrate_v2.forward.geometry import pixel_to_REta as _v2_pixel_to_REta

from ..spec import IntegrationSpec, DISTORTION_NAMES


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
           "_build_p_coeffs_from_spec"]
