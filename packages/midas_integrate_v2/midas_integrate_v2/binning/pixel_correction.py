"""Per-pixel intensity corrections taken from the spec flags.

``IntegrationSpec`` carries ``SolidAngleCorrection``, ``PolarizationCorrection``,
``PolarizationFraction`` and ``PolarizationPlaneEtaDeg`` (read from a v1 paramstest by
``spec_from_v1_paramstest``). v1's mapper applies them; until this module existed the hard /
subpixel / polygon geometries and their variance versions did not read them at all, so
``PolarizationCorrection 1`` in a paramstest was applied by v1 and not applied by
``midas-integrate-v2`` (measured on a polarized, cos^3-weighted synthetic frame: 4-14 % in ring
areas, flags on == flags off to the last bit).

Convention, identical to :func:`midas_integrate_v2.corrections.integrate_with_corrections` and to
v1's ``corrected /= sa; corrected /= polFactor``:

    I_corrected(pixel) = I(pixel) / (SA(pixel) * POL(pixel))

and the per-bin normaliser is unchanged, so the output is the mean of the corrected pixels.
The factors are evaluated once per geometry, at the pixel centre, from the same (R, eta) grid the
binning uses (distortion, panel shifts and residual map included).

Precedence (same as ``integrate_with_corrections``): an explicit per-pixel ``correction`` passed to
a ``*_with_variance`` integrator replaces the spec-derived divisor, it is not stacked on it. That
keeps the hand-built ``correction = sa * pf`` recipe in ``manuals/pdf/README.md`` correct when the
spec also has the flags set.

To opt out for one geometry: ``geom.pixel_divisor = None``.

Opt-in distance normalisation (``spec.IntensityRefDistance_um`` = L_ref > 0, default 0 = off): the divisor is also multiplied by
(L_ref / Lsd)^2, so the output is I * (Lsd / L_ref)^2 / (SA * POL). With L_ref = 1e6 um this is GSAS-II's convention (it multiplies
by Lsd[m]^2): intensities from different sample-to-detector distances become comparable. It works with or without the SA / POL flags.

Limitation: the solid-angle factor is the cartesian tilt-aware one; for ``lattice='hex_offset_y'``
it is not a hex-lattice formula (same as ``integrate_with_corrections``).
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch

from ..forward import pixel_to_REta_from_spec
from ..spec import IntegrationSpec

#: Floors, copied from ``integrate_with_corrections`` so the two paths agree.
_SA_FLOOR = 1e-12
_POL_FLOOR = 1e-6


def mean_pixel_um(spec: IntegrationSpec, *, dtype, device) -> torch.Tensor:
    """The pixel pitch (um) in which ``R_px`` from ``pixel_to_REta`` is expressed: ``px_mean = (pxY_eff + pxZ_eff) / 2``
    (forward/geometry.py). Cartesian: the spec's pxY, pxZ; hex_offset_y: derived from the apothem. pxY/pxZ are plain floats on the spec,
    so no gradient flows through this value."""
    return torch.as_tensor(spec.mean_pixel_um(), dtype=dtype, device=device)


def distance_ref_um(spec: IntegrationSpec) -> float:
    """``spec.IntensityRefDistance_um`` (0 = distance normalisation off)."""
    return float(getattr(spec, "IntensityRefDistance_um", 0.0) or 0.0)


def spec_corrections_enabled(spec: IntegrationSpec) -> bool:
    return bool(int(getattr(spec, "SolidAngleCorrection", 0) or 0)
                or int(getattr(spec, "PolarizationCorrection", 0) or 0)
                or distance_ref_um(spec) > 0.0)


def pixel_divisor_from_spec(spec: IntegrationSpec, *, differentiable: bool = False) -> Optional[torch.Tensor]:
    """``(n_pixels_z * n_pixels_y,)`` float64 divisor, or ``None`` when both flags are off.

    ``differentiable=True`` keeps the autograd graph to the spec tensors (used by ``SoftBinGeometry``, the refinement path, so the
    correction's dependence on the geometry is not silently dropped); the hard-binning geometries use a detached constant."""
    if not spec_corrections_enabled(spec):
        return None
    spec.validate()          # e.g. resolves a pixel-valued RhoD, which changes R; geometry.from_spec does the same first
    from midas_calibrate_v2.forward.geometry import build_tilt_matrix
    from ..corrections.intensity import (
        POL_PLANE_HORIZONTAL_ETA_DEG, PolarizationCorrection, SolidAngleCorrection,
    )

    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    dt, dev = spec.dtype(), spec.device()
    with (contextlib.nullcontext() if differentiable else torch.no_grad()):
        ys = torch.arange(NY, dtype=dt, device=dev)
        zs = torch.arange(NZ, dtype=dt, device=dev)
        Z, Y = torch.meshgrid(zs, ys, indexing="ij")
        pix = pixel_to_REta_from_spec(Y, Z, spec)
        R_flat = pix.R_px.reshape(-1)
        Eta_flat = pix.eta_deg.reshape(-1)
        div = torch.ones(NY * NZ, dtype=R_flat.dtype, device=R_flat.device)
        _pyz = spec.effective_pxYZ()          # hex: apothem-derived; pxZ unset: square
        pxY = torch.as_tensor(_pyz[0], dtype=R_flat.dtype, device=R_flat.device)
        pxZ = torch.as_tensor(_pyz[1], dtype=R_flat.dtype, device=R_flat.device)
        if int(getattr(spec, "SolidAngleCorrection", 0) or 0):
            sa = SolidAngleCorrection()(
                Y.reshape(-1), Z.reshape(-1),
                Ycen=spec.BC_y, Zcen=spec.BC_z,
                TRs=build_tilt_matrix(spec.tx, spec.ty, spec.tz),
                Lsd=spec.Lsd, pxY=pxY, pxZ=pxZ,
            )
            div = div * sa.clamp(min=_SA_FLOOR)
        if int(getattr(spec, "PolarizationCorrection", 0) or 0):
            pf = PolarizationCorrection(
                pol_fraction=float(getattr(spec, "PolarizationFraction", 0.99)),
                pol_plane_eta_deg=float(getattr(spec, "PolarizationPlaneEtaDeg",
                                                POL_PLANE_HORIZONTAL_ETA_DEG)),
            )(R_flat, Eta_flat, Lsd=spec.Lsd, px=mean_pixel_um(spec, dtype=R_flat.dtype, device=R_flat.device))   # R_px is in MEAN-pixel units
            div = div * pf.clamp(min=_POL_FLOOR)
        ref = distance_ref_um(spec)
        if ref > 0.0:
            # I_out = I * (Lsd / L_ref)^2  <=>  divisor *= (L_ref / Lsd)^2 ; Lsd keeps its graph on the soft path
            div = div * (ref / torch.as_tensor(spec.Lsd, dtype=div.dtype, device=div.device)) ** 2
    div = div.to(torch.float64)
    return div if differentiable else div.detach()


def apply_pixel_divisor(image: torch.Tensor, geom) -> torch.Tensor:
    """Divide ``image`` (``(..., NZ, NY)``, already in the post-ImTransOpt frame) by the
    geometry's spec-derived divisor. No-op when the geometry has none."""
    d = getattr(geom, "pixel_divisor", None)
    if d is None:
        return image
    d = d.reshape(geom.n_pixels_z, geom.n_pixels_y).to(device=image.device)
    return image / d.to(image.dtype if image.dtype.is_floating_point else torch.float64)


__all__ = ["pixel_divisor_from_spec", "apply_pixel_divisor", "spec_corrections_enabled", "mean_pixel_um", "distance_ref_um"]
