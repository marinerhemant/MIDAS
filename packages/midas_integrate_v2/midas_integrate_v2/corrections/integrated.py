"""End-to-end differentiable integration with corrections wired in.

:func:`integrate_with_corrections` is the joint-refinement entry point:
given an :class:`IntegrationSpec` (geometry), an image, and any of
``residual`` (a thin-plate spline ``nn.Module``) or ``per_ring_offsets``
(δr_k ``nn.Module``), it produces the 2D integrated array with gradient
flowing back to every refinable parameter — geometry, distortion,
spline weights, and per-ring offsets.

Pipeline order (matches v1's apply order plus the v2 additions):

    R, η = pixel_to_REta(Y, Z; spec)              # geometry + distortion
    R   += residual(Y, Z)                         # Stage-4 spline
    R   += per_ring_offsets(R, ring_centres)      # F2 fix
    profile = soft_bin(image; R, η, RMin, RBin, EtaMin, EtaBin)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..binning.trans_opt import apply_trans_opt_forward, needs_trans_opt
from ..diff.soft_bin import soft_bin_indices_weights
from ..forward import eval_pixel_REta, pixel_to_REta_from_spec
from ..spec import IntegrationSpec


def integrate_with_corrections(
    image: torch.Tensor,
    spec: IntegrationSpec,
    *,
    residual: Optional[nn.Module] = None,
    per_ring_offsets: Optional[nn.Module] = None,
    ring_R_centres_px: Optional[torch.Tensor] = None,
    capture_radius_px: Optional[float] = None,
    polarization: Optional[nn.Module] = None,
    solid_angle: Optional[nn.Module] = None,
    apply_trans_opt: bool = True,
    learnable_mask: Optional[nn.Module] = None,
    empty_subtraction: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Differentiable integrate with optional Stage-4 spline + δr_k.

    Parameters
    ----------
    image :
        Detector image, shape ``(NrPixelsZ, NrPixelsY)``.
    spec :
        :class:`IntegrationSpec` carrying geometry / distortion / binning.
    residual :
        Optional :class:`RBFResidualCorrection` evaluating ``ΔR(Y, Z)``;
        applied as ``R += ΔR(Y, Z)`` (units of pixels).
    per_ring_offsets :
        Optional :class:`PerRingOffsets` providing δr_k. Requires
        ``ring_R_centres_px``.
    ring_R_centres_px :
        ``(n_rings,)`` tensor of ideal ring radii in pixels. Used to
        assign each pixel to its nearest ring for the δr_k lookup.
    capture_radius_px :
        Pixels farther than this from any ring centre receive zero δr_k
        contribution. Defaults to no clipping.

    Returns
    -------
    Tensor of shape ``(n_eta, n_r)`` with autograd hooked up to
    ``spec`` tensors, ``residual`` parameters, and ``per_ring_offsets``
    parameters that are ``requires_grad=True``.
    """
    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    dt, dev = spec.dtype(), spec.device()

    if apply_trans_opt and spec.TransOpt and needs_trans_opt(spec.TransOpt):
        image = apply_trans_opt_forward(
            image, spec.TransOpt, NrPixelsY=NY, NrPixelsZ=NZ,
        )

    # Empty-cell / dark baseline subtraction. Runs before the learnable
    # mask so the mask sees the post-subtraction image (its sparsity
    # prior should reward "the empty cell is clean", not raw counts).
    if empty_subtraction is not None:
        image = empty_subtraction(image)

    # Learnable mask: differentiable per-pixel weight in (0, 1).
    # Multiplying the image by this routes gradient back to the mask
    # parameters, letting the optimiser auto-detect bad pixels.
    if learnable_mask is not None:
        image = image * learnable_mask()

    ys = torch.arange(NY, dtype=dt, device=dev)
    zs = torch.arange(NZ, dtype=dt, device=dev)
    Z, Y = torch.meshgrid(zs, ys, indexing="ij")
    out = pixel_to_REta_from_spec(Y, Z, spec)
    R = out.R_px
    Eta = out.eta_deg

    if residual is not None:
        R = R + residual(Y, Z)

    if per_ring_offsets is not None:
        if ring_R_centres_px is None:
            raise ValueError(
                "ring_R_centres_px required when per_ring_offsets is set"
            )
        R = per_ring_offsets(R, ring_R_centres_px,
                              capture_radius_px=capture_radius_px)

    R_flat = R.reshape(-1)
    Eta_flat = Eta.reshape(-1)
    img_flat = image.to(dtype=R_flat.dtype).reshape(-1)

    # Per-pixel multiplicative corrections (divide raw counts by the
    # correction factor — same convention as v1 ``corrected /= sa`` etc).
    if solid_angle is not None or polarization is not None:
        pxY_t = torch.as_tensor(spec.pxY, dtype=R_flat.dtype,
                                 device=R_flat.device)
        pxZ_t = torch.as_tensor(spec.pxZ, dtype=R_flat.dtype,
                                 device=R_flat.device)
    if solid_angle is not None:
        # Exact tilt-aware solid angle: needs per-pixel (Y, Z) and the
        # tilt matrix, not just radial R. Build the tilt matrix from
        # the spec's tilts each call (small fixed cost).
        from ..forward.pixels import _build_p_coeffs_from_spec  # noqa: F401
        from midas_calibrate_v2.forward.geometry import build_tilt_matrix
        TRs = build_tilt_matrix(spec.tx, spec.ty, spec.tz)
        Y_flat = Y.reshape(-1)
        Z_flat = Z.reshape(-1)
        sa = solid_angle(
            Y_flat, Z_flat,
            Ycen=spec.BC_y, Zcen=spec.BC_z, TRs=TRs,
            Lsd=spec.Lsd, pxY=pxY_t, pxZ=pxZ_t,
        )
        img_flat = img_flat / sa.clamp(min=1e-12)
    if polarization is not None:
        pf = polarization(R_flat, Eta_flat, Lsd=spec.Lsd, px=pxY_t)
        img_flat = img_flat / pf.clamp(min=1e-6)

    n_r, n_eta = spec.n_r_bins, spec.n_eta_bins
    rb0, rb1, rw0, rw1 = soft_bin_indices_weights(
        R_flat, R_min=spec.RMin, R_bin_size=spec.RBinSize, n_r=n_r,
    )
    eb0, eb1, ew0, ew1 = soft_bin_indices_weights(
        Eta_flat, R_min=spec.EtaMin, R_bin_size=spec.EtaBinSize, n_r=n_eta,
    )
    flat = torch.zeros(n_eta * n_r, dtype=img_flat.dtype, device=img_flat.device)
    for ei, ew in ((eb0, ew0), (eb1, ew1)):
        for ri, rw in ((rb0, rw0), (rb1, rw1)):
            idx = ei * n_r + ri
            flat = flat.index_add(0, idx, img_flat * ew * rw)
    return flat.reshape(n_eta, n_r)


__all__ = ["integrate_with_corrections"]
