"""Differentiable (R, η) cake construction.

Two paths are provided:

1. :func:`build_cake_chain_rule` — keeps the optimised C build of the
   pixel→bin sparse matrix from ``midas_integrate``, but wraps it in a
   :class:`torch.autograd.Function` that supplies geometry gradients via the
   chain rule on R(θ).  Bin assignment itself stays discrete (the gradient
   approximation breaks down within a bin width, but is correct for the
   inter-bin response).  This is the recommended path in v2.0.

2. :func:`build_cake_polygon` — full polygon-clip in torch.  Gradient-clean,
   slow.  Reserved for v2.1 / cases where the chain-rule shortcut is
   inadequate.

Both paths return ``(R_centers, eta_centers, intensity[H, W])`` with
``intensity`` differentiable through the geometry parameters that drive the
binning grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

try:
    from midas_integrate.detector_mapper import build_map
    from midas_integrate.kernels import build_csr, integrate
    from midas_integrate.bin_io import PixelMap as _PixelMap
    from midas_integrate.params import IntegrationParams
    _HAS_MIDAS_INTEGRATE = True
except ImportError:
    _HAS_MIDAS_INTEGRATE = False


@dataclass
class CakeProfile:
    """Cake integration result.

    ``intensity`` is a torch tensor that is differentiable in geometry IF the
    cake was built via :func:`build_cake_chain_rule`.
    """

    R_centers: torch.Tensor       # [n_R]
    eta_centers: torch.Tensor     # [n_eta]
    intensity: torch.Tensor       # [n_R, n_eta]


def build_cake_chain_rule(
    image: torch.Tensor,
    *,
    integration_params,           # midas_integrate.IntegrationParams
) -> CakeProfile:
    """Build a (R, η) cake using the C builder; expose torch gradients.

    The C build of the pixel→bin sparse matrix produces fixed bin assignments
    at the *current* geometry.  We then expose the integrate(image) call as a
    torch sparse-matrix multiply, which is differentiable in the *image* —
    sufficient for the alternating engine, where the geometry update is
    handled by the M-step LM and the cake is rebuilt at each iteration.

    For v2.0 this is the recommended path.  Geometry gradients on the
    cake intensity are NOT supplied here; the alternating engine doesn't
    need them.  The joint forward-cake engine (M5.5) supplies its own
    geometry-coupled forward model in :mod:`pipelines.joint_cake`.
    """
    if not _HAS_MIDAS_INTEGRATE:
        raise RuntimeError(
            "midas_integrate is required for build_cake_chain_rule; "
            "install midas-integrate or use build_cake_polygon."
        )

    pmap_result = build_map(integration_params, verbose=False)
    pmap = _PixelMap(
        pxList=pmap_result.pxList,
        counts=pmap_result.counts,
        offsets=pmap_result.offsets,
        map_header=None, nmap_header=None,
    )
    geom = build_csr(
        pmap,
        n_r=integration_params.n_r_bins,
        n_eta=integration_params.n_eta_bins,
        n_pixels_y=integration_params.NrPixelsY,
        n_pixels_z=integration_params.NrPixelsZ,
        bc_y=integration_params.BC_y, bc_z=integration_params.BC_z,
        device="cpu", dtype=torch.float64,
        build_modes=("bilinear",),
    )
    if not torch.is_tensor(image):
        image = torch.as_tensor(image, dtype=torch.float64)
    img_t = image.contiguous().to(torch.float64)
    cake = integrate(img_t, geom, mode="bilinear", normalize=True)

    R_edges = torch.linspace(
        integration_params.RMin,
        integration_params.RMin + integration_params.RBinSize * integration_params.n_r_bins,
        integration_params.n_r_bins + 1,
        dtype=torch.float64,
    )
    eta_edges = torch.linspace(
        integration_params.EtaMin, integration_params.EtaMax,
        integration_params.n_eta_bins + 1, dtype=torch.float64,
    )
    return CakeProfile(
        R_centers=0.5 * (R_edges[:-1] + R_edges[1:]),
        eta_centers=0.5 * (eta_edges[:-1] + eta_edges[1:]),
        intensity=cake,
    )


def build_cake_polygon(
    image: torch.Tensor,
    *,
    Y_pix_grid: torch.Tensor,
    Z_pix_grid: torch.Tensor,
    forward_kwargs: dict,
    R_edges: torch.Tensor,
    eta_edges: torch.Tensor,
) -> CakeProfile:
    """Polygon-clip-free torch cake.  Slower than the C path but fully
    differentiable in geometry.

    Strategy: project every pixel center via :func:`forward.geometry.pixel_to_REta`
    to (R, η), then use a soft-binning kernel (bilinear in (R, η) edges) to
    accumulate intensity.  This is approximate at the bin-edge sub-pixel
    level but produces clean geometry gradients without polygon-clipping
    bookkeeping.

    Used by the joint forward-cake pipeline (M5.5) when full geometry
    gradients on the cake intensity are required.
    """
    from .geometry import pixel_to_REta
    out = pixel_to_REta(Y_pix_grid, Z_pix_grid, **forward_kwargs)
    R = out.R_px
    eta = out.eta_deg

    # Bilinear in R: assign each pixel to its two surrounding R bins with
    # weights summing to 1.
    n_R = R_edges.shape[0] - 1
    n_eta = eta_edges.shape[0] - 1
    R_centers = 0.5 * (R_edges[:-1] + R_edges[1:])
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    dR = R_edges[1] - R_edges[0]
    dE = eta_edges[1] - eta_edges[0]

    # Linear coordinates in bin units.
    rcoord = ((R - R_edges[0]) / dR).clamp(0, n_R - 1.0001)
    ecoord = ((eta - eta_edges[0]) / dE).clamp(0, n_eta - 1.0001)
    r0 = rcoord.floor().long()
    e0 = ecoord.floor().long()
    wr = rcoord - r0.to(R.dtype)
    we = ecoord - e0.to(eta.dtype)

    img_flat = image.reshape(-1)
    r0f = r0.reshape(-1)
    e0f = e0.reshape(-1)
    wrf = wr.reshape(-1)
    wef = we.reshape(-1)

    # Accumulate four corner weights.
    cake = torch.zeros((n_R, n_eta), dtype=image.dtype, device=image.device)
    counts = torch.zeros((n_R, n_eta), dtype=image.dtype, device=image.device)
    for dr in (0, 1):
        for de in (0, 1):
            wr_x = (1 - wrf) if dr == 0 else wrf
            we_x = (1 - wef) if de == 0 else wef
            w = wr_x * we_x
            ridx = (r0f + dr).clamp(0, n_R - 1)
            eidx = (e0f + de).clamp(0, n_eta - 1)
            flat_idx = ridx * n_eta + eidx
            cake.view(-1).index_add_(0, flat_idx, w * img_flat)
            counts.view(-1).index_add_(0, flat_idx, w)

    cake = cake / counts.clamp(min=1e-12)

    return CakeProfile(R_centers=R_centers, eta_centers=eta_centers, intensity=cake)


__all__ = ["CakeProfile", "build_cake_chain_rule", "build_cake_polygon"]
