"""M-step: refine geometry from a list of fitted (R_fit, η_fit) data points.

Cost: residuals  r_i = √w_i · (1 − R_obs_i / R_pred_i)  where
    R_obs_i  = pixel_to_REta_torch(Y_pix_i, Z_pix_i, geometry).R         (in pixel coords)
    R_pred_i = Lsd · tan(2θ_ring_i) / px

with weights from SNR / radius / ring as configured.  We invoke
midas_peakfit.lm_solve_generic with an autograd Jacobian (since N_geom ≤ 23,
this is cheap).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from midas_peakfit import GenericLMConfig, lm_solve_generic, u_to_x

from .geometry_torch import predict_R_at_pixel, predict_two_theta_from_d
from .param_vector import (
    N_GEOM,
    bounds,
    collapse_unrefined_bounds,
    pack,
    refine_mask,
    unpack,
)
from .params import CalibrationParams
from .rings import RingTable


@dataclass
class FittedPoint:
    Y_pix: float
    Z_pix: float
    ring_idx: int      # index into RingTable
    snr: float = 1.0
    # Peak height over a LINEAR baseline across the radial window, divided by
    # the scatter at the window ends.  ``snr`` above is peak/mean inside the
    # window, which scores a ring sitting on a sloping background as strong;
    # this one does not, so it is what the per-ring quality filter aggregates.
    # Kept separate so ``SNRMin`` keeps its historical meaning.
    snr_baseline: float = 0.0


@dataclass
class RefineResult:
    params: CalibrationParams
    cost: float
    rc: int
    mean_strain_uE: float        # mean |1 - R_obs/R_pred|·1e6, in microstrain
    n_points_used: int


def _build_residual_function(
    pts_Y: torch.Tensor, pts_Z: torch.Tensor, pts_ring_idx: torch.Tensor,
    rt_d: torch.Tensor, rt_two_theta: torch.Tensor,
    weights: torch.Tensor, px: float, rho_d: float,
    refine_wavelength: bool,
):
    """Build a residual closure for lm_solve_generic.

    Returns a callable  residual_fn(u, lo, hi) -> r [B=1, M].
    """
    def _residual_fn(u, lo, hi):
        # u is [1, N_GEOM], single-problem batch.
        x = u_to_x(u, lo, hi).squeeze(0)   # [N_GEOM]
        R_obs = predict_R_at_pixel(pts_Y, pts_Z, x, px=px, rho_d=rho_d)
        if refine_wavelength:
            two_theta = predict_two_theta_from_d(rt_d[pts_ring_idx], x[21])
        else:
            two_theta = rt_two_theta[pts_ring_idx]
        R_pred = x[0] * torch.tan(two_theta * 0.017453292519943295) / px
        # Strain residual, weighted.
        r = (1.0 - R_obs / R_pred) * weights
        return r.unsqueeze(0)
    return _residual_fn


def refine_geometry(
    params: CalibrationParams,
    rt: RingTable,
    fits: list[FittedPoint],
    *,
    device: str = "cpu",
    dtype=torch.float64,
    max_iter: int = 200,
    huber_delta: float | None = None,
    verbose: bool = False,
) -> RefineResult:
    """One M-step pass: refine geometry from the supplied (Y_pix, Z_pix, ring) points.

    Returns updated params, final cost, return code, mean strain (μϵ).
    """
    if not fits:
        raise ValueError("no fitted points provided — E-step produced no usable data")

    # Pack initial geometry & bounds, freeze unrefined params.
    x0 = pack(params, dtype=dtype, device=device)
    lo, hi = bounds(params, dtype=dtype, device=device)
    mask = refine_mask(params).to(device)
    lo, hi = collapse_unrefined_bounds(lo, hi, x0, mask)

    pts_Y = torch.tensor([p.Y_pix for p in fits], dtype=dtype, device=device)
    pts_Z = torch.tensor([p.Z_pix for p in fits], dtype=dtype, device=device)
    pts_ring_idx = torch.tensor([p.ring_idx for p in fits], dtype=torch.long, device=device)

    rt_d = torch.tensor(rt.d_spacing, dtype=dtype, device=device)
    rt_tt = torch.tensor(rt.two_theta_deg, dtype=dtype, device=device)

    # Weights — basic SNR weighting; ring & radius weighting can be layered later.
    w = torch.ones(len(fits), dtype=dtype, device=device)
    if params.WeightBySNR:
        snr = torch.tensor([p.snr for p in fits], dtype=dtype, device=device)
        med = snr.median().clamp(min=1e-6)
        w = w * (snr / med).clamp(min=0.1, max=10.0)

    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    rho_d = params.RhoD if params.RhoD > 0 else params.MaxRingRad

    residual_fn = _build_residual_function(
        pts_Y, pts_Z, pts_ring_idx, rt_d, rt_tt,
        weights=w, px=px, rho_d=rho_d,
        refine_wavelength=params.Refine.get("Wavelength", False),
    )

    cfg = GenericLMConfig(
        max_iter=max_iter,
        ftol_rel=1e-9,
        xtol_rel=1e-9,
        huber_delta=huber_delta,
        verbose=verbose,
    )
    x_final, cost, rc = lm_solve_generic(
        x0.unsqueeze(0), lo.unsqueeze(0), hi.unsqueeze(0),
        residual_fn=residual_fn,
        config=cfg,
    )
    x_final = x_final.squeeze(0)
    new_params = unpack(x_final, params)

    # Compute mean strain at the final params for reporting.
    R_obs = predict_R_at_pixel(pts_Y, pts_Z, x_final, px=px, rho_d=rho_d)
    if new_params.Refine.get("Wavelength", False):
        two_theta = predict_two_theta_from_d(rt_d[pts_ring_idx], x_final[21])
    else:
        two_theta = rt_tt[pts_ring_idx]
    R_pred = x_final[0] * torch.tan(two_theta * 0.017453292519943295) / px
    strain = (1.0 - R_obs / R_pred).abs()
    mean_strain_uE = float(strain.mean()) * 1e6

    return RefineResult(
        params=new_params,
        cost=float(cost.item()),
        rc=int(rc.item()),
        mean_strain_uE=mean_strain_uE,
        n_points_used=len(fits),
    )
