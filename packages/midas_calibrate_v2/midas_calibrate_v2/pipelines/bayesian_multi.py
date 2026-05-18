"""Bayesian multi-image calibration — MAP + Laplace covariance.

Mirrors ``pipelines/bayesian.py`` for the multi-image / joint case. Runs the
existing ``autocalibrate_multi`` LM, then computes a Gauss-Newton (Fisher)
covariance at the converged MAP using the same per-image pseudo-strain
residual stack the LM saw. The covariance is reported in bounded
``x``-space (same convention as single-image ``laplace_at_map``).

The Fisher path is used by default (J·diag(1/σ²)·J via ``jacfwd``)
because it is ~10–50× faster than the full ``hessian`` for the typical
multi-image parameter count (~20–30 refined params) and gives the same
leading-order behaviour when residuals are close to Gaussian. Set
``method="hessian"`` to force the full path for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_peakfit.laplace import LaplaceResult, report_laplace
from midas_peakfit.reparam import x_to_u, u_to_x

from ..forward.panels import PanelLayout
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.pack import (
    pack_multi, refined_indices, refined_bounds, unpack_spec,
)
from ..parameters.spec import CalibrationSpec, MultiImageSpec
from ._common import run_estep_v1
from .multi import MultiResult, autocalibrate_multi, build_multi_spec


@dataclass
class MultiBayesianResult:
    multi_result: MultiResult
    laplace: LaplaceResult


def _build_multi_indices(multi_spec: MultiImageSpec, info,
                          *, dtype, device):
    """Mirror of ``multi._build_multi_indices`` — returns (refined_idx, lo, hi)
    on ``device``. Inlined here to avoid coupling to the private helper."""
    s_spec = CalibrationSpec(parameters=multi_spec.shared)
    lo_s, hi_s = refined_bounds(s_spec, info.shared_info, dtype=dtype, device=device)
    lo_pieces = [lo_s]
    hi_pieces = [hi_s]
    ref_idx_pieces = [refined_indices(info.shared_info)]
    for img_dict, img_info in zip(multi_spec.per_image, info.per_image_info):
        i_spec = CalibrationSpec(parameters=img_dict)
        lo_i, hi_i = refined_bounds(i_spec, img_info, dtype=dtype, device=device)
        lo_pieces.append(lo_i)
        hi_pieces.append(hi_i)
        ref_idx_pieces.append(refined_indices(img_info))
    refined_idx = torch.cat(ref_idx_pieces).to(device=device)
    return refined_idx, torch.cat(lo_pieces), torch.cat(hi_pieces)


def _flat_refined_names(multi_spec: MultiImageSpec, info) -> List[str]:
    """Names matching the flat refined-parameter vector — used to label
    ``LaplaceResult.refined_names`` for the multi-image case.

    Order matches ``_build_multi_indices``: shared block first, then per-image
    blocks in image order. Per-image parameter names are suffixed with the
    image index so the caller can tell ``Lsd_img0`` from ``Lsd_img1``.
    """
    names: List[str] = []
    for n, r in zip(info.shared_info.names, info.shared_info.refined):
        if r:
            names.append(n)
    for k, img_info in enumerate(info.per_image_info):
        for n, r in zip(img_info.names, img_info.refined):
            if r:
                names.append(f"{n}_img{k}")
    return names


def sigma_cc_at_multi_map(
    multi_spec: MultiImageSpec,
    v1_per_image: List[V1Params],
    images: List[np.ndarray],
    darks: Optional[List[Optional[np.ndarray]]] = None,
    *,
    panel_layout: Optional[PanelLayout] = None,
    method: str = "fisher",
    laplace_ridge: float = 1e-9,
    sigma_r: Optional[float] = None,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> LaplaceResult:
    """Compute Laplace covariance at a multi-image MAP, *skipping* the LM step.

    Use this when the MAP geometry already lives in ``multi_spec`` (and in
    the per-image ``v1_per_image``) — e.g. loaded from a previously-saved
    ``autocalibrate_multi`` result. Re-runs the E-step once at the current
    geometry, then computes the Fisher (default) or Hessian covariance.
    """
    if method not in ("fisher", "hessian"):
        raise ValueError(f"method must be 'fisher' or 'hessian'; got {method!r}")
    if darks is None:
        darks = [None] * len(v1_per_image)

    fits_per_image = [
        run_estep_v1(v1, img, dark=drk, dtype=dtype, device=device)
        for v1, img, drk in zip(v1_per_image, images, darks)
    ]

    x_full, info = pack_multi(multi_spec, dtype=dtype, device=device)
    refined_idx, lo, hi = _build_multi_indices(
        multi_spec, info, dtype=dtype, device=device,
    )

    s_spec = CalibrationSpec(parameters=multi_spec.shared)
    per_specs = [CalibrationSpec(parameters=d) for d in multi_spec.per_image]

    x_ref_map = x_full.index_select(0, refined_idx)
    u_map = x_to_u(x_ref_map, lo, hi)

    def r_of_u(u: torch.Tensor) -> torch.Tensor:
        x_ref = u_to_x(u, lo, hi)
        x_full_now = x_full.clone()
        x_full_now[refined_idx] = x_ref
        shared_dict = unpack_spec(x_full_now, info.shared_info, s_spec)
        per_dicts = [unpack_spec(x_full_now, img_info, ps)
                      for img_info, ps in zip(info.per_image_info, per_specs)]
        r_pieces: List[torch.Tensor] = []
        for fits, per_d in zip(fits_per_image, per_dicts):
            merged = {**shared_dict, **per_d}
            r = pseudo_strain_residual(
                fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, merged,
                rho_d=fits.rho_d, weights=fits.weights,
                panel_layout=panel_layout, panel_idx=fits.panel_idx,
            )
            r_pieces.append(r)
        return torch.cat(r_pieces)

    if method == "fisher":
        from torch.func import jacfwd, jacrev
        with torch.no_grad():
            r_map = r_of_u(u_map.detach().clone())
        if sigma_r is None:
            sigma_r = float(torch.sqrt((r_map ** 2).mean()).item())
            if verbose:
                print(f"[bayesian-multi] empirical σ_r at MAP = {sigma_r:.3e}",
                      flush=True)
        n_params = int(u_map.numel())
        m_resid = int(r_map.numel())
        if n_params < m_resid:
            if verbose:
                print(f"[bayesian-multi] jacfwd over {n_params} params "
                      f"× {m_resid} residual rows...", flush=True)
            J = jacfwd(r_of_u)(u_map.detach().clone())
        else:
            J = jacrev(r_of_u)(u_map.detach().clone())
        F_u = J.transpose(0, 1) @ J / (sigma_r ** 2)
    else:
        def nll_of_u(u: torch.Tensor) -> torch.Tensor:
            r = r_of_u(u)
            return 0.5 * (r * r).sum()
        H = torch.autograd.functional.hessian(nll_of_u, u_map.detach().clone())
        F_u = 0.5 * (H + H.transpose(-1, -2))

    eye = torch.eye(F_u.shape[0], dtype=F_u.dtype, device=F_u.device)
    F_reg = F_u + laplace_ridge * eye
    try:
        cov_u = torch.linalg.inv(F_reg)
    except torch._C._LinAlgError:
        cov_u = torch.linalg.pinv(F_reg)

    s = torch.sigmoid(u_map)
    dxdu = (hi - lo) * s * (1.0 - s)
    cov_x = cov_u * dxdu.unsqueeze(0) * dxdu.unsqueeze(1)
    sigma = torch.sqrt(torch.diag(cov_x).clamp(min=0.0))

    refined_names = _flat_refined_names(multi_spec, info)
    return LaplaceResult(
        map_unpacked={},
        refined_names=refined_names,
        refined_offsets=list(range(len(refined_names))),
        refined_sizes=[1] * len(refined_names),
        map_refined=x_ref_map.detach(),
        cov=cov_x,
        sigma_per_dim=sigma,
    )


def autocalibrate_multi_bayesian(
    v1_per_image: List[V1Params],
    images: List[np.ndarray],
    darks: Optional[List[Optional[np.ndarray]]] = None,
    *,
    multi_spec: Optional[MultiImageSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter_map: int = 5,
    lm_max_iter: int = 200,
    method: str = "fisher",
    laplace_ridge: float = 1e-9,
    sigma_r: Optional[float] = None,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> MultiBayesianResult:
    """Joint multi-image calibration with Laplace covariance at MAP."""
    if multi_spec is None:
        multi_spec = build_multi_spec(v1_per_image)

    # Step 1: MAP via the existing LM driver.
    map_result = autocalibrate_multi(
        v1_per_image=v1_per_image,
        images=images, darks=darks,
        multi_spec=multi_spec, panel_layout=panel_layout,
        n_iter=n_iter_map, lm_max_iter=lm_max_iter,
        dtype=dtype, device=device, verbose=verbose,
    )

    # Step 2: covariance at MAP (skip-MAP helper does the E-step + Fisher).
    laplace = sigma_cc_at_multi_map(
        multi_spec=multi_spec,
        v1_per_image=v1_per_image,
        images=images, darks=darks,
        panel_layout=panel_layout,
        method=method, laplace_ridge=laplace_ridge, sigma_r=sigma_r,
        dtype=dtype, device=device, verbose=verbose,
    )

    if verbose:
        print(report_laplace(laplace))

    return MultiBayesianResult(multi_result=map_result, laplace=laplace)


__all__ = [
    "MultiBayesianResult",
    "autocalibrate_multi_bayesian",
    "sigma_cc_at_multi_map",
]
