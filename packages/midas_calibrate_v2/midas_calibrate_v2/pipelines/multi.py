"""Multi-image / multi-distance joint calibration.

Each image has per-image parameters (Lsd, BC, tilts) plus a shared block
(distortion harmonics, panels, pxY, pxZ).  Joint loss = Σ pseudo-strain.

Wright2022's grid panel calibration falls out as a special case (multiple
beam positions, shared panel shifts).  This is the milestone that
operationally unlocks pxY / pxZ / d-spacing fitting because the
rank-deficiency from a single image disappears when multiple geometries
share the intrinsic parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_peakfit import GenericLMConfig
from midas_peakfit.reparam import x_to_u, u_to_x
from midas_peakfit import lm_solve_generic

from ..compat.from_v1 import spec_from_v1_params
from ..forward.panels import PanelLayout
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.parameter import Parameter
from ..parameters.pack import (
    pack_multi, refined_indices, refined_bounds,
    unpack_spec, MultiPackInfo, pack_spec,
)
from ..parameters.spec import CalibrationSpec, MultiImageSpec
from ._common import FittedDataset, run_estep_v1


def build_multi_spec(
    v1_per_image: List[V1Params],
    *,
    shared_names: Optional[List[str]] = None,
) -> MultiImageSpec:
    """Construct a MultiImageSpec from per-image v1 params.

    Default shared block: pxY, pxZ, RhoD, all v2 distortion params, panel_*.
    Per-image: Lsd, BC_y, BC_z, ty, tz, Wavelength, Parallax, tx.
    """
    if shared_names is None:
        from ..forward.distortion import P_COEF_NAMES
        shared_names = (["pxY", "pxZ", "RhoD"]
                        + list(P_COEF_NAMES)
                        + ["panel_delta_yz", "panel_delta_theta",
                           "panel_delta_lsd", "panel_delta_p2"])

    specs = [spec_from_v1_params(p) for p in v1_per_image]
    shared_names = [n for n in shared_names if n in specs[0].parameters]
    return MultiImageSpec.from_calibration_specs(specs, shared_names)


@dataclass
class MultiResult:
    multi_spec: MultiImageSpec
    shared_unpacked: Dict[str, torch.Tensor]
    per_image_unpacked: List[Dict[str, torch.Tensor]]
    cost: float
    rc: int
    # Empirical residual-correction map built post-MAP (port of v1 C
    # dg_residual_corr_lookup).  None when ``build_residual_corr=False``.
    # Shape: ``[NrPixelsZ, NrPixelsY]`` float64, ΔR in pixels.
    residual_corr_map: Optional[torch.Tensor] = None
    # Mean weighted strain (µε) re-evaluated at MAP after the residual
    # map is applied — the honest "post-correction" number.  None when
    # ``build_residual_corr=False``.
    post_residual_strain_uE: Optional[List[float]] = None


def _build_multi_indices(multi_spec: MultiImageSpec, info: MultiPackInfo,
                          dtype, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (refined_idx, lo, hi) over the multi-image flat tensor.

    ``refined_indices`` returns CPU tensors by default; move the concatenated
    result onto ``device`` so it lines up with ``x_full`` (which ``pack_multi``
    already places on ``device``). Without this, ``index_select`` raises a
    cross-device error the moment the caller is on a GPU.
    """
    s_spec = CalibrationSpec(parameters=multi_spec.shared)
    lo_s, hi_s = refined_bounds(s_spec, info.shared_info, dtype=dtype, device=device)
    lo_pieces = [lo_s]; hi_pieces = [hi_s]
    ref_idx_pieces = [refined_indices(info.shared_info)]
    for img_dict, img_info in zip(multi_spec.per_image, info.per_image_info):
        i_spec = CalibrationSpec(parameters=img_dict)
        lo_i, hi_i = refined_bounds(i_spec, img_info, dtype=dtype, device=device)
        lo_pieces.append(lo_i); hi_pieces.append(hi_i)
        ref_idx_pieces.append(refined_indices(img_info))
    refined_idx = torch.cat(ref_idx_pieces).to(device=device)
    return refined_idx, torch.cat(lo_pieces), torch.cat(hi_pieces)


def autocalibrate_multi(
    v1_per_image: List[V1Params],
    images: List[np.ndarray],
    darks: Optional[List[Optional[np.ndarray]]] = None,
    *,
    multi_spec: Optional[MultiImageSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter: int = 5,
    lm_max_iter: int = 200,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
    build_residual_corr: bool = True,
    residual_corr_outlier_pct: float = 90.0,
    residual_corr_path: Optional[str] = None,
) -> MultiResult:
    """Joint calibration over multiple images.

    After LM/EM convergence, optionally builds an empirical residual
    correction map from per-fit ``ΔR = R_forward - R_ideal`` and stores it on
    the result.  The map absorbs systematic deviations the harmonic
    distortion polynomial cannot capture (port of v1 C
    ``dg_residual_corr_lookup``) and brings v2 to v1-AutoCal accuracy.  When
    ``residual_corr_path`` is set, the map is also persisted as a
    v1-compatible binary readable by :mod:`midas_integrate` and
    ``CalibrantIntegratorOMP``.
    """
    # ---- Sanity-check RhoD against detector geometry on every input image.
    # RhoD is stored in pixels in CalibrationParams; we validate the µm
    # equivalent (× pxY) so wrong-unit mistakes raise here instead of
    # producing silent garbage in the distortion stage.
    from ..forward.sanity import check_rho_d_um
    for i, v1 in enumerate(v1_per_image):
        rho_d_px = v1.RhoD if v1.RhoD > 0 else v1.MaxRingRad
        check_rho_d_um(
            RhoD_um=float(rho_d_px) * float(v1.pxY),
            NrPixelsY=int(v1.NrPixelsY), NrPixelsZ=int(v1.NrPixelsZ),
            BC_y=float(v1.BC_y), BC_z=float(v1.BC_z),
            pxY=float(v1.pxY), pxZ=float(v1.pxZ if v1.pxZ > 0 else v1.pxY),
            strict=True,
        )
    n_imgs = len(v1_per_image)
    if len(images) != n_imgs:
        raise ValueError("len(images) must match len(v1_per_image)")
    if darks is None:
        darks = [None] * n_imgs

    if multi_spec is None:
        multi_spec = build_multi_spec(v1_per_image)

    cost_final = float("inf")
    rc_final = 1
    shared_dict_final: Dict[str, torch.Tensor] = {}
    per_dicts_final: List[Dict[str, torch.Tensor]] = []

    for it in range(n_iter):
        # E-step per image at current geometry.
        fits_per_image: List[FittedDataset] = []
        for v1, img, drk in zip(v1_per_image, images, darks):
            fits_per_image.append(
                run_estep_v1(v1, img, dark=drk, dtype=dtype, device=device)
            )

        x_full, info = pack_multi(multi_spec, dtype=dtype, device=device)
        refined_idx, lo, hi = _build_multi_indices(multi_spec, info, dtype, device)
        x_ref = x_full.index_select(0, refined_idx)

        s_spec = CalibrationSpec(parameters=multi_spec.shared)
        per_specs = [CalibrationSpec(parameters=d) for d in multi_spec.per_image]

        # Σ panel = 0 zero-sum constraint (Wright 2022 §3.2): if the
        # multi_spec carries the flag (set via add_panel_zero_sum_constraint
        # on the shared block), append the constraint residual exactly once
        # to the joint residual (panels are shared, so one set of zero-sum
        # rows applies to all images).
        _zs_active = bool(getattr(multi_spec, "zero_sum_panels", False))
        _zs_lambda = float(getattr(multi_spec, "zero_sum_lambda", 1e6))
        if _zs_active:
            from ..loss.constraints import zero_sum_residual

        def residual_fn(u, lo_, hi_):
            x_ref_now = u_to_x(u, lo_, hi_).squeeze(0)
            x_full_now = x_full.clone()
            x_full_now[refined_idx] = x_ref_now
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
            if _zs_active:
                zs = zero_sum_residual(shared_dict, lambda_zs=_zs_lambda)
                if zs.numel() > 0:
                    r_pieces.append(zs)
            return torch.cat(r_pieces).unsqueeze(0)

        x_final, cost, rc = lm_solve_generic(
            x_ref.unsqueeze(0), lo.unsqueeze(0), hi.unsqueeze(0),
            residual_fn=residual_fn,
            config=GenericLMConfig(max_iter=lm_max_iter, ftol_rel=1e-9, xtol_rel=1e-9),
        )

        with torch.no_grad():
            x_ref_final = x_final.squeeze(0)
            x_full_final = x_full.clone()
            x_full_final[refined_idx] = x_ref_final
            shared_dict_final = unpack_spec(x_full_final, info.shared_info, s_spec)
            per_dicts_final = [unpack_spec(x_full_final, img_info, ps)
                                for img_info, ps in zip(info.per_image_info, per_specs)]

        # Push back into spec inits and v1 params for next E-step.
        for name, val in shared_dict_final.items():
            init = val.detach().cpu() if val.numel() > 1 else float(val.detach())
            multi_spec.shared[name].init = init
            for v1 in v1_per_image:
                if hasattr(v1, name) and val.numel() == 1:
                    try:
                        cur = getattr(v1, name)
                        setattr(v1, name, type(cur)(float(val.detach())))
                    except Exception:
                        pass
        for img_idx, per_d in enumerate(per_dicts_final):
            for name, val in per_d.items():
                multi_spec.per_image[img_idx][name].init = (
                    val.detach().cpu() if val.numel() > 1 else float(val.detach())
                )
                v1 = v1_per_image[img_idx]
                if hasattr(v1, name) and val.numel() == 1:
                    try:
                        cur = getattr(v1, name)
                        setattr(v1, name, type(cur)(float(val.detach())))
                    except Exception:
                        pass

        cost_final = float(cost.item())
        rc_final = int(rc.item())
        if verbose:
            # Honest strain at MAP: rebuild E-step at post-LM geometry. The
            # in-loop residual_fn still holds pre-LM fits_per_image, so
            # evaluating it at x_final reports the drift between pre-LM peak
            # positions and post-LM forward prediction — not fit quality.
            # v1_per_image has already been pushed to x_final above.
            with torch.no_grad():
                fits_at_map = [run_estep_v1(v1, img, dark=drk,
                                             dtype=dtype, device=device)
                                for v1, img, drk in zip(v1_per_image, images, darks)]
                r_pieces_map = []
                for fits, per_d in zip(fits_at_map, per_dicts_final):
                    merged = {**shared_dict_final, **per_d}
                    r = pseudo_strain_residual(
                        fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, merged,
                        rho_d=fits.rho_d, weights=fits.weights,
                        panel_layout=panel_layout, panel_idx=fits.panel_idx,
                    )
                    r_pieces_map.append(r)
                mean_uE = float(torch.cat(r_pieces_map).abs().mean()) * 1e6
            print(f"[multi iter {it}] cost={cost_final:.6e}  rc={rc_final}  "
                  f"strain={mean_uE:8.1f}μϵ across {n_imgs} images")

    # ---- Post-MAP empirical residual-correction map (v1 parity stage) -----
    # After LM/EM converges, fit a smooth ΔR(Y, Z) spline to the per-fit
    # residuals (R_forward - R_ideal) and store it as a detector-resolution
    # grid.  Subsequent forward calls add ΔR via differentiable bilinear
    # lookup; this is the v2 port of v1 C ``dg_residual_corr_lookup`` and
    # closes the ~50–100 µε per-pixel gap between v2 and AutoCal/v1.
    residual_map: Optional[torch.Tensor] = None
    post_strain: Optional[List[float]] = None
    if build_residual_corr:
        from ..forward.residual_corr import (
            build_residual_corr_map, save_residual_corr_bin,
        )
        from ..forward.bragg import R_ideal_px

        # Aggregate non-outlier (Y, Z, ΔR_µm) across all images so the map
        # is shared (matches the multi-distance shared-detector setup).
        Y_all, Z_all, dR_all = [], [], []
        post_strain = []
        # Re-use fits_per_image and post-LM unpacked dicts from the LAST
        # EM iteration (still in scope from the for-loop).
        for fits, per_d in zip(fits_per_image, per_dicts_final):
            merged = {**shared_dict_final, **per_d}
            with torch.no_grad():
                r_un = pseudo_strain_residual(
                    fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, merged,
                    rho_d=fits.rho_d, weights=None,
                    panel_layout=panel_layout, panel_idx=fits.panel_idx,
                )
                # Default px for radial-µm conversion (mean of pxY/pxZ).
                pxY = float(merged.get("pxY", torch.as_tensor(200.0)))
                pxZ = float(merged.get("pxZ", torch.as_tensor(pxY)))
                px_mean = 0.5 * (pxY + pxZ)
                R_ideal = R_ideal_px(
                    fits.ring_two_theta_deg,
                    merged["Lsd"].detach(),
                    torch.as_tensor(px_mean, dtype=fits.Y_pix.dtype),
                )
                # r_un = 1 - R_fwd/R_ideal  →  ΔR_px = R_fwd - R_ideal = -R_ideal·r_un
                delta_R_um = (-R_ideal * r_un) * px_mean
                abs_r = r_un.abs().cpu().numpy()
                if abs_r.size > 0:
                    cutoff = float(np.percentile(abs_r, residual_corr_outlier_pct))
                    keep = torch.as_tensor(abs_r < cutoff)
                    Y_all.append(fits.Y_pix[keep].detach().cpu())
                    Z_all.append(fits.Z_pix[keep].detach().cpu())
                    dR_all.append(delta_R_um[keep].detach().cpu())

        if Y_all and sum(t.numel() for t in Y_all) >= 50:
            Y_cat = torch.cat(Y_all)
            Z_cat = torch.cat(Z_all)
            dR_cat = torch.cat(dR_all)
            NrPixelsY = int(v1_per_image[0].NrPixelsY)
            NrPixelsZ = int(v1_per_image[0].NrPixelsZ)
            pxY0 = float(v1_per_image[0].pxY)
            if verbose:
                print(f"[multi] building residual corr map from "
                      f"{Y_cat.numel()} non-outlier fits across {n_imgs} images...",
                      flush=True)
            residual_map = build_residual_corr_map(
                Y_cat, Z_cat, dR_cat,
                NrPixelsY=NrPixelsY, NrPixelsZ=NrPixelsZ, pxY=pxY0,
                dtype=dtype,
            ).to(device=device)
            if residual_corr_path is not None:
                save_residual_corr_bin(residual_map, residual_corr_path)
                if verbose:
                    print(f"[multi] saved residual map -> {residual_corr_path}",
                          flush=True)

            # Honest post-residual strain: rebuild E-step at MAP, evaluate
            # with the new map applied.
            with torch.no_grad():
                fits_post = [run_estep_v1(v1, img, dark=drk,
                                           dtype=dtype, device=device)
                              for v1, img, drk in zip(v1_per_image, images, darks)]
                for fits, per_d in zip(fits_post, per_dicts_final):
                    merged = {**shared_dict_final, **per_d,
                              "residual_corr_map": residual_map}
                    r = pseudo_strain_residual(
                        fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, merged,
                        rho_d=fits.rho_d, weights=fits.weights,
                        panel_layout=panel_layout, panel_idx=fits.panel_idx,
                    )
                    post_strain.append(float(r.abs().mean()) * 1e6)
                if verbose:
                    pretty = ", ".join(f"img{i}={s:.1f}μϵ" for i, s in enumerate(post_strain))
                    print(f"[multi] strain after residual map: {pretty}",
                          flush=True)
        elif verbose:
            print(f"[multi] residual corr map skipped: only "
                  f"{sum(t.numel() for t in Y_all)} non-outlier fits "
                  f"(need >=50)", flush=True)

    return MultiResult(
        multi_spec=multi_spec,
        shared_unpacked=shared_dict_final,
        per_image_unpacked=per_dicts_final,
        cost=cost_final, rc=rc_final,
        residual_corr_map=residual_map,
        post_residual_strain_uE=post_strain,
    )


__all__ = ["build_multi_spec", "MultiResult", "autocalibrate_multi"]
