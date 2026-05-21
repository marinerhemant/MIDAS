"""Single-image v2 pipeline — drop-in replacement for v1 autocalibrate.

The control flow is the alternating engine (E-step from v1, M-step from v2's
LM over the parameter spec).  v2's value over v1 here is:
  - pxY, pxZ, tx are first-class refinable.
  - Per-panel parameters live in the same spec.
  - Priors and additional losses can be layered without rewriting the closure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_peakfit import GenericLMConfig

from ..compat.from_v1 import spec_from_v1_params
from ..forward.panels import PanelLayout
from ..inference.lm import lm_minimise
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.spec import CalibrationSpec
from ._common import FittedDataset, run_estep_v1


@dataclass
class IterRecord:
    iteration: int
    n_fitted: int
    cost: float
    rc: int
    mean_strain_uE: float
    Lsd: float
    BC_y: float
    BC_z: float
    ty: float
    tz: float
    # Robust strain summaries: ACZ rejects outlier fits before reporting the
    # calibration quality, so the plain mean over *all* fits overstates it.
    # median_strain_uE and trim_strain_uE (mean after dropping the worst 5%)
    # are the apples-to-apples numbers vs the C tool.
    median_strain_uE: float = 0.0
    trim_strain_uE: float = 0.0


@dataclass
class CalibrationResult:
    spec: CalibrationSpec
    unpacked: dict
    history: List[IterRecord] = field(default_factory=list)
    fits_final: Optional[FittedDataset] = None
    # Empirical residual-correction map built post-MAP (port of v1 C
    # dg_residual_corr_lookup).  None when ``build_residual_corr=False``.
    residual_corr_map: Optional[torch.Tensor] = None
    post_residual_strain_uE: Optional[float] = None


def autocalibrate(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter: int = 5,
    lm_max_iter: int = 200,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
    build_residual_corr: bool = True,
    residual_corr_outlier_pct: float = 90.0,
    residual_corr_path: Optional[str] = None,
) -> CalibrationResult:
    """Run alternating E↔M with the v2 spec.

    Parameters
    ----------
    v1_params : v1 CalibrationParams
        Used for the E-step (cake build, peak extraction) and as a default
        spec source if ``spec`` is None.
    image, dark : numpy arrays.
    spec : optional v2 CalibrationSpec
        Overrides the v1-derived spec.  Use this to enable extra refinable
        parameters (pxY, pxZ, panels) or attach priors.
    panel_layout : optional :class:`PanelLayout` for multi-panel detectors.
    """
    v1_params.validate()
    # Resolve RhoD to µm (RhoD enters only as ρ = R_um / RhoD). Auto-detect
    # the units of the supplied value, and default to the BC-to-farthest-edge
    # distance for the automated / from-scratch case.
    from ..forward.sanity import resolve_rho_d_um
    rho_d_um, _rho_how = resolve_rho_d_um(
        v1_params.RhoD if v1_params.RhoD > 0 else v1_params.MaxRingRad,
        NrPixelsY=int(v1_params.NrPixelsY), NrPixelsZ=int(v1_params.NrPixelsZ),
        BC_y=float(v1_params.BC_y), BC_z=float(v1_params.BC_z),
        pxY=float(v1_params.pxY),
        pxZ=float(v1_params.pxZ if v1_params.pxZ > 0 else v1_params.pxY),
    )
    if verbose:
        print(f"[autocalibrate] RhoD resolved to {rho_d_um:.1f} µm ({_rho_how})")
    v1_params.RhoD = rho_d_um   # canonical µm for E-step + forward distortion
    if spec is None:
        spec = spec_from_v1_params(v1_params)

    history: List[IterRecord] = []
    fits_final: Optional[FittedDataset] = None
    unpacked = None

    for it in range(n_iter):
        # E-step (v1, proven).  Uses current v1_params geometry.
        fits = run_estep_v1(v1_params, image, dark=dark, dtype=dtype, device=device)
        # Multi-panel detectors: tag each fitted point with its panel so the
        # M-step can refine per-panel rigid-body shifts.  The E-step has no
        # panel awareness, so we assign indices here from the panel mask.
        if panel_layout is not None and fits.panel_idx is None:
            from ..forward.panels import panel_idx_for_points
            fits.panel_idx = panel_idx_for_points(panel_layout, fits.Y_pix, fits.Z_pix)

        # M-step closure operates on the v2 unpacked dict.
        def residual_fn(unpacked_now: dict) -> torch.Tensor:
            r = pseudo_strain_residual(
                fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unpacked_now,
                rho_d=fits.rho_d, weights=fits.weights,
                panel_layout=panel_layout, panel_idx=fits.panel_idx,
                ring_idx=fits.ring_idx,
            )
            return r

        unpacked, cost, rc = lm_minimise(
            spec, residual_fn,
            config=GenericLMConfig(max_iter=lm_max_iter, ftol_rel=1e-9, xtol_rel=1e-9),
            dtype=dtype, device=device,
        )

        # Push refined values back into v1 params for the next E-step.
        for name, val in unpacked.items():
            scalar = float(val.detach().reshape(-1)[0]) if val.numel() == 1 else None
            if scalar is None:
                continue
            if hasattr(v1_params, name):
                cur = getattr(v1_params, name)
                try:
                    setattr(v1_params, name, type(cur)(scalar))
                except Exception:
                    setattr(v1_params, name, scalar)

        # Compute mean strain at the converged unpacked dict, plus robust
        # summaries (median + 5%-trimmed mean) for ACZ-comparable reporting.
        with torch.no_grad():
            r_final = residual_fn(unpacked)
            abs_r = r_final.abs()
            mean_strain_uE = float(abs_r.mean()) * 1e6
            median_strain_uE = float(abs_r.median()) * 1e6
            if abs_r.numel() >= 20:
                cut = torch.quantile(abs_r, 0.95)
                inl = abs_r[abs_r <= cut]
                trim_strain_uE = float(inl.mean()) * 1e6 if inl.numel() else mean_strain_uE
            else:
                trim_strain_uE = mean_strain_uE

        rec = IterRecord(
            iteration=it, n_fitted=int(fits.Y_pix.numel()),
            cost=cost, rc=rc, mean_strain_uE=mean_strain_uE,
            Lsd=float(unpacked["Lsd"]),
            BC_y=float(unpacked["BC_y"]), BC_z=float(unpacked["BC_z"]),
            ty=float(unpacked["ty"]), tz=float(unpacked["tz"]),
            median_strain_uE=median_strain_uE, trim_strain_uE=trim_strain_uE,
        )
        history.append(rec)
        fits_final = fits
        if verbose:
            print(f"[v2 iter {it}] n_fits={rec.n_fitted:4d}  rc={rc}  "
                  f"strain={mean_strain_uE:8.1f}μϵ "
                  f"(med={median_strain_uE:6.1f}, trim5%={trim_strain_uE:6.1f})  "
                  f"Lsd={rec.Lsd:.2f}  BC=({rec.BC_y:.3f},{rec.BC_z:.3f})  "
                  f"ty={rec.ty:.4f}  tz={rec.tz:.4f}")

        if len(history) >= 2:
            prev = history[-2].mean_strain_uE
            cur_ms = history[-1].mean_strain_uE
            if cur_ms < 1.0 or abs(prev - cur_ms) < 0.01 * max(prev, 1.0):
                if verbose:
                    print(f"[v2 iter {it}] converged")
                break

    # ---- Post-MAP empirical residual-correction map (v1 parity stage).
    residual_map = None
    post_strain = None
    if build_residual_corr and fits_final is not None and unpacked:
        from ..forward.residual_corr import (
            build_residual_corr_map, save_residual_corr_bin,
        )
        from ..forward.bragg import R_ideal_px
        with torch.no_grad():
            r_un = pseudo_strain_residual(
                fits_final.Y_pix, fits_final.Z_pix,
                fits_final.ring_two_theta_deg, unpacked,
                rho_d=fits_final.rho_d, weights=None,
                panel_layout=panel_layout, panel_idx=fits_final.panel_idx,
                ring_idx=fits_final.ring_idx,
            )
            pxY = float(unpacked.get("pxY", torch.as_tensor(v1_params.pxY)))
            pxZ = float(unpacked.get("pxZ", torch.as_tensor(pxY)))
            px_mean = 0.5 * (pxY + pxZ)
            R_ideal = R_ideal_px(
                fits_final.ring_two_theta_deg,
                unpacked["Lsd"].detach(),
                torch.as_tensor(px_mean, dtype=fits_final.Y_pix.dtype),
            )
            delta_R_um = (-R_ideal * r_un) * px_mean
            abs_r = r_un.abs().cpu().numpy()
            if abs_r.size >= 50:
                import numpy as _np
                cutoff = float(_np.percentile(abs_r, residual_corr_outlier_pct))
                keep = torch.as_tensor(abs_r < cutoff)
                if int(keep.sum()) >= 50:
                    if verbose:
                        print(f"[v2] building residual corr map from "
                              f"{int(keep.sum())} non-outlier fits...",
                              flush=True)
                    residual_map = build_residual_corr_map(
                        fits_final.Y_pix[keep].detach().cpu(),
                        fits_final.Z_pix[keep].detach().cpu(),
                        delta_R_um[keep].detach().cpu(),
                        NrPixelsY=int(v1_params.NrPixelsY),
                        NrPixelsZ=int(v1_params.NrPixelsZ),
                        pxY=pxY, dtype=dtype,
                    ).to(device=device)
                    if residual_corr_path is not None:
                        save_residual_corr_bin(residual_map, residual_corr_path)
                        if verbose:
                            print(f"[v2] saved residual map -> {residual_corr_path}",
                                  flush=True)
                    # Honest post-residual strain at MAP.
                    fits_post = run_estep_v1(v1_params, image, dark=dark,
                                              dtype=dtype, device=device)
                    if panel_layout is not None and fits_post.panel_idx is None:
                        from ..forward.panels import panel_idx_for_points
                        fits_post.panel_idx = panel_idx_for_points(
                            panel_layout, fits_post.Y_pix, fits_post.Z_pix)
                    unpacked_with_map = {**unpacked, "residual_corr_map": residual_map}
                    r_post = pseudo_strain_residual(
                        fits_post.Y_pix, fits_post.Z_pix,
                        fits_post.ring_two_theta_deg, unpacked_with_map,
                        rho_d=fits_post.rho_d, weights=fits_post.weights,
                        panel_layout=panel_layout, panel_idx=fits_post.panel_idx,
                        ring_idx=fits_post.ring_idx,
                    )
                    abs_post = r_post.abs()
                    post_strain = float(abs_post.mean()) * 1e6
                    post_med = float(abs_post.median()) * 1e6
                    if abs_post.numel() >= 20:
                        cut = torch.quantile(abs_post, 0.95)
                        inl = abs_post[abs_post <= cut]
                        post_trim = float(inl.mean()) * 1e6 if inl.numel() else post_strain
                    else:
                        post_trim = post_strain
                    if verbose:
                        print(f"[v2] strain after residual map: {post_strain:.1f} μϵ "
                              f"(med={post_med:.1f}, trim5%={post_trim:.1f})",
                              flush=True)
                    # Guard: the empirical residual map can overfit and *worsen*
                    # strain on low-fit-count / off-panel cases. Keep it only if
                    # it actually reduced the post-MAP strain; otherwise discard.
                    pre_strain = history[-1].mean_strain_uE if history else None
                    if pre_strain is not None and post_strain >= pre_strain:
                        if verbose:
                            print(f"[v2] residual map did not help "
                                  f"({pre_strain:.1f} -> {post_strain:.1f} μϵ); "
                                  f"discarding it", flush=True)
                        residual_map = None
                        post_strain = pre_strain

    # Always report the achieved strain, even when no residual map was applied.
    if post_strain is None and history:
        post_strain = history[-1].mean_strain_uE

    return CalibrationResult(spec=spec, unpacked=unpacked or {},
                              history=history, fits_final=fits_final,
                              residual_corr_map=residual_map,
                              post_residual_strain_uE=post_strain)


__all__ = ["IterRecord", "CalibrationResult", "autocalibrate"]
