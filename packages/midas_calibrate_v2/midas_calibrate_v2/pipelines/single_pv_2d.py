"""Single-image alternating engine using the **2-D** pV peak fitter.

Same control flow as :mod:`single_pv` (cake → batched pV → invert →
M-step → repeat) but the 2-D fitter operates on (R, η) windows so each
peak benefits from much more data per region.  Captures tangential
broadening that the 1-D fit smears into its noise tail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from midas_calibrate.estep import integrate_cake
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate.rings import RingTable, build_ring_table
from midas_integrate.geometry import build_tilt_matrix, invert_REta_to_pixel_batch
from midas_peakfit import GenericLMConfig

from ..compat.from_v1 import spec_from_v1_params
from ..forward.panels import PanelLayout, invert_panel_shifts
from ..forward.peak_fit_2d import fit_cake_per_ring_2d, BatchedFits2D
from ..inference.lm import lm_minimise
from ..loss.diagnostics import strain_summary
from ..loss.pseudo_strain import pseudo_strain_residual
from ..loss.robust_trim import stratified_trim, evaluate_full_strain
from ..parameters.spec import CalibrationSpec
from ..seed.auto_max_ring import auto_detect_max_ring
from ._common import FittedDataset, filter_ring_table


@dataclass
class IterRecord2D:
    iteration: int
    n_fitted: int
    cost: float
    rc: int
    mean_strain_uE: float
    full_mean_uE: float
    full_med_uE: float
    Lsd: float
    BC_y: float
    BC_z: float
    ty: float
    tz: float


@dataclass
class PV2DResult:
    spec: CalibrationSpec
    unpacked: dict
    history: List[IterRecord2D]
    fits_final: Optional[FittedDataset] = None


def _bake_2d_to_dataset(
    fits: BatchedFits2D, v1: V1Params, rt: RingTable, dtype, device,
    panel_layout: Optional[PanelLayout] = None,
    panel_delta_yz: Optional[torch.Tensor] = None,
    panel_delta_theta: Optional[torch.Tensor] = None,
    fix_panel_id: int = 0,
) -> FittedDataset:
    """Same recipe as ``single_pv._bake_fits_to_dataset`` but for 2-D fits."""
    px = 0.5 * (v1.pxY + v1.pxZ) if v1.pxZ > 0 else v1.pxY
    rho_d = v1.RhoD if v1.RhoD > 0 else v1.MaxRingRad
    TRs = build_tilt_matrix(v1.tx, v1.ty, v1.tz)
    R = fits.R_fit.detach().cpu().numpy()
    # The 2D fitter recovers a fitted η offset within the window; use the
    # fitted η (= η_bin_center + offset) for the inversion target.
    Eta = (fits.eta_deg + fits.eta_offset).detach().cpu().numpy()
    p_kw = {f"p{i}": float(getattr(v1, f"p{i}")) for i in range(15)}
    Y_pix, Z_pix = invert_REta_to_pixel_batch(
        R, Eta,
        Ycen=v1.BC_y, Zcen=v1.BC_z, TRs=TRs,
        Lsd=v1.Lsd, RhoD=rho_d, px=px, parallax=v1.Parallax,
        **p_kw,
    )
    rt_tt = torch.tensor(rt.two_theta_deg, dtype=dtype, device=device)
    Y_t = torch.tensor(Y_pix, dtype=dtype, device=device)
    Z_t = torch.tensor(Z_pix, dtype=dtype, device=device)
    panel_idx = None
    if panel_layout is not None and panel_layout.panel_index_mask is not None:
        mask_np = panel_layout.panel_index_mask.cpu().numpy()
        H, W = mask_np.shape
        Yi = np.clip(np.round(Y_pix).astype(int), 0, H - 1)
        Zi = np.clip(np.round(Z_pix).astype(int), 0, W - 1)
        panel_idx = torch.tensor(mask_np[Yi, Zi], dtype=torch.long, device=device)
        if panel_delta_yz is None:
            panel_delta_yz = torch.zeros(panel_layout.n_panels(), 2,
                                          dtype=dtype, device=device)
        if panel_delta_theta is None:
            panel_delta_theta = torch.zeros(panel_layout.n_panels(),
                                              dtype=dtype, device=device)
        Y_t, Z_t = invert_panel_shifts(
            Y_t, Z_t, panel_idx, panel_layout,
            panel_delta_yz, panel_delta_theta,
            fix_panel_id=fix_panel_id,
        )
    return FittedDataset(
        Y_pix=Y_t, Z_pix=Z_t,
        ring_idx=fits.ring_idx, snr=fits.snr,
        ring_two_theta_deg=rt_tt[fits.ring_idx],
        rho_d=torch.as_tensor(rho_d, dtype=dtype, device=device),
        weights=torch.ones(R.shape[0], dtype=dtype, device=device),
        panel_idx=panel_idx,
        rt=rt,
    )


def autocalibrate_pv_2d(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter: int = 3,
    half_window_R_px: float = 6.0,
    half_window_eta_deg: float = 3.0,
    pv_max_iter: int = 50,
    snip_window: int = 0,
    snr_min: float = 3.0,
    lm_max_iter: int = 100,
    huber_delta: Optional[float] = None,
    trim_mode: str = "stratified",
    trim_keep_pct: float = 90.0,
    trim_use_mad: bool = True,
    trim_mad_k: float = 5.0,
    trim_n_eta_buckets: int = 8,
    trim_min_per_cell: int = 3,
    distribution_report: bool = True,
    auto_max_ring: bool = True,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> PV2DResult:
    """Alternating engine using the 2-D pV peak fitter."""
    v1_params.validate()
    if spec is None:
        spec = spec_from_v1_params(v1_params)

    history: List[IterRecord2D] = []
    fits_final: Optional[FittedDataset] = None
    unpacked = None

    for it in range(n_iter):
        rt = build_ring_table(v1_params)
        max_ring_eff = getattr(spec, "max_ring_number", 0)
        if auto_max_ring and max_ring_eff == 0 and it == 0:
            img_for_snr = (image - dark) if dark is not None else image
            mr = auto_detect_max_ring(
                rt.r_ideal_px, v1_params.NrPixelsY, v1_params.NrPixelsZ,
                v1_params.BC_y, v1_params.BC_z, data=img_for_snr,
            )
            if mr > 0:
                spec.max_ring_number = mr
                max_ring_eff = mr
                if verbose:
                    print(f"  [pv2d iter {it}] auto-max-ring: keeping {mr} of "
                          f"{len(rt.ring_nr)} rings", flush=True)
        rt = filter_ring_table(
            rt,
            rings_to_exclude=getattr(spec, "rings_to_exclude", ()),
            max_ring_number=max_ring_eff,
        )
        img_used = (image - dark) if dark is not None else image
        cake = integrate_cake(v1_params, img_used, rt)

        cake_t = torch.as_tensor(cake.intensity, dtype=dtype, device=device)
        R_centers = torch.as_tensor(cake.R_centers, dtype=dtype, device=device)
        eta_centers = torch.as_tensor(cake.eta_centers, dtype=dtype, device=device)
        rt_R_ideal = torch.as_tensor(rt.r_ideal_px, dtype=dtype, device=device)

        bf = fit_cake_per_ring_2d(
            cake_t, R_centers, eta_centers, rt_R_ideal,
            half_window_R_px=half_window_R_px,
            half_window_eta_deg=half_window_eta_deg,
            max_iter=pv_max_iter, snr_min=snr_min,
            snip_window=snip_window,
            dtype=dtype, device=device, verbose=verbose,
        )
        if verbose:
            offsets = (bf.R_fit - rt_R_ideal[bf.ring_idx]).detach().cpu().numpy()
            print(f"  [pV-2D] |R_fit - R_ideal|: med={np.median(np.abs(offsets)):.3f} "
                  f"q90={np.quantile(np.abs(offsets), 0.9):.3f} px", flush=True)

        # Bake to FittedDataset using current panel state from spec.
        cur_dyz = (spec.parameters["panel_delta_yz"].init
                    if "panel_delta_yz" in spec.parameters
                    and isinstance(spec.parameters["panel_delta_yz"].init, torch.Tensor)
                    else None)
        cur_dth = (spec.parameters["panel_delta_theta"].init
                    if "panel_delta_theta" in spec.parameters
                    and isinstance(spec.parameters["panel_delta_theta"].init, torch.Tensor)
                    else None)
        fix_id = getattr(spec, "fix_panel_id", 0)
        fits_ds = _bake_2d_to_dataset(
            bf, v1_params, rt, dtype, device,
            panel_layout=panel_layout,
            panel_delta_yz=cur_dyz, panel_delta_theta=cur_dth,
            fix_panel_id=fix_id,
        )
        # SNR filter on the bake output.
        keep_snr = (fits_ds.snr >= snr_min) & torch.isfinite(fits_ds.Y_pix) & torch.isfinite(fits_ds.Z_pix)
        fits_ds = FittedDataset(
            Y_pix=fits_ds.Y_pix[keep_snr], Z_pix=fits_ds.Z_pix[keep_snr],
            ring_idx=fits_ds.ring_idx[keep_snr], snr=fits_ds.snr[keep_snr],
            ring_two_theta_deg=fits_ds.ring_two_theta_deg[keep_snr],
            rho_d=fits_ds.rho_d,
            weights=(fits_ds.weights[keep_snr] if fits_ds.weights is not None else None),
            panel_idx=(fits_ds.panel_idx[keep_snr] if fits_ds.panel_idx is not None else None),
            rt=fits_ds.rt,
        )
        full_fits_ds = fits_ds

        def residual_fn(unpacked_now: dict) -> torch.Tensor:
            return pseudo_strain_residual(
                fits_ds.Y_pix, fits_ds.Z_pix, fits_ds.ring_two_theta_deg,
                unpacked_now,
                rho_d=fits_ds.rho_d, weights=fits_ds.weights,
                panel_layout=panel_layout, panel_idx=fits_ds.panel_idx,
            )

        # Stratified trim (default).  Reuse same logic as single_pv.
        if trim_mode != "off":
            with torch.no_grad():
                from ..parameters.pack import pack_spec, unpack_spec
                x_full, info = pack_spec(spec, dtype=dtype, device=device)
                unpacked0 = unpack_spec(x_full, info, spec)
                r_pre = residual_fn(unpacked0).abs()
            n_before = int(fits_ds.Y_pix.numel())
            if trim_mode == "stratified":
                with torch.no_grad():
                    keep, trim_report = stratified_trim(
                        r_pre, fits_ds.ring_idx, fits_ds.ring_two_theta_deg,
                        panel_idx=fits_ds.panel_idx,
                        keep_pct=trim_keep_pct,
                        n_eta_buckets=trim_n_eta_buckets,
                        min_per_cell=trim_min_per_cell,
                        use_mad=trim_use_mad,
                        mad_k=trim_mad_k,
                    )
                if verbose:
                    print(f"  [pv2d iter {it}] stratified trim: {n_before} → {int(keep.sum())} "
                          f"({100.0 * (1 - int(keep.sum()) / max(n_before, 1)):.1f}% rejected)",
                          flush=True)
                    print("  " + trim_report.render().replace("\n", "\n  "), flush=True)
            else:
                cutoff = float(torch.quantile(r_pre, trim_keep_pct / 100.0))
                keep = r_pre <= cutoff
            fits_ds = FittedDataset(
                Y_pix=fits_ds.Y_pix[keep], Z_pix=fits_ds.Z_pix[keep],
                ring_idx=fits_ds.ring_idx[keep], snr=fits_ds.snr[keep],
                ring_two_theta_deg=fits_ds.ring_two_theta_deg[keep],
                rho_d=fits_ds.rho_d,
                weights=(fits_ds.weights[keep] if fits_ds.weights is not None else None),
                panel_idx=(fits_ds.panel_idx[keep] if fits_ds.panel_idx is not None else None),
                rt=fits_ds.rt,
            )

            def residual_fn(unpacked_now: dict) -> torch.Tensor:    # rebound
                return pseudo_strain_residual(
                    fits_ds.Y_pix, fits_ds.Z_pix, fits_ds.ring_two_theta_deg,
                    unpacked_now,
                    rho_d=fits_ds.rho_d, weights=fits_ds.weights,
                    panel_layout=panel_layout, panel_idx=fits_ds.panel_idx,
                )

        unpacked, cost, rc = lm_minimise(
            spec, residual_fn,
            config=GenericLMConfig(max_iter=lm_max_iter, ftol_rel=1e-9,
                                    xtol_rel=1e-9, huber_delta=huber_delta),
            dtype=dtype, device=device,
        )

        for name, val in unpacked.items():
            if val.numel() == 1 and hasattr(v1_params, name):
                cur = getattr(v1_params, name)
                try:
                    setattr(v1_params, name, type(cur)(float(val.detach())))
                except Exception:
                    pass
            if name in spec.parameters:
                if val.numel() == 1:
                    spec.parameters[name].init = float(val.detach())
                else:
                    spec.parameters[name].init = val.detach().cpu()

        with torch.no_grad():
            r_final = residual_fn(unpacked)
            mean_uE = float(r_final.abs().mean()) * 1e6

        def _full_resid(u):
            return pseudo_strain_residual(
                full_fits_ds.Y_pix, full_fits_ds.Z_pix,
                full_fits_ds.ring_two_theta_deg, u,
                rho_d=full_fits_ds.rho_d, weights=full_fits_ds.weights,
                panel_layout=panel_layout, panel_idx=full_fits_ds.panel_idx,
            )
        full_mean_uE, full_med_uE, full_rms_uE = evaluate_full_strain(_full_resid, unpacked)
        full_mean_uE *= 1e6; full_med_uE *= 1e6; full_rms_uE *= 1e6

        if verbose:
            print(f"  [pv2d iter {it}] strain={mean_uE:7.2f}  "
                  f"FULL: mean={full_mean_uE:7.2f}  med={full_med_uE:7.2f}  "
                  f"rms={full_rms_uE:7.2f} μϵ", flush=True)
            if distribution_report:
                with torch.no_grad():
                    r_full_uE = _full_resid(unpacked).abs() * 1e6
                from ..forward.geometry import pixel_to_REta
                from ..forward.distortion import build_p_coeffs as _build_pc
                p_eff = unpacked
                p_coeffs = _build_pc(p_eff, dtype=dtype)
                eta_per_fit = pixel_to_REta(
                    full_fits_ds.Y_pix, full_fits_ds.Z_pix,
                    Lsd=p_eff["Lsd"], BC_y=p_eff["BC_y"], BC_z=p_eff["BC_z"],
                    tx=p_eff.get("tx", torch.zeros((), dtype=dtype)),
                    ty=p_eff["ty"], tz=p_eff["tz"], p_coeffs=p_coeffs,
                    parallax=p_eff.get("Parallax", torch.zeros((), dtype=dtype)),
                    pxY=p_eff["pxY"], pxZ=p_eff.get("pxZ", p_eff["pxY"]),
                    rho_d=full_fits_ds.rho_d,
                    panel_layout=panel_layout, panel_idx=full_fits_ds.panel_idx,
                    delta_yz=p_eff.get("panel_delta_yz"),
                    delta_theta=p_eff.get("panel_delta_theta"),
                ).eta_deg.detach()
                print(strain_summary(
                    r_full_uE,
                    ring_idx=full_fits_ds.ring_idx,
                    eta_deg=eta_per_fit,
                    panel_idx=full_fits_ds.panel_idx,
                ), flush=True)

        rec = IterRecord2D(
            iteration=it, n_fitted=int(fits_ds.Y_pix.numel()),
            cost=cost, rc=rc, mean_strain_uE=mean_uE,
            full_mean_uE=full_mean_uE, full_med_uE=full_med_uE,
            Lsd=float(unpacked["Lsd"]),
            BC_y=float(unpacked["BC_y"]), BC_z=float(unpacked["BC_z"]),
            ty=float(unpacked["ty"]), tz=float(unpacked["tz"]),
        )
        history.append(rec)
        fits_final = fits_ds

        if len(history) >= 2:
            prev = history[-2].full_med_uE
            cur_ms = full_med_uE
            if (cur_ms < 1.0 or abs(prev - cur_ms) < 0.01 * max(prev, 1.0)
                    or cur_ms > prev * 1.05):
                if verbose:
                    print(f"  [pv2d iter {it}] terminating ({prev:.2f} → {cur_ms:.2f})",
                          flush=True)
                break

    return PV2DResult(spec=spec, unpacked=unpacked or {},
                       history=history, fits_final=fits_final)


__all__ = ["PV2DResult", "IterRecord2D", "autocalibrate_pv_2d"]
