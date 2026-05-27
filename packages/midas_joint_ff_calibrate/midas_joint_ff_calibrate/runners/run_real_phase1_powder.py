"""Phase 1 of the real-data validation: powder-only baseline calibration.

Runs paper-3's differentiable Bayesian calibration on a single CeO2 image,
with the actual Pilatus3 2M-CdTe 6×8 panel layout decoded from the data
(panel size 243×195 px; Y gaps (1,7,1,7,1); Z gaps 17 uniform).  Writes:

    <output>/phase1_powder_map.json    MAP for every refined parameter
    <output>/phase1_powder_sigma.csv   marginal σ per refined parameter
    <output>/phase1_powder_panels.csv  per-panel δyz/δθ/δLsd/δp2 with σ

The output paramstest serves as the starting calibration for Phase 2
(midas-ff-pipeline) and as the powder-only baseline against which the
joint-refinement MAP (Phase 3) is compared.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import tifffile

import midas_peakfit as mp
from midas_calibrate_v2.compat.from_v1 import (
    spec_from_v1_file, add_panel_parameters, add_panel_zero_sum_constraint,
)
from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv
from midas_calibrate.params import CalibrationParams as V1Params


# --------------------------------------------------------------------
# Pilatus3 2M-CdTe 6×8 layout decoded from CeO2 image
# (see Phase 0 inventory in dev/paper/REAL_DATA_PLAN.md).
# --------------------------------------------------------------------
N_PANELS_Y = 6
N_PANELS_Z = 8
PANEL_SIZE_Y = 243
PANEL_SIZE_Z = 195
GAPS_Y = (1, 7, 1, 7, 1)      # 5 non-uniform gaps in Y
GAPS_Z = (17, 17, 17, 17, 17, 17, 17)  # 7 uniform gaps in Z


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paramstest", type=Path, required=True,
                        help="Path to v1-format powder paramstest (e.g. powder_starting.txt)")
    parser.add_argument("--image", type=Path, required=True,
                        help="Path to the CeO2 calibration image TIFF")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--n-iter", type=int, default=4,
                        help="Outer iterations of the four-stage pipeline (default 4)")
    parser.add_argument("--half-window", type=float, default=8.0,
                        help="Half-window for peak fitting (px, default 8)")
    parser.add_argument("--trim", type=float, default=5.0,
                        help="Stratified-trim residual percentile (default 5%%)")
    parser.add_argument("--snr-min", type=float, default=8.0,
                        help="Minimum SNR for accepting a PV fit (default 8)")
    parser.add_argument("--no-panel-refine", action="store_true",
                        help="Skip per-panel refinement (single rigid detector)")
    parser.add_argument("--calibrant", default="CeO2",
                        choices=["CeO2", "LaB6", "Au", "Si"],
                        help="Calibrant identity (sets SpaceGroup + LatticeConstant)")
    args = parser.parse_args(argv)

    CALIBRANTS = {
        "CeO2": (225, (5.411651, 5.411651, 5.411651, 90.0, 90.0, 90.0)),  # NIST SRM 674b
        "LaB6": (221, (4.156813, 4.156813, 4.156813, 90.0, 90.0, 90.0)),  # NIST SRM 660c
        "Au":   (225, (4.0782,   4.0782,   4.0782,   90.0, 90.0, 90.0)),  # gold (standard)
        "Si":   (227, (5.4310205, 5.4310205, 5.4310205, 90.0, 90.0, 90.0)),  # NIST SRM 640
    }

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Phase 1 — Powder-only baseline (CeO2 / Pilatus3 2M-CdTe)")
    print("=" * 70)

    # ----- 1. Load paramstest and image
    print(f"\n>> paramstest: {args.paramstest}")
    v1 = V1Params.from_file(str(args.paramstest))
    # The user's powder_starting.txt omits SpaceGroup / LatticeConstant /
    # ring lookup --- fill them in from the calibrant choice.
    sg, lat = CALIBRANTS[args.calibrant]
    if v1.SpaceGroup == 0:
        v1.SpaceGroup = sg
    if v1.LatticeConstant[0] == 0:
        v1.LatticeConstant = lat
    # paper-3 needs some sensible defaults for the ring search.
    if getattr(v1, "RBinSize", 0.0) <= 0.0 or v1.RBinSize > 0.5:
        v1.RBinSize = 0.25
    if getattr(v1, "EtaBinSize", 0.0) <= 0.0:
        v1.EtaBinSize = 5.0
    if v1.MaxRingRad <= 0 and getattr(v1, "RMax", 0):
        v1.MaxRingRad = float(v1.RMax)
    v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
    if getattr(v1, "Width", 0.0) <= 0.0:
        v1.Width = 800.0
    print(f"   calibrant   = {args.calibrant}  (SG={v1.SpaceGroup}, a={v1.LatticeConstant[0]:.6f} Å)")
    print(f"   Lsd = {v1.Lsd:.2f} µm,  BC = ({v1.BC_y:.3f}, {v1.BC_z:.3f}) px")
    print(f"   tilts (tx, ty, tz) = ({v1.tx:.4f}, {v1.ty:.4f}, {v1.tz:.4f}) deg")
    print(f"   λ = {v1.Wavelength:.6f} Å,  pxY = pxZ = {v1.pxY} µm")
    print(f"   RhoD = {v1.RhoD:.2f},  MaxRingRad = {v1.MaxRingRad:.0f}")

    print(f"\n>> image: {args.image}")
    img = tifffile.imread(str(args.image))
    # tifffile loads Pilatus TIFFs as (rows=Z, cols=Y) which is ALREADY
    # MIDAS convention --- do NOT transpose.  Verified empirically by
    # sweeping all 8 standard image transforms and choosing the one with
    # the smallest starting-params pseudo-strain (identity wins).
    print(f"   shape = {img.shape},  dtype = {img.dtype}")
    print(f"   min/median/max = {img.min()} / {np.median(img):.1f} / {img.max()}")

    # ----- 2. Build CalibrationSpec from v1 + add panel parameters
    print(f"\n>> Building CalibrationSpec …")
    spec = spec_from_v1_file(str(args.paramstest))

    # Pilatus3 2M-CdTe non-uniform Y gaps require explicit per-axis spec.
    layout = PanelLayout.regular(
        n_y=N_PANELS_Y, n_z=N_PANELS_Z,
        sy=PANEL_SIZE_Y, sz=PANEL_SIZE_Z,
        gap_y=GAPS_Y, gap_z=GAPS_Z,
    )
    print(f"   PanelLayout: {N_PANELS_Y}×{N_PANELS_Z} = {layout.n_panels()} panels, "
          f"panel size ({PANEL_SIZE_Y}, {PANEL_SIZE_Z}) px")
    print(f"   Y gaps: {GAPS_Y},  Z gaps: 17 px uniform")

    if not args.no_panel_refine:
        add_panel_parameters(
            spec, n_panels=layout.n_panels(),
            tol_shift_px=4.0, tol_rot_deg=2.0,
            tol_lsd_um=2000.0, tol_p2=2e-2,
            enable_lsd=True, enable_p2=True,
        )
        add_panel_zero_sum_constraint(spec, lambda_zs=1e6)
        print(f"   add_panel_parameters: refined panel_delta_yz / _theta / _lsd / _p2 + Σ=0 gauge")

    # ----- 3. Run paper-3 single-image differentiable calibration
    print(f"\n>> autocalibrate_pv …  (this writes to {args.output})")
    res = autocalibrate_pv(
        v1, img, dark=None, spec=spec, panel_layout=layout,
        n_iter=args.n_iter, half_window_px=args.half_window,
        snr_min=args.snr_min, max_per_ring=None,
        trim_mode="stratified_multfactor", trim_residual_pct=args.trim,
        huber_delta=None, reuse_fits=True, lm_max_iter=80,
        verbose=True,
    )

    # ----- 4. Save MAP + σ.  PVCalibrationResult has .spec, .unpacked,
    # .history, .fits_final --- but NO Laplace.  Run Fisher J'J explicitly
    # on the trimmed final fit set (paper-3 idiom; see
    # dev/paper/runners/run_uncertainty_pilatus_panels.py).
    print(f"\n>> Saving MAP and σ tables to {args.output}")
    map_unpacked = res.unpacked
    fits_ds = res.fits_final
    spec_final = res.spec

    laplace = None
    if fits_ds is not None:
        from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
        from midas_calibrate_v2.loss.robust_trim import multfactor_trim
        from midas_peakfit import fisher_at_map

        Y = fits_ds.Y_pix; Z = fits_ds.Z_pix; tt = fits_ds.ring_two_theta_deg
        rho_d = fits_ds.rho_d
        panel_idx = fits_ds.panel_idx

        with torch.no_grad():
            r0 = pseudo_strain_residual(
                Y, Z, tt, map_unpacked,
                rho_d=rho_d, panel_layout=layout, panel_idx=panel_idx,
            )
            keep, _ = multfactor_trim(r0, factor=2.0)
        Yk = Y[keep]; Zk = Z[keep]; ttk = tt[keep]
        pidxk = panel_idx[keep] if panel_idx is not None else None
        sigma_r = float(((r0 * r0).mean()) ** 0.5)
        print(f"   trimmed N = {int(keep.sum())},  σ_r = {sigma_r:.3e}")

        def residual_fn(unp):
            return pseudo_strain_residual(
                Yk, Zk, ttk, unp,
                rho_d=rho_d, panel_layout=layout, panel_idx=pidxk,
            )

        try:
            laplace = fisher_at_map(spec_final, residual_fn, map_unpacked,
                                     sigma_r=sigma_r, dtype=torch.float64,
                                     device="cpu")
            print(f"   Fisher J'J done, {len(laplace.refined_names)} refined param blocks")
        except Exception as e:
            print(f"   Fisher failed: {type(e).__name__}: {e}")

    map_dict: Dict[str, float | list] = {}
    for name, val in map_unpacked.items():
        if isinstance(val, torch.Tensor):
            v = val.detach().cpu().tolist()
        else:
            v = val
        map_dict[name] = v
    (args.output / "phase1_powder_map.json").write_text(json.dumps(map_dict, indent=2))

    if laplace is not None:
        sig_rows = []
        for name, offset, size in zip(laplace.refined_names,
                                        laplace.refined_offsets,
                                        laplace.refined_sizes):
            for k in range(size):
                idx = offset + k
                tag = f"{name}[{k}]" if size > 1 else name
                sig_rows.append((tag, float(laplace.map_refined[idx]),
                                  float(laplace.sigma_per_dim[idx])))
        with open(args.output / "phase1_powder_sigma.csv", "w") as f:
            f.write("name,map,sigma\n")
            for n, m, s in sig_rows:
                f.write(f"{n},{m:.10e},{s:.6e}\n")

        # Per-panel breakdown
        with open(args.output / "phase1_powder_panels.csv", "w") as f:
            f.write("panel_id,dy,sigma_dy,dz,sigma_dz,dtheta,sigma_dtheta,dLsd,sigma_dLsd,dp2,sigma_dp2\n")
            block_names = ("panel_delta_yz", "panel_delta_theta",
                            "panel_delta_lsd", "panel_delta_p2")
            block_data = {n: map_unpacked.get(n) for n in block_names}
            block_sigma = {n: [] for n in block_names}
            for name, offset, size in zip(laplace.refined_names,
                                            laplace.refined_offsets,
                                            laplace.refined_sizes):
                if name in block_names:
                    for k in range(size):
                        block_sigma[name].append(float(laplace.sigma_per_dim[offset + k]))
            for k in range(layout.n_panels()):
                d_yz = block_data.get("panel_delta_yz")
                d_th = block_data.get("panel_delta_theta")
                d_lsd = block_data.get("panel_delta_lsd")
                d_p2 = block_data.get("panel_delta_p2")
                dy = float(d_yz[k, 0]) if d_yz is not None else 0.0
                dz = float(d_yz[k, 1]) if d_yz is not None else 0.0
                dth = float(d_th[k]) if d_th is not None else 0.0
                dl = float(d_lsd[k]) if d_lsd is not None else 0.0
                dp = float(d_p2[k]) if d_p2 is not None else 0.0
                sy_yz = block_sigma["panel_delta_yz"]
                sy_dy = sy_yz[2 * k] if len(sy_yz) > 2 * k else float("nan")
                sy_dz = sy_yz[2 * k + 1] if len(sy_yz) > 2 * k + 1 else float("nan")
                sy_th = block_sigma["panel_delta_theta"][k] if len(block_sigma["panel_delta_theta"]) > k else float("nan")
                sy_lsd = block_sigma["panel_delta_lsd"][k] if len(block_sigma["panel_delta_lsd"]) > k else float("nan")
                sy_p2 = block_sigma["panel_delta_p2"][k] if len(block_sigma["panel_delta_p2"]) > k else float("nan")
                f.write(f"{k},{dy:.6e},{sy_dy:.6e},{dz:.6e},{sy_dz:.6e},"
                        f"{dth:.6e},{sy_th:.6e},{dl:.6e},{sy_lsd:.6e},"
                        f"{dp:.6e},{sy_p2:.6e}\n")

    print(f"\n>> Phase 1 complete.  Headline MAP:")
    for n in ("Lsd", "BC_y", "BC_z", "ty", "tz", "Wavelength"):
        if n in map_dict:
            print(f"   {n:>16s} = {map_dict[n]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
