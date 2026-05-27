"""Phase 3 of the real-data validation: joint refinement.

Reads the Phase 1 powder-only MAP + Phase 2 grain-fit outputs, builds a
joint ParameterSpec with HEDM nuisance blocks (per-grain euler / pos /
lattice), constructs a joint residual closure (powder pseudo-strain +
per-grain spot pixel residuals), and runs the alternating driver of
paper §4.4 option 2.

Writes:
    <output>/phase3_joint_map.json
    <output>/phase3_joint_cost_history.csv
    <output>/phase3_joint_per_panel_sigma.csv
    <output>/phase3_joint_diagnostics.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tifffile
import torch

import midas_peakfit as mp
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import (
    spec_from_v1_file, add_panel_parameters, add_panel_zero_sum_constraint,
)
from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv

from midas_diffract import HEDMForwardModel
from midas_diffract.forward import HEDMGeometry
from midas_diffract.hkls import hkls_for_forward_model
from midas_hkls import Lattice, SpaceGroup
from midas_fit_grain import HEDMResidualBundle, hedm_spot_residual
from midas_fit_grain.matching import MatchResult, associate, ring_slot_lookup
from midas_fit_grain.observations import ObservedSpots

from midas_joint_ff_calibrate.spec import build_joint_spec
from midas_joint_ff_calibrate.loss import joint_residual, JointWeights


# Pilatus3 2M-CdTe 6×8 layout (decoded in Phase 0)
N_PANELS_Y = 6
N_PANELS_Z = 8
PANEL_SIZE_Y = 243
PANEL_SIZE_Z = 195
GAPS_Y = (1, 7, 1, 7, 1)
GAPS_Z = (17, 17, 17, 17, 17, 17, 17)


# Grain/SpotMatrix loaders + observation builder live in the shared module
# so the lightweight grain-geometry refiner (grain_refine) reuses one
# definition (no dual tree).
from midas_joint_ff_calibrate.grain_observations import (  # noqa: E402
    euler_zxz_from_om,
    load_grains_csv,
    load_spot_matrix,
    load_phase2_grains_and_spots,
)

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1-dir", type=Path, required=True,
                   help="Phase 1 output dir (contains phase1_powder_map.json)")
    p.add_argument("--phase2-layer-dir", type=Path, required=True,
                   help="Phase 2 LayerNr_1 dir (contains Grains.csv, ExtraInfo.bin)")
    p.add_argument("--paramstest-powder", type=Path, required=True)
    p.add_argument("--paramstest-hedm", type=Path, required=True)
    p.add_argument("--calibration-image", type=Path, required=True,
                   help="CeO2 calibration TIFF")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibrant", default="CeO2",
                   choices=["CeO2", "LaB6", "Au", "Si"])
    p.add_argument("--w-powder", type=float, default=1.0e4)
    p.add_argument("--w-hedm",   type=float, default=10.0)
    p.add_argument("--lambda-gauge", type=float, default=1.0e6)
    p.add_argument("--sigma-prior-px", type=float, default=0.5,
                   help="Gaussian prior σ on panel_delta_yz (default 0.5 px, Pilatus3 spec)")
    p.add_argument("--n-outer-max", type=int, default=5)
    p.add_argument("--n-grains-limit", type=int, default=0,
                   help="If >0, refine only the first N grains by completeness")
    p.add_argument("--max-iter", type=int, default=80,
                   help="LM max iterations (default 80; bump higher for full "
                        "convergence at scale)")
    args = p.parse_args(argv)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args.output.mkdir(parents=True, exist_ok=True)

    CALIBRANTS = {
        "CeO2": (225, (5.411651, 5.411651, 5.411651, 90.0, 90.0, 90.0)),
        "LaB6": (221, (4.156813, 4.156813, 4.156813, 90.0, 90.0, 90.0)),
        "Au":   (225, (4.0782,   4.0782,   4.0782,   90.0, 90.0, 90.0)),
        "Si":   (227, (5.4310205, 5.4310205, 5.4310205, 90.0, 90.0, 90.0)),
    }

    print("=" * 70)
    print(" Phase 3 — Joint differentiable refinement")
    print("=" * 70)

    # 1) Load Phase 1 MAP (anchored powder calibration)
    phase1_map = json.loads((args.phase1_dir / "phase1_powder_map.json").read_text())
    print(f"\n>> Phase 1 MAP:")
    for k in ("Lsd", "BC_y", "BC_z", "ty", "tz", "Wavelength"):
        if k in phase1_map:
            print(f"   {k:>10s} = {phase1_map[k]}")

    # 2) Load Phase 2 grains + spots
    print(f"\n>> Loading Phase 2 grains + spots from {args.phase2_layer_dir}")
    grain_eulers_init, grain_pos_init, grain_lat_init, spots, pkey, _smdict = \
        load_phase2_grains_and_spots(args.phase2_layer_dir)

    # 3) Build powder spec from Phase 1 paramstest + apply Phase 1 MAP
    v1 = V1Params.from_file(str(args.paramstest_powder))
    sg, lat = CALIBRANTS[args.calibrant]
    if v1.SpaceGroup == 0: v1.SpaceGroup = sg
    if v1.LatticeConstant[0] == 0: v1.LatticeConstant = lat
    v1.RBinSize = 0.25 if v1.RBinSize <= 0 else v1.RBinSize
    v1.EtaBinSize = 5.0 if v1.EtaBinSize <= 0 else v1.EtaBinSize
    v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
    v1.Width = 800.0 if v1.Width <= 0 else v1.Width

    # Apply Phase 1 MAP to v1 (so cake integration uses refined geometry)
    v1.Lsd  = float(phase1_map["Lsd"])
    v1.BC_y = float(phase1_map["BC_y"])
    v1.BC_z = float(phase1_map["BC_z"])
    v1.ty   = float(phase1_map["ty"])
    v1.tz   = float(phase1_map["tz"])

    layout = PanelLayout.regular(
        n_y=N_PANELS_Y, n_z=N_PANELS_Z,
        sy=PANEL_SIZE_Y, sz=PANEL_SIZE_Z,
        gap_y=GAPS_Y, gap_z=GAPS_Z,
    )
    print(f"\n>> PanelLayout: {layout.n_panels()} panels")

    spec = spec_from_v1_file(str(args.paramstest_powder))
    spec.SpaceGroup = v1.SpaceGroup
    spec.LatticeConstant = v1.LatticeConstant
    add_panel_parameters(spec, n_panels=layout.n_panels(),
        tol_shift_px=4.0, tol_rot_deg=2.0,
        tol_lsd_um=2000.0, tol_p2=2e-2,
        enable_lsd=True, enable_p2=True)

    # Apply Phase 1 MAP per-panel deltas as initial values
    for name in ("panel_delta_yz", "panel_delta_theta",
                  "panel_delta_lsd", "panel_delta_p2"):
        if name in phase1_map and name in spec.parameters:
            arr = np.array(phase1_map[name], dtype=np.float64)
            spec.set_init(name, torch.from_numpy(arr))

    # Apply Gaussian prior on panel_delta_yz (paper §3.3)
    spec.parameters["panel_delta_yz"].prior = mp.GaussianPrior(
        mean=0.0, std=args.sigma_prior_px)

    # 4) Build powder calibration fit set by re-running paper-3's
    #    single-PV pipeline with lm_max_iter=0 (peak fits only, no LM).
    print(f"\n>> Building powder fit set (calibration image)")
    img = tifffile.imread(str(args.calibration_image))
    pv_res = autocalibrate_pv(
        v1, img, dark=None, spec=spec, panel_layout=layout,
        n_iter=1, half_window_px=12, snr_min=2.0, max_per_ring=None,
        trim_mode="stratified_multfactor", trim_residual_pct=5.0,
        reuse_fits=False, lm_max_iter=0, verbose=False,
    )
    fits = pv_res.fits_final
    print(f"   {fits.Y_pix.numel()} powder peaks across "
          f"{len(set(fits.panel_idx.tolist()))} panels")

    # 5) Build HEDMResidualBundle from Phase 2's spots.
    print(f"\n>> Building HEDMResidualBundle …")
    v1_hedm = V1Params.from_file(str(args.paramstest_hedm))

    # Adhoc reader for HEDM-only keys the calibration V1Params doesn't carry.
    hedm_keys = {
        "OmegaFirstFile": 0.0, "OmegaStep": 0.0, "NrFilesPerSweep": 1440,
        "NrPixelsY": 2048, "NrPixelsZ": 2048, "MinEta": 6.0,
    }
    for line in args.paramstest_hedm.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) >= 2 and toks[0] in hedm_keys:
            try:
                hedm_keys[toks[0]] = float(toks[1]) if "." in toks[1] or toks[0].startswith(("Omega", "Min")) else int(toks[1])
            except ValueError:
                pass
    for k, v in hedm_keys.items():
        setattr(v1_hedm, k, v)
    # hkls for the sample (NMC811 hexagonal, space group 166 -- a, b, c, α, β, γ)
    sg_sample = SpaceGroup.from_number(v1_hedm.SpaceGroup)
    lat_sample = Lattice.for_system(
        "hexagonal" if v1_hedm.SpaceGroup in (143, 144, 145, 146, 147, 148, 149, 150,
                                                151, 152, 153, 154, 155, 156, 157,
                                                158, 159, 160, 161, 162, 163, 164,
                                                165, 166, 167)
        else "cubic",
        a=v1_hedm.LatticeConstant[0],
        c=v1_hedm.LatticeConstant[2] if v1_hedm.LatticeConstant[2] != v1_hedm.LatticeConstant[0] else None,
    )
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg_sample, lat_sample,
        wavelength_A=float(phase1_map["Wavelength"]),
        two_theta_max_deg=20.0, expand_equivalents=True,
    )

    geom = HEDMGeometry(
        Lsd=float(phase1_map["Lsd"]),
        y_BC=float(phase1_map["BC_y"]),
        z_BC=float(phase1_map["BC_z"]),
        px=v1.pxY, omega_start=v1_hedm.OmegaFirstFile,
        omega_step=v1_hedm.OmegaStep,
        n_frames=v1_hedm.NrFilesPerSweep,
        n_pixels_y=v1_hedm.NrPixelsY, n_pixels_z=v1_hedm.NrPixelsZ,
        min_eta=v1_hedm.MinEta,
        wavelength=float(phase1_map["Wavelength"]),
        tx=0.0, ty=float(phase1_map["ty"]), tz=float(phase1_map["tz"]),
        flip_y=True, multi_mode="layered",
    )
    model = HEDMForwardModel(hkls_cart, thetas, geom, hkls_int=hkls_int.float())

    # Optionally subset grains (by descending confidence) to keep first-pass refinement tractable.
    confidence = pkey["confidence"]
    order = np.argsort(-confidence)
    n_grains_total = grain_eulers_init.shape[0]
    if args.n_grains_limit > 0:
        n_grains_total = min(n_grains_total, args.n_grains_limit)
    keep_idx = order[:n_grains_total]
    grain_eulers_init = grain_eulers_init[keep_idx]
    grain_pos_init = grain_pos_init[keep_idx]
    grain_lat_init = grain_lat_init[keep_idx]
    spots = [spots[i] for i in keep_idx]
    print(f"   refining {n_grains_total} grains "
          f"(confidence ∈ [{confidence[keep_idx].min():.3f}, "
          f"{confidence[keep_idx].max():.3f}])")

    # 6) Build per-grain ObservedSpots + initial MatchResult.
    # Ring-slot lookup is shared across grains; predicted ring slot is the
    # bucket of model.thetas (rad) onto the sorted unique observed ring 2θ.
    obs_ring_nrs = sorted({int(r) for bag in spots for r in bag.get("ring_nr", [])})
    if len(obs_ring_nrs) == 0:
        raise RuntimeError("No observed rings in any grain's spot bag")
    # Build pred-ring-slot by nearest 2θ (deg) from model.thetas (rad) onto
    # observed-ring 2θ taken from hkls.csv (Phase 2's canonical hkls).
    hkls_csv = args.phase2_layer_dir / "hkls.csv"
    ring_two_theta = {}
    with open(hkls_csv) as f:
        next(f)  # header
        for line in f:
            cols = line.split()
            if len(cols) < 11:
                continue
            rn = int(cols[4]); tt = float(cols[9])
            ring_two_theta.setdefault(rn, tt)
    ring_tt_arr = np.array([ring_two_theta[r] for r in obs_ring_nrs], dtype=np.float64)
    pred_tt_deg = np.rad2deg(2 * model.thetas.detach().cpu().numpy())
    diffs = np.abs(pred_tt_deg[:, None] - ring_tt_arr[None, :])
    # Only assign a ring slot if 2θ matches within 0.05 deg; otherwise -1.
    nearest = diffs.argmin(axis=1)
    nearest_d = diffs[np.arange(diffs.shape[0]), nearest]
    pred_ring_slot_np = np.where(nearest_d < 0.05, nearest, -1)
    pred_ring_slot = torch.from_numpy(pred_ring_slot_np).long()

    observations: List[ObservedSpots] = []
    matches: List[MatchResult] = []
    grain_radius_um = pkey["radius"][keep_idx]
    for g in range(n_grains_total):
        bag = spots[g]
        if "spot_id" not in bag or len(bag["spot_id"]) == 0:
            # Empty grain — still need a stub entry; emit zero-length match.
            observations.append(ObservedSpots(
                spot_id=torch.zeros(0, dtype=torch.int64),
                ring_nr=torch.zeros(0, dtype=torch.int64),
                y_lab=torch.zeros(0, dtype=torch.float64),
                z_lab=torch.zeros(0, dtype=torch.float64),
                omega=torch.zeros(0, dtype=torch.float64),
                eta=torch.zeros(0, dtype=torch.float64),
                two_theta=torch.zeros(0, dtype=torch.float64),
                grain_radius=torch.zeros(0, dtype=torch.float64),
                fit_rmse=torch.zeros(0, dtype=torch.float64),
                y_orig=torch.zeros(0, dtype=torch.float64),
                z_orig=torch.zeros(0, dtype=torch.float64),
                omega_ini=torch.zeros(0, dtype=torch.float64),
                mask_touched=torch.zeros(0, dtype=torch.bool),
            ))
            matches.append(MatchResult(
                k_idx=torch.zeros(0, dtype=torch.int64),
                m_idx=torch.zeros(0, dtype=torch.int64),
                mask=torch.zeros(0, dtype=torch.bool),
                delta_omega=torch.zeros(0, dtype=torch.float64),
                delta_eta=torch.zeros(0, dtype=torch.float64),
            ))
            continue
        S = len(bag["spot_id"])
        ring_nr = torch.from_numpy(bag["ring_nr"]).long()
        omega_rad = torch.from_numpy(np.deg2rad(bag["omega"])).double()
        eta_rad = torch.from_numpy(np.deg2rad(bag["eta"])).double()
        theta_rad = torch.from_numpy(np.deg2rad(bag["theta"])).double()
        two_theta = 2.0 * theta_rad
        y_lab = torch.from_numpy(bag["y_lab"]).double()
        z_lab = torch.from_numpy(bag["z_lab"]).double()
        observations.append(ObservedSpots(
            spot_id=torch.from_numpy(bag["spot_id"]).long(),
            ring_nr=ring_nr,
            y_lab=y_lab, z_lab=z_lab,
            omega=omega_rad, eta=eta_rad,
            two_theta=two_theta,
            grain_radius=torch.full((S,), float(grain_radius_um[g]), dtype=torch.float64),
            fit_rmse=torch.zeros(S, dtype=torch.float64),
            y_orig=y_lab.clone(), z_orig=z_lab.clone(),
            omega_ini=omega_rad.clone(),
            mask_touched=torch.zeros(S, dtype=torch.bool),
        ))
        # Forward at initial euler/pos/lattice to get pred (omega, eta) for matching.
        eu_g = torch.from_numpy(grain_eulers_init[g][None, None, :]).double()
        po_g = torch.from_numpy(grain_pos_init[g][None, None, :]).double()
        la_g = torch.from_numpy(grain_lat_init[g][None, :]).double()
        pred = model(eu_g, po_g, lattice_params=la_g)
        # (1, 1, K, M) → (K, M)
        pred_omega = pred.omega.squeeze(0).squeeze(0).double()
        pred_eta   = pred.eta.squeeze(0).squeeze(0).double()
        pred_valid = pred.valid.squeeze(0).squeeze(0).bool()
        obs_slot = ring_slot_lookup(obs_ring_nrs, ring_nr)
        m = associate(
            obs_ring_nr=ring_nr,
            obs_omega=omega_rad,
            obs_eta=eta_rad,
            pred_ring_slot=pred_ring_slot,
            pred_omega=pred_omega,
            pred_eta=pred_eta,
            pred_valid=pred_valid,
            obs_ring_slot=obs_slot,
            omega_tolerance=math.radians(2.0),
            eta_tolerance=math.radians(3.0),
        )
        matches.append(m)

    n_matched = sum(int(m.mask.sum()) for m in matches)
    n_total = sum(int(o.spot_id.numel()) for o in observations)
    print(f"   matched {n_matched}/{n_total} observed spots ({100*n_matched/max(n_total,1):.1f}%)")

    # 7) Add per-grain blocks to spec.
    spec = build_joint_spec(
        powder_spec=spec,
        grain_eulers_init=torch.from_numpy(grain_eulers_init).double(),
        grain_positions_init=torch.from_numpy(grain_pos_init).double(),
        grain_lattices_init=torch.from_numpy(grain_lat_init).double(),
        refine_grain_orientation=True,
        refine_grain_position=True,
        refine_grain_strain=False,
    )
    # Override default bounds — build_joint_spec leaves grain blocks unbounded,
    # so `fallback_span=2.0` collapses them to init±2 (microscopic for µm pos).
    # And v1 tols (Lsd ±15mm, BC ±20px, tilts ±3°) are too loose to use
    # alongside the Phase 1 MAP anchor: tighten so LM uses Phase 1 as a true
    # warm start, not a free initial point.
    spec.parameters["grain_euler"].bounds = (-2 * math.pi, 2 * math.pi)
    spec.parameters["grain_pos"].bounds = (-1000.0, 1000.0)  # µm, well past
    spec.parameters["Lsd"].bounds = (
        float(phase1_map["Lsd"]) - 2000.0, float(phase1_map["Lsd"]) + 2000.0)
    spec.parameters["BC_y"].bounds = (
        float(phase1_map["BC_y"]) - 5.0, float(phase1_map["BC_y"]) + 5.0)
    spec.parameters["BC_z"].bounds = (
        float(phase1_map["BC_z"]) - 5.0, float(phase1_map["BC_z"]) + 5.0)
    spec.parameters["ty"].bounds = (
        float(phase1_map["ty"]) - 1.0, float(phase1_map["ty"]) + 1.0)
    spec.parameters["tz"].bounds = (
        float(phase1_map["tz"]) - 1.0, float(phase1_map["tz"]) + 1.0)
    print(f"\n>> Joint spec: {len(spec.refined_names())} refined blocks, "
          f"{sum(int(spec.parameters[n].init_tensor().numel()) for n in spec.refined_names())} "
          f"refined dims")

    # 8) Build residual closures (powder pseudo-strain + HEDM pixel) and joint loss.
    bundle = HEDMResidualBundle(
        model=model, observations=observations, matches=matches, kind="pixel",
    )

    def powder_fn(u):
        return pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, u,
            rho_d=u.get("RhoD", torch.tensor(float(v1.RhoD), dtype=torch.float64)),
            panel_layout=layout, panel_idx=fits.panel_idx,
            ring_idx=fits.ring_idx,
            ring_d_spacing_A=fits.ring_d_spacing_A,
        )

    def hedm_fn(u):
        return hedm_spot_residual(u, bundle)

    weights = JointWeights(
        w_powder=args.w_powder, w_hedm=args.w_hedm,
        lambda_gauge=args.lambda_gauge,
    )

    def joint_fn(u):
        return joint_residual(
            u, powder_residual_fn=powder_fn, hedm_residual_fn=hedm_fn,
            spec=spec, weights=weights, gauge_blocks=[],
        )

    # 9) Run joint LM (one-shot; AlternatingDriver is an option for later passes).
    print(f"\n>> Joint LM (w_powder={weights.w_powder}, "
          f"w_hedm={weights.w_hedm}, λ_gauge={weights.lambda_gauge:.1e}) …")
    import time
    t0 = time.time()
    unpacked_init = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    cost_init = float((joint_fn(unpacked_init) ** 2).sum().item())
    unpacked, cost, rc = mp.lm_minimise(
        spec, joint_fn,
        config=mp.GenericLMConfig(max_iter=args.max_iter,
                                   ftol_rel=1e-9, xtol_rel=1e-9),
        fallback_span=2.0,
    )
    dt = time.time() - t0
    print(f"     rc={rc}  cost: {cost_init:.4e} → {cost:.4e}  time={dt:.1f}s")

    # 10) Save outputs.
    map_out = {}
    for name, par in spec.parameters.items():
        v = unpacked[name]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
            map_out[name] = v.tolist() if v.ndim > 0 else float(v)
        else:
            map_out[name] = float(v)
    map_out["_cost_init"] = cost_init
    map_out["_cost_map"] = float(cost)
    map_out["_lm_rc"] = int(rc)
    map_out["_n_grains_refined"] = int(n_grains_total)
    map_out["_n_spots_matched"] = int(n_matched)
    map_out["_n_spots_total"] = int(n_total)
    (args.output / "phase3_joint_map.json").write_text(json.dumps(map_out, indent=2))
    print(f"\n   ✓ Saved phase3_joint_map.json")

    # Compact diagnostics summary.
    diag_lines = [
        f"Phase 3 joint MAP — {n_grains_total} grains, "
        f"{n_matched}/{n_total} matched spots",
        f"  cost: {cost_init:.4e} → {cost:.4e}  (Δ = {cost-cost_init:+.2e}, rc={rc})",
        f"  Lsd:   {float(phase1_map['Lsd']):.2f}  →  {float(unpacked['Lsd']):.2f}  µm",
        f"  BC_y:  {float(phase1_map['BC_y']):.4f}  →  {float(unpacked['BC_y']):.4f}  px",
        f"  BC_z:  {float(phase1_map['BC_z']):.4f}  →  {float(unpacked['BC_z']):.4f}  px",
        f"  ty:    {float(phase1_map['ty']):.6f}  →  {float(unpacked['ty']):.6f}  deg",
        f"  tz:    {float(phase1_map['tz']):.6f}  →  {float(unpacked['tz']):.6f}  deg",
    ]
    if "panel_delta_yz" in unpacked:
        pd = unpacked["panel_delta_yz"].detach().cpu().numpy()
        diag_lines.append(
            f"  panel_delta_yz σ: y={pd[:,0].std():.3f} px,  z={pd[:,1].std():.3f} px  "
            f"(over {layout.n_panels()} panels)")
    (args.output / "phase3_joint_diagnostics.txt").write_text("\n".join(diag_lines) + "\n")
    print("\n".join("   " + l for l in diag_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
