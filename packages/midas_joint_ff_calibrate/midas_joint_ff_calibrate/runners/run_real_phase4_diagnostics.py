"""Phase 4 of the real-data validation: identifiability diagnostics at
the Phase 3 joint MAP.

Re-creates the analogues of the synthetic paper figures on real data:
    - Fisher block rank on per-panel delta_yz (powder-only / HEDM-only / joint),
      both data-only (gauge-free) and gauge-included variants
    - Fisher rank on (Lsd, Wavelength) — the rank-1 gauge demo from §9
    - All-blocks Fisher table across logical refined blocks
    - Per-panel σ comparison CSV (joint vs powder-only vs HEDM-only)

Writes (under <output>/):
    phase4_per_panel_sigma.csv       — per-panel σ_y / σ_z under three modalities
    phase4_gauge_demo.csv            — (Lsd, λ) Fisher under three modalities
    phase4_all_blocks_fisher.csv     — block-level (rank, cond, σ_max, σ_med)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile
import torch

import midas_peakfit as mp
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import (
    spec_from_v1_file, add_panel_parameters,
)
from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv

from midas_diffract import HEDMForwardModel
from midas_diffract.forward import HEDMGeometry
from midas_diffract.hkls import hkls_for_forward_model
from midas_hkls import Lattice, SpaceGroup
from midas_fit_grain import HEDMResidualBundle, hedm_spot_residual
from midas_fit_grain.matching import associate, ring_slot_lookup
from midas_fit_grain.observations import ObservedSpots
from midas_fit_grain.matching import MatchResult

from midas_joint_ff_calibrate.spec import build_joint_spec
from midas_joint_ff_calibrate.loss import joint_residual, JointWeights
from midas_joint_ff_calibrate.pipelines.identifiability import fisher_block_rank

# Phase 3 setup helpers (same loaders).
from midas_joint_ff_calibrate.runners.run_real_phase3_joint import (
    load_phase2_grains_and_spots, load_grains_csv,
)

# Pilatus3 2M-CdTe layout (from Phase 0)
N_PANELS_Y = 6
N_PANELS_Z = 8
PANEL_SIZE_Y = 243
PANEL_SIZE_Z = 195
GAPS_Y = (1, 7, 1, 7, 1)
GAPS_Z = (17, 17, 17, 17, 17, 17, 17)

CALIBRANTS = {
    "CeO2": (225, (5.411651, 5.411651, 5.411651, 90.0, 90.0, 90.0)),
}


def _build_setup(args):
    """Re-build powder spec + HEDM bundle + joint spec, identical to Phase 3."""
    phase3_map = json.loads((args.phase3_dir / "phase3_joint_map.json").read_text())

    print(">> Loading Phase 2 grains + spots")
    grain_eulers_init, grain_pos_init, grain_lat_init, spots, pkey, _sm = \
        load_phase2_grains_and_spots(args.phase2_layer_dir)

    v1 = V1Params.from_file(str(args.paramstest_powder))
    sg, lat = CALIBRANTS[args.calibrant]
    if v1.SpaceGroup == 0: v1.SpaceGroup = sg
    if v1.LatticeConstant[0] == 0: v1.LatticeConstant = lat
    v1.RBinSize = 0.25 if v1.RBinSize <= 0 else v1.RBinSize
    v1.EtaBinSize = 5.0 if v1.EtaBinSize <= 0 else v1.EtaBinSize
    v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
    v1.Width = 800.0 if v1.Width <= 0 else v1.Width
    v1.Lsd  = float(phase3_map["Lsd"])
    v1.BC_y = float(phase3_map["BC_y"])
    v1.BC_z = float(phase3_map["BC_z"])
    v1.ty   = float(phase3_map["ty"])
    v1.tz   = float(phase3_map["tz"])

    layout = PanelLayout.regular(
        n_y=N_PANELS_Y, n_z=N_PANELS_Z,
        sy=PANEL_SIZE_Y, sz=PANEL_SIZE_Z,
        gap_y=GAPS_Y, gap_z=GAPS_Z,
    )
    print(f">> PanelLayout: {layout.n_panels()} panels")

    spec = spec_from_v1_file(str(args.paramstest_powder))
    spec.SpaceGroup = v1.SpaceGroup
    spec.LatticeConstant = v1.LatticeConstant
    add_panel_parameters(spec, n_panels=layout.n_panels(),
        tol_shift_px=4.0, tol_rot_deg=2.0,
        tol_lsd_um=2000.0, tol_p2=2e-2,
        enable_lsd=True, enable_p2=True)

    # Seed panel blocks at Phase 3 MAP values.
    for name in ("panel_delta_yz", "panel_delta_theta",
                  "panel_delta_lsd", "panel_delta_p2"):
        if name in phase3_map and name in spec.parameters:
            arr = np.array(phase3_map[name], dtype=np.float64)
            spec.set_init(name, torch.from_numpy(arr))
    spec.parameters["panel_delta_yz"].prior = mp.GaussianPrior(
        mean=0.0, std=args.sigma_prior_px)

    print(">> Building powder fit set (calibration image)")
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

    print(">> Building HEDMResidualBundle …")
    v1_hedm = V1Params.from_file(str(args.paramstest_hedm))
    hedm_keys = {
        "OmegaFirstFile": 0.0, "OmegaStep": 0.0, "NrFilesPerSweep": 1440,
        "NrPixelsY": 2048, "NrPixelsZ": 2048, "MinEta": 6.0,
    }
    for line in args.paramstest_hedm.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line: continue
        toks = line.split()
        if len(toks) >= 2 and toks[0] in hedm_keys:
            try:
                hedm_keys[toks[0]] = float(toks[1]) if "." in toks[1] or toks[0].startswith(("Omega","Min")) else int(toks[1])
            except ValueError:
                pass
    for k, v in hedm_keys.items():
        setattr(v1_hedm, k, v)

    sg_sample = SpaceGroup.from_number(v1_hedm.SpaceGroup)
    lat_sample = Lattice.for_system(
        "hexagonal" if v1_hedm.SpaceGroup in range(143, 168) else "cubic",
        a=v1_hedm.LatticeConstant[0],
        c=v1_hedm.LatticeConstant[2] if v1_hedm.LatticeConstant[2] != v1_hedm.LatticeConstant[0] else None,
    )
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg_sample, lat_sample,
        wavelength_A=float(phase3_map["Wavelength"]),
        two_theta_max_deg=20.0, expand_equivalents=True,
    )

    geom = HEDMGeometry(
        Lsd=float(phase3_map["Lsd"]), y_BC=float(phase3_map["BC_y"]),
        z_BC=float(phase3_map["BC_z"]),
        px=v1.pxY, omega_start=v1_hedm.OmegaFirstFile,
        omega_step=v1_hedm.OmegaStep, n_frames=v1_hedm.NrFilesPerSweep,
        n_pixels_y=v1_hedm.NrPixelsY, n_pixels_z=v1_hedm.NrPixelsZ,
        min_eta=v1_hedm.MinEta, wavelength=float(phase3_map["Wavelength"]),
        tx=0.0, ty=float(phase3_map["ty"]), tz=float(phase3_map["tz"]),
        flip_y=True, multi_mode="layered",
    )
    model = HEDMForwardModel(hkls_cart, thetas, geom, hkls_int=hkls_int.float())

    # Per-grain bundle build (mimics Phase 3 setup).
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
    print(f"   refining {n_grains_total} grains")

    obs_ring_nrs = sorted({int(r) for bag in spots for r in bag.get("ring_nr", [])})
    hkls_csv = args.phase2_layer_dir / "hkls.csv"
    ring_two_theta = {}
    with open(hkls_csv) as f:
        next(f)
        for line in f:
            cols = line.split()
            if len(cols) >= 11:
                rn = int(cols[4]); tt = float(cols[9])
                ring_two_theta.setdefault(rn, tt)
    ring_tt_arr = np.array([ring_two_theta[r] for r in obs_ring_nrs], dtype=np.float64)
    pred_tt_deg = np.rad2deg(2 * model.thetas.detach().cpu().numpy())
    diffs = np.abs(pred_tt_deg[:, None] - ring_tt_arr[None, :])
    nearest = diffs.argmin(axis=1)
    nearest_d = diffs[np.arange(diffs.shape[0]), nearest]
    pred_ring_slot = torch.from_numpy(np.where(nearest_d < 0.05, nearest, -1)).long()

    observations: List[ObservedSpots] = []
    matches: List[MatchResult] = []
    grain_radius_um = pkey["radius"][keep_idx]
    for g in range(n_grains_total):
        bag = spots[g]
        S = len(bag.get("spot_id", []))
        if S == 0:
            obs = ObservedSpots(*([torch.zeros(0, dtype=torch.float64)]*12 + [torch.zeros(0, dtype=torch.bool)]))
            mr  = MatchResult(*([torch.zeros(0, dtype=torch.int64)]*2 + [torch.zeros(0, dtype=torch.bool)] + [torch.zeros(0, dtype=torch.float64)]*2))
        else:
            ring_nr = torch.from_numpy(bag["ring_nr"]).long()
            omega_rad = torch.from_numpy(np.deg2rad(bag["omega"])).double()
            eta_rad = torch.from_numpy(np.deg2rad(bag["eta"])).double()
            theta_rad = torch.from_numpy(np.deg2rad(bag["theta"])).double()
            y_lab = torch.from_numpy(bag["y_lab"]).double()
            z_lab = torch.from_numpy(bag["z_lab"]).double()
            obs = ObservedSpots(
                spot_id=torch.from_numpy(bag["spot_id"]).long(),
                ring_nr=ring_nr, y_lab=y_lab, z_lab=z_lab,
                omega=omega_rad, eta=eta_rad, two_theta=2.0*theta_rad,
                grain_radius=torch.full((S,), float(grain_radius_um[g]), dtype=torch.float64),
                fit_rmse=torch.zeros(S, dtype=torch.float64),
                y_orig=y_lab.clone(), z_orig=z_lab.clone(),
                omega_ini=omega_rad.clone(),
                mask_touched=torch.zeros(S, dtype=torch.bool),
            )
            eu_g = torch.from_numpy(grain_eulers_init[g][None, None, :]).double()
            po_g = torch.from_numpy(grain_pos_init[g][None, None, :]).double()
            la_g = torch.from_numpy(grain_lat_init[g][None, :]).double()
            pred = model(eu_g, po_g, lattice_params=la_g)
            mr = associate(
                obs_ring_nr=ring_nr, obs_omega=omega_rad, obs_eta=eta_rad,
                pred_ring_slot=pred_ring_slot,
                pred_omega=pred.omega.squeeze(0).squeeze(0).double(),
                pred_eta=pred.eta.squeeze(0).squeeze(0).double(),
                pred_valid=pred.valid.squeeze(0).squeeze(0).bool(),
                obs_ring_slot=ring_slot_lookup(obs_ring_nrs, ring_nr),
                omega_tolerance=math.radians(2.0), eta_tolerance=math.radians(3.0),
            )
        observations.append(obs)
        matches.append(mr)

    spec = build_joint_spec(
        powder_spec=spec,
        grain_eulers_init=torch.from_numpy(grain_eulers_init).double(),
        grain_positions_init=torch.from_numpy(grain_pos_init).double(),
        grain_lattices_init=torch.from_numpy(grain_lat_init).double(),
        refine_grain_orientation=True, refine_grain_position=True,
        refine_grain_strain=False,
    )
    # Apply Phase 3 MAP grain values
    if "grain_euler" in phase3_map:
        spec.set_init("grain_euler", torch.tensor(phase3_map["grain_euler"], dtype=torch.float64))
    if "grain_pos" in phase3_map:
        spec.set_init("grain_pos", torch.tensor(phase3_map["grain_pos"], dtype=torch.float64))
    spec.parameters["grain_euler"].bounds = (-2*math.pi, 2*math.pi)
    spec.parameters["grain_pos"].bounds = (-1000.0, 1000.0)
    spec.parameters["Lsd"].bounds = (float(phase3_map["Lsd"])-2000, float(phase3_map["Lsd"])+2000)
    spec.parameters["BC_y"].bounds = (float(phase3_map["BC_y"])-5, float(phase3_map["BC_y"])+5)
    spec.parameters["BC_z"].bounds = (float(phase3_map["BC_z"])-5, float(phase3_map["BC_z"])+5)
    spec.parameters["ty"].bounds = (float(phase3_map["ty"])-1, float(phase3_map["ty"])+1)
    spec.parameters["tz"].bounds = (float(phase3_map["tz"])-1, float(phase3_map["tz"])+1)

    bundle = HEDMResidualBundle(model=model, observations=observations, matches=matches, kind="pixel")

    def powder_fn(u):
        return pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, u,
            rho_d=u.get("RhoD", torch.tensor(float(v1.RhoD), dtype=torch.float64)),
            panel_layout=layout, panel_idx=fits.panel_idx,
            ring_idx=fits.ring_idx, ring_d_spacing_A=fits.ring_d_spacing_A,
        )

    def hedm_fn(u):
        return hedm_spot_residual(u, bundle)

    return spec, powder_fn, hedm_fn, layout, n_grains_total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase3-dir", type=Path, required=True)
    p.add_argument("--phase2-layer-dir", type=Path, required=True)
    p.add_argument("--paramstest-powder", type=Path, required=True)
    p.add_argument("--paramstest-hedm", type=Path, required=True)
    p.add_argument("--calibration-image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibrant", default="CeO2", choices=list(CALIBRANTS))
    p.add_argument("--w-powder", type=float, default=1.0e4)
    p.add_argument("--w-hedm", type=float, default=10.0)
    p.add_argument("--lambda-gauge", type=float, default=1.0e6)
    p.add_argument("--sigma-prior-px", type=float, default=0.5)
    p.add_argument("--n-grains-limit", type=int, default=0)
    args = p.parse_args(argv)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args.output.mkdir(parents=True, exist_ok=True)

    if not (args.phase3_dir / "phase3_joint_map.json").exists():
        print(f"   phase3_joint_map.json not found in {args.phase3_dir}; run Phase 3 first.")
        return 1

    print("=" * 70)
    print(" Phase 4 — Identifiability diagnostics at Phase 3 MAP")
    print("=" * 70)

    spec, powder_fn, hedm_fn, layout, n_grains = _build_setup(args)

    W_POWDER, W_HEDM = args.w_powder, args.w_hedm
    weights = JointWeights(w_powder=W_POWDER, w_hedm=W_HEDM, lambda_gauge=args.lambda_gauge)

    # Gauge-included closures (mirror what LM saw)
    def powder_only_g(u):
        return joint_residual(u, powder_residual_fn=powder_fn,
            hedm_residual_fn=lambda _u: torch.zeros(0, dtype=torch.float64),
            spec=spec, weights=weights, gauge_blocks=[])
    def hedm_only_g(u):
        return joint_residual(u, powder_residual_fn=lambda _u: torch.zeros(0, dtype=torch.float64),
            hedm_residual_fn=hedm_fn, spec=spec, weights=weights, gauge_blocks=[])
    def joint_g(u):
        return joint_residual(u, powder_residual_fn=powder_fn,
            hedm_residual_fn=hedm_fn, spec=spec, weights=weights, gauge_blocks=[])

    # Gauge-free closures (data-only identifiability — paper §9 claim)
    def powder_only(u): return W_POWDER * powder_fn(u)
    def hedm_only(u):   return W_HEDM * hedm_fn(u)
    def joint(u):       return torch.cat([W_POWDER * powder_fn(u), W_HEDM * hedm_fn(u)])

    # Evaluate at MAP (spec init = MAP after _build_setup)
    unpacked = {n: spec.parameters[n].init_tensor() for n in spec.parameters}

    # ----- 1. Per-panel σ_yz under three modalities (gauge-included)
    print("\n>> Fisher block rank on panel_delta_yz (gauge-INCLUDED, what LM sees)")
    sigma_reports = {}
    for label, fn in [("powder-only", powder_only_g), ("hedm-only", hedm_only_g), ("joint", joint_g)]:
        rep = fisher_block_rank(spec, fn, unpacked,
            block_names=["panel_delta_yz"], sigma_r=1.0, fallback_span=2.0)
        sigma_reports[label] = rep
        sig = rep.sigma_per_dim
        print(f"     {label:12s}  rank={rep.rank:3d}/{sig.numel():3d}  "
              f"cond={rep.condition_number:.2e}  "
              f"σ_max={float(sig.max()):.3e}  σ_med={float(sig.median()):.3e}")

    # Per-panel CSV: panel_id, sigma_y_<modality>, sigma_z_<modality>
    csv_path = args.output / "phase4_per_panel_sigma.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["panel_id"]
        for lbl in ("powder-only", "hedm-only", "joint"):
            cols += [f"sigma_y_{lbl}", f"sigma_z_{lbl}"]
        w.writerow(cols)
        n_p = layout.n_panels()
        sig_yz = {lbl: rep.sigma_per_dim.numpy().reshape(n_p, 2)
                  for lbl, rep in sigma_reports.items()}
        for pi in range(n_p):
            row = [pi]
            for lbl in ("powder-only", "hedm-only", "joint"):
                row += [float(sig_yz[lbl][pi, 0]), float(sig_yz[lbl][pi, 1])]
            w.writerow(row)
    print(f"   ✓ Saved {csv_path}")

    # ----- 2. Gauge-free Fisher (data-only)
    print("\n>> Fisher block rank on panel_delta_yz (gauge-FREE, data-only)")
    for label, fn in [("powder-only", powder_only), ("hedm-only", hedm_only), ("joint", joint)]:
        rep = fisher_block_rank(spec, fn, unpacked,
            block_names=["panel_delta_yz"], sigma_r=1.0, fallback_span=2.0)
        print(f"     {label:12s}  rank={rep.rank:3d}/{rep.sigma_per_dim.numel():3d}  "
              f"cond={rep.condition_number:.2e}")

    # ----- 3. (Lsd, Wavelength) gauge demo
    print("\n>> Gauge demo: (Lsd, Wavelength) — paper §9 rank-1 deficiency claim")
    gauge_rows = [["modality", "rank", "cond", "sigma_Lsd_um", "sigma_lambda_uA"]]
    for label, fn in [("powder-only", powder_only), ("hedm-only", hedm_only), ("joint", joint)]:
        try:
            rep = fisher_block_rank(spec, fn, unpacked,
                block_names=["Lsd", "Wavelength"], sigma_r=1.0, fallback_span=2.0)
            sig = rep.sigma_per_dim
            sig_lsd = float(sig[0])
            sig_lam = float(sig[1]) * 1e6  # Å → µÅ
            gauge_rows.append([label, rep.rank, f"{rep.condition_number:.3e}",
                               f"{sig_lsd:.3e}", f"{sig_lam:.3e}"])
            print(f"     {label:12s}  rank={rep.rank}/2  cond={rep.condition_number:.2e}  "
                  f"σ_Lsd={sig_lsd:.3e} µm  σ_λ={sig_lam:.3e} µÅ")
        except Exception as e:
            print(f"     {label:12s}  skip: {e}")
    gauge_csv = args.output / "phase4_gauge_demo.csv"
    with open(gauge_csv, "w", newline="") as f:
        csv.writer(f).writerows(gauge_rows)
    print(f"   ✓ Saved {gauge_csv}")

    # ----- 4. All-blocks Fisher table
    print("\n>> All-blocks Fisher (gauge-free)")
    logical = [
        ("Lsd",                   ["Lsd"]),
        ("BC_y",                  ["BC_y"]),
        ("BC_z",                  ["BC_z"]),
        ("Lsd+BC",                ["Lsd", "BC_y", "BC_z"]),
        ("panel_delta_yz",        ["panel_delta_yz"]),
        ("Lsd+BC+panel_delta_yz", ["Lsd", "BC_y", "BC_z", "panel_delta_yz"]),
    ]
    table_rows = [["block", "modality", "n_dim", "rank", "cond", "sigma_max", "sigma_med"]]
    refined_set = set(spec.refined_names())
    for label, names in logical:
        missing = [n for n in names if n not in refined_set]
        if missing:
            print(f"     skip {label!r}: unrefined dims {missing}")
            continue
        for modality, fn in [("powder-only", powder_only), ("hedm-only", hedm_only), ("joint", joint)]:
            try:
                rep = fisher_block_rank(spec, fn, unpacked, block_names=names,
                    sigma_r=1.0, fallback_span=2.0)
                sig = rep.sigma_per_dim
                table_rows.append([
                    label, modality, sig.numel(), rep.rank,
                    f"{rep.condition_number:.3e}",
                    f"{float(sig.max()):.3e}", f"{float(sig.median()):.3e}",
                ])
                print(f"   {label:25s} {modality:12s}  rank={rep.rank}/{sig.numel()}  "
                      f"cond={rep.condition_number:.2e}  σ_max={float(sig.max()):.3e}")
            except Exception as e:
                print(f"   {label:25s} {modality:12s}  err: {e}")
    table_csv = args.output / "phase4_all_blocks_fisher.csv"
    with open(table_csv, "w", newline="") as f:
        csv.writer(f).writerows(table_rows)
    print(f"   ✓ Saved {table_csv}")

    print("\n>> Phase 4 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
