"""midas-grain-odf CLI.

Usage
-----
    midas-grain-odf fit \\
        --geometry geometry.json \\
        --grains grains.csv \\
        --spots spots.csv \\
        --frames frames.npy \\
        --output results.h5 \\
        [--odf-type particle|bingham|voxel] \\
        [--K 64] \\
        [--theta-max-deg 1.0] \\
        [--patch-P 31] \\
        [--patch-F 7] \\
        [--grain-id ID]    # restrict to one grain (for debugging)

The defaults are tuned for synthetic / small-spread data; expect to raise
``patch-P`` to 60–100 for real-data spreads beyond ~0.3°.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

from midas_grain_odf.forward_helpers import HEDMForwardModel
from midas_diffract.forward import HEDMGeometry  # noqa: F401 -- imported for type docs
from midas_grain_odf.inversion import fit_grain_odf
from midas_grain_odf.io import (
    GrainOdfConfig,
    GrainRow,
    build_spot_indexer,
    crop_patches_from_frames,
    enumerate_hkls,
    load_frames,
    load_grains_csv,
    load_spots_csv,
    write_odf_results_h5,
)
from midas_grain_odf.odf import BinghamMixtureODF, ParticleODF, VoxelGridODF


def _build_model(cfg: GrainOdfConfig):
    """Construct an HEDMForwardModel from a GrainOdfConfig."""
    from midas_diffract.forward import HEDMGeometry as _HG
    geometry = _HG(
        Lsd=cfg.Lsd, y_BC=cfg.y_BC, z_BC=cfg.z_BC, px=cfg.px,
        omega_start=cfg.omega_start, omega_step=cfg.omega_step,
        n_frames=cfg.n_frames,
        n_pixels_y=cfg.n_pixels_y, n_pixels_z=cfg.n_pixels_z,
        min_eta=cfg.min_eta, wavelength=cfg.wavelength,
        flip_y=True,
    )
    G, th, intt = enumerate_hkls(cfg)
    model = HEDMForwardModel(
        hkls=G, thetas=th, geometry=geometry, hkls_int=intt,
        device=torch.device("cpu"),
        compile=getattr(cfg, "compile", False),
    )
    return model.to(torch.float64)


def _make_odf(odf_type: str, R_avg, K: int, theta_max: float):
    if odf_type == "particle":
        return ParticleODF(R_avg=R_avg, K=K, theta_max=theta_max,
                           seed=0).to(torch.float64)
    if odf_type == "bingham":
        n_modes = max(1, K // 32)
        return BinghamMixtureODF(R_avg=R_avg, n_modes=n_modes,
                                  K_per_mode=max(8, K // n_modes),
                                  theta_max=theta_max,
                                  seed=0).to(torch.float64)
    if odf_type == "voxel":
        n_per_axis = max(3, int(round(K ** (1.0 / 3.0))))
        return VoxelGridODF(R_avg=R_avg, n_per_axis=n_per_axis,
                             theta_max=theta_max).to(torch.float64)
    raise ValueError(f"unknown odf-type: {odf_type!r}")


def cmd_fit(args: argparse.Namespace) -> int:
    cfg = GrainOdfConfig.from_json(args.geometry)
    model = _build_model(cfg)
    grains = load_grains_csv(args.grains)
    if args.grain_id is not None:
        grains = [g for g in grains if g.grain_id == args.grain_id]
        if not grains:
            print(f"grain_id {args.grain_id} not found in {args.grains}")
            return 2
    spots_df = load_spots_csv(args.spots)
    frames = load_frames(args.frames)
    print(f"Loaded {len(grains)} grain(s), {len(spots_df)} spot rows, "
          f"frames shape {frames.shape}")

    indexer = build_spot_indexer(grains, spots_df, model.hkls_int)
    print(f"Built per-grain indexers for {len(indexer)} grain(s)")

    theta_max = math.radians(args.theta_max_deg)
    results = {}
    for grain in grains:
        if grain.grain_id not in indexer:
            print(f"grain {grain.grain_id}: no matched spots, skipping")
            continue
        gx = indexer[grain.grain_id]
        S = int(gx["spot_indexer"].numel())
        print(f"\nFitting grain {grain.grain_id}  ({S} spots)")

        # Crop patches around measured spot anchors (which is where Stage-1
        # Delta lands the patch). Out-of-bound regions zero-padded.
        patches = crop_patches_from_frames(
            frames, gx["meas_y"], gx["meas_z"], gx["meas_f"],
            patch_F=args.patch_F, patch_P=args.patch_P,
        )

        odf = _make_odf(args.odf_type, grain.R_avg, args.K, theta_max)
        t0 = time.time()
        result = fit_grain_odf(
            odf=odf, model=model, position=grain.position,
            measured_y=gx["meas_y"], measured_z=gx["meas_z"],
            measured_f=gx["meas_f"],
            measured_patches=patches,
            spot_indexer=gx["spot_indexer"],
            lattice_params=grain.lattice,
            patch_F=args.patch_F, patch_P=args.patch_P,
            sigma_yz=args.sigma_yz, sigma_f=args.sigma_f,
            delta_iters=args.delta_iters,
            inner_steps=args.inner_steps,
            lr_axis_angle=args.lr_axis_angle,
            lr_logits=args.lr_logits,
            verbose=args.verbose,
        )
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s  initial loss = {result.losses[0]:.3e}  "
              f"final loss = {result.losses[-1]:.3e}  "
              f"converged = {result.converged}")

        R_rec, w_rec = result.odf.sample()
        results[grain.grain_id] = {
            "R_samples": R_rec.detach(),
            "weights": w_rec.detach(),
            "delta_y": result.delta_y,
            "delta_z": result.delta_z,
            "delta_f": result.delta_f,
            "keep": result.keep.to(torch.bool),
            "loss_history": torch.tensor(result.losses, dtype=torch.float64),
            "odf_type": args.odf_type,
            "converged": bool(result.converged),
            "delta_iters_run": int(result.delta_iters_run),
        }

    if not results:
        print("no grains were fit; nothing to write")
        return 1

    write_odf_results_h5(results, args.output)
    print(f"\nWrote results for {len(results)} grain(s) to {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midas-grain-odf",
        description="Per-grain ODF inversion from FF-HEDM data.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="Fit per-grain ODFs.")
    fit.add_argument("--geometry", required=True,
                     help="JSON file with detector geometry + lattice.")
    fit.add_argument("--grains", required=True, help="grains.csv path")
    fit.add_argument("--spots", required=True, help="spots.csv path")
    fit.add_argument("--frames", required=True,
                     help="frame stack (.npy / .npz / .h5)")
    fit.add_argument("--output", required=True, help="output .h5 path")
    fit.add_argument("--odf-type", default="particle",
                     choices=["particle", "bingham", "voxel"])
    fit.add_argument("--K", type=int, default=64,
                     help="ODF capacity (samples / modes / grid size).")
    fit.add_argument("--theta-max-deg", type=float, default=1.0,
                     help="Trust-region half-width in degrees.")
    fit.add_argument("--patch-F", type=int, default=7)
    fit.add_argument("--patch-P", type=int, default=31)
    fit.add_argument("--sigma-yz", type=float, default=1.0)
    fit.add_argument("--sigma-f", type=float, default=0.6)
    fit.add_argument("--delta-iters", type=int, default=2)
    fit.add_argument("--inner-steps", type=int, default=400)
    fit.add_argument("--lr-axis-angle", type=float, default=1e-4)
    fit.add_argument("--lr-logits", type=float, default=0.1)
    fit.add_argument("--grain-id", type=int, default=None,
                     help="Run only this grain (for debugging).")
    fit.add_argument("--verbose", action="store_true")
    fit.set_defaults(func=cmd_fit)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
