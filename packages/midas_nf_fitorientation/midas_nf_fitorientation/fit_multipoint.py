"""``FitOrientationParametersMultiPoint`` replacement: joint multi-voxel
calibration.

Refines a single global calibration set (per-distance ``Lsd``,
``y_BC``/``z_BC``, sample-axis ``tx/ty/tz``, optional ``wedge``)
together with per-voxel orientations across up to 200 voxels. The
parameter vector has dimension ``3 + 3·nLayers + 3·nSpots`` plus 1 if
wedge is refined.

The C code uses a NM → CRS2 → NM → CRS2 → NM ladder repeated
``NumIterations`` times to escape local optima via genetic-style
global search. PyTorch L-BFGS is local-only, so we replace the global
phase with **multi-start L-BFGS**: the outer loop runs
``NumIterations`` independent L-BFGS attempts, each seeded with a
random perturbation of the previous best (within the tanh box). The
overall best is kept. For well-seeded calibration this matches the C
behaviour in practice; if the seed is far from optimum the user
should bump ``NumIterations``.

Three-phase schedule per multi-start trial:

1. Per-voxel Eulers only (independent, can be parallelised).
2. Calibration only, all voxels' Eulers fixed.
3. Joint refinement of everything.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .fit_kernel import LBFGSConfig, run_lbfgs
from .io import read_grid, read_hkls, read_mic_gridpoints, read_orientations
from .obs_volume import ObsVolume
from .params import FitParams, parse_paramfile
from .reparam import LsdEncoding, TanhBox
from .soft_overlap import (
    GeometryOverrides,
    auto_sigma_px,
    build_forward_model,
    soft_overlap,
    soft_overlap_loss,
)


def _multipoint_loss(
    model,
    obs,
    box_eulers: List[TanhBox],
    positions_um: torch.Tensor,
    sigma_px: float,
    geom_ov: GeometryOverrides,
) -> torch.Tensor:
    """Mean ``1 − soft_overlap`` across all voxels under one calibration.

    The forward model is called once per voxel because each voxel's
    Eulers are an independent leaf — batching across voxels is
    possible but would require a single shared (V, 3) Euler tensor,
    which is not how :class:`TanhBox` is structured. v0.1 keeps it
    simple; v0.2 can fuse if profiling demands.
    """
    losses = []
    for vi in range(positions_um.shape[0]):
        pos = positions_um[vi : vi + 1]                 # (1, 3)
        eul = box_eulers[vi].x.unsqueeze(0)             # (1, 3)
        overlap = soft_overlap(model, obs, eul, pos, sigma_px, geom_ov)
        losses.append(1.0 - overlap)
    return torch.stack(losses).mean()


def fit_multipoint_run(
    paramfile: str,
    n_cpus: int = 1,
    *,
    device: str = "auto",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    seed: int = 0,
    lbfgs_config: Optional[LBFGSConfig] = None,
) -> dict:
    """Joint multi-voxel calibration. Replaces
    :program:`FitOrientationParametersMultiPoint`.

    Voxels and seed Eulers come from the paramfile's ``GridPoints``
    block (the C code's existing convention).
    """
    p = parse_paramfile(paramfile)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    out_dir = Path(p.out_dir)
    grid_points = list(p.grid_points)
    if not grid_points:
        # No explicit GridPoints block: derive them from the
        # reconstructed text .mic (the C GridPoints columns are exactly
        # a .mic row), keeping the highest-confidence voxels up to the
        # C cap of 200. Mirrors the convention in
        # FitOrientationParametersMultiPoint.c.
        if not p.mic_file_text:
            raise ValueError(
                "paramfile has no GridPoints entries and no MicFileText "
                "to derive them from. Add a GridPoints block or set "
                "MicFileText to a reconstructed .mic."
            )
        mic_path = out_dir / p.mic_file_text
        if not mic_path.exists():
            raise FileNotFoundError(
                f"paramfile has no GridPoints entries and MicFileText "
                f"{mic_path} does not exist. Run the reconstruction first "
                f"or add an explicit GridPoints block."
            )
        grid_points = read_mic_gridpoints(
            mic_path, min_confidence=p.min_confidence, max_points=200,
        )
        if not grid_points:
            raise ValueError(
                f"no voxels in {mic_path} passed MinConfidence="
                f"{p.min_confidence}; lower MinConfidence or add an "
                f"explicit GridPoints block."
            )
        if verbose:
            print(f"No GridPoints block; derived {len(grid_points)} voxels "
                  f"from {mic_path.name} (MinConfidence={p.min_confidence}).")

    hkl_table = read_hkls(out_dir)
    if p.rings_to_use:
        hkl_table = hkl_table.filter_rings(p.rings_to_use)

    grid_unused = read_grid(out_dir, p.grid_file_name)  # only for sanity

    obs = ObsVolume.from_spotsinfo(
        out_dir / "SpotsInfo.bin",
        n_distances=p.n_distances,
        n_frames=p.n_frames_per_distance,
        n_y=p.n_pixels_y, n_z=p.n_pixels_z,
        device=torch_device, dtype=torch.float32,
        packed=False,                       # dense float for the soft path
    )
    model = build_forward_model(
        p, hkl_table.hkls_int.astype(np.float64),
        device=torch_device, dtype=dtype,
        hkls_cart=hkl_table.hkls_cart.astype(np.float64),
    )

    n_spots = len(grid_points)
    if verbose:
        print(f"Multipoint calibration: {n_spots} voxels, "
              f"{p.n_distances} distances, "
              f"{'wedge ON' if p.refine_wedge else 'wedge OFF'}")

    # Per-voxel positions (centroids) and Euler seeds from GridPoints.
    positions_np = np.zeros((n_spots, 3), dtype=np.float64)
    seed_eulers_np = np.zeros((n_spots, 3), dtype=np.float64)
    for i, (xc, yc, _ud, e1, e2, e3) in enumerate(grid_points):
        positions_np[i] = (xc, yc, 0.0)
        seed_eulers_np[i] = (e1, e2, e3)
    positions_um = torch.tensor(positions_np, device=torch_device, dtype=dtype)
    sigma_px = auto_sigma_px(p.grid_size_um / 2.0, p.px,
                              p.gaussian_splat_sigma_px)

    cfg = lbfgs_config or LBFGSConfig()

    Lsds = torch.tensor(p.Lsd, device=torch_device, dtype=dtype)
    enc0 = LsdEncoding.from_lsds(Lsds)
    tilts0 = torch.tensor(
        [[p.tx, p.ty, p.tz]] * p.n_distances,
        device=torch_device, dtype=dtype,
    )

    # Globals tracked across multi-start trials.
    best_overall_loss = float("inf")
    best_overall_state: dict = {}

    rng = torch.Generator(device="cpu").manual_seed(seed)

    n_trials = max(1, p.num_iterations)
    for trial in range(n_trials):
        if verbose:
            print(f"\n--- Multi-start trial {trial+1}/{n_trials} ---")

        # Per-voxel Euler boxes
        tol_rad = p.orient_tol * math.pi / 180.0
        box_eulers = [
            TanhBox(
                torch.tensor(seed_eulers_np[i], device=torch_device, dtype=dtype),
                tol_rad,
            )
            for i in range(n_spots)
        ]
        # Calibration boxes
        box_lsd0 = TanhBox(enc0.Lsd0, p.lsd_tol)
        box_lsd_delta = (
            TanhBox(enc0.deltas, p.lsd_rel_tol)
            if enc0.deltas.numel() > 0 else None
        )
        box_ybc = TanhBox(
            torch.tensor(p.ybc, device=torch_device, dtype=dtype),
            p.bc_tol_a,
        )
        box_zbc = TanhBox(
            torch.tensor(p.zbc, device=torch_device, dtype=dtype),
            p.bc_tol_b,
        )
        box_tilts = TanhBox(tilts0, p.tilts_tol)
        box_wedge = (
            TanhBox(
                torch.tensor(p.wedge, device=torch_device, dtype=dtype),
                p.wedge_tol,
            )
            if p.refine_wedge else None
        )

        # Trials > 0: perturb the unbounded leaves so we sample a
        # different basin. Scale 0.3 = 30% of one tolerance width.
        if trial > 0:
            for be in box_eulers:
                be.perturb(0.3, generator=rng)
            box_lsd0.perturb(0.3, generator=rng)
            if box_lsd_delta is not None:
                box_lsd_delta.perturb(0.3, generator=rng)
            box_ybc.perturb(0.3, generator=rng)
            box_zbc.perturb(0.3, generator=rng)
            box_tilts.perturb(0.3, generator=rng)
            if box_wedge is not None:
                box_wedge.perturb(0.3, generator=rng)

        def make_geom_ov() -> GeometryOverrides:
            if box_lsd_delta is None:
                Lsd = box_lsd0.x
            else:
                Lsd = LsdEncoding(box_lsd0.x, box_lsd_delta.x).decode()
            return GeometryOverrides(
                Lsd=Lsd,
                y_BC=box_ybc.x,
                z_BC=box_zbc.x,
                tilts=box_tilts.x,
                wedge=box_wedge.x if box_wedge is not None else None,
            )

        def tikhonov_terms():
            if p.tikhonov_calibration <= 0.0:
                return []
            lam = p.tikhonov_calibration
            terms = [
                box_lsd0.tikhonov(p.tikhonov_sigma_lsd, lam),
                box_ybc.tikhonov(p.tikhonov_sigma_bc, lam),
                box_zbc.tikhonov(p.tikhonov_sigma_bc, lam),
                box_tilts.tikhonov(p.tikhonov_sigma_tilts, lam),
            ]
            if box_lsd_delta is not None:
                terms.append(box_lsd_delta.tikhonov(p.tikhonov_sigma_lsd, lam))
            if box_wedge is not None:
                terms.append(box_wedge.tikhonov(p.tikhonov_sigma_wedge, lam))
            return terms

        # ---- Phase 1: Eulers only (per voxel, independent) ----
        for vi, be in enumerate(box_eulers):
            ov_fixed = make_geom_ov()
            # Detach calibration so Phase 1's gradient is Eulers-only.
            ov_fixed = GeometryOverrides(
                Lsd=ov_fixed.Lsd.detach() if ov_fixed.Lsd is not None else None,
                y_BC=ov_fixed.y_BC.detach() if ov_fixed.y_BC is not None else None,
                z_BC=ov_fixed.z_BC.detach() if ov_fixed.z_BC is not None else None,
                tilts=ov_fixed.tilts.detach() if ov_fixed.tilts is not None else None,
                wedge=ov_fixed.wedge.detach() if ov_fixed.wedge is not None else None,
            )

            def closure_eul(be=be, vi=vi, ov_fixed=ov_fixed):
                be.u.grad = None
                pos = positions_um[vi : vi + 1]
                loss = soft_overlap_loss(
                    model, obs, be.x, pos, sigma_px, ov_fixed,
                )
                loss.backward()
                return loss

            run_lbfgs(closure_eul, [be.u], cfg)

        # ---- Phase 2: Calibration only ----
        calib_leaves = [box_lsd0.u, box_ybc.u, box_zbc.u, box_tilts.u]
        if box_lsd_delta is not None:
            calib_leaves.append(box_lsd_delta.u)
        if box_wedge is not None:
            calib_leaves.append(box_wedge.u)

        # Detach Eulers for Phase 2
        eulers_fixed = [be.x.detach().clone() for be in box_eulers]

        def closure_calib():
            for leaf in calib_leaves:
                leaf.grad = None
            ov = make_geom_ov()
            losses = []
            for vi in range(n_spots):
                pos = positions_um[vi : vi + 1]
                eul = eulers_fixed[vi].unsqueeze(0)
                overlap = soft_overlap(
                    model, obs, eul, pos, sigma_px, ov,
                )
                losses.append(1.0 - overlap)
            loss = torch.stack(losses).mean()
            for t in tikhonov_terms():
                loss = loss + t
            loss.backward()
            return loss

        run_lbfgs(closure_calib, calib_leaves, cfg)

        # ---- Phase 3: Joint ----
        joint_leaves = calib_leaves + [be.u for be in box_eulers]

        def closure_joint():
            for leaf in joint_leaves:
                leaf.grad = None
            ov = make_geom_ov()
            losses = []
            for vi in range(n_spots):
                pos = positions_um[vi : vi + 1]
                eul = box_eulers[vi].x.unsqueeze(0)
                overlap = soft_overlap(
                    model, obs, eul, pos, sigma_px, ov,
                )
                losses.append(1.0 - overlap)
            loss = torch.stack(losses).mean()
            for t in tikhonov_terms():
                loss = loss + t
            loss.backward()
            return loss

        cfg_joint = LBFGSConfig(
            lr=cfg.lr * 0.5, max_iter=cfg.max_iter, max_outer=cfg.max_outer,
            stop_loss=cfg.stop_loss,
        )
        t0 = time.perf_counter()
        res = run_lbfgs(closure_joint, joint_leaves, cfg_joint)
        trial_secs = time.perf_counter() - t0

        if verbose:
            print(f"Trial {trial+1}: final loss={res.final_loss:.6f} "
                  f"({trial_secs:.1f} s)")

        if res.final_loss < best_overall_loss:
            best_overall_loss = res.final_loss
            ov = make_geom_ov()
            best_overall_state = {
                "trial": trial,
                "loss": res.final_loss,
                "Lsd": ov.Lsd.detach().cpu().numpy().tolist(),
                "y_BC": ov.y_BC.detach().cpu().numpy().tolist(),
                "z_BC": ov.z_BC.detach().cpu().numpy().tolist(),
                "tilts": ov.tilts.detach().cpu().numpy().tolist(),
                "wedge": (
                    float(ov.wedge.detach())
                    if ov.wedge is not None else None
                ),
                "voxel_eulers_rad": [
                    be.x.detach().cpu().numpy().tolist()
                    for be in box_eulers
                ],
                "voxel_individual_overlaps": [
                    float(soft_overlap(
                        model, obs,
                        be.x.detach().unsqueeze(0),
                        positions_um[i : i + 1],
                        sigma_px, ov,
                    )) for i, be in enumerate(box_eulers)
                ],
            }

    if verbose:
        print(f"\nBest result from trial {best_overall_state['trial']+1}: "
              f"avg overlap = {1.0 - best_overall_state['loss']:.6f}")
        for d in range(p.n_distances):
            print(f"Layer {d}: Lsd={best_overall_state['Lsd'][d]:.4f}, "
                  f"BC=({best_overall_state['y_BC'][d]:.4f}, "
                  f"{best_overall_state['z_BC'][d]:.4f})")
        for d in range(p.n_distances):
            print(f"Tilts[{d}]: tx={best_overall_state['tilts'][d][0]:.4f}, "
                  f"ty={best_overall_state['tilts'][d][1]:.4f}, "
                  f"tz={best_overall_state['tilts'][d][2]:.4f}")
        if best_overall_state.get("wedge") is not None:
            print(f"Wedge: {best_overall_state['wedge']:.4f}")

    return best_overall_state
