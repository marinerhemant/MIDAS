"""``FitOrientationParameters`` replacement: single-voxel calibration.

Refines orientation jointly with detector geometry (per-distance Lsd,
per-distance ``y_BC``/``z_BC``, sample-axis tilts ``tx/ty/tz``, and
optionally ``wedge``) at one voxel position. Used by the calibration
pipelines, not by per-voxel orientation reconstruction.

Two phases:

1. Eulers-only L-BFGS (matches the C 3-DoF first phase's basin).
2. Joint Eulers + calibration L-BFGS, all parameters reparameterised
   via :class:`TanhBox`. Optional Tikhonov pull on the calibration
   block via the ``TikhonovCalibration`` paramfile knob.

The C code uses one Nelder-Mead phase over all 6+3·nLayers parameters.
We split into two phases because L-BFGS is more sensitive to bad
initial steps than NM, and orientation gradients can be noisy when
calibration is mis-set; the staging avoids the calibration parameters
drifting badly during the orientation-recovery phase.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .fit_kernel import LBFGSConfig, run_lbfgs
from .io import read_grid, read_hkls, read_orientations
from .obs_volume import ObsVolume
from .params import FitParams, parse_paramfile
from .reparam import LsdEncoding, TanhBox
from .screen import orientmat_to_euler_zxz, screen
from .soft_overlap import (
    GeometryOverrides,
    auto_sigma_px,
    build_forward_model,
    soft_overlap_loss,
)


def fit_parameters_run(
    paramfile: str,
    voxel_idx: int,
    n_cpus: int = 1,
    *,
    device: str = "auto",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    lbfgs_config: Optional[LBFGSConfig] = None,
) -> dict:
    """Run ``FitOrientationParameters`` on a single voxel.

    Parameters
    ----------
    paramfile : str
        Path to the C-style paramfile.
    voxel_idx : int
        1-based voxel index, matching the C executable's ``argv[2]``.
    n_cpus : int
        Currently informational; PyTorch's intra-op pool is used.

    Returns
    -------
    dict
        Best Euler / Lsd / BC / tilts / wedge / fraction-overlap.
    """
    p = parse_paramfile(paramfile)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    out_dir = Path(p.out_dir)

    grid = read_grid(out_dir, p.grid_file_name)
    orientations = read_orientations(out_dir)
    hkl_table = read_hkls(out_dir)
    if p.rings_to_use:
        hkl_table = hkl_table.filter_rings(p.rings_to_use)

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

    # 1-based → 0-based index, matching the C convention.
    vi = voxel_idx - 1
    if vi < 0 or vi >= grid.n_voxels:
        raise IndexError(
            f"voxel index {voxel_idx} outside grid [1, {grid.n_voxels}]"
        )
    xs = float(grid.xs[vi])
    ys = float(grid.ys[vi])
    gs = float(grid.gs[vi])
    sigma_px = auto_sigma_px(gs, p.px, p.gaussian_splat_sigma_px)
    pos_um = torch.tensor([xs, ys, 0.0], device=torch_device, dtype=dtype)

    if verbose:
        print(f"Fitting voxel {voxel_idx} at (xs={xs:.3f}, ys={ys:.3f})")

    # ---- screen this single voxel ----
    screen_result = screen(
        grid, orientations, obs, p,
        voxel_indices=np.array([vi]),
        dtype=dtype,
    )
    n_winners = len(screen_result.winners)
    if n_winners == 0:
        raise RuntimeError(
            f"voxel {voxel_idx}: no orientations passed MinFracAccept"
        )
    if verbose:
        print(f"Screen produced {n_winners} candidate orientations")

    eulers_seed = orientmat_to_euler_zxz(orientations.matrices)

    cfg = lbfgs_config or LBFGSConfig()

    # ---- Phase 1: Eulers only, per candidate. Best becomes seed for Phase 2.
    best_frac = -1.0
    best_eul = None
    best_orient_idx = -1
    for w in screen_result.winners:
        euler_seed = torch.tensor(
            eulers_seed[w.orient_idx], device=torch_device, dtype=dtype,
        )
        tol_rad = p.orient_tol * math.pi / 180.0
        box_eul = TanhBox(euler_seed, tol_rad)

        def closure_eul():
            box_eul.u.grad = None
            loss = soft_overlap_loss(
                model, obs, box_eul.x, pos_um, sigma_px,
            )
            loss.backward()
            return loss

        res = run_lbfgs(closure_eul, [box_eul.u], cfg)
        frac = max(0.0, 1.0 - res.final_loss)
        if frac > best_frac:
            best_frac = frac
            best_eul = box_eul.x.detach().clone()
            best_orient_idx = w.orient_idx

    if verbose:
        print(f"Phase 1 best: orient {best_orient_idx}, frac {best_frac:.4f}")

    # ---- Phase 2: joint Eulers + calibration ----
    Lsds = torch.tensor(p.Lsd, device=torch_device, dtype=dtype)
    enc0 = LsdEncoding.from_lsds(Lsds)

    box_eul = TanhBox(best_eul, p.orient_tol * math.pi / 180.0)
    box_lsd0 = TanhBox(enc0.Lsd0, p.lsd_tol)
    if enc0.deltas.numel() > 0:
        box_lsd_delta = TanhBox(enc0.deltas, p.lsd_rel_tol)
    else:
        box_lsd_delta = None

    box_ybc = TanhBox(
        torch.tensor(p.ybc, device=torch_device, dtype=dtype),
        p.bc_tol_a,
    )
    box_zbc = TanhBox(
        torch.tensor(p.zbc, device=torch_device, dtype=dtype),
        p.bc_tol_b,
    )
    tilts0 = torch.tensor(
        [[p.tx, p.ty, p.tz]] * p.n_distances,
        device=torch_device, dtype=dtype,
    )
    box_tilts = TanhBox(tilts0, p.tilts_tol)
    if p.refine_wedge:
        wedge0 = torch.tensor(p.wedge, device=torch_device, dtype=dtype)
        box_wedge = TanhBox(wedge0, p.wedge_tol)
    else:
        box_wedge = None

    leaves = [
        box_eul.u, box_lsd0.u, box_ybc.u, box_zbc.u, box_tilts.u,
    ]
    if box_lsd_delta is not None:
        leaves.append(box_lsd_delta.u)
    if box_wedge is not None:
        leaves.append(box_wedge.u)

    def closure_joint():
        for leaf in leaves:
            leaf.grad = None

        if box_lsd_delta is None:
            Lsd = box_lsd0.x
        else:
            Lsd = LsdEncoding(box_lsd0.x, box_lsd_delta.x).decode()
        ov = GeometryOverrides(
            Lsd=Lsd,
            y_BC=box_ybc.x,
            z_BC=box_zbc.x,
            tilts=box_tilts.x,
            wedge=box_wedge.x if box_wedge is not None else None,
        )

        tikhonov_terms = []
        if p.tikhonov_calibration > 0.0:
            lam = p.tikhonov_calibration
            tikhonov_terms.append(
                box_lsd0.tikhonov(p.tikhonov_sigma_lsd, lam)
            )
            if box_lsd_delta is not None:
                tikhonov_terms.append(
                    box_lsd_delta.tikhonov(p.tikhonov_sigma_lsd, lam)
                )
            tikhonov_terms.append(box_ybc.tikhonov(p.tikhonov_sigma_bc, lam))
            tikhonov_terms.append(box_zbc.tikhonov(p.tikhonov_sigma_bc, lam))
            tikhonov_terms.append(
                box_tilts.tikhonov(p.tikhonov_sigma_tilts, lam)
            )
            if box_wedge is not None:
                tikhonov_terms.append(
                    box_wedge.tikhonov(p.tikhonov_sigma_wedge, lam)
                )

        loss = soft_overlap_loss(
            model, obs, box_eul.x, pos_um, sigma_px,
            geom_ov=ov, tikhonov_terms=tikhonov_terms,
        )
        loss.backward()
        return loss

    cfg_joint = LBFGSConfig(
        lr=cfg.lr * 0.5, max_iter=cfg.max_iter, max_outer=cfg.max_outer,
        stop_loss=cfg.stop_loss,
    )
    t0 = time.perf_counter()
    res = run_lbfgs(closure_joint, leaves, cfg_joint)
    fit_secs = time.perf_counter() - t0
    final_frac = max(0.0, 1.0 - res.final_loss)

    final_eul = box_eul.x.detach()
    final_lsd = (
        box_lsd0.x.detach()
        if box_lsd_delta is None
        else LsdEncoding(box_lsd0.x, box_lsd_delta.x).decode().detach()
    )
    final_ybc = box_ybc.x.detach()
    final_zbc = box_zbc.x.detach()
    final_tilts = box_tilts.x.detach()
    final_wedge = box_wedge.x.detach() if box_wedge is not None else None

    result = {
        "voxel_idx": voxel_idx,
        "n_winners": n_winners,
        "frac_overlap": final_frac,
        "fit_seconds": fit_secs,
        "euler_rad": final_eul.cpu().numpy().tolist(),
        "Lsd": final_lsd.cpu().numpy().tolist(),
        "y_BC": final_ybc.cpu().numpy().tolist(),
        "z_BC": final_zbc.cpu().numpy().tolist(),
        "tilts": final_tilts.cpu().numpy().tolist(),
        "wedge": (
            float(final_wedge) if final_wedge is not None else None
        ),
    }

    if verbose:
        print(f"\nFinal Best Result")
        print(f"Euler angles: {result['euler_rad']}, Confidence: {final_frac:.6f}")
        print(f"Tilts: {result['tilts']}")
        for d in range(p.n_distances):
            print(f"Layer {d}: Lsd={result['Lsd'][d]:.4f}, "
                  f"BCs=({result['y_BC'][d]:.4f}, {result['z_BC'][d]:.4f})")
        if final_wedge is not None:
            print(f"Wedge: {float(final_wedge):.6f}")

    return result
