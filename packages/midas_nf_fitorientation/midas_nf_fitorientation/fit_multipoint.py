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
    overrides,
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
    # ONE shared tilt triple, not one per distance.
    #
    # tx/ty/tz describe how the detector is MOUNTED. In a multi-distance NF scan
    # the same physical detector is translated along the beam, so the tilts are a
    # property of the detector, not of the position -- there is one set, shared.
    #
    # This used to be [[tx,ty,tz]] * n_distances wrapped in a single TanhBox,
    # which gave every distance its own free tilts. That is unphysical and it
    # showed: a refinement returned Tilts[0] != Tilts[1] for one detector at two
    # DetZ positions. It also inflates the parameter count by 3*(nD-1) against a
    # dataset that cannot constrain them separately.
    #
    # Kept as a (3,) leaf and broadcast to (n_distances, 3) at use, so the
    # optimiser sees three tilt parameters total regardless of distance count.
    tilts0 = torch.tensor(
        [p.tx, p.ty, p.tz], device=torch_device, dtype=dtype,
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
            # box_tilts.x is the shared (3,) triple; the geometry wants one row
            # per distance. expand() is a view, so the gradient from every
            # distance accumulates back into the SAME three leaves -- which is
            # exactly the shared-tilt constraint.
            tilts_shared = box_tilts.x.unsqueeze(0).expand(p.n_distances, 3)
            return GeometryOverrides(
                Lsd=Lsd,
                y_BC=box_ybc.x,
                z_BC=box_zbc.x,
                tilts=tilts_shared,
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

        # ---- Baseline: the objective BEFORE any optimisation --------------
        # Without this there is no way to tell an improvement from noise. On
        # trial 0 the boxes are unperturbed, so this is the loss at the SEED
        # geometry exactly -- the number that says whether the objective can
        # even see a geometry you already believe in. (A seed loss of ~1.0, i.e.
        # ~0 overlap, at a geometry known to give hard confidence ~1.0 means the
        # objective is not tracking the reconstruction and the whole refinement
        # is optimising noise.)
        with torch.no_grad():
            trial_start_loss = float(
                _multipoint_loss(model, obs, box_eulers, positions_um,
                                 sigma_px, make_geom_ov())
            )
        if trial == 0:
            seed_loss = trial_start_loss
            if verbose:
                print(f"  SEED geometry loss = {seed_loss:.6f} "
                      f"(overlap {1.0 - seed_loss:.6f})")
                if seed_loss > 0.99:
                    print("  WARNING: the seed geometry scores ~zero overlap. "
                          "If it is known-good, the objective is not tracking "
                          "the data (check sigma_px / GridSize) and any "
                          "'improvement' below is noise.")
        elif verbose:
            print(f"  start loss = {trial_start_loss:.6f}")

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
            gain = trial_start_loss - res.final_loss
            print(f"Trial {trial+1}: {trial_start_loss:.6f} -> "
                  f"{res.final_loss:.6f} (improved {gain:+.6f}) "
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
        # State the gain against the seed, not just the final value: a good
        # absolute number with no gain means the seed was already there, and a
        # tiny gain on a near-1.0 loss means the objective saw nothing.
        _gain = seed_loss - best_overall_state['loss']
        print(f"  vs SEED: loss {seed_loss:.6f} -> "
              f"{best_overall_state['loss']:.6f} (improved {_gain:+.6f})")
        if seed_loss > 0.99:
            print("  DO NOT ADOPT THIS GEOMETRY without an independent check: "
                  "the seed scored ~zero overlap, so the objective is not "
                  "tracking the data on this dataset.")
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


# ---------------------------------------------------------------------------
#  Hard-FracOverlap multipoint calibration -- TRUE equivalent of the C
# ---------------------------------------------------------------------------

def fit_multipoint_hard_run(
    paramfile: str,
    n_cpus: int = 1,
    *,
    device: str = "auto",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    seed: int = 0,
    max_iter: int = 20000,
    global_iters: int = 40,
) -> dict:
    """Joint multi-voxel calibration against the HARD FracOverlap.

    This is the true equivalent of ``FitOrientationParametersMultiPoint``, not a
    differentiable approximation of it. The C objective is
    (FitOrientationParametersMultiPoint.c:140-176)::

        netResult += FracOverlap      # per GridPoint
        netResult /= nSpots           # mean
        return 1 - netResult          # minimise

    and it is optimised with derivative-free NLopt (NELDERMEAD / CRS2_LM), so
    nothing ever required the objective to be differentiable. The soft
    Gaussian-splat surrogate used by :func:`fit_multipoint_run` is a *different*
    objective, and on real data it under-reported badly enough to be unusable --
    0.055 where the C's hard confidence was 0.852 on the same inputs.

    Optimising the reported quantity directly removes the surrogate gap entirely
    and makes the number here comparable, value for value, with the C's
    "Original val" / "Final value".

    Parameter vector, laid out exactly as the C's ``x``
    (FitOrientationParametersMultiPoint.c:127-139)::

        x[0:3]                        tx, ty, tz          (SHARED across layers)
        x[3]                          Lsd[0]
        x[3+i]        i=1..nL-1       Lsd[i] = Lsd[i-1] + x[3+i]   (cumulative)
        x[3+nL : 3+2nL]               ybc per layer
        x[3+2nL : 3+3nL]              zbc per layer
        x[3+3nL + 3i : +3]            Eulers of voxel i

    It also uses the PACKED obs volume, so it needs ~1 bit/pixel instead of the
    dense float32 the soft path requires (1.9 GB vs 56 GiB on a 3600-frame scan).
    """
    from scipy.optimize import minimize

    p = parse_paramfile(paramfile)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    out_dir = Path(p.out_dir)

    grid_points = list(p.grid_points)
    if not grid_points:
        mic_path = out_dir / p.mic_file_text
        grid_points = read_mic_gridpoints(
            mic_path, min_confidence=p.min_confidence, max_points=200,
        )
    n_spots = len(grid_points)

    hkl_table = read_hkls(out_dir)
    if p.rings_to_use:
        hkl_table = hkl_table.filter_rings(p.rings_to_use)

    # PACKED: the hard path reads bits, so no dense float volume is needed.
    obs = ObsVolume.from_spotsinfo(
        out_dir / "SpotsInfo.bin",
        n_distances=p.n_distances,
        n_frames=p.n_frames_per_distance,
        n_y=p.n_pixels_y, n_z=p.n_pixels_z,
        device=torch_device, dtype=torch.uint8, packed=True,
    )
    model = build_forward_model(
        p, hkl_table.hkls_int.astype(np.float64),
        device=torch_device, dtype=dtype,
        hkls_cart=hkl_table.hkls_cart.astype(np.float64),
    )

    positions_np = np.zeros((n_spots, 3), dtype=np.float64)
    seed_eulers_np = np.zeros((n_spots, 3), dtype=np.float64)
    for i, (xc, yc, _ud, e1, e2, e3) in enumerate(grid_points):
        positions_np[i] = (xc, yc, 0.0)
        seed_eulers_np[i] = (e1, e2, e3)
    positions_um = torch.tensor(positions_np, device=torch_device, dtype=dtype)

    nL = p.n_distances
    n_geom = 3 + 3 * nL

    # ---- seed vector in the C's layout ------------------------------------
    x0 = np.zeros(n_geom + 3 * n_spots, dtype=np.float64)
    x0[0], x0[1], x0[2] = p.tx, p.ty, p.tz
    x0[3] = p.Lsd[0]
    for i in range(1, nL):
        x0[3 + i] = p.Lsd[i] - p.Lsd[i - 1]        # cumulative delta, as the C
    for i in range(nL):
        x0[3 + nL + i] = p.ybc[i]
        x0[3 + 2 * nL + i] = p.zbc[i]
    x0[n_geom:] = seed_eulers_np.reshape(-1)

    # ---- bounds straight from the paramfile tolerances ---------------------
    lo = x0.copy()
    hi = x0.copy()
    tt = p.tilts_tol
    lo[0:3] -= tt; hi[0:3] += tt
    lo[3] -= p.lsd_tol; hi[3] += p.lsd_tol
    for i in range(1, nL):
        lo[3 + i] -= p.lsd_rel_tol; hi[3 + i] += p.lsd_rel_tol
    for i in range(nL):
        lo[3 + nL + i] -= p.bc_tol_a; hi[3 + nL + i] += p.bc_tol_a
        lo[3 + 2 * nL + i] -= p.bc_tol_b; hi[3 + 2 * nL + i] += p.bc_tol_b
    ot = math.radians(p.orient_tol)
    lo[n_geom:] -= ot; hi[n_geom:] += ot
    bounds = list(zip(lo.tolist(), hi.tolist()))

    def unpack(x):
        tilts = torch.tensor(
            [[x[0], x[1], x[2]]] * nL, device=torch_device, dtype=dtype,
        )
        lsd = [float(x[3])]
        for i in range(1, nL):
            lsd.append(lsd[-1] + float(x[3 + i]))
        Lsd = torch.tensor(lsd, device=torch_device, dtype=dtype)
        ybc = torch.tensor(
            [float(x[3 + nL + i]) for i in range(nL)],
            device=torch_device, dtype=dtype)
        zbc = torch.tensor(
            [float(x[3 + 2 * nL + i]) for i in range(nL)],
            device=torch_device, dtype=dtype)
        eul = torch.tensor(
            np.asarray(x[n_geom:], dtype=np.float64).reshape(n_spots, 3),
            device=torch_device, dtype=dtype)
        return GeometryOverrides(Lsd=Lsd, y_BC=ybc, z_BC=zbc, tilts=tilts), eul

    n_eval = [0]

    # ---- CPU dispatch tuning -------------------------------------------
    # The objective is DISPATCH-bound, not compute-bound: one evaluation is
    # ~700 data elements spread over ~728 torch ops, so per-op overhead
    # dominates and the usual knobs invert.
    #
    #   * MORE THREADS IS SLOWER.  Measured on 12 voxels x 3 distances:
    #     1 thread 1.930 ms, 4 -> 1.975, 16 -> 2.052, 32 -> 2.536 ms.
    #     Intra-op threading cannot pay for itself on ~1400-element tensors.
    #   * fp32 buys NOTHING (1.874 vs 1.879 ms) for the same reason.
    #
    # So pin to a single thread while this objective runs, and restore after.
    _prev_threads = torch.get_num_threads()
    if str(device) == "cpu" or torch_device.type == "cpu":
        torch.set_num_threads(1)

    # torch.compile fuses the op storm: forward 1.499 -> 0.316 ms (4.7x),
    # verified identical (frame_nr / y_pixel / valid all equal).  Guarded --
    # if compilation is unavailable or the result differs, fall back silently.
    _model_fast = model
    try:
        _cand = torch.compile(model, dynamic=False)
        with torch.no_grad():
            _se = torch.tensor(seed_eulers_np, device=torch_device, dtype=dtype)
            _a = model(_se, positions_um)
            _b = _cand(_se, positions_um)
        if (torch.allclose(_a.frame_nr, _b.frame_nr)
                and torch.allclose(_a.y_pixel, _b.y_pixel)
                and torch.allclose(_a.z_pixel, _b.z_pixel)
                and torch.equal(_a.valid, _b.valid)):
            _model_fast = _cand
    except Exception:
        pass

    def _fractions_loop(eul, geom_ov):
        """Per-voxel loop. Reference implementation; parity target."""
        out = []
        with torch.no_grad(), overrides(model, geom_ov):
            for vi in range(n_spots):
                sp = model(eul[vi:vi + 1], positions_um[vi:vi + 1])
                out.append(float(obs.hard_fraction(
                    sp.frame_nr, sp.y_pixel, sp.z_pixel, sp.valid)))
        return out

    def _fractions_batched(eul, geom_ov):
        """All voxels in ONE forward call. Returns per-voxel fractions.

        The forward model stacks N grains **k-major**: the returned leading
        axis is ``K*N`` ordered as ``[k0g0, k0g1, ... k1g0, k1g1, ...]``, so
        the reshape is ``(K, N, M)`` followed by a transpose -- NOT
        ``(N, K, M)``.  Getting that backwards yields per-voxel fractions that
        differ by ~2e-2 while the MEAN still matches to 10 decimal places,
        which is exactly how a previous batching attempt slipped through.

        Reshaping is also what keeps this a *mean of per-voxel fractions*.
        Passing the un-reshaped ``(K*N, M)`` straight to ``hard_fraction``
        collapses everything into ONE pooled ``total_matched/total_predicted``
        -- a different quantity that happens to coincide only when every voxel
        has the same denominator.

        Verified elementwise against ``_fractions_loop``: max|diff| = 0.0,
        7.7x faster on CPU (18.55 -> 2.42 ms/eval for 12 voxels x 3 distances).
        """
        with torch.no_grad(), overrides(model, geom_ov):
            sp = _model_fast(eul, positions_um)
            NK, M = sp.frame_nr.shape
            if NK % n_spots:
                return None                      # unexpected layout; caller falls back
            K = NK // n_spots
            fr = sp.frame_nr.reshape(K, n_spots, M).transpose(0, 1)
            va = sp.valid.reshape(K, n_spots, M).transpose(0, 1)
            yp = sp.y_pixel.reshape(nL, K, n_spots, M).transpose(1, 2)
            zp = sp.z_pixel.reshape(nL, K, n_spots, M).transpose(1, 2)
            f = obs.hard_fraction(fr, yp, zp, va)
        if f.numel() != n_spots:
            return None
        return f

    # One-time parity gate: the batched path is only used if it reproduces the
    # loop exactly on the seed. A wrong stacking order is silent otherwise --
    # the aggregate agrees while every per-voxel value is wrong.
    _g0, _e0 = unpack(x0)
    _ref = _fractions_loop(_e0, _g0)
    _bat = _fractions_batched(_e0, _g0)
    use_batched = (
        _bat is not None
        and max(abs(float(a) - float(b)) for a, b in zip(_ref, _bat)) < 1e-12
    )
    if verbose:
        print(f"  objective: {'BATCHED' if use_batched else 'per-voxel loop'} "
              f"({'parity verified' if use_batched else 'batched path failed parity'})",
              flush=True)

    def objective(x) -> float:
        geom_ov, eul = unpack(x)
        if use_batched:
            f = _fractions_batched(eul, geom_ov)
            if f is not None:
                n_eval[0] += 1
                return 1.0 - float(f.sum()) / n_spots
        total = sum(_fractions_loop(eul, geom_ov))
        n_eval[0] += 1
        return 1.0 - total / n_spots

    seed_val = objective(x0)
    if verbose:
        print(f"Multipoint (HARD FracOverlap, C-equivalent): "
              f"{n_spots} voxels, {nL} distances", flush=True)
        print(f"  Original val: {1.0 - seed_val:.10f}   "
              f"(this is the C's 'Original val')", flush=True)

    # Local -> GLOBAL -> local ladder, repeated NumIterations times.
    #
    # The C alternates NLOPT_LN_NELDERMEAD with NLOPT_GN_CRS2_LM (a global
    # controlled random search) five times per iteration, precisely because a
    # single local method in 3 + 3*nL + 3*nSpots dimensions stalls. A lone
    # scipy Nelder-Mead did exactly that here: 20001 evaluations for +0.0009
    # against the C's +0.0119, finishing on the seed geometry to 4 decimals.
    #
    # scipy has no CRS2; differential_evolution is the closest bounded global
    # search. It is seeded with the current best (`x0=`) and given a small
    # budget per round, so it perturbs broadly without dominating the runtime.
    from scipy.optimize import differential_evolution
    from concurrent.futures import ThreadPoolExecutor

    # ---- CONDITIONING: build the initial simplex from the TOLERANCES ------
    #
    # scipy's default Nelder-Mead simplex is 5% of each value, or an ABSOLUTE
    # 0.00025 when the value is exactly zero.  With this parameter vector that
    # spans FIVE ORDERS OF MAGNITUDE in step/tolerance:
    #
    #     tx,ty,tz   value 0.000    step 0.00025   tol 1.00   ratio 2.5e-04
    #     Lsd0       value 6162     step 308       tol 500    ratio 6.2e-01
    #     ybc0       value 2701     step 135       tol 2.00   ratio 6.8e+01
    #
    # The tilts are seeded at exactly 0, so their first step is 0.00025 deg --
    # about 4000x smaller than their box, and ~200x below the scale at which
    # the objective responds (measured: no change until ~0.05 deg).  Since the
    # objective is also QUANTISED at 1/(spots*voxels), a 0.00025 deg step
    # returns a BIT-IDENTICAL value, so the tilts can never move.  Meanwhile
    # ybc takes a 135 px first step against a 2 px tolerance.
    #
    # Same failure as the FF refiner's gradient imbalance: the simplex is
    # degenerate before the search starts.  Fix: scale every step to that
    # parameter's own box, so all coordinates are explored comparably.
    _lo = np.asarray([b[0] for b in bounds], dtype=float)
    _hi = np.asarray([b[1] for b in bounds], dtype=float)
    _halfwidth = 0.5 * (_hi - _lo)

    def _simplex(xc):
        """(N+1, N) simplex centred on xc with tolerance-scaled steps."""
        step = 0.3 * _halfwidth
        step[step <= 0] = 1e-6
        sim = np.repeat(np.asarray(xc, dtype=float)[None, :], len(xc) + 1, axis=0)
        for i in range(len(xc)):
            s = step[i] if xc[i] + step[i] <= _hi[i] else -step[i]
            sim[i + 1, i] = np.clip(xc[i] + s, _lo[i], _hi[i])
        return sim

    # fatol must not be far below the objective's QUANTISATION, or Nelder-Mead
    # can never satisfy it and always burns the full maxfev budget.
    _fatol = max(1e-9, 0.1 / max(n_spots * 40, 1))

    t0 = time.perf_counter()
    x_best = x0.copy()
    f_best = float(seed_val)
    n_rounds = max(1, p.num_iterations)

    # Threads, not processes: the objective closes over the packed ObsVolume
    # (13 GB here), and scipy's default multiprocessing `workers` would pickle
    # it to every worker.  Threads share it, and torch releases the GIL for
    # tensor ops.  differential_evolution already runs updating="deferred",
    # which is the precondition for a parallel map.
    _nw = max(1, int(n_cpus or 1))
    _pool = ThreadPoolExecutor(_nw) if _nw > 1 else None
    _workers = _pool.map if _pool is not None else 1
    if verbose and _pool is not None:
        print(f"  global phase: {_nw} threads (deferred updating)", flush=True)

    for rnd in range(n_rounds):
        r = minimize(
            objective, x_best, method="Nelder-Mead", bounds=bounds,
            options=dict(maxiter=max_iter, maxfev=max_iter,
                         xatol=1e-7, fatol=_fatol, adaptive=True,
                         initial_simplex=_simplex(x_best)),
        )
        if float(r.fun) < f_best:
            f_best, x_best = float(r.fun), r.x.copy()
        if verbose:
            print(f"  round {rnd+1}/{n_rounds} local : "
                  f"{1.0 - f_best:.10f}", flush=True)

        gr = differential_evolution(
            objective, bounds, x0=x_best, seed=seed + rnd,
            maxiter=global_iters, popsize=8, tol=1e-8,
            mutation=(0.3, 0.9), recombination=0.9,
            polish=False, init="sobol", updating="deferred",
            workers=_workers,
        )
        if float(gr.fun) < f_best:
            f_best, x_best = float(gr.fun), gr.x.copy()
        if verbose:
            print(f"  round {rnd+1}/{n_rounds} global: "
                  f"{1.0 - f_best:.10f}", flush=True)

    # final local polish from the best point found
    r = minimize(
        objective, x_best, method="Nelder-Mead", bounds=bounds,
        options=dict(maxiter=max_iter, maxfev=max_iter,
                     xatol=1e-7, fatol=_fatol, adaptive=True,
                     initial_simplex=_simplex(x_best)),
    )
    if float(r.fun) < f_best:
        f_best, x_best = float(r.fun), r.x.copy()
    if _pool is not None:
        _pool.shutdown(wait=True)

    class _R:
        pass
    res = _R()
    res.x = x_best
    res.fun = f_best
    secs = time.perf_counter() - t0

    best_val = 1.0 - float(res.fun)
    if verbose:
        print(f"  Final value:  {best_val:.10f}   "
              f"({n_eval[0]} evals, {secs:.1f} s)")
        print(f"  improvement:  {best_val - (1.0 - seed_val):+.10f}")

    geom_ov, eul = unpack(res.x)
    lsd_out = geom_ov.Lsd.tolist()
    if verbose:
        for d in range(nL):
            print(f"Layer {d}: Lsd={lsd_out[d]:.4f}, "
                  f"BC=({float(geom_ov.y_BC[d]):.4f}, "
                  f"{float(geom_ov.z_BC[d]):.4f})")
        print(f"Tilts (shared): tx={res.x[0]:.4f}, ty={res.x[1]:.4f}, "
              f"tz={res.x[2]:.4f}")

    return dict(
        seed_frac_overlap=1.0 - seed_val,
        final_frac_overlap=best_val,
        Lsd=lsd_out,
        y_BC=geom_ov.y_BC.tolist(),
        z_BC=geom_ov.z_BC.tolist(),
        tilts=[float(res.x[0]), float(res.x[1]), float(res.x[2])],
        eulers=eul.tolist(),
        n_evals=n_eval[0],
        seconds=secs,
    )
