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

import json
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from .fit_kernel import LBFGSConfig, run_lbfgs
from .geom_multistart import (
    analyse_basins,
    build_geometry_starts,
    default_n_starts,
    geom_dict,
    geometry_layout,
    print_multimodal_warning,
    scanned_params,
)
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


#: File names written next to each other by both multipoint drivers.
RESULT_JSON_NAME = "multipoint_result.json"
REFINED_PARAMS_NAME = "params_refined.txt"

#: The hard objective is a mean of fractions in [0, 1]; within this of 1.0
#: every predicted spot of every chosen voxel is already inside an observed
#: spot, and the objective is flat.
SATURATION_EPS = 1e-9


def _resolve_result_dir(p: FitParams, result_dir: Optional[str]) -> Path:
    """Where the multipoint result goes: explicit arg, else the paramfile's
    ``OutputDirectory``, else the current working directory."""
    if result_dir:
        d = Path(result_dir)
    elif p.output_dir:
        d = Path(p.output_dir)
    else:
        d = Path.cwd()
    d.mkdir(parents=True, exist_ok=True)
    return d


def refined_paramfile_text(
    text: str,
    *,
    Lsd: Sequence[float],
    y_BC: Sequence[float],
    z_BC: Sequence[float],
    tilts: Sequence[float],
    wedge: Optional[float] = None,
    header: str = "",
) -> str:
    """Return ``text`` (a paramfile) with its geometry lines replaced.

    ``Lsd`` and ``BC`` are per distance and are replaced IN ORDER (the i-th
    ``Lsd`` line gets ``Lsd[i]``), which is how the parser assigns them.
    ``tx``/``ty``/``tz`` and, when ``wedge`` is given, ``Wedge`` are replaced
    in place; a key that is absent is appended. Every other line -- comments,
    tolerances, ``LsdTol``/``BCTol`` (keys are matched on the whole first
    token, never as a prefix) -- is kept verbatim.
    """
    lines = text.splitlines()
    out: List[str] = []
    i_lsd = i_bc = 0
    seen = set()
    scalars = {"tx": tilts[0], "ty": tilts[1], "tz": tilts[2]}
    if wedge is not None:
        scalars["Wedge"] = wedge
    for line in lines:
        tok = line.split()
        key = tok[0] if tok else ""
        if key == "Lsd" and i_lsd < len(Lsd):
            out.append(f"Lsd {Lsd[i_lsd]:.6f}")
            i_lsd += 1
        elif key == "BC" and i_bc < len(y_BC):
            out.append(f"BC {y_BC[i_bc]:.6f} {z_BC[i_bc]:.6f}")
            i_bc += 1
        elif key in scalars:
            out.append(f"{key} {scalars[key]:.8f}")
            seen.add(key)
        else:
            out.append(line)
    for key, val in scalars.items():
        if key not in seen:
            out.append(f"{key} {val:.8f}")
    body = "\n".join(out) + "\n"
    return (header + body) if header else body


def write_multipoint_outputs(
    paramfile: str,
    p: FitParams,
    result: dict,
    result_dir: Optional[str] = None,
) -> dict:
    """Write ``multipoint_result.json`` and ``params_refined.txt``.

    Before this existed the refined geometry was only PRINTED, and only under
    ``--verbose``, so a production run left nothing on disk to adopt. The
    refined paramfile is the input paramfile with only Lsd / BC / tx / ty /
    tz (and Wedge when it was refined) replaced, so it can be fed straight
    back to the reconstruction. Returns the two paths as strings.
    """
    d = _resolve_result_dir(p, result_dir)
    json_path = d / RESULT_JSON_NAME
    par_path = d / REFINED_PARAMS_NAME
    tilts = result["tilts"]
    header = (
        f"# Refined by midas-nf-fit-multipoint ({result.get('objective')}) "
        f"from {Path(paramfile).resolve()}\n"
        f"# objective {result.get('seed_frac_overlap')} -> "
        f"{result.get('final_frac_overlap')}; see {RESULT_JSON_NAME}\n"
    )
    if result.get("under_determined"):
        header += ("# WARNING: geometry UNDER-DETERMINED (objective "
                   "saturated / flat); do not adopt without a check.\n")
    if result.get("multimodal"):
        header += ("# WARNING: geometry NOT UNIQUELY DETERMINED: "
                   f"{result.get('n_competitive_basins')} distinct basins "
                   "score within the multimodal margin of the best (in: "
                   f"{', '.join(result.get('ambiguous_params') or [])}); "
                   "see geometry_trials / basins in the json.\n")
    text = refined_paramfile_text(
        Path(paramfile).read_text(),
        Lsd=result["Lsd"], y_BC=result["y_BC"], z_BC=result["z_BC"],
        tilts=tilts,
        wedge=result["wedge"] if result.get("wedge_refined") else None,
        header=header,
    )
    par_path.write_text(text)
    paths = dict(result_json=str(json_path), params_refined=str(par_path))
    payload = dict(result)
    payload.update(paramfile=str(Path(paramfile).resolve()), **paths)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return paths


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
    result_dir: Optional[str] = None,
) -> dict:
    """Joint multi-voxel calibration. Replaces
    :program:`FitOrientationParametersMultiPoint`.

    Voxels and seed Eulers come from the paramfile's ``GridPoints``
    block (the C code's existing convention).

    The refined geometry is always written to ``multipoint_result.json`` and
    ``params_refined.txt`` in ``result_dir`` (default: ``OutputDirectory``,
    else the cwd) and a summary is always printed; ``verbose`` only adds the
    per-trial detail.
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
    print(f"Multipoint calibration (soft surrogate): {n_spots} voxels, "
          f"{p.n_distances} distances, "
          f"{'wedge ON' if p.refine_wedge else 'wedge OFF'}", flush=True)

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

    # ---- GEOMETRY multi-start (issue #16) ---------------------------------
    # Every trial used to start the geometry at the paramfile value; only
    # trials after the first were perturbed, by ~0.3 of a tolerance, and
    # NumIterations defaults to 1. On real data the objective is multimodal
    # in the wedge, so the answer depended on where the wedge was started.
    # Each trial now has its own geometry START: the seed, a deterministic
    # scan of the wedge (and optionally the tilts) over its tolerance box,
    # and random starts inside the boxes (fixed RNG seed).
    geom_names, geom_seed, geom_hw = geometry_layout(p)
    geom_starts = build_geometry_starts(
        geom_names, geom_seed, geom_hw,
        scan=scanned_params(p), n_scan=p.multipoint_geom_scan,
        n_total=default_n_starts(p), rng_seed=seed,
    )
    geometry_trials: List[dict] = []
    nL = p.n_distances

    n_trials = len(geom_starts)
    print(f"  geometry multi-start: {n_trials} start(s) "
          f"({', '.join(g.label for g in geom_starts)})", flush=True)
    for trial in range(n_trials):
        gstart = geom_starts[trial]
        if verbose:
            print(f"\n--- Multi-start trial {trial+1}/{n_trials} "
                  f"[{gstart.label}] ---")

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

        # Trials > 0: the geometry starts at this trial's geometry start
        # (the boxes stay centred on the seed), and the Eulers are perturbed
        # by 0.3 of a tolerance as before.
        if trial > 0:
            for be in box_eulers:
                be.perturb(0.3, generator=rng)
            gx = torch.tensor(gstart.x, device=torch_device, dtype=dtype)
            box_tilts.set_x(gx[0:3])
            box_lsd0.set_x(gx[3:4])
            if box_lsd_delta is not None:
                box_lsd_delta.set_x(gx[4:3 + nL])
            box_ybc.set_x(gx[3 + nL:3 + 2 * nL])
            box_zbc.set_x(gx[3 + 2 * nL:3 + 3 * nL])
            if box_wedge is not None:
                box_wedge.set_x(gx[3 + 3 * nL])

        def geom_vector() -> np.ndarray:
            parts = [box_tilts.x, box_lsd0.x]
            if box_lsd_delta is not None:
                parts.append(box_lsd_delta.x)
            parts += [box_ybc.x, box_zbc.x]
            if box_wedge is not None:
                parts.append(box_wedge.x.reshape(1))
            return torch.cat([t.detach().reshape(-1) for t in parts]
                             ).cpu().numpy().astype(np.float64)

        start_vec = geom_vector()

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

        with torch.no_grad():
            trial_end_loss = float(
                _multipoint_loss(model, obs, box_eulers, positions_um,
                                 sigma_px, make_geom_ov())
            )
        end_vec = geom_vector()
        geometry_trials.append(dict(
            trial=trial,
            label=gstart.label,
            start=start_vec.tolist(),
            end=end_vec.tolist(),
            start_geometry=geom_dict(geom_names, start_vec),
            end_geometry=geom_dict(geom_names, end_vec),
            start_frac_overlap=1.0 - trial_start_loss,
            end_frac_overlap=1.0 - trial_end_loss,
        ))
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

    # The summary is the RESULT, so it is printed unconditionally; only the
    # per-trial detail above is gated on verbose.
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
    # Tilts are ONE shared triple (see tilts0 above); every row is identical.
    _t = best_overall_state['tilts'][0]
    print(f"Tilts (shared): tx={_t[0]:.4f}, ty={_t[1]:.4f}, tz={_t[2]:.4f}")
    if best_overall_state.get("wedge") is not None:
        print(f"Wedge: {best_overall_state['wedge']:.4f}")

    _print_trial_table(geometry_trials, scanned_params(p) or geom_names)
    basins = analyse_basins(
        geometry_trials, geom_names, geom_hw,
        basin_frac=p.multipoint_basin_frac,
        rel_margin=p.multipoint_basin_margin,
    )
    print_multimodal_warning(basins, "soft")

    summary = dict(
        objective="soft",
        seed_frac_overlap=1.0 - seed_loss,
        final_frac_overlap=1.0 - best_overall_state["loss"],
        Lsd=best_overall_state["Lsd"],
        y_BC=best_overall_state["y_BC"],
        z_BC=best_overall_state["z_BC"],
        tilts=[float(v) for v in _t],
        wedge=(best_overall_state["wedge"]
               if best_overall_state.get("wedge") is not None
               else float(p.wedge)),
        wedge_refined=bool(p.refine_wedge),
        n_voxels=n_spots,
        best_trial=best_overall_state["trial"],
        eulers=best_overall_state["voxel_eulers_rad"],
        voxel_individual_overlaps=best_overall_state[
            "voxel_individual_overlaps"],
        seed_tracks_data=bool(seed_loss <= 0.99),
        **_multistart_fields(geom_names, geometry_trials, basins),
    )
    paths = write_multipoint_outputs(paramfile, p, summary, result_dir)
    print(f"Wrote {paths['result_json']}")
    print(f"Wrote {paths['params_refined']}", flush=True)
    best_overall_state.update(paths)
    best_overall_state.update(
        _multistart_fields(geom_names, geometry_trials, basins))
    return best_overall_state


def _multistart_fields(names, trials, basins) -> dict:
    """The geometry multi-start record that goes into the result json."""
    return dict(
        geometry_names=list(names),
        geometry_trials=trials,
        multimodal=basins["multimodal"],
        n_basins=basins["n_basins"],
        n_competitive_basins=basins["n_competitive_basins"],
        ambiguous_params=basins["ambiguous_params"],
        basins=basins["basins"],
        multimodal_criteria=basins["criteria"],
    )


def _print_trial_table(trials: List[dict], show: Sequence[str]) -> None:
    """One line per geometry start: start -> end of the scanned parameters
    and the objective. Always printed; it is the evidence for (or against)
    a unique geometry."""
    if len(trials) <= 1:
        return
    print(f"  geometry trials ({len(trials)}):")
    for t in trials:
        se = "  ".join(
            f"{n} {t['start_geometry'][n]:.5g}->{t['end_geometry'][n]:.5g}"
            for n in show if n in t["start_geometry"])
        print(f"    [{t['label']}] overlap {t['start_frac_overlap']:.6f}"
              f" -> {t['end_frac_overlap']:.6f}   {se}", flush=True)


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
    result_dir: Optional[str] = None,
    compile_model: bool = True,
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
        [x[3+3nL]                     wedge, ONLY with RefineWedge 1]
        x[n_geom + 3i : +3]           Eulers of voxel i

    The C never refines the wedge; ``RefineWedge 1`` (with ``WedgeTol``) is
    the extension the soft path already honoured, and it is honoured here the
    same way: one extra geometry slot, bounded to ``Wedge +/- WedgeTol``.
    With ``RefineWedge 0`` the layout is exactly the C's.

    It also uses the PACKED obs volume, so it needs ~1 bit/pixel instead of the
    dense float32 the soft path requires (1.9 GB vs 56 GiB on a 3600-frame scan).

    SATURATION. The objective is a mean of per-voxel fractions and tops out at
    exactly 1.0 once every predicted spot of every chosen voxel lands inside an
    observed spot. With ~10-20 voxels on real data that happens easily (Au at
    1-ID: 0.745 -> 1.0000 in round 1), and from then on the objective is FLAT:
    the geometry returned is one arbitrary point on a plateau, not a
    measurement. This is detected (``saturated``), the geometry parameters the
    objective cannot see around the optimum are listed (``flat_params``, a
    +/- quarter-tolerance probe), and ``under_determined`` is set and warned
    about loudly if either holds. The objective itself is unchanged.

    Output: always prints a summary and writes ``multipoint_result.json`` and
    ``params_refined.txt`` to ``result_dir`` (default ``OutputDirectory``, else
    the cwd). ``verbose`` adds per-round detail only.
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
    # Wedge slot, appended to the geometry block only when refined, so that
    # RefineWedge 0 keeps the C's exact layout. This path used to ignore
    # RefineWedge entirely: no slot, no bound, no report.
    refine_wedge = bool(p.refine_wedge)
    i_wedge = 3 + 3 * nL if refine_wedge else None
    n_geom = 3 + 3 * nL + (1 if refine_wedge else 0)
    geom_names = (
        ["tx", "ty", "tz", "Lsd[0]"]
        + [f"dLsd[{i}]" for i in range(1, nL)]
        + [f"ybc[{i}]" for i in range(nL)]
        + [f"zbc[{i}]" for i in range(nL)]
        + (["wedge"] if refine_wedge else [])
    )

    # ---- seed vector in the C's layout ------------------------------------
    x0 = np.zeros(n_geom + 3 * n_spots, dtype=np.float64)
    x0[0], x0[1], x0[2] = p.tx, p.ty, p.tz
    x0[3] = p.Lsd[0]
    for i in range(1, nL):
        x0[3 + i] = p.Lsd[i] - p.Lsd[i - 1]        # cumulative delta, as the C
    for i in range(nL):
        x0[3 + nL + i] = p.ybc[i]
        x0[3 + 2 * nL + i] = p.zbc[i]
    if refine_wedge:
        x0[i_wedge] = p.wedge
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
    if refine_wedge:
        lo[i_wedge] -= p.wedge_tol; hi[i_wedge] += p.wedge_tol
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
        # Wedge is degrees, as in the paramfile and GeometryOverrides. When it
        # is not refined it is left to the model, which was built with p.wedge.
        wedge = (
            torch.tensor(float(x[i_wedge]), device=torch_device, dtype=dtype)
            if refine_wedge else None
        )
        return GeometryOverrides(
            Lsd=Lsd, y_BC=ybc, z_BC=zbc, tilts=tilts, wedge=wedge,
        ), eul

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

    # Forward call with the geometry as EXPLICIT inputs. The overrides are
    # applied INSIDE the (possibly compiled) function.
    #
    # This used to be `torch.compile(model)` called inside
    # `with overrides(model, geom_ov)`. overrides() swaps the geometry by
    # object.__setattr__ on the module, which a compiled nn.Module does NOT
    # see: the compiled graph keeps reading the registered tilts / Lsd / BC
    # / wedge, i.e. the SEED geometry, for every call. Measured with torch
    # 2.9.1 (backend aot_eager): a BC + tilt override moved eager spots by
    # 13.0 px and the compiled ones by 0.0 px. Wherever inductor works
    # (Linux; it fails on the Mac and fell back to eager there, hiding it)
    # the hard objective was therefore BLIND to every geometry parameter:
    # geometry starts scored exactly the seed value and never moved, and
    # only the Eulers were refined. The parity gate did not catch it
    # because it compared compiled vs eager at the seed only, with no
    # override -- the one point where ignoring the override is harmless.
    def _fwd(eul, Lsd, ybc, zbc, tilts, wedge):
        with overrides(model, GeometryOverrides(
                Lsd=Lsd, y_BC=ybc, z_BC=zbc, tilts=tilts, wedge=wedge)):
            return model(eul, positions_um)

    def _fwd_args(eul, geom_ov):
        return (eul, geom_ov.Lsd, geom_ov.y_BC, geom_ov.z_BC, geom_ov.tilts,
                geom_ov.wedge)

    # Parity gate, at the seed AND at a probe geometry with EVERY refined
    # geometry coordinate moved by half its tolerance: a compiled forward
    # that ignored the geometry would pass the first and fail the second.
    # torch.compile fuses the op storm (forward 1.499 -> 0.316 ms, 4.7x);
    # if it is unavailable or fails either check, the eager forward is used.
    _fwd_fast = _fwd
    compiled_forward = False
    compile_note = "disabled by caller"
    if compile_model:
        try:
            _cand = torch.compile(_fwd, dynamic=False)
            x_probe = x0.copy()
            x_probe[:n_geom] = np.clip(
                x0[:n_geom] + 0.25 * (hi[:n_geom] - lo[:n_geom]),
                lo[:n_geom], hi[:n_geom])
            ok = True
            with torch.no_grad():
                for _x in (x0, x_probe):
                    _g, _e = unpack(_x)
                    _a = _fwd(*_fwd_args(_e, _g))
                    _b = _cand(*_fwd_args(_e, _g))
                    ok = ok and bool(
                        torch.allclose(_a.frame_nr, _b.frame_nr)
                        and torch.allclose(_a.y_pixel, _b.y_pixel)
                        and torch.allclose(_a.z_pixel, _b.z_pixel)
                        and torch.equal(_a.valid, _b.valid))
            if ok:
                _fwd_fast, compiled_forward = _cand, True
                compile_note = "compiled; parity verified at seed and probe"
            else:
                compile_note = ("compiled forward FAILED parity at seed/probe "
                                "geometry; using eager")
        except Exception as exc:              # compiler unavailable etc.
            compile_note = f"compile unavailable ({type(exc).__name__})"
    if verbose or (compile_model and not compiled_forward
                   and "FAILED" in compile_note):
        print(f"  forward: {compile_note}", flush=True)

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
        with torch.no_grad():
            sp = _fwd_fast(*_fwd_args(eul, geom_ov))
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
    print(f"Multipoint (HARD FracOverlap, C-equivalent): "
          f"{n_spots} voxels, {nL} distances, "
          f"{'wedge ON' if refine_wedge else 'wedge OFF'}", flush=True)
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

    # Best value after each stage, so a plateau is visible in the output.
    history: List[dict] = [dict(stage="seed", frac_overlap=1.0 - f_best)]
    saturated_at: Optional[str] = (
        "seed" if (1.0 - f_best) >= 1.0 - SATURATION_EPS else None
    )

    def _record(stage: str) -> None:
        nonlocal saturated_at
        v = 1.0 - f_best
        history.append(dict(stage=stage, frac_overlap=v))
        if saturated_at is None and v >= 1.0 - SATURATION_EPS:
            saturated_at = stage
            # Say it NOW, not only at the end: every later stage is searching
            # a flat objective and cannot change the answer.
            print(f"  WARNING: hard objective SATURATED at 1.0 after {stage}; "
                  f"the geometry is no longer determined by the data (see "
                  f"the end-of-run summary).", flush=True)

    # ---- GEOMETRY multi-start (issue #16) ---------------------------------
    # The ladder below starts from ONE point, the paramfile geometry. Its
    # global phase samples the whole box, but with a small budget in a
    # 3 + 3*nL + 3*nSpots dimensional space, so a weakly-determined, multimodal
    # coordinate (the wedge, on real AlON NF) is not reliably explored. Run a
    # local search from each geometry start (seed, deterministic scan of the
    # wedge / optionally tilts, random starts in the boxes; Eulers at their
    # seeds), record every start and end, and hand the best to the ladder.
    # With a single start (RefineWedge 0, NumIterations <= 1, no
    # MultipointGeomStarts) this is skipped and the run is exactly as before.
    assert geometry_layout(p)[0] == geom_names   # one layout, both paths
    geom_starts = build_geometry_starts(
        geom_names, x0[:n_geom], _halfwidth[:n_geom],
        scan=scanned_params(p), n_scan=p.multipoint_geom_scan,
        n_total=default_n_starts(p), rng_seed=seed,
    )
    geometry_trials: List[dict] = []

    def _trial_record(label, xs, fs, xe, fe):
        geometry_trials.append(dict(
            trial=len(geometry_trials), label=label,
            start=[float(v) for v in xs[:n_geom]],
            end=[float(v) for v in xe[:n_geom]],
            start_geometry=geom_dict(geom_names, xs[:n_geom]),
            end_geometry=geom_dict(geom_names, xe[:n_geom]),
            start_frac_overlap=1.0 - float(fs),
            end_frac_overlap=1.0 - float(fe),
        ))

    if len(geom_starts) > 1:
        print(f"  geometry multi-start: {len(geom_starts)} starts "
              f"({', '.join(g.label for g in geom_starts)})", flush=True)
        for gs in geom_starts:
            xs = x0.copy()
            xs[:n_geom] = np.clip(gs.x, _lo[:n_geom], _hi[:n_geom])
            fs = objective(xs)
            r = minimize(
                objective, xs, method="Nelder-Mead", bounds=bounds,
                options=dict(maxiter=max_iter, maxfev=max_iter,
                             xatol=1e-7, fatol=_fatol, adaptive=True,
                             initial_simplex=_simplex(xs)),
            )
            _trial_record(gs.label, xs, fs, r.x, r.fun)
            if float(r.fun) < f_best:
                f_best, x_best = float(r.fun), r.x.copy()
            if verbose:
                print(f"    [{gs.label}] {1.0 - fs:.10f} -> "
                      f"{1.0 - float(r.fun):.10f}", flush=True)
        _record("geometry starts")
    x_ladder_start = x_best.copy()
    f_ladder_start = f_best
    _i_ladder = len(history)

    for rnd in range(n_rounds):
        r = minimize(
            objective, x_best, method="Nelder-Mead", bounds=bounds,
            options=dict(maxiter=max_iter, maxfev=max_iter,
                         xatol=1e-7, fatol=_fatol, adaptive=True,
                         initial_simplex=_simplex(x_best)),
        )
        if float(r.fun) < f_best:
            f_best, x_best = float(r.fun), r.x.copy()
        _record(f"round {rnd+1} local")
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
        _record(f"round {rnd+1} global")
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
    _record("final polish")
    if _pool is not None:
        _pool.shutdown(wait=True)

    class _R:
        pass
    res = _R()
    res.x = x_best
    res.fun = f_best
    secs = time.perf_counter() - t0

    best_val = 1.0 - float(res.fun)

    # ---- is the returned geometry actually DETERMINED? ---------------------
    # Probe each geometry coordinate by +/- a quarter of its tolerance
    # half-width. A coordinate whose move changes the objective in NEITHER
    # direction sits on a plateau: the data do not pin it, and its returned
    # value is whatever the optimiser happened to stop on. 2*n_geom extra
    # evaluations, negligible next to the search.
    flat_params: List[str] = []
    for j in range(n_geom):
        step = 0.25 * _halfwidth[j]
        if step <= 0:
            continue
        moved = False
        for sgn in (+1.0, -1.0):
            xp = res.x.copy()
            xp[j] = float(np.clip(xp[j] + sgn * step, _lo[j], _hi[j]))
            if xp[j] == res.x[j]:
                continue
            if abs(objective(xp) - float(res.fun)) > SATURATION_EPS:
                moved = True
                break
        if not moved:
            flat_params.append(geom_names[j])

    saturated = saturated_at is not None
    # Flat across rounds: nothing after round 1 changed the value. On its own
    # this is also what a converged run looks like, so it is REPORTED but is
    # not by itself the under-determined verdict -- the probe above is.
    _after_r1 = [h["frac_overlap"] for h in history[_i_ladder + 1:]]
    flat_across_rounds = bool(
        n_rounds >= 2 and _after_r1
        and max(_after_r1) - min(_after_r1) <= SATURATION_EPS
    )
    under_determined = bool(saturated or flat_params)

    # ---- summary: ALWAYS printed (it is the result, not chatter) ----------
    print(f"  Final value:  {best_val:.10f}   "
          f"({n_eval[0]} evals, {secs:.1f} s)")
    print(f"  improvement:  {best_val - (1.0 - seed_val):+.10f}")

    geom_ov, eul = unpack(res.x)
    lsd_out = geom_ov.Lsd.tolist()
    for d in range(nL):
        print(f"Layer {d}: Lsd={lsd_out[d]:.4f}, "
              f"BC=({float(geom_ov.y_BC[d]):.4f}, "
              f"{float(geom_ov.z_BC[d]):.4f})")
    print(f"Tilts (shared): tx={res.x[0]:.4f}, ty={res.x[1]:.4f}, "
          f"tz={res.x[2]:.4f}")
    wedge_out = float(res.x[i_wedge]) if refine_wedge else float(p.wedge)
    print(f"Wedge: {wedge_out:.4f} "
          f"({'refined' if refine_wedge else 'fixed, RefineWedge 0'})")

    if under_determined:
        bar = "!" * 72
        print(bar)
        print("WARNING: the multipoint geometry is UNDER-DETERMINED.")
        if saturated:
            print(f"  The hard objective saturated at 1.0 (at: {saturated_at})."
                  f" Every predicted spot of the {n_spots} chosen voxels is"
                  f" already inside an observed spot, so the objective is"
                  f" flat and the geometry above is one arbitrary point on a"
                  f" plateau.")
        if flat_params:
            print(f"  Moving these by +/-25% of their tolerance does not change"
                  f" the objective: {', '.join(flat_params)}")
        print("  Do NOT adopt it as is. Use --objective soft, and/or more (and")
        print("  more widely spread) voxels, and/or tighter tolerances, then")
        print("  check the result against an independent measurement.")
        print(bar, flush=True)
    elif flat_across_rounds:
        print("  note: nothing after round 1 changed the objective.")

    # The ladder's own end point is a trial too (its global phase can change
    # basin), so it takes part in the basin analysis.
    _trial_record("ladder" if len(geom_starts) > 1 else "seed",
                  x_ladder_start, f_ladder_start, res.x, res.fun)
    _print_trial_table(geometry_trials, scanned_params(p) or geom_names)
    basins = analyse_basins(
        geometry_trials, geom_names, _halfwidth[:n_geom],
        basin_frac=p.multipoint_basin_frac,
        rel_margin=p.multipoint_basin_margin,
    )
    print_multimodal_warning(basins, "hard")

    result = dict(
        objective="hard",
        seed_frac_overlap=1.0 - seed_val,
        final_frac_overlap=best_val,
        Lsd=lsd_out,
        y_BC=geom_ov.y_BC.tolist(),
        z_BC=geom_ov.z_BC.tolist(),
        tilts=[float(res.x[0]), float(res.x[1]), float(res.x[2])],
        wedge=wedge_out,
        wedge_refined=refine_wedge,
        n_voxels=n_spots,
        eulers=eul.tolist(),
        n_evals=n_eval[0],
        seconds=secs,
        history=history,
        saturated=saturated,
        saturated_at=saturated_at,
        flat_across_rounds=flat_across_rounds,
        flat_params=flat_params,
        under_determined=under_determined,
        compiled_forward=compiled_forward,
        **_multistart_fields(geom_names, geometry_trials, basins),
    )
    paths = write_multipoint_outputs(paramfile, p, result, result_dir)
    print(f"Wrote {paths['result_json']}")
    print(f"Wrote {paths['params_refined']}", flush=True)
    result.update(paths)
    return result
