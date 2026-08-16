"""midas-defect-rods CLI: end-to-end rod detection on a voxel NPZ.

Pipeline:
  1. Load `voxels_layer*.npz` (from `demk_volume.py`) -> `VoxelCloud`.
  2. Run `seed_index.find_seed_orientation` to get an average orientation
     U and (optionally) refined (a, c).
  3. Run `rod_detect.find_rods` with that U to produce rod direction tags
     in the crystal frame.
  4. Write the rod catalog as JSON + CSV.
  5. Write a 3-panel Plotly HTML with the cloud + detected rods overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .. import __version__
from ..data_io import load_voxel_npz
from ..lattice import cual2_crystal
from ..rod_detect import find_rods
from ..seed_index import find_seed_orientation

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-defect"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--voxels", required=True,
                   help="path to voxels_layer*.npz produced by demk_volume.py")
    p.add_argument("--out-dir", required=True,
                   help="directory to write rod catalog + HTML")
    p.add_argument("--cual2-a", type=float, default=None,
                   help="lattice constant a (Å); defaults to canonical 6.066")
    p.add_argument("--cual2-c", type=float, default=None,
                   help="lattice constant c (Å); defaults to canonical 4.874")
    p.add_argument("--seed-tol-q-rel", type=float, default=0.02,
                   help="|q| matching tolerance for seed indexing")
    p.add_argument("--seed-tol-angle-deg", type=float, default=5.0,
                   help="direction tolerance (deg) for seed-indexer scoring")
    p.add_argument("--seed-n-bright", type=int, default=20,
                   help="bright-core count for seed indexing")
    p.add_argument("--seed-min-sep", type=float, default=0.05,
                   help="minimum separation between bright cores (1/Å)")
    p.add_argument("--seed-refine-lattice", action="store_true",
                   help="jointly refine (a, c) along with U")
    p.add_argument("--rod-n-cores", type=int, default=200,
                   help="bright cores for rod pair-seeding")
    p.add_argument("--rod-r-tube", type=float, default=0.05,
                   help="tube radius for inlier counting (1/Å)")
    p.add_argument("--rod-L-min", type=float, default=1.0,
                   help="minimum rod length (1/Å)")
    p.add_argument("--rod-L-max", type=float, default=None,
                   help="maximum rod length (1/Å); kills span-the-dataset artifacts")
    p.add_argument("--rod-N-min", type=int, default=200,
                   help="minimum inlier count for an accepted rod")
    p.add_argument("--cloud-min-intensity", type=float, default=None,
                   help="pre-filter voxels with intensity < this (kills diffuse haze)")
    p.add_argument("--rod-max-scoring", type=int, default=200_000,
                   help="cap on voxels used during candidate scoring")
    p.add_argument("--nms-direction-tol-deg", type=float, default=5.0,
                   help="NMS direction tolerance (deg); rods within this "
                        "angle of each other are de-duplicated")
    p.add_argument("--nms-pivot-perp-tol", type=float, default=0.3,
                   help="NMS perpendicular-pivot tolerance (1/Å); rods with "
                        "similar direction AND perp-distance below this are "
                        "considered duplicates")
    p.add_argument("--no-html", action="store_true",
                   help="skip HTML rendering (catalog only)")
    p.add_argument("--max-bright-html", type=int, default=80_000)
    p.add_argument("--max-haze-html", type=int, default=200_000)
    p.add_argument("--device", default=None,
                   help="torch device override (e.g. cpu, cuda, mps)")


def main(argv: list[str] | None = None) -> int:
    parser = _midas_make_parser(prog="midas-defect-rods",
                                     description=__doc__)
    _add_args(parser)
    args = parser.parse_args(argv)

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load
    print(f"[{time.time()-t0:6.1f}s] loading {args.voxels} ...", flush=True)
    cloud = load_voxel_npz(args.voxels)
    print(f"[{time.time()-t0:6.1f}s]   {cloud.n_voxels()} voxels, "
          f"|q| range {cloud.q_mag.min():.3f}..{cloud.q_mag.max():.3f} 1/Å",
          flush=True)

    crystal = cual2_crystal(
        a=args.cual2_a if args.cual2_a is not None else 6.066,
        c=args.cual2_c if args.cual2_c is not None else 4.874,
    )

    # 2) seed indexer
    print(f"[{time.time()-t0:6.1f}s] running seed indexer ...", flush=True)
    seed = find_seed_orientation(
        cloud.qx, cloud.qy, cloud.qz, cloud.intensity,
        crystal=crystal,
        n_bright=args.seed_n_bright,
        min_separation=args.seed_min_sep,
        tol_q_rel=args.seed_tol_q_rel,
        tol_angle_deg=args.seed_tol_angle_deg,
        refine_lattice=args.seed_refine_lattice,
        n_refine_steps=400,
        refine_lr=1e-2,
        device=args.device,
    )
    print(f"[{time.time()-t0:6.1f}s]   seed score {seed.score} / "
          f"{args.seed_n_bright} cores, a={seed.a:.4f} c={seed.c:.4f}, "
          f"final_loss={seed.final_loss:.4e}", flush=True)

    seed_path = out_dir / f"seed_layer{cloud.layer_start_filenr}.json"
    with seed_path.open("w") as fh:
        json.dump({
            "U_flat": seed.U_flat().tolist(),
            "a_A": seed.a,
            "c_A": seed.c,
            "score": int(seed.score),
            "n_bright_cores": int(len(seed.matched_centroids)),
            "n_matched": int(sum(h is not None for h in seed.matched_hkls)),
            "final_loss": seed.final_loss,
            "midas_defect_version": __version__,
        }, fh, indent=2)
    print(f"[{time.time()-t0:6.1f}s]   wrote {seed_path}", flush=True)

    # rebuild a Crystal with the refined (a, c) for downstream shell-crossings
    refined_crystal = cual2_crystal(a=seed.a, c=seed.c)

    # 3) rod detection
    print(f"[{time.time()-t0:6.1f}s] running rod detection ...", flush=True)
    rods = find_rods(
        cloud.qx, cloud.qy, cloud.qz, cloud.intensity,
        n_cores=args.rod_n_cores,
        r_tube=args.rod_r_tube,
        L_min=args.rod_L_min,
        L_max=args.rod_L_max,
        N_min_inliers=args.rod_N_min,
        cloud_min_intensity=args.cloud_min_intensity,
        max_voxels_for_scoring=args.rod_max_scoring,
        nms_direction_tol_deg=args.nms_direction_tol_deg,
        nms_pivot_perp_tol=args.nms_pivot_perp_tol,
        crystal=refined_crystal,
        U=seed.U,
        device=args.device,
    )
    print(f"[{time.time()-t0:6.1f}s]   {len(rods)} rods detected", flush=True)
    for i, r in enumerate(rods[:10]):
        hkl = r.defect_normal_hkl
        hstr = f"({hkl[0]}{hkl[1]}{hkl[2]})" if hkl is not None else "?"
        n_shells = len(r.shells_crossed)
        print(f"    rod {i+1}: {hstr}  length={r.length:.2f}  "
              f"ΣI={r.integrated_intensity:.2e}  n_in={r.n_inliers}  "
              f"shells_crossed={n_shells}", flush=True)

    # 4) catalog
    cat_json = out_dir / f"rods_layer{cloud.layer_start_filenr}.json"
    with cat_json.open("w") as fh:
        json.dump({"rods": [r.as_dict() for r in rods],
                   "midas_defect_version": __version__,
                   "voxel_npz": str(args.voxels)},
                  fh, indent=2)
    print(f"[{time.time()-t0:6.1f}s]   wrote {cat_json}", flush=True)

    cat_csv = out_dir / f"rods_layer{cloud.layer_start_filenr}.csv"
    with cat_csv.open("w") as fh:
        fh.write(
            "rank,defect_normal_hkl,dir_x,dir_y,dir_z,pivot_x,pivot_y,pivot_z,"
            "length,n_inliers,integrated_intensity,n_shells_crossed\n"
        )
        for i, r in enumerate(rods):
            hkl = r.defect_normal_hkl or (0, 0, 0)
            fh.write(
                f"{i+1},({hkl[0]} {hkl[1]} {hkl[2]}),"
                f"{r.direction[0]:.6f},{r.direction[1]:.6f},{r.direction[2]:.6f},"
                f"{r.pivot[0]:.6f},{r.pivot[1]:.6f},{r.pivot[2]:.6f},"
                f"{r.length:.4f},{r.n_inliers},"
                f"{r.integrated_intensity:.4e},{len(r.shells_crossed)}\n"
            )
    print(f"[{time.time()-t0:6.1f}s]   wrote {cat_csv}", flush=True)

    # 5) HTML
    if not args.no_html:
        from ..viz import render_rod_overlay_html
        html_path = out_dir / f"rods_layer{cloud.layer_start_filenr}.html"
        print(f"[{time.time()-t0:6.1f}s] rendering HTML ...", flush=True)
        render_rod_overlay_html(
            cloud, rods,
            crystal=refined_crystal,
            html_path=str(html_path),
            n_bright=args.max_bright_html,
            n_haze=args.max_haze_html,
        )
        sz = html_path.stat().st_size / 1e6
        print(f"[{time.time()-t0:6.1f}s]   wrote {html_path}  ({sz:.1f} MB)",
              flush=True)

    print(f"[{time.time()-t0:6.1f}s] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
