"""midas-defect-polytype CLI: per-grain 9R satellite-ladder inventory on a voxel NPZ.

Pipeline:
  1. Load ``voxels_layer*.npz`` -> ``VoxelCloud`` (q-vectors + intensity + omega).
  2. Seed an average orientation ``U`` (``seed_index.find_seed_orientation``) -- in
     the package ``U @ G_crystal`` convention, so no OM transpose is needed.
  3. Pick the activated <111> (``detect_activated_111_axis``).
  4. Build + decontaminate the n*G/3 ladder (fundamentals vs forbidden-gap 9R
     satellites), resolve the G/3 & 2G/3 doublets with the omega Ewald-artifact
     test, fit the modulation tilt, and compute the periodic/aperiodic balance.
  5. Write a JSON summary.

This emits aggregate + polarity-level numbers only -- never a parent-grain vs
twin-lamella identity (FF cannot attribute it; see :mod:`midas_defect.attribution`).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .. import __version__
from ..data_io import load_voxel_npz
from ..lattice import fcc_cu_crystal
from ..polytype import (
    build_satellite_ladder,
    decontaminate_ladder,
    detect_activated_111_axis,
    fit_modulation_tilt,
    periodic_aperiodic_balance,
    resolve_satellite_doublet,
)
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
    p.add_argument("--voxels", required=True, help="voxels_layer*.npz path")
    p.add_argument("--out", required=True, help="output JSON summary path")
    p.add_argument("--a-fcc", type=float, default=3.6356, help="FCC a (Å)")
    p.add_argument("--seed-n-bright", type=int, default=20)
    p.add_argument("--seed-tol-q-rel", type=float, default=0.02)
    p.add_argument("--seed-tol-angle-deg", type=float, default=5.0)
    p.add_argument("--decontam-tol", type=float, default=0.25,
                   help="fundamental-vs-satellite 3D distance tol (1/Å)")
    p.add_argument("--device", default="cpu")


def main(argv: list[str] | None = None) -> int:
    parser = _midas_make_parser(
        prog="midas-defect-polytype",
        description="Per-grain 9R satellite-ladder inventory from a voxel NPZ.")
    _add_args(parser)
    args = parser.parse_args(argv)
    t0 = time.time()

    cloud = load_voxel_npz(args.voxels)
    q = np.stack([cloud.qx, cloud.qy, cloud.qz], axis=1)
    I = cloud.intensity
    om = cloud.omega_deg
    print(f"[{time.time()-t0:5.1f}s] {cloud.n_voxels()} voxels loaded", flush=True)

    B0 = 2 * math.pi / args.a_fcc
    G = math.sqrt(3) * B0
    g3 = G / 3.0

    seed = find_seed_orientation(
        cloud.qx, cloud.qy, cloud.qz, I,
        n_bright=args.seed_n_bright, tol_q_rel=args.seed_tol_q_rel,
        tol_angle_deg=args.seed_tol_angle_deg, device=args.device)
    U = np.asarray(seed.U, float)
    print(f"[{time.time()-t0:5.1f}s] seed U scored {seed.score}", flush=True)

    act = detect_activated_111_axis(q, I, G, crystal_to_sample_OM=U)
    axis = np.asarray(act["a_sample"], float)

    cr = fcc_cu_crystal()
    lad = decontaminate_ladder(
        build_satellite_ladder(q, I, axis, G), U, cr, tol_inv_A=args.decontam_tol)

    doublets = {}
    splits, orders = [], []
    for nm, n in (("G/3", 1), ("2G/3", 2)):
        d = resolve_satellite_doublet(q, I, om, axis, n * g3)
        doublets[nm] = {"verdict": d.verdict, "n_members": d.n_members,
                        "azimuth_deg": d.azimuth_deg,
                        "is_twin_polarity": d.is_twin_polarity}
        if d.n_members == 2:
            pa = np.asarray(d.members[0]["perp_vec"])
            pb = np.asarray(d.members[1]["perp_vec"])
            splits.append(float(np.linalg.norm(pa - pb)))
            orders.append(n)

    tilt = fit_modulation_tilt(orders, splits, g3) if len(orders) >= 2 else None
    balance = periodic_aperiodic_balance(q, I, axis, G)

    summary = {
        "version": __version__,
        "voxels": args.voxels,
        "n_voxels": int(cloud.n_voxels()),
        "a_fcc": args.a_fcc,
        "G_magnitude": G,
        "activated_axis_sample": axis.tolist(),
        "activated_axis_intensities": np.asarray(act["all_intensities"]).tolist(),
        "ladder": {
            "n_fundamentals": lad.metadata.get("n_fundamentals"),
            "n_satellites": lad.metadata.get("n_satellites"),
            "rungs": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in r.items()} for r in lad.rungs],
        },
        "doublets": doublets,
        "modulation_tilt": tilt,
        "fault_balance": {k: v for k, v in balance.items() if k != "per_position"},
        "attribution_note": ("members are polarity/character only; parent-grain vs "
                             "twin-lamella needs pf-/NF-HEDM (attribution.py)"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[{time.time()-t0:5.1f}s] wrote {out}  "
          f"({summary['ladder']['n_fundamentals']} fund, "
          f"{summary['ladder']['n_satellites']} satellites)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
