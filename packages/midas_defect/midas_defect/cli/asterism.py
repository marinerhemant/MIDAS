"""midas-defect-asterism CLI: per-hkl 3-D Gaussian fits + ODF aggregation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .. import __version__
from ..asterism_fit import fit_asterism_patches
from ..data_io import load_voxel_npz
from ..lattice import cual2_crystal
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
                   help="directory to write asterism catalog")
    p.add_argument("--seed-json", default=None,
                   help="optional path to a seed_layerXXX.json produced by "
                        "midas-defect-rods; if omitted, run seed indexer here")
    p.add_argument("--cual2-a", type=float, default=6.066)
    p.add_argument("--cual2-c", type=float, default=4.874)
    p.add_argument("--q-max", type=float, default=10.0,
                   help="max |q| (1/Å) for hkl prediction")
    p.add_argument("--crop-half", type=float, default=0.10,
                   help="base crop half-width (1/Å) around each predicted hkl")
    p.add_argument("--crop-q-scale", type=float, default=0.03,
                   help="extra half-width scaled by |q|")
    p.add_argument("--min-voxels", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--sigma-init", type=float, default=0.05)
    p.add_argument("--loss", choices=["lsq", "sqrt_w", "poisson"],
                   default="lsq",
                   help="fit loss: lsq | sqrt_w (Poisson-style weighting) | "
                        "poisson (full NLL)")
    p.add_argument("--output-residual", default=None,
                   help="if set, write a Bragg-residual voxel NPZ to this path. "
                        "Run midas-defect-rods on the residual to find rods that "
                        "remain after Bragg subtraction.")
    p.add_argument("--residual-det-bin", type=int, default=4,
                   help="det_bin used in the input NPZ (needed to rehydrate "
                        "indices for the residual NPZ)")
    p.add_argument("--seed-n-bright", type=int, default=50)
    p.add_argument("--seed-tol-q-rel", type=float, default=0.05)
    p.add_argument("--seed-tol-angle-deg", type=float, default=10.0)
    p.add_argument("--device", default=None)


def main(argv: list[str] | None = None) -> int:
    parser = _midas_make_parser(prog="midas-defect-asterism",
                                     description=__doc__)
    _add_args(parser)
    args = parser.parse_args(argv)

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.time()-t0:6.1f}s] loading {args.voxels} ...", flush=True)
    cloud = load_voxel_npz(args.voxels)
    print(f"[{time.time()-t0:6.1f}s]   {cloud.n_voxels()} voxels, "
          f"|q| range {cloud.q_mag.min():.3f}..{cloud.q_mag.max():.3f} 1/Å",
          flush=True)

    crystal = cual2_crystal(a=args.cual2_a, c=args.cual2_c)

    if args.seed_json:
        with open(args.seed_json) as fh:
            seed = json.load(fh)
        U = np.asarray(seed["U_flat"]).reshape(3, 3)
        a, c = float(seed["a_A"]), float(seed["c_A"])
        print(f"[{time.time()-t0:6.1f}s] using seed from {args.seed_json}: "
              f"a={a:.4f} c={c:.4f}", flush=True)
    else:
        print(f"[{time.time()-t0:6.1f}s] running seed indexer ...", flush=True)
        seed = find_seed_orientation(
            cloud.qx, cloud.qy, cloud.qz, cloud.intensity,
            crystal=crystal,
            n_bright=args.seed_n_bright,
            tol_q_rel=args.seed_tol_q_rel,
            tol_angle_deg=args.seed_tol_angle_deg,
            refine_lattice=True,
            n_refine_steps=400, refine_lr=1e-2,
            device=args.device,
        )
        U, a, c = seed.U, seed.a, seed.c
        print(f"[{time.time()-t0:6.1f}s]   seed score {seed.score}/"
              f"{args.seed_n_bright}, a={a:.4f} c={c:.4f}", flush=True)

    print(f"[{time.time()-t0:6.1f}s] fitting asterism patches up to "
          f"|q|={args.q_max} 1/Å ...", flush=True)
    fits = fit_asterism_patches(
        cloud.qx, cloud.qy, cloud.qz, cloud.intensity,
        U=U, a=a, c=c, crystal=crystal,
        q_max_inv_A=args.q_max,
        crop_halfwidth=args.crop_half,
        crop_q_scale=args.crop_q_scale,
        min_voxels=args.min_voxels,
        sigma_init=args.sigma_init,
        n_steps=args.n_steps, lr=args.lr,
        loss_kind=args.loss,
        device=args.device,
    )
    n_pred_total = len(fits)   # only fits actually returned
    print(f"[{time.time()-t0:6.1f}s]   fit {n_pred_total} hkls", flush=True)

    # catalog
    cat_csv = out_dir / f"asterism_layer{cloud.layer_start_filenr}.csv"
    with cat_csv.open("w") as fh:
        fh.write("h,k,l,"
                 "q_pred_x,q_pred_y,q_pred_z,"
                 "q_fit_x,q_fit_y,q_fit_z,"
                 "amplitude,baseline,"
                 "sigma_min,sigma_mid,sigma_max,"
                 "isotropy,n_voxels,integrated_intensity,final_loss\n")
        for f in fits:
            sg = np.sort(f.sigma_eig)
            fh.write(
                f"{f.hkl[0]},{f.hkl[1]},{f.hkl[2]},"
                f"{f.q_pred[0]:.6f},{f.q_pred[1]:.6f},{f.q_pred[2]:.6f},"
                f"{f.q_fit[0]:.6f},{f.q_fit[1]:.6f},{f.q_fit[2]:.6f},"
                f"{f.amplitude:.4e},{f.baseline:.4e},"
                f"{sg[0]:.5f},{sg[1]:.5f},{sg[2]:.5f},"
                f"{f.isotropy():.4f},{f.n_voxels},"
                f"{f.integrated_intensity:.4e},{f.final_loss:.4e}\n"
            )
    print(f"[{time.time()-t0:6.1f}s]   wrote {cat_csv}", flush=True)

    cat_json = out_dir / f"asterism_layer{cloud.layer_start_filenr}.json"
    with cat_json.open("w") as fh:
        json.dump({
            "fits": [
                {
                    "hkl": list(f.hkl),
                    "q_pred": f.q_pred.tolist(),
                    "q_fit":  f.q_fit.tolist(),
                    "amplitude": f.amplitude,
                    "baseline": f.baseline,
                    "sigma_eig": f.sigma_eig.tolist(),
                    "sigma_axes": f.sigma_axes.tolist(),
                    "isotropy": f.isotropy(),
                    "n_voxels": f.n_voxels,
                    "integrated_intensity": f.integrated_intensity,
                    "final_loss": f.final_loss,
                    "converged": f.converged,
                }
                for f in fits
            ],
            "U_flat": np.asarray(U).reshape(-1).tolist(),
            "a_A": float(a),
            "c_A": float(c),
            "midas_defect_version": __version__,
            "voxel_npz": str(args.voxels),
        }, fh, indent=2)
    print(f"[{time.time()-t0:6.1f}s]   wrote {cat_json}", flush=True)

    # short summary
    if fits:
        iso = np.array([f.isotropy() for f in fits])
        sigma_max = np.array([f.sigma_eig.max() for f in fits])
        print(f"[{time.time()-t0:6.1f}s] summary: isotropy median {np.median(iso):.3f}, "
              f"min {iso.min():.3f}, max {iso.max():.3f}; "
              f"σ_max median {np.median(sigma_max):.4f} 1/Å",
              flush=True)

    # residual-cloud NPZ
    if args.output_residual is not None:
        from ..asterism_fit import build_bragg_residual_intensity
        from ..data_io import save_voxel_npz_from_cloud
        print(f"[{time.time()-t0:6.1f}s] building Bragg residual cloud ...",
              flush=True)
        resid = build_bragg_residual_intensity(
            cloud.qx, cloud.qy, cloud.qz, cloud.intensity, fits,
            clip_negative=True,
        )
        save_voxel_npz_from_cloud(
            cloud, resid, args.output_residual,
            det_bin=args.residual_det_bin,
        )
        kept = int((resid > 0).sum())
        print(f"[{time.time()-t0:6.1f}s]   wrote residual NPZ "
              f"{args.output_residual}  ({kept} non-zero voxels remain)",
              flush=True)

    print(f"[{time.time()-t0:6.1f}s] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
