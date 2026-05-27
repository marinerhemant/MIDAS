"""CLI for midas-joint-ff-calibrate.

Subcommands:
  grain-tx   Refine tx (and Wedge) from reconstructed grain spots — the
             powder-blind geometry the pipeline ran with tx=0. Writes a
             corrected paramstest for a pipeline re-run.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _grain_tx(args) -> int:
    from midas_joint_ff_calibrate.grain_refine import refine_geometry_from_grains

    refine = tuple(s.strip() for s in args.refine.split(",") if s.strip())
    res = refine_geometry_from_grains(
        paramstest=args.paramstest, layer_dir=args.layer_dir,
        refine_params=refine, kind=args.kind, max_grains=args.max_grains,
        max_iter=args.max_iter, two_theta_max_deg=args.two_theta_max,
        refine_grain_strain=not args.no_strain, with_powder=args.with_powder,
        out_paramstest=args.out, device=args.device,
    )
    print(f"\n  grains={res.n_grains}  matched spots={res.n_spots_matched}  rc={res.rc}")
    print(f"  cost: {res.cost_init:.4e} → {res.cost_final:.4e}")
    for k, v in res.refined.items():
        print(f"  {k}: {v:+.6f}")
    if res.paramstest_out:
        print(f"  wrote corrected paramstest → {res.paramstest_out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="midas-joint-ff-calibrate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    gx = sub.add_parser("grain-tx", help="refine tx/Wedge from grain spots")
    gx.add_argument("--paramstest", type=Path, required=True,
                    help="paramstest the pipeline ran with (tx≈0, full geometry)")
    gx.add_argument("--layer-dir", type=Path, required=True,
                    help="pipeline layer dir (Grains.csv + SpotMatrix.csv + hkls.csv)")
    gx.add_argument("--refine", default="tx,Wedge",
                    help="comma-separated geometry blocks to refine (default tx,Wedge)")
    gx.add_argument("--kind", default="angular", choices=("angular", "internal_angle"),
                    help="η-sensitive loss; 'pixel' is disabled (blind to tx)")
    gx.add_argument("--max-grains", type=int, default=50)
    gx.add_argument("--max-iter", type=int, default=50)
    gx.add_argument("--two-theta-max", type=float, default=20.0)
    gx.add_argument("--no-strain", action="store_true",
                    help="freeze per-grain lattice (default: refine strain)")
    gx.add_argument("--with-powder", action="store_true",
                    help="full joint (powder + grains); not yet wired here")
    gx.add_argument("--out", type=Path, default=None,
                    help="write corrected paramstest here for the re-run")
    gx.add_argument("--device", default="cpu")
    gx.set_defaults(func=_grain_tx)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
