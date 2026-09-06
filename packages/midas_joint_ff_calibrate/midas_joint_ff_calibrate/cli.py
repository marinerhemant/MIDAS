"""CLI for midas-joint-ff-calibrate.

Subcommands:
  grain-tx   Refine tx (and Wedge) from reconstructed grain spots — the
             powder-blind geometry the pipeline ran with tx=0. Writes a
             corrected paramstest for a pipeline re-run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-joint-ff-calibrate"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _parse_fix(items) -> dict:
    """Turn ``--fix KEY=V`` / ``--fix KEY=V1,V2,...`` into ``fix_values``.

    Pinning is not the same as freezing. Freezing keeps whatever the parameter
    file happened to say; pinning replaces it with a value you know from
    somewhere else — a lattice measured on a standard, grain positions a
    focused beam already defines — and holds it there while the rest refines.

    A single row broadcasts to every grain, so a LaB6 lattice is six numbers
    rather than six per grain.
    """
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(
                f"--fix wants KEY=VALUE, got {item!r}. "
                "Examples: --fix tx=0.048  "
                "--fix grain_lattice=4.1569,4.1569,4.1569,90,90,90")
        key, raw = item.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if not key or not raw:
            raise SystemExit(f"--fix {item!r} is missing a key or a value.")
        try:
            vals = [float(x) for x in raw.split(",") if x.strip() != ""]
        except ValueError:
            raise SystemExit(
                f"--fix {key}: {raw!r} is not a number or comma-separated "
                "list of numbers.")
        if not vals:
            raise SystemExit(f"--fix {key}: no value given.")
        out[key] = vals[0] if len(vals) == 1 else vals
    return out


def _grain_tx(args) -> int:
    from midas_joint_ff_calibrate.grain_refine import refine_geometry_from_grains

    refine = tuple(s.strip() for s in args.refine.split(",") if s.strip())
    fix_values = _parse_fix(getattr(args, "fix", None))
    res = refine_geometry_from_grains(
        paramstest=args.paramstest, layer_dir=args.layer_dir,
        refine_params=refine, kind=args.kind, max_grains=args.max_grains,
        max_iter=args.max_iter, two_theta_max_deg=args.two_theta_max,
        refine_grain_strain=not args.no_strain, with_powder=args.with_powder,
        strain_bound=args.strain_bound, lattice_source=args.lattice_seed,
        out_paramstest=args.out, device=args.device,
        fix_values=fix_values or None,
    )
    for k, v in (fix_values or {}).items():
        print(f"  pinned {k} = {v}")
    if args.kind != "angular":
        print(f"  NOTE: --kind {args.kind} is accepted but ignored; the loss "
              "is the (Y,Z) position residual either way.")
    print(f"\n  grains={res.n_grains}  matched spots={res.n_spots_matched}  rc={res.rc}")
    print(f"  lattice seed: {args.lattice_seed}   "
          f"grain strain: {'FROZEN at 0' if args.no_strain else 'refined'}")
    print(f"  cost: {res.cost_init:.4e} → {res.cost_final:.4e}")
    strain_ue = getattr(res, "grain_strain_ue", None)
    if strain_ue:
        g = strain_ue
        print(f"  per-grain strain (nuisance): median {g['median']:.0f} µε  "
              f"p90 {g['p90']:.0f} µε  max {g['max']:.0f} µε")
    for k, v in res.refined.items():
        print(f"  {k}: {v:+.6f}")
    for msg in getattr(res, "conditioning", []):
        print(f"  NOTE: {msg}")
    for nm in getattr(res, "at_bounds", []):
        print(f"  *** {nm} finished ON a bound — not a measurement, do not use it ***")
    if res.paramstest_out:
        print(f"  wrote corrected paramstest → {res.paramstest_out}")
    if getattr(res, "at_bounds", []):
        print("  Re-run with fewer free parameters, or more grains.")
        return 1
    return 0


def main(argv=None) -> int:
    p = _midas_make_parser(prog="midas-joint-ff-calibrate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    gx = sub.add_parser("grain-tx", help="refine tx/Wedge from grain spots")
    gx.add_argument("--paramstest", type=Path, required=True,
                    help="paramstest the pipeline ran with (tx≈0, full geometry)")
    gx.add_argument("--layer-dir", type=Path, required=True,
                    help="pipeline layer dir (Grains.csv + SpotMatrix.csv + hkls.csv)")
    gx.add_argument("--refine", default="tx,Wedge",
                    help="comma-separated geometry blocks to refine (default tx,Wedge)")
    gx.add_argument("--fix", action="append", metavar="KEY=VALUE",
                    help="pin a parameter to a value you KNOW, and hold it "
                         "there while the rest refines. Repeatable. Distinct "
                         "from simply leaving it out of --refine, which keeps "
                         "whatever the parameter file said. A single row "
                         "broadcasts to every grain, e.g. a measured LaB6 "
                         "lattice: --fix "
                         "grain_lattice=4.1569,4.1569,4.1569,90,90,90 (the "
                         "SEED lattice, which is frozen either way — to hold "
                         "the grains unstrained use --no-strain or --fix "
                         "grain_strain=0,0,0,0,0,0) ; or focused-beam grain "
                         "positions: --fix grain_pos=0,0,0")
    gx.add_argument("--kind", default="angular", choices=("angular", "internal_angle"),
                    help="ACCEPTED BUT IGNORED. make_residual uses the "
                         "FitMultipleGrains (Y,Z) position loss whatever this "
                         "says; re-deriving observed (R,η) from raw pixels hit "
                         "a flipped-η / broken-2θ convention mismatch. Kept "
                         "only so existing command lines still parse.")
    gx.add_argument("--max-grains", type=int, default=50)
    gx.add_argument("--max-iter", type=int, default=50)
    gx.add_argument("--two-theta-max", type=float, default=20.0)
    gx.add_argument("--no-strain", action="store_true",
                    help="freeze per-grain strain at zero, i.e. hold every "
                         "grain at its seed lattice (default: refine a "
                         "dimensionless per-grain strain jointly with the "
                         "geometry, so a real strain is not charged to tx). "
                         "Costs 6 free parameters per grain — see "
                         "--max-grains. Until issue #70 this flag did nothing "
                         "in either direction.")
    gx.add_argument("--strain-bound", type=float, default=0.02, metavar="EPS",
                    help="half-width of the per-grain strain box, "
                         "dimensionless (default 0.02 = 20000 µε)")
    gx.add_argument("--lattice-seed", default="per_grain",
                    choices=("per_grain", "header"),
                    help="per-grain seed lattice: 'per_grain' (default) uses "
                         "each grain's own fitted a,b,c,α,β,γ from Grains.csv; "
                         "'header' tiles the nominal LatticeConstant, which is "
                         "what every run did before issue #70")
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
