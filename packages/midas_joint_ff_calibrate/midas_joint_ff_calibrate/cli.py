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
        out_paramstest=args.out, device=args.device,
        fix_values=fix_values or None,
    )
    for k, v in (fix_values or {}).items():
        print(f"  pinned {k} = {v}")
    print(f"\n  grains={res.n_grains}  matched spots={res.n_spots_matched}  rc={res.rc}")
    print(f"  cost: {res.cost_init:.4e} → {res.cost_final:.4e}")
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
                         "grain_lattice=4.1569,4.1569,4.1569,90,90,90 ; or "
                         "focused-beam grain positions: --fix grain_pos=0,0,0")
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
