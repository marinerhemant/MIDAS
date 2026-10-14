"""CLI for midas-joint-ff-calibrate.

Subcommands:
  grain-tx       Refine tx from reconstructed grain spots — the powder-blind
                 detector roll. Writes a corrected paramstest for a re-run.
  friedel-wedge  Measure the absolute rotation-axis Wedge and the OmegaStep
                 scale from Friedel pairs (the FitWedge.c method).
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


def _friedel_wedge(args) -> int:
    from midas_joint_ff_calibrate.friedel_wedge import friedel_wedge

    res = friedel_wedge(
        args.paramstest, args.layer_dir, pair_source=args.pairs,
        fit_omega_scale=not args.no_omega_scale,
        max_cot=args.max_cot, ome_tol=args.ome_tol, eta_tol=args.eta_tol,
        n_boot=args.n_boot, min_pairs=args.min_pairs,
        use_positions=not args.no_positions, out_paramstest=args.out,
        write_omega_step=args.write_omega_step, loss=args.loss)
    src = (f"from {res.n_grains} grains" if res.pair_source == "grains"
           else "from the raw peak list")
    print(f"\n  Friedel pairs used: {res.n_pairs}  {src}   "
          f"robust omega scatter {res.sigma_deg:.4f} deg")
    if res.pair_source == "peaks":
        print(f"  estimated wrong pairs (decoy windows at 180 +- 3 deg): "
              f"{100 * res.chance_pair_fraction:.1f} %")
    print(f"  Wedge (absolute): {res.wedge_deg:+.5f} +- {res.wedge_se_deg:.5f} deg "
          f"(SE: bootstrap over {'grains' if res.pair_source == 'grains' else 'pairs'}; "
          f"estimator: {res.estimator})")
    if res.class_counts:
        print(f"  pair members on the omega frame grid: {100 * res.snapped_member_fraction:.0f} % "
              f"(phase {res.grid_phase_frames:.3f} frame)")
        for c in sorted(res.class_counts):
            sg = res.class_sigma_deg.get(c)
            print(f"    pairs with {c} snapped member(s): {res.class_counts[c]:6d}"
                  + (f"   residual scatter {sg:.4f} deg" if sg is not None else ""))
    if not args.no_omega_scale:
        print(f"  OmegaStep: {res.omega_step_in:.6f} -> {res.omega_step_fit:.6f} deg/frame "
              f"(relative {res.omega_scale_eps:+.2e} +- {res.omega_scale_se:.1e})")
    if res.splits:
        print("  split check (one fixed axis tilt => one wedge):")
        for name, (w, se, n) in res.splits.items():
            print(f"    {name:26s} {w:+.5f} +- {se:.5f}  (n={n})")
    for msg in res.warnings:
        print(f"  NOTE: {msg}")
    if res.paramstest_out:
        print(f"  wrote {res.paramstest_out} (Wedge"
              f"{' + OmegaStep' if args.write_omega_step else ''})")
    return 0


def main(argv=None) -> int:
    p = _midas_make_parser(prog="midas-joint-ff-calibrate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    gx = sub.add_parser("grain-tx", help="refine tx (detector roll) from grain spots")
    gx.add_argument("--paramstest", type=Path, required=True,
                    help="paramstest the pipeline ran with (tx≈0, full geometry)")
    gx.add_argument("--layer-dir", type=Path, required=True,
                    help="pipeline layer dir (Grains.csv + SpotMatrix.csv + hkls.csv)")
    gx.add_argument("--refine", default="tx",
                    help="comma-separated geometry blocks to refine (default tx). "
                         "Measure Wedge with friedel-wedge: grain-tx compares only "
                         "on-detector (Y,Z) with the grain poses frozen, never the "
                         "observed omega, so one pass recovers only part of a wedge.")
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

    fw = sub.add_parser(
        "friedel-wedge",
        help="measure the absolute rotation-axis Wedge (and the OmegaStep scale) "
             "from Friedel pairs -- use this, not grain-tx, for Wedge")
    fw.add_argument("--paramstest", type=Path, required=True,
                    help="parameter file the reconstruction ran with (the layer "
                         "paramstest.txt or the file given to --params)")
    fw.add_argument("--layer-dir", type=Path, required=True,
                    help="layer dir with SpotMatrix.csv (OmeRaw) and Grains.csv")
    fw.add_argument("--out", type=Path, default=None,
                    help="write a copy of --paramstest with the measured Wedge")
    fw.add_argument("--write-omega-step", action="store_true",
                    help="also write the fitted OmegaStep into --out")
    fw.add_argument("--no-omega-scale", action="store_true",
                    help="hold OmegaStep fixed (NOT recommended: a step error then "
                         "biases the wedge, oppositely above and below the beam)")
    fw.add_argument("--pairs", choices=("peaks", "grains"), default="peaks",
                    help="'peaks' (default): pair the raw peak list, independent of the "
                         "reconstruction; 'grains': pair within each grain's matched "
                         "spots (selected around the run's own predictions)")
    fw.add_argument("--max-cot", type=float, default=3.0,
                    help="drop pairs with |cot eta| above this (near the poles)")
    fw.add_argument("--ome-tol", type=float, default=None,
                    help="pairing window on |omega2 - omega1 - 180| (deg; default "
                         "0.5 peaks / 1.0 grains)")
    fw.add_argument("--eta-tol", type=float, default=None,
                    help="pairing window on the mirrored eta (deg; default 0.6 "
                         "peaks / 1.0 grains)")
    fw.add_argument("--loss", choices=("quant", "ls", "huber"), default="quant",
                    help="'quant' (default): quantisation-aware weighted least squares "
                         "(pairs with omegas on the frame grid are weighted by their own "
                         "variance and trimmed at half a frame per snapped member); 'ls': "
                         "plain least squares; 'huber': the legacy MAD-scaled Huber, biased "
                         "toward 0 when many omegas sit on the frame grid")
    fw.add_argument("--n-boot", type=int, default=100)
    fw.add_argument("--min-pairs", type=int, default=200)
    fw.add_argument("--no-positions", action="store_true",
                    help="ignore grain positions (rays from the origin)")
    fw.set_defaults(func=_friedel_wedge)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
