"""Command-line interface — drop-in replacement for GetHKLList."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

from .hkl_gen import generate_hkls
from .lattice import Lattice
from .space_group import SpaceGroup, list_space_groups

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-hkls"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _build_parser() -> argparse.ArgumentParser:
    p = _midas_make_parser(prog="midas-hkls", description="Generate HKL list (sginfo replacement)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="Generate an HKL list")
    g.add_argument("--sg", required=True, help="Space group: number, HM symbol, or Hall symbol")
    g.add_argument("--lat", nargs=6, type=float, metavar=("a", "b", "c", "alpha", "beta", "gamma"),
                   required=True, help="Lattice constants (Å, Å, Å, deg, deg, deg)")
    g.add_argument("--wavelength", type=float, required=True, help="X-ray wavelength (Å)")
    g.add_argument("--two-theta-max", type=float, default=None, help="Maximum 2θ in degrees")
    g.add_argument("--d-min", type=float, default=None, help="Minimum d-spacing in Å")
    g.add_argument("--output", "-o", type=Path, default=None, help="CSV output path (default: stdout)")
    g.add_argument("--ext", default="", help="Setting extension (sginfo extension code)")

    z = sub.add_parser(
        "zarr",
        help="Drop-in replacement for GetHKLListZarr. Reads params from a "
             "MIDAS Zarr archive and writes the legacy 11-column hkls.csv.",
    )
    z.add_argument("zarr_path", help="Path to the MIDAS Zarr archive (.zip).")
    z.add_argument("--result-folder", default=None,
                   help="Override the result folder (default: ResultFolder from "
                        "the Zarr archive, falling back to the archive's parent dir).")

    sub.add_parser("list", help="List all 230 space groups in the canonical table")

    info = sub.add_parser("info", help="Print space group info (operations, centering, Laue)")
    info.add_argument("--sg", required=True, help="Space group: number, HM, or Hall")
    info.add_argument("--ops", action="store_true", help="Print all symmetry operations")

    return p


def _resolve_space_group(spec: str, ext: str = "") -> SpaceGroup:
    spec = spec.strip()
    if spec.isdigit():
        return SpaceGroup.from_number(int(spec), extension=ext)
    if " " in spec or spec.startswith("-"):
        return SpaceGroup.from_hall(spec)
    try:
        return SpaceGroup.from_hm(spec)
    except ValueError:
        return SpaceGroup.from_hall(spec)


def _emit_csv(refs, fp) -> None:
    w = csv.writer(fp)
    w.writerow(["ring_nr", "h", "k", "l", "d_spacing", "two_theta_deg", "multiplicity"])
    for r in refs:
        w.writerow([r.ring_nr, r.h, r.k, r.l, f"{r.d_spacing:.8g}",
                    f"{r.two_theta_deg:.8g}", r.multiplicity])


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "list":
        for num, hall, hm in list_space_groups():
            print(f"{num:>3}  {hall:<28}  {hm}")
        return 0

    if args.cmd == "info":
        sg = _resolve_space_group(args.sg)
        print(f"Number       : {sg.number}")
        print(f"Hall symbol  : {sg.hall_symbol!r}")
        print(f"HM symbol    : {sg.hm_symbol!r}")
        print(f"Crystal sys  : {sg.crystal_system}")
        print(f"Laue class   : {sg.laue_class}")
        print(f"Centering    : {sg.centering}")
        print(f"Order        : {sg.order}")
        print(f"Centric      : {sg.is_centrosymmetric()}")
        if args.ops:
            for i, op in enumerate(sg.operations):
                print(f"  {i + 1:3d}: {op.to_xyz()}")
        return 0

    if args.cmd == "zarr":
        from .zarr_compat import generate_hkls_from_zarr
        out = generate_hkls_from_zarr(args.zarr_path, result_folder=args.result_folder)
        print(f"Wrote hkls.csv to {out}", file=sys.stderr)
        return 0

    if args.cmd == "gen":
        sg = _resolve_space_group(args.sg, ext=args.ext)
        lat = Lattice(*args.lat)
        refs = generate_hkls(
            sg, lat,
            wavelength_A=args.wavelength,
            d_min=args.d_min,
            two_theta_max_deg=args.two_theta_max,
        )
        if args.output is None:
            _emit_csv(refs, sys.stdout)
        else:
            with args.output.open("w", newline="") as fp:
                _emit_csv(refs, fp)
            print(f"Wrote {len(refs)} reflections to {args.output}", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
