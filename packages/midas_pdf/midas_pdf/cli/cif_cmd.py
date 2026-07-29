"""``midas-pdf-cif`` — inspect a CIF or round-trip it through midas-hkls.

Usage::

    midas-pdf-cif info structure.cif                     # print JSON summary
    midas-pdf-cif convert in.cif out.cif                 # normalise + rewrite
    midas-pdf-cif convert supercell.cif out.xyz          # (future) format map
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..cif import (
    read_cif_to_crystal,
    write_crystal_to_cif,
    parse_cif,
)
from ._common import print_json


def _cmd_info(args: argparse.Namespace) -> int:
    crystal = read_cif_to_crystal(args.path)
    payload = {
        "path": str(args.path),
        "name": crystal.name,
        "space_group": {
            "number": int(crystal.space_group.number),
            "hm_symbol": getattr(crystal.space_group, "hm_symbol", None),
            "crystal_system": getattr(crystal.space_group,
                                        "crystal_system", None),
        },
        "lattice": {
            "a": crystal.lattice.a, "b": crystal.lattice.b,
            "c": crystal.lattice.c,
            "alpha": crystal.lattice.alpha,
            "beta":  crystal.lattice.beta,
            "gamma": crystal.lattice.gamma,
        },
        "atoms_asu": [
            {"element": a.element,
             "fract": [float(x) for x in a.fract],
             "occupancy": float(a.occupancy),
             "B_iso": float(a.B_iso),
             "label": a.label}
            for a in crystal.atoms
        ],
        "n_atoms_asu": int(len(crystal.atoms)),
    }
    print_json(payload)
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    src = Path(args.src)
    dst = Path(args.dst)
    if dst.suffix.lower() != ".cif":
        raise SystemExit(
            f"midas-pdf-cif convert: destination must be a .cif "
            f"(got {dst.suffix!r}). Other output formats are on the roadmap.")
    crystal = read_cif_to_crystal(src)
    written = write_crystal_to_cif(crystal, dst)
    print(f"wrote {written}")
    return 0


def _cmd_raw(args: argparse.Namespace) -> int:
    """Dump the raw parsed CIFData (keys + loops) — useful for debugging
    space-group resolution or atom-loop ingestion."""
    cif = parse_cif(args.path)
    payload = {
        "keys": cif.keys,
        "n_loops": len(cif.loops),
        "loops": [
            {"keys": loop_keys, "n_rows": len(rows)}
            for loop_keys, rows in cif.loops
        ],
    }
    print_json(payload)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="midas-pdf-cif",
        description="Inspect or round-trip a CIF file via midas-hkls.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="print summary of a CIF")
    p_info.add_argument("path", type=Path)
    p_info.set_defaults(func=_cmd_info)

    p_convert = sub.add_parser(
        "convert", help="read CIF and rewrite (normalised)")
    p_convert.add_argument("src", type=Path)
    p_convert.add_argument("dst", type=Path)
    p_convert.set_defaults(func=_cmd_convert)

    p_raw = sub.add_parser("raw", help="dump raw parsed CIFData")
    p_raw.add_argument("path", type=Path)
    p_raw.set_defaults(func=_cmd_raw)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
