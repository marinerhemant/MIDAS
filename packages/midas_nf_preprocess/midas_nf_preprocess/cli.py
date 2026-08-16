"""Umbrella CLI: ``midas-nf-preprocess <subcommand> ...``.

Subcommands:

  - ``hex-grid``         : generate the voxel grid (port of MakeHexGrid)
  - ``tomo-filter``      : mask the grid by a tomography image / bbox
  - ``diffr-spots``      : forward-simulate diffraction spots (port of MakeDiffrSpots)
  - ``process-images``   : raw TIFF -> SpotsInfo.bin (port of ProcessImagesCombined)

Each subcommand mirrors its standalone ``python -m`` invocation:

  python -m midas_nf_preprocess.hex_grid <args>          ==
  midas-nf-preprocess hex-grid <args>
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .diffr_spots import cli as diffr_spots_cli
from .hex_grid import cli as hex_grid_cli
from .process_images import cli as process_images_cli
from .seed_orientations import cli as seed_orientations_cli
from .tomo_filter import cli as tomo_filter_cli

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-nf-preprocess"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



_SUBCOMMANDS = {
    "hex-grid": hex_grid_cli,
    "tomo-filter": tomo_filter_cli,
    "diffr-spots": diffr_spots_cli,
    "process-images": process_images_cli,
    "seed-orientations": seed_orientations_cli,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = _midas_make_parser(
        prog="midas-nf-preprocess",
        description="Differentiable PyTorch port of NF-HEDM preprocessing.",
    )
    parser.add_argument(
        "--version", action="version", version=f"midas-nf-preprocess {__version__}"
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", required=True, metavar="<subcommand>"
    )
    for name, mod in _SUBCOMMANDS.items():
        sp = subparsers.add_parser(name, help=mod.__doc__.splitlines()[0] if mod.__doc__ else None)
        mod.add_arguments(sp)
        sp.set_defaults(_run=mod.run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    sys.exit(main())
