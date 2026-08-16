"""CLI dispatch for ``midas-transforms``.

The umbrella command ``midas-transforms <stage> [args]`` plus four
sub-CLIs that mirror the C-binary argv contracts:

  ``midas-merge-peaks <zarr_path>``
  ``midas-calc-radius <zarr_path>``
  ``midas-fit-setup <zarr_path>``
  ``midas-bin-data``    (no positional args, reads from the cwd)

Each sub-CLI is a thin wrapper around the corresponding library function.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-transforms"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _common_argparser(prog: str, description: str) -> argparse.ArgumentParser:
    p = _midas_make_parser(prog=prog, description=description)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    p.add_argument("--dtype", choices=["float32", "float64"], default=None)
    p.add_argument("--version", action="version", version=f"midas-transforms {__version__}")
    return p


def merge_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-merge-peaks", "Frame-by-frame mutual-nearest merge of consolidated peakfit output.")
    p.add_argument("zarr_path", help="Path to the MIDAS Zarr archive (.zip).")
    p.add_argument("--result-folder", default=None,
                   help="Override the result folder (default: directory of zarr_path).")
    p.add_argument("--allpeaks-ps-bin", default=None,
                   help="Override the AllPeaks_PS.bin path "
                        "(default: <result-folder>/Temp/AllPeaks_PS.bin).")
    p.add_argument("--allpeaks-px-bin", default=None,
                   help="Override the AllPeaks_PX.bin path "
                        "(default: <result-folder>/Temp/AllPeaks_PX.bin). "
                        "Used only when UsePixelOverlap=1.")
    p.add_argument("--overlap-length", type=float, default=None,
                   help="Centroid distance threshold in px (default: from Zarr params, fallback 2.0).")
    p.add_argument("--use-pixel-overlap", type=int, choices=[0, 1], default=None,
                   help="Override Zarr's UsePixelOverlap flag (0=centroid, 1=pixel-overlap).")
    args = p.parse_args(argv)

    from .merge import merge_overlapping_peaks
    from .params import read_zarr_params
    rf = Path(args.result_folder) if args.result_folder else Path(args.zarr_path).parent
    zp = read_zarr_params(args.zarr_path)
    overlap = args.overlap_length if args.overlap_length is not None else zp.OverlapLength
    use_px = bool(args.use_pixel_overlap) if args.use_pixel_overlap is not None else None
    merge_overlapping_peaks(
        zarr_path=args.zarr_path,
        allpeaks_ps_bin=args.allpeaks_ps_bin,
        allpeaks_px_bin=args.allpeaks_px_bin,
        result_folder=rf,
        overlap_length=overlap,
        skip_frame=zp.SkipFrame,
        use_maxima_positions=bool(zp.UseMaximaPositions),
        use_pixel_overlap=use_px,
        end_nr=zp.EndNr if zp.EndNr > 0 else None,
        device=args.device, dtype=args.dtype,
        write=True,
    )
    print(f"midas-merge-peaks {__version__}: wrote Result_*.csv and MergeMap.csv to {rf}", file=sys.stderr)
    return 0


def radius_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-calc-radius", "Per-spot ring/Bragg/grain-volume calculation.")
    p.add_argument("zarr_path", help="Path to the MIDAS Zarr archive (.zip).")
    p.add_argument("--result-folder", default=None)
    args = p.parse_args(argv)

    from .params import read_zarr_params
    from .radius import calc_radius
    rf = Path(args.result_folder) if args.result_folder else Path(args.zarr_path).parent
    zp = read_zarr_params(args.zarr_path)
    calc_radius(
        result_folder=rf, zarr_params=zp,
        end_nr=zp.EndNr if zp.EndNr > 0 else None,
        device=args.device, dtype=args.dtype, write=True,
    )
    print(f"midas-calc-radius {__version__}: wrote Radius_*.csv to {rf}", file=sys.stderr)
    return 0


def fit_setup_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-fit-setup", "Per-spot tilt+distortion+wedge correction, filtering, and paramstest.txt writer.")
    p.add_argument("zarr_path", help="Path to the MIDAS Zarr archive (.zip).")
    p.add_argument("--result-folder", default=None)
    p.add_argument("--no-fit", action="store_true", help="Force DoFit=0 (skip the geometry refine).")
    args = p.parse_args(argv)

    from .fit_setup import fit_setup
    from .params import read_zarr_params
    rf = Path(args.result_folder) if args.result_folder else Path(args.zarr_path).parent
    zp = read_zarr_params(args.zarr_path)
    do_fit = False if args.no_fit else (zp.DoFit == 1)
    fit_setup(
        result_folder=rf, zarr_params=zp,
        end_nr=zp.EndNr if zp.EndNr > 0 else None,
        do_fit=do_fit,
        device=args.device, dtype=args.dtype, write=True,
    )
    print(f"midas-fit-setup {__version__}: wrote InputAll.csv et al to {rf}", file=sys.stderr)
    return 0


def bin_data_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-bin-data", "Bin spots into Spots.bin / ExtraInfo.bin / Data.bin / nData.bin.")
    p.add_argument("--result-folder", default=".")
    args = p.parse_args(argv)

    from .bin_data import bin_data
    bin_data(
        result_folder=args.result_folder,
        device=args.device, dtype=args.dtype, write=True,
    )
    print(f"midas-bin-data {__version__}: wrote Spots.bin / ExtraInfo.bin / Data.bin / nData.bin to {args.result_folder}", file=sys.stderr)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Umbrella command: ``midas-transforms <stage> [args]``."""
    parser = _midas_make_parser(
        prog="midas-transforms",
        description="Pure-Python/PyTorch FF-HEDM transforms (merge / radius / fit-setup / bin-data).",
    )
    parser.add_argument("--version", action="version", version=f"midas-transforms {__version__}")
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("merge-peaks", add_help=False)
    sub.add_parser("calc-radius", add_help=False)
    sub.add_parser("fit-setup", add_help=False)
    sub.add_parser("bin-data", add_help=False)
    sub.add_parser("pipeline", add_help=False)

    # Parse only the first positional, dispatch the rest.
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help(sys.stderr)
        return 2
    if argv[0] in ("--version", "-V"):
        parser.parse_args(argv)
        return 0
    # An explicit --help is a REQUEST, not a usage error: it belongs on stdout
    # with exit 0. Without this it fell through to the unknown-stage branch
    # below, which prints to stderr and returns 2 -- so `midas-transforms
    # --help > x` wrote an empty file and any script checking the exit status
    # saw a failure. The no-argv and unknown-stage cases below are genuine
    # usage errors and correctly stay on stderr with 2.
    if argv[0] in ("-h", "--help"):
        parser.print_help(sys.stdout)
        return 0
    stage, rest = argv[0], argv[1:]
    if stage == "merge-peaks":
        return merge_main(rest)
    if stage == "calc-radius":
        return radius_main(rest)
    if stage == "fit-setup":
        return fit_setup_main(rest)
    if stage == "bin-data":
        return bin_data_main(rest)
    if stage == "pipeline":
        return pipeline_main(rest)

    parser.print_help(sys.stderr)
    return 2


def pipeline_main(argv: Optional[List[str]] = None) -> int:
    p = _common_argparser("midas-transforms pipeline", "Run all four stages on-device with no disk round-trips between them.")
    p.add_argument("zarr_path", help="Path to the MIDAS Zarr archive (.zip).")
    p.add_argument("--out-dir", default=None, help="Output directory (default: dir of zarr_path).")
    p.add_argument("--allpeaks-ps-bin", default=None)
    args = p.parse_args(argv)

    from .pipeline import Pipeline
    pipe = Pipeline.from_zarr(
        args.zarr_path,
        allpeaks_ps_bin=args.allpeaks_ps_bin,
        device=args.device, dtype=args.dtype,
    )
    pipe.run()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.zarr_path).parent
    pipe.dump(out_dir)
    print(f"midas-transforms pipeline {__version__}: wrote 9 files to {out_dir}", file=sys.stderr)
    return 0
