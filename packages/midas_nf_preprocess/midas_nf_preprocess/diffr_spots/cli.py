"""``midas-nf-preprocess diffr-spots`` subcommand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .params import DiffrSpotsParams
from .pipeline import DiffrSpotsPipeline


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "parameter_file", help="MIDAS parameter file (Lsd, SeedOrientations, etc.)"
    )
    parser.add_argument(
        "--device", default=None, help="cpu | cuda | mps (default: auto-detect)."
    )
    parser.add_argument(
        "--dtype", default=None, help="fp32 | fp64 (default: fp64 on cpu)."
    )
    parser.add_argument(
        "--hkls-csv",
        default=None,
        help=(
            "Override hkls.csv path (default: <output dir>/hkls.csv, i.e. "
            "--output-dir, then OutputDirectory; DataDirectory only if absent there)."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Override SeedOrientations path (default: from parameter file).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for Key.bin / DiffractionSpots.bin / OrientMat.bin.",
    )


def run(args: argparse.Namespace) -> int:
    params = DiffrSpotsParams.from_paramfile(args.parameter_file)
    pipe = DiffrSpotsPipeline(
        params,
        device=args.device,
        dtype=args.dtype,
        hkls_csv=args.hkls_csv,
        seed_orientations_csv=args.seeds,
        output_dir=args.output_dir,
    )
    print(f"hkls.csv: {pipe.hkls_csv_path}")
    result, paths = pipe.run(output_dir=args.output_dir)
    total = int(result.counts.sum().item())
    print(
        f"Wrote {len(paths)} files to {Path(paths['Key.bin']).parent} "
        f"({pipe.n_orientations} orientations, {total} total spots)"
    )
    for name, p in paths.items():
        print(f"  {name}: {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midas-nf-preprocess diffr-spots",
        description="Differentiable diffraction-spot prediction (port of MakeDiffrSpots).",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
