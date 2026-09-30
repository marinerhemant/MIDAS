"""``midas-nf-preprocess seed-orientations`` subcommand.

Three methods, selected with ``--method``:

  - ``cache``       : use the pre-computed lookup files in
                      ``NF_HEDM/seedOrientations/`` (or ``--seed-dir``).
                      Those files are not in git; ``--build-cache`` builds a
                      missing one deterministically.
  - ``from-scratch`` : sample uniformly at ``--resolution-deg`` (default 1.5)
                       and reduce to the FZ.
  - ``from-grains`` : parse an FF-HEDM ``Grains.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dispatch import generate_seeds
from .from_cache import SeedCacheNotFound
from .from_grains import read_grains_orientations
from .io import write_seeds_csv, write_seeds_with_lattice_csv


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        choices=("cache", "from-scratch", "from-grains"),
        required=True,
        help="Generation strategy.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path. With --method=from-grains the format is the "
             "11-column GenSeedOrientationsFF2NFHEDM layout (quat + lattice + "
             "GrainID). For other methods it is a 4-column quaternion CSV.",
    )

    sg_group = parser.add_mutually_exclusive_group()
    sg_group.add_argument(
        "--space-group",
        type=int,
        default=None,
        help="Space group number (1-230). Required for --method=cache and "
             "--method=from-scratch unless --crystal-system is given.",
    )
    sg_group.add_argument(
        "--crystal-system",
        choices=(
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        ),
        default=None,
        help="Friendly crystal-system name; resolved to a representative space group.",
    )

    parser.add_argument(
        "--seed-dir",
        default=None,
        help="Override the cache directory (used by --method=cache). Default: "
             "$MIDAS_NF_SEED_DIR, else NF_HEDM/seedOrientations/ in the source "
             "tree and then ~/.cache/midas/nf_seed_orientations.",
    )
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="With --method=cache: if the cache file for this symmetry is "
             "missing (a clean checkout has none -- they are not in git), build "
             "it deterministically (legacy-cache density, seed 42) into "
             "--seed-dir, else "
             "$MIDAS_NF_SEED_DIR, else ~/.cache/midas/nf_seed_orientations.",
    )
    parser.add_argument(
        "--grains-file",
        default=None,
        help="FF Grains.csv path (required for --method=from-grains).",
    )

    parser.add_argument(
        "--resolution-deg",
        type=float,
        default=1.5,
        help="Target misorientation spacing for --method=from-scratch (default: 1.5).",
    )
    parser.add_argument(
        "--n-master",
        type=int,
        default=None,
        help="Override the resolution-derived sample count (--method=from-scratch).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the Shoemake sampler (--method=from-scratch).",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Skip near-duplicate FZ representative culling.",
    )

    parser.add_argument(
        "--device", default=None, help="cpu | cuda | mps (default: auto-detect)."
    )
    parser.add_argument(
        "--dtype", default=None, help="fp32 | fp64 (default: fp64 on cpu)."
    )


def run(args: argparse.Namespace) -> int:
    method = args.method.replace("-", "_")

    if method == "from_grains":
        if args.grains_file is None:
            raise SystemExit("--grains-file is required when --method=from-grains")
        grains = read_grains_orientations(
            args.grains_file, device=args.device, dtype=args.dtype
        )
        n = write_seeds_with_lattice_csv(grains, args.output)
        print(f"Wrote {args.output} ({n} grains)")
        return 0

    if args.space_group is None and args.crystal_system is None:
        raise SystemExit(
            "--method={cache,from-scratch} requires --space-group or --crystal-system"
        )

    try:
        seeds = generate_seeds(
            method=method,
            space_group=args.space_group,
            crystal_system=args.crystal_system,
            resolution_deg=args.resolution_deg,
            n_master=args.n_master,
            seed=args.seed,
            deduplicate=not args.no_deduplicate,
            seed_dir=args.seed_dir,
            device=args.device,
            dtype=args.dtype,
            build_cache=getattr(args, "build_cache", False),
        )
    except SeedCacheNotFound as exc:
        raise SystemExit(f"error: {exc}")
    write_seeds_csv(seeds, args.output)
    print(f"Wrote {args.output} ({seeds.shape[0]} orientations)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midas-nf-preprocess seed-orientations",
        description="Generate NF-HEDM seed orientations (cache, from scratch, or from FF Grains.csv).",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
