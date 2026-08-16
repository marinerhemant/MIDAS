"""CLI entry point for midas-index.

Drop-in replacement for the IndexerOMP / IndexerGPU positional argv:

    midas-index <param_file> <block_nr> <n_blocks> <n_spots_to_index> <num_procs>

Optional flags (extension over C binaries):

    --device {cpu,cuda,mps}       override auto-detection (env: MIDAS_INDEX_DEVICE)
    --dtype  {float32,float64}    override per-device default (env: MIDAS_INDEX_DTYPE)
    --version                     print version and exit

Behaviour mirrors `FF_HEDM/src/IndexerOMP.c::main` exactly:
  - Reads <param_file> for all 33 paramstest.txt keys.
  - Computes [startRowNr, endRowNr] from (block_nr, n_blocks, n_spots_to_index).
  - Writes BestPos_<block_nr>.csv plus consolidated binaries to OutputFolder.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-index"


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
    parser = _midas_make_parser(
        prog="midas-index",
        description="Pure-Python/PyTorch FF-HEDM indexer (drop-in for IndexerOMP/IndexerGPU).",
    )
    parser.add_argument("param_file", help="Path to paramstest.txt")
    parser.add_argument("block_nr", type=int, help="Block index for sharded run")
    parser.add_argument("n_blocks", type=int, help="Total number of blocks")
    parser.add_argument(
        "n_spots_to_index", type=int, help="Total number of seed spots to process"
    )
    parser.add_argument(
        "num_procs",
        type=int,
        help="Threads on CPU (passed to torch.set_num_threads); ignored on GPU/MPS",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default=None)
    parser.add_argument(
        "--group-size", type=int, default=None,
        help="Seeds packed into a single forward+match batch. Smaller means "
             "less peak memory, larger means more throughput. Default "
             "auto-picks per device (cuda=64, mps=8, cpu=8). Overrides "
             "env MIDAS_INDEX_GROUP_SIZE.",
    )
    parser.add_argument("--version", action="version", version=f"midas-index {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    import numpy as np

    from .indexer import Indexer
    from .io.output import (
        close_output_files,
        open_output_files,
        write_full_record,
        write_seed_record,
    )

    indexer = Indexer.from_param_file(
        args.param_file, device=args.device, dtype=args.dtype,
    )
    indexer.load_observations()
    obs = indexer._observations
    assert obs is not None

    spot_ids = obs["spot_ids"]
    n_total = int(min(args.n_spots_to_index, len(spot_ids)))

    result = indexer.run(
        block_nr=args.block_nr,
        n_blocks=args.n_blocks,
        n_spots_to_index=n_total,
        num_procs=args.num_procs,
        seed_group_size=args.group_size,
    )

    # Build a spot_id -> offset map (offset is the row in SpotsToIndex.csv).
    sid_to_offset: dict[int, int] = {
        int(sid): i for i, sid in enumerate(spot_ids[:n_total].tolist())
    }

    output_folder = indexer.params.OutputFolder
    fd_best, fd_full = open_output_files(output_folder, n_total, args.block_nr)
    try:
        for seed in result.seeds:
            offset = sid_to_offset.get(int(seed.spot_id), -1)
            if offset < 0:
                continue
            write_seed_record(fd_best, seed, offset)
            if seed.matched_pairs is not None:
                pairs_np = seed.matched_pairs.numpy().astype(np.float64)
                write_full_record(fd_full, pairs_np, offset)
    finally:
        close_output_files(fd_best, fd_full)

    print(
        f"midas-index {__version__}: block {args.block_nr}/{args.n_blocks} "
        f"completed. {len(result.seeds)} seeds indexed -> {output_folder}/IndexBest.bin",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
