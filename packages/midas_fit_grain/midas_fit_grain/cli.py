"""CLI entry point — mirrors ``FitPosOrStrainsOMP`` argv shape.

Usage::

    midas-fit-grain paramstest.txt <blockNr> <numBlocks> <numLines> <numProcs> \\
                    [--solver lbfgs] [--loss pixel] [--mode iterative] [--csv]

Slots into ff_MIDAS.py at:
    ``cmd = f\"{bin_dir}/{refine_bin} paramstest.txt {blockNr} ...\"``
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from .config import FitConfig
from .device import apply_cpu_threads, resolve_device, resolve_dtype
from .driver import refine_block_from_disk

LOG = logging.getLogger("midas_fit_grain.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midas-fit-grain",
        description=(
            "PyTorch single/multi-grain refiner — drop-in replacement for "
            "FitPosOrStrainsOMP / FitPosOrStrainsGPU."
        ),
    )
    # Positional args mirror the C binary (paramstest blockNr numBlocks numLines numProcs).
    p.add_argument("param_file", help="paramstest.txt produced by FitSetupParamsAllZarr / ff_MIDAS")
    p.add_argument("block_nr", type=int, nargs="?", default=0,
                   help="0-based block index (default 0)")
    p.add_argument("num_blocks", type=int, nargs="?", default=1,
                   help="total number of blocks (default 1)")
    p.add_argument("num_lines", type=int, nargs="?", default=0,
                   help="number of grain seeds in SpotsToIndex.csv (default: read from disk)")
    p.add_argument("num_procs", type=int, nargs="?", default=0,
                   help="CPU thread count for torch.set_num_threads (default: 0=auto)")

    # Refinement-package-specific switches.
    p.add_argument("--solver", choices=["lbfgs", "adam", "lm", "nelder_mead", "lm_batched"],
                   default="lbfgs",
                   help="Optimizer (default: lbfgs)")
    p.add_argument("--mode", choices=["iterative", "all_at_once"],
                   default=None,
                   help="Iterative re-match phases vs single joint fit. "
                        "Default: iterative if FitAllAtOnce=0, else all_at_once.")
    p.add_argument("--loss", choices=["full3d", "angular", "internal_angle"],
                   default="full3d",
                   help="Residual definition (default: pixel — C parity)")
    p.add_argument("--device", default=None,
                   help="Override MIDAS_FIT_GRAIN_DEVICE (cuda|mps|cpu)")
    p.add_argument("--dtype", default=None,
                   help="Override MIDAS_FIT_GRAIN_DTYPE (float32|float64)")
    p.add_argument("--max-iter", type=int, default=200,
                   help="Outer-iteration cap per phase (default: 200)")
    p.add_argument("--ftol", type=float, default=1e-7,
                   help="Relative-loss convergence threshold (default: 1e-7)")
    p.add_argument("--xtol", type=float, default=1e-9,
                   help="Parameter-delta convergence threshold (default: 1e-9)")
    p.add_argument("--csv", action="store_true",
                   help="Also dump a human-readable FitBest.csv next to the binary")
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose >= 1:
        level = logging.INFO
    if verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    if args.num_procs > 0:
        apply_cpu_threads(args.num_procs, device)

    cfg = FitConfig.from_param_file(
        args.param_file,
        solver=args.solver, mode=args.mode, loss=args.loss,
        device=device, dtype=dtype,
        max_iter=args.max_iter, ftol=args.ftol, xtol=args.xtol,
    )
    LOG.info("midas-fit-grain start  block=%d/%d  solver=%s  mode=%s  loss=%s "
             "device=%s  dtype=%s",
             args.block_nr, args.num_blocks, cfg.solver, cfg.mode, cfg.loss,
             device, dtype)

    n_grains = refine_block_from_disk(
        cfg=cfg, param_file=args.param_file,
        block_nr=args.block_nr, num_blocks=args.num_blocks,
        num_lines=args.num_lines if args.num_lines > 0 else None,
        device=device, dtype=dtype,
        also_write_csv=args.csv,
    )
    LOG.info("done — refined %d grains (block %d/%d)",
             n_grains, args.block_nr, args.num_blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
