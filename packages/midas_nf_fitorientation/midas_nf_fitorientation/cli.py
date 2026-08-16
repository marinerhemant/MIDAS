"""Command-line entry points: drop-in replacements for the three C
executables.

Argument signatures intentionally mirror the C executables so existing
shell scripts and workflow drivers can swap binaries without changing
their command lines.

- ``midas-nf-fit-orientation params.txt blockNr nBlocks nCPUs``
- ``midas-nf-fit-parameters  params.txt rowNr [nCPUs]``
- ``midas-nf-fit-multipoint  params.txt [nCPUs]``

Common flags (parsed via ``argparse`` after the positional args):

- ``--device {auto,cpu,cuda}``  default ``auto``
- ``--fp32``                    use float32 (default float64)
- ``--screen-only``             stop after Phase 1, dump screen_cpu.csv
- ``--verbose``                 chatty progress
"""
from __future__ import annotations

import argparse
import sys
from typing import List

import torch

from . import __version__
from .fit_kernel import LBFGSConfig
from .fit_multipoint import fit_multipoint_run
from .fit_orientation import fit_orientation_run
from .fit_parameters import fit_parameters_run

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-nf-fitorientation"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _parse_common(args: List[str]) -> argparse.Namespace:
    """Strip the optional flags from the argv tail, returning a
    namespace with ``device``, ``fp32``, ``screen_only``, ``verbose``,
    and the residual positional args.
    """
    pp = _midas_make_parser(add_help=False)
    pp.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    pp.add_argument("--fp32", action="store_true")
    pp.add_argument("--screen-only", action="store_true")
    pp.add_argument("--verbose", action="store_true")
    # Multi-process block fan-out: MicWriter zeroes the whole output file when
    # it creates it, so concurrent blocks must NOT create. The parent
    # pre-allocates (MicWriter.allocate) and passes this to every worker.
    pp.add_argument("--no-create-output", dest="create_output",
                    action="store_false", default=True,
                    help="Open existing MicFileBinary r+ instead of creating "
                         "and zeroing it. Required when running concurrent "
                         "blocks against one shared output file.")
    pp.add_argument("--lbfgs-max-outer", type=int, default=20)
    pp.add_argument("--lbfgs-max-iter", type=int, default=20)
    pp.add_argument("--refine", default="nm-batched",
                    choices=["nm-batched", "nm-serial", "lbfgs+nm", "lbfgs"],
                    help="(fit-orientation only) Phase-2 refinement strategy. "
                         "'nm-batched' (default) runs the vectorised PyTorch "
                         "Nelder-Mead over every (voxel, winner) in one "
                         "batched forward call per iteration — ~10–20× "
                         "faster than 'nm-serial' on GPU. The other options "
                         "are kept for ablation: 'nm-serial' (per-winner "
                         "scipy NM, parity oracle), 'lbfgs+nm' (soft warmup "
                         "+ NM polish), 'lbfgs' (legacy soft-only).")
    pp.add_argument("--nm-max-iter", type=int, default=200,
                    help="(fit-orientation only) Max NM iterations per "
                         "candidate orientation. C uses 5000.")
    pp.add_argument("--nm-batch-size", type=int, default=4096,
                    help="(fit-orientation, --refine nm-batched only) Max "
                         "(voxel × winner) problems run through one batched "
                         "NM call. Larger batches use more GPU memory.")
    pp.add_argument("-h", "--help", action="store_true")
    return pp.parse_known_args(args)


def _print_version() -> None:
    print(f"midas-nf-fitorientation {__version__}")


# ---------------------------------------------------------------------------
#  fit-orientation (replaces FitOrientationOMP)
# ---------------------------------------------------------------------------

def fit_orientation_main(argv: List[str] | None = None) -> int:
    """CLI for ``midas-nf-fit-orientation``.

    Usage::

        midas-nf-fit-orientation params.txt blockNr nBlocks nCPUs [flags]
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    common, rest = _parse_common(argv)
    _print_version()
    if common.help:
        # An explicit --help is a REQUEST: stdout, exit 0. Sharing this
        # branch with the too-few-arguments case below made `--help`
        # return 1, so every script and smoke test that runs it saw a
        # failure.
        print(fit_orientation_main.__doc__)
        return 0
    if len(rest) < 4:
        print(fit_orientation_main.__doc__, file=sys.stderr)
        return 1
    paramfile, block_nr, n_blocks, n_cpus = rest[:4]
    cfg = LBFGSConfig(
        max_outer=common.lbfgs_max_outer,
        max_iter=common.lbfgs_max_iter,
    )
    fit_orientation_run(
        paramfile,
        block_nr=int(block_nr),
        n_blocks=int(n_blocks),
        n_cpus=int(n_cpus),
        device=common.device,
        dtype=torch.float32 if common.fp32 else torch.float64,
        screen_only=common.screen_only,
        verbose=common.verbose,
        create_output=common.create_output,
        lbfgs_config=cfg,
        refine=common.refine,
        nm_max_iter=common.nm_max_iter,
        nm_batch_size=common.nm_batch_size,
    )
    return 0


# ---------------------------------------------------------------------------
#  fit-parameters (replaces FitOrientationParameters)
# ---------------------------------------------------------------------------

def fit_parameters_main(argv: List[str] | None = None) -> int:
    """CLI for ``midas-nf-fit-parameters``.

    Usage::

        midas-nf-fit-parameters params.txt rowNr [nCPUs] [flags]
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    common, rest = _parse_common(argv)
    _print_version()
    if common.help:
        print(fit_parameters_main.__doc__)
        return 0
    if len(rest) < 2:
        print(fit_parameters_main.__doc__, file=sys.stderr)
        return 1
    paramfile, row_nr = rest[0], int(rest[1])
    n_cpus = int(rest[2]) if len(rest) >= 3 else 1
    cfg = LBFGSConfig(
        max_outer=common.lbfgs_max_outer,
        max_iter=common.lbfgs_max_iter,
    )
    fit_parameters_run(
        paramfile,
        voxel_idx=row_nr,
        n_cpus=n_cpus,
        device=common.device,
        dtype=torch.float32 if common.fp32 else torch.float64,
        verbose=common.verbose,
        lbfgs_config=cfg,
    )
    return 0


# ---------------------------------------------------------------------------
#  fit-multipoint (replaces FitOrientationParametersMultiPoint)
# ---------------------------------------------------------------------------

def fit_multipoint_main(argv: List[str] | None = None) -> int:
    """CLI for ``midas-nf-fit-multipoint``.

    Usage::

        midas-nf-fit-multipoint params.txt [nCPUs] [flags]
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    common, rest = _parse_common(argv)
    _print_version()
    if common.help:
        print(fit_multipoint_main.__doc__)
        return 0
    if len(rest) < 1:
        print(fit_multipoint_main.__doc__, file=sys.stderr)
        return 1
    paramfile = rest[0]
    n_cpus = int(rest[1]) if len(rest) >= 2 else 1
    cfg = LBFGSConfig(
        max_outer=common.lbfgs_max_outer,
        max_iter=common.lbfgs_max_iter,
    )
    fit_multipoint_run(
        paramfile,
        n_cpus=n_cpus,
        device=common.device,
        dtype=torch.float32 if common.fp32 else torch.float64,
        verbose=common.verbose,
        lbfgs_config=cfg,
    )
    return 0


if __name__ == "__main__":
    # `python -m midas_nf_fitorientation.cli {orientation|parameters|multipoint} args...`
    if len(sys.argv) < 2:
        print("Usage: python -m midas_nf_fitorientation.cli "
              "{orientation|parameters|multipoint} <args>")
        sys.exit(1)
    sub = sys.argv[1]
    sub_argv = sys.argv[2:]
    if sub == "orientation":
        sys.exit(fit_orientation_main(sub_argv))
    elif sub == "parameters":
        sys.exit(fit_parameters_main(sub_argv))
    elif sub == "multipoint":
        sys.exit(fit_multipoint_main(sub_argv))
    else:
        print(f"unknown subcommand {sub!r}")
        sys.exit(1)
