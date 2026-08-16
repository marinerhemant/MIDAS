"""Command-line entry point: ``peakfit_torch``.

Drop-in compatible with the C tool's positional args:
    peakfit_torch DataFile blockNr nBlocks numProcs [ResultFolder] [fitPeaks]

Plus our extensions:
    --device {cpu,cuda}
    --dtype {float32,float64}
    --batch-size N
    --validate-against PATH
    --deterministic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from midas_peakfit import __version__
from midas_peakfit.orchestrator import run

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-peakfit"


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
    p = _midas_make_parser(
        prog="peakfit_torch",
        description=(
            "Differentiable PyTorch peak-fitting for FF-HEDM Zarr archives. "
            "Drop-in replacement for PeaksFittingOMPZarrRefactor."
        ),
    )
    p.add_argument("DataFile", help="Path to .MIDAS.zip Zarr archive")
    p.add_argument("blockNr", type=int, help="Block index (0-based)")
    p.add_argument("nBlocks", type=int, help="Total number of blocks")
    p.add_argument("numProcs", type=int, help="Number of CPU workers (advisory)")
    p.add_argument(
        "ResultFolder", nargs="?", default=None,
        help="Override the Zarr-stored ResultFolder",
    )
    p.add_argument(
        "fitPeaks", nargs="?", type=int, default=None,
        help="Override the Zarr-stored doPeakFit (0|1)",
    )
    p.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default=None,
        help="Compute device. Default auto-detects: cuda > mps > cpu. "
             "MPS forces float32 (Apple's MPS backend has no float64 support).",
    )
    p.add_argument(
        "--dtype", choices=["float32", "float64"], default="float64",
        help="Numeric precision for fitting (default: float64)",
    )
    p.add_argument(
        "--batch-size", type=int, default=4096,
        help="Cross-frame region batch threshold (advisory; default 4096)",
    )
    p.add_argument(
        "--validate-against", type=str, default=None,
        help="Path to a C-produced AllPeaks_PS.bin to compare against",
    )
    p.add_argument(
        "--deterministic", action="store_true",
        help="Force deterministic algorithms (fp64 only)",
    )
    p.add_argument(
        "--producer", choices=["process", "thread"], default="process",
        help="CPU producer model (default: process; use 'thread' if pickling overhead matters)",
    )
    p.add_argument(
        "--interleave-blocks", action="store_true",
        help=(
            "Interleave frames across blocks: block N gets every nBlocks-th frame "
            "starting at N. Better load balancing when peak density correlates "
            "with omega (clusters of dense overlapping spots)."
        ),
    )
    p.add_argument(
        "--compute-uncertainty", action="store_true",
        help=(
            "Emit per-peak 1-σ uncertainties (Hessian-based) alongside the "
            "fitted parameters. Adds a sibling AllPeaks_PS_unc.bin file with "
            "9 σ columns per peak (BG, Imax, R, Eta, Mu, σGR, σLR, σGEta, σLEta)."
        ),
    )
    p.add_argument(
        "--moment-uncertainty", action="store_true",
        help=(
            "Emit per-peak model-free moment-based σ alongside fitted "
            "parameters (Modregger et al., J. Appl. Cryst. 58, 1653, 2025). "
            "Adds a sibling AllPeaks_PS_moment.bin file with 6 columns per "
            "peak: M_0, quality_flag (0/1/2 = OK/marginal/deep-Poisson), "
            "u_M1_R [px], u_M1_Eta [deg], u_M2_R [px²], u_M2_Eta [deg²]."
        ),
    )
    p.add_argument("--version", action="version", version=f"peakfit_torch {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(f"peakfit_torch {__version__}")

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}, dtype: {args.dtype}")

    summary = run(
        data_file=args.DataFile,
        block_nr=args.blockNr,
        n_blocks=args.nBlocks,
        num_procs=args.numProcs,
        result_folder_cli=args.ResultFolder,
        fit_peaks_cli=args.fitPeaks,
        device=device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        deterministic=args.deterministic,
        producer=args.producer,
        interleave_blocks=args.interleave_blocks,
        compute_uncertainty=args.compute_uncertainty,
        compute_moments=args.moment_uncertainty,
    )

    if args.validate_against is not None:
        from midas_peakfit.compat.parity import parity_check
        rep = parity_check(args.validate_against, summary["ps_path"])
        print()
        print(rep.summary())
        if not rep.passing():
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
