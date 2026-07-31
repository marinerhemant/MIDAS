"""Umbrella CLI: ``midas-nf-pipeline <subcommand> ...``.

Subcommands::

    run            Run the full NF pipeline for one or more layers
                   (single-resolution or multi-resolution).
    parse-mic      Run the ParseMic byte-parity port.
    mic2grains     Run the Mic2GrainsList byte-parity port.
    consolidate    (Re-)build a consolidated HDF5 from existing outputs.
    refine-params  Calibration / parameter-refinement modes from
                   midas-nf-fitorientation (single-point + multi-point).

Single-resolution is just multi-resolution with ``NumLoops=0``, so the
``run`` subcommand handles both. Multi-layer is always supported via
``--start-layer`` / ``--end-layer`` (default: 1..1).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__


# ---------------------------------------------------------------------------
#  Subcommands
# ---------------------------------------------------------------------------

def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("paramFN", help="MIDAS parameter file.")
    p.add_argument("--n-cpus", "-nCPUs", dest="nCPUs", type=int, default=1)
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda (default: auto)")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "fp32", "fp64", "float32", "float64"],
                   help="auto = fp32 on cuda/mps, fp64 on cpu (matches the "
                        "midas_nf_preprocess default).")
    p.add_argument("--fit-gpus", dest="fitGpus", default="",
                   help="Comma-separated GPU ids to shard the FITTING stage "
                        "across, e.g. '0,1'. Each GPU gets a disjoint block of "
                        "voxels in its own process, writing into one shared "
                        "pre-allocated output. Empty = single process.")
    p.add_argument("--refine", default="nm-batched",
                   choices=["nm-triton", "nm-batched", "nm-serial",
                            "lbfgs+nm", "lbfgs"],
                   help="midas-nf-fitorientation phase-2 refine strategy. "
                        "nm-triton is auto-selected on cuda when Triton is "
                        "installed (fastest); nm-batched is the safe default.")
    p.add_argument("--skip-validation", action="store_true",
                   help="skip midas-params preflight on the parameter file.")
    p.add_argument("--strict-validation", action="store_true",
                   help="exit on midas-params validation errors (default: warn).")
    p.add_argument("--ff-seed-orientations", dest="ffSeedOrientations",
                   action="store_true",
                   help="Loop 0 starts from FF Grains.csv seeds "
                        "(default: cache-based seed generation).")
    p.add_argument("--no-image-processing", dest="doImageProcessing",
                   action="store_const", const=0, default=1,
                   help="Skip ProcessImagesCombined (assumes SpotsInfo.bin "
                        "already exists in DataDirectory).")
    p.add_argument("--start-layer", dest="startLayerNr", type=int, default=1)
    p.add_argument("--end-layer", dest="endLayerNr", type=int, default=1)
    p.add_argument("--result-folder", dest="resultFolder", default="",
                   help="Overrides OutputDirectory in param file. "
                        "Per-layer outputs go into <resultFolder>/LayerNr_<n>/.")
    p.add_argument("--min-confidence", dest="minConfidence", type=float, default=0.6,
                   help="MinConfidence for the per-layer Mic2GrainsList run "
                        "after the last loop (default: 0.6).")
    p.add_argument("--resume", default="",
                   help="Path to pipeline H5 to resume from "
                        "(auto-detects last completed stage).")
    p.add_argument("--restart-from", dest="restartFrom", default="",
                   help="Stage label to restart from "
                        "(e.g. loop_1_seeded, loop_2_unseeded).")
    p.add_argument("--install-dir", default=None,
                   help="Path to MIDAS install (for the seedOrientations cache; "
                        "default: env var MIDAS_INSTALL_DIR / MIDAS_HOME).")


def _resolve_nf_dtype(device: str, dtype_arg: str) -> str:
    """``auto`` → fp32 on cuda/mps, fp64 on cpu. Aliases fp32→float32 etc."""
    if dtype_arg == "auto":
        # auto-detect cuda when device=auto
        if device == "auto":
            try:
                import torch
                effective = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                effective = "cpu"
        else:
            effective = device
        return "float32" if effective in ("cuda", "mps") else "float64"
    return {"fp32": "float32", "fp64": "float64"}.get(dtype_arg, dtype_arg)


def cmd_run(args: argparse.Namespace) -> int:
    import logging as _logging
    _logger = _logging.getLogger("midas_nf_pipeline.cli")

    # midas-params preflight (soft dependency, mirrors FF pipeline).
    try:
        from midas_params.hook import preflight_validate
    except ImportError:
        preflight_validate = None
    if preflight_validate is not None:
        ok = preflight_validate(
            param_file=args.paramFN,
            pipeline="nf",
            skip=getattr(args, "skip_validation", False),
            strict=getattr(args, "strict_validation", False),
            logger=_logger,
        )
        if not ok:
            return 1

    # Resolve --dtype auto.
    args.dtype = _resolve_nf_dtype(args.device, args.dtype)
    if getattr(args, "_dtype_was_auto", False) or True:
        _logger.info("auto: dtype=%s (device=%s)", args.dtype, args.device)

    from .workflows import run_multi_layer
    install_dir = (
        args.install_dir
        or __import__("os").environ.get("MIDAS_INSTALL_DIR")
        or __import__("os").environ.get("MIDAS_HOME")
    )
    out_h5 = run_multi_layer(args, install_dir=install_dir)
    print(f"Pipeline finished. Consolidated H5 outputs: {out_h5}")
    return 0


def _add_parse_mic_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("paramFN", help="MIDAS parameter file (defines "
                                   "MicFileBinary, MicFileText, ...).")


def cmd_parse_mic(args: argparse.Namespace) -> int:
    from .parse_mic import parse_mic_from_paramfile
    out = parse_mic_from_paramfile(args.paramFN)
    print(f"ParseMic produced: {sorted(out.values())}")
    return 0


def _add_mic2grains_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("paramFN")
    p.add_argument("micFile")
    p.add_argument("outFile")
    p.add_argument("doNeighborSearch", nargs="?", type=int, default=0)
    p.add_argument("nCPUs", nargs="?", type=int, default=1)
    p.add_argument("minConfOverride", nargs="?", type=float, default=None)


def cmd_mic2grains(args: argparse.Namespace) -> int:
    from .mic2grains import Mic2GrainsParams, mic_to_grains
    n = mic_to_grains(Mic2GrainsParams(
        param_file=args.paramFN,
        mic_file=args.micFile,
        out_file=args.outFile,
        do_neighbor_search=args.doNeighborSearch,
        n_cpus=args.nCPUs,
        min_conf_override=args.minConfOverride,
    ))
    print(f"Values written: {n} unique grains found.")
    return 0


def _add_consolidate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("micFile", help="Path to a text .mic file.")
    p.add_argument("--paramFN", default=None,
                   help="Optional parameter file (text stored in provenance).")
    p.add_argument("--output", default=None,
                   help="Output H5 path (default: <stem>_consolidated.h5).")


def cmd_consolidate(args: argparse.Namespace) -> int:
    from .consolidate import generate_consolidated_hdf5
    param_text = ""
    if args.paramFN and Path(args.paramFN).exists():
        with open(args.paramFN) as f:
            param_text = f.read()
    out = generate_consolidated_hdf5(
        args.micFile, param_text=param_text, output_path=args.output,
    )
    print(f"Consolidated H5 saved: {out}")
    return 0


def _add_refine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("paramFN")
    p.add_argument("--multi-point", action="store_true",
                   help="Use FitOrientationParametersMultiPoint mode.")
    p.add_argument("--row-nr", type=int, default=0,
                   help="Single-point row number (0-based, ignored for "
                        "--multi-point).")
    p.add_argument("--n-cpus", dest="nCPUs", type=int, default=1)
    p.add_argument("--device", default="auto")


def cmd_refine_params(args: argparse.Namespace) -> int:
    if args.multi_point:
        from midas_nf_fitorientation import fit_multipoint_run
        fit_multipoint_run(args.paramFN, n_cpus=args.nCPUs, device=args.device)
    else:
        from midas_nf_fitorientation import fit_parameters_run
        # fit_parameters_run(paramfile, voxel_idx, n_cpus, *, device, ...) --
        # there is no `row_nr` parameter and `voxel_idx` is a required
        # positional (fit_parameters.py:45-54).
        fit_parameters_run(
            args.paramFN, args.row_nr,
            n_cpus=args.nCPUs, device=args.device,
        )
    return 0


# ---------------------------------------------------------------------------
#  Top-level parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midas-nf-pipeline",
        description="Pure-Python NF-HEDM pipeline orchestrator.",
    )
    p.add_argument("--version", action="version",
                   version=f"midas-nf-pipeline {__version__}")
    sub = p.add_subparsers(dest="subcommand", required=True, metavar="<subcommand>")

    sp = sub.add_parser("run", help="Run the NF pipeline (single + multi-layer).")
    _add_run_args(sp)
    sp.set_defaults(_run=cmd_run)

    sp = sub.add_parser("parse-mic", help="Run ParseMic.")
    _add_parse_mic_args(sp)
    sp.set_defaults(_run=cmd_parse_mic)

    sp = sub.add_parser("mic2grains", help="Run Mic2GrainsList.")
    _add_mic2grains_args(sp)
    sp.set_defaults(_run=cmd_mic2grains)

    sp = sub.add_parser("consolidate", help="(Re-)build a consolidated HDF5.")
    _add_consolidate_args(sp)
    sp.set_defaults(_run=cmd_consolidate)

    sp = sub.add_parser("refine-params", help="Parameter refinement.")
    _add_refine_args(sp)
    sp.set_defaults(_run=cmd_refine_params)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    sys.exit(main())
