#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
integrate_and_refine.py — Combined MIDAS Integration + Peak Fitting / Refinement

Runs the full pipeline from raw diffraction data to refined crystal structures:

    1. MIDAS integration  (integrator.py or integrator_batch_process.py → .zarr.zip)
    2. Peak Fitting / Refinement (gsas_ii_refine.py  →  .gpx per histogram)

This wrapper delegates to the two existing scripts, passing through all
necessary arguments.

Backends
--------
The script supports two integration backends selected via ``--backend``:

**batch** (default)
    Uses ``integrator.py`` (CPU / OpenMP).  You provide individual data files
    with ``-dataFN`` (and optionally ``-startFileNr`` / ``-endFileNr``).

**stream**
    Uses ``integrator_batch_process.py`` (GPU / CUDA).  You provide a folder
    of images with ``--folder`` or connect to a live detector via ``--pva``.
    This backend also produces an HDF5 file alongside the zarr.zip.

Prerequisites
-------------
- MIDAS   (integrator.py, integrator_batch_process.py, and their C/CUDA binaries)
- GSAS-II (conda install gsas2full -c briantoby)
- zarr == 2.18.3

Usage
-----
CPU batch (individual files)::

    python $MIDAS_INSTALL_DIR/utils/integrate_and_refine.py \\
        --backend batch \\
        -paramFN  ps.txt \\
        -dataFN   data/sample_000001.h5 \\
        --cif     CeO2.cif \\
        --out     results/ \\
        -nCPUs    8

GPU streaming (folder of images)::

    python $MIDAS_INSTALL_DIR/utils/integrate_and_refine.py \\
        --backend stream \\
        --param-file ps.txt \\
        --folder  /data/experiment/scan_01/ \\
        --cif     CeO2.cif \\
        --out     results/ \\
        -nCPUs    8

GPU streaming (live PVA)::

    python $MIDAS_INSTALL_DIR/utils/integrate_and_refine.py \\
        --backend stream \\
        --param-file ps.txt \\
        --pva --pva-ip 10.54.105.139 \\
        --cif     CeO2.cif \\
        --out     results/ \\
        -nCPUs    8

Environment
-----------
MIDAS_INSTALL_DIR
    (Auto-detected from script location.) Override with this env var if needed.

GSASII_PATH
    Path to the GSAS-II source directory, if not installed via conda.
    See ``gsas_ii_refine.py --help`` for details.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve MIDAS installation directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = os.environ.get("MIDAS_INSTALL_DIR", str(_SCRIPT_DIR.parent))
MIDAS_UTILS = Path(MIDAS_HOME) / "utils"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("integrate_and_refine")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1a: CPU Batch Integration  (integrator.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_batch_integrator(args: argparse.Namespace) -> Path:
    """Run MIDAS integrator.py (CPU batch) and return the zarr.zip path.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (must contain integrator-relevant fields).

    Returns
    -------
    Path
        Path to the generated .zarr.zip file.
    """
    integrator_script = MIDAS_UTILS / "integrator.py"
    if not integrator_script.exists():
        log.error("integrator.py not found at %s", integrator_script)
        sys.exit(1)

    cmd = [
        sys.executable, str(integrator_script),
        "-paramFN", str(args.paramFN),
        "-dataFN", str(args.dataFN),
        "-resultFolder", str(args.resultFolder),
    ]

    if args.darkFN:
        cmd += ["-darkFN", str(args.darkFN)]
    if args.dataLoc:
        cmd += ["-dataLoc", str(args.dataLoc)]
    if args.darkLoc:
        cmd += ["-darkLoc", str(args.darkLoc)]
    if args.numFrameChunks != -1:
        cmd += ["-numFrameChunks", str(args.numFrameChunks)]
    if args.preProcThresh != -1:
        cmd += ["-preProcThresh", str(args.preProcThresh)]
    if args.startFileNr != -1:
        cmd += ["-startFileNr", str(args.startFileNr)]
    if args.endFileNr != -1:
        cmd += ["-endFileNr", str(args.endFileNr)]
    cmd += ["-nCPUs", str(args.nCPUs)]
    cmd += ["-nCPUsLocal", str(args.nCPUsLocal)]

    log.info("═══ Stage 1: MIDAS Integration (batch / CPU) ═══")
    log.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("integrator.py failed (exit code %d)", result.returncode)
        sys.exit(result.returncode)

    # Find the .zarr.zip output
    result_dir = Path(args.resultFolder)
    zarr_files = sorted(result_dir.glob("*.zarr.zip"))
    if not zarr_files:
        # Also check for deeply nested outputs
        zarr_files = sorted(result_dir.rglob("*.caked.hdf.zarr.zip"))

    if not zarr_files:
        log.error("No .zarr.zip files found in %s after integration", result_dir)
        sys.exit(1)

    log.info("Integration produced %d zarr file(s)", len(zarr_files))
    return zarr_files[-1]  # Return the latest one


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1b: GPU Streaming Integration  (integrator_batch_process.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_stream_integrator(args: argparse.Namespace) -> Path:
    """Run integrator_batch_process.py (GPU stream) and return the zarr.zip path.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (must contain stream-relevant fields).

    Returns
    -------
    Path
        Path to the generated .zarr.zip file.
    """
    stream_script = MIDAS_UTILS / "integrator_batch_process.py"
    if not stream_script.exists():
        log.error("integrator_batch_process.py not found at %s", stream_script)
        sys.exit(1)

    cmd = [
        sys.executable, str(stream_script),
        "--param-file", str(args.param_file),
    ]

    # Data source: folder or PVA (mutually exclusive)
    if args.pva:
        cmd.append("--pva")
        if args.pva_ip:
            cmd += ["--pva-ip", str(args.pva_ip)]
    elif args.folder:
        cmd += ["--folder", str(args.folder)]
    else:
        log.error("Stream backend requires --folder or --pva")
        sys.exit(1)

    if args.dark:
        cmd += ["--dark", str(args.dark)]
    if args.output_h5:
        cmd += ["--output-h5", str(args.output_h5)]
    if args.stream_data_loc:
        cmd += ["--data-loc", str(args.stream_data_loc)]
    if args.compress:
        cmd.append("--compress")

    # Zarr output (always enabled so refinement can use it)
    if args.zarr_output:
        cmd += ["--zarr-output", str(args.zarr_output)]

    log.info("═══ Stage 1: MIDAS Integration (stream / GPU) ═══")
    log.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("integrator_batch_process.py failed (exit code %d)",
                  result.returncode)
        sys.exit(result.returncode)

    # Find the .zarr.zip output
    # The stream integrator creates it alongside the HDF5
    if args.zarr_output:
        zarr_path = Path(args.zarr_output)
    elif args.output_h5:
        zarr_path = Path(args.output_h5).with_suffix("").with_suffix(".zarr.zip")
    else:
        # Default name used by integrator_batch_process.py
        zarr_path = Path("integrator_output.zarr.zip")

    if not zarr_path.exists():
        # Fallback: search for any zarr.zip in the current directory
        zarr_files = sorted(Path(".").glob("*.zarr.zip"))
        if zarr_files:
            zarr_path = zarr_files[-1]
        else:
            log.error("No .zarr.zip found after stream integration")
            sys.exit(1)

    log.info("Stream integration complete. Zarr: %s", zarr_path)
    return zarr_path


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Peak Fitting & Refinement  (gsas_ii_refine.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_refinement(zarr_file: Path, args: argparse.Namespace) -> None:
    """Run gsas_ii_refine.py on the integrated zarr.zip file.

    Parameters
    ----------
    zarr_file : Path
        Path to the .zarr.zip file from integration.
    args : argparse.Namespace
        Parsed arguments (must contain refinement-relevant fields).
    """
    refine_script = MIDAS_UTILS / "gsas_ii_refine.py"
    if not refine_script.exists():
        log.error("gsas_ii_refine.py not found at %s", refine_script)
        sys.exit(1)

    refinement_dir = Path(args.out)

    cmd = [
        sys.executable, str(refine_script),
        "--data", str(zarr_file),
        "--out", str(refinement_dir),
        "--bkg-terms", str(args.bkg_terms),
        "--nCPUs", str(args.nCPUs),
    ]
    cmd += ["--cif"] + [str(c) for c in args.cif]

    if args.instprm:
        cmd += ["--instprm", str(args.instprm)]
    if args.limits:
        cmd += ["--limits", str(args.limits[0]), str(args.limits[1])]
    if args.no_atoms:
        cmd.append("--no-atoms")
    if args.no_export:
        cmd.append("--no-export")
    if args.verbose:
        cmd.append("-v")

    log.info("═══ Stage 2: Peak Fitting & Refinement ═══")
    log.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("gsas_ii_refine.py failed (exit code %d)", result.returncode)
        sys.exit(result.returncode)

    log.info("Refinement complete. Results in: %s", refinement_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Combined MIDAS Integration + Peak Fitting / Refinement pipeline.

            Stage 1: Integrates raw data → .zarr.zip (choose backend with --backend)
            Stage 2: Runs gsas_ii_refine.py for staged peak fitting/refinement

            Backends:
              batch  — Uses integrator.py (CPU/OpenMP). Provide files with -dataFN.
              stream — Uses integrator_batch_process.py (GPU/CUDA).
                       Provide --folder or --pva for data source.

            GSAS-II must be accessible.  Install via conda:
                conda install gsas2full -c briantoby
            Or set env variable:
                export GSASII_PATH=/path/to/GSAS-II
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Backend selection ────────────────────────────────────────────────
    p.add_argument(
        "--backend", choices=["batch", "stream"], default="batch",
        help="Integration backend: 'batch' (CPU, integrator.py) or "
             "'stream' (GPU, integrator_batch_process.py). Default: batch.",
    )

    # ── Batch backend arguments (integrator.py) ──────────────────────────
    batch = p.add_argument_group(
        "Batch integration (--backend batch)",
        "Arguments for the CPU batch integrator. Ignored when --backend stream.",
    )
    batch.add_argument(
        "-paramFN",
        help="MIDAS parameter file (batch backend).",
    )
    batch.add_argument(
        "-dataFN",
        help="Path to the first data file (raw HDF5).",
    )
    batch.add_argument(
        "-resultFolder", default=".",
        help="Folder for integration results (default: cwd).",
    )
    batch.add_argument("-darkFN", default="", help="Dark file path.")
    batch.add_argument("-dataLoc", default="exchange/data",
                       help="HDF5 data location key.")
    batch.add_argument("-darkLoc", default="exchange/dark",
                       help="HDF5 dark location key.")
    batch.add_argument("-numFrameChunks", type=int, default=-1,
                       help="Frame chunks (-1 = disabled).")
    batch.add_argument("-preProcThresh", type=int, default=-1,
                       help="Pre-process threshold (-1 = disabled).")
    batch.add_argument("-startFileNr", type=int, default=-1,
                       help="Start file number (-1 = auto).")
    batch.add_argument("-endFileNr", type=int, default=-1,
                       help="End file number (-1 = single file).")
    batch.add_argument("-nCPUsLocal", type=int, default=4,
                       help="CPUs per integrator instance.")

    # ── Stream backend arguments (integrator_batch_process.py) ───────────
    stream = p.add_argument_group(
        "Stream integration (--backend stream)",
        "Arguments for the GPU streaming integrator. Ignored when --backend batch.",
    )
    stream.add_argument(
        "--param-file",
        help="MIDAS parameter file (stream backend).",
    )
    stream.add_argument(
        "--folder",
        help="Source folder of image files (e.g. .tif, .ge, .h5).",
    )
    stream.add_argument(
        "--pva", action="store_true",
        help="Enable PVA live-streaming mode.",
    )
    stream.add_argument("--pva-ip", help="PVA server IP address.")
    stream.add_argument(
        "--output-h5", default=None,
        help="HDF5 output filename (default: integrator_output.h5).",
    )
    stream.add_argument("--dark", default=None,
                        help="Dark field file for background subtraction.")
    stream.add_argument("--stream-data-loc", default=None,
                        help="HDF5 dataset location within data files.")
    stream.add_argument("--compress", action="store_true",
                        help="Enable hybrid compression.")
    stream.add_argument("--zarr-output", default=None,
                        help="Custom zarr.zip output filename.")

    # ── Refinement arguments ─────────────────────────────────────────────
    refine = p.add_argument_group("Refinement (Stage 2)")
    refine.add_argument(
        "--cif", "-c", nargs="+", required=True,
        help="CIF file(s) for the crystallographic phase(s).",
    )
    refine.add_argument(
        "--out", "-o", default="refinement/",
        help="Output directory for .gpx projects (default: refinement/).",
    )
    refine.add_argument(
        "--instprm", default=None,
        help="Optional .instprm file for GSAS-II instrument parameters.",
    )
    refine.add_argument(
        "--bkg-terms", type=int, default=6,
        help="Chebyshev background terms (default: 6).",
    )
    refine.add_argument(
        "--limits", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
        help="2θ limits in degrees.",
    )
    refine.add_argument(
        "--no-atoms", action="store_true",
        help="Skip atomic position refinement.",
    )
    refine.add_argument(
        "--no-export", action="store_true",
        help="Skip CIF/CSV export.",
    )

    # ── Shared arguments ─────────────────────────────────────────────────
    shared = p.add_argument_group("Shared")
    shared.add_argument(
        "-nCPUs", type=int, default=1,
        help="Number of CPUs for parallel integration and refinement.",
    )
    shared.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output.",
    )

    # ── Mode control ─────────────────────────────────────────────────────
    mode = p.add_argument_group("Pipeline control")
    mode.add_argument(
        "--skip-integration", action="store_true",
        help="Skip Stage 1 and use an existing .zarr.zip file "
             "(specify with --zarr-file).",
    )
    mode.add_argument(
        "--zarr-file", default=None,
        help="Path to an existing .zarr.zip file (use with --skip-integration).",
    )
    mode.add_argument(
        "--skip-refinement", action="store_true",
        help="Run only Stage 1 (integration). Skip peak fitting/refinement.",
    )

    args = p.parse_args()

    # Validate backend-specific required args
    if not args.skip_integration:
        if args.backend == "batch":
            if not args.paramFN or not args.dataFN:
                p.error("Batch backend requires -paramFN and -dataFN")
        elif args.backend == "stream":
            if not args.param_file:
                p.error("Stream backend requires --param-file")
            if not args.folder and not args.pva:
                p.error("Stream backend requires --folder or --pva")

    return args


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Stage 1: Integration ─────────────────────────────────────────────
    if args.skip_integration:
        if not args.zarr_file:
            log.error("--skip-integration requires --zarr-file")
            sys.exit(1)
        zarr_file = Path(args.zarr_file)
        if not zarr_file.exists():
            log.error("zarr file not found: %s", zarr_file)
            sys.exit(1)
        log.info("Skipping integration, using: %s", zarr_file)
    elif args.backend == "batch":
        zarr_file = run_batch_integrator(args)
    elif args.backend == "stream":
        zarr_file = run_stream_integrator(args)

    # ── Stage 2: Refinement ──────────────────────────────────────────────
    if args.skip_refinement:
        log.info("Skipping refinement stage (--skip-refinement)")
    else:
        run_refinement(zarr_file, args)

    log.info("═══ Pipeline complete ═══")


if __name__ == "__main__":
    main()
