#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
integrate_and_refine.py — Combined MIDAS Integration + Rietveld Refinement

Runs the full pipeline from raw diffraction data to refined crystal structures:

    1. MIDAS integration  (integrator.py  →  .zarr.zip)
    2. Rietveld refinement (rietveld_refine.py  →  .gpx per histogram)

This wrapper delegates to the two existing scripts, passing through all
necessary arguments.

Prerequisites
-------------
- MIDAS   (integrator.py and its C binaries)
- GSAS-II (conda install gsas2full -c briantoby)
- zarr == 2.18.3

Usage
-----
::

    python $MIDAS_INSTALL_DIR/utils/integrate_and_refine.py \\
        --paramFN  ps.txt \\
        --dataFN   data/sample_000001.h5 \\
        --cif      CeO2.cif \\
        --out      results/ \\
        --nCPUs    8

The integration stage uses ``integrator.py`` to convert the raw data into
a .zarr.zip file.  The refinement stage then reads that .zarr.zip and
performs a staged Rietveld refinement for each histogram.

Environment
-----------
MIDAS_INSTALL_DIR
    (Auto-detected from script location.) Override with this env var if needed.

GSASII_PATH
    Path to the GSAS-II source directory, if not installed via conda.
    See ``rietveld_refine.py --help`` for details.
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


def run_integrator(args: argparse.Namespace) -> Path:
    """Run MIDAS integrator.py and return the path to the output zarr.zip.

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

    log.info("═══ Stage 1: MIDAS Integration ═══")
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


def run_refinement(zarr_file: Path, args: argparse.Namespace) -> None:
    """Run rietveld_refine.py on the integrated zarr.zip file.

    Parameters
    ----------
    zarr_file : Path
        Path to the .zarr.zip file from integration.
    args : argparse.Namespace
        Parsed arguments (must contain refinement-relevant fields).
    """
    refine_script = MIDAS_UTILS / "rietveld_refine.py"
    if not refine_script.exists():
        log.error("rietveld_refine.py not found at %s", refine_script)
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

    log.info("═══ Stage 2: Rietveld Refinement ═══")
    log.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("rietveld_refine.py failed (exit code %d)", result.returncode)
        sys.exit(result.returncode)

    log.info("Refinement complete. Results in: %s", refinement_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Combined MIDAS Integration + Rietveld Refinement pipeline.

            Stage 1: Runs integrator.py to convert raw data → .zarr.zip
            Stage 2: Runs rietveld_refine.py for staged Rietveld refinement

            GSAS-II must be accessible.  Install via conda:
                conda install gsas2full -c briantoby
            Or set env variable:
                export GSASII_PATH=/path/to/GSAS-II/GSASII
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Integrator arguments ──────────────────────────────────────────────
    integ = p.add_argument_group("Integration (Stage 1)")
    integ.add_argument(
        "-paramFN", required=True,
        help="MIDAS parameter file.",
    )
    integ.add_argument(
        "-dataFN", required=True,
        help="Path to the first data file (raw HDF5).",
    )
    integ.add_argument(
        "-resultFolder", default=".",
        help="Folder for integration results (default: cwd).",
    )
    integ.add_argument("-darkFN", default="", help="Dark file path.")
    integ.add_argument("-dataLoc", default="exchange/data", help="HDF5 data location key.")
    integ.add_argument("-darkLoc", default="exchange/dark", help="HDF5 dark location key.")
    integ.add_argument("-numFrameChunks", type=int, default=-1, help="Frame chunks (-1 = disabled).")
    integ.add_argument("-preProcThresh", type=int, default=-1, help="Pre-process threshold (-1 = disabled).")
    integ.add_argument("-startFileNr", type=int, default=-1, help="Start file number (-1 = auto).")
    integ.add_argument("-endFileNr", type=int, default=-1, help="End file number (-1 = single file).")
    integ.add_argument("-nCPUsLocal", type=int, default=4, help="CPUs per integrator instance.")

    # ── Refinement arguments ──────────────────────────────────────────────
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

    # ── Shared arguments ──────────────────────────────────────────────────
    shared = p.add_argument_group("Shared")
    shared.add_argument(
        "-nCPUs", type=int, default=1,
        help="Number of CPUs for parallel integration and refinement.",
    )
    shared.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output.",
    )

    # ── Mode control ──────────────────────────────────────────────────────
    mode = p.add_argument_group("Pipeline control")
    mode.add_argument(
        "--skip-integration", action="store_true",
        help="Skip Stage 1 and use an existing .zarr.zip file (specify with --zarr-file).",
    )
    mode.add_argument(
        "--zarr-file", default=None,
        help="Path to an existing .zarr.zip file (use with --skip-integration).",
    )
    mode.add_argument(
        "--skip-refinement", action="store_true",
        help="Run only Stage 1 (integration). Skip Rietveld refinement.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Stage 1: Integration ──────────────────────────────────────────────
    if args.skip_integration:
        if not args.zarr_file:
            log.error("--skip-integration requires --zarr-file")
            sys.exit(1)
        zarr_file = Path(args.zarr_file)
        if not zarr_file.exists():
            log.error("zarr file not found: %s", zarr_file)
            sys.exit(1)
        log.info("Skipping integration, using: %s", zarr_file)
    else:
        zarr_file = run_integrator(args)

    # ── Stage 2: Refinement ──────────────────────────────────────────────
    if args.skip_refinement:
        log.info("Skipping refinement stage (--skip-refinement)")
    else:
        run_refinement(zarr_file, args)

    log.info("═══ Pipeline complete ═══")


if __name__ == "__main__":
    main()
