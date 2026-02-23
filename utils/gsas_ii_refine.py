#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gsas_ii_refine.py — Automated Peak Fitting and Refinement using GSASIIscriptable

This script processes a GSAS-II `.zarr.zip` archive containing caked histograms
and performs a staged peak fitting/refinement using the GSAS-II scripting API.  The stages
mirror best-practice for synchrotron powder-diffraction data:

    1. Background + Scale
    2. Unit Cell
    3. Peak Profile (U, V, W, then X, Y)
    4. Atomic positions (X) + thermal parameters (U)

The resulting .gpx project is fully compatible with the GSAS-II GUI, so
users can inspect, adjust, and continue refining interactively.

Each histogram (lineout) from the zarr file is refined **independently** in
a separate GSAS-II project.  When ``--nCPUs`` > 1 these refinements run in
parallel via multiprocessing.

Prerequisites
-------------
- GSAS-II  (``conda install gsas2full -c briantoby``)
- zarr == 2.18.3  (``pip install zarr==2.18.3``)

GSAS-II Installation
--------------------
The script auto-detects GSAS-II when installed via conda (gsas2full).
If GSAS-II is installed elsewhere, set the environment variable::

    export GSASII_PATH=/path/to/GSAS-II/GSASII

Example
-------
::

    python $MIDAS_INSTALL_DIR/utils/gsas_ii_refine.py \\
        --data  output/sample.zarr.zip \\
        --cif   CeO2.cif \\
        --out   refinement/ \\
        --bkg-terms 6 \\
        --nCPUs 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# GSAS-II import — works whether gsas2full is on sys.path or needs explicit
# path injection.
# ---------------------------------------------------------------------------
def _import_gsasii():
    """Import GSASIIscriptable, handling various installation layouts.

    Search order:
      1. ``from GSASII import GSASIIscriptable``  (pip / conda install)
      2. ``import GSASIIscriptable``                (GSASII on sys.path)
      3. ``GSASII_PATH`` environment variable        (manual install)
    """
    try:
        from GSASII import GSASIIscriptable as G2sc
        return G2sc
    except ImportError:
        pass

    try:
        import GSASIIscriptable as G2sc
        return G2sc
    except ImportError:
        pass

    gsas_path = os.environ.get("GSASII_PATH")
    if gsas_path:
        sys.path.insert(0, gsas_path)
        try:
            from GSASII import GSASIIscriptable as G2sc
            return G2sc
        except ImportError:
            pass
        # Also try with GSASII subdirectory on path
        gsasii_subdir = os.path.join(gsas_path, "GSASII")
        if os.path.isdir(gsasii_subdir):
            sys.path.insert(0, gsasii_subdir)
        try:
            import GSASIIscriptable as G2sc
            return G2sc
        except ImportError:
            pass

    print(
        "ERROR: Cannot import GSASIIscriptable.\n"
        "Install GSAS-II via conda:\n"
        "    conda install gsas2full -c briantoby\n"
        "Or set GSASII_PATH to point to your GSAS-II source directory:\n"
        "    export GSASII_PATH=/path/to/GSAS-II/GSASII",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gsas_ii_refine")


# ---------------------------------------------------------------------------
# Refinement recipe builder
# ---------------------------------------------------------------------------
def build_refinement_recipe(
    bkg_terms: int = 6,
    refine_atoms: bool = True,
    two_theta_limits: Optional[list] = None,
) -> list[dict]:
    """Return a multi-stage refinement recipe.

    Each dict is passed to ``gpx.do_refinements([pardict])``.
    Stages are cumulative — earlier flags stay on.

    Parameters
    ----------
    bkg_terms : int
        Number of Chebyshev background coefficients.
    refine_atoms : bool
        Whether to refine atomic positions and thermal parameters.
    two_theta_limits : list, optional
        [low, high] 2θ limits in degrees.
    """

    stage1 = {
        "set": {
            "Background": {"no. coeffs": bkg_terms, "refine": True},
            "Sample Parameters": ["Scale"],
        }
    }
    if two_theta_limits is not None:
        stage1["set"]["Limits"] = two_theta_limits

    stage2 = {"set": {"Cell": True}}
    stage3a = {"set": {"Instrument Parameters": ["U", "V", "W"]}}
    stage3b = {"set": {"Instrument Parameters": ["X", "Y", "SH/L"]}}

    stages = [stage1, stage2, stage3a, stage3b]

    if refine_atoms:
        stages.append({"set": {"Atoms": {"all": "XU"}}})

    return stages


# ---------------------------------------------------------------------------
# Single-histogram refinement (runs in a worker process)
# ---------------------------------------------------------------------------
def _refine_single_histogram(
    hist_index: int,
    data_file: str,
    cif_files: list[str],
    out_dir: str,
    instprm_file: Optional[str],
    bkg_terms: int,
    refine_atoms: bool,
    two_theta_limits: Optional[list],
    export_cif: bool,
    export_csv: bool,
) -> dict:
    """Refine a single histogram (lineout) from the zarr file.

    This function is designed to run in a separate process so that multiple
    histograms can be refined in parallel.  It creates its own GSAS-II
    project, imports only the requested histogram, and runs the full staged
    refinement.

    Parameters
    ----------
    hist_index : int
        Index of the histogram (lineout) to refine.
    data_file, cif_files, out_dir, instprm_file, bkg_terms, refine_atoms,
    two_theta_limits, export_cif, export_csv
        Same meaning as in ``run_refinement()``.

    Returns
    -------
    dict
        Per-histogram result summary.
    """
    G2sc = _import_gsasii()

    gpx_name = f"hist_{hist_index:04d}.gpx"
    gpx_path = str(Path(out_dir) / gpx_name)

    result = {"histogram_index": hist_index, "gpx": gpx_path}

    try:
        # Create a project for this single histogram
        gpx = G2sc.G2Project(newgpx=gpx_path)

        # Import powder histogram, selecting only this one lineout
        gpx.add_powder_histogram(
            data_file,
            instprm_file if instprm_file else "",
            fmthint="MIDAS zarr",
            databank=hist_index + 1,  # 1-based bank number for selection
        )

        if len(gpx.histograms()) == 0:
            result["status"] = "skipped"
            result["message"] = f"Histogram {hist_index} could not be loaded"
            return result

        # Add phase(s)
        for cif in cif_files:
            gpx.add_phase(
                cif,
                phasename=Path(cif).stem,
                histograms=gpx.histograms(),
                fmthint="CIF",
            )

        # Run staged refinement
        recipe = build_refinement_recipe(
            bkg_terms=bkg_terms,
            refine_atoms=refine_atoms,
            two_theta_limits=two_theta_limits,
        )

        for i, stage in enumerate(recipe, 1):
            try:
                gpx.do_refinements([stage])
            except Exception as e:
                result.setdefault("warnings", []).append(
                    f"Stage {i} issue: {str(e)}"
                )

        # Collect results
        try:
            h0 = gpx.histogram(0)
            result["Rwp"] = h0.get_wR()
        except Exception:
            pass

        for phase in gpx.phases():
            try:
                cell = phase.get_cell()
                result.setdefault("phases", []).append({
                    "name": phase.name,
                    "cell": {
                        "a": cell[0], "b": cell[1], "c": cell[2],
                        "alpha": cell[3], "beta": cell[4], "gamma": cell[5],
                        "volume": cell[6],
                    },
                })
            except Exception:
                result.setdefault("phases", []).append({"name": phase.name})

        # Save & export
        gpx.save()

        out_stem = str(Path(gpx_path).with_suffix(""))
        if export_cif:
            try:
                for phase in gpx.phases():
                    phase.export_CIF(out_stem + f"_{phase.name}.cif")
            except Exception:
                pass
        if export_csv:
            try:
                for j, h in enumerate(gpx.histograms()):
                    h.Export(out_stem + f"_data", ".csv", "hist")
            except Exception:
                pass

        result["status"] = "success"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Main entry point: refine all histograms (optionally in parallel)
# ---------------------------------------------------------------------------
def run_refinement(
    data_file: str,
    cif_files: list[str],
    output_dir: str,
    instprm_file: Optional[str] = None,
    bkg_terms: int = 6,
    refine_atoms: bool = True,
    two_theta_limits: Optional[list] = None,
    export_cif: bool = True,
    export_csv: bool = True,
    n_cpus: int = 1,
) -> dict:
    """Refine every histogram from a MIDAS .zarr.zip file.

    When ``n_cpus > 1`` histograms are refined **in parallel**, each in its
    own GSAS-II project and process.  Results are collected into a single
    summary JSON.

    Parameters
    ----------
    data_file : str
        Path to the MIDAS .zarr.zip caked output file.
    cif_files : list[str]
        CIF file(s) defining the phase(s).
    output_dir : str
        Directory for all output .gpx projects and exports.
    instprm_file : str, optional
        .instprm override.
    bkg_terms : int
        Chebyshev background terms (default 6).
    refine_atoms : bool
        Refine atomic positions + thermal parameters.
    two_theta_limits : list, optional
        [low, high] 2θ limits.
    export_cif, export_csv : bool
        Whether to export refined structures / data.
    n_cpus : int
        Number of parallel workers.

    Returns
    -------
    dict
        Summary across all histograms.
    """
    data_path = Path(data_file).resolve()
    if not data_path.exists():
        log.error("Data file not found: %s", data_path)
        sys.exit(1)

    for cif in cif_files:
        if not Path(cif).exists():
            log.error("CIF file not found: %s", cif)
            sys.exit(1)

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover how many histograms (lineouts) the file contains ──────────
    # We do a quick probe by loading the zarr file directly.
    n_hist = _count_histograms(str(data_path))
    if n_hist == 0:
        log.error("No valid histograms found in %s", data_path)
        sys.exit(1)

    log.info(
        "Found %d histogram(s) in %s — refining with %d worker(s)",
        n_hist, data_path.name, min(n_cpus, n_hist),
    )

    # ── Dispatch refinements ──────────────────────────────────────────────
    common_kwargs = dict(
        data_file=str(data_path),
        cif_files=[str(Path(c).resolve()) for c in cif_files],
        out_dir=str(out_dir),
        instprm_file=instprm_file,
        bkg_terms=bkg_terms,
        refine_atoms=refine_atoms,
        two_theta_limits=two_theta_limits,
        export_cif=export_cif,
        export_csv=export_csv,
    )

    results_list = []

    if n_cpus <= 1 or n_hist == 1:
        # Serial path
        for idx in range(n_hist):
            log.info("Refining histogram %d / %d ...", idx + 1, n_hist)
            res = _refine_single_histogram(hist_index=idx, **common_kwargs)
            results_list.append(res)
            _log_histogram_result(res)
    else:
        # Parallel path
        workers = min(n_cpus, n_hist)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_refine_single_histogram, hist_index=idx, **common_kwargs): idx
                for idx in range(n_hist)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    res = {
                        "histogram_index": idx,
                        "status": "failed",
                        "error": str(e),
                    }
                results_list.append(res)
                _log_histogram_result(res)

    # Sort by index for deterministic output
    results_list.sort(key=lambda r: r.get("histogram_index", 0))

    # ── Aggregate summary ─────────────────────────────────────────────────
    summary = {
        "data_file": str(data_path),
        "total_histograms": n_hist,
        "succeeded": sum(1 for r in results_list if r.get("status") == "success"),
        "failed": sum(1 for r in results_list if r.get("status") == "failed"),
        "skipped": sum(1 for r in results_list if r.get("status") == "skipped"),
        "histograms": results_list,
    }

    # Compute average Rwp across successful refinements
    rwps = [r["Rwp"] for r in results_list if "Rwp" in r]
    if rwps:
        summary["mean_Rwp"] = sum(rwps) / len(rwps)

    summary_path = out_dir / "refinement_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Summary written: %s", summary_path)

    return summary


def _count_histograms(data_file: str) -> int:
    """Count the number of valid lineouts in a MIDAS zarr.zip file.

    Mirrors the logic in G2pwd_MIDAS.py: a lineout needs ≥20 unmasked points.
    """
    try:
        import zarr
        import numpy as np
    except ImportError:
        log.error("zarr and numpy are required (pip install zarr==2.18.3 numpy)")
        sys.exit(1)

    try:
        fp = zarr.open(data_file, mode="r")
    except Exception:
        # zarr 3.x workaround
        import asyncio
        async def _open():
            store = await zarr.storage.ZipStore.open(data_file, mode="r")
            return zarr.open_group(store, mode="r")
        fp = asyncio.run(_open())

    remap = np.array(fp["REtaMap"])
    Nbins, Nazim = remap[1].shape
    n_images = len(fp["OmegaSumFrame"])

    # Unmasked = area > 0
    unmasked = [(remap[3][:, i] != 0) for i in range(Nazim)]
    valid_azm = [i for i in range(Nazim) if sum(unmasked[i]) > 20]

    return n_images * len(valid_azm)


def _log_histogram_result(res: dict) -> None:
    """Log a one-line summary for a histogram result."""
    idx = res.get("histogram_index", "?")
    status = res.get("status", "unknown")
    rwp = res.get("Rwp")
    if status == "success" and rwp is not None:
        log.info("  Histogram %s  →  Rwp=%.3f%%", idx, rwp)
    elif status == "failed":
        log.warning("  Histogram %s  →  FAILED: %s", idx, res.get("error", ""))
    else:
        log.info("  Histogram %s  →  %s", idx, status)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Automated Peak Fitting / Refinement with GSAS-II on MIDAS caked data.

            This script takes a zarr.zip file generated by the MIDAS integrator and
            performs a staged peak fitting refinement on every histogram (lineout).
            It uses multiprocessing to refine multiple histograms in parallel.

            The resulting .gpx projects can be opened in the GSAS-II GUI for
            manual inspection.

            GSAS-II setup
            -------------
            Install via conda:    conda install gsas2full -c briantoby
            Or set env variable:  export GSASII_PATH=/path/to/GSAS-II/GSASII
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--data", "-d", required=True,
        help="Path to the MIDAS .zarr.zip caked output file.",
    )
    p.add_argument(
        "--cif", "-c", nargs="+", required=True,
        help="One or more CIF files defining the crystallographic phase(s).",
    )
    p.add_argument(
        "--out", "-o", default="refinement/",
        help="Output directory for .gpx projects and exports (default: refinement/).",
    )
    p.add_argument(
        "--instprm", "-i", default=None,
        help="Optional .instprm file for instrument parameters.",
    )
    p.add_argument(
        "--bkg-terms", type=int, default=6,
        help="Number of Chebyshev background coefficients (default: 6).",
    )
    p.add_argument(
        "--limits", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
        help="2θ limits in degrees (default: full range from data).",
    )
    p.add_argument(
        "--no-atoms", action="store_true",
        help="Skip atomic position / thermal parameter refinement.",
    )
    p.add_argument(
        "--no-export", action="store_true",
        help="Skip CIF and CSV exports after refinement.",
    )
    p.add_argument(
        "--nCPUs", type=int, default=1,
        help="Number of parallel workers for histogram refinement (default: 1).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = run_refinement(
        data_file=args.data,
        cif_files=args.cif,
        output_dir=args.out,
        instprm_file=args.instprm,
        bkg_terms=args.bkg_terms,
        refine_atoms=not args.no_atoms,
        two_theta_limits=args.limits,
        export_cif=not args.no_export,
        export_csv=not args.no_export,
        n_cpus=args.nCPUs,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  Peak Fitting / Refinement Complete")
    print("=" * 60)
    print(f"  Histograms : {results['total_histograms']}")
    print(f"  Succeeded  : {results['succeeded']}")
    print(f"  Failed     : {results['failed']}")
    print(f"  Skipped    : {results['skipped']}")
    if "mean_Rwp" in results:
        print(f"  Mean Rwp   : {results['mean_Rwp']:.3f}%")
    print(f"  Output dir : {args.out}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
