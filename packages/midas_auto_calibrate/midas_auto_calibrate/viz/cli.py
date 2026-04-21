"""CLI entry points for the viz submodule.

Declared in pyproject.toml's [project.scripts]:
    midas-calib-inspect  -> main_inspect (static bundle)
    midas-calib-viewer   -> main_viewer  (PyQt5 interactive, [viz-gui] extra)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main_inspect(argv: list[str] | None = None) -> int:
    """Render the five-plot inspection bundle from a calibration run.

    Takes the work-dir of a previous ``midas-auto-calibrate`` invocation
    (the one containing ``calibrant.stdout`` and the ``*.corr.csv``) and
    writes PNGs next to it.
    """
    parser = argparse.ArgumentParser(
        prog="midas-calib-inspect",
        description="Render static PNG plots for a calibration result.",
    )
    parser.add_argument(
        "work_dir",
        type=Path,
        help="Directory holding calibrant.stdout and <rawFN>.corr.csv "
             "from a prior midas-auto-calibrate run.",
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Path to the calibrant image (for rings_overlay plot). "
             "Optional — inspect skips rings_overlay if absent.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory for PNGs (default: <work_dir>).",
    )
    parser.add_argument(
        "--prefix", default="calib",
        help="Filename prefix for written PNGs (default: 'calib').",
    )
    args = parser.parse_args(argv)

    try:
        from .static import inspect
    except ImportError as exc:
        sys.stderr.write(
            f"midas-calib-inspect requires the [viz] extra "
            f"(pip install 'midas-auto-calibrate[viz]'): {exc}\n"
        )
        return 2

    result = _load_result(args.work_dir)
    out_dir = args.out or args.work_dir
    written = inspect(result, image=args.image, out_dir=out_dir, prefix=args.prefix)

    for name, path in written.items():
        print(f"  {name:>11} -> {path}")
    if not written:
        sys.stderr.write("No plots written — corr.csv / convergence missing.\n")
        return 1
    return 0


def main_viewer(argv: list[str] | None = None) -> int:
    """Launch the interactive PyQt5 CalibrantViewer.

    Requires the ``[viz-gui]`` extra. Falls back with a helpful error
    message when PyQt5 isn't available.
    """
    parser = argparse.ArgumentParser(prog="midas-calib-viewer")
    parser.add_argument("work_dir", type=Path,
                        help="Directory holding the calibration result.")
    args = parser.parse_args(argv)

    try:
        from .viewer import launch
    except ImportError as exc:
        sys.stderr.write(
            f"midas-calib-viewer requires the [viz-gui] extra "
            f"(pip install 'midas-auto-calibrate[viz-gui]'): {exc}\n"
        )
        return 2

    result = _load_result(args.work_dir)
    return launch(result)


def _load_result(work_dir: Path):
    """Rebuild a minimal CalibrationResult from an existing work directory.

    We can't re-run the binary just to viz — we just need enough state
    for the plots to work. Most plots read the CSV outputs directly and
    tolerate minimal geometry.
    """
    from ..calibrate import CalibrationResult, _parse_final_geometry
    from .._config import CalibrationConfig

    work = Path(work_dir)
    stdout_path = work / "calibrant.stdout"
    if not stdout_path.exists():
        sys.stderr.write(
            f"No calibrant.stdout in {work}. Was this the output of a "
            f"midas-auto-calibrate run?\n"
        )
        sys.exit(1)

    stdout = stdout_path.read_text()
    # Config defaults are OK — we only need them for fallback geometry.
    geom = _parse_final_geometry(stdout, CalibrationConfig())

    corr_candidates = list(work.glob("*.corr.csv"))
    corr_csv = corr_candidates[0] if corr_candidates else None

    conv_candidates = list(work.glob("*.convergence_history.csv"))
    conv_hist = None
    if conv_candidates:
        from ..calibrate import _load_convergence_history
        conv_hist = _load_convergence_history(conv_candidates[0])

    return CalibrationResult(
        geometry=geom,
        pseudo_strain=geom.mean_strain,
        pseudo_strain_std=geom.std_strain,
        convergence_history=conv_hist,
        corr_csv_path=corr_csv,
        stdout=stdout,
        work_dir=work,
    )
