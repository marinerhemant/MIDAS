"""Preflight-validation hook for MIDAS workflow scripts.

Designed to be dropped into ff_MIDAS.py / nf_MIDAS.py / pf_MIDAS.py with a
single import + one call, so expensive pipelines fail fast on obvious
parameter-file mistakes.

Usage inside a workflow script::

    try:
        from midas_params.hook import preflight_validate
    except ImportError:
        preflight_validate = None

    if preflight_validate is not None:
        ok = preflight_validate(
            param_file=args.paramFN,
            pipeline="ff",                  # "ff" | "nf" | "pf" | "ri"
            skip=args.skipValidation,
            strict=False,                   # True → exit(1) on errors
            logger=logger,
        )
        if not ok:
            sys.exit(1)

Behavior:
  - `skip=True` short-circuits (useful for CI and legacy configs).
  - `strict=False` (default): prints warnings + errors but returns True,
    so the pipeline continues. This is the safer default for existing users.
  - `strict=True`: returns False on any error, caller decides whether to exit.
"""

from __future__ import annotations

import sys
from typing import Any


def preflight_validate(
    param_file: str,
    pipeline: str,
    skip: bool = False,
    strict: bool = False,
    logger: Any = None,
    use_color: bool | None = None,
) -> bool:
    """Validate a parameter file before running a MIDAS workflow.

    Returns True if it's safe to proceed, False if `strict=True` and errors
    were found.
    """
    if skip:
        _log(logger, "info", "Parameter validation skipped (--skipValidation).")
        return True

    from .schema import Path as _Path
    from .validator import format_report, validate

    try:
        path = _Path(pipeline.lower())
    except ValueError:
        _log(logger, "warning",
             f"Unknown pipeline {pipeline!r}; skipping validation.")
        return True

    try:
        report = validate(param_file, path)
    except FileNotFoundError:
        _log(logger, "error", f"Parameter file not found: {param_file}")
        return not strict
    except Exception as e:
        # Any internal validator crash should NEVER prevent the pipeline
        # from running — just log and continue.
        _log(logger, "warning", f"Parameter validation crashed: {e}")
        return True

    if report.ok and not report.warnings:
        _log(logger, "info", f"Parameter file passed validation ({param_file}).")
        return True

    # Print the pretty report to stderr (bypasses logger formatting so error
    # messages don't get buried in logger prefixes)
    if use_color is None:
        use_color = sys.stderr.isatty()
    print(format_report(report, use_color=use_color), file=sys.stderr)

    summary = (f"Parameter validation: "
               f"{len(report.errors)} error(s), {len(report.warnings)} warning(s).")
    if report.errors and strict:
        _log(logger, "error", f"{summary} Refusing to start (strict mode).")
        return False
    elif report.errors:
        _log(logger, "warning",
             f"{summary} Continuing anyway — pass --strictValidation to fail.")
        return True
    else:
        _log(logger, "warning", f"{summary} Continuing.")
        return True


def _log(logger: Any, level: str, msg: str) -> None:
    """Log to the caller's logger if present, else print to stderr."""
    if logger is not None and hasattr(logger, level):
        getattr(logger, level)(msg)
    else:
        prefix = {"info": "[INFO]", "warning": "[WARN]", "error": "[ERROR]"}.get(
            level, "[LOG]"
        )
        print(f"{prefix} {msg}", file=sys.stderr)


# ─── Runtime-default resolution (FF / PF) ────────────────────────────────────


def resolve_runtime_defaults(
    param_file: str,
    num_frame_chunks: int,
    pre_proc_thresh: int,
    n_cpus: int,
    logger: Any = None,
) -> tuple[int, int]:
    """Fill in smarter defaults for `-numFrameChunks` and `-preProcThresh`
    when the caller left them at the sentinel `-1`.

    Rules:
      - `numFrameChunks`: if -1, set to `n_cpus * 4` (clamped to ≥ 1).
      - `preProcThresh`:  if -1, set to min of the intensity column in
                          all `RingThresh` entries in the param file.
                          If no `RingThresh` entries exist, leave at -1
                          (pre-processing threshold not applied) and warn.

    Both workflows (ff_MIDAS.py, pf_MIDAS.py) call this after argparse
    and after preflight validation.

    Returns the (possibly adjusted) pair. Values the caller explicitly
    passed (non-`-1`) are returned unchanged.
    """
    if num_frame_chunks == -1:
        num_frame_chunks = max(1, int(n_cpus) * 4)
        _log(logger, "info",
             f"numFrameChunks auto-set to {num_frame_chunks} (nCPUs × 4)")

    if pre_proc_thresh == -1:
        min_thresh = _min_ringthresh(param_file)
        if min_thresh is not None:
            pre_proc_thresh = int(min_thresh)
            _log(logger, "info",
                 f"preProcThresh auto-set to {pre_proc_thresh} "
                 f"(min RingThresh intensity)")
        else:
            _log(logger, "warning",
                 "preProcThresh left at -1: no RingThresh entries found in "
                 "param file (pre-processing threshold not applied)")
    return num_frame_chunks, pre_proc_thresh


def _min_ringthresh(param_file: str) -> float | None:
    """Parse the param file and return the minimum of the second token on
    each `RingThresh` line. Returns None if no `RingThresh` entries exist
    or parsing fails."""
    try:
        from .parser import parse_raw
    except ImportError:
        return None
    try:
        raw, _ = parse_raw(param_file)
    except Exception:
        return None
    rings = raw.get("RingThresh", [])
    thresholds: list[float] = []
    for tokens in rings:
        if len(tokens) >= 2:
            try:
                thresholds.append(float(tokens[1]))
            except ValueError:
                # Malformed entry — skip; the validator will flag it
                pass
    if not thresholds:
        return None
    return min(thresholds)
