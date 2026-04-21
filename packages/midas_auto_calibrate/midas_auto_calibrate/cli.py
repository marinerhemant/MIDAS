"""Command-line entry points.

These are stubs declared in pyproject.toml's [project.scripts] so the package
installs cleanly. Each implementation lands in its own week per the release
plan (calibrate.py, validate.py, benchmark.py).
"""

from __future__ import annotations

import sys


def _not_implemented(name: str) -> int:
    sys.stderr.write(
        f"{name}: implementation pending. See "
        "packages/midas_auto_calibrate/ release plan for timeline.\n"
    )
    return 2


def main_calibrate() -> int:
    return _not_implemented("midas-auto-calibrate")


def main_validate() -> int:
    return _not_implemented("midas-calib-validate")


def main_benchmark() -> int:
    return _not_implemented("midas-calib-benchmark")
