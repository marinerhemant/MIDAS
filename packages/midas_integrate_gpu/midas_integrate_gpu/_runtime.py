"""Runtime diagnostics for the GPU backend.

Users should call :func:`check_environment` before invoking GPU
integration — it surfaces actionable errors (wrong platform, missing
driver, binary not shipped) instead of letting subprocess failures
bubble up from the C binary.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EnvironmentCheck:
    """Result of :func:`check_environment`. Falsy when anything is wrong."""

    platform_ok: bool
    binary_available: bool
    nvidia_smi_found: bool
    driver_version: Optional[str] = None
    problems: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return not self.problems


def _package_bin_dir() -> Optional[Path]:
    from importlib import resources
    try:
        d = resources.files("midas_integrate_gpu").joinpath("_bin")
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    try:
        return Path(str(d))
    except TypeError:
        return None


def _nvidia_driver_version() -> Optional[str]:
    """Parse nvidia-smi's CUDA driver version. None if nvidia-smi absent."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    # "535.104.05\n" or multi-GPU list
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def check_environment() -> EnvironmentCheck:
    """Verify the GPU wheel can actually run.

    Returns an :class:`EnvironmentCheck` that's truthy when everything
    looks good. Inspect ``.problems`` for actionable error strings when
    the check fails.
    """
    problems: list[str] = []

    # 1. Platform — Linux x86_64 only for v0.1.0.
    system = platform.system()
    machine = platform.machine()
    platform_ok = system == "Linux" and machine in ("x86_64", "amd64")
    if not platform_ok:
        problems.append(
            f"midas-integrate-gpu requires Linux x86_64; got {system}/{machine}. "
            "macOS (no CUDA) and Windows (v0.2 roadmap) are unsupported."
        )

    # 2. Shipped binary — skipped silently when the wheel was installed
    #    without the CUDA bits (e.g. source install on a non-CUDA host).
    bin_dir = _package_bin_dir()
    binary_available = bool(bin_dir and (bin_dir / "MIDASIntegratorGPU").exists())
    if not binary_available:
        problems.append(
            "MIDASIntegratorGPU binary not found inside the wheel. "
            "Install via `pip install midas-integrate-gpu` on a Linux "
            "x86_64 host, or build from source inside the CUDA toolkit "
            "container."
        )

    # 3. NVIDIA driver — fail early if the user has no driver at all.
    driver_version = _nvidia_driver_version()
    nvidia_smi_found = driver_version is not None
    if not nvidia_smi_found:
        problems.append(
            "nvidia-smi not found on PATH. Install the NVIDIA driver "
            "(≥ 525 recommended for CUDA 12 runtime compatibility)."
        )

    return EnvironmentCheck(
        platform_ok=platform_ok,
        binary_available=binary_available,
        nvidia_smi_found=nvidia_smi_found,
        driver_version=driver_version,
        problems=tuple(problems),
    )


def is_gpu_available() -> bool:
    """True iff :func:`check_environment` passes (simple boolean form)."""
    return bool(check_environment())
