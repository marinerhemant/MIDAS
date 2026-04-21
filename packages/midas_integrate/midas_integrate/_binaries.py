"""Locate C binaries shipped inside the wheel (CPU + optional GPU).

Discovery order (first match wins):
    1. Inside this package's _bin directory (importlib.resources).
    2. Inside midas_auto_calibrate's _bin directory, if installed — the
       two packages ship complementary binaries that can call each other
       via subprocess (e.g. MIDASIntegrator uses GetHKLList).
    3. Inside midas_integrate_gpu's _bin directory (optional GPU wheel).
    4. $MIDAS_BIN/<name>   — explicit override for in-tree dev builds.
    5. $MIDAS_INSTALL_DIR/bin/<name> — legacy MIDAS env var.
    6. shutil.which(name) — on PATH.
    7. Raise MidasBinaryNotFoundError listing every location tried.
"""

from __future__ import annotations

import os
import shutil
from importlib import resources
from pathlib import Path


class MidasBinaryNotFoundError(RuntimeError):
    """Raised when a MIDAS C binary can't be located."""


def _package_bin_dir(package_name: str) -> Path | None:
    """Return the ``_bin`` directory shipped inside ``package_name``, or None.

    Returns None on package absence, missing directory, or importlib quirks
    (resources.files can raise on namespace packages).
    """
    try:
        bin_dir = resources.files(package_name).joinpath("_bin")
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    try:
        return Path(str(bin_dir))
    except TypeError:
        return None


def midas_bin(name: str, *, bin_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a MIDAS binary to an absolute Path."""
    tried: list[str] = []

    if bin_dir is not None:
        p = Path(bin_dir) / name
        tried.append(str(p))
        if p.exists():
            return p.resolve()

    # Check our own wheel, then the sister packages' wheels.
    for package in ("midas_integrate", "midas_auto_calibrate", "midas_integrate_gpu"):
        pkg_bin = _package_bin_dir(package)
        if pkg_bin is None:
            continue
        p = pkg_bin / name
        tried.append(str(p))
        if p.exists():
            return p.resolve()

    env_bin = os.environ.get("MIDAS_BIN")
    if env_bin:
        p = Path(env_bin) / name
        tried.append(str(p))
        if p.exists():
            return p.resolve()

    env_install = os.environ.get("MIDAS_INSTALL_DIR")
    if env_install:
        p = Path(env_install) / "bin" / name
        tried.append(str(p))
        if p.exists():
            return p.resolve()

    on_path = shutil.which(name)
    tried.append(f"PATH({name})")
    if on_path:
        return Path(on_path).resolve()

    raise MidasBinaryNotFoundError(
        f"Could not locate MIDAS binary {name!r}. Tried: " + ", ".join(tried)
    )
