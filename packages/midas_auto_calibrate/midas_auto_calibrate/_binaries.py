"""Locate C binaries shipped inside the wheel.

Discovery order (first match wins):
    1. Inside this package's _bin directory (importlib.resources).
    2. $MIDAS_BIN/<name>   — explicit override for dev with in-tree builds.
    3. $MIDAS_INSTALL_DIR/bin/<name> — legacy MIDAS env var.
    4. shutil.which(name) — on PATH.
    5. Raise MidasBinaryNotFoundError listing every location tried.
"""

from __future__ import annotations

import os
import shutil
from importlib import resources
from pathlib import Path


class MidasBinaryNotFoundError(RuntimeError):
    """Raised when a MIDAS C binary can't be located."""


def _candidate_from_package() -> Path | None:
    try:
        bin_dir = resources.files("midas_auto_calibrate").joinpath("_bin")
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    try:
        # bin_dir may be a MultiplexedPath for namespace packages; coerce.
        return Path(str(bin_dir))
    except TypeError:
        return None


def midas_bin(name: str, *, bin_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a MIDAS binary to an absolute Path.

    Parameters
    ----------
    name : str
        Executable name (e.g. ``"MIDASCalibrant"``).
    bin_dir : path-like, optional
        Explicit directory to search first; bypasses the env-var chain.
    """
    tried: list[str] = []

    if bin_dir is not None:
        p = Path(bin_dir) / name
        tried.append(str(p))
        if p.exists():
            return p.resolve()

    pkg_bin = _candidate_from_package()
    if pkg_bin is not None:
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
