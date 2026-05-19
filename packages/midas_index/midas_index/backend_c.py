"""Wrapper around the unified C indexer binary (`midas_indexer`).

The binary is built optionally at `pip install` time by scikit-build-core
(see ``packages/midas_index/CMakeLists.txt``). If OpenMP wasn't available
at install time the binary is absent — :func:`available` returns ``False``
and callers should fall back to the Python+numba backend (``backend="python"``,
the default of :meth:`Indexer.run`).

CLI contract (mirrors the unified binary at ``c_src/IndexerUnified.c``):

    midas_indexer paramstest.txt blockNr nBlocks nWork numProcs

- ``paramstest.txt`` — full path to the parameter file. The binary derives
  the input dir (Spots.bin, Data.bin, nData.bin, hkls.csv, positions.csv)
  from ``dirname(OutputFolder)``.
- ``nWork`` — FF mode: ``nSpotsToIndex`` (number of seeds in
  ``SpotsToIndex.csv``); PF mode: ``numScans`` (NB: actual ``numScans`` is
  re-read from ``positions.csv`` for safety; the argv value is informational).
- ``numProcs`` — OpenMP thread count passed to the binary's parallel region.

Mode auto-detect inside the binary: PF iff ``positions.csv`` has > 1 row,
else FF.
"""

from __future__ import annotations

import importlib.resources
import os
import subprocess
from pathlib import Path

__all__ = ["available", "binary_path", "run_indexer", "CBackendUnavailableError"]


class CBackendUnavailableError(RuntimeError):
    """Raised when the user asks for a C backend but the binary isn't present."""


def binary_path() -> Path:
    """Return the path to the bundled ``midas_indexer`` binary.

    The file may or may not exist on disk — use :func:`available` to test.

    Search order (first existing wins; otherwise the first candidate is
    returned for diagnostic purposes):

    1. ``<importlib.resources.files('midas_index')>/bin/midas_indexer``
       — the standard scikit-build-core wheel layout.
    2. ``<sys.prefix>/lib/python*/site-packages/midas_index/bin/midas_indexer``
       — covers the scikit-build-core editable-install case, where
       ``importlib.resources.files()`` resolves to the SOURCE dir (no
       binary there) but the binary lives in site-packages.
    3. ``<sys.exec_prefix>/lib/python*/site-packages/midas_index/bin/midas_indexer``
       — virtualenv edge case.
    """
    import sys

    candidates: list[Path] = []
    try:
        res = importlib.resources.files("midas_index") / "bin" / "midas_indexer"
        candidates.append(Path(str(res)))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for prefix in {sys.prefix, sys.exec_prefix}:
        c = Path(prefix) / "lib" / pyver / "site-packages" / "midas_index" / "bin" / "midas_indexer"
        candidates.append(c)
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0] if candidates else Path("midas_indexer")


def available() -> bool:
    """``True`` if the bundled ``midas_indexer`` binary is present and executable."""
    p = binary_path()
    return p.is_file() and os.access(p, os.X_OK)


def run_indexer(
    paramstest: str | os.PathLike[str],
    *,
    block_nr: int = 0,
    n_blocks: int = 1,
    n_work: int,
    num_procs: int = 1,
    extra_env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Invoke the unified ``midas_indexer`` binary.

    Parameters
    ----------
    paramstest
        Path to the parameter file. May be relative or absolute.
    block_nr, n_blocks
        Sharding params: process voxels/seeds in
        ``[block_nr * stride, (block_nr+1) * stride)``.
    n_work
        FF mode: ``nSpotsToIndex``. PF mode: ``numScans`` (the binary
        re-reads from positions.csv too; this argv value is informational).
    num_procs
        OpenMP thread count for the binary's parallel region.
    extra_env
        Optional environment-variable overrides (e.g.
        ``{"OMP_NUM_THREADS": "1"}`` for bit-deterministic single-threaded runs).
    cwd
        Working directory the binary runs in. Defaults to
        ``dirname(paramstest)`` so relative paths inside paramstest.txt
        resolve the way the C ReadParams expects.

    Returns
    -------
    subprocess.CompletedProcess
        Captured stdout/stderr (bytes) and returncode. Caller decides how
        to interpret a non-zero exit (the binary prints diagnostics to stderr).

    Raises
    ------
    CBackendUnavailableError
        If the bundled ``midas_indexer`` binary is not on disk. Re-install
        midas-index with a working OpenMP toolchain (macOS:
        ``brew install libomp``) or use ``backend="python"``.
    """
    if not available():
        raise CBackendUnavailableError(
            f"midas-index C backend binary not found at {binary_path()}. "
            "Re-install midas-index with a working OpenMP toolchain "
            "(macOS: `brew install libomp`; Linux: gcc with libgomp), or "
            "switch to backend='python'."
        )

    paramstest = Path(paramstest).resolve()
    if cwd is None:
        cwd = paramstest.parent

    cmd = [
        str(binary_path()),
        str(paramstest),
        str(int(block_nr)),
        str(int(n_blocks)),
        str(int(n_work)),
        str(int(num_procs)),
    ]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, check=False)
