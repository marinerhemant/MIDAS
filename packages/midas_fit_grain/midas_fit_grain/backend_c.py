"""Wrapper around the unified C refiner binary (`midas_fitgrain`).

Mirrors ``midas_index.backend_c``. The binary is built optionally at
``pip install`` time by scikit-build-core (see
``packages/midas_fit_grain/CMakeLists.txt``). If OpenMP / a C compiler wasn't
available at install time the binary is absent — :func:`available` returns
``False`` and callers fall back to the PyTorch refiner.

CLI contract (mirrors ``c_src/FitUnified.c`` and the indexer):

    midas_fitgrain paramstest.txt blockNr nBlocks nWork numProcs

- ``paramstest.txt`` — full path to the parameter file. The binary derives the
  input dir (Spots.bin, Data.bin, nData.bin, hkls.csv, positions.csv, the
  IndexBest seeds) from ``dirname(OutputFolder)``.
- ``nWork`` — FF mode: ``nSpotsToIndex`` (seeds in ``SpotsToIndex.csv``);
  PF mode: ``numScans`` (re-read from ``positions.csv`` for safety).
- ``numProcs`` — OpenMP thread count for the binary's parallel region.

Mode auto-detect inside the binary follows the indexer contract: PF iff
``positions.csv`` has > 1 row, else FF (the nScans==1 degeneracy).
"""

from __future__ import annotations

import importlib.resources
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

__all__ = ["available", "binary_path", "run_refiner", "CBackendUnavailableError"]

_BIN = "midas_fitgrain"
_PKG = "midas_fit_grain"


class CBackendUnavailableError(RuntimeError):
    """Raised when the user asks for the C backend but the binary isn't present."""


def binary_path() -> Path:
    """Return the path to the bundled ``midas_fitgrain`` binary.

    The file may or may not exist on disk — use :func:`available` to test.
    Search order mirrors ``midas_index.backend_c`` (wheel layout first, then
    site-packages fallbacks for scikit-build-core editable installs).
    """
    import sys

    candidates: list[Path] = []
    try:
        res = importlib.resources.files(_PKG) / "bin" / _BIN
        candidates.append(Path(str(res)))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for prefix in {sys.prefix, sys.exec_prefix}:
        candidates.append(
            Path(prefix) / "lib" / pyver / "site-packages" / _PKG / "bin" / _BIN
        )
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0] if candidates else Path(_BIN)


def available() -> bool:
    """``True`` if the bundled ``midas_fitgrain`` binary is present and executable."""
    p = binary_path()
    return p.is_file() and os.access(p, os.X_OK)


def run_refiner(
    paramstest: str | os.PathLike[str],
    *,
    block_nr: int = 0,
    n_blocks: int = 1,
    n_work: int,
    num_procs: int = 1,
    extra_env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    line_cb: Optional[Callable[[str], None]] = None,
) -> subprocess.CompletedProcess[bytes]:
    """Invoke the unified ``midas_fitgrain`` binary.

    Parameters mirror :func:`midas_index.backend_c.run_indexer`. ``n_work`` is
    ``nSpotsToIndex`` (FF) or ``numScans`` (PF). ``cwd`` defaults to
    ``dirname(paramstest)`` so relative paths in paramstest resolve as the C
    ReadParams expects.

    ``line_cb`` streams stdout line by line instead of buffering it to exit.
    ``FitUnified.c`` prints ``  progress: N/M seeds, R seeds/s`` ~200 times per
    run and ``fflush``es each because stdout is a pipe; the default
    ``capture_output`` path discards that liveness by holding everything until
    the process ends.

    Raises
    ------
    CBackendUnavailableError
        If the bundled binary is not on disk. Re-install midas-fit-grain with a
        working OpenMP toolchain (macOS: ``brew install libomp``) or use the
        PyTorch refiner.
    """
    if not available():
        raise CBackendUnavailableError(
            f"midas-fit-grain C backend binary not found at {binary_path()}. "
            "Re-install midas-fit-grain with a working OpenMP toolchain "
            "(macOS: `brew install libomp`; Linux: gcc with libgomp), or use "
            "the PyTorch refiner."
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
    if line_cb is None:
        return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True,
                              check=False)
    return _run_streaming(cmd, cwd=str(cwd), env=env, line_cb=line_cb)


def _run_streaming(
    cmd: list[str], *, cwd: str, env: dict[str, str],
    line_cb: Callable[[str], None],
) -> subprocess.CompletedProcess[bytes]:
    """Run ``cmd``, feeding each stdout line to ``line_cb`` as it arrives.

    Returns the same ``CompletedProcess`` shape as the buffered path, stdout
    included. stderr is spooled to a temp file rather than a second pipe --
    draining one pipe while the other fills is how these deadlock.

    (Copy of ``midas_index.backend_c._run_streaming``: the two packages are
    independent by design and neither should depend on the other for a
    twenty-line subprocess helper. Fix both together.)
    """
    chunks: list[bytes] = []
    with tempfile.TemporaryFile() as errf:
        popen = subprocess.Popen(cmd, cwd=cwd, env=env,
                                 stdout=subprocess.PIPE, stderr=errf)
        try:
            assert popen.stdout is not None
            for raw in iter(popen.stdout.readline, b""):
                chunks.append(raw)
                try:
                    line_cb(raw.decode("utf-8", errors="replace"))
                except Exception:
                    pass            # progress reporting must never fail a run
        finally:
            if popen.stdout is not None:
                popen.stdout.close()
            popen.wait()
        errf.seek(0)
        err = errf.read()
    return subprocess.CompletedProcess(
        cmd, popen.returncode, stdout=b"".join(chunks), stderr=err)
