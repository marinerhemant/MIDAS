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
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

LOG = logging.getLogger(__name__)

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
    line_cb: Optional[Callable[[str], None]] = None,
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
    line_cb
        Optional per-line stdout callback, invoked while the binary runs.

        ``IndexerUnified.c`` prints ``  progress: N/M voxels, R vox/s`` about
        200 times per run and ``fflush``es each one *specifically because
        stdout is a pipe*. The default path here uses ``capture_output``, which
        buffers in this process until exit and throws that away -- a PF
        indexing stage can run eight hours with no sign of life. Passing
        ``line_cb`` streams instead, so the progress reaches the caller live.

        stderr is spooled to a temp file rather than a second pipe: draining
        one pipe while the other fills is how these deadlock.

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
    # Provenance. A campaign's run logs recorded the paramstest path but never
    # which binary produced the output; when two runs of "the same thing"
    # disagreed later, the binary had been rebuilt and the question was
    # unanswerable. Path + size + sha256 of the actual executable, and the
    # parameter file it was handed, cost nothing and settle it.
    _log_invocation(cmd, paramstest)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if line_cb is None:
        return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True,
                              check=False)
    return _run_streaming(cmd, cwd=str(cwd), env=env, line_cb=line_cb)


def binary_fingerprint() -> dict:
    """Path, size and sha256 of the bundled ``midas_indexer``.

    Provenance for a run. ``midas_index.__version__`` is NOT sufficient on its
    own: a rebuild from a different tree can carry the same version string, and
    on this project a released patch version was once (wrongly) blamed for a
    result change that turned out to be an invocation difference. The hash is
    what actually identifies the executable.
    """
    import hashlib

    p = binary_path()
    out = {"path": str(p), "exists": p.exists(), "sha256": None, "size": None}
    if p.exists():
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
        out["size"] = p.stat().st_size
    return out


def _log_invocation(cmd: list[str], paramstest: Path) -> None:
    """Emit the binary's identity and the parameter file it was handed."""
    try:
        from . import __version__ as _v
    except Exception:                                     # noqa: BLE001
        _v = "unknown"
    try:
        fp = binary_fingerprint()
        LOG.info("midas_indexer: %s (midas-index %s, %s bytes, sha256 %s)",
                 fp["path"], _v, fp["size"],
                 (fp["sha256"] or "?")[:16])
        LOG.info("midas_indexer: paramstest=%s", paramstest)
        LOG.info("midas_indexer: argv=%s", " ".join(cmd[1:]))
    except Exception:                                     # noqa: BLE001
        # Provenance logging must never be the reason a run fails.
        pass


def _run_streaming(
    cmd: list[str], *, cwd: str, env: dict[str, str],
    line_cb: Callable[[str], None],
) -> subprocess.CompletedProcess[bytes]:
    """Run ``cmd``, feeding each stdout line to ``line_cb`` as it arrives.

    Returns the same ``CompletedProcess`` shape as the buffered path, stdout
    included, so callers cannot tell the two apart afterwards.

    (``midas_fit_grain.backend_c`` carries a copy of this: the two packages are
    independent by design and neither should depend on the other for a
    twenty-line subprocess helper. Fix both together.)
    """
    chunks: list[bytes] = []
    with tempfile.TemporaryFile() as errf:
        popen = subprocess.Popen(cmd, cwd=cwd, env=env,
                                 stdout=subprocess.PIPE, stderr=errf)
        try:
            assert popen.stdout is not None
            # readline(), not `for line in popen.stdout`, to be explicit that
            # each line is handed over the moment the binary flushes it.
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
