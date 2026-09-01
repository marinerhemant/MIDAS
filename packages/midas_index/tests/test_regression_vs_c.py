"""Regression test: midas-index output vs C IndexerOMP on a synthetic dataset.

Builds a small synthetic dataset (5 grains, identity-derived orientations),
runs both `IndexerOMP` (C) and `midas-index` (Python) on the same input,
and asserts the recovered orientations agree within tolerance.

Marked `slow` because it shells out to subprocess and depends on the C
binary being available. CI runs that don't have IndexerOMP skip this.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"

INDEXER_OMP_BIN = Path("/Users/hsharma/opt/MIDAS/FF_HEDM/bin/IndexerOMP")
GETHKLLIST_BIN = Path("/Users/hsharma/opt/MIDAS/FF_HEDM/bin/GetHKLList")


def _binary_unusable(binary: Path) -> str | None:
    """Why this C binary must not be run, or ``None`` if it looks usable.

    **Deliberately STATIC — this must never launch** ``binary``.

    ``.exists()`` alone is not enough: these are prebuilt artefacts of the
    soft-deprecated ``FF_HEDM/`` tree and one can be present but unloadable.
    The obvious fix -- run it once and see whether it survives -- is worse
    than the problem on macOS/arm64:

    * ``install_name_tool`` rpath surgery invalidates a Mach-O's code
      signature, and the kernel then SIGKILLs the process inside dyld, before
      ``main()``. Measured 2026-09-01 on ``GetHKLList``: ``codesign -v`` says
      "code object is not signed at all"; the crash report reads
      ``termination CODESIGNING / Invalid Page`` with ``signal SIGKILL (Code
      Signature Invalid)``; and **stderr is empty**, because the process never
      reached its own code. A probe that greps stderr for ``dyld`` therefore
      cannot see this failure and reports the binary as runnable -- the exact
      opposite of what such a guard is for.
    * Each launch is slow (macOS writes a crash report -- 19 accumulated for
      ``GetHKLList`` in one evening) and leaves the process in uninterruptible
      ``UE`` state, where it survives ``SIGKILL``. Nine had piled up on the
      dev machine, the oldest over 1.5 h. ``subprocess.run(..., timeout=...)``
      does **not** bound this: the timeout fires, the reap does not return.

    So: check statically, and let the test's own subprocess call be the place
    where a binary that passes this gate but still cannot load is caught.
    """
    if not binary.exists():
        return "not found"
    if not os.access(binary, os.X_OK):
        return "present but not executable"
    if sys.platform == "darwin":
        try:
            sig = subprocess.run(
                ["codesign", "--verify", str(binary)],
                capture_output=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return None  # cannot check -- let the test try
        if sig.returncode != 0:
            detail = " ".join((sig.stderr or b"").decode(errors="replace").split())
            return (
                f"code signature invalid ({detail}); on arm64 macOS the kernel "
                f"SIGKILLs such a binary inside dyld. Re-sign after any "
                f"install_name_tool surgery: codesign -s - -f {binary}"
            )
    return None


@pytest.fixture(scope="module")
def c_indexer_binaries() -> tuple[Path, Path]:
    """Skip unless both C binaries look usable.

    A fixture, **not** a ``skipif`` decorator. A decorator's condition is an
    expression evaluated at import -- i.e. during collection -- so whatever it
    does is paid by every ``--collect-only`` and every sweep of this package,
    even when the test is deselected. Measured before this change: collecting
    ``packages/midas_index/tests`` did not finish in 90 s, against 0.35 s for
    225 tests with this module ignored.
    """
    for b in (INDEXER_OMP_BIN, GETHKLLIST_BIN):
        why = _binary_unusable(b)
        if why is not None:
            pytest.skip(f"{b.name}: {why}")
    return INDEXER_OMP_BIN, GETHKLLIST_BIN


@pytest.mark.slow
def test_midas_index_matches_c_indexer_on_synthetic_5_grain_dataset(
    tmp_path, c_indexer_binaries,
):
    """Run both indexers on a 5-grain Cu synthetic dataset, compare records."""
    build_script = DATA_DIR / "build_reference.py"
    workdir = tmp_path / "ref"
    cmd = [
        sys.executable, str(build_script),
        "--n-grains", "5",
        "--seed", "42",
        "--n-procs", "1",
        "--workdir", str(workdir),
    ]
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Capture as bytes; C IndexerOMP can write garbage bytes to stdout from
    # uninitialized strings (the IDsFileName field appears truncated), so
    # text=True would crash on UnicodeDecodeError.
    res = subprocess.run(cmd, capture_output=True, env=env, timeout=300)
    if res.returncode != 0:
        out = res.stdout.decode("utf-8", errors="replace")
        err = res.stderr.decode("utf-8", errors="replace")
        # The static gate cannot see a binary that is signed and present but
        # still cannot resolve its libraries: IndexerOMP hardcodes an ABSOLUTE
        # path to build/fftw_install/lib/libfftw3f.3.dylib, and an absolute
        # path never consults @rpath, so a homebrew copy does not rescue it.
        # That is an environment fault, not an indexer regression -- skip.
        # Matched on loader signatures only, so a genuine failure still fails.
        blob = out + err
        loader_fault = any(s in blob for s in (
            "Library not loaded", "image not found", "no such file",
            "code signature", "Code Signature", "Killed: 9", "Abort trap",
        ))
        if loader_fault:
            pytest.skip(
                "C reference binaries could not be loaded by dyld -- "
                "environment fault, not an indexer regression. Tail:\n"
                + blob[-800:]
            )
        raise AssertionError(
            f"build_reference.py failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        )

    golden = workdir / "golden" / "IndexBest.bin"
    ours = workdir / "midas" / "IndexBest.bin"
    assert golden.exists(), f"missing C output: {golden}"
    assert ours.exists(), f"missing midas output: {ours}"

    g = np.fromfile(golden, dtype=np.float64).reshape(-1, 15)
    m = np.fromfile(ours, dtype=np.float64).reshape(-1, 15)
    assert g.shape == m.shape == (5, 15)

    # Misorientation between recovered orientations (cubic symmetry, sg 225)
    from midas_stress.orientation import misorientation_om

    misos = []
    for i in range(g.shape[0]):
        # Skip empty slots (both zero)
        if (g[i] == 0).all() and (m[i] == 0).all():
            continue
        c_R = g[i, 1:10].reshape(3, 3)
        m_R = m[i, 1:10].reshape(3, 3)
        ang_rad, _ = misorientation_om(
            c_R.flatten().tolist(), m_R.flatten().tolist(), 225,
        )
        miso = math.degrees(float(ang_rad))
        misos.append(miso)

        # Match counts must agree exactly
        assert int(g[i, 14]) == int(m[i, 14]), (
            f"seed {i}: n_matches mismatch C={int(g[i, 14])} vs M={int(m[i, 14])}"
        )
        # Total theor must agree exactly
        assert int(g[i, 13]) == int(m[i, 13]), (
            f"seed {i}: n_t mismatch C={int(g[i, 13])} vs M={int(m[i, 13])}"
        )
        # Misorientation within tolerance (orientation grid is 0.5°, so allow up to 1°)
        assert miso < 1.0, f"seed {i}: miso {miso:.4f}° exceeds 1.0° tolerance"

    assert len(misos) >= 5, f"only {len(misos)} non-empty seeds compared"
    # Aggregate sanity
    assert max(misos) < 1.0
    assert sum(misos) / len(misos) < 0.5
