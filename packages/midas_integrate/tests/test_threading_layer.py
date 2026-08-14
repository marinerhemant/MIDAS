"""The numba threading layer, pinned because getting it wrong SEGFAULTS.

The mapper kernels are ``@njit(parallel=True)``. numba's default layer on macOS
is OpenMP, and these run inside processes that have already loaded a different
OpenMP runtime -- torch and numpy each ship one, which is why
``KMP_DUPLICATE_LIB_OK=TRUE`` is needed to import them together at all. Two
OpenMP runtimes in one process crash when the parallel region opens.

Measured on an Apple-silicon Mac: ``midas_calibrate_v2.calibrate()`` died with
SIGSEGV inside ``map_kernel``, in-process and in a child process. With
``workqueue`` the identical call completes and matches Linux to every digit
printed. A segfault cannot be caught, so it cannot be regression-tested
directly; what IS testable is the setting that avoids it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _layer_after_import(env_value):
    """Import the mapper in a clean process and report the resulting layer."""
    env = dict(os.environ)
    env.pop("NUMBA_THREADING_LAYER", None)
    if env_value is not None:
        env["NUMBA_THREADING_LAYER"] = env_value
    out = subprocess.run(
        [sys.executable, "-c",
         "import midas_integrate._mapper_numba, os;"
         "print(os.environ.get('NUMBA_THREADING_LAYER', '<unset>'))"],
        capture_output=True, text=True, env=env, timeout=180,
    )
    assert out.returncode == 0, out.stderr[-800:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only guard")
def test_macos_defaults_to_workqueue():
    """Without this the calibration pipeline segfaults on macOS."""
    assert _layer_after_import(None) == "workqueue"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only guard")
@pytest.mark.parametrize("choice", ["omp", "tbb", "workqueue"])
def test_an_explicit_choice_is_never_overridden(choice):
    """``setdefault``, not assignment: a user who has chosen a layer -- to
    reproduce a result, or because they know their build is safe -- must keep
    it, even the one that crashes here."""
    assert _layer_after_import(choice) == choice


@pytest.mark.skipif(sys.platform == "darwin", reason="checks the non-macOS path")
def test_other_platforms_are_left_alone():
    """Linux is not touched: omp/tbb are faster and do not crash there."""
    assert _layer_after_import(None) == "<unset>"
