"""Completeness weighting in the C indexer: raw must not move, weighted must.

Drives ``CompareSpots`` directly through a C harness that ``#include``s
``IndexerUnified.c`` (the function is ``static``). The alternative would be a
full synthetic indexing fixture — ``Spots.bin`` plus the ~1 GB binned
``Data.bin``/``nData.bin`` — to exercise twenty lines of arithmetic.

What the harness pins, in the order that matters:

1. **raw is unaffected.** ``ConfidenceMetric 0`` returns the integer ratio it
   always did, even with weights present in ``hkls[i][10]``. Every historical
   result depends on this.
2. **a uniform weight is exactly a no-op.** Both sides of the ratio scale
   together. This is the check that a *correct* implementation passes and a
   *never-applied* one also passes — which is precisely why (3) exists, and
   why the same trap was recorded at
   ``midas_nf_fitorientation/screen.py:330-334``.
3. **a non-uniform weight moves the number, in the right direction.**
   Down-weighting a matched reflection lowers completeness; down-weighting a
   missed one raises it. Both are checked against the closed-form value, not
   just a sign.
4. **``filtered`` removes a forbidden reflection from BOTH sides** — a missed
   reflection with |F|² = 0 takes completeness to 1.0, because it was never
   observable.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
_C_SRC = _PKG / "c_src"

_VERSION_H = """#ifndef MIDAS_VERSION_H
#define MIDAS_VERSION_H
#define MIDAS_VERSION "test"
#define MIDAS_GIT_HASH ""
#define MIDAS_GIT_DATE ""
#define MIDAS_VERSION_STRING "midas-index v" MIDAS_VERSION
#endif
"""


def test_comparespots_weighting(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler available")

    (tmp_path / "midas_version.h").write_text(_VERSION_H)
    exe = tmp_path / "cs_weight"
    cmd = [
        cc, "-std=gnu99", "-fopenmp", "-O2",
        "-I", str(_C_SRC), "-I", str(tmp_path),
        str(_PKG / "tests" / "comparespots_weight_test.c"),
        str(_C_SRC / "MIDAS_Math.c"),
        str(_C_SRC / "GetMisorientation.c"),
        str(_C_SRC / "forward.c"),
        "-lm", "-o", str(exe),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if "omp.h" in r.stderr or "fopenmp" in r.stderr:
            pytest.skip(f"OpenMP unavailable: {r.stderr[:200]}")
        pytest.fail(f"compile failed:\n{r.stderr[-3000:]}")

    out = subprocess.run([str(exe)], capture_output=True, text=True)
    print(out.stdout)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS (comparespots weighting)" in out.stdout

    # The harness prints each ratio; re-assert the two that carry the result so
    # a silent change to the harness's own thresholds cannot hide a regression.
    lines = {l.split("]")[0].lstrip("["): l for l in out.stdout.splitlines()
             if l.startswith("[")}
    assert "0.7500000000" in lines["raw"]
    assert "0.7500000000" in lines["uniform"], "uniform weight must be a no-op"
    assert "0.6774193548" in lines["matched   down"]
    assert "0.9677419355" in lines["unmatched down"]
    assert "1.0000000000" in lines["filtered"]
