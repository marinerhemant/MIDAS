"""Compile and run the C forward-parity harness as a pytest.

Proves the unified forward (``c_src/forward.c``) reproduces:
  * the legacy indexer ``CalcDiffrSpots`` BIT-IDENTICALLY, and
  * the legacy refiner ``CalcDiffractionSpots`` to ULP (~3e-9, by design R2).

Skips gracefully if a C compiler or the legacy refiner source is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
_C_SRC = _PKG / "c_src"
# Frozen pre-unification refiner forward, kept in-tree as the parity reference
# (the shipped CalcDiffractionSpots.c is now an adapter over the shared forward).
_REFINER_REF = _PKG / "tests" / "legacy_refiner_forward.c"


@pytest.mark.parity
def test_forward_bit_parity(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler available")

    exe = tmp_path / "ck_parity"
    cmd = [
        cc, "-O2", "-I", str(_C_SRC),
        str(_PKG / "tests" / "parity_test.c"),
        str(_C_SRC / "forward.c"),
        str(_C_SRC / "MIDAS_Math.c"),
        str(_REFINER_REF),
        "-lm", "-o", str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    print(out)

    assert "PASS (bit-identical)" in out, out
    # indexer line must report zero bit-mismatches
    idx_line = next(l for l in out.splitlines() if l.startswith("[indexer]"))
    assert "BIT-mismatches: 0" in idx_line, idx_line
