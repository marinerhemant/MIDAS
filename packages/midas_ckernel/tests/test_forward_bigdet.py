"""The BigDetector mask path, which ``test_forward_parity.py`` cannot cover.

The parity harness runs with ``BigDetSize = 0`` on purpose -- it has to, since
the legacy bodies it compares against are the *indexer's*, which never had a
mask. So the mask branch in ``forward.c`` has no coverage from it at all.

That branch matters as of 2026-08-22: the indexer now passes a real
``MidasCkBigDet`` (reversing plan ruling #5), so a reflection predicted onto a
dead pixel leaves both sides of the completeness ratio. The C harness this test
drives checks the three things that can silently go wrong:

* an all-bits-set mask must be indistinguishable from no mask;
* clearing a spot's cell must remove it from ``nSpotsFracCalc`` -- the
  completeness *denominator* -- and not merely from the emitted list;
* a grid too small for the predicted spots must drop them, not read past the
  end of the bitset. ``yl``/``zl`` are bounded only by ``BoxSizes``, commonly
  +/-1e6 um, while the grid is sized to the detector, so this is reachable in
  ordinary use rather than a contrived case.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
_C_SRC = _PKG / "c_src"


@pytest.mark.parity
def test_forward_bigdet_mask(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler available")

    exe = tmp_path / "ck_bigdet"
    cmd = [
        cc, "-O2", "-I", str(_C_SRC),
        str(_PKG / "tests" / "bigdet_test.c"),
        str(_C_SRC / "forward.c"),
        str(_C_SRC / "MIDAS_Math.c"),
        "-lm", "-o", str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = subprocess.run(
        [str(exe)], check=True, capture_output=True, text=True
    ).stdout
    print(out)

    assert "PASS (bigdet)" in out, out
    for tag in ("[all-set]", "[cleared]", "[bounds]"):
        line = next((l for l in out.splitlines() if l.startswith(tag)), None)
        assert line is not None and line.endswith("PASS"), f"{tag}: {out}"
