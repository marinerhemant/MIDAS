"""Unit tests for ``stages/transforms._write_layer_extra`` (PF y-offset).

Regression guard for E2(d): an earlier version looked up the legacy-C
column name ``"YOrig(NoWedgeCorr)"`` — which no header variant ever
contained — so the second lab-frame Y shift was a silent no-op. The shift
must hit ``YLab`` + ``YOrigDetCor`` (both lab um), never ``YRawPx``
(detector pixels), and must fail loud on an alien header.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.stages.transforms import _write_layer_extra

NEW_HEADER = (
    "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta "
    "OmegaIni YOrigDetCor ZOrigDetCor YRawPx ZRawPx "
    "OmegaDetCor IntegratedIntensity RawSumIntensity maskTouched FitRMSE"
)
# Pre-fix (mislabeled cols 11+) header written by deployed <=0.7.2 wheels.
OLD_HEADER = (
    "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta "
    "OmegaIni YOrigDetCor ZOrigDetCor YOrigNoWedge ZOrigNoWedge "
    "IntegratedIntensity RawSumIntensity FitRMSE maskTouched FitErrCode"
)

LSD = 1_000_000.0


def _write_src(path: Path, header: str) -> np.ndarray:
    """Two valid spots + one zero-padded (GrainRadius=0) row."""
    rows = np.zeros((3, 18), dtype=np.float64)
    #        YLab    ZLab   Ome  GR   SID  Ring  Eta  Ttheta
    rows[0, :8] = [100.0, 500.0, 10.0, 5.0, 1.0, 2.0, -11.31, 0.03]
    rows[1, :8] = [-250.0, 400.0, 20.0, 5.0, 2.0, 2.0, 32.0, 0.03]
    rows[2, 4] = 3.0                       # zero-padded row, SpotID only
    rows[:2, 9] = rows[:2, 0] + 1.0        # YOrigDetCor (lab um, pre-wedge)
    rows[:2, 10] = rows[:2, 1]             # ZOrigDetCor
    rows[:2, 11] = 1024.5                  # YRawPx (detector px)
    rows[:2, 12] = 767.25                  # ZRawPx
    with open(path, "w") as f:
        f.write(header + "\n")
        np.savetxt(f, rows, fmt="%.6f")
    return rows


@pytest.mark.parametrize("header", [NEW_HEADER, OLD_HEADER],
                         ids=["new-header", "old-mislabeled-header"])
def test_y_offset_shifts_both_lab_columns(tmp_path: Path, header: str):
    src, dst = tmp_path / "src.csv", tmp_path / "dst.csv"
    rows = _write_src(src, header)
    y_pos = 42.0
    _write_layer_extra(src_extra=src, dst_extra=dst, y_position=y_pos, Lsd=LSD)

    out = np.loadtxt(dst, skiprows=1)
    # Valid rows: YLab (col 0) and YOrigDetCor (col 9) both shifted.
    np.testing.assert_allclose(out[:2, 0], rows[:2, 0] + y_pos, atol=1e-6)
    np.testing.assert_allclose(out[:2, 9], rows[:2, 9] + y_pos, atol=1e-6)
    # Raw detector pixels (cols 11/12) untouched.
    np.testing.assert_allclose(out[:2, 11], rows[:2, 11], atol=1e-9)
    np.testing.assert_allclose(out[:2, 12], rows[:2, 12], atol=1e-9)
    # Zero-padded row untouched.
    np.testing.assert_allclose(out[2, [0, 9]], 0.0, atol=1e-9)

    # Eta/Ttheta recomputed from the SHIFTED y (CalcEtaAngleAll convention).
    y, z = out[0, 0], out[0, 1]
    norm = math.hypot(y, z)
    eta = math.degrees(math.acos(z / norm))
    if y > 0:
        eta = -eta
    assert out[0, 6] == pytest.approx(eta, abs=1e-5)
    assert out[0, 7] == pytest.approx(math.degrees(math.atan(norm / LSD)), abs=1e-5)


def test_missing_lab_column_fails_loud(tmp_path: Path):
    src, dst = tmp_path / "src.csv", tmp_path / "dst.csv"
    bad_header = NEW_HEADER.replace("YOrigDetCor", "SomethingElse")
    _write_src(src, bad_header)
    with pytest.raises(ValueError, match="YOrigDetCor"):
        _write_layer_extra(src_extra=src, dst_extra=dst, y_position=1.0, Lsd=LSD)
    assert not dst.exists(), "must not write an unshifted layer CSV"


def test_empty_src_passthrough(tmp_path: Path):
    src, dst = tmp_path / "src.csv", tmp_path / "dst.csv"
    src.write_text(NEW_HEADER + "\n")
    _write_layer_extra(src_extra=src, dst_extra=dst, y_position=1.0, Lsd=LSD)
    assert dst.exists()
