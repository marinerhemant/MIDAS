"""Round-trip tests for ExtraInfo / OrientPosFit / FitBest / Key binaries."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from midas_fit_grain.io_binary import (
    EXTRA_INFO_NCOLS,
    FIT_BEST_NCOLS,
    MAX_NHKLS_DEFAULT,
    ORIENT_POS_FIT_NCOLS,
    ExtraInfoSpot,
    GrainResult,
    read_extra_info,
    read_fit_best,
    read_key,
    read_orient_pos_fit,
    write_fit_best_row,
    write_key_row,
    write_orient_pos_fit_row,
    write_process_key_row,
)


def test_extra_info_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((23, EXTRA_INFO_NCOLS)).astype(np.float64)
    p = tmp_path / "ExtraInfo.bin"
    arr.tofile(p)
    out = read_extra_info(p, mmap=False)
    assert out.shape == arr.shape
    np.testing.assert_array_equal(out, arr)
    out_mmap = read_extra_info(p, mmap=True)
    np.testing.assert_array_equal(out_mmap, arr)


def test_extra_info_size_check(tmp_path):
    p = tmp_path / "ExtraInfo.bin"
    np.zeros(EXTRA_INFO_NCOLS - 1, dtype=np.float64).tofile(p)
    with pytest.raises(ValueError, match="multiple"):
        read_extra_info(p, mmap=False)


def test_extra_info_spot_dataclass():
    row = np.arange(EXTRA_INFO_NCOLS, dtype=np.float64) + 0.5
    s = ExtraInfoSpot.from_row(row)
    assert s.YLab == 0.5
    assert s.SpotID == 4
    assert s.RingNumber == 5
    assert s.maskTouched == 14.5
    assert s.FitRMSE == 15.5


def test_grain_result_layout():
    g = GrainResult(
        SpotID=42,
        OrientMat=np.arange(9, dtype=np.float64),
        Position=np.array([1.0, 2.0, 3.0]),
        LatticeFit=np.array([4.04, 4.04, 4.04, 90.0, 90.0, 90.0]),
        ErrorPos=0.1, ErrorOme=0.2, ErrorAngle=0.3,
        meanRadius=50.0, completeness=0.99,
    )
    row = g.to_row()
    assert row.shape == (ORIENT_POS_FIT_NCOLS,)
    # Layout per FitPosOrStrainsOMP.c:3007-3025:
    #   [0]=SpotID, [1..9]=OrientMat, [10]=SpotID, [11..13]=Pos,
    #   [14]=SpotID, [15..20]=Lattice, [21]=SpotID,
    #   [22]=ErrorPos(pos,um), [23]=ErrorOme(ome,deg), [24]=ErrorAngle(IA,deg),
    #   [25]=meanRadius, [26]=completeness
    assert row[0] == 42 and row[10] == 42 and row[14] == 42 and row[21] == 42
    np.testing.assert_array_equal(row[1:10], np.arange(9))
    np.testing.assert_array_equal(row[11:14], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(row[15:21], [4.04, 4.04, 4.04, 90, 90, 90])
    assert row[22] == 0.1 and row[23] == 0.2 and row[24] == 0.3
    assert row[25] == 50.0 and row[26] == 0.99


def test_orient_pos_fit_pwrite_strided(tmp_path):
    """Writing rows out of order should land at the right offsets."""
    p = tmp_path / "OrientPosFit.bin"
    g0 = GrainResult(
        SpotID=10, OrientMat=np.zeros(9), Position=np.array([1, 2, 3]),
        LatticeFit=np.array([1, 2, 3, 4, 5, 6]),
        ErrorPos=0.1, ErrorOme=0.2, ErrorAngle=0.3,
        meanRadius=10.0, completeness=0.5,
    )
    g3 = GrainResult(
        SpotID=99, OrientMat=np.ones(9), Position=np.array([7, 8, 9]),
        LatticeFit=np.array([10, 20, 30, 40, 50, 60]),
        ErrorPos=1.1, ErrorOme=1.2, ErrorAngle=1.3,
        meanRadius=20.0, completeness=0.9,
    )
    write_orient_pos_fit_row(p, 0, g0)
    write_orient_pos_fit_row(p, 3, g3)

    expected_size = 4 * ORIENT_POS_FIT_NCOLS * 8
    assert p.stat().st_size == expected_size

    out = read_orient_pos_fit(p, n_grains=4)
    np.testing.assert_array_equal(out[0], g0.to_row())
    np.testing.assert_array_equal(out[1], np.zeros(ORIENT_POS_FIT_NCOLS))
    np.testing.assert_array_equal(out[2], np.zeros(ORIENT_POS_FIT_NCOLS))
    np.testing.assert_array_equal(out[3], g3.to_row())


def test_key_pwrite_strided(tmp_path):
    p = tmp_path / "Key.bin"
    write_key_row(p, 0, spot_id=11, n_spots_comp=200)
    write_key_row(p, 5, spot_id=22, n_spots_comp=180)
    out = read_key(p, n_grains=6)
    assert out[0].tolist() == [11, 200]
    assert out[5].tolist() == [22, 180]
    assert out[1].tolist() == [0, 0]


def test_fit_best_pwrite_sparse_block(tmp_path):
    p = tmp_path / "FitBest.bin"
    n_spots = 7
    rng = np.random.default_rng(1)
    payload = rng.standard_normal((n_spots, FIT_BEST_NCOLS))
    write_fit_best_row(p, row_nr=2, per_spot=payload, max_nhkls=64)
    arr = read_fit_best(p, n_grains=3, max_nhkls=64)
    np.testing.assert_array_equal(arr[2, :n_spots], payload)
    # Trailing rows still zero.
    np.testing.assert_array_equal(arr[2, n_spots:], np.zeros(
        (64 - n_spots, FIT_BEST_NCOLS)))


def test_fit_best_rejects_overflow(tmp_path):
    p = tmp_path / "FitBest.bin"
    with pytest.raises(ValueError, match="exceeds"):
        write_fit_best_row(p, 0, np.zeros((10, FIT_BEST_NCOLS)), max_nhkls=5)


def test_fit_best_rejects_wrong_shape(tmp_path):
    p = tmp_path / "FitBest.bin"
    with pytest.raises(ValueError, match="22"):
        write_fit_best_row(p, 0, np.zeros((4, 21)))


def test_process_key_roundtrip(tmp_path):
    """pwrite past EOF leaves the file at offset+payload — no full-stride pad."""
    p = tmp_path / "ProcessKey.bin"
    ids = np.array([7, 8, 9, 10, 11], dtype=np.int32)
    write_process_key_row(p, 1, ids, max_nhkls=16)
    raw = np.fromfile(p, dtype=np.int32)
    # On disk: row 0 hole (16 zeros from sparse seek) + 5 written ints.
    assert raw.size == 16 + 5
    assert raw[:16].tolist() == [0] * 16
    assert raw[16:16 + 5].tolist() == [7, 8, 9, 10, 11]


def test_process_key_full_population(tmp_path):
    """Writing every row up to n_grains-1 fills the file to the full stride."""
    p = tmp_path / "ProcessKey.bin"
    n_grains = 3
    max_nhkls = 8
    for r in range(n_grains):
        ids = np.full(max_nhkls, r + 1, dtype=np.int32)
        write_process_key_row(p, r, ids, max_nhkls=max_nhkls)
    raw = np.fromfile(p, dtype=np.int32)
    assert raw.size == n_grains * max_nhkls
    raw = raw.reshape(n_grains, max_nhkls)
    for r in range(n_grains):
        assert (raw[r] == r + 1).all()
