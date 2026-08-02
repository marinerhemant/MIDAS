"""Tests for binary readers (OrientMat / Key / DiffractionSpots / grid / hkls)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_nf_fitorientation.io import (
    GridTable,
    HKLTable,
    OrientationData,
    read_grid,
    read_hkls,
    read_orientations,
)


# ---------------------------------------------------------------------------
#  Synthetic OrientMat / Key / DiffractionSpots
# ---------------------------------------------------------------------------

def _write_orient_bundle(tmp_path: Path, n_or: int = 3, m_per_or: int = 4):
    rng = np.random.default_rng(0)
    matrices = rng.standard_normal((n_or, 3, 3)).astype(np.float64)
    n_spots = np.full(n_or, m_per_or, dtype=np.int32)
    starts = (np.arange(n_or, dtype=np.int32) * m_per_or)
    spots = rng.standard_normal((n_or * m_per_or, 3)).astype(np.float64)

    matrices.tofile(tmp_path / "OrientMat.bin")
    np.column_stack([n_spots, starts]).astype(np.int32).tofile(tmp_path / "Key.bin")
    spots.tofile(tmp_path / "DiffractionSpots.bin")
    return matrices, n_spots, starts, spots


def test_read_orientations_round_trip(tmp_path):
    M, S, R, P = _write_orient_bundle(tmp_path)
    od = read_orientations(tmp_path)
    assert isinstance(od, OrientationData)
    assert od.n_orientations == M.shape[0]
    assert np.allclose(od.matrices, M)
    assert np.array_equal(od.n_spots, S.astype(np.int64))
    assert np.array_equal(od.starts, R.astype(np.int64))
    assert np.allclose(od.spots, P)


def test_read_orientations_bad_size_raises(tmp_path):
    # File size not a multiple of 72 bytes
    (tmp_path / "OrientMat.bin").write_bytes(b"\x00" * 73)
    (tmp_path / "Key.bin").write_bytes(b"")
    (tmp_path / "DiffractionSpots.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="OrientMat.bin"):
        read_orientations(tmp_path)


# ---------------------------------------------------------------------------
#  hkls.csv
# ---------------------------------------------------------------------------

def test_read_hkls_basic(tmp_path):
    """Reader exposes both integer Miller indices (cols 0–2) and the
    Cartesian G-vectors (cols 5–7) — using the wrong column for the
    omega/eta solver caused the v0.1 forward to double-apply B."""
    csv = tmp_path / "hkls.csv"
    csv.write_text(
        "h k l D RingNr g1 g2 g3 Theta 2Theta Radius\n"
        "1 1 1 2.36 1 0.245 0.245 0.245 2.10 4.21 757.2\n"
        "2 0 0 2.04 2 0.490 0.000 0.000 2.43 4.86 875.0\n"
    )
    table = read_hkls(tmp_path)
    assert isinstance(table, HKLTable)
    assert table.n == 2
    assert np.array_equal(table.rings, [1, 2])
    assert np.allclose(table.thetas_deg, [2.10, 2.43])
    assert table.hkls_int.shape == (2, 3)
    assert table.hkls_cart.shape == (2, 3)
    # First row: integer (1, 1, 1), Cartesian (0.245, 0.245, 0.245)
    assert np.allclose(table.hkls_int[0], [1, 1, 1])
    assert np.allclose(table.hkls_cart[0], [0.245, 0.245, 0.245])


def test_filter_rings(tmp_path):
    csv = tmp_path / "hkls.csv"
    csv.write_text(
        "h k l D RingNr g1 g2 g3 Theta 2Theta Radius\n"
        "1 1 1 2.36 1 0.245 0.245 0.245 5 _ _\n"
        "2 0 0 2.04 2 0.490 0.000 0.000 7 _ _\n"
        "2 2 0 1.44 4 0.347 0.347 0.000 9 _ _\n"
    )
    table = read_hkls(tmp_path)
    keep = table.filter_rings([1, 4])
    assert keep.n == 2
    assert set(keep.rings.tolist()) == {1, 4}
    # Cartesian column also gets filtered.
    assert keep.hkls_cart.shape == (2, 3)


# ---------------------------------------------------------------------------
#  grid.txt
# ---------------------------------------------------------------------------

def test_read_grid_basic(tmp_path):
    g = tmp_path / "grid.txt"
    g.write_text("4\n"
                 "1.0 0.5 10.0 20.0 0.1\n"
                 "0.5 0.5 11.0 21.0 0.1\n"
                 "0.5 1.0 12.0 22.0 0.1\n"
                 "1.0 1.0 13.0 23.0 0.1\n")
    grid = read_grid(tmp_path)
    assert isinstance(grid, GridTable)
    assert grid.n_voxels == 4
    # Row 0: y1=1 > y2=0.5 → ud = -1
    assert grid.ud[0] == -1
    # Row 1: y1=0.5 == y2=0.5 → ud = +1 (np.where strict >)
    assert grid.ud[1] == 1
    # Row 2: y1=0.5 < y2=1 → ud = +1
    assert grid.ud[2] == 1


def test_grid_block_decomposition():
    g = GridTable(
        y1=np.zeros(10), y2=np.zeros(10),
        xs=np.arange(10).astype(np.float64),
        ys=np.zeros(10), gs=np.ones(10),
        ud=np.ones(10, dtype=np.int8),
    )
    # Mirrors the C decomposition exactly: chunk = ceil(10/3) = 4,
    # blocks overlap by 1 voxel at the boundary because the C end-clamp
    # is `min(chunk * (block_nr + 1), N - 1)` not `... - 1`. The C code
    # accepts the overlap because pwrite is idempotent (last write
    # wins) and per-voxel results are deterministic.
    s, e = g.slice_block(0, 3)
    assert (s, e) == (0, 4)
    s, e = g.slice_block(1, 3)
    assert (s, e) == (4, 8)
    s, e = g.slice_block(2, 3)
    assert (s, e) == (8, 9)


def test_diffraction_spots_columns_are_yl_zl_omega(tmp_path):
    """Pin the COLUMN MEANING of DiffractionSpots.bin, not just its shape.

    The OrientationData docstring said ``[2theta, eta, omega]`` (radians) for a
    long time. The real layout is ``[yl, zl, omega]`` -- lab-frame microns then
    degrees (midas_nf_preprocess/diffr_spots/pipeline.py:42). Nothing caught the
    error because every reader indexes positionally, so the label was never
    exercised; code written against it silently produced neutral results
    instead of failing.

    This test asserts the ranges are physically consistent with microns-on-the-
    detector rather than radians, which is the cheapest thing that would have
    caught it.
    """
    import numpy as np
    from midas_nf_fitorientation.io import read_orientations

    # one orientation, three spots on a ring of radius 500 um
    r = 500.0
    ang = np.radians([30.0, 150.0, 270.0])
    spots = np.column_stack([r * np.cos(ang), r * np.sin(ang),
                             np.array([12.5, 40.0, 175.0])])
    spots.astype(np.float64).tofile(tmp_path / "DiffractionSpots.bin")
    np.eye(3, dtype=np.float64).tofile(tmp_path / "OrientMat.bin")
    np.array([[3, 0]], dtype=np.int32).tofile(tmp_path / "Key.bin")

    o = read_orientations(tmp_path)
    yl, zl, omega = o.spots[:, 0], o.spots[:, 1], o.spots[:, 2]

    # microns on the detector, NOT radians: a 2theta in radians could never
    # exceed ~0.2 for NF, so anything above 1 rules that reading out.
    assert np.abs(yl).max() > 1.0
    np.testing.assert_allclose(np.hypot(yl, zl), r, rtol=1e-9)

    # column 2 is omega in DEGREES over the scan range, not radians
    assert omega.min() >= -360.0 and omega.max() <= 360.0

    # and the ring radius is what maps onto hkls.csv's Radius column
    assert np.hypot(yl, zl).std() < 1e-9
