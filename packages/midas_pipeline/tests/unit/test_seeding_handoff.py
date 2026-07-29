"""Tests for seeding/handoff.py (Stage D of merged-FF seeding).

Verifies the Grains.csv → UniqueOrientations.csv conversion:
- Parses the ``%GrainID`` header + ``O11..O33`` column order.
- Emits the 14-col layout the per-voxel scanning indexer reads.
- Handles a missing header → ValueError with a clear message.
- Optional dedup_misorientation_deg drops symmetry-equivalent OMs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.seeding.handoff import (
    _parse_grains_csv,
    grains_csv_to_unique_orientations,
)


def _make_grains_csv(path: Path, oms: np.ndarray,
                     grain_ids=None) -> None:
    """Write a minimal Grains.csv with the %GrainID header + O11..O33 cols."""
    n = oms.shape[0]
    if grain_ids is None:
        grain_ids = np.arange(1, n + 1)
    # Header line, then n data rows. Add a couple of distractor columns
    # before the OM block so we exercise the dynamic column-finder.
    cols = ["%GrainID", "x", "y", "z",
            "O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
            "extra"]
    lines = [" ".join(cols)]
    for i, om in enumerate(oms):
        row = [str(int(grain_ids[i])), "0.0", "0.0", "0.0",
               *(f"{v:.9f}" for v in om.ravel()), "1.0"]
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_grains_csv_extracts_om_block(tmp_path: Path):
    om1 = np.eye(3).ravel()
    om2 = np.diag([1, -1, -1]).astype(np.float64).ravel()
    grains = tmp_path / "Grains.csv"
    _make_grains_csv(grains, np.stack([om1, om2]), grain_ids=[7, 12])
    oms, ids = _parse_grains_csv(grains)
    assert oms.shape == (2, 9)
    np.testing.assert_allclose(oms[0], om1)
    np.testing.assert_allclose(oms[1], om2)
    assert ids == [7, 12]


def test_parse_grains_csv_missing_header_raises(tmp_path: Path):
    grains = tmp_path / "Grains.csv"
    grains.write_text("not a header\n1 2 3 4 5 6 7 8 9 10 11 12 13 14\n")
    # Header detection keys off the OM columns (the ID column is spelled both
    # 'GrainID' and 'ID' across ProcessGrains versions), so the message names
    # those rather than '%GrainID'.
    with pytest.raises(ValueError, match="not a ProcessGrains output file"):
        _parse_grains_csv(grains)


def test_parse_grains_csv_missing_om_column_raises(tmp_path: Path):
    grains = tmp_path / "Grains.csv"
    grains.write_text(
        "%GrainID x y z O11 O12 O13 O21 O22 O23 O31 O32 OWRONG\n"
        "1 0 0 0 1 0 0 0 1 0 0 0 1\n"
    )
    with pytest.raises(ValueError, match="O33"):
        _parse_grains_csv(grains)


def test_parse_grains_csv_skips_blank_and_comment_rows(tmp_path: Path):
    grains = tmp_path / "Grains.csv"
    grains.write_text(
        "%GrainID O11 O12 O13 O21 O22 O23 O31 O32 O33\n"
        "1 1 0 0 0 1 0 0 0 1\n"
        "\n"
        "% comment\n"
        "2 -1 0 0 0 -1 0 0 0 1\n"
    )
    oms, ids = _parse_grains_csv(grains)
    assert oms.shape == (2, 9)
    assert ids == [1, 2]


# ---------------------------------------------------------------------------
# UniqueOrientations.csv writer
# ---------------------------------------------------------------------------


def test_grains_csv_to_unique_orientations_layout(tmp_path: Path):
    om = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64).ravel()
    _make_grains_csv(tmp_path / "Grains.csv",
                     np.stack([om]), grain_ids=[42])
    out = tmp_path / "UniqueOrientations.csv"
    n = grains_csv_to_unique_orientations(
        tmp_path / "Grains.csv", out,
    )
    assert n == 1
    arr = np.loadtxt(out)
    # 14 cols: 5 key + 9 OM.
    assert arr.shape == (14,) or arr.shape == (1, 14)
    arr = np.atleast_2d(arr)
    assert arr[0, 0] == 42.0                  # grainID
    np.testing.assert_array_equal(arr[0, 1:5], [0, 0, 0, 0])  # key padding
    np.testing.assert_allclose(arr[0, 5:14], om)


def test_grains_csv_to_unique_orientations_empty_input(tmp_path: Path):
    """No grains → empty file (don't crash)."""
    grains = tmp_path / "Grains.csv"
    grains.write_text("%GrainID O11 O12 O13 O21 O22 O23 O31 O32 O33\n")
    out = tmp_path / "UniqueOrientations.csv"
    n = grains_csv_to_unique_orientations(grains, out)
    assert n == 0
    assert out.exists()


def test_grains_csv_to_unique_orientations_writes_n_rows(tmp_path: Path):
    """Three distinct OMs → three rows."""
    oms = np.stack([
        np.eye(3).ravel(),
        np.diag([-1, -1, 1]).astype(np.float64).ravel(),
        np.diag([-1, 1, -1]).astype(np.float64).ravel(),
    ])
    _make_grains_csv(tmp_path / "Grains.csv", oms, grain_ids=[1, 2, 3])
    out = tmp_path / "UniqueOrientations.csv"
    n = grains_csv_to_unique_orientations(tmp_path / "Grains.csv", out)
    assert n == 3
    arr = np.loadtxt(out)
    assert arr.shape == (3, 14)


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_dedup_collapses_symmetry_equivalent_oms(tmp_path: Path):
    """For cubic SG, the identity and a 90° rotation about z are
    symmetry-equivalent. With dedup, both rows should collapse to one.
    """
    eye = np.eye(3).ravel()
    rot_z_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float64).ravel()
    _make_grains_csv(tmp_path / "Grains.csv",
                     np.stack([eye, rot_z_90]), grain_ids=[1, 2])
    out = tmp_path / "UniqueOrientations.csv"
    n_no_dedup = grains_csv_to_unique_orientations(
        tmp_path / "Grains.csv", out,
        space_group=225,
        dedup_misorientation_deg=0.0,                # no dedup
    )
    assert n_no_dedup == 2
    n_dedup = grains_csv_to_unique_orientations(
        tmp_path / "Grains.csv", out,
        space_group=225,
        dedup_misorientation_deg=1.0,                # 1° threshold
    )
    assert n_dedup == 1                              # collapsed
