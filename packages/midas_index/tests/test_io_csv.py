"""Tests for CSV / text readers (hkls.csv, SpotsToIndex.csv, Grains.csv)."""

import textwrap

import numpy as np
import pytest

from midas_index.io import (
    read_grains_csv,
    read_hkls_csv,
    read_spots_to_index_csv,
    write_spots_to_index_csv,
)


# ---------------------------------------------------------------------- hkls

_HKLS_BODY = """\
h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius
1 -1 -1 3.124 1 0.184 -0.184 -0.184 1.585 3.171 36445.0
1 1 1 3.124 1 0.184 0.184 0.184 1.585 3.171 36445.0
2 0 0 2.706 2 0.213 0.0 0.0 1.832 3.665 42100.0
2 0 0 2.706 2 0.213 0.0 0.0 1.832 3.665 42100.0
1 1 0 3.605 99 0.150 0.150 0.0 1.370 2.741 30000.0
"""


def test_read_hkls_keep_all(tmp_path):
    p = tmp_path / "hkls.csv"
    p.write_text(_HKLS_BODY)
    real, ints = read_hkls_csv(p)
    assert real.shape == (5, 7)
    assert ints.shape == (5, 4)
    # Layout check: cols [g1, g2, g3, ring_nr, d_spacing, theta, radius]
    assert real[0, 3] == 1.0   # ring_nr
    assert real[0, 4] == pytest.approx(3.124, rel=1e-6)
    assert ints[0, 3] == 1     # ring_nr in int form too


def test_read_hkls_filter_by_ring(tmp_path):
    p = tmp_path / "hkls.csv"
    p.write_text(_HKLS_BODY)
    real, ints = read_hkls_csv(p, ring_numbers=[1, 2])
    assert real.shape == (4, 7)        # 5th row (ring 99) dropped
    assert ints.shape == (4, 4)
    assert set(np.unique(ints[:, 3].tolist())) == {1, 2}


def test_read_hkls_empty_ring_filter_returns_empty(tmp_path):
    p = tmp_path / "hkls.csv"
    p.write_text(_HKLS_BODY)
    real, ints = read_hkls_csv(p, ring_numbers=[])
    assert real.shape == (0, 7)
    assert ints.shape == (0, 4)


# ---------------------------------------------------------------------- spots-to-index


def test_read_spots_to_index_one_int_per_line(tmp_path):
    p = tmp_path / "SpotsToIndex.csv"
    p.write_text("17\n42\n8\n")
    ids = read_spots_to_index_csv(p)
    assert ids.tolist() == [17, 42, 8]


def test_read_spots_to_index_two_int_per_line_first_only(tmp_path):
    # Mode A writes "newID origID" — IndexerOMP.c:2313 reads only first %d
    p = tmp_path / "SpotsToIndex.csv"
    p.write_text("100 1\n200 2\n300 3\n")
    ids = read_spots_to_index_csv(p)
    assert ids.tolist() == [100, 200, 300]


def test_write_spots_to_index_roundtrip_pairs(tmp_path):
    p = tmp_path / "SpotsToIndex.csv"
    write_spots_to_index_csv(p, [(100, 1), (200, 2)])
    ids = read_spots_to_index_csv(p)
    assert ids.tolist() == [100, 200]
    # Verify the on-disk format matches mode-A two-int layout
    text = p.read_text().splitlines()
    assert text == ["100 1", "200 2"]


def test_write_spots_to_index_roundtrip_singles(tmp_path):
    p = tmp_path / "SpotsToIndex.csv"
    write_spots_to_index_csv(p, [10, 20, 30])
    ids = read_spots_to_index_csv(p)
    assert ids.tolist() == [10, 20, 30]


# ---------------------------------------------------------------------- grains


_GRAINS_BODY = """\
%NumGrains 2
%BeamCenter 0.000000
%BeamThickness 200.000000
%GlobalPosition 0.000000
%NumPhases 1
%PhaseInfo
%	SpaceGroup:225
%	Lattice Parameter: 4.080000 4.080000 4.080000 90.000000 90.000000 90.000000
%GrainID	O11	O12	O13	O21	O22	O23	O31	O32	O33	X	Y	Z	a	b	c	alpha	beta	gamma	DiffPos	DiffOme	DiffAngle	GrainRadius
1\t1.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0\t10.0\t20.0\t30.0\tx\ty\tz\tw\tv\tu\tt\ts\tr\t50.0
2\t0.0\t-1.0\t0.0\t1.0\t0.0\t0.0\t0.0\t0.0\t1.0\t-15.0\t5.0\t-2.5\tx\ty\tz\tw\tv\tu\tt\ts\tr\t75.0
"""


def test_read_grains_csv(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text(_GRAINS_BODY)
    g = read_grains_csv(p)

    assert g["ids"].tolist() == [1, 2]
    np.testing.assert_array_equal(
        g["orient_mat"][0],
        np.eye(3),
    )
    np.testing.assert_array_equal(
        g["orient_mat"][1],
        np.array([[0.0, -1.0, 0.0],
                  [1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]]),
    )
    np.testing.assert_array_equal(g["positions"][0], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(g["positions"][1], [-15.0, 5.0, -2.5])
    assert g["radii"].tolist() == [50.0, 75.0]


def test_read_grains_csv_bad_header(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text("not a numgrains line\n")
    with pytest.raises(ValueError, match="NumGrains"):
        read_grains_csv(p)


# --------------------------------------------------- grains: current formats
#
# The fixture above is a 23-column %GrainID file. Grains.csv is 53 columns
# today and is written under BOTH %GrainID (c_parity_emit) and %ID (io/csv),
# and this reader had a fixed eight-line preamble skip plus a hard-coded
# tokens[22] for GrainRadius. Both happen to be right at 53 columns, but
# nothing here proved it.

_GRAINS_53_NAMES = (
    [f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
       "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
    + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2",
       "DiffPosPre", "DiffOmePre", "DiffAnglePre",
       "DiffPosPost", "DiffOmePost", "DiffAnglePost"]
)

_PREAMBLE_9 = (
    "%NumGrains {n}\n"
    "%BeamCenter 0.000000\n"
    "%BeamThickness 200.000000\n"
    "%GlobalPosition 0.000000\n"
    "%NumPhases 1\n"
    "%PhaseInfo\n"
    "%\tSpaceGroup:225\n"
    "%\tLattice Parameter: 4.080000 4.080000 4.080000 90.000000 90.000000 90.000000\n"
)


def _grains_53(id_token="GrainID", *, trailing_tab=False, extra_preamble="",
               grains=((7, 11.0, 22.0, 33.0, 61.5), (9, -4.0, 5.0, -6.0, 88.25))):
    """A current-format 53-column Grains.csv under either ID spelling."""
    head = "%" + "\t".join([id_token] + _GRAINS_53_NAMES) + "\n"
    body = ""
    for gid, x, y, z, radius in grains:
        row = [str(gid)]
        row += [f"{v:.6f}" for v in (1, 0, 0, 0, 1, 0, 0, 0, 1)]   # O11..O33
        row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
        row += [f"{v:.6f}" for v in (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)]
        row += ["0.500000", "0.100000", "0.200000",
                f"{radius:.6f}", "0.970000"]
        row += ["0.000000"] * 9 + ["0.000000"] * 9          # eFab, eKen
        row += ["12.500000", "1", "0.100000", "0.200000", "0.300000"]
        row += ["0.600000"] * 6                              # Diff*Pre/Post
        assert len(row) == 53, len(row)
        body += "\t".join(row) + ("\t\n" if trailing_tab else "\n")
    return (_PREAMBLE_9.format(n=len(grains)) + extra_preamble + head + body)


@pytest.mark.parametrize("id_token", ["GrainID", "ID"])
@pytest.mark.parametrize("trailing_tab", [False, True])
def test_read_grains_csv_53col_both_id_spellings(tmp_path, id_token,
                                                 trailing_tab):
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53(id_token, trailing_tab=trailing_tab))
    g = read_grains_csv(p)
    assert g["ids"].tolist() == [7, 9]
    np.testing.assert_allclose(g["orient_mat"][0], np.eye(3))
    np.testing.assert_allclose(g["positions"][0], [11.0, 22.0, 33.0])
    np.testing.assert_allclose(g["positions"][1], [-4.0, 5.0, -6.0])
    # GrainRadius, NOT Confidence and NOT the first eFab component.
    np.testing.assert_allclose(g["radii"], [61.5, 88.25])


def test_read_grains_csv_tolerates_a_longer_preamble(tmp_path):
    """The '%' block is nine lines today but is written per-phase.

    A second phase adds a SpaceGroup + Lattice Parameter pair; the old fixed
    eight-line skip would then have fed the column-header line to int() as a
    data row and, past that, mis-set the grain count.
    """
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53(extra_preamble="%\tSpaceGroup:229\n"
                                           "%\tLattice Parameter: 2.87 2.87 2.87 90 90 90\n"))
    g = read_grains_csv(p)
    assert g["ids"].tolist() == [7, 9]
    np.testing.assert_allclose(g["radii"], [61.5, 88.25])


def test_read_grains_csv_finds_radius_when_it_is_not_column_22(tmp_path):
    """21-column legacy files put GrainRadius at 19, not 22.

    tokens[22] raised IndexError (swallowed, radius 0) and the flat
    ``len(tokens) < 23`` guard skipped every row, so the reader returned zero
    grains from a perfectly readable file -- silently.
    """
    cols = (["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
             "X", "Y", "Z", "E11", "E22", "E33", "E12", "E13", "E23",
             "GrainRadius", "Confidence"])
    row = (["3"] + [f"{v}" for v in (1, 0, 0, 0, 1, 0, 0, 0, 1)]
           + ["1.0", "2.0", "3.0"] + ["1e-4"] * 6 + ["44.0", "0.9"])
    p = tmp_path / "Grains.csv"
    p.write_text(_PREAMBLE_9.format(n=1)
                 + "%" + "\t".join(["ID"] + cols) + "\n"
                 + "\t".join(row) + "\n")
    g = read_grains_csv(p)
    assert g["ids"].tolist() == [3]
    np.testing.assert_allclose(g["positions"][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(g["radii"], [44.0])


def test_read_grains_csv_refuses_a_reordered_orientation_block(tmp_path):
    """C reads this same file with a fixed sscanf and cannot follow a header.

    A moved OM/position block must therefore raise here rather than let the
    two halves of mode A read different numbers out of one file.
    """
    names = ["X", "Y", "Z"] + [f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    p = tmp_path / "Grains.csv"
    p.write_text(_PREAMBLE_9.format(n=1)
                 + "%" + "\t".join(["GrainID"] + names) + "\n"
                 + "\t".join(["1"] + ["0"] * 12) + "\n")
    with pytest.raises(ValueError, match="column order has changed"):
        read_grains_csv(p)


def test_read_grains_csv_without_a_named_header_stays_positional(tmp_path):
    """Old NF Mic2GrainsList wrote a prose header naming no columns.

    There is nothing to resolve by name, so the reader must fall back to the
    positional layout C uses -- exactly what it did before.
    """
    p = tmp_path / "Grains.csv"
    p.write_text(
        "%NumGrains 1\n"
        "%GrainID OrientMat(9) X Y Z LatC(6) 0 0 0 Radius Confidence\n"
        "%a\n%b\n%c\n%d\n%e\n%f\n%g\n"
        "5 1 0 0 0 1 0 0 0 1 7 8 9 "
        "4.08 4.08 4.08 90 90 90 0 0 0 33.0 1\n")
    g = read_grains_csv(p)
    assert g["ids"].tolist() == [5]
    np.testing.assert_allclose(g["positions"][0], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(g["radii"], [33.0])
