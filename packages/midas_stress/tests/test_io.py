"""Tests for io.py — header-driven Grains.csv parsing and bundled data."""

import os

import numpy as np
import pytest

import midas_stress as ms
from midas_stress.io import read_grains_csv, example_data_path


def test_example_data_path_exists():
    p = example_data_path()
    assert os.path.exists(p)
    assert p.endswith("GrainsSim.csv")


def test_read_grainssim_keys():
    g = read_grains_csv(example_data_path())
    assert "orientations" in g
    assert "positions" in g
    assert "lattice_params" in g
    assert "strain" in g          # d-spacing (strain-gauge) form
    assert "strain_lattice" in g  # lattice-parameter alternate
    assert "radii" in g
    assert "confidences" in g
    assert "euler_angles" in g


def test_read_grainssim_shapes():
    g = read_grains_csv(example_data_path())
    N = g["orientations"].shape[0]
    assert N == 250
    assert g["orientations"].shape == (N, 3, 3)
    assert g["positions"].shape == (N, 3)
    assert g["lattice_params"].shape == (N, 6)
    assert g["strain"].shape == (N, 3, 3)
    assert g["strain_lattice"].shape == (N, 3, 3)
    assert g["radii"].shape == (N,)
    assert g["confidences"].shape == (N,)


def test_read_grainssim_orientation_is_rotation():
    g = read_grains_csv(example_data_path())
    U = g["orientations"]
    # Determinant should be +/-1 (orthogonal); for MIDAS output |det|≈1
    dets = np.linalg.det(U)
    np.testing.assert_allclose(np.abs(dets), 1.0, atol=1e-4)


def test_read_grainssim_radius_positive():
    g = read_grains_csv(example_data_path())
    assert (g["radii"] > 0).all()


def test_strain_default_is_d_spacing():
    """The 'strain' key should map to the eKen (d-spacing) columns.

    In GrainsSim.csv all strains are zero, so instead verify that
    both strain forms are zero (consistency) and that reading them
    yields arrays of the expected shape. The primary 'strain' key
    is the d-spacing / strain-gauge form (historically eKen);
    'strain_lattice' is the lattice-parameter form (historically
    eFab).
    """
    g = read_grains_csv(example_data_path())
    # GrainsSim has zero strain by construction
    np.testing.assert_allclose(g["strain"], 0.0, atol=1e-12)
    np.testing.assert_allclose(g["strain_lattice"], 0.0, atol=1e-12)


def test_header_parsing_missing_file():
    with pytest.raises(FileNotFoundError):
        read_grains_csv("/nonexistent/path/Grains.csv")


def test_strain_rescaled_to_dimensionless(tmp_path):
    """Verify MIDAS microstrain convention is rescaled to dimensionless.

    MIDAS writes strain tensors as fractional_strain * 1e6 (microstrain).
    The reader should divide by 1e6 so values are ready to feed into
    Hooke's law. Construct a minimal CSV with a single grain whose
    stored eKen value is 1e3 microstrain; after read, it should be 1e-3.
    """
    import numpy as np
    csv_text = (
        "%NumGrains 1\n"
        "%BeamCenter 0\n"
        "%BeamThickness 200\n"
        "%GlobalPosition 0\n"
        "%NumPhases 1\n"
        "%PhaseInfo\n"
        "%\tSpaceGroup:225\n"
        "%\tLattice Parameter: 4.0 4.0 4.0 90.0 90.0 90.0\n"
        "%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t"
        "X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\t"
        "DiffPos\tDiffOme\tDiffAngle\tGrainRadius\tConfidence\t"
        "eFab11\teFab12\teFab13\teFab21\teFab22\teFab23\teFab31\teFab32\teFab33\t"
        "eKen11\teKen12\teKen13\teKen21\teKen22\teKen23\teKen31\teKen32\teKen33\t"
        "RMSErrorStrain\tPhaseNr\tEul0\tEul1\tEul2\n"
        # Data row: eKen* stored as 1000 (microstrain), eFab* as 500
        "1\t1\t0\t0\t0\t1\t0\t0\t0\t1\t"      # orientation I3
        "0\t0\t0\t4.0\t4.0\t4.0\t90\t90\t90\t"
        "0\t0\t0\t100\t1.0\t"
        "500\t0\t0\t0\t500\t0\t0\t0\t500\t"
        "1000\t0\t0\t0\t1000\t0\t0\t0\t1000\t"
        "0\t1\t0\t0\t0\n"
    )
    p = tmp_path / "mini.csv"
    p.write_text(csv_text)
    g = read_grains_csv(str(p))
    # Strain stored as 1000 microstrain -> should be read as 1e-3
    assert np.isclose(g['strain'][0, 0, 0], 1e-3, atol=1e-12)
    assert np.isclose(g['strain'][0, 1, 1], 1e-3, atol=1e-12)
    assert np.isclose(g['strain_lattice'][0, 0, 0], 5e-4, atol=1e-12)


# ---------------------------------------------------------------------------
# Header-spelling regressions: %ID vs %GrainID, and trailing tabs.
#
# midas_process_grains ships TWO writers with different ID spellings --
# compute/c_parity_emit.py writes "%GrainID..." and io/csv.py writes "%ID..."
# -- and the reader used to anchor on "%GrainID" alone, so every %ID file on
# disk (the majority of current output) raised
#     ValueError: Could not locate '%GrainID ...' column header line
# The bundled GrainsSim.csv is 47-col %GrainID, which is exactly why the whole
# %ID half of the corpus went untested.  Fixtures below cover both spellings,
# both widths, and the trailing-tab form the C writer emits.
# ---------------------------------------------------------------------------

_PREAMBLE = (
    "%NumGrains {n}\n"
    "%BeamCenter 0\n"
    "%BeamThickness 200\n"
    "%GlobalPosition 0\n"
    "%NumPhases 1\n"
    "%PhaseInfo\n"
    "%\tSpaceGroup:225\n"
    "%\tLattice Parameter: 4.0 4.0 4.0 90.0 90.0 90.0\n"
)

_COLS_53 = (
    ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
     "X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
     "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
    + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2",
       "DiffPosPre", "DiffOmePre", "DiffAnglePre",
       "DiffPosPost", "DiffOmePost", "DiffAnglePost"]
)

#: Values for the 52 non-ID columns of a single 53-column grain row.
_ROW_53 = (
    [1, 0, 0, 0, 1, 0, 0, 0, 1]                 # O11..O33 = I3
    + [10.0, 20.0, 30.0]                        # X Y Z (um)
    + [4.0, 4.0, 4.0, 90.0, 90.0, 90.0]         # a b c alpha beta gamma
    + [0.5, 0.1, 0.2, 100.0, 0.97]              # DiffPos/Ome/Angle Radius Conf
    + [500, 0, 0, 0, 500, 0, 0, 0, 500]         # eFab (microstrain)
    + [1000, 0, 0, 0, 1000, 0, 0, 0, 1000]      # eKen (microstrain)
    + [12.5, 1, 0.10, 0.20, 0.30]               # RMSErrorStrain PhaseNr Eul0-2
    + [0.6, 0.15, 0.25, 0.5, 0.1, 0.2]          # Diff*Pre / Diff*Post
)


def _grains_53(id_token="ID", *, trailing_tab=False, n=2):
    """A 53-column Grains.csv under either ID spelling.

    ``trailing_tab`` reproduces what the C ProcessGrains writer actually
    emits: every data row ends in a tab, giving a spurious empty final field
    to any reader that does a bare ``split('\\t')``.
    """
    head = "%" + "\t".join([id_token] + _COLS_53) + "\n"
    rows = ""
    for gid in range(1, n + 1):
        row = "\t".join([str(gid)] + [f"{v}" for v in _ROW_53])
        rows += row + ("\t\n" if trailing_tab else "\n")
    return _PREAMBLE.format(n=n) + head + rows


def _grains_21(id_token="ID", *, trailing_tab=False):
    """A 21-column legacy file: cols 13-18 are a Voigt STRAIN, not a lattice."""
    cols = (["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
             "X", "Y", "Z", "E11", "E22", "E33", "E12", "E13", "E23",
             "GrainRadius", "Confidence"])
    head = "%" + "\t".join([id_token] + cols) + "\n"
    vals = [1, 0, 0, 0, 1, 0, 0, 0, 1, 10.0, 20.0, 30.0,
            1e-4, 2e-4, 3e-4, 0.0, 0.0, 0.0, 100.0, 0.97]
    row = "\t".join(["7"] + [f"{v}" for v in vals])
    return (_PREAMBLE.format(n=1) + head + row
            + ("\t\n" if trailing_tab else "\n"))


@pytest.mark.parametrize("id_token", ["ID", "GrainID"])
@pytest.mark.parametrize("trailing_tab", [False, True])
def test_read_53col_both_id_spellings(tmp_path, id_token, trailing_tab):
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53(id_token, trailing_tab=trailing_tab))
    g = read_grains_csv(str(p))
    assert g["columns"][0] == id_token
    assert len(g["columns"]) == 53
    assert g["raw"].shape == (2, 53)
    np.testing.assert_array_equal(g["grain_ids"], [1, 2])
    np.testing.assert_allclose(g["positions"][0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(g["lattice_params"][0],
                               [4.0, 4.0, 4.0, 90.0, 90.0, 90.0])
    # microstrain -> dimensionless on read
    np.testing.assert_allclose(g["strain"][0, 0, 0], 1e-3)
    np.testing.assert_allclose(g["strain_lattice"][0, 0, 0], 5e-4)
    np.testing.assert_allclose(g["rms_error"][0], 12.5)
    np.testing.assert_array_equal(g["phase"], [1, 1])
    np.testing.assert_allclose(g["euler_angles"][0], [0.10, 0.20, 0.30])
    np.testing.assert_allclose(g["radii"], [100.0, 100.0])


@pytest.mark.parametrize("id_token", ["ID", "GrainID"])
@pytest.mark.parametrize("trailing_tab", [False, True])
def test_read_21col_legacy_has_no_lattice_or_strain(tmp_path, id_token,
                                                    trailing_tab):
    """Cols 13-18 of a 21-col file are E11..E23, NOT a b c alpha beta gamma.

    Because every block is resolved by NAME, the lattice and the eFab/eKen
    strains must simply be ABSENT -- never silently filled from the Voigt
    block sitting at the same positions.
    """
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_21(id_token, trailing_tab=trailing_tab))
    g = read_grains_csv(str(p))
    assert g["raw"].shape == (1, 21)
    np.testing.assert_array_equal(g["grain_ids"], [7])
    np.testing.assert_allclose(g["orientations"][0], np.eye(3))
    assert "lattice_params" not in g
    assert "strain" not in g
    assert "strain_lattice" not in g


def test_prose_header_is_rejected_loudly(tmp_path):
    """A prose header names no columns; guessing at it would be worse than
    raising. (Old NF Mic2GrainsList output looked like this.)"""
    p = tmp_path / "Grains.csv"
    p.write_text(
        "%NumGrains 1\n"
        "%GrainID OrientMat(9) X Y Z LatC(6) 0 0 0 Radius Confidence\n"
        "1 1 0 0 0 1 0 0 0 1 1 2 3 3.6 3.6 3.6 90 90 90 0 0 0 5 0.9\n")
    with pytest.raises(ValueError, match="O11"):
        read_grains_csv(str(p))


@pytest.mark.parametrize("path", [
    "/Users/hsharma/Desktop/analysis/demk/fcc_reanalysis/Grains_L1_local.csv",
    "/Users/hsharma/Desktop/analysis/au3_ff_sidecar/LayerNr_1/Grains.csv",
])
def test_real_percent_id_files_on_disk(path):
    """Smoke test against the real %ID files that used to crash this reader.

    Skipped where the analysis tree is not present (CI, other machines).
    """
    if not os.path.exists(path):
        pytest.skip(f"analysis file not present: {path}")
    g = read_grains_csv(path)
    assert g["columns"][0] in ("ID", "GrainID")
    assert g["raw"].shape[0] > 0
    assert g["orientations"].shape[1:] == (3, 3)


# ---------------------------------------------------------------------------
# Shipped-data contract: GrainsSim.csv Eul0/1/2 are RADIANS and describe the
# SAME orientation as that row's O11..O33.
#
# They used to be degrees, written by tests/generate_grains.py with a
# hand-rolled decomposition whose first and third angles were additionally
# swapped and offset by 180 deg against C's OrientMat2Euler. Nothing raised:
# every reader in the tree documents these columns as radians and simply
# believed them, and midas_plotting.read_grains reported a 1.97 rad OM/Euler
# disagreement on the file example_data_path() hands to users.
# ---------------------------------------------------------------------------

def test_grainssim_euler_is_radians():
    """Degrees would put values outside [0, 2pi] on a 250-grain sample."""
    g = read_grains_csv(example_data_path())
    e = g["euler_angles"]
    assert e.shape == (250, 3)
    assert e.min() >= 0.0
    assert e.max() <= 2 * np.pi
    # A degrees file spans ~+/-180 and would trip this immediately.
    assert e.max() > np.pi, "suspiciously narrow range for a random SO(3) set"


def test_grainssim_euler_matches_its_own_orientation_matrix():
    """Rebuilding the OM from Eul0/1/2 must return the stored O11..O33.

    The 1e-3 tolerance is the file's own precision floor, not slack: the OM
    columns are stored to 6 decimals, and MIDAS's acos-based decomposition is
    ill-conditioned when the third angle is near 0 or pi, so a 1e-6 rounding
    of the matrix can move an angle by ~2e-4 rad. The pre-fix file failed this
    by ~2, not by ~1e-3.
    """
    from midas_stress.orientation import euler_to_orient_mat

    g = read_grains_csv(example_data_path())
    e, U = g["euler_angles"], g["orientations"]
    rebuilt = np.array([np.asarray(euler_to_orient_mat(e[i])).reshape(3, 3)
                        for i in range(len(e))])
    np.testing.assert_allclose(rebuilt, U, atol=1e-3)


def test_grainssim_euler_round_trips_through_the_canonical_converter():
    """orient_mat_to_euler(OM) must reproduce the stored Eul0/1/2.

    This is the direction that catches a swap: Eul0 and Eul2 were transposed
    (with a 180 deg offset), which the OM-rebuild test above would also catch
    but this one localises to the column pair.
    """
    from midas_stress.orientation import orient_mat_to_euler

    g = read_grains_csv(example_data_path())
    e, U = g["euler_angles"], g["orientations"]
    derived = np.array([np.asarray(orient_mat_to_euler(U[i])).ravel()
                        for i in range(len(e))])
    np.testing.assert_allclose(derived, e, atol=1e-5)
