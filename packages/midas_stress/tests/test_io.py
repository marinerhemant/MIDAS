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
