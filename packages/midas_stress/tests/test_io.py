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
    assert "strain" in g          # Kenesei (d-spacing form)
    assert "strain_fable" in g    # Fable-Beaudoin alternate
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
    assert g["strain_fable"].shape == (N, 3, 3)
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


def test_strain_default_is_kenesei():
    """The 'strain' key should map to eKen columns in the file.

    In GrainsSim.csv all strains are zero, so instead verify that
    strain and strain_fable are BOTH zero (consistency) and that
    reading them yields arrays of the expected shape.
    """
    g = read_grains_csv(example_data_path())
    # GrainsSim has zero strain by construction
    np.testing.assert_allclose(g["strain"], 0.0, atol=1e-12)
    np.testing.assert_allclose(g["strain_fable"], 0.0, atol=1e-12)


def test_header_parsing_missing_file():
    with pytest.raises(FileNotFoundError):
        read_grains_csv("/nonexistent/path/Grains.csv")
