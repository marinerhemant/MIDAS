"""Tests for hooke.py — Hooke's law stress computation."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_stress.hooke import hooke_stress
from midas_stress.materials import cubic_stiffness, get_stiffness, STIFFNESS_LIBRARY


class TestHooke:
    def test_zero_strain_zero_stress(self):
        C = get_stiffness("Au")
        eps = np.zeros((3, 3))
        sig = hooke_stress(eps, C, frame="grain")
        np.testing.assert_allclose(sig, np.zeros((3, 3)), atol=1e-15)

    def test_hydrostatic_cubic(self):
        C11, C12, C44 = 192.9, 163.8, 41.5  # Au
        C = cubic_stiffness(C11, C12, C44)
        eps_val = 0.001
        eps = eps_val * np.eye(3)
        sig = hooke_stress(eps, C, frame="grain")
        expected_hydro = (C11 + 2 * C12) * eps_val
        np.testing.assert_allclose(np.diag(sig), expected_hydro, rtol=1e-12)
        np.testing.assert_allclose(sig - np.diag(np.diag(sig)), 0, atol=1e-12)

    def test_lab_frame_with_identity(self):
        C = get_stiffness("Cu")
        eps = np.diag([0.001, -0.0005, -0.0005])
        sig_grain = hooke_stress(eps, C, frame="grain")
        sig_lab = hooke_stress(eps, C, orient=np.eye(3), frame="lab")
        np.testing.assert_allclose(sig_lab, sig_grain, atol=1e-12)

    def test_stress_trace_preserved(self):
        C = get_stiffness("Fe")
        eps_grain = np.diag([0.001, -0.0003, -0.0007])
        sig_grain = hooke_stress(eps_grain, C, frame="grain")
        orient = Rotation.random(random_state=42).as_matrix()
        eps_lab = orient @ eps_grain @ orient.T
        sig_lab = hooke_stress(eps_lab, C, orient=orient, frame="lab")
        np.testing.assert_allclose(np.trace(sig_grain), np.trace(sig_lab), atol=1e-10)

    def test_material_library(self):
        for mat in STIFFNESS_LIBRARY:
            C = get_stiffness(mat)
            assert C.shape == (6, 6)
            eigvals = np.linalg.eigvalsh(C)
            assert np.all(eigvals > 0), f"{mat} stiffness not positive definite"

    def test_orient_required_for_lab(self):
        C = get_stiffness("Cu")
        eps = np.diag([0.001, 0, 0])
        with pytest.raises(ValueError, match="orient required"):
            hooke_stress(eps, C, frame="lab")
