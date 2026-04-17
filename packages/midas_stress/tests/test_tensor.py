"""Tests for tensor.py — Voigt conversions, A-matrix, strain, frame transforms, 6x6 rotation."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_stress.tensor import (
    tensor_to_voigt,
    voigt_to_tensor,
    lattice_params_to_A_matrix,
    lattice_params_to_strain,
    strain_grain_to_lab,
    strain_lab_to_grain,
    rotation_voigt_mandel,
)


# ===================================================================
#  Voigt notation
# ===================================================================

class TestVoigt:
    def test_roundtrip(self):
        T = np.array([[1, 4, 5], [4, 2, 6], [5, 6, 3]], dtype=float)
        v = tensor_to_voigt(T)
        T2 = voigt_to_tensor(v)
        np.testing.assert_allclose(T, T2, atol=1e-15)

    def test_identity(self):
        I = np.eye(3)
        v = tensor_to_voigt(I)
        np.testing.assert_allclose(v, [1, 1, 1, 0, 0, 0])

    def test_shear_scaling(self):
        T = np.zeros((3, 3))
        T[0, 1] = T[1, 0] = 0.5
        v = tensor_to_voigt(T)
        assert abs(v[3] - math.sqrt(2) * 0.5) < 1e-15

    def test_frobenius_norm_preserved(self):
        T = np.array([[1, 0.3, 0.1], [0.3, 2, 0.2], [0.1, 0.2, 3]], dtype=float)
        v = tensor_to_voigt(T)
        np.testing.assert_allclose(
            np.linalg.norm(T, 'fro'), np.linalg.norm(v), atol=1e-14
        )

    def test_batch(self):
        T = np.random.randn(5, 3, 3)
        T = 0.5 * (T + np.swapaxes(T, -1, -2))
        v = tensor_to_voigt(T)
        assert v.shape == (5, 6)
        T2 = voigt_to_tensor(v)
        np.testing.assert_allclose(T, T2, atol=1e-14)


# ===================================================================
#  A-matrix and strain from lattice params
# ===================================================================

class TestAMatrix:
    def test_cubic(self):
        latc = np.array([4.08, 4.08, 4.08, 90, 90, 90], dtype=float)
        A = lattice_params_to_A_matrix(latc)
        assert abs(A[0, 0] - 4.08) < 1e-10
        assert abs(A[2, 2] - 4.08) < 1e-10

    def test_zero_strain(self):
        latc = np.array([4.08, 4.08, 4.08, 90, 90, 90], dtype=float)
        eps = lattice_params_to_strain(latc, latc)
        np.testing.assert_allclose(eps, np.zeros((3, 3)), atol=1e-14)

    def test_tensile_strain(self):
        latc0 = np.array([4.08, 4.08, 4.08, 90, 90, 90], dtype=float)
        latc1 = np.array([4.09, 4.08, 4.08, 90, 90, 90], dtype=float)
        eps = lattice_params_to_strain(latc1, latc0)
        assert eps[0, 0] > 0
        assert abs(eps[0, 1]) < 1e-10

    def test_batch(self):
        latc0 = np.array([4.08, 4.08, 4.08, 90, 90, 90], dtype=float)
        latc1 = np.tile(latc0, (3, 1))
        latc1[0, 0] += 0.01
        latc1[1, 1] += 0.01
        eps = lattice_params_to_strain(latc1, latc0)
        assert eps.shape == (3, 3, 3)


# ===================================================================
#  Coordinate frame transformations
# ===================================================================

class TestFrameTransform:
    def test_identity_orientation(self):
        eps = np.array([[0.001, 0, 0], [0, -0.0005, 0], [0, 0, -0.0005]])
        orient = np.eye(3)
        eps_lab = strain_grain_to_lab(eps, orient)
        np.testing.assert_allclose(eps_lab, eps, atol=1e-15)

    def test_roundtrip(self):
        eps = np.array([[0.001, 0.0002, 0], [0.0002, -0.0005, 0.0001],
                        [0, 0.0001, -0.0005]])
        orient = Rotation.random(random_state=42).as_matrix()
        eps_lab = strain_grain_to_lab(eps, orient)
        eps_back = strain_lab_to_grain(eps_lab, orient)
        np.testing.assert_allclose(eps_back, eps, atol=1e-14)

    def test_trace_preserved(self):
        eps = np.array([[0.003, 0.001, 0], [0.001, -0.001, 0.0005],
                        [0, 0.0005, -0.002]])
        orient = Rotation.random(random_state=99).as_matrix()
        eps_lab = strain_grain_to_lab(eps, orient)
        np.testing.assert_allclose(np.trace(eps), np.trace(eps_lab), atol=1e-14)


# ===================================================================
#  6x6 Voigt rotation matrix
# ===================================================================

class TestVoigtRotation:
    def test_identity(self):
        M = rotation_voigt_mandel(np.eye(3))
        np.testing.assert_allclose(M, np.eye(6), atol=1e-14)

    def test_orthogonal(self):
        orient = Rotation.random(random_state=42).as_matrix()
        M = rotation_voigt_mandel(orient)
        np.testing.assert_allclose(M @ M.T, np.eye(6), atol=1e-13)

    def test_det_one(self):
        orient = Rotation.random(random_state=42).as_matrix()
        M = rotation_voigt_mandel(orient)
        assert abs(np.linalg.det(M) - 1.0) < 1e-12

    def test_consistent_with_tensor_transform(self):
        """M^T @ eps_voigt should equal voigt(U @ eps @ U^T) (grain->lab)."""
        orient = Rotation.random(random_state=7).as_matrix()
        eps = np.array([[0.001, 0.0003, -0.0001],
                        [0.0003, -0.0005, 0.0002],
                        [-0.0001, 0.0002, -0.0005]])
        # U @ eps @ U^T is grain->lab
        eps_rotated = orient @ eps @ orient.T
        v_expected = tensor_to_voigt(eps_rotated)
        # rotation_voigt_mandel returns lab->grain; transpose gives grain->lab
        M = rotation_voigt_mandel(orient)
        v_actual = M.T @ tensor_to_voigt(eps)
        np.testing.assert_allclose(v_actual, v_expected, atol=1e-14)

    def test_batch(self):
        orients = Rotation.random(5, random_state=42).as_matrix()
        M = rotation_voigt_mandel(orients)
        assert M.shape == (5, 6, 6)
