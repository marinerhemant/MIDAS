"""Tests for frames.py — MIDAS/APS/sample frame conversions."""

import math

import numpy as np
import pytest

from midas_stress.frames import (
    R_MIDAS_TO_APS,
    R_APS_TO_MIDAS,
    lab_to_sample_rotation,
    vector_midas_to_aps,
    vector_aps_to_midas,
    orient_midas_to_aps,
    orient_aps_to_midas,
    tensor_midas_to_aps,
    tensor_aps_to_midas,
    tensor_lab_to_sample,
    grains_midas_to_sample,
)


class TestRotationMatrices:
    def test_r_midas_to_aps_is_orthogonal(self):
        np.testing.assert_allclose(R_MIDAS_TO_APS @ R_MIDAS_TO_APS.T, np.eye(3), atol=1e-15)

    def test_r_midas_to_aps_det_one(self):
        assert abs(np.linalg.det(R_MIDAS_TO_APS) - 1.0) < 1e-15

    def test_roundtrip(self):
        np.testing.assert_allclose(R_MIDAS_TO_APS @ R_APS_TO_MIDAS, np.eye(3), atol=1e-15)

    def test_axis_mapping(self):
        """MIDAS X (beam) -> APS Z (beam)."""
        x_midas = np.array([1, 0, 0])
        x_aps = R_MIDAS_TO_APS @ x_midas
        np.testing.assert_allclose(x_aps, [0, 0, 1], atol=1e-15)

        # MIDAS Y (OB) -> APS X (OB)
        y_midas = np.array([0, 1, 0])
        y_aps = R_MIDAS_TO_APS @ y_midas
        np.testing.assert_allclose(y_aps, [1, 0, 0], atol=1e-15)

        # MIDAS Z (up) -> APS Y (up)
        z_midas = np.array([0, 0, 1])
        z_aps = R_MIDAS_TO_APS @ z_midas
        np.testing.assert_allclose(z_aps, [0, 1, 0], atol=1e-15)


class TestLabToSample:
    def test_zero_omega_is_identity(self):
        for frame in ("midas", "aps"):
            R = lab_to_sample_rotation(0.0, frame)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_orthogonality(self):
        for frame in ("midas", "aps"):
            R = lab_to_sample_rotation(37.5, frame)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_aps_rotates_about_y(self):
        """APS lab-to-sample is a rotation about Y (up)."""
        R = lab_to_sample_rotation(90.0, "aps")
        # 90-deg rotation about Y: X->-Z, Z->X, Y unchanged
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 0, 1], atol=1e-14)
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 1, 0], atol=1e-14)

    def test_midas_rotates_about_z(self):
        """MIDAS lab-to-sample is a rotation about Z (up)."""
        R = lab_to_sample_rotation(90.0, "midas")
        # 90-deg rotation about Z: X->-Y, Y->X, Z unchanged
        np.testing.assert_allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-14)
        np.testing.assert_allclose(R @ [0, 0, 1], [0, 0, 1], atol=1e-14)


class TestVectorConversions:
    def test_roundtrip(self):
        v = np.array([1.5, -2.3, 0.7])
        v2 = vector_aps_to_midas(vector_midas_to_aps(v))
        np.testing.assert_allclose(v2, v, atol=1e-15)

    def test_batch(self):
        v = np.random.randn(10, 3)
        v_aps = vector_midas_to_aps(v)
        assert v_aps.shape == (10, 3)
        v_back = vector_aps_to_midas(v_aps)
        np.testing.assert_allclose(v_back, v, atol=1e-14)


class TestOrientConversions:
    def test_roundtrip(self):
        from scipy.spatial.transform import Rotation
        U = Rotation.random(random_state=42).as_matrix()
        U2 = orient_aps_to_midas(orient_midas_to_aps(U))
        np.testing.assert_allclose(U2, U, atol=1e-14)

    def test_batch(self):
        from scipy.spatial.transform import Rotation
        U = Rotation.random(5, random_state=42).as_matrix()
        U_aps = orient_midas_to_aps(U)
        assert U_aps.shape == (5, 3, 3)
        U_back = orient_aps_to_midas(U_aps)
        np.testing.assert_allclose(U_back, U, atol=1e-14)


class TestTensorConversions:
    def test_roundtrip(self):
        T = np.array([[1, 0.5, 0.1], [0.5, 2, 0.2], [0.1, 0.2, 3]])
        T2 = tensor_aps_to_midas(tensor_midas_to_aps(T))
        np.testing.assert_allclose(T2, T, atol=1e-14)

    def test_trace_preserved(self):
        T = np.array([[100, 10, 5], [10, -50, 3], [5, 3, -50]])
        T_aps = tensor_midas_to_aps(T)
        np.testing.assert_allclose(np.trace(T), np.trace(T_aps), atol=1e-12)

    def test_eigenvalues_preserved(self):
        """Frame change preserves eigenvalues (physical invariants)."""
        T = np.array([[100, 10, 5], [10, -50, 3], [5, 3, -50]])
        T_aps = tensor_midas_to_aps(T)
        eig_midas = np.sort(np.linalg.eigvalsh(T))
        eig_aps = np.sort(np.linalg.eigvalsh(T_aps))
        np.testing.assert_allclose(eig_midas, eig_aps, atol=1e-12)

    def test_batch(self):
        T = np.random.randn(5, 3, 3)
        T = 0.5 * (T + np.swapaxes(T, -1, -2))
        T_aps = tensor_midas_to_aps(T)
        assert T_aps.shape == (5, 3, 3)
        T_back = tensor_aps_to_midas(T_aps)
        np.testing.assert_allclose(T_back, T, atol=1e-14)

    def test_lab_to_sample_zero_omega(self):
        T = np.array([[100, 10, 5], [10, -50, 3], [5, 3, -50]])
        T_sam = tensor_lab_to_sample(T, 0.0, "midas")
        np.testing.assert_allclose(T_sam, T, atol=1e-14)


class TestGrainsPipeline:
    def test_identity_at_zero_omega_midas(self):
        """With target_frame='midas' and omega=0, everything is identity."""
        N = 3
        from scipy.spatial.transform import Rotation
        U = Rotation.random(N, random_state=42).as_matrix()
        pos = np.random.randn(N, 3) * 100
        eps = np.random.randn(N, 3, 3) * 0.001
        eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))

        out = grains_midas_to_sample(U, pos, eps, omega_deg=0.0,
                                      target_frame="midas")
        np.testing.assert_allclose(out['orientations'], U, atol=1e-14)
        np.testing.assert_allclose(out['positions'], pos, atol=1e-14)
        np.testing.assert_allclose(out['strains'], eps, atol=1e-14)

    def test_trace_preserved_through_pipeline(self):
        N = 5
        from scipy.spatial.transform import Rotation
        U = Rotation.random(N, random_state=7).as_matrix()
        pos = np.random.randn(N, 3) * 100
        eps = np.random.randn(N, 3, 3) * 0.001
        eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))

        out = grains_midas_to_sample(U, pos, eps, omega_deg=45.0,
                                      target_frame="aps")
        traces_in = np.trace(eps, axis1=-2, axis2=-1)
        traces_out = np.trace(out['strains'], axis1=-2, axis2=-1)
        np.testing.assert_allclose(traces_in, traces_out, atol=1e-13)
