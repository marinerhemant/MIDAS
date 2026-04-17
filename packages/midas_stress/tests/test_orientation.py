"""Tests for orientation.py — Euler angles, quaternions, misorientation."""

import math

import numpy as np
import pytest

from midas_stress.orientation import (
    euler_to_orient_mat,
    orient_mat_to_quat,
    orient_mat_to_euler,
    quaternion_product,
    quat_to_orient_mat,
    euler_to_orient_mat_batch,
    make_symmetries,
    misorientation,
    misorientation_om,
    fundamental_zone,
    axis_angle_to_orient_mat,
    rodrigues_to_orient_mat,
)


class TestEulerOrientMat:
    def test_identity(self):
        om = euler_to_orient_mat([0, 0, 0])
        np.testing.assert_allclose(np.array(om).reshape(3, 3), np.eye(3), atol=1e-14)

    def test_roundtrip(self):
        euler = [0.5, 1.0, 1.5]
        om = euler_to_orient_mat(euler)
        euler_back = orient_mat_to_euler(om)
        om2 = euler_to_orient_mat(euler_back)
        np.testing.assert_allclose(om, om2, atol=1e-12)

    def test_orthogonality(self):
        euler = [0.3, 0.7, 2.1]
        om = np.array(euler_to_orient_mat(euler)).reshape(3, 3)
        np.testing.assert_allclose(om @ om.T, np.eye(3), atol=1e-13)
        np.testing.assert_allclose(np.linalg.det(om), 1.0, atol=1e-13)


class TestQuaternion:
    def test_identity(self):
        om = list(np.eye(3).ravel())
        q = orient_mat_to_quat(om)
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-14)

    def test_quat_to_om_roundtrip(self):
        euler = [0.8, 1.2, 0.4]
        om = euler_to_orient_mat(euler)
        q = orient_mat_to_quat(om)
        om2 = quat_to_orient_mat(q)
        np.testing.assert_allclose(om, om2, atol=1e-12)

    def test_product_associativity(self):
        q1 = np.array([0.5, 0.5, 0.5, 0.5])
        q2 = np.array([0.70711, 0.70711, 0, 0])
        q3 = np.array([0.70711, 0, 0.70711, 0])
        lhs = quaternion_product(quaternion_product(q1, q2), q3)
        rhs = quaternion_product(q1, quaternion_product(q2, q3))
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)


class TestBatchEuler:
    def test_matches_single(self):
        eulers = np.array([
            [0.1, 0.5, 1.0],
            [1.0, 0.3, 2.0],
            [0.0, 0.0, 0.0],
        ])
        batch = euler_to_orient_mat_batch(eulers)
        for i in range(len(eulers)):
            single = euler_to_orient_mat(eulers[i])
            np.testing.assert_allclose(batch[i], single, atol=1e-12)


class TestSymmetries:
    def test_cubic(self):
        n, sym = make_symmetries(225)  # Fm-3m (FCC)
        assert n == 24

    def test_triclinic(self):
        n, sym = make_symmetries(1)
        assert n == 1

    def test_hexagonal(self):
        n, sym = make_symmetries(194)  # P6_3/mmc (HCP)
        assert n == 12


class TestMisorientation:
    def test_zero_misorientation(self):
        euler = [0.5, 1.0, 1.5]
        ang, axis = misorientation(euler, euler, 225)
        assert abs(ang) < 1e-10

    def test_known_cubic_rotation(self):
        # 90-degree rotation about [001] is a symmetry operation for cubic
        euler1 = [0, 0, 0]
        om1 = euler_to_orient_mat(euler1)
        # Rotate by 90 deg about z
        om2_mat = axis_angle_to_orient_mat([0, 0, 1], 90.0)
        om2 = list(om2_mat.ravel())
        ang, _ = misorientation_om(om1, om2, 225)
        # Should be 0 because 90-deg z rotation is a cubic symmetry
        assert ang < 1e-6

    def test_symmetry_of_arguments(self):
        euler1 = [0.3, 0.8, 1.2]
        euler2 = [1.0, 0.5, 2.0]
        ang1, _ = misorientation(euler1, euler2, 225)
        ang2, _ = misorientation(euler2, euler1, 225)
        np.testing.assert_allclose(ang1, ang2, atol=1e-10)


class TestFundamentalZone:
    def test_idempotent(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q1 = fundamental_zone(q, 225)
        q2 = fundamental_zone(q1, 225)
        np.testing.assert_allclose(q1, q2, atol=1e-12)


class TestAxisAngle:
    def test_identity(self):
        R = axis_angle_to_orient_mat([0, 0, 1], 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_90_deg_z(self):
        R = axis_angle_to_orient_mat([0, 0, 1], 90.0)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_orthogonality(self):
        R = axis_angle_to_orient_mat([1, 1, 1], 60.0)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-13)


class TestRodrigues:
    def test_zero(self):
        R = rodrigues_to_orient_mat([0, 0, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_orthogonality(self):
        R = rodrigues_to_orient_mat([0.1, 0.2, 0.3])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
