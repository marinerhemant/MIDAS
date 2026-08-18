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
    matrix_mult_f33,
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

    def test_precomputed_sym_matches_default(self):
        """Caller-supplied sym table must give identical output to the default path."""
        q = np.array([0.1, 0.4, 0.5, 0.7], dtype=np.float64)
        q = q / np.linalg.norm(q)
        n_sym, sym = make_symmetries(225)
        sym_arr = np.asarray(sym, dtype=np.float64)
        q_default = fundamental_zone(q, 225)
        q_with_sym = fundamental_zone(q, sym=sym_arr)
        np.testing.assert_allclose(q_with_sym, q_default, atol=1e-14)

    def test_precomputed_sym_overrides_space_group(self):
        """If both `space_group` and `sym` are passed, `sym` wins (documented contract)."""
        q = np.array([0.3, 0.2, 0.6, 0.7], dtype=np.float64)
        q = q / np.linalg.norm(q)
        _, sym_cubic = make_symmetries(225)
        # Result with cubic sym should equal calling with space_group=225 alone.
        q_a = fundamental_zone(q, sym=np.asarray(sym_cubic, dtype=np.float64))
        # Sending a different space_group along with the cubic sym should not change the result.
        q_b = fundamental_zone(q, space_group=1, sym=np.asarray(sym_cubic, dtype=np.float64))
        np.testing.assert_allclose(q_a, q_b, atol=1e-14)

    def test_missing_both_args_raises(self):
        with pytest.raises(ValueError, match="space_group"):
            fundamental_zone([1.0, 0.0, 0.0, 0.0])

    def test_bad_sym_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            fundamental_zone([1.0, 0.0, 0.0, 0.0], sym=np.zeros((3, 3)))


class TestMatrixMultF33:
    def test_identity_left(self):
        M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        out = matrix_mult_f33(np.eye(3), M)
        np.testing.assert_allclose(out, M, atol=1e-14)

    def test_identity_right(self):
        M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        out = matrix_mult_f33(M, np.eye(3))
        np.testing.assert_allclose(out, M, atol=1e-14)

    def test_matches_numpy_matmul(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 3))
        np.testing.assert_allclose(matrix_mult_f33(A, B), A @ B, atol=1e-14)

    def test_matches_calcmiso_reference(self):
        """Bit-parity check against utils/calcMiso.MatrixMultF33 reference impl."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 3))
        # Inlined reference from utils/calcMiso.py:304-310 to avoid sys.path tricks.
        ref = np.zeros((3, 3))
        for r in range(3):
            ref[r, 0] = A[r, 0]*B[0, 0] + A[r, 1]*B[1, 0] + A[r, 2]*B[2, 0]
            ref[r, 1] = A[r, 0]*B[0, 1] + A[r, 1]*B[1, 1] + A[r, 2]*B[2, 1]
            ref[r, 2] = A[r, 0]*B[0, 2] + A[r, 1]*B[1, 2] + A[r, 2]*B[2, 2]
        np.testing.assert_allclose(matrix_mult_f33(A, B), ref, atol=1e-14)

    def test_accepts_list_input(self):
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        B = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
        np.testing.assert_allclose(matrix_mult_f33(A, B), np.asarray(B, dtype=float), atol=1e-14)


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


# ---------------------------------------------------------------------------
# Rodrigues -> matrix: the ANGLE, not just the structure
# ---------------------------------------------------------------------------

def test_rodrigues_to_orient_mat_has_the_right_rotation_angle():
    """|r| = tan(theta/2), so the matrix must rotate by exactly 2*atan(|r|).

    This is the assertion the suite was missing. Until it existed the function
    returned a proper rotation about the correct axis at the WRONG angle,
    inflated by 1/cos^2(theta/2) -- 30 deg came back as 32.154, 60 as 80, 90 as
    180 -- and every existing test passed: [0,0,0] is the one input where the
    error vanishes, det/orthogonality are preserved by it, and the numpy/torch
    comparison agreed because both backends carried it.

    120 deg is deliberately NOT in this list. theta/cos^2(theta/2) = 480 there,
    and 480 mod 360 = 120, so the broken function returns the correct angle by
    coincidence and a test using it would certify the bug.
    """
    import math
    import numpy as np

    for deg in (5.0, 30.0, 60.0, 90.0, 170.0):
        r = math.tan(math.radians(deg) / 2.0)
        R = np.asarray(rodrigues_to_orient_mat([0.0, 0.0, r]), dtype=float)
        cos_t = (np.trace(R) - 1.0) / 2.0
        got = math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))
        assert got == pytest.approx(deg, abs=1e-9), f"{deg} deg came back as {got}"


def test_rodrigues_to_orient_mat_preserves_the_axis():
    """The old code got the axis right and the angle wrong; assert both, so a
    future change cannot trade one for the other."""
    import numpy as np

    rod = np.array([0.3, -0.5, 0.2])
    R = np.asarray(rodrigues_to_orient_mat(rod), dtype=float)
    evals, evecs = np.linalg.eig(R)
    axis = np.real(evecs[:, int(np.argmin(np.abs(evals - 1.0)))])
    axis /= np.linalg.norm(axis)
    expect = rod / np.linalg.norm(rod)
    assert abs(float(expect @ axis)) == pytest.approx(1.0, abs=1e-12)


def test_rodrigues_agrees_with_axis_angle_construction():
    """Cross-check against an INDEPENDENT constructor.

    The pre-existing tests compared the function to itself (numpy vs torch) and
    to properties it satisfies while wrong. axis_angle_to_orient_mat builds the
    same rotation by a different route, so it can disagree.
    """
    import math
    import numpy as np

    for deg in (30.0, 60.0, 90.0):
        r = math.tan(math.radians(deg) / 2.0)
        a = np.asarray(axis_angle_to_orient_mat([0.0, 0.0, 1.0], deg), dtype=float)
        b = np.asarray(rodrigues_to_orient_mat([0.0, 0.0, r]), dtype=float)
        assert np.abs(a - b).max() < 1e-12
