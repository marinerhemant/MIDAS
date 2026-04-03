"""Tests for stress_strain.py utilities.

Run with:
    cd /Users/hsharma/opt/MIDAS/utils
    python -m pytest tests/test_stress_strain.py -v
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stress_strain import (
    tensor_to_voigt, voigt_to_tensor,
    lattice_params_to_A_matrix, lattice_params_to_strain,
    strain_grain_to_lab, strain_lab_to_grain,
    rotation_voigt_mandel,
    hooke_stress, cubic_stiffness, get_stiffness,
    volume_average_stress_constraint,
    hydrostatic_deviatoric_decomposition,
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
        # Mandel: v[5] = sqrt(2)*T[0,1] = sqrt(2)*0.5
        assert abs(v[5] - math.sqrt(2) * 0.5) < 1e-15

    def test_frobenius_norm_preserved(self):
        """Mandel convention preserves Frobenius norm."""
        T = np.array([[1, 0.3, 0.1], [0.3, 2, 0.2], [0.1, 0.2, 3]], dtype=float)
        v = tensor_to_voigt(T)
        np.testing.assert_allclose(
            np.linalg.norm(T, 'fro'), np.linalg.norm(v), atol=1e-14
        )

    def test_batch(self):
        T = np.random.randn(5, 3, 3)
        T = 0.5 * (T + np.swapaxes(T, -1, -2))  # symmetrize
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
        # For cubic: A should be a*I (with Fable convention ordering)
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
        # Tensile in a-direction: eps[0,0] > 0
        assert eps[0, 0] > 0
        # No shear for cubic with only a-stretch
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
        np.random.seed(42)
        # Random rotation
        from scipy.spatial.transform import Rotation
        orient = Rotation.random().as_matrix()
        eps_lab = strain_grain_to_lab(eps, orient)
        eps_back = strain_lab_to_grain(eps_lab, orient)
        np.testing.assert_allclose(eps_back, eps, atol=1e-14)

    def test_trace_preserved(self):
        """Rotation preserves trace (hydrostatic component)."""
        eps = np.array([[0.003, 0.001, 0], [0.001, -0.001, 0.0005],
                        [0, 0.0005, -0.002]])
        from scipy.spatial.transform import Rotation
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
        """M should be orthogonal: M @ M^T = I."""
        from scipy.spatial.transform import Rotation
        orient = Rotation.random(random_state=42).as_matrix()
        M = rotation_voigt_mandel(orient)
        np.testing.assert_allclose(M @ M.T, np.eye(6), atol=1e-13)

    def test_det_one(self):
        from scipy.spatial.transform import Rotation
        orient = Rotation.random(random_state=42).as_matrix()
        M = rotation_voigt_mandel(orient)
        assert abs(np.linalg.det(M) - 1.0) < 1e-12

    def test_consistent_with_tensor_transform(self):
        """M @ eps_voigt should equal voigt(U @ eps @ U^T)."""
        from scipy.spatial.transform import Rotation
        orient = Rotation.random(random_state=7).as_matrix()
        eps = np.array([[0.001, 0.0003, -0.0001],
                        [0.0003, -0.0005, 0.0002],
                        [-0.0001, 0.0002, -0.0005]])
        # Tensor transform
        eps_rotated = orient @ eps @ orient.T
        v_expected = tensor_to_voigt(eps_rotated)
        # Voigt transform
        M = rotation_voigt_mandel(orient)
        v_actual = M @ tensor_to_voigt(eps)
        np.testing.assert_allclose(v_actual, v_expected, atol=1e-14)

    def test_batch(self):
        from scipy.spatial.transform import Rotation
        orients = Rotation.random(5, random_state=42).as_matrix()
        M = rotation_voigt_mandel(orients)
        assert M.shape == (5, 6, 6)


# ===================================================================
#  Hooke's law
# ===================================================================

class TestHooke:
    def test_zero_strain_zero_stress(self):
        C = get_stiffness("Au")
        eps = np.zeros((3, 3))
        sig = hooke_stress(eps, C, frame="grain")
        np.testing.assert_allclose(sig, np.zeros((3, 3)), atol=1e-15)

    def test_hydrostatic_cubic(self):
        """Hydrostatic strain in cubic: sigma = (C11 + 2*C12) * eps * I."""
        C11, C12, C44 = 192.9, 163.8, 41.5  # Au
        C = cubic_stiffness(C11, C12, C44)
        eps_val = 0.001
        eps = eps_val * np.eye(3)
        sig = hooke_stress(eps, C, frame="grain")
        expected_hydro = (C11 + 2 * C12) * eps_val
        np.testing.assert_allclose(np.diag(sig), expected_hydro, rtol=1e-12)
        np.testing.assert_allclose(sig - np.diag(np.diag(sig)), 0, atol=1e-12)

    def test_lab_frame_with_identity(self):
        """Lab frame with identity orientation should equal grain frame."""
        C = get_stiffness("Cu")
        eps = np.diag([0.001, -0.0005, -0.0005])
        sig_grain = hooke_stress(eps, C, frame="grain")
        sig_lab = hooke_stress(eps, C, orient=np.eye(3), frame="lab")
        np.testing.assert_allclose(sig_lab, sig_grain, atol=1e-12)

    def test_stress_trace_preserved(self):
        """Trace of stress should be independent of orientation."""
        C = get_stiffness("Fe")
        eps_grain = np.diag([0.001, -0.0003, -0.0007])
        sig_grain = hooke_stress(eps_grain, C, frame="grain")

        from scipy.spatial.transform import Rotation
        orient = Rotation.random(random_state=42).as_matrix()
        eps_lab = orient @ eps_grain @ orient.T
        sig_lab = hooke_stress(eps_lab, C, orient=orient, frame="lab")
        np.testing.assert_allclose(np.trace(sig_grain), np.trace(sig_lab), atol=1e-10)

    def test_material_library(self):
        """All materials in library should produce valid stiffness matrices."""
        from stress_strain import STIFFNESS_LIBRARY
        for mat in STIFFNESS_LIBRARY:
            C = get_stiffness(mat)
            assert C.shape == (6, 6)
            # Stiffness should be positive definite
            eigvals = np.linalg.eigvalsh(C)
            assert np.all(eigvals > 0), f"{mat} stiffness not positive definite"


# ===================================================================
#  Volume-average stress constraint (B1)
# ===================================================================

class TestVolumeAverageConstraint:
    def test_unloaded_zero_average(self):
        """After correction with zero applied stress, volume average should be zero."""
        N = 10
        np.random.seed(42)
        stresses = np.random.randn(N, 3, 3) * 100  # random stresses in MPa
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.random.rand(N) + 0.1

        corrected = volume_average_stress_constraint(stresses, volumes)

        V_total = volumes.sum()
        w = volumes / V_total
        avg = np.sum(w[:, None, None] * corrected, axis=0)
        np.testing.assert_allclose(avg, np.zeros((3, 3)), atol=1e-10)

    def test_with_applied_stress(self):
        """Volume average should equal applied stress."""
        N = 5
        np.random.seed(123)
        stresses = np.random.randn(N, 3, 3) * 50
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.array([1.0, 2.0, 3.0, 1.5, 0.5])
        applied = np.diag([100.0, 0.0, 0.0])  # uniaxial 100 MPa

        corrected = volume_average_stress_constraint(stresses, volumes, applied)

        V_total = volumes.sum()
        w = volumes / V_total
        avg = np.sum(w[:, None, None] * corrected, axis=0)
        np.testing.assert_allclose(avg, applied, atol=1e-10)

    def test_voigt_input(self):
        """Should work with Voigt notation input."""
        N = 3
        sig_voigt = np.array([[100, 0, 0, 0, 0, 0],
                              [-50, 50, 0, 0, 0, 0],
                              [0, 0, 100, 0, 0, 0]], dtype=float)
        volumes = np.ones(N)
        corrected = volume_average_stress_constraint(sig_voigt, volumes)
        assert corrected.shape == (N, 6)
        avg = corrected.mean(axis=0)
        np.testing.assert_allclose(avg, np.zeros(6), atol=1e-10)

    def test_preserves_relative_differences(self):
        """Uniform correction should preserve grain-to-grain differences."""
        stresses = np.array([
            np.diag([100, 0, 0]),
            np.diag([200, 0, 0]),
        ], dtype=float)
        volumes = np.array([1.0, 1.0])
        corrected = volume_average_stress_constraint(stresses, volumes)
        diff_before = stresses[1] - stresses[0]
        diff_after = corrected[1] - corrected[0]
        np.testing.assert_allclose(diff_before, diff_after, atol=1e-14)


# ===================================================================
#  Hydrostatic-deviatoric decomposition (B2)
# ===================================================================

class TestHydrostaticDeviatoric:
    def test_decomposition_sums_to_original(self):
        """hydrostatic + deviatoric should equal corrected total."""
        N = 5
        np.random.seed(42)
        stresses = np.random.randn(N, 3, 3) * 100
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.random.rand(N) + 0.1

        hydro, dev, corrected = hydrostatic_deviatoric_decomposition(
            stresses, volumes
        )

        I = np.eye(3)
        reconstructed = hydro[:, None, None] * I[None, :, :] + dev
        np.testing.assert_allclose(corrected, reconstructed, atol=1e-12)

    def test_deviatoric_is_traceless(self):
        """Deviatoric part should have zero trace."""
        N = 8
        np.random.seed(7)
        stresses = np.random.randn(N, 3, 3) * 50
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.ones(N)

        _, dev, _ = hydrostatic_deviatoric_decomposition(stresses, volumes)

        traces = np.trace(dev, axis1=-2, axis2=-1)
        np.testing.assert_allclose(traces, np.zeros(N), atol=1e-12)

    def test_equilibrium_maintained(self):
        """Volume average of corrected stress should equal applied stress."""
        N = 6
        np.random.seed(99)
        stresses = np.random.randn(N, 3, 3) * 80
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.random.rand(N) + 0.5
        applied = np.diag([200.0, -100.0, 50.0])

        _, _, corrected = hydrostatic_deviatoric_decomposition(
            stresses, volumes, applied
        )

        V_total = volumes.sum()
        w = volumes / V_total
        avg = np.sum(w[:, None, None] * corrected, axis=0)
        np.testing.assert_allclose(avg, applied, atol=1e-10)

    def test_unloaded_hydrostatic_zero_average(self):
        """For unloaded sample, average hydrostatic should be zero."""
        N = 10
        np.random.seed(42)
        stresses = np.random.randn(N, 3, 3) * 100
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.random.rand(N) + 0.1

        hydro, _, _ = hydrostatic_deviatoric_decomposition(stresses, volumes)

        V_total = volumes.sum()
        w = volumes / V_total
        avg_hydro = np.sum(w * hydro)
        np.testing.assert_allclose(avg_hydro, 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
