"""Cross-validation against Park's MATLAB tools (matlab_tools/hedm).

Replicates the exact pipeline from parseGrainData_OneLayer_ff.m in Python
and verifies that midas-stress produces identical 3x3 stress tensors.

Park's pipeline:
  1. RMat = reshape(Grains_csv(i, 2:10), 3, 3)'
  2. RMat = RLab2Sam * R_ESRF2APS * RMat     (crystal -> APS sample)
  3. strain_sample = RLab2Sam * R_ESRF2APS * strain_midas * R_ESRF2APS' * RLab2Sam' / 1e6
  4. T = VectorizedCOBMatrix(RMat)            (6x6, order: [11,22,33,23,13,12])
  5. C_sample = T * C_xstal * T'
  6. stress_vec = C_sample * strain_vec
  7. stress_3x3 = MatrixOfStressStrainVectorInVM(stress_vec)
"""

import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import midas_stress as ms


# ===================================================================
#  Reimplementation of Park's MATLAB functions in Python
# ===================================================================

def park_R_ESRF2APS():
    """QuatOfESRF2APS.m: the cyclic permutation matrix."""
    return np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)


def park_RLab2Sam(omega_deg, crd_system="APS"):
    """Lab-to-sample rotation from parseGrainData_OneLayer_ff.m."""
    c = math.cos(math.radians(omega_deg))
    s = math.sin(math.radians(omega_deg))
    if crd_system == "APS":
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    elif crd_system == "ESRF":
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def park_VectorizedCOBMatrix(R, order="11-22-33-23-13-12"):
    """VectorizedCOBMatrix.m: 6x6 change-of-basis for R*A*R'.

    Default order: [11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12]
    """
    T = np.zeros((6, 6))
    if order == "11-22-33-23-13-12":
        # Shear pairs: index 3=(1,2)=yz, index 4=(0,2)=xz, index 5=(0,1)=xy
        for i in range(3):
            for j in range(3):
                T[i, j] = R[i, j]**2

        # Normal-shear (row < 3, col >= 3)
        for i in range(3):
            T[i, 3] = math.sqrt(2) * R[i, 1] * R[i, 2]  # yz
            T[i, 4] = math.sqrt(2) * R[i, 0] * R[i, 2]  # xz
            T[i, 5] = math.sqrt(2) * R[i, 0] * R[i, 1]  # xy

        # Shear-normal (row >= 3, col < 3)
        # Row 3 = yz: pairs (1,2)
        for j in range(3):
            T[3, j] = math.sqrt(2) * R[1, j] * R[2, j]
            T[4, j] = math.sqrt(2) * R[0, j] * R[2, j]
            T[5, j] = math.sqrt(2) * R[0, j] * R[1, j]

        # Shear-shear
        T[3, 3] = R[1, 1]*R[2, 2] + R[1, 2]*R[2, 1]
        T[3, 4] = R[1, 0]*R[2, 2] + R[1, 2]*R[2, 0]
        T[3, 5] = R[1, 0]*R[2, 1] + R[1, 1]*R[2, 0]

        T[4, 3] = R[0, 1]*R[2, 2] + R[0, 2]*R[2, 1]
        T[4, 4] = R[0, 0]*R[2, 2] + R[0, 2]*R[2, 0]
        T[4, 5] = R[0, 0]*R[2, 1] + R[0, 1]*R[2, 0]

        T[5, 3] = R[0, 1]*R[1, 2] + R[0, 2]*R[1, 1]
        T[5, 4] = R[0, 0]*R[1, 2] + R[0, 2]*R[1, 0]
        T[5, 5] = R[0, 0]*R[1, 1] + R[0, 1]*R[1, 0]

    return T


def park_VectorOfStressStrainMatrixInVM(T_3x3, order="11-22-33-23-13-12"):
    """VectorOfStressStrainMatrixInVM.m: 3x3 -> 6-vec with sqrt(2) shear."""
    s2 = math.sqrt(2)
    if order == "11-22-33-23-13-12":
        return np.array([
            T_3x3[0, 0], T_3x3[1, 1], T_3x3[2, 2],
            s2 * T_3x3[1, 2], s2 * T_3x3[0, 2], s2 * T_3x3[0, 1],
        ])


def park_MatrixOfStressStrainVectorInVM(v, order="11-22-33-23-13-12"):
    """MatrixOfStressStrainVectorInVM.m: 6-vec -> 3x3."""
    s2i = 1.0 / math.sqrt(2)
    T = np.zeros((3, 3))
    if order == "11-22-33-23-13-12":
        T[0, 0] = v[0]
        T[1, 1] = v[1]
        T[2, 2] = v[2]
        T[1, 2] = T[2, 1] = v[3] * s2i
        T[0, 2] = T[2, 0] = v[4] * s2i
        T[0, 1] = T[1, 0] = v[5] * s2i
    return T


def park_BuildElasticityMatrix_cubic(C11, C12, C44):
    """BuildElasticityMatrix.m for cubic."""
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = C12
    C[3, 3] = C[4, 4] = C[5, 5] = 2.0 * C44
    return C


def park_full_pipeline(orient_midas, strain_midas, C11, C12, C44, omega_deg=0):
    """Full Park MATLAB pipeline: MIDAS grain -> stress in APS sample frame.

    Parameters
    ----------
    orient_midas : ndarray (3, 3) — orientation matrix from MIDAS (crystal -> MIDAS lab)
    strain_midas : ndarray (3, 3) — strain tensor in MIDAS lab frame (natural units)
    C11, C12, C44 : float — elastic constants (GPa)
    omega_deg : float

    Returns
    -------
    stress_3x3 : ndarray (3, 3) — stress in APS sample frame (GPa)
    """
    R_E2A = park_R_ESRF2APS()
    R_L2S = park_RLab2Sam(omega_deg, "APS")

    # Transform orientation: crystal -> APS sample
    RMat = R_L2S @ R_E2A @ orient_midas

    # Transform strain: MIDAS lab -> APS sample
    R_total = R_L2S @ R_E2A
    strain_sample = R_total @ strain_midas @ R_total.T

    # Voigt-Mandel vectorization (Park ordering: yz, xz, xy)
    strain_vec = park_VectorOfStressStrainMatrixInVM(strain_sample)

    # Build stiffness and rotate to sample frame
    C_xstal = park_BuildElasticityMatrix_cubic(C11, C12, C44)
    T = park_VectorizedCOBMatrix(RMat)
    C_sample = T @ C_xstal @ T.T

    # Hooke's law
    stress_vec = C_sample @ strain_vec

    # Back to 3x3
    return park_MatrixOfStressStrainVectorInVM(stress_vec)


# ===================================================================
#  midas-stress pipeline
# ===================================================================

def midas_full_pipeline(orient_midas, strain_midas, C11, C12, C44, omega_deg=0):
    """midas-stress pipeline: MIDAS grain -> stress in APS sample frame.

    Parameters
    ----------
    Same as park_full_pipeline.

    Returns
    -------
    stress_3x3 : ndarray (3, 3) — stress in APS sample frame (GPa)
    """
    # Frame conversion
    sam = ms.grains_midas_to_sample(
        orient_midas[np.newaxis],
        np.zeros((1, 3)),  # positions don't matter for stress
        strain_midas[np.newaxis],
        omega_deg=omega_deg,
        target_frame="aps",
    )

    # Hooke's law
    C = ms.cubic_stiffness(C11, C12, C44)
    stress = ms.hooke_stress(
        strain=sam['strains'],
        stiffness=C,
        orient=sam['orientations'],
        frame="lab",
    )
    return stress[0]


# ===================================================================
#  Tests
# ===================================================================

class TestCrossValidationPark:
    """Verify midas-stress matches Park's MATLAB pipeline exactly."""

    def _random_grain(self, seed):
        """Generate a random grain with orientation and strain."""
        rng = np.random.default_rng(seed)
        orient = Rotation.random(random_state=seed).as_matrix()
        # Realistic microstrain-level symmetric strain
        strain = rng.normal(0, 500e-6, (3, 3))
        strain = 0.5 * (strain + strain.T)
        return orient, strain

    def test_cu_omega0(self):
        """Cu grain at omega=0: both pipelines should agree."""
        orient, strain = self._random_grain(42)
        C11, C12, C44 = 168.4, 121.4, 75.4

        stress_park = park_full_pipeline(orient, strain, C11, C12, C44, omega_deg=0)
        stress_midas = midas_full_pipeline(orient, strain, C11, C12, C44, omega_deg=0)

        np.testing.assert_allclose(stress_midas, stress_park, atol=1e-12,
                                   err_msg="Cu omega=0: Park vs midas-stress mismatch")

    def test_fe_omega45(self):
        """Fe grain at omega=45: both pipelines should agree."""
        orient, strain = self._random_grain(7)
        C11, C12, C44 = 231.4, 134.7, 116.4

        stress_park = park_full_pipeline(orient, strain, C11, C12, C44, omega_deg=45)
        stress_midas = midas_full_pipeline(orient, strain, C11, C12, C44, omega_deg=45)

        np.testing.assert_allclose(stress_midas, stress_park, atol=1e-12,
                                   err_msg="Fe omega=45: Park vs midas-stress mismatch")

    def test_ni_omega90(self):
        """Ni grain at omega=90."""
        orient, strain = self._random_grain(99)
        C11, C12, C44 = 246.5, 147.3, 124.7

        stress_park = park_full_pipeline(orient, strain, C11, C12, C44, omega_deg=90)
        stress_midas = midas_full_pipeline(orient, strain, C11, C12, C44, omega_deg=90)

        np.testing.assert_allclose(stress_midas, stress_park, atol=1e-12,
                                   err_msg="Ni omega=90: Park vs midas-stress mismatch")

    def test_batch_100_grains(self):
        """100 random Cu grains at omega=0: all should match."""
        C11, C12, C44 = 168.4, 121.4, 75.4
        max_err = 0.0
        for seed in range(100):
            orient, strain = self._random_grain(seed)
            stress_park = park_full_pipeline(orient, strain, C11, C12, C44)
            stress_midas = midas_full_pipeline(orient, strain, C11, C12, C44)
            err = np.max(np.abs(stress_midas - stress_park))
            max_err = max(max_err, err)

        assert max_err < 1e-11, f"Max error over 100 grains: {max_err:.2e}"

    def test_stress_symmetry(self):
        """Resulting stress tensor should be symmetric."""
        orient, strain = self._random_grain(42)
        C11, C12, C44 = 168.4, 121.4, 75.4

        stress = midas_full_pipeline(orient, strain, C11, C12, C44)
        np.testing.assert_allclose(stress, stress.T, atol=1e-14)

    def test_trace_invariance(self):
        """Trace of stress should be same in MIDAS and APS frames."""
        orient, strain = self._random_grain(42)
        C11, C12, C44 = 168.4, 121.4, 75.4

        # Compute stress in MIDAS frame directly (no frame conversion)
        C = ms.cubic_stiffness(C11, C12, C44)
        stress_midas = ms.hooke_stress(strain, C, orient=orient, frame="lab")

        # Compute stress in APS frame
        stress_aps = midas_full_pipeline(orient, strain, C11, C12, C44)

        np.testing.assert_allclose(np.trace(stress_midas), np.trace(stress_aps),
                                   atol=1e-12)
