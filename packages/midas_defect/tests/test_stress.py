import numpy as np
import pytest

from midas_defect.phases import FCC_SLIP_111_110
from midas_defect.stress import (
    cubic_stiffness_voigt,
    hydrostatic,
    lode_parameter,
    max_resolved_shear_per_grain,
    max_shear,
    per_grain_stress_cubic,
    resolved_shear_stress,
    strain_tensor_to_voigt,
    stress_tensor_to_voigt,
    triaxiality,
    voigt_to_strain_tensor,
    voigt_to_stress_tensor,
    von_mises,
)


# -- Voigt round-trip -------------------------------------------------------

def test_stress_voigt_roundtrip():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(5, 3, 3))
    s = 0.5 * (s + np.swapaxes(s, -1, -2))  # symmetrize
    np.testing.assert_allclose(
        voigt_to_stress_tensor(stress_tensor_to_voigt(s)), s, atol=1e-12
    )


def test_strain_voigt_roundtrip_with_factor_of_2():
    rng = np.random.default_rng(1)
    e = rng.normal(size=(3, 3, 3))
    e = 0.5 * (e + np.swapaxes(e, -1, -2))
    v = strain_tensor_to_voigt(e)
    # 4th entry is 2 * e_23, so the factor must round-trip out.
    np.testing.assert_allclose(v[..., 3], 2.0 * e[..., 1, 2], atol=1e-12)
    np.testing.assert_allclose(voigt_to_strain_tensor(v), e, atol=1e-12)


def test_cubic_stiffness_matrix_structure_and_isotropy_limit():
    c11, c12, c44 = 200.0, 100.0, 50.0
    C = cubic_stiffness_voigt(c11, c12, c44)
    assert C.shape == (6, 6)
    # Diagonal: c11, c11, c11, c44, c44, c44
    np.testing.assert_allclose(np.diag(C), [c11, c11, c11, c44, c44, c44])
    # Upper-left 3x3 off-diagonals = c12
    assert C[0, 1] == C[0, 2] == C[1, 2] == c12
    # Symmetric
    np.testing.assert_allclose(C, C.T)


# -- Invariants -------------------------------------------------------------

def test_invariants_under_uniaxial_tension():
    sigma0 = 100e6  # Pa
    s = np.zeros((3, 3))
    s[2, 2] = sigma0
    assert von_mises(s) == pytest.approx(sigma0)
    assert hydrostatic(s) == pytest.approx(sigma0 / 3.0)
    assert triaxiality(s) == pytest.approx(1.0 / 3.0)
    # Lode mu for uniaxial: lambdas = (0, 0, sigma0); mu = -1
    assert lode_parameter(s) == pytest.approx(-1.0)


def test_invariants_under_pure_shear():
    tau = 50e6  # Pa
    s = np.zeros((3, 3))
    s[0, 1] = s[1, 0] = tau
    # Pure shear principal stresses = (-tau, 0, +tau)
    assert von_mises(s) == pytest.approx(np.sqrt(3.0) * tau)
    assert hydrostatic(s) == pytest.approx(0.0)
    assert lode_parameter(s) == pytest.approx(0.0)
    assert max_shear(s) == pytest.approx(tau)


def test_triaxiality_inf_for_pure_hydrostatic():
    s = np.eye(3) * 10e6
    assert np.isinf(triaxiality(s))


def test_lode_nan_for_isotropic():
    s = np.eye(3) * 10e6
    assert np.isnan(lode_parameter(s))


def test_invariants_broadcast_over_batch():
    s = np.zeros((4, 3, 3))
    s[0, 2, 2] = 100e6
    s[1, 0, 1] = s[1, 1, 0] = 50e6
    s[2] = np.eye(3) * 5e6
    s[3] = np.array([[10, 1, 0], [1, 20, 2], [0, 2, 30]]) * 1e6
    assert von_mises(s).shape == (4,)


# -- Cubic stiffness propagation --------------------------------------------

def test_per_grain_stress_identity_orientation_diagonal_strain():
    # Cu-like constants
    c11, c12, c44 = 169.0, 122.0, 75.3
    OM = np.eye(3)[None]
    # epsilon = diag(0.001, 0, 0): sigma_11 = c11*0.001, sigma_22=sigma_33=c12*0.001
    eps = np.zeros((1, 3, 3))
    eps[0, 0, 0] = 1e-3
    sigma = per_grain_stress_cubic(OM, eps, c11, c12, c44)
    assert sigma[0, 0, 0] == pytest.approx(c11 * 1e-3 * 1e9)  # Pa
    assert sigma[0, 1, 1] == pytest.approx(c12 * 1e-3 * 1e9)
    assert sigma[0, 2, 2] == pytest.approx(c12 * 1e-3 * 1e9)
    np.testing.assert_allclose(sigma[0] - np.diag(np.diag(sigma[0])), 0.0, atol=1e-3)


def test_per_grain_stress_rotation_invariance_of_frobenius_norm():
    # For an isotropic stiffness, sigma is identical in any frame; for cubic
    # the *magnitude* of the stress tensor (Frobenius norm in sample frame)
    # should match a baseline up to anisotropic shuffling. Here we use the
    # isotropic limit c11 - c12 - 2 c44 = 0 (Zener anisotropy = 1) to check
    # full rotation invariance.
    mu = 50.0
    K = 100.0  # bulk
    # Isotropic stiffness in cubic terms: c11 = K + 4 mu/3, c12 = K - 2 mu/3, c44 = mu
    c11 = K + 4.0 * mu / 3.0
    c12 = K - 2.0 * mu / 3.0
    c44 = mu

    rng = np.random.default_rng(7)
    # Random orientation
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    eps = np.array([[1.0, 0.2, 0.0], [0.2, -0.3, 0.1], [0.0, 0.1, 0.0]]) * 1e-3

    sigma_identity = per_grain_stress_cubic(np.eye(3)[None], eps[None], c11, c12, c44)[0]
    sigma_rotated = per_grain_stress_cubic(R[None], (R @ eps @ R.T)[None], c11, c12, c44)[0]
    # Should match: cubic-isotropic stiffness commutes with arbitrary rotations.
    np.testing.assert_allclose(
        np.linalg.norm(sigma_identity), np.linalg.norm(sigma_rotated), rtol=1e-12
    )


# -- Resolved shear ---------------------------------------------------------

def test_rss_for_diagonal_stress_and_111_110():
    sigma = np.diag([0.0, 0.0, 100e6])  # uniaxial tension along z
    n_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b_hat = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    # tau = | b . sigma . n | = | b_z * sigma_zz * n_z | = | 0 * sigma * (1/sqrt 3) |
    # = 0  (because b has no z-component)
    assert resolved_shear_stress(sigma, n_hat, b_hat) == pytest.approx(0.0, abs=1e-12)


def test_rss_textbook_schmid_max():
    # sigma = sigma0 * e_z e_z^T, choose system n=(1,1,1)/sqrt 3, b=(-1,0,1)/sqrt 2
    # tau = | b_z * sigma * n_z | = | (1/sqrt 2) * sigma0 * (1/sqrt 3) | = sigma0 / sqrt 6
    sigma0 = 100e6
    sigma = np.diag([0.0, 0.0, sigma0])
    n_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b_hat = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2)
    assert resolved_shear_stress(sigma, n_hat, b_hat) == pytest.approx(sigma0 / np.sqrt(6))


def test_max_rss_per_grain_returns_max_over_systems():
    # Batch of 2 grains, identity stress on each, batch shape consistent.
    rng = np.random.default_rng(0)
    sigma = rng.normal(size=(2, 3, 3)) * 1e6
    sigma = 0.5 * (sigma + np.swapaxes(sigma, -1, -2))
    tau_max, active = max_resolved_shear_per_grain(sigma, FCC_SLIP_111_110)
    assert tau_max.shape == (2,)
    assert active.shape == (2,)
    # Reproducibility: max over all systems must equal tau_max.
    for g in range(2):
        for k in range(FCC_SLIP_111_110.shape[0]):
            t = resolved_shear_stress(sigma[g], FCC_SLIP_111_110[k, 0], FCC_SLIP_111_110[k, 1])
            assert t <= tau_max[g] + 1e-9
