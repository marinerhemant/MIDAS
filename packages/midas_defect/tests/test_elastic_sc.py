"""Tests for cpfem.elastic_sc: Reuss / Voigt / Kröner self-consistent prediction."""

from __future__ import annotations

import numpy as np
import pytest

from midas_defect.cpfem.elastic_sc import (
    _bond_rotation_matrices,
    _isotropic_voigt_stiffness,
    _voigt_average_isotropic,
    _reuss_average_isotropic,
    hill_average_isotropic,
    per_grain_lab_stiffness,
    reuss_per_grain,
    voigt_per_grain,
    kroner_self_consistent,
    per_grain_energy,
)
from midas_defect.stress.cubic_anisotropic import cubic_stiffness_voigt


# Cu single-crystal stiffness, GPa
C11, C12, C44 = 168.4, 121.4, 75.4


def test_bond_rotation_identity():
    """OM = I → Bond rotation = I_6."""
    M = _bond_rotation_matrices(np.eye(3)[None])
    assert np.allclose(M[0], np.eye(6), atol=1e-12)


def test_bond_rotation_orthogonal():
    """For any rotation R, M @ M^T should produce a Bond-compatible structure
    (note: Bond matrix M is NOT orthogonal in general — but C_lab = M C_xtal M^T
    must remain symmetric)."""
    rng = np.random.default_rng(42)
    # random rotation
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    M = _bond_rotation_matrices(Q[None])[0]
    C_xtal = cubic_stiffness_voigt(C11, C12, C44)
    C_lab = M @ C_xtal @ M.T
    assert np.allclose(C_lab, C_lab.T, atol=1e-9)


def test_isotropic_averages_consistent():
    Kv, mv = _voigt_average_isotropic(cubic_stiffness_voigt(C11, C12, C44))
    Kr, mr = _reuss_average_isotropic(cubic_stiffness_voigt(C11, C12, C44))
    Kh, mh = hill_average_isotropic(C11, C12, C44)
    # Reuss ≤ Hill ≤ Voigt
    assert Kr <= Kh <= Kv + 1e-9
    assert mr <= mh <= mv + 1e-9
    # Cu Hill values: K_VRH ≈ 137 GPa, μ_VRH ≈ 47 GPa
    assert abs(Kh - 137.07) < 0.5
    assert abs(mh - 47.34) < 0.5


def test_isotropic_stiffness_recovers_K_mu():
    K, mu = 137.0, 47.0
    C = _isotropic_voigt_stiffness(K, mu)
    K2, mu2 = _voigt_average_isotropic(C)
    K3, mu3 = _reuss_average_isotropic(C)
    assert abs(K2 - K) < 1e-9 and abs(mu2 - mu) < 1e-9
    assert abs(K3 - K) < 1e-9 and abs(mu3 - mu) < 1e-9


def test_per_grain_lab_stiffness_identity():
    """OM = I → C_lab = C_xtal."""
    C_lab = per_grain_lab_stiffness(np.eye(3)[None], C11, C12, C44)
    assert np.allclose(C_lab[0], cubic_stiffness_voigt(C11, C12, C44), atol=1e-9)


def test_per_grain_lab_stiffness_symmetric():
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    C_lab = per_grain_lab_stiffness(Q[None], C11, C12, C44)[0]
    assert np.allclose(C_lab, C_lab.T, atol=1e-9)
    # Eigenvalues all positive (stable)
    assert (np.linalg.eigvalsh(C_lab) > 0).all()


def test_reuss_bound_recovers_macroscopic_stress():
    """Reuss = uniform stress = σ_macro everywhere."""
    n = 50
    rng = np.random.default_rng(0)
    OMs = np.array([_random_rotation(rng) for _ in range(n)])
    sigma_macro = np.diag([100.0, 0.0, 0.0])  # MPa
    sigma, eps = reuss_per_grain(OMs, sigma_macro, C11, C12, C44)
    assert np.allclose(sigma, sigma_macro[None], atol=1e-9)
    # Per-grain ε varies through orientation-dependent compliance
    assert eps.std() > 1e-6


def test_voigt_bound_recovers_macroscopic_strain():
    n = 50
    rng = np.random.default_rng(1)
    OMs = np.array([_random_rotation(rng) for _ in range(n)])
    eps_macro = np.array([[1e-3, 0, 0], [0, -3e-4, 0], [0, 0, -3e-4]])
    sigma, eps = voigt_per_grain(OMs, eps_macro, C11, C12, C44)
    assert np.allclose(eps, eps_macro[None], atol=1e-12)
    # Per-grain σ varies through orientation-dependent stiffness
    assert sigma.std() > 1e-3


def test_isotropic_polycrystal_eshelby_recovers_macro():
    """For a randomly textured polycrystal under Kröner SC, the volume-averaged
    stress should converge to the macroscopic stress (within numerical
    tolerance)."""
    rng = np.random.default_rng(2)
    n = 500
    OMs = np.array([_random_rotation(rng) for _ in range(n)])
    sigma_macro = np.array([[150.0, 0, 0], [0, 0, 0], [0, 0, 0]])  # MPa
    res = kroner_self_consistent(OMs, sigma_macro, C11, C12, C44)
    sig_avg = res["sigma"].mean(axis=0)
    # The Kröner scheme uses an isotropic effective medium — recovers macro
    # stress to within a few percent for n=500 grains.
    assert abs(sig_avg[0, 0] - sigma_macro[0, 0]) / sigma_macro[0, 0] < 0.05
    assert abs(sig_avg[1, 1]) < 10.0  # MPa


def test_kroner_K_mu_match_hill():
    """For a sufficiently large random texture, the Kröner SC (K*, μ*) should
    fall between the Reuss and Voigt bounds and near the Hill average."""
    rng = np.random.default_rng(3)
    n = 1000
    OMs = np.array([_random_rotation(rng) for _ in range(n)])
    sigma_macro = np.diag([100.0, 0.0, 0.0])
    res = kroner_self_consistent(OMs, sigma_macro, C11, C12, C44)

    Kv, muv = _voigt_average_isotropic(cubic_stiffness_voigt(C11, C12, C44))
    Kr, mur = _reuss_average_isotropic(cubic_stiffness_voigt(C11, C12, C44))
    K_h, mu_h = hill_average_isotropic(C11, C12, C44)
    # For cubic crystals K_Voigt = K_Reuss; SC K should be within ~1 % of the
    # cubic bulk modulus (any iteration noise from mixing bulk and shear in
    # the isotropic Eshelby tensor).
    assert abs(res["K_eff"] - Kv) / Kv < 0.01
    # Shear modulus should lie in the strict [Reuss, Voigt] band
    assert mur <= res["mu_eff"] <= muv
    # And within ~25% of Hill (loose tolerance — SC isn't Hill exactly)
    assert abs(res["mu_eff"] - mu_h) / mu_h < 0.25


def test_per_grain_energy_positive_under_load():
    rng = np.random.default_rng(4)
    n = 100
    OMs = np.array([_random_rotation(rng) for _ in range(n)])
    sigma_macro = np.diag([200.0, 0.0, 0.0])
    sigma, eps = reuss_per_grain(OMs, sigma_macro, C11, C12, C44)
    U = per_grain_energy(sigma, eps)
    assert (U > 0).all()


def test_sigma3_twin_asymmetry_nontrivial():
    """For a Σ3 twin pair (60° rotation about a <111> axis), the matrix and
    twin variants should have DIFFERENT predicted elastic energy under a
    generic non-axial load. The asymmetry exists in the elastic prediction
    alone (no plasticity needed)."""
    # Matrix orientation: identity
    OM_matrix = np.eye(3)
    # Twin orientation: 60° rotation about [1,1,1]/√3 applied to matrix
    axis = np.array([1, 1, 1]) / np.sqrt(3)
    theta = np.radians(60)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    OM_twin = R @ OM_matrix
    OMs = np.stack([OM_matrix, OM_twin])

    # Off-axis load (so cubic anisotropy matters)
    sigma_macro = np.array([[80.0, 30.0, 10.0],
                            [30.0, -20.0, 5.0],
                            [10.0, 5.0, -60.0]])  # MPa

    sig_R, eps_R = reuss_per_grain(OMs, sigma_macro, C11, C12, C44)
    U_R = per_grain_energy(sig_R, eps_R)
    # Matrix vs twin should differ
    assert abs(U_R[0] - U_R[1]) / max(abs(U_R[0]), abs(U_R[1])) > 1e-6


def _random_rotation(rng):
    """Uniform random rotation in SO(3) via QR."""
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q
