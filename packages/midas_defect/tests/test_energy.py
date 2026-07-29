import numpy as np
import pytest

from midas_defect.energy import (
    elastic_energy_density_cubic,
    elastic_energy_density_isotropic,
    energy_balance_closure,
    twin_boundary_energy_density,
    volume_weighted_energy_per_variant,
)


# -- Isotropic Lamé ---------------------------------------------------------

def test_isotropic_uniaxial_tension_closed_form():
    # Uniaxial strain along z (no Poisson): eps_zz = e0, others zero.
    # U = 1/2 lambda * e0^2 + mu * e0^2.
    lam, mu = 100.0, 50.0  # GPa
    e0 = 1e-3
    eps = np.zeros((1, 3, 3))
    eps[0, 2, 2] = e0
    U = elastic_energy_density_isotropic(eps, lam, mu)
    expected_Pa = (0.5 * lam * e0**2 + mu * e0**2) * 1e9
    assert U[0] == pytest.approx(expected_Pa, rel=1e-12)


def test_isotropic_pure_shear_closed_form():
    # Pure shear eps_12 = eps_21 = g/2 -> tr = 0; eps:eps = 2 (g/2)^2 = g^2/2
    # U = mu * g^2 / 2
    lam, mu = 0.0, 75.0
    g = 1e-3
    eps = np.zeros((1, 3, 3))
    eps[0, 0, 1] = eps[0, 1, 0] = g / 2.0
    U = elastic_energy_density_isotropic(eps, lam, mu)
    assert U[0] == pytest.approx(mu * g * g / 2.0 * 1e9, rel=1e-12)


# -- Cubic anisotropic vs isotropic limit -----------------------------------

def test_cubic_reduces_to_isotropic_in_zener_unity_limit():
    # Zener anisotropy A = 2 c44 / (c11 - c12) = 1 -> isotropic limit.
    mu = 50.0  # GPa
    K = 100.0
    c11 = K + 4 * mu / 3
    c12 = K - 2 * mu / 3
    c44 = mu
    rng = np.random.default_rng(0)
    eps_raw = rng.normal(size=(5, 3, 3))
    eps = 0.5 * (eps_raw + np.swapaxes(eps_raw, -1, -2)) * 1e-3
    # Random orientation per grain
    OM = np.tile(np.eye(3)[None], (5, 1, 1))

    U_cubic = elastic_energy_density_cubic(OM, eps, c11, c12, c44)

    # Convert (K, mu) to Lamé: lambda = K - 2 mu / 3
    lam = K - 2 * mu / 3
    U_iso = elastic_energy_density_isotropic(eps, lam, mu)
    np.testing.assert_allclose(U_cubic, U_iso, rtol=1e-12)


# -- Volume weighting -------------------------------------------------------

def test_volume_weighted_per_variant_uses_radius_cubed():
    # Two variants; variant 0 has small grains, variant 1 has large.
    U = np.array([100.0, 100.0, 200.0, 200.0])
    radii = np.array([1.0, 1.0, 2.0, 2.0])
    var = np.array([0, 0, 1, 1])
    out = volume_weighted_energy_per_variant(U, radii, var)
    assert out["U_mean_per_variant"][0] == pytest.approx(100.0)
    assert out["U_mean_per_variant"][1] == pytest.approx(200.0)
    # V_total for variant 1 = 2 * (4/3 pi 2^3) = 8 * V_variant_0; check ratio.
    assert out["V_total_per_variant"][1] / out["V_total_per_variant"][0] == pytest.approx(8.0)
    assert out["ratio"][(1, 0)] == pytest.approx(2.0)


def test_volume_weighted_drops_nan_grains():
    U = np.array([100.0, np.nan, 50.0])
    radii = np.array([1.0, 1.0, 1.0])
    var = np.array([0, 0, 0])
    out = volume_weighted_energy_per_variant(U, radii, var)
    assert out["U_mean_per_variant"][0] == pytest.approx(75.0)


# -- Balance closure --------------------------------------------------------

def test_twin_boundary_energy_density_basic():
    # gamma = 0.04 J/m^2, L = 1e-7 m (= 100 nm). U_TB = 2*0.04/1e-7 = 8e5 J/m^3.
    U_TB = twin_boundary_energy_density(0.04, 1e-7)
    assert U_TB == pytest.approx(8e5)


def test_twin_boundary_rejects_nonpositive_L():
    with pytest.raises(ValueError, match="L_lamella must be positive"):
        twin_boundary_energy_density(0.04, 0.0)


def test_energy_balance_closure_unity_when_match():
    out = energy_balance_closure(U_matrix=1.8e6, U_twin=1.0e6, gamma_TB=0.04, L_lamella=1e-7)
    # dU = 8e5, U_TB = 8e5 -> ratio = 1.0
    assert out["closure_ratio"] == pytest.approx(1.0)
    assert out["within_unity_pct"] == pytest.approx(0.0)


def test_energy_balance_closure_half():
    out = energy_balance_closure(U_matrix=1.4e6, U_twin=1.0e6, gamma_TB=0.04, L_lamella=1e-7)
    # dU = 4e5, U_TB = 8e5 -> ratio = 0.5
    assert out["closure_ratio"] == pytest.approx(0.5)
    assert out["within_unity_pct"] == pytest.approx(50.0)
