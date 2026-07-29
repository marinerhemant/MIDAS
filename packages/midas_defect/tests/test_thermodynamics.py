import numpy as np
import pytest

from midas_defect.thermodynamics import (
    mk_evolve,
    taylor_implied_total_rho,
    variant_specific_k2,
    wh_visible_fraction,
)


# -- Mecking-Kocks ----------------------------------------------------------

def test_mk_evolve_saturates_to_k1_over_k2_squared():
    k1, k2 = 1e8, 5.0
    rho = mk_evolve(np.linspace(0.0, 5.0, 500), k1=k1, k2=k2, rho_init=1e10)
    sat = (k1 / k2) ** 2
    # rho should approach sat at large strain.
    assert rho[-1] == pytest.approx(sat, rel=0.01)


def test_mk_evolve_returns_initial_at_eps_zero():
    rho0 = 5e10
    out = mk_evolve(np.array([0.0]), k1=1e8, k2=5.0, rho_init=rho0)
    assert out[0] == pytest.approx(rho0, rel=1e-9)


def test_mk_evolve_monotonic_increasing_when_above_initial():
    rho = mk_evolve(np.linspace(0.0, 2.0, 200), k1=1e8, k2=5.0, rho_init=1e10)
    diffs = np.diff(rho)
    assert (diffs > 0).all()


def test_mk_evolve_rejects_nonpositive_k2():
    with pytest.raises(ValueError, match="k2 must be positive"):
        mk_evolve(np.array([0.0]), k1=1.0, k2=0.0)


def test_variant_specific_k2_recovers_sqrt_ratio():
    # rho_sat ratio 4:1 -> k_1/k_2 ratio 2:1 -> k_2 ratio 0.5:1
    out = variant_specific_k2({"matrix": 4e12, "twin": 1e12})
    np.testing.assert_allclose(out["k_ratio_per_variant"]["matrix"], np.sqrt(4e12))
    np.testing.assert_allclose(out["k_ratio_per_variant"]["twin"], np.sqrt(1e12))
    # k_2_twin / k_2_matrix = k_ratio_matrix / k_ratio_twin = sqrt(4) = 2.0
    assert out["k2_ratio_pairs"][("twin", "matrix")] == pytest.approx(2.0)


def test_variant_specific_k2_absolute_when_k1_given():
    out = variant_specific_k2({"matrix": 1e12, "twin": 4e12}, k1_literature=1e9)
    assert out["k2_absolute"]["matrix"] == pytest.approx(1e9 / np.sqrt(1e12))
    assert out["k2_absolute"]["twin"] == pytest.approx(1e9 / np.sqrt(4e12))


# -- Taylor inversion -------------------------------------------------------

def test_taylor_implied_total_rho_recovers_textbook_cu_at_700_MPa():
    # Cu at 700 MPa flow stress: rho_total ~ 4e15 m^-2 (canonical literature).
    rho = taylor_implied_total_rho(7.0e8)
    # Order of magnitude check.
    assert 1e15 < rho < 1e16


def test_taylor_implied_total_rho_scales_quadratically():
    rho_a = taylor_implied_total_rho(1.0e8)
    rho_b = taylor_implied_total_rho(2.0e8)
    assert rho_b / rho_a == pytest.approx(4.0, rel=1e-12)


def test_taylor_implied_total_rho_rejects_zero_denominator():
    with pytest.raises(ValueError, match="must be positive"):
        taylor_implied_total_rho(1.0e8, alpha=0.0)


def test_wh_visible_fraction_basic():
    rho_total = taylor_implied_total_rho(7.0e8)
    rho_WH = 0.15 * rho_total
    f = wh_visible_fraction(rho_WH, 7.0e8)
    assert f == pytest.approx(0.15, rel=1e-9)
