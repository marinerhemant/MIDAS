import numpy as np
import pytest

from midas_defect.phases import (
    BCC_SLIP_110_111,
    BCC_SLIP_112_111,
    BCC_SLIP_123_111,
    BCC_TWIN_112_111,
    FCC_PARTIAL_111_112,
    FCC_SLIP_111_110,
    FCC_TWIN_111_112,
    GAMMA_TWIN_BCC,
    GAMMA_TWIN_FCC,
    CrystalPhase,
    bravais_to_miller,
    gamma_twin_hcp_tensile,
    hcp_systems,
    miller_to_bravais,
    n_systems,
)
from midas_defect.phases.hcp import (
    direction_bravais_to_cart,
    plane_normal_bravais_to_cart,
)


def _check_systems_table(systems: np.ndarray, expected_count: int):
    assert systems.shape == (expected_count, 2, 3)
    np.testing.assert_allclose(np.linalg.norm(systems[:, 0, :], axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(systems[:, 1, :], axis=1), 1.0, atol=1e-12)
    dots = np.sum(systems[:, 0, :] * systems[:, 1, :], axis=1)
    np.testing.assert_allclose(dots, 0.0, atol=1e-12)


def test_fcc_slip_table_has_12_orthogonal_unit_systems():
    _check_systems_table(FCC_SLIP_111_110, 12)


def test_fcc_twin_table_has_12_orthogonal_unit_systems():
    _check_systems_table(FCC_TWIN_111_112, 12)


def test_fcc_partials_share_geometry_with_twin_family():
    np.testing.assert_array_equal(FCC_PARTIAL_111_112, FCC_TWIN_111_112)


def test_bcc_slip_tables_have_correct_counts():
    _check_systems_table(BCC_SLIP_110_111, 12)
    _check_systems_table(BCC_SLIP_112_111, 12)
    _check_systems_table(BCC_SLIP_123_111, 24)


def test_bcc_twin_table_has_12_orthogonal_unit_systems():
    _check_systems_table(BCC_TWIN_112_111, 12)


def test_gamma_twin_constants():
    assert GAMMA_TWIN_FCC == pytest.approx(1.0 / np.sqrt(2.0))
    assert GAMMA_TWIN_BCC == pytest.approx(1.0 / np.sqrt(2.0))


# -- HCP --------------------------------------------------------------------

def test_hcp_bravais_miller_roundtrip_a1():
    # [2 -1 -1 0] is +a1 axis; 3-index is (3, 0, 0); roundtrip back.
    U, V, W = bravais_to_miller(2, -1, -1, 0)
    assert (U, V, W) == (3, 0, 0)
    u, v, t, w = miller_to_bravais(U, V, W)
    # The simplest reduction collapses the common factor 3.
    assert (u, v, t, w) == (2, -1, -1, 0)


def test_hcp_bravais_miller_roundtrip_c():
    U, V, W = bravais_to_miller(0, 0, 0, 1)
    assert (U, V, W) == (0, 0, 1)
    assert miller_to_bravais(U, V, W) == (0, 0, 0, 1)


def test_hcp_a1_axis_unit_vector_is_ex():
    a1 = direction_bravais_to_cart(2, -1, -1, 0, c_over_a=1.6)
    np.testing.assert_allclose(a1, np.array([1.0, 0.0, 0.0]), atol=1e-12)


def test_hcp_basal_plane_normal_is_ez():
    n = plane_normal_bravais_to_cart(0, 0, 0, 1, c_over_a=1.6)
    np.testing.assert_allclose(n, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_hcp_prismatic_plane_normal_recovers_expected_direction():
    # (1 0 -1 0): n_cart proportional to (1, 1/sqrt(3), 0).
    n = plane_normal_bravais_to_cart(1, 0, -1, 0, c_over_a=1.6)
    expected = np.array([1.0, 1.0 / np.sqrt(3), 0.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(n, expected, atol=1e-12)


def test_hcp_basal_family_count_and_orthogonality():
    systems = hcp_systems("basal_a", c_over_a=1.624)
    _check_systems_table(systems, 3)
    # Basal plane normals all = +z (or -z up to sign).
    for i in range(3):
        np.testing.assert_allclose(np.abs(systems[i, 0]), [0.0, 0.0, 1.0], atol=1e-12)


def test_hcp_prismatic_family_count_and_orthogonality():
    _check_systems_table(hcp_systems("prismatic_a", c_over_a=1.624), 3)


def test_hcp_pyramidal_a_family_count_and_orthogonality():
    _check_systems_table(hcp_systems("pyramidal_a", c_over_a=1.624), 6)


def test_hcp_pyramidal_ca_family_count_and_orthogonality():
    _check_systems_table(hcp_systems("pyramidal_ca", c_over_a=1.624), 6)


def test_hcp_twin_tensile_family_count_and_orthogonality():
    _check_systems_table(hcp_systems("twin_tensile", c_over_a=1.624), 6)


def test_hcp_twin_compressive_family_count_and_orthogonality():
    _check_systems_table(hcp_systems("twin_compressive", c_over_a=1.624), 6)


def test_gamma_twin_tensile_vanishes_at_ideal_c_over_a():
    # Tensile twin shear is zero at c/a = sqrt(3).
    assert gamma_twin_hcp_tensile(np.sqrt(3.0)) == pytest.approx(0.0, abs=1e-12)


def test_gamma_twin_tensile_at_mg_c_over_a():
    # Mg c/a = 1.624; |(c/a)^2 - 3| / (sqrt(3) c/a) = |1.624^2 - 3| / (sqrt(3)*1.624)
    val = gamma_twin_hcp_tensile(1.624)
    expected = abs(1.624**2 - 3) / (np.sqrt(3) * 1.624)
    assert val == pytest.approx(expected, rel=1e-12)


def test_hcp_systems_rejects_unknown_family():
    with pytest.raises(KeyError, match="unknown HCP family"):
        hcp_systems("not_a_family", c_over_a=1.6)


def test_miller_bravais_constraint_enforced():
    with pytest.raises(ValueError, match="Miller-Bravais"):
        direction_bravais_to_cart(1, 1, 1, 0, c_over_a=1.6)  # u+v+t != 0
    with pytest.raises(ValueError, match="Miller-Bravais"):
        plane_normal_bravais_to_cart(1, 1, 1, 0, c_over_a=1.6)


# -- n_systems --------------------------------------------------------------

def test_n_systems_dispatches_by_phase():
    assert n_systems(CrystalPhase.FCC, "slip") == 12
    assert n_systems(CrystalPhase.FCC, "twin") == 12
    assert n_systems(CrystalPhase.BCC, "slip") == 12
    assert n_systems(CrystalPhase.BCC, "twin") == 12
    assert n_systems(CrystalPhase.HCP, "slip") == 3
    assert n_systems(CrystalPhase.HCP, "twin") == 6
    with pytest.raises(ValueError, match="kind must be"):
        n_systems(CrystalPhase.FCC, "partial")
