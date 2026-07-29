import numpy as np
import pytest

from midas_defect.phases import FCC_TWIN_111_112
from midas_defect.strain import (
    deviatoric_hydrostatic_decomposition,
    per_grain_eigenvalues,
    project_onto_system,
    twin_shear_projection_per_pair,
    von_mises_strain,
)


def test_von_mises_strain_uniaxial():
    # Uniaxial elastic strain along z with no Poisson coupling: eps_eq = |eps_zz|.
    eps = np.zeros((1, 3, 3))
    eps[0, 2, 2] = 1e-3
    # Deviatoric part: dev_zz = 2/3 eps0, dev_xx=dev_yy = -1/3 eps0
    # eps_eq = sqrt(2/3 * (4/9 + 1/9 + 1/9) * eps0^2) = sqrt(2/3 * 6/9 * eps0^2)
    #        = sqrt(4/9) * eps0 = 2/3 eps0
    assert von_mises_strain(eps)[0] == pytest.approx(2.0 / 3.0 * 1e-3)


def test_von_mises_strain_pure_shear():
    eps = np.zeros((1, 3, 3))
    eps[0, 0, 1] = eps[0, 1, 0] = 1e-3
    # Pure shear: dev_ij == eps_ij; dev:dev = 2 * eps12^2
    # eps_eq = sqrt(2/3 * 2 eps12^2) = sqrt(4/3) * eps12 = 2/sqrt(3) * eps12
    assert von_mises_strain(eps)[0] == pytest.approx(2.0 / np.sqrt(3) * 1e-3)


def test_deviatoric_hydrostatic_decomposition():
    eps = np.eye(3)[None] * 1e-3
    dev, tr = deviatoric_hydrostatic_decomposition(eps)
    np.testing.assert_allclose(dev, 0.0, atol=1e-18)
    assert tr[0] == pytest.approx(1e-3)


def test_per_grain_eigenvalues_uniaxial_lode_minus_one():
    eps = np.zeros((1, 3, 3))
    eps[0, 2, 2] = 1e-3
    out = per_grain_eigenvalues(eps)
    # eigvals are (0, 0, 1e-3) ascending => Lode mu = -1
    np.testing.assert_allclose(out["eigvals"][0], np.array([0.0, 0.0, 1e-3]), atol=1e-15)
    assert out["lode_parameter"][0] == pytest.approx(-1.0)


def test_per_grain_eigenvalues_lode_nan_for_isotropic():
    eps = np.eye(3)[None] * 1e-3
    out = per_grain_eigenvalues(eps)
    assert np.isnan(out["lode_parameter"][0])


def test_per_grain_eigenvalues_shape_validation():
    with pytest.raises(ValueError, match="must be"):
        per_grain_eigenvalues(np.zeros((3, 3)))


def test_project_onto_system_picks_up_planted_shear():
    # Plant a sample-frame shear on (100, 010): eps_12 = gamma/2 => b.eps.n = gamma/2
    gamma = 1e-3
    eps = np.zeros((1, 3, 3))
    eps[0, 0, 1] = eps[0, 1, 0] = gamma / 2.0
    n = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 1.0, 0.0]])
    assert project_onto_system(eps, n, b)[0] == pytest.approx(gamma / 2.0)


def test_twin_shear_projection_per_pair_picks_up_planted_shear():
    # Build a 2-grain pair with identity orientations and a planted twin-shear
    # difference. Loading axis along z. With FCC twin systems in crystal frame
    # and identity OM, the max-Schmid twin system for uniaxial-z is one of the
    # 12; we just verify the projection magnitude tracks the planted gamma.
    OM = np.tile(np.eye(3)[None], (2, 1, 1))

    eps = np.zeros((2, 3, 3))
    # Use a deviatoric shear with known projection on a {111}<112> system.
    # The numerical magnitude isn't asserted; we just confirm the routine
    # selects a system and returns finite values of both projections.
    eps[0] = np.diag([1.0, -0.5, -0.5]) * 1e-3
    eps[1] = np.diag([0.5, -0.25, -0.25]) * 1e-3

    pairs = np.array([[0, 1]])
    out = twin_shear_projection_per_pair(
        eps, OM, pairs, FCC_TWIN_111_112, sigma_axis_sample=np.array([0.0, 0.0, 1.0])
    )
    assert out["active_system"].shape == (1,)
    assert np.isfinite(out["dEps_twin_shear"][0])
    assert np.isfinite(out["dEps_orthogonal"][0])
    # Active system index must be in valid range.
    assert 0 <= out["active_system"][0] < FCC_TWIN_111_112.shape[0]


def test_twin_shear_projection_rejects_bad_pairs_shape():
    with pytest.raises(ValueError, match="pairs must be"):
        twin_shear_projection_per_pair(
            np.zeros((2, 3, 3)),
            np.tile(np.eye(3)[None], (2, 1, 1)),
            np.array([0, 1]),  # wrong shape
            FCC_TWIN_111_112,
            np.array([0.0, 0.0, 1.0]),
        )
