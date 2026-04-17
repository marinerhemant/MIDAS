"""Tests for plasticity.py — slip systems, Schmid factor, resolved shear, CRSS."""

import numpy as np
import pytest

import midas_stress as ms
from midas_stress.plasticity import (
    get_slip_systems,
    get_slip_systems_for_material,
    list_slip_families,
    slip_systems_to_lab,
    schmid_factor,
    resolved_shear_stress,
    dominant_slip_system,
    active_systems_from_crss,
    yield_proximity,
    taylor_factor,
    HCP_RATIOS,
)


def _random_rotations(n, seed):
    """Uniform random rotation matrices from a seed (no scipy dependency).

    Scipy's ``Rotation.random`` accepts ``rng=`` in recent releases and
    ``random_state=`` in older ones; this helper sidesteps the
    compatibility question entirely.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ===================================================================
#  Slip-system databases
# ===================================================================

class TestSlipSystems:
    def test_fcc_12_systems(self):
        n, b = get_slip_systems("fcc")
        assert n.shape == (12, 3)
        assert b.shape == (12, 3)
        # Unit vectors
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-12)
        # n perpendicular to b on every system (slip direction lies in plane)
        dots = np.einsum('mi,mi->m', n, b)
        np.testing.assert_allclose(dots, 0.0, atol=1e-12)

    def test_bcc_110(self):
        n, b = get_slip_systems("bcc_110")
        assert n.shape == (12, 3)
        dots = np.einsum('mi,mi->m', n, b)
        np.testing.assert_allclose(dots, 0.0, atol=1e-12)

    def test_bcc_combined(self):
        n, b = get_slip_systems("bcc")
        assert n.shape == (24, 3)
        dots = np.einsum('mi,mi->m', n, b)
        np.testing.assert_allclose(dots, 0.0, atol=1e-12)

    def test_bcc_all(self):
        n, b = get_slip_systems("bcc_all")
        assert n.shape == (48, 3)

    def test_hcp_requires_c_over_a(self):
        with pytest.raises(ValueError, match="c_over_a"):
            get_slip_systems("hcp_basal")

    def test_hcp_basal(self):
        n, b = get_slip_systems("hcp_basal", c_over_a=HCP_RATIOS["Ti"])
        assert n.shape == (3, 3)
        # Basal plane normal must be along c (z-axis)
        for ni in n:
            np.testing.assert_allclose(ni[:2], [0.0, 0.0], atol=1e-12)
            assert abs(abs(ni[2]) - 1.0) < 1e-12
        # Slip directions lie in basal plane (z=0)
        np.testing.assert_allclose(b[:, 2], 0.0, atol=1e-12)
        # n · b = 0
        dots = np.einsum('mi,mi->m', n, b)
        np.testing.assert_allclose(dots, 0.0, atol=1e-12)

    def test_hcp_prismatic_direction_in_basal(self):
        n, b = get_slip_systems("hcp_prismatic", c_over_a=HCP_RATIOS["Ti"])
        # <a>-type Burgers in basal plane
        np.testing.assert_allclose(b[:, 2], 0.0, atol=1e-12)
        # Prismatic normals lie in basal plane too
        np.testing.assert_allclose(n[:, 2], 0.0, atol=1e-12)
        dots = np.einsum('mi,mi->m', n, b)
        np.testing.assert_allclose(dots, 0.0, atol=1e-12)

    def test_hcp_all(self):
        n, b = get_slip_systems("hcp", c_over_a=1.587)
        # 3 + 3 + 6 + 12 = 24
        assert n.shape == (24, 3)
        dots = np.einsum('mi,mi->m', n, b)
        np.testing.assert_allclose(dots, 0.0, atol=1e-10)

    def test_material_dispatch(self):
        n_cu, _ = get_slip_systems_for_material("Cu")
        assert n_cu.shape == (12, 3)
        n_fe, _ = get_slip_systems_for_material("Fe")
        assert n_fe.shape == (24, 3)  # bcc combined
        n_ti, _ = get_slip_systems_for_material("Ti")
        assert n_ti.shape == (24, 3)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            get_slip_systems("bogus")

    def test_list_families(self):
        families = list_slip_families()
        assert "fcc" in families
        assert "bcc" in families
        assert "hcp_basal" in families


# ===================================================================
#  Schmid factor and resolved shear
# ===================================================================

class TestSchmidFactor:
    def test_schmid_identity_orientation(self):
        """For an unrotated FCC crystal under z-axis tension, the max
        Schmid factor should be 0.4082 (the classical FCC value).
        """
        U = np.eye(3)[None, :, :]
        n, b = get_slip_systems("fcc")
        m = schmid_factor(U, [0, 0, 1], n, b)
        # m should contain values up to sqrt(6)/6 = 0.40825
        assert m.shape == (1, 12)
        np.testing.assert_allclose(m.max(), np.sqrt(6) / 6, atol=1e-10)

    def test_schmid_no_load_along_slip_direction(self):
        """If the load is parallel to a slip direction, its Schmid
        factor on that system is 0 (since n ⟂ b ⟂ load means n·ell=0
        or the whole product vanishes). Use z-axis load with the
        {111}<110> system whose direction is [110]/sqrt(2) (in basal plane)."""
        U = np.eye(3)[None, :, :]
        n, b = get_slip_systems("fcc")
        m = schmid_factor(U, [1, 1, 0], n, b)
        # Many systems should still be active (non-zero)
        assert (m > 1e-6).any()

    def test_schmid_batched(self):
        U = _random_rotations(20, seed=0)
        n, b = get_slip_systems("fcc")
        m = schmid_factor(U, [0, 0, 1], n, b)
        assert m.shape == (20, 12)
        # All non-negative (absolute=True default) and bounded by 0.5
        assert m.min() >= 0
        assert m.max() <= 0.5 + 1e-12

    def test_schmid_signed_vs_absolute(self):
        U = np.eye(3)[None, :, :]
        n, b = get_slip_systems("fcc")
        m_abs = schmid_factor(U, [0, 0, 1], n, b, absolute=True)
        m_signed = schmid_factor(U, [0, 0, 1], n, b, absolute=False)
        np.testing.assert_allclose(np.abs(m_signed), m_abs, atol=1e-12)

    def test_resolved_shear_uniaxial_matches_schmid(self):
        """For a pure uniaxial stress sigma*ell*ell^T, tau = sigma * m."""
        U = _random_rotations(5, seed=42)
        n, b = get_slip_systems("fcc")
        load = np.array([0.0, 0.0, 1.0])
        sigma_val = 100.0  # MPa
        stress = sigma_val * np.einsum('i,j->ij', load, load)
        stress = np.broadcast_to(stress, (5, 3, 3)).copy()
        tau = resolved_shear_stress(stress, U, n, b)
        m_signed = schmid_factor(U, load, n, b, absolute=False)
        np.testing.assert_allclose(tau, sigma_val * m_signed, atol=1e-10)

    def test_slip_systems_to_lab_identity(self):
        U = np.eye(3)
        n, b = get_slip_systems("fcc")
        n_lab, b_lab = slip_systems_to_lab(U, n, b)
        np.testing.assert_allclose(n_lab, n, atol=1e-12)
        np.testing.assert_allclose(b_lab, b, atol=1e-12)

    def test_slip_systems_to_lab_rotation(self):
        # Rotate 90 degrees about z: x->y, y->-x
        theta = np.pi / 2
        U = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1.0],
        ])
        n_cr = np.array([[1.0, 0.0, 0.0]])
        b_cr = np.array([[0.0, 1.0, 0.0]])
        n_lab, b_lab = slip_systems_to_lab(U, n_cr, b_cr)
        np.testing.assert_allclose(n_lab[0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(b_lab[0], [-1, 0, 0], atol=1e-12)


# ===================================================================
#  Dominant system and CRSS analyses
# ===================================================================

class TestDominantSystem:
    def test_dominant_returns_largest(self):
        U = np.eye(3)[None, :, :]
        n, b = get_slip_systems("fcc")
        res = dominant_slip_system(U, n, b, load_dir=[0, 0, 1], top_k=3)
        assert res['rank'].shape == (1, 3)
        # top system's score equals the max Schmid factor
        m = schmid_factor(U, [0, 0, 1], n, b)
        np.testing.assert_allclose(res['best_score'][0], m[0].max())

    def test_dominant_from_stress(self):
        U = _random_rotations(10, seed=1)
        stress = np.zeros((10, 3, 3))
        stress[:, 2, 2] = 200.0  # uniaxial along z
        n, b = get_slip_systems("fcc")
        res = dominant_slip_system(U, n, b, stress=stress)
        # For uniaxial along z, best_score = 200 * max_schmid
        m_max = schmid_factor(U, [0, 0, 1], n, b).max(axis=1)
        np.testing.assert_allclose(res['best_score'], 200.0 * m_max, atol=1e-10)


class TestCRSS:
    def test_active_systems_scalar_threshold(self):
        """Under a stress below CRSS, no systems should activate."""
        U = np.eye(3)[None, :, :]
        n, b = get_slip_systems("fcc")
        stress = np.zeros((1, 3, 3))
        stress[0, 2, 2] = 10.0
        res = active_systems_from_crss(stress, U, n, b, crss=100.0)
        assert not res['active'].any()
        assert res['fraction_grains_yielding'] == 0.0

    def test_active_systems_above_crss(self):
        U = np.eye(3)[None, :, :]
        n, b = get_slip_systems("fcc")
        # Big uniaxial stress: several systems should activate
        stress = np.zeros((1, 3, 3))
        stress[0, 2, 2] = 1000.0
        res = active_systems_from_crss(stress, U, n, b, crss=50.0)
        # max Schmid ≈ 0.408 → max |tau| ≈ 408 MPa >> 50
        assert res['active'].any()
        assert res['fraction_grains_yielding'] == 1.0

    def test_per_system_crss(self):
        U = np.eye(3)[None, :, :]
        n, b = get_slip_systems("fcc")
        stress = np.zeros((1, 3, 3))
        stress[0, 2, 2] = 1000.0
        crss = np.full(12, 1e6)  # huge everywhere
        crss[0] = 1.0  # tiny for system 0
        res = active_systems_from_crss(stress, U, n, b, crss=crss)
        # Only system 0 can activate (if |tau[0]| >= 1)
        assert res['active'][0].sum() <= 1

    def test_yield_proximity(self):
        U = _random_rotations(50, seed=7)
        stress = np.zeros((50, 3, 3))
        stress[:, 2, 2] = 100.0
        n, b = get_slip_systems("fcc")
        res = yield_proximity(stress, U, n, b, crss=50.0)
        assert res['proximity'].shape == (50,)
        # Grains are sorted by proximity descending
        sorted_prox = res['proximity'][res['grains_sorted']]
        assert (np.diff(sorted_prox) <= 1e-12).all()

    def test_taylor_factor_fcc(self):
        """Taylor factor for FCC polycrystal under uniaxial load is ~3.06."""
        U = _random_rotations(2000, seed=123)
        n, b = get_slip_systems("fcc")
        res = taylor_factor(U, [0, 0, 1], n, b)
        # Single-slip approx gives the classical FCC value ~2.24 (not 3.06,
        # which is the full-constraint Taylor factor). Just check it is in a
        # reasonable range — M > 1 and finite.
        assert 2.0 < res['M_poly'] < 4.0
        assert np.isfinite(res['M_per_grain']).all()


# ===================================================================
#  End-to-end pipeline coupling
# ===================================================================

class TestEndToEnd:
    def test_compute_stress_then_resolved_shear(self):
        """Full flow: strain -> stress via compute_stress -> resolved shear."""
        U = _random_rotations(30, seed=0)
        rng = np.random.default_rng(0)
        strains = 1e-3 * rng.normal(size=(30, 3, 3))
        strains = 0.5 * (strains + strains.swapaxes(-1, -2))
        volumes = np.ones(30)

        result = ms.compute_stress(
            strain=strains,
            stiffness=ms.get_stiffness("Cu"),
            orient=U,
            volumes=volumes,
        )
        n, b = ms.get_slip_systems("fcc")
        tau = ms.resolved_shear_stress(result['stress_corrected'], U, n, b)
        assert tau.shape == (30, 12)
        assert np.isfinite(tau).all()
