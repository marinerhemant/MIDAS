"""Tests for equilibrium.py — volume-average and hydrostatic-deviatoric constraints."""

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from midas_stress.equilibrium import (
    volume_average_stress_constraint,
    hydrostatic_deviatoric_decomposition,
    hydrostatic_deviatoric_decomposition_weighted,
    equilibrium_correction_uncertainty,
    d0_correction_strain_level,
)
from midas_stress.materials import (
    d0_sensitivity, d0_sensitivity_table,
    get_stiffness, cubic_stiffness, hexagonal_stiffness,
)


class TestVolumeAverageConstraint:
    def test_unloaded_zero_average(self):
        N = 10
        np.random.seed(42)
        stresses = np.random.randn(N, 3, 3) * 100
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.random.rand(N) + 0.1

        corrected = volume_average_stress_constraint(stresses, volumes)

        V_total = volumes.sum()
        w = volumes / V_total
        avg = np.sum(w[:, None, None] * corrected, axis=0)
        np.testing.assert_allclose(avg, np.zeros((3, 3)), atol=1e-10)

    def test_with_applied_stress(self):
        N = 5
        np.random.seed(123)
        stresses = np.random.randn(N, 3, 3) * 50
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.array([1.0, 2.0, 3.0, 1.5, 0.5])
        applied = np.diag([100.0, 0.0, 0.0])

        corrected = volume_average_stress_constraint(stresses, volumes, applied)

        V_total = volumes.sum()
        w = volumes / V_total
        avg = np.sum(w[:, None, None] * corrected, axis=0)
        np.testing.assert_allclose(avg, applied, atol=1e-10)

    def test_voigt_input(self):
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
        stresses = np.array([
            np.diag([100, 0, 0]),
            np.diag([200, 0, 0]),
        ], dtype=float)
        volumes = np.array([1.0, 1.0])
        corrected = volume_average_stress_constraint(stresses, volumes)
        diff_before = stresses[1] - stresses[0]
        diff_after = corrected[1] - corrected[0]
        np.testing.assert_allclose(diff_before, diff_after, atol=1e-14)


class TestHydrostaticDeviatoric:
    def test_decomposition_sums_to_original(self):
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
        N = 8
        np.random.seed(7)
        stresses = np.random.randn(N, 3, 3) * 50
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.ones(N)

        _, dev, _ = hydrostatic_deviatoric_decomposition(stresses, volumes)
        traces = np.trace(dev, axis1=-2, axis2=-1)
        np.testing.assert_allclose(traces, np.zeros(N), atol=1e-12)

    def test_equilibrium_maintained(self):
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


# ===================================================================
#  Confidence-weighted variant
# ===================================================================

class TestWeightedDecomposition:
    def test_matches_unweighted_with_uniform_confidence(self):
        """With all confidences = 1.0, should match the unweighted version."""
        N = 10
        np.random.seed(42)
        stresses = np.random.randn(N, 3, 3) * 100
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.random.rand(N) + 0.1
        confidences = np.ones(N)

        h1, d1, c1 = hydrostatic_deviatoric_decomposition(stresses, volumes)
        h2, d2, c2, info = hydrostatic_deviatoric_decomposition_weighted(
            stresses, volumes, confidences)

        np.testing.assert_allclose(h1, h2, atol=1e-12)
        np.testing.assert_allclose(c1, c2, atol=1e-12)

    def test_low_confidence_filtered(self):
        """Grains below min_confidence should not affect the correction."""
        N = 20
        np.random.seed(7)
        stresses = np.random.randn(N, 3, 3) * 80
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.random.rand(N) + 0.1
        confidences = np.ones(N)
        confidences[0:3] = 0.1  # 3 low-confidence grains

        _, _, _, info = hydrostatic_deviatoric_decomposition_weighted(
            stresses, volumes, confidences, min_confidence=0.5)
        assert info['n_grains_used'] == N - 3
        assert info['n_grains_total'] == N

    def test_all_grains_receive_correction(self):
        """Even low-confidence grains should be corrected."""
        N = 10
        np.random.seed(99)
        stresses = np.random.randn(N, 3, 3) * 50 + 30 * np.eye(3)
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.ones(N)
        confidences = np.ones(N)
        confidences[0] = 0.05

        h, _, corrected, _ = hydrostatic_deviatoric_decomposition_weighted(
            stresses, volumes, confidences, min_confidence=0.5)
        # All grains should have their hydrostatic shifted
        assert corrected.shape == (N, 3, 3)
        # The low-confidence grain (idx=0) is still corrected
        assert not np.allclose(corrected[0], stresses[0])

    def test_uncertainty_decreases_with_more_grains(self):
        """Standard error should decrease as N increases."""
        np.random.seed(42)
        se_values = []
        for N in [20, 100, 500]:
            stresses = np.random.randn(N, 3, 3) * 80
            stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
            volumes = np.ones(N)
            info = equilibrium_correction_uncertainty(stresses, volumes)
            se_values.append(info['hydrostatic_se_MPa'])
        # SE should strictly decrease
        assert se_values[0] > se_values[1] > se_values[2]

    def test_effective_n_equals_n_for_equal_weights(self):
        """Kish's effective N = N when all weights are equal."""
        N = 50
        stresses = np.random.randn(N, 3, 3)
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.ones(N)
        info = equilibrium_correction_uncertainty(stresses, volumes)
        np.testing.assert_allclose(info['effective_n'], N, atol=1e-10)

    def test_effective_n_less_than_n_for_unequal_weights(self):
        """Kish's effective N < N when weights are unequal."""
        N = 50
        stresses = np.random.randn(N, 3, 3)
        stresses = 0.5 * (stresses + np.swapaxes(stresses, -1, -2))
        volumes = np.ones(N)
        volumes[0] = 100  # one dominant grain
        info = equilibrium_correction_uncertainty(stresses, volumes)
        assert info['effective_n'] < N


# ===================================================================
#  d0 sensitivity
# ===================================================================

class TestD0Sensitivity:
    def test_cu_sensitivity(self):
        """Cu: cubic, should be pure hydrostatic."""
        s = d0_sensitivity("Cu")
        assert s['is_pure_hydrostatic']
        assert s['bulk_modulus_GPa'] > 0

    def test_ti_sensitivity_not_pure_hydrostatic(self):
        """Ti: hexagonal, d0 error has deviatoric component."""
        s = d0_sensitivity("Ti")
        assert not s['is_pure_hydrostatic']
        assert s['hydrostatic_fraction'] < 1.0

    def test_all_materials(self):
        table = d0_sensitivity_table()
        assert len(table) == 9
        for mat, s in table.items():
            assert s['sensitivity_MPa_per_ppm'] > 0
            assert s['bulk_modulus_GPa'] > 0

    def test_w_has_highest_sensitivity(self):
        """Tungsten (stiffest) should have highest d0 sensitivity."""
        table = d0_sensitivity_table()
        w_sens = table['W']['sensitivity_MPa_per_100ppm']
        for mat, s in table.items():
            if mat != 'W':
                assert w_sens > s['sensitivity_MPa_per_100ppm']


# ===================================================================
#  Strain-level d0 correction
# ===================================================================

class TestD0CorrectionStrainLevel:
    def _make_test_data(self, material, n_grains, eps_iso_inject, seed=42):
        """Create synthetic data with a known d0 error and zero true strain."""
        C = get_stiffness(material)
        orients = Rotation.random(n_grains, random_state=seed).as_matrix()
        # Zero true strain — only the d0 error
        strains_true = np.zeros((n_grains, 3, 3))
        strains_measured = strains_true + eps_iso_inject * np.eye(3)
        volumes = np.ones(n_grains)
        return C, orients, strains_true, strains_measured, volumes

    def _make_test_data_noisy(self, material, n_grains, eps_iso_inject, seed=42):
        """Create synthetic data with d0 error + random measurement noise."""
        rng = np.random.default_rng(seed)
        C = get_stiffness(material)
        orients = Rotation.random(n_grains, random_state=seed).as_matrix()
        strains_true = rng.normal(0, 200e-6, (n_grains, 3, 3))
        strains_true = 0.5 * (strains_true + np.swapaxes(strains_true, -1, -2))
        strains_measured = strains_true + eps_iso_inject * np.eye(3)
        volumes = rng.lognormal(0, 0.5, n_grains)
        return C, orients, strains_true, strains_measured, volumes

    def test_cubic_recovers_eps_iso_exact(self):
        """For cubic Cu with zero noise, should recover eps_iso exactly."""
        eps_inject = 100e-6
        C, orients, _, strains_meas, volumes = self._make_test_data(
            "Cu", 200, eps_inject)
        result = d0_correction_strain_level(
            strains_meas, C, orients, volumes)
        np.testing.assert_allclose(result['eps_iso'], eps_inject, rtol=1e-10)

    def test_hexagonal_recovers_eps_iso_exact(self):
        """For hexagonal Ti with zero noise, should recover eps_iso exactly."""
        eps_inject = 100e-6
        C, orients, _, strains_meas, volumes = self._make_test_data(
            "Ti", 200, eps_inject)
        result = d0_correction_strain_level(
            strains_meas, C, orients, volumes)
        np.testing.assert_allclose(result['eps_iso'], eps_inject, rtol=1e-10)

    def test_residual_reduced(self):
        """Residual norm should decrease substantially after correction."""
        eps_inject = 200e-6
        C, orients, _, strains_meas, volumes = self._make_test_data(
            "Cu", 100, eps_inject)
        result = d0_correction_strain_level(
            strains_meas, C, orients, volumes)
        assert result['residual_norm_after'] < result['residual_norm_before'] * 1e-10

    def test_noisy_reduces_residual(self):
        """With noise, residual should still decrease significantly."""
        eps_inject = 200e-6
        C, orients, _, strains_meas, volumes = self._make_test_data_noisy(
            "Cu", 500, eps_inject)
        result = d0_correction_strain_level(
            strains_meas, C, orients, volumes)
        assert result['residual_norm_after'] < result['residual_norm_before'] * 0.1

    def test_hexagonal_per_grain_accuracy(self):
        """For Ti (HCP), strain-level correction gives better per-grain
        stresses than the naive hydrostatic-only shift.

        The naive approach applies a uniform hydrostatic shift in the lab
        frame. For HCP, the d0 artifact has an orientation-dependent
        deviatoric component, so the naive correction leaves per-grain
        errors. The strain-level approach removes the artifact before
        Hooke's law and produces exact per-grain stresses (when the
        true strain is zero).
        """
        from midas_stress.hooke import hooke_stress

        eps_inject = 100e-6
        C, orients, strains_true, strains_meas, volumes = self._make_test_data(
            "Ti", 200, eps_inject, seed=7)

        # Compute true stresses (from zero true strain -> all zeros)
        stresses_true = hooke_stress(strains_true, C, orient=orients, frame="lab")

        # Strain-level correction
        result = d0_correction_strain_level(
            strains_meas, C, orients, volumes)
        err_strain = np.max(np.abs(result['stresses_corrected'] - stresses_true))

        # Naive stress-level hydrostatic shift
        stresses_raw = hooke_stress(strains_meas, C, orient=orients, frame="lab")
        _, _, naive_corrected = hydrostatic_deviatoric_decomposition(
            stresses_raw, volumes)
        err_naive = np.max(np.abs(naive_corrected - stresses_true))

        # Strain-level should give exact per-grain recovery (zero error)
        assert err_strain < 1e-10
        # Naive should have nonzero per-grain error for HCP
        assert err_naive > 1e-5
        # And strain-level should be orders of magnitude better
        assert err_strain < err_naive * 1e-6


# ===================================================================
#  Two-step correction
# ===================================================================

class TestCorrectD0TwoStep:
    def test_isotropic_matches_strain_level(self):
        """For isotropic error, 2-step should match strain-level."""
        from midas_stress.equilibrium import correct_d0
        N = 200
        C = get_stiffness("Cu")
        orients = Rotation.random(N, random_state=42).as_matrix()
        volumes = np.ones(N)
        strains = np.zeros((N, 3, 3)) + 100e-6 * np.eye(3)

        result = correct_d0(strains, C, orients, volumes)
        err_1dof = np.max(np.abs(result['stresses_corrected']))
        err_2step = np.max(np.abs(result['stresses_2step']))
        # Both should be near zero for isotropic cubic
        assert err_1dof < 1e-10
        assert err_2step < 1e-10

    def test_anisotropic_2step_beats_1dof(self):
        """For anisotropic HCP error, 2-step should beat 1-DOF alone."""
        from midas_stress.equilibrium import correct_d0
        from midas_stress.hooke import hooke_stress
        N = 200
        C = get_stiffness("Ti")
        orients = Rotation.random(N, random_state=7).as_matrix()
        volumes = np.ones(N)

        ref_true = np.array([2.951, 2.951, 4.684, 90.0, 90.0, 120.0])
        ref_wrong = ref_true.copy()
        ref_wrong[2] *= 1.0005  # only c wrong

        from midas_stress.tensor import lattice_params_to_strain
        lp = np.tile(ref_true, (N, 1))
        strains_wrong = lattice_params_to_strain(lp, ref_wrong)
        stresses_true = hooke_stress(
            lattice_params_to_strain(lp, ref_true), C, orient=orients, frame="lab")

        result = correct_d0(strains_wrong, C, orients, volumes)

        err_1dof = np.sqrt(np.sum(
            (result['stresses_corrected'] - stresses_true)**2, axis=(-2,-1))).mean()
        err_2step = np.sqrt(np.sum(
            (result['stresses_2step'] - stresses_true)**2, axis=(-2,-1))).mean()

        assert err_2step < err_1dof


# ===================================================================
#  d0 recovery
# ===================================================================

class TestRecoverD0:
    def test_cubic_exact_recovery(self):
        """With zero true strain, should recover a0 exactly."""
        from midas_stress.equilibrium import recover_d0
        N = 200
        C = get_stiffness("Cu")
        orients = Rotation.random(N, random_state=42).as_matrix()
        volumes = np.ones(N)
        a0_true = 4.080
        # All grains have exactly the true lattice parameter
        lattice_params = np.tile([a0_true]*3 + [90.0]*3, (N, 1))
        # Assume wrong d0
        a0_wrong = 4.084
        ref_wrong = np.array([a0_wrong]*3 + [90.0]*3)
        result = recover_d0(lattice_params, ref_wrong, C, orients, volumes)
        a0_rec = result['reference_recovered'][0]
        err_ppm = abs(a0_rec - a0_true) / a0_true * 1e6
        assert err_ppm < 2.0  # < 2 ppm residual (second-order term)

    def test_hexagonal_recovery(self):
        """Should work for hexagonal Ti too."""
        from midas_stress.equilibrium import recover_d0
        N = 200
        C = get_stiffness("Ti")
        orients = Rotation.random(N, random_state=7).as_matrix()
        volumes = np.ones(N)
        ref_true = np.array([2.951, 2.951, 4.684, 90.0, 90.0, 120.0])
        lattice_params = np.tile(ref_true, (N, 1))
        # Assume wrong d0 (500 ppm)
        ref_wrong = ref_true.copy()
        ref_wrong[:3] *= 1.0005
        result = recover_d0(lattice_params, ref_wrong, C, orients, volumes)
        err_a = abs(result['reference_recovered'][0] - ref_true[0]) / ref_true[0] * 1e6
        err_c = abs(result['reference_recovered'][2] - ref_true[2]) / ref_true[2] * 1e6
        assert err_a < 1.0  # sub-ppm
        assert err_c < 1.0
        # Angles should be unchanged
        np.testing.assert_allclose(result['reference_recovered'][3:],
                                   ref_true[3:], atol=1e-10)
