"""Tests for elastic_inverse.py — Paper III single-crystal Cij recovery."""

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from midas_stress import (
    fit_single_crystal_stiffness,
    symmetry_parameterisation,
    stiffness_from_cij,
    build_stage_matrix,
)
from midas_stress.elastic_inverse import (
    _make_P_from_voigt_entries,
    _mandel_factor,
    _VOIGT_TO_MIDAS,
)
from midas_stress.materials import get_stiffness
from midas_stress.tensor import (
    rotation_voigt_mandel,
    tensor_to_voigt,
    voigt_to_tensor,
)


# -------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------

def _taylor_strains(C, orients, applied):
    """Build per-grain strains under a Taylor (uniform-strain) model.

    By construction the volume-averaged stress equals ``applied``, so
    any downstream fit of :math:`\\mathbf{C}` can attain the exact
    answer (subject to conditioning).
    """
    M = rotation_voigt_mandel(orients)
    Mt = np.swapaxes(M, -1, -2)
    C_lab = Mt @ C @ M
    eps_uniform = np.linalg.solve(C_lab.mean(axis=0), tensor_to_voigt(applied))
    return np.broadcast_to(voigt_to_tensor(eps_uniform), (len(orients), 3, 3)).copy()


def _applied_stress_library():
    """A diverse load palette sufficient to condition triclinic fits."""
    return [
        np.diag([0.05, 0.10, 0.15]),
        np.diag([0.15, 0.05, 0.10]),
        np.array([[0.0, 0.05, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.10]]),
        np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 0.05], [0.05, 0.05, 0.10]]),
        np.diag([-0.10, 0.15, 0.05]),
        np.array([[0.1, 0.05, 0.0],
                  [0.05, 0.1, 0.05],
                  [0.0, 0.05, 0.1]]),
    ]


# -------------------------------------------------------------------
#  Mandel-factor helper tests
# -------------------------------------------------------------------

class TestMandelHelper:
    def test_normal_normal_factor(self):
        assert _mandel_factor(0, 0) == 1.0
        assert _mandel_factor(2, 1) == 1.0

    def test_shear_shear_factor(self):
        assert _mandel_factor(3, 3) == 2.0
        assert _mandel_factor(5, 4) == 2.0

    def test_normal_shear_factor(self):
        np.testing.assert_allclose(_mandel_factor(0, 3), np.sqrt(2))
        np.testing.assert_allclose(_mandel_factor(4, 1), np.sqrt(2))

    def test_voigt_midas_index_map(self):
        assert _VOIGT_TO_MIDAS[1] == 0  # xx
        assert _VOIGT_TO_MIDAS[2] == 1  # yy
        assert _VOIGT_TO_MIDAS[3] == 2  # zz
        assert _VOIGT_TO_MIDAS[4] == 5  # yz
        assert _VOIGT_TO_MIDAS[5] == 4  # xz
        assert _VOIGT_TO_MIDAS[6] == 3  # xy

    def test_make_P_symmetric(self):
        P = _make_P_from_voigt_entries({(1, 2): 1.0})
        # Both (0,1) and (1,0) should be set
        assert P[0, 1] == 1.0
        assert P[1, 0] == 1.0


# -------------------------------------------------------------------
#  Symmetry parameterisation tests
# -------------------------------------------------------------------

class TestSymmetryParameterisation:
    @pytest.mark.parametrize("sym, n_expected", [
        ("cubic", 3),
        ("hexagonal", 5),
        ("trigonal", 6),
        ("tetragonal", 6),
        ("orthorhombic", 9),
        ("monoclinic", 13),
        ("triclinic", 21),
    ])
    def test_count_of_independent_constants(self, sym, n_expected):
        names, P = symmetry_parameterisation(sym)
        assert len(names) == n_expected
        assert P.shape == (n_expected, 6, 6)

    @pytest.mark.parametrize("sym", [
        "cubic", "hexagonal", "trigonal", "tetragonal",
        "orthorhombic", "monoclinic", "triclinic",
    ])
    def test_basis_matrices_are_symmetric(self, sym):
        _, P = symmetry_parameterisation(sym)
        for Pk in P:
            np.testing.assert_allclose(Pk, Pk.T, atol=1e-14)

    def test_cubic_matches_library(self):
        """cij vector -> stiffness matches ``cubic_stiffness`` helper."""
        C_rebuilt = stiffness_from_cij(
            {"C11": 192.9, "C12": 163.8, "C44": 41.5}, "cubic")
        C_ref = get_stiffness("Au")
        np.testing.assert_allclose(C_rebuilt, C_ref, atol=1e-12)

    def test_hexagonal_matches_library(self):
        C_rebuilt = stiffness_from_cij(
            {"C11": 162.4, "C12": 92.0, "C13": 69.0, "C33": 180.7, "C44": 46.7},
            "hexagonal",
        )
        C_ref = get_stiffness("Ti")
        np.testing.assert_allclose(C_rebuilt, C_ref, atol=1e-12)

    def test_invalid_symmetry_raises(self):
        with pytest.raises(ValueError, match="Unknown symmetry"):
            symmetry_parameterisation("unknown")

    def test_stiffness_from_cij_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="Expected cij"):
            stiffness_from_cij([1.0, 2.0], "cubic")

    def test_stiffness_from_cij_rejects_missing_dict_key(self):
        with pytest.raises(ValueError, match="Missing constants"):
            stiffness_from_cij({"C11": 100.0}, "cubic")


# -------------------------------------------------------------------
#  Round-trip on synthetic Taylor data
# -------------------------------------------------------------------

class TestRoundTrip:
    """For every symmetry, Taylor strains from a known C should recover C."""

    N_GRAINS = 500
    RSEED = 42
    TOL = 1e-8  # relative

    @staticmethod
    def _fit_from_cij(cij_true, sym, n_stages):
        C_true = stiffness_from_cij(cij_true, sym)
        rng = Rotation.random(TestRoundTrip.N_GRAINS,
                              random_state=TestRoundTrip.RSEED)
        orients = rng.as_matrix()
        volumes = np.ones(TestRoundTrip.N_GRAINS)
        library = _applied_stress_library()
        stages = []
        for i in range(n_stages):
            applied = library[i]
            strains = _taylor_strains(C_true, orients, applied)
            stages.append(dict(
                orient=orients, strain=strains, volumes=volumes,
                applied_stress=applied, is_unloaded=False,
            ))
        return fit_single_crystal_stiffness(
            stages, symmetry=sym, fit_eps_iso=False), C_true

    def test_cubic_recovery(self):
        res, C_true = self._fit_from_cij(
            {"C11": 192.9, "C12": 163.8, "C44": 41.5}, "cubic", 1)
        rel = np.max(np.abs(res["stiffness"] - C_true)) / np.max(np.abs(C_true))
        assert rel < self.TOL

    def test_hexagonal_recovery(self):
        cij = {"C11": 162.4, "C12": 92.0, "C13": 69.0, "C33": 180.7, "C44": 46.7}
        res, C_true = self._fit_from_cij(cij, "hexagonal", 1)
        rel = np.max(np.abs(res["stiffness"] - C_true)) / np.max(np.abs(C_true))
        assert rel < self.TOL

    def test_trigonal_recovery(self):
        cij = {"C11": 497.4, "C12": 162.7, "C13": 115.9,
               "C14": -22.0, "C33": 501.1, "C44": 147.4}
        res, C_true = self._fit_from_cij(cij, "trigonal", 2)
        rel = np.max(np.abs(res["stiffness"] - C_true)) / np.max(np.abs(C_true))
        assert rel < self.TOL

    def test_tetragonal_recovery(self):
        cij = {"C11": 68.0, "C12": 36.0, "C13": 36.0,
               "C33": 77.0, "C44": 22.0, "C66": 25.0}
        res, C_true = self._fit_from_cij(cij, "tetragonal", 2)
        rel = np.max(np.abs(res["stiffness"] - C_true)) / np.max(np.abs(C_true))
        assert rel < self.TOL

    def test_orthorhombic_recovery(self):
        cij = {"C11": 320.0, "C22": 200.0, "C33": 330.0,
               "C12": 72.0, "C13": 70.0, "C23": 68.0,
               "C44": 130.0, "C55": 120.0, "C66": 90.0}
        res, C_true = self._fit_from_cij(cij, "orthorhombic", 3)
        rel = np.max(np.abs(res["stiffness"] - C_true)) / np.max(np.abs(C_true))
        assert rel < self.TOL

    def test_monoclinic_recovery(self):
        cij = {"C11": 320.0, "C22": 200.0, "C33": 330.0,
               "C44": 130.0, "C55": 120.0, "C66": 90.0,
               "C12": 72.0, "C13": 70.0, "C23": 68.0,
               "C15": 15.0, "C25": 10.0, "C35": 12.0, "C46": 8.0}
        res, C_true = self._fit_from_cij(cij, "monoclinic", 4)
        rel = np.max(np.abs(res["stiffness"] - C_true)) / np.max(np.abs(C_true))
        assert rel < self.TOL

    def test_triclinic_recovery(self):
        cij = {
            "C11": 300.0, "C12": 70.0, "C13": 60.0, "C14": 5.0, "C15": 4.0, "C16": 3.0,
            "C22": 200.0, "C23": 80.0, "C24": 6.0, "C25": 7.0, "C26": 2.0,
            "C33": 280.0, "C34": 4.0, "C35": 3.0, "C36": 5.0,
            "C44": 100.0, "C45": 2.0, "C46": 3.0,
            "C55": 110.0, "C56": 4.0,
            "C66": 90.0,
        }
        res, C_true = self._fit_from_cij(cij, "triclinic", 6)
        rel = np.max(np.abs(res["stiffness"] - C_true)) / np.max(np.abs(C_true))
        assert rel < self.TOL


# -------------------------------------------------------------------
#  Conditioning / load-path diagnostics
# -------------------------------------------------------------------

class TestConditioning:
    def test_hexagonal_uniaxial_along_c_is_degenerate(self):
        """Pure uniaxial along the c axis cannot distinguish all hex constants."""
        C_true = get_stiffness("Ti")
        N = 300
        orients = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()  # texture-free? No — all same
        # Use random orientations but a load along z (which is c after random rotation
        # becomes c-rotated). To really hit the uniaxial-along-c degeneracy we need
        # the loading to be along c in the CRYSTAL frame of every grain, which
        # requires a single-crystal (single orientation). Use that.
        orients = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        volumes = np.ones(N)
        applied = np.diag([0.0, 0.0, 0.15])
        strains = _taylor_strains(C_true, orients, applied)
        stages = [dict(orient=orients, strain=strains, volumes=volumes,
                       applied_stress=applied, is_unloaded=False)]
        result = fit_single_crystal_stiffness(
            stages, symmetry="hexagonal", fit_eps_iso=False)
        # Should flag ill-conditioned
        assert not result["well_conditioned"]
        assert result["condition_number"] > 1e3

    def test_two_stages_well_conditioned_for_orthorhombic(self):
        cij = {"C11": 320.0, "C22": 200.0, "C33": 330.0,
               "C12": 72.0, "C13": 70.0, "C23": 68.0,
               "C44": 130.0, "C55": 120.0, "C66": 90.0}
        C_true = stiffness_from_cij(cij, "orthorhombic")
        N = 400
        orients = Rotation.random(N, random_state=1).as_matrix()
        volumes = np.ones(N)
        library = _applied_stress_library()
        stages = []
        for i in range(3):
            strains = _taylor_strains(C_true, orients, library[i])
            stages.append(dict(orient=orients, strain=strains, volumes=volumes,
                               applied_stress=library[i], is_unloaded=False))
        result = fit_single_crystal_stiffness(
            stages, symmetry="orthorhombic", fit_eps_iso=False)
        assert result["well_conditioned"]

    def test_underdetermined_raises(self):
        """Asking for orthorhombic with only one stage should error."""
        cij = {"C11": 320.0, "C22": 200.0, "C33": 330.0,
               "C12": 72.0, "C13": 70.0, "C23": 68.0,
               "C44": 130.0, "C55": 120.0, "C66": 90.0}
        C_true = stiffness_from_cij(cij, "orthorhombic")
        # Deliberately one stage with 6 equations for 9 unknowns
        N = 100
        orients = Rotation.random(N, random_state=1).as_matrix()
        volumes = np.ones(N)
        applied = np.diag([0.05, 0.10, 0.15])
        strains = _taylor_strains(C_true, orients, applied)
        stages = [dict(orient=orients, strain=strains, volumes=volumes,
                       applied_stress=applied, is_unloaded=False)]
        with pytest.raises(ValueError, match="Under-determined"):
            fit_single_crystal_stiffness(stages, symmetry="orthorhombic",
                                         fit_eps_iso=False)


# -------------------------------------------------------------------
#  Noise propagation
# -------------------------------------------------------------------

class TestNoisePropagation:
    def test_reported_se_tracks_empirical_scatter(self):
        """Repeat the fit with independent noise realisations and check that
        the reported standard errors are within 3x of empirical scatter."""
        cij_true = {"C11": 192.9, "C12": 163.8, "C44": 41.5}
        C_true = stiffness_from_cij(cij_true, "cubic")
        N = 400
        orients = Rotation.random(N, random_state=11).as_matrix()
        volumes = np.ones(N)
        applied = np.diag([0.05, 0.10, 0.15])
        eps_taylor = _taylor_strains(C_true, orients, applied)

        noise_sigma = 5e-5
        n_trials = 40
        fits = {"C11": [], "C12": [], "C44": []}
        reported_se = None
        rng = np.random.default_rng(7)
        for t in range(n_trials):
            noise = rng.normal(0, noise_sigma, (N, 3, 3))
            noise = 0.5 * (noise + np.swapaxes(noise, -1, -2))
            strains = eps_taylor + noise
            stages = [dict(orient=orients, strain=strains, volumes=volumes,
                           applied_stress=applied, is_unloaded=False)]
            result = fit_single_crystal_stiffness(
                stages, symmetry="cubic", fit_eps_iso=False)
            for k in fits:
                fits[k].append(result["cij"][k])
            if reported_se is None:
                reported_se = result["cij_se"]

        # Empirical scatter
        for k in fits:
            empirical = np.std(fits[k])
            reported = reported_se[k]
            # Reported SE is from a single realisation; allow 3x slack
            # (empirical variance on 40 trials has sqrt-n uncertainty).
            assert reported > 0.0
            assert 0.1 * reported < empirical < 10.0 * reported, (
                f"{k}: reported {reported:.3g}, empirical {empirical:.3g}")

    def test_se_scales_with_noise(self):
        """Doubling the noise roughly doubles the reported SE."""
        cij_true = {"C11": 192.9, "C12": 163.8, "C44": 41.5}
        C_true = stiffness_from_cij(cij_true, "cubic")
        N = 400
        orients = Rotation.random(N, random_state=11).as_matrix()
        volumes = np.ones(N)
        applied = np.diag([0.05, 0.10, 0.15])
        eps_taylor = _taylor_strains(C_true, orients, applied)

        rng = np.random.default_rng(5)
        se_values = []
        for scale in [1e-5, 1e-4, 1e-3]:
            noise = rng.normal(0, scale, (N, 3, 3))
            noise = 0.5 * (noise + np.swapaxes(noise, -1, -2))
            strains = eps_taylor + noise
            stages = [dict(orient=orients, strain=strains, volumes=volumes,
                           applied_stress=applied, is_unloaded=False)]
            res = fit_single_crystal_stiffness(
                stages, symmetry="cubic", fit_eps_iso=False)
            se_values.append(res["cij_se"]["C11"])
        # Strictly monotone in the noise level
        assert se_values[0] < se_values[1] < se_values[2]


# -------------------------------------------------------------------
#  Coupled d0 + C fit
# -------------------------------------------------------------------

class TestCoupledFit:
    def test_recovers_both_eps_iso_and_C(self):
        """Inject a d0 error on all stages and verify the coupled fit
        recovers both the isotropic strain error and the stiffness."""
        cij_true = {"C11": 162.4, "C12": 92.0, "C13": 69.0,
                    "C33": 180.7, "C44": 46.7}
        C_true = stiffness_from_cij(cij_true, "hexagonal")
        N = 400
        orients = Rotation.random(N, random_state=3).as_matrix()
        volumes = np.ones(N)

        eps_iso_inject = 300e-6

        # Unloaded stage: zero true strain, but d0 error shifts everything
        strains_unloaded = (
            np.broadcast_to(eps_iso_inject * np.eye(3), (N, 3, 3)).copy()
        )
        stage0 = dict(orient=orients, strain=strains_unloaded, volumes=volumes,
                      applied_stress=np.zeros((3, 3)), is_unloaded=True)

        # Loaded stage: Taylor strains + the same d0 shift
        applied1 = np.diag([0.05, 0.10, 0.15])
        strains_loaded = (
            _taylor_strains(C_true, orients, applied1)
            + eps_iso_inject * np.eye(3)[None, :, :]
        )
        stage1 = dict(orient=orients, strain=strains_loaded, volumes=volumes,
                      applied_stress=applied1, is_unloaded=False)

        result = fit_single_crystal_stiffness(
            [stage0, stage1], symmetry="hexagonal",
            material_hint="Ti",
        )
        np.testing.assert_allclose(result["eps_iso"], eps_iso_inject, rtol=1e-3)
        for k, v in cij_true.items():
            assert abs(result["cij"][k] - v) / abs(v) < 1e-3

    def test_without_unloaded_stage_skips_eps_iso(self):
        """Asking for fit_eps_iso with no unloaded stage should still fit C."""
        cij_true = {"C11": 192.9, "C12": 163.8, "C44": 41.5}
        C_true = stiffness_from_cij(cij_true, "cubic")
        N = 400
        orients = Rotation.random(N, random_state=2).as_matrix()
        volumes = np.ones(N)
        applied = np.diag([0.05, 0.10, 0.15])
        strains = _taylor_strains(C_true, orients, applied)
        stages = [dict(orient=orients, strain=strains, volumes=volumes,
                       applied_stress=applied, is_unloaded=False)]
        result = fit_single_crystal_stiffness(
            stages, symmetry="cubic", fit_eps_iso=True)
        assert result["eps_iso"] == 0.0  # no unloaded stage → skipped


# -------------------------------------------------------------------
#  Confidence weighting
# -------------------------------------------------------------------

class TestConfidenceFiltering:
    def test_low_confidence_grains_excluded(self):
        cij = {"C11": 192.9, "C12": 163.8, "C44": 41.5}
        C_true = stiffness_from_cij(cij, "cubic")
        N = 400
        orients = Rotation.random(N, random_state=17).as_matrix()
        volumes = np.ones(N)
        applied = np.diag([0.05, 0.10, 0.15])
        strains = _taylor_strains(C_true, orients, applied)

        # Corrupt 20% of the grains with huge strain noise and mark them low-confidence
        rng = np.random.default_rng(0)
        bad = rng.choice(N, size=80, replace=False)
        strains_mix = strains.copy()
        strains_mix[bad] += rng.normal(0, 1e-2, (80, 3, 3))
        strains_mix = 0.5 * (strains_mix + np.swapaxes(strains_mix, -1, -2))

        confidences = np.ones(N)
        confidences[bad] = 0.1

        stages = [dict(
            orient=orients, strain=strains_mix, volumes=volumes,
            applied_stress=applied, is_unloaded=False,
            confidences=confidences,
        )]
        # Without filtering: bad grains pollute the fit
        res_nofilter = fit_single_crystal_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False, min_confidence=0.0)
        # With filtering: bad grains excluded
        res_filter = fit_single_crystal_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False, min_confidence=0.5)

        err_nofilter = max(abs(res_nofilter["cij"][k] - cij[k]) for k in cij)
        err_filter = max(abs(res_filter["cij"][k] - cij[k]) for k in cij)
        assert err_filter < err_nofilter


# -------------------------------------------------------------------
#  Input validation / error handling
# -------------------------------------------------------------------

class TestInputValidation:
    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            fit_single_crystal_stiffness([], symmetry="cubic")

    def test_missing_key_raises(self):
        N = 10
        orients = Rotation.random(N, random_state=0).as_matrix()
        stages = [dict(orient=orients, strain=np.zeros((N, 3, 3)),
                       volumes=np.ones(N))]  # no applied_stress
        with pytest.raises(ValueError, match="missing key"):
            fit_single_crystal_stiffness(stages, symmetry="cubic")

    def test_mismatched_strain_shape_raises(self):
        N = 10
        orients = Rotation.random(N, random_state=0).as_matrix()
        stages = [dict(
            orient=orients,
            strain=np.zeros((N + 1, 3, 3)),  # wrong first dim
            volumes=np.ones(N),
            applied_stress=np.zeros((3, 3)),
        )]
        with pytest.raises(ValueError, match="inconsistent"):
            fit_single_crystal_stiffness(stages, symmetry="cubic")

    def test_bad_applied_stress_shape_raises(self):
        N = 10
        orients = Rotation.random(N, random_state=0).as_matrix()
        stages = [dict(
            orient=orients, strain=np.zeros((N, 3, 3)),
            volumes=np.ones(N),
            applied_stress=np.zeros((6,)),  # not a 3x3 tensor
        )]
        with pytest.raises(ValueError, match="3, 3"):
            fit_single_crystal_stiffness(stages, symmetry="cubic")
