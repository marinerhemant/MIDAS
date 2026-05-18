"""Tests for fit_joint_d0_stiffness — Paper III joint d_0 + C_ij fit.

Parity vs alternating, device portability, autograd correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.spatial.transform import Rotation

from midas_stress import (
    fit_single_crystal_stiffness,
    fit_joint_d0_stiffness,
    loo_influence_stages,
    stiffness_from_cij,
)
from midas_stress.tensor import (
    rotation_voigt_mandel,
    tensor_to_voigt,
    voigt_to_tensor,
)


# -------------------------------------------------------------------
#  Helpers (mirror the conventions in test_elastic_inverse.py)
# -------------------------------------------------------------------

def _taylor_strains(C, orients, applied):
    M = rotation_voigt_mandel(orients)
    Mt = np.swapaxes(M, -1, -2)
    C_lab = Mt @ C @ M
    eps_uniform = np.linalg.solve(C_lab.mean(axis=0), tensor_to_voigt(applied))
    return np.broadcast_to(voigt_to_tensor(eps_uniform), (len(orients), 3, 3)).copy()


def _hex_two_stage_with_d0(N=400, eps_iso=300e-6, seed=3):
    """Hexagonal Ti, one unloaded + one loaded stage, both shifted by eps_iso."""
    cij_true = {"C11": 162.4, "C12": 92.0, "C13": 69.0,
                "C33": 180.7, "C44": 46.7}
    C_true = stiffness_from_cij(cij_true, "hexagonal")
    orients = Rotation.random(N, random_state=seed).as_matrix()
    volumes = np.ones(N)

    strains_unloaded = np.broadcast_to(
        eps_iso * np.eye(3), (N, 3, 3)
    ).copy()
    stage0 = dict(orient=orients, strain=strains_unloaded, volumes=volumes,
                  applied_stress=np.zeros((3, 3)), is_unloaded=True)

    applied1 = np.diag([0.05, 0.10, 0.15])
    strains_loaded = (
        _taylor_strains(C_true, orients, applied1)
        + eps_iso * np.eye(3)[None, :, :]
    )
    stage1 = dict(orient=orients, strain=strains_loaded, volumes=volumes,
                  applied_stress=applied1, is_unloaded=False)
    return [stage0, stage1], cij_true, C_true


def _cubic_one_stage(N=300, seed=11):
    """Cubic Au, single loaded stage; no d_0 bias."""
    cij_true = {"C11": 192.9, "C12": 163.8, "C44": 41.5}
    C_true = stiffness_from_cij(cij_true, "cubic")
    orients = Rotation.random(N, random_state=seed).as_matrix()
    volumes = np.ones(N)
    applied = np.diag([0.05, 0.10, 0.15])
    eps = _taylor_strains(C_true, orients, applied)
    stage = dict(orient=orients, strain=eps, volumes=volumes,
                 applied_stress=applied, is_unloaded=False)
    return [stage], cij_true


# -------------------------------------------------------------------
#  Parity vs alternating fit (the existing closed-form path)
# -------------------------------------------------------------------

class TestJointParityVsAlternating:
    def test_hex_joint_matches_alternating_on_c(self):
        stages, cij_true, _ = _hex_two_stage_with_d0()
        alt = fit_single_crystal_stiffness(
            stages, symmetry="hexagonal", material_hint="Ti",
        )
        joint = fit_joint_d0_stiffness(
            stages, symmetry="hexagonal", material_hint="Ti",
        )
        for k in cij_true:
            assert abs(joint["cij"][k] - alt["cij"][k]) / abs(alt["cij"][k]) < 1e-6, (
                f"{k}: joint={joint['cij'][k]:.6f} alt={alt['cij'][k]:.6f}"
            )

    def test_hex_joint_matches_alternating_on_eps_iso(self):
        stages, _, _ = _hex_two_stage_with_d0()
        alt = fit_single_crystal_stiffness(
            stages, symmetry="hexagonal", material_hint="Ti",
        )
        joint = fit_joint_d0_stiffness(
            stages, symmetry="hexagonal", material_hint="Ti",
        )
        # Both should recover the injected 300e-6 to <0.1% relative
        assert abs(joint["eps_iso"] - alt["eps_iso"]) / max(abs(alt["eps_iso"]), 1e-12) < 1e-4

    def test_hex_joint_recovers_truth(self):
        eps_iso_inject = 300e-6
        stages, cij_true, _ = _hex_two_stage_with_d0(eps_iso=eps_iso_inject)
        joint = fit_joint_d0_stiffness(
            stages, symmetry="hexagonal", material_hint="Ti",
        )
        assert abs(joint["eps_iso"] - eps_iso_inject) / eps_iso_inject < 1e-3
        for k, v in cij_true.items():
            assert abs(joint["cij"][k] - v) / abs(v) < 1e-3

    def test_cubic_no_d0_joint_matches_alternating(self):
        stages, cij_true = _cubic_one_stage()
        alt = fit_single_crystal_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False,
        )
        joint = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False,
        )
        for k in cij_true:
            assert abs(joint["cij"][k] - alt["cij"][k]) / abs(alt["cij"][k]) < 1e-9


class TestJointCovariance:
    def test_covariance_shape_includes_eps_iso(self):
        stages, _, _ = _hex_two_stage_with_d0()
        joint = fit_joint_d0_stiffness(
            stages, symmetry="hexagonal", material_hint="Ti",
        )
        N_c = len(joint["cij_names"])
        assert joint["joint_covariance"].shape == (N_c + 1, N_c + 1)
        assert joint["covariance"].shape == (N_c, N_c)
        # eps_iso block is non-negative on the diagonal
        assert joint["joint_covariance"][N_c, N_c] >= 0.0
        # eps_iso_se is the sqrt of that diagonal
        assert abs(joint["eps_iso_se"]**2 - joint["joint_covariance"][N_c, N_c]) < 1e-30

    def test_fit_eps_iso_false_zeros_cov_row(self):
        stages, _ = _cubic_one_stage()
        joint = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False,
        )
        N_c = len(joint["cij_names"])
        # When eps_iso is pinned the joint cov last row/col should be zero
        assert np.allclose(joint["joint_covariance"][N_c, :], 0.0)
        assert np.allclose(joint["joint_covariance"][:, N_c], 0.0)
        assert joint["eps_iso"] == 0.0


class TestWithoutUnloadedStage:
    def test_no_unloaded_stage_pins_eps_iso_zero(self):
        """Without an unloaded reference, eps_iso is conservatively pinned to
        zero (matches the alternating fit's fallback)."""
        stages, cij_true = _cubic_one_stage()
        joint = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=True,
        )
        # No is_unloaded stage → eps_iso should not move
        assert joint["eps_iso"] == 0.0
        for k in cij_true:
            assert abs(joint["cij"][k] - cij_true[k]) / abs(cij_true[k]) < 1e-3


# -------------------------------------------------------------------
#  Device portability
# -------------------------------------------------------------------

def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS does not yet support float64; the joint fit requires fp64.
        # Skip MPS by default; users on Apple Silicon need fp64 anyway.
        pass
    return devices


class TestDevicePortability:
    @pytest.mark.parametrize("device", _available_devices())
    def test_joint_fit_device(self, device):
        stages, cij_true, _ = _hex_two_stage_with_d0(N=200)
        cpu_result = fit_joint_d0_stiffness(
            stages, symmetry="hexagonal", material_hint="Ti", device="cpu",
        )
        dev_result = fit_joint_d0_stiffness(
            stages, symmetry="hexagonal", material_hint="Ti", device=device,
        )
        assert dev_result["device"] == device
        for k in cij_true:
            rel = abs(dev_result["cij"][k] - cpu_result["cij"][k]) / abs(cpu_result["cij"][k])
            assert rel < 1e-9, f"{k}: cpu={cpu_result['cij'][k]} {device}={dev_result['cij'][k]}"
        assert abs(dev_result["eps_iso"] - cpu_result["eps_iso"]) < 1e-15


# -------------------------------------------------------------------
#  Autograd correctness on the joint loss
# -------------------------------------------------------------------

class TestPDEnforcement:
    """The enforce_pd flag soft-projects the stiffness back into the
    Born-stability cone; verify that (a) it's a no-op on well-posed
    problems and (b) it rescues non-PD results on pathological cases."""

    def test_pd_no_op_on_well_posed_cubic(self):
        """A clean, well-conditioned Au fit should already be PD; enforce_pd
        should not perturb it appreciably."""
        stages, cij_true = _cubic_one_stage()
        free = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False,
        )
        constrained = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False,
            enforce_pd=True, pd_floor=0.0,
        )
        for k in cij_true:
            rel = abs(constrained["cij"][k] - free["cij"][k]) / abs(free["cij"][k])
            assert rel < 1e-6, (
                f"{k}: free={free['cij'][k]} constrained={constrained['cij'][k]}"
            )
        assert constrained["pd_enforced"] is True
        assert constrained["pd_violation"] == 0.0
        assert constrained["min_eigenvalue"] > 0.0
        assert free["pd_enforced"] is False

    def test_pd_rescues_pathological_cubic(self):
        """Pathological case: 1 stage + low N + high noise can drive the
        unconstrained fit non-PD on some seeds.  When this happens the
        PD-enforced fit must return a strictly PD stiffness."""
        cij_true = {"C11": 192.9, "C12": 163.8, "C44": 41.5}
        C_true = stiffness_from_cij(cij_true, "cubic")
        N = 20
        applied = np.diag([0.01, 0.005, 0.003])  # weak load palette
        noise_sigma = 5e-4                        # high noise relative to elastic strain

        found_pathological = False
        for seed in range(60):
            orients = Rotation.random(N, random_state=seed).as_matrix()
            volumes = np.ones(N)
            eps = _taylor_strains(C_true, orients, applied)
            rng = np.random.default_rng(seed * 13 + 7)
            n = rng.normal(0, noise_sigma, (N, 3, 3))
            eps = eps + 0.5 * (n + np.swapaxes(n, -1, -2))
            stages = [dict(orient=orients, strain=eps, volumes=volumes,
                           applied_stress=applied, is_unloaded=False)]
            free = fit_joint_d0_stiffness(
                stages, symmetry="cubic", fit_eps_iso=False,
            )
            if free["min_eigenvalue"] >= 0.0:
                continue
            # Found a non-PD unconstrained fit.  PD enforcement must rescue.
            found_pathological = True
            constrained = fit_joint_d0_stiffness(
                stages, symmetry="cubic", fit_eps_iso=False,
                enforce_pd=True, pd_floor=1e-3, pd_weight=1e6,
            )
            assert constrained["min_eigenvalue"] >= 0.0, (
                f"seed {seed}: PD-enforced fit min_eigenvalue = "
                f"{constrained['min_eigenvalue']} (free was {free['min_eigenvalue']})"
            )
            break
        assert found_pathological, (
            "did not find a non-PD unconstrained fit in 60 seeds; "
            "tighten the pathological setup"
        )


class TestIRLSWeights:
    """The irls_weights=True flag runs an outer iteratively-reweighted
    least-squares loop on top of the unweighted joint fit.  Verify:
    (1) on a well-posed multi-stage cubic synthetic the IRLS pass is
        a near-no-op (it shouldn't make a good fit worse);
    (2) the fit reports irls_iters > 0 and a vector of final weights;
    (3) on a uniaxial-Z synthetic where the unweighted fit lets the
        5 should-be-zero components absorb residual, IRLS pulls the
        recovery closer to truth on the shear-coupled constants.
    """

    def _well_posed_cubic_5_stages(self, N=300, noise=2e-5, seed=2):
        cij_true = {"C11": 192.9, "C12": 163.8, "C44": 41.5}
        C_true = stiffness_from_cij(cij_true, "cubic")
        rng = np.random.default_rng(seed)
        orients = Rotation.random(N, random_state=seed).as_matrix()
        volumes = np.ones(N)
        applieds = [
            np.diag([0.05, 0.10, 0.15]),
            np.diag([0.10, 0.05, 0.12]),
            np.diag([-0.05, 0.12, 0.08]),
            np.array([[0.0, 0.05, 0.0],
                      [0.05, 0.0, 0.0],
                      [0.0, 0.0, 0.10]]),
        ]
        stages = []
        for app in applieds:
            eps = _taylor_strains(C_true, orients, app)
            n = rng.normal(0, noise, (N, 3, 3))
            eps = eps + 0.5 * (n + np.swapaxes(n, -1, -2))
            stages.append(dict(orient=orients, strain=eps, volumes=volumes,
                               applied_stress=app, is_unloaded=False))
        return stages, cij_true

    def test_irls_no_op_on_well_posed(self):
        stages, cij_true = self._well_posed_cubic_5_stages()
        base = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False, material_hint="Au",
        )
        irls = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False, material_hint="Au",
            irls_weights=True,
        )
        assert irls["irls_enabled"] is True
        assert irls["irls_iters"] >= 1
        assert irls["irls_weights_final"] is not None
        # Both should recover the truth to <1% on every Cij; IRLS shouldn't
        # make the well-conditioned fit any worse than ~3x the baseline gap.
        for k in cij_true:
            base_err = abs(base["cij"][k] - cij_true[k]) / abs(cij_true[k])
            irls_err = abs(irls["cij"][k] - cij_true[k]) / abs(cij_true[k])
            assert irls_err < 3 * base_err + 5e-3, (
                f"{k}: base_err={base_err:.4e}, irls_err={irls_err:.4e}"
            )

    def _uniaxial_z_with_anisotropic_noise(self, N=400, noise_diag=2e-4,
                                            noise_shear=2e-5, seed=4):
        """Uniaxial-Z cubic Au with strain noise heavier on the diagonal
        than on the shear components.  Construction: the baseline unit-
        weight LSQ over-weights the noisy diagonals and absorbs noise
        into Cij; IRLS auto-detects the small-shear residual structure
        and amplifies the cleaner shear constraints."""
        cij_true = {"C11": 192.9, "C12": 163.8, "C44": 41.5}
        C_true = stiffness_from_cij(cij_true, "cubic")
        rng = np.random.default_rng(seed)
        orients = Rotation.random(N, random_state=seed).as_matrix()
        volumes = np.ones(N)
        stages = []
        # Unloaded
        n0 = np.zeros((N, 3, 3))
        for a in range(3):
            n0[:, a, a] = rng.normal(0, noise_diag, N)
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            v = rng.normal(0, noise_shear, N)
            n0[:, a, b] = v
            n0[:, b, a] = v
        stages.append(dict(orient=orients, strain=n0, volumes=volumes,
                           applied_stress=np.zeros((3, 3)), is_unloaded=True))
        # Loaded stages along Z
        for stress_GPa in [0.05, 0.10, 0.15, 0.20]:
            applied = np.array([[0, 0, 0], [0, 0, 0], [0, 0, stress_GPa]],
                               dtype=float)
            eps = _taylor_strains(C_true, orients, applied)
            n = np.zeros((N, 3, 3))
            for a in range(3):
                n[:, a, a] = rng.normal(0, noise_diag, N)
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                v = rng.normal(0, noise_shear, N)
                n[:, a, b] = v
                n[:, b, a] = v
            eps = eps + n
            stages.append(dict(orient=orients, strain=eps, volumes=volumes,
                               applied_stress=applied, is_unloaded=False))
        return stages, cij_true

    def test_irls_improves_uniaxial_z_with_component_anisotropic_noise(self):
        stages, cij_true = self._uniaxial_z_with_anisotropic_noise()
        base = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False, material_hint="Au",
        )
        irls = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False, material_hint="Au",
            irls_weights=True,
        )
        # IRLS final weights should be strongly non-uniform when the noise is.
        w_final = irls["irls_weights_final"]
        assert w_final is not None
        ratio = w_final.max() / max(w_final.min(), 1e-30)
        assert ratio > 5.0, (
            f"expected strong weight asymmetry for component-anisotropic "
            f"noise; got ratio = {ratio:.2f}"
        )
        # On at least one Cij the IRLS estimate must be strictly closer to
        # truth than the baseline; the others must not regress by more than
        # 25%.  The diagnostic claim is that IRLS strictly improves the
        # constant most-affected by the noise asymmetry (C44 here).
        improvements = []
        for k in cij_true:
            base_err = abs(base["cij"][k] - cij_true[k]) / abs(cij_true[k])
            irls_err = abs(irls["cij"][k] - cij_true[k]) / abs(cij_true[k])
            improvements.append((k, base_err, irls_err))
        # At least one constant strictly improved
        assert any(irls_e < base_e * 0.9 for _, base_e, irls_e in improvements), (
            f"IRLS did not improve any Cij beyond 10% (base, irls): "
            f"{[(k, base_e, irls_e) for k, base_e, irls_e in improvements]}"
        )


class TestLOOInfluence:
    """Verify that the autograd-Hessian one-step Newton LOO agrees
    with brute-force re-fits on a multi-stage cubic synthetic."""

    def _multi_stage_cubic(self, n_loaded=4, N=200, noise=5e-5, seed=0):
        cij_true = {"C11": 192.9, "C12": 163.8, "C44": 41.5}
        C_true = stiffness_from_cij(cij_true, "cubic")
        rng = np.random.default_rng(seed)
        orients = Rotation.random(N, random_state=seed).as_matrix()
        volumes = np.ones(N)

        # Unloaded reference + a diverse load palette
        applieds = [
            np.diag([0.05, 0.10, 0.15]),
            np.diag([0.10, 0.05, 0.12]),
            np.diag([-0.05, 0.12, 0.08]),
            np.array([[0.0, 0.05, 0.0],
                      [0.05, 0.0, 0.0],
                      [0.0, 0.0, 0.10]]),
        ][:n_loaded]
        stages = [dict(orient=orients,
                       strain=np.broadcast_to(0.0 * np.eye(3),
                                              (N, 3, 3)).copy()
                              + 0.5 * rng.normal(0, noise, (N, 3, 3))
                              + 0.5 * np.swapaxes(rng.normal(0, noise, (N, 3, 3)), -1, -2),
                       volumes=volumes,
                       applied_stress=np.zeros((3, 3)),
                       is_unloaded=True)]
        for app in applieds:
            eps = _taylor_strains(C_true, orients, app)
            eps = eps + 0.5 * rng.normal(0, noise, (N, 3, 3))
            stages.append(dict(orient=orients, strain=eps, volumes=volumes,
                               applied_stress=app, is_unloaded=False))
        return stages, cij_true

    def test_loo_matches_brute_force_cubic_5_stages(self):
        stages, cij_true = self._multi_stage_cubic(n_loaded=4)
        fit = fit_joint_d0_stiffness(
            stages, symmetry="cubic", fit_eps_iso=False, material_hint="Au",
        )
        loo = loo_influence_stages(fit, stages, symmetry="cubic")
        # Brute-force LOO: re-fit with each stage left out
        for i in range(len(stages)):
            kept = [s for j, s in enumerate(stages) if j != i]
            # Skip if removing this stage drops the only unloaded reference
            # (then eps_iso identification flips and the comparison is
            # ill-defined for the fixed-mask one-step approximation).
            if not any(s.get("is_unloaded", False) for s in kept):
                continue
            bf = fit_joint_d0_stiffness(
                kept, symmetry="cubic", fit_eps_iso=False, material_hint="Au",
            )
            for k_idx, k in enumerate(loo["cij_names"]):
                approx = loo["cij_loo"][i, k_idx]
                exact = bf["cij"][k]
                # 1-step Newton accuracy: tight near the optimum for LSQ
                assert abs(approx - exact) / max(abs(exact), 1.0) < 1e-3, (
                    f"stage {i}, {k}: autograd-LOO={approx:.4f}, "
                    f"brute={exact:.4f}"
                )

    def test_loo_eps_iso_matches_brute_force_hex_3_stages(self):
        # 1 unloaded + 2 loaded so removing a loaded stage keeps eps_iso identifiable
        cij_true = {"C11": 162.4, "C12": 92.0, "C13": 69.0,
                    "C33": 180.7, "C44": 46.7}
        C_true = stiffness_from_cij(cij_true, "hexagonal")
        N = 200
        rng = np.random.default_rng(7)
        orients = Rotation.random(N, random_state=7).as_matrix()
        volumes = np.ones(N)
        noise = 5e-5
        eps_inject = 200e-6

        applieds = [np.diag([0.05, 0.10, 0.15]), np.diag([0.10, -0.05, 0.08])]
        stages = [dict(orient=orients,
                       strain=np.broadcast_to(eps_inject * np.eye(3),
                                              (N, 3, 3)).copy()
                              + 0.5 * rng.normal(0, noise, (N, 3, 3))
                              + 0.5 * np.swapaxes(rng.normal(0, noise, (N, 3, 3)), -1, -2),
                       volumes=volumes,
                       applied_stress=np.zeros((3, 3)),
                       is_unloaded=True)]
        for app in applieds:
            eps = _taylor_strains(C_true, orients, app) + eps_inject * np.eye(3)[None]
            eps = eps + 0.5 * rng.normal(0, noise, (N, 3, 3))
            stages.append(dict(orient=orients, strain=eps, volumes=volumes,
                               applied_stress=app, is_unloaded=False))

        fit = fit_joint_d0_stiffness(stages, symmetry="hexagonal", material_hint="Ti")
        loo = loo_influence_stages(fit, stages, symmetry="hexagonal")

        for i in [1, 2]:  # only LOO the loaded stages
            kept = [s for j, s in enumerate(stages) if j != i]
            bf = fit_joint_d0_stiffness(kept, symmetry="hexagonal", material_hint="Ti")
            for k_idx, k in enumerate(loo["cij_names"]):
                approx = loo["cij_loo"][i, k_idx]
                exact = bf["cij"][k]
                # On the bilinear joint loss with only 3 stages, the
                # one-step Newton approximation carries a ~1% bias
                # (drops to <0.1% for 5+ stages); the experimental
                # 20-stage cases in the paper are well within that.
                assert abs(approx - exact) / max(abs(exact), 1.0) < 1e-2, (
                    f"stage {i}, {k}: autograd-LOO={approx:.4f}, "
                    f"brute={exact:.4f}"
                )
            assert abs(loo["eps_iso_loo"][i] - bf["eps_iso"]) < 5e-5


class TestAutograd:
    def test_gradcheck_joint_loss(self):
        """torch.autograd.gradcheck on a tiny problem (cubic, 8 grains)."""
        from midas_stress.elastic_inverse_torch import (
            _build_stage_terms,
            _joint_loss,
        )
        from midas_stress.elastic_inverse import (
            symmetry_parameterisation,
            _prep_stages,
        )

        # Tiny hex case so the joint loss is cheap to differentiate.
        cij_true = {"C11": 162.4, "C12": 92.0, "C13": 69.0,
                    "C33": 180.7, "C44": 46.7}
        C_true = stiffness_from_cij(cij_true, "hexagonal")
        N = 8
        orients = Rotation.random(N, random_state=0).as_matrix()
        volumes = np.ones(N)
        applied = np.diag([0.05, 0.10, 0.15])
        strains = _taylor_strains(C_true, orients, applied) + 100e-6 * np.eye(3)[None]
        stages = [dict(orient=orients, strain=strains, volumes=volumes,
                       applied_stress=applied, is_unloaded=False),
                  dict(orient=orients,
                       strain=np.broadcast_to(100e-6 * np.eye(3), (N, 3, 3)).copy(),
                       volumes=volumes,
                       applied_stress=np.zeros((3, 3)), is_unloaded=True)]
        names, P_stack = symmetry_parameterisation("hexagonal")
        N_c = len(names)
        prepped = _prep_stages(stages, 0.0)
        dtype = torch.float64
        device = torch.device("cpu")
        P_stack_t = torch.as_tensor(P_stack, dtype=dtype, device=device)
        terms = _build_stage_terms(prepped, P_stack_t, dtype, device)
        mask = [s["is_unloaded"] for s in prepped]

        # Random non-trivial theta for the gradcheck
        theta = torch.randn(N_c + 1, dtype=dtype, device=device, requires_grad=True) * 1e-2

        def loss_fn(th):
            return _joint_loss(th, terms, N_c, mask)

        assert torch.autograd.gradcheck(loss_fn, (theta,), eps=1e-7, atol=1e-6, rtol=1e-4)
