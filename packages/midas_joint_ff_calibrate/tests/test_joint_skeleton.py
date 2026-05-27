"""Minimal joint-loss tests using a 2D toy problem.

Two synthetic modalities sharing one geometry parameter ``Lsd``:
  - Modality A constrains ``Lsd`` and ``BC_y`` (proxy for powder)
  - Modality B constrains ``BC_y`` and ``BC_z`` (proxy for HEDM eta info)

Either modality alone is rank-deficient on the ``Lsd``+``BC`` triplet;
joint is full-rank.  This is the same identifiability story as paper-4
in miniature.
"""
from __future__ import annotations

import torch

import midas_peakfit as mp
import midas_joint_ff_calibrate as mjf
from midas_joint_ff_calibrate.loss import JointWeights, joint_residual
from midas_joint_ff_calibrate.pipelines.identifiability import fisher_block_rank
from midas_joint_ff_calibrate.pipelines.alternating import AlternatingDriver
from midas_joint_ff_calibrate.pipelines.full_joint import FullJointDriver


def _make_problem():
    """Build a tiny ParameterSpec + two residual closures."""
    torch.manual_seed(42)
    spec = mp.ParameterSpec()
    spec.add(mp.Parameter("Lsd", init=900.0, bounds=(800.0, 1100.0)))
    spec.add(mp.Parameter("BC_y", init=510.0, bounds=(500.0, 530.0)))
    spec.add(mp.Parameter("BC_z", init=510.0, bounds=(500.0, 530.0)))
    # HEDM-like nuisance (single grain, just ``orient``):
    spec.add(mp.Parameter("grain_lattice", init=torch.tensor([[5.4, 5.4, 5.4, 90, 90, 90]],
                                                              dtype=torch.float64),
                          refined=False))

    Lsd_true, BCy_true, BCz_true = 1000.0, 520.0, 515.0

    # Modality A: 50 measurements that linearly constrain (Lsd, BC_y).
    # Form r_a[i] = a_i * Lsd + b_i * BC_y - c_i, fit at the true values to ~0.
    a_a = torch.linspace(0.5, 1.5, 50, dtype=torch.float64)
    b_a = torch.linspace(0.2, 0.8, 50, dtype=torch.float64)
    c_a = a_a * Lsd_true + b_a * BCy_true + 1e-3 * torch.randn(50, dtype=torch.float64)

    # Modality B: 50 measurements that constrain (BC_y, BC_z).
    a_b = torch.linspace(0.3, 1.0, 50, dtype=torch.float64)
    b_b = torch.linspace(0.7, 1.4, 50, dtype=torch.float64)
    c_b = a_b * BCy_true + b_b * BCz_true + 1e-3 * torch.randn(50, dtype=torch.float64)

    def residual_a(unpacked):
        return a_a * unpacked["Lsd"] + b_a * unpacked["BC_y"] - c_a

    def residual_b(unpacked):
        return a_b * unpacked["BC_y"] + b_b * unpacked["BC_z"] - c_b

    return spec, residual_a, residual_b, (Lsd_true, BCy_true, BCz_true)


def test_joint_residual_shape():
    spec, residual_a, residual_b, _ = _make_problem()
    x, info = mp.pack_spec(spec)
    unpacked = mp.unpack_spec(x, info, spec)
    r = joint_residual(
        unpacked,
        powder_residual_fn=residual_a,
        hedm_residual_fn=residual_b,
        spec=spec,
        weights=JointWeights(w_powder=1.0, w_hedm=1.0, lambda_gauge=1e6),
        gauge_blocks=[],   # no panel block in this toy problem
    )
    assert r.numel() == 100   # 50 + 50; no gauge / prior rows


def test_joint_lm_recovers_truth():
    spec, residual_a, residual_b, truth = _make_problem()
    Lsd_true, BCy_true, BCz_true = truth

    def joint_fn(unpacked):
        return joint_residual(
            unpacked,
            powder_residual_fn=residual_a,
            hedm_residual_fn=residual_b,
            spec=spec,
            gauge_blocks=[],
        )

    unpacked, cost, rc = mp.lm_minimise(
        spec, joint_fn,
        config=mp.GenericLMConfig(max_iter=200, ftol_rel=1e-12),
        fallback_span=1.0,
    )
    assert rc == 0
    assert abs(float(unpacked["Lsd"]) - Lsd_true) < 0.1
    assert abs(float(unpacked["BC_y"]) - BCy_true) < 0.1
    assert abs(float(unpacked["BC_z"]) - BCz_true) < 0.1


def test_fisher_block_rank_powder_only_rank_deficient():
    """Modality A alone constrains (Lsd, BC_y) but NOT BC_z — so the
    Fisher block on {Lsd, BC_y, BC_z} should be rank 2 (not 3).
    """
    spec, residual_a, _, _ = _make_problem()
    # Set MAP at the truth so jacrev is accurate.
    map_unpacked = {
        "Lsd": torch.tensor(1000.0, dtype=torch.float64),
        "BC_y": torch.tensor(520.0, dtype=torch.float64),
        "BC_z": torch.tensor(515.0, dtype=torch.float64),
        "grain_lattice": spec.parameters["grain_lattice"].init,
    }
    rep = fisher_block_rank(
        spec, residual_a, map_unpacked,
        block_names=["Lsd", "BC_y", "BC_z"],
        sigma_r=1.0,
    )
    # BC_z is unconstrained by modality A → block is rank-deficient.
    assert rep.rank < 3, f"Expected rank<3, got {rep.rank} (cond={rep.condition_number:.2e})"


def test_fisher_block_rank_joint_full_rank():
    """The joint loss covers (Lsd, BC_y, BC_z) — Fisher block is full rank."""
    spec, residual_a, residual_b, _ = _make_problem()

    def joint_fn(unpacked):
        return joint_residual(
            unpacked,
            powder_residual_fn=residual_a,
            hedm_residual_fn=residual_b,
            spec=spec,
            gauge_blocks=[],
        )

    map_unpacked = {
        "Lsd": torch.tensor(1000.0, dtype=torch.float64),
        "BC_y": torch.tensor(520.0, dtype=torch.float64),
        "BC_z": torch.tensor(515.0, dtype=torch.float64),
        "grain_lattice": spec.parameters["grain_lattice"].init,
    }
    rep = fisher_block_rank(
        spec, joint_fn, map_unpacked,
        block_names=["Lsd", "BC_y", "BC_z"],
        sigma_r=1.0,
    )
    assert rep.rank == 3


def test_full_joint_driver_runs():
    spec, residual_a, residual_b, truth = _make_problem()
    Lsd_true, BCy_true, BCz_true = truth

    def joint_fn(unpacked):
        return joint_residual(
            unpacked,
            powder_residual_fn=residual_a,
            hedm_residual_fn=residual_b,
            spec=spec,
            gauge_blocks=[],
        )

    drv = FullJointDriver(
        spec=spec,
        residual_fn=joint_fn,
        lm_config=mp.GenericLMConfig(max_iter=100, ftol_rel=1e-10),
        sigma_r=1e-3,   # 1mε noise floor in modality A
        compute_laplace=True,
    )
    res = drv.run()
    assert res.rc == 0
    assert res.laplace is not None
    # σ on Lsd should be small (well-determined by modality A's 50 points).
    sig_Lsd = float(res.laplace.sigma_per_dim[
        res.laplace.refined_names.index("Lsd")
    ])
    assert sig_Lsd > 0
    assert sig_Lsd < 1.0   # micrometers — well below 1 mm
