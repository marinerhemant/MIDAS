"""Solver comparison tests on the synthetic fixture.

Spreads each solver across the 3 loss families and 2 modes; checks that
each solver makes meaningful progress and ends with a small loss. Tighter
solver-specific tolerances live here rather than in
``test_refine_grain.py`` so the cross-solver matrix doesn't blow up the
already-slow test_refine_grain runtime.
"""

from __future__ import annotations

import math

import pytest
import torch

from midas_fit_grain import FitConfig, refine_grain

from ._synthetic import fixture_to_observed, gt_match, make_synthetic

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


@pytest.fixture(scope="module")
def fix():
    return make_synthetic(device=torch.device("cpu"), dtype=torch.float64)


def _build_cfg(fix, *, solver, loss, mode="all_at_once",
               max_iter=None, ftol=1e-12, xtol=1e-12):
    # Nelder-Mead is derivative-free → much slower; needs more iters.
    if max_iter is None:
        max_iter = 5000 if solver == "nelder_mead" else 200
    return FitConfig(
        Lsd=fix.model.Lsd, px=fix.px, Wavelength=0.1729,
        LatticeConstant=tuple(fix.gt_lattice.tolist()),
        SpaceGroup=225,
        RingNumbers=fix.ring_numbers,
        RingRadii=[1.0] * len(fix.ring_numbers),
        OmegaRanges=[(-180.0, 180.0)],
        BoxSizes=[(-1e6, 1e6, -1e6, 1e6)],
        MarginEta=5.0, MarginOme=2.0,
        EtaBinSize=2.0, OmeBinSize=2.0,
        MinEta=6.0,
        solver=solver, mode=mode, loss=loss,
        max_iter=max_iter, ftol=ftol, xtol=xtol,
        phase_steps=(8, 8, 8, 8),
    )


def _misori_deg(eul_a, eul_b):
    from midas_diffract import HEDMForwardModel
    Ra = HEDMForwardModel.euler2mat(eul_a)
    Rb = HEDMForwardModel.euler2mat(eul_b)
    trace = (Ra.T @ Rb).diagonal().sum()
    cos_ang = ((trace - 1) / 2).clamp(-1.0, 1.0)
    return float(torch.acos(cos_ang)) * RAD2DEG


# Solver × loss matrix (all_at_once, GT match precomputed).
@pytest.mark.parametrize("solver", ["lbfgs", "lm", "nelder_mead"])
@pytest.mark.parametrize("loss", ["full3d", "angular", "internal_angle"])
def test_solver_makes_progress(fix, solver, loss):
    """Every (solver, loss) pair must drop the loss by ≥1000× from a tight
    seed, and the resulting orientation must be within 0.05° on a tight
    seed (or 0.5° for nelder_mead which converges slower)."""
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = _build_cfg(fix, solver=solver, loss=loss)

    init_pos = fix.gt_position.clone() + torch.tensor([0.5, -0.3, 0.2],
                                                      dtype=torch.float64)
    init_eul = fix.gt_euler.clone() + 0.02 * DEG2RAD
    init_lat = fix.gt_lattice.clone()
    match_seed = gt_match(fix, device=torch.device("cpu"),
                          dtype=torch.float64)

    res = refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot, precomputed_match=match_seed,
    )

    initial_loss = res.history[0]
    final_loss = res.history[-1]
    assert final_loss < initial_loss * 1e-3 or final_loss < 1e-6, (
        f"{solver}/{loss}: loss {initial_loss:.3g} -> {final_loss:.3g} "
        f"(want >1000x or <1e-6)"
    )

    # Orientation tolerance per (solver, loss).
    if loss == "pixel":
        # phi1 conditioning issue on the cubic synthetic.
        tol = 0.06
    elif solver == "nelder_mead":
        tol = 0.5
    else:
        tol = 0.005

    mis_deg = _misori_deg(res.euler, fix.gt_euler)
    assert mis_deg < tol, (
        f"{solver}/{loss}: misori = {mis_deg:.4f}° (want <{tol}°)"
    )


@pytest.mark.parametrize("solver", ["lbfgs", "lm"])
def test_lm_handles_phi1_conditioning(fix, solver):
    """LM should outperform L-BFGS on the phi1-poorly-conditioned axis.

    With the same seed, LM (Marquardt-damped Gauss-Newton) should converge
    closer to GT than L-BFGS on this synthetic. The C reference uses
    Nelder-Mead which sits between the two.
    """
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = _build_cfg(fix, solver=solver, loss="full3d", mode="all_at_once",
                     max_iter=300, ftol=1e-14, xtol=1e-14)

    init_pos = fix.gt_position.clone() + torch.tensor([1.0, -0.5, 0.3],
                                                      dtype=torch.float64)
    init_eul = fix.gt_euler.clone() + 0.05 * DEG2RAD
    init_lat = fix.gt_lattice.clone()
    match_seed = gt_match(fix, device=torch.device("cpu"),
                          dtype=torch.float64)

    res = refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot, precomputed_match=match_seed,
    )
    mis_deg = _misori_deg(res.euler, fix.gt_euler)
    pos_err = (res.position - fix.gt_position).norm().item()

    # Both must recover position well.
    assert pos_err < 0.05, (
        f"{solver}: pos err {pos_err:.3f}um after refinement"
    )
    # LM should reach the smooth basin and recover phi1; L-BFGS may not.
    if solver == "lm":
        assert mis_deg < 0.005, f"LM misori {mis_deg:.4f}°"
    else:
        assert mis_deg < 0.06, f"LBFGS misori {mis_deg:.4f}°"
