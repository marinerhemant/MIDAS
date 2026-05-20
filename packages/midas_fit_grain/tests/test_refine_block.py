"""Multi-grain batched refinement tests.

Builds a synthetic block with several grains at different orientations,
runs ``refine_block``, checks that each grain recovers its ground truth.
"""

from __future__ import annotations

import math

import pytest
import torch

from midas_fit_grain import (
    FitConfig, GrainFitResult, MatchResult, refine_block,
)
from midas_fit_grain.batch import MatchBatch

from ._synthetic import fixture_to_observed, gt_match, make_synthetic

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def _misori_deg(eul_a, eul_b):
    from midas_diffract import HEDMForwardModel
    Ra = HEDMForwardModel.euler2mat(eul_a)
    Rb = HEDMForwardModel.euler2mat(eul_b)
    trace = (Ra.T @ Rb).diagonal().sum()
    cos_ang = ((trace - 1) / 2).clamp(-1.0, 1.0)
    return float(torch.acos(cos_ang)) * RAD2DEG


def _build_cfg(fix, *, solver, mode="all_at_once", loss="full3d",
               max_iter=200, ftol=1e-12, xtol=1e-12):
    return FitConfig(
        Lsd=fix.model.Lsd, px=fix.px, Wavelength=0.1729,
        LatticeConstant=tuple(fix.gt_lattice.tolist()), SpaceGroup=225,
        RingNumbers=fix.ring_numbers, RingRadii=[1.0] * len(fix.ring_numbers),
        OmegaRanges=[(-180.0, 180.0)],
        BoxSizes=[(-1e6, 1e6, -1e6, 1e6)],
        MarginEta=5.0, MarginOme=2.0, EtaBinSize=2.0, OmeBinSize=2.0,
        MinEta=6.0,
        solver=solver, mode=mode, loss=loss,
        max_iter=max_iter, ftol=ftol, xtol=xtol,
        phase_steps=(8, 8, 8, 8),
    )


def test_refine_block_single_grain_matches_refine_grain():
    """A B=1 block should give the same answer as refine_grain on its own."""
    from midas_fit_grain import refine_grain
    fix = make_synthetic(device=torch.device("cpu"), dtype=torch.float64)
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = _build_cfg(fix, solver="lm")
    init_pos = fix.gt_position.clone() + torch.tensor(
        [1.0, -0.5, 0.3], dtype=torch.float64
    )
    init_eul = fix.gt_euler.clone() + 0.05 * DEG2RAD
    init_lat = fix.gt_lattice.clone()
    match_seed = gt_match(fix, device=torch.device("cpu"), dtype=torch.float64)

    block = refine_block(
        cfg, model=fix.model,
        grains_obs=[obs],
        init_positions=init_pos.view(1, 3),
        init_eulers=init_eul.view(1, 3),
        init_lattices=init_lat.view(1, 6),
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_matches=[match_seed],
    )
    single = refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot, precomputed_match=match_seed,
    )

    assert len(block.grains) == 1
    g = block.grains[0]
    # LM is deterministic; should land in the same numerical neighborhood.
    assert (g.position - single.position).norm().item() < 1e-6
    assert (g.euler - single.euler).norm().item() < 1e-6
    assert (g.lattice - single.lattice).norm().item() < 1e-6


def test_refine_block_three_grains_lm():
    """Three grains at different orientations, all recovered together by LM."""
    fix = make_synthetic(device=torch.device("cpu"), dtype=torch.float64)
    base_obs = fixture_to_observed(fix, device=torch.device("cpu"),
                                   dtype=torch.float64)

    # We reuse the synthetic spots as B=3 identical-content grains, each
    # given a different perturbed seed. The ground truth is the same; what
    # we test is that the batched optimizer drives all 3 to GT.
    grains = [base_obs] * 3
    matches = [gt_match(fix, device=torch.device("cpu"),
                        dtype=torch.float64)] * 3
    init_pos = torch.stack([
        fix.gt_position + torch.tensor([1.0, -0.5, 0.3], dtype=torch.float64),
        fix.gt_position + torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        fix.gt_position + torch.tensor([-0.5, 0.7, -0.2], dtype=torch.float64),
    ])
    init_eul = torch.stack([
        fix.gt_euler + 0.05 * DEG2RAD,
        fix.gt_euler.clone(),
        fix.gt_euler - 0.03 * DEG2RAD,
    ])
    init_lat = torch.stack([fix.gt_lattice] * 3)

    cfg = _build_cfg(fix, solver="lm")
    block = refine_block(
        cfg, model=fix.model,
        grains_obs=grains,
        init_positions=init_pos,
        init_eulers=init_eul,
        init_lattices=init_lat,
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_matches=matches,
    )
    assert len(block.grains) == 3
    for b, g in enumerate(block.grains):
        mis = _misori_deg(g.euler, fix.gt_euler)
        pos_err = (g.position - fix.gt_position).norm().item()
        # LM is the strict-tolerance solver — should drive every grain
        # to the synthetic's machine-precision basin.
        assert mis < 0.01, f"grain {b} misori {mis:.5f}°"
        assert pos_err < 0.01, f"grain {b} pos_err {pos_err:.5f} um"


def test_refine_block_lbfgs_three_grains():
    """L-BFGS on a 3-grain block should also converge (slightly looser tol)."""
    fix = make_synthetic(device=torch.device("cpu"), dtype=torch.float64)
    base_obs = fixture_to_observed(fix, device=torch.device("cpu"),
                                   dtype=torch.float64)
    matches = [gt_match(fix, device=torch.device("cpu"),
                        dtype=torch.float64)] * 3
    grains = [base_obs] * 3
    init_pos = torch.stack([fix.gt_position + 0.5,
                            fix.gt_position - 0.3,
                            fix.gt_position + 0.2])
    init_eul = torch.stack([fix.gt_euler + 0.02 * DEG2RAD,
                            fix.gt_euler.clone(),
                            fix.gt_euler - 0.01 * DEG2RAD])
    init_lat = torch.stack([fix.gt_lattice] * 3)

    cfg = _build_cfg(fix, solver="lbfgs", max_iter=400)
    block = refine_block(
        cfg, model=fix.model, grains_obs=grains,
        init_positions=init_pos, init_eulers=init_eul, init_lattices=init_lat,
        pred_ring_slot=fix.pred_ring_slot, precomputed_matches=matches,
    )
    assert block.converged
    for g in block.grains:
        pos_err = (g.position - fix.gt_position).norm().item()
        assert pos_err < 0.1
