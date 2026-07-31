"""A refiner that returns its input must never look like a successful fit.

Found chasing FF grain positions ~158 µm off the C reference
(`FitPosOrStrainsOMP`) on the 1-ID GE5 Au3 scan, 2026-07-30. Refining the
IDENTICAL seeds from the C indexer's ``IndexBest.bin``:

    float64 (cpu or cuda): moved 149.7 µm, median |Δpos| vs C = 13.4 µm
    float32 (cpu or cuda): moved   0.0 µm — 20/20 grains, exactly

fp32 did not refine position *at all*; it emitted the seed. Mechanism
(reproduced by ``test_fp32_position_is_left_on_the_seed`` below):
``torch.optim.LBFGS``'s strong-Wolfe line search takes one improving step and
then returns t = 0 forever, so the loss repeats bit-for-bit, the ``ftol``
counter reaches 8, and ``minimize_lbfgs`` truthfully reports that the loss
stopped changing — which the caller reads as success.

Two separable defects, and only the second is fixable here:

  * the numerical one (fp32 cannot drive this line search) — handled by
    defaulting FF refinement to float64, ``midas_pipeline.config.
    RefinementConfig.dtype``;
  * the SILENCE — nothing downstream could tell that a grain was never
    refined. That is what these tests pin, via ``BlockFitResult.
    max_position_move_um``, judged against a scale that carries units: the
    detector pixel. A block in which not one grain's position moved by even a
    pixel-equivalent did no useful work, whatever ``converged`` says.
"""

from __future__ import annotations

import math

import torch

from midas_fit_grain import FitConfig, refine_block
from midas_fit_grain.solvers.lbfgs import minimize_lbfgs

from ._synthetic import fixture_to_observed, gt_match, make_synthetic

DEG2RAD = math.pi / 180.0
SEED_OFFSET_UM = (90.0, -60.0, 110.0)      # |offset| ~154 µm, a real seed error


def _cfg(fix):
    return FitConfig(
        Lsd=fix.model.Lsd, px=fix.px, Wavelength=0.1729,
        LatticeConstant=tuple(fix.gt_lattice.tolist()), SpaceGroup=225,
        RingNumbers=fix.ring_numbers, RingRadii=[1.0] * len(fix.ring_numbers),
        OmegaRanges=[(-180.0, 180.0)],
        BoxSizes=[(-1e6, 1e6, -1e6, 1e6)],
        MarginEta=5.0, MarginOme=2.0, EtaBinSize=2.0, OmeBinSize=2.0,
        MinEta=6.0,
        solver="lbfgs", mode="all_at_once", loss="full3d",
        max_iter=200, ftol=1e-5, xtol=1e-5,
        phase_steps=(8, 8, 8, 8),
        Rsample=2000.0, Hbeam=2000.0,
    )


def _refine(dtype):
    """Refine one grain from a seed displaced ~154 µm from ground truth."""
    dev = torch.device("cpu")
    fix = make_synthetic(device=dev, dtype=dtype)
    obs = fixture_to_observed(fix, device=dev, dtype=dtype)
    match = gt_match(fix, device=dev, dtype=dtype)
    init_pos = fix.gt_position.clone() + torch.tensor(SEED_OFFSET_UM,
                                                      dtype=dtype)
    blk = refine_block(
        _cfg(fix), model=fix.model, grains_obs=[obs],
        init_positions=init_pos.view(1, 3),
        init_eulers=(fix.gt_euler.clone() + 0.05 * DEG2RAD).view(1, 3),
        init_lattices=fix.gt_lattice.clone().view(1, 6),
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_matches=[match],
    )
    g = blk.grains[0]
    return dict(
        blk=blk, floor=_cfg(fix).px / 1000.0,
        moved=float((g.position.double() - init_pos.double()).norm()),
        err=float((g.position.double() - fix.gt_position.double()).norm()),
        seed_err=float((init_pos.double() - fix.gt_position.double()).norm()),
    )


# ── the control: fp64 genuinely refines ──────────────────────────────────

def test_float64_refines_the_position_and_is_not_flagged():
    r = _refine(torch.float64)
    assert r["seed_err"] > 100.0, "fixture no longer displaces the seed"
    assert r["moved"] > 0.5 * r["seed_err"], (
        f"fp64 moved only {r['moved']:.2f} µm from a seed "
        f"{r['seed_err']:.2f} µm off GT"
    )
    assert r["err"] < 1.0, f"fp64 missed ground truth by {r['err']:.2f} µm"
    assert r["blk"].n_unmoved_position == 0
    assert r["blk"].n_grains == 1
    # Well clear of the driver's px/1000 "did nothing" floor. NB a good fit
    # can legitimately move less than ONE pixel (this one moves 0.77 px), so
    # the floor has to sit far below a pixel.
    assert r["blk"].max_position_move_um > r["floor"], (
        f"fp64 moved {r['blk'].max_position_move_um:.4g} µm, under the "
        f"{r['floor']:.4g} µm floor — the driver would flag a good fit"
    )


# ── the failure: fp32 leaves the seed, and SAYS SO ───────────────────────

def test_fp32_leaves_the_position_essentially_on_the_seed_and_is_flagged():
    """Documents the numerical failure. It is allowed to stay broken (the
    dtype default protects production) — it is not allowed to be silent.

    fp32 does not necessarily return the seed BIT-identically; measured here
    it creeps ~5e-4 µm (2.5e-06 px) while the seed is 154 µm off ground
    truth. So the guard is measurable movement, not bit equality."""
    r = _refine(torch.float32)
    if r["moved"] > 0.5 * r["seed_err"]:
        # fp32 started working; then it must not be flagged either.
        assert r["blk"].max_position_move_um > r["floor"]
        return
    assert r["err"] > 0.5 * r["seed_err"], "no movement but the error shrank?"
    assert r["blk"].max_position_move_um < r["floor"], (
        f"fp32 moved {r['blk'].max_position_move_um:.4g} µm; the driver only "
        f"warns below {r['floor']:.4g} µm, so this failure would ship "
        f"silently again"
    )
    assert r["blk"].n_grains == 1


def test_movement_stats_are_measured_against_the_seed():
    """max/median must describe distance from the SEED, not from zero."""
    dev = torch.device("cpu")
    fix = make_synthetic(device=dev, dtype=torch.float64)
    obs = fixture_to_observed(fix, device=dev, dtype=torch.float64)
    match = gt_match(fix, device=dev, dtype=torch.float64)
    init_pos = fix.gt_position.clone()          # seed exactly at GT
    blk = refine_block(
        _cfg(fix), model=fix.model, grains_obs=[obs],
        init_positions=init_pos.view(1, 3),
        init_eulers=fix.gt_euler.clone().view(1, 3),
        init_lattices=fix.gt_lattice.clone().view(1, 6),
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_matches=[match],
    )
    ref = float((blk.grains[0].position.double() - init_pos.double()).norm())
    assert abs(blk.max_position_move_um - ref) < 1e-9
    assert abs(blk.median_position_move_um - ref) < 1e-9   # B == 1


# ── solver diagnostics are reported, and never acted on ──────────────────

def test_solver_reports_frozen_steps_and_gradients():
    p = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def closure():
        if p.grad is not None:
            p.grad = None
        loss = (p ** 2).sum() + 1.0
        loss.backward()
        return loss

    res = minimize_lbfgs(closure, [p], max_iter=5)
    for key in ("frozen_steps", "grad_inf", "grad_inf_0", "converged",
                "history", "n_iter", "final_loss"):
        assert key in res, f"missing {key} in solver result"
    assert isinstance(res["frozen_steps"], int)


def test_entry_gradient_is_measured_before_any_step():
    """``grad_inf_0`` must be the gradient at the seed. Capturing it after the
    first step (an earlier attempt) makes it useless as a reference and
    false-positived on fits whose seed already sat near the optimum."""
    p = torch.full((1,), 3.0, dtype=torch.float64, requires_grad=True)

    def closure():
        if p.grad is not None:
            p.grad = None
        loss = (p ** 2).sum()
        loss.backward()
        return loss

    res = minimize_lbfgs(closure, [p], max_iter=20)
    # d(p²)/dp at p=3 is 6; anything else means we sampled after moving.
    assert abs(res["grad_inf_0"] - 6.0) < 1e-9, res["grad_inf_0"]
    # And the solve should still work.
    assert abs(float(p.detach())) < 1e-6
