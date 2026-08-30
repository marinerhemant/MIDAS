"""A fit with no matched spots must not report success.

The mask in ``grain_residuals`` zeroes every row of the residual when nothing
matches, so ``(res*res).sum()`` came out as ``0.0`` -- the LOWEST attainable
loss. A total failure therefore outranked every real grain in any best-of-N or
loss sort, while also reporting ``converged=True`` and returning its own seed.

There is nothing to restore here: with no matched spots there are no
observations, hence no gradient and no information. These tests pin the
*contract* instead -- failure is legible and sorts last.
"""

from __future__ import annotations

import math

import pytest
import torch

from midas_fit_grain import FitConfig, MatchResult, refine_grain

from ._synthetic import fixture_to_observed, gt_match, make_synthetic

DEG2RAD = math.pi / 180.0


def _empty_match(fix, obs) -> MatchResult:
    """An all-False association: the state a seed too far off the truth lands in."""
    m = gt_match(fix, device=torch.device("cpu"), dtype=torch.float64)
    return MatchResult(
        k_idx=m.k_idx, m_idx=m.m_idx,
        mask=torch.zeros_like(m.mask, dtype=torch.bool),
        delta_omega=m.delta_omega, delta_eta=m.delta_eta,
    )


def _cfg(fix, **kw):
    base = dict(
        Lsd=fix.model.Lsd, px=fix.px, Wavelength=0.1729,
        LatticeConstant=tuple(fix.gt_lattice.tolist()), SpaceGroup=225,
        RingNumbers=fix.ring_numbers, RingRadii=[1.0] * len(fix.ring_numbers),
        OmegaRanges=[(-180.0, 180.0)],
        BoxSizes=[(-1e6, 1e6, -1e6, 1e6)],
        MarginEta=5.0, MarginOme=0.5, EtaBinSize=2.0, OmeBinSize=2.0,
        MinEta=6.0, solver="lm", mode="all_at_once", loss="full3d",
        max_iter=200, ftol=1e-12, xtol=1e-12, phase_steps=(8, 8, 8, 8),
    )
    base.update(kw)
    return FitConfig(**base)


def _run(seed_err_deg, *, empty_match=False, **cfgkw):
    dev, dt = torch.device("cpu"), torch.float64
    fix = make_synthetic(device=dev, dtype=dt)
    obs = fixture_to_observed(fix, device=dev, dtype=dt)
    cfg = _cfg(fix, **cfgkw)
    init_eul = fix.gt_euler.clone() + seed_err_deg * DEG2RAD
    return fix, refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=fix.gt_position.clone(),
        init_euler=init_eul,
        init_lattice=fix.gt_lattice.clone(),
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_match=_empty_match(fix, obs) if empty_match else None,
    ), init_eul


def test_no_match_loss_is_inf_not_zero():
    """The headline bug: an unmatched fit must not score better than a good one."""
    _, bad, seed = _run(0.0, empty_match=True)
    assert bad.n_matched == 0
    assert bad.final_loss == float("inf"), (
        f"unmatched fit reported final_loss={bad.final_loss!r}; 0.0 would make "
        f"it beat every real grain in a loss sort"
    )
    assert not bad.converged, "a fit with no matched spots cannot have converged"


def test_no_match_returns_the_seed_unmoved():
    """Not a defect -- with no data there is nothing to move toward. Pinned so
    the contract is explicit: the caller gets its own seed back, clearly
    labelled as a failure rather than as a converged fit."""
    fix, bad, seed = _run(0.0, empty_match=True)
    assert torch.allclose(bad.euler, seed, atol=1e-12)
    assert not bad.converged


def test_failure_never_wins_a_best_of_n():
    """The operational consequence, stated as the sort a caller would write."""
    _, good, _ = _run(0.05)
    _, bad, _ = _run(0.0, empty_match=True)
    assert good.n_matched > 0 and bad.n_matched == 0
    best = min([good, bad], key=lambda r: r.final_loss)
    assert best is good, "the failed grain won the best-of-N"


def test_good_fit_still_reports_a_finite_loss():
    """The guard must not swallow real fits."""
    _, good, _ = _run(0.05)
    assert good.n_matched > 0
    assert math.isfinite(good.final_loss)
    assert good.final_loss >= 0.0


def test_zero_margin_raises_rather_than_matching_nothing():
    """FitConfig defaults MarginOme/MarginEta to 0.0, which empties the mask."""
    dev, dt = torch.device("cpu"), torch.float64
    fix = make_synthetic(device=dev, dtype=dt)
    obs = fixture_to_observed(fix, device=dev, dtype=dt)
    for kw in ({"MarginOme": 0.0}, {"MarginEta": 0.0}):
        with pytest.raises(ValueError, match="must both be > 0"):
            refine_grain(
                _cfg(fix, **kw), model=fix.model, obs=obs,
                init_position=fix.gt_position.clone(),
                init_euler=fix.gt_euler.clone(),
                init_lattice=fix.gt_lattice.clone(),
                pred_ring_slot=fix.pred_ring_slot,
            )


def test_empty_ring_radii_raises_rather_than_treating_um_as_deg():
    """500 µm * DEG2RAD = 8.727 rad -- four orders too large, silently."""
    dev, dt = torch.device("cpu"), torch.float64
    fix = make_synthetic(device=dev, dtype=dt)
    obs = fixture_to_observed(fix, device=dev, dtype=dt)
    with pytest.raises(ValueError, match="RingRadii is empty"):
        refine_grain(
            _cfg(fix, RingRadii=[]), model=fix.model, obs=obs,
            init_position=fix.gt_position.clone(),
            init_euler=fix.gt_euler.clone(),
            init_lattice=fix.gt_lattice.clone(),
            pred_ring_slot=fix.pred_ring_slot,
        )
