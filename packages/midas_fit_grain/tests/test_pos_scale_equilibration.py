"""``pos_scale`` must equilibrate the gradient blocks, not be a fixed guess.

L-BFGS applies ONE step length to the concatenated (pos_scaled, euler,
lattice) vector, so a block whose gradient is far smaller than the others
barely moves. The historical fixed ``pos_scale = 100`` left the FF
orientation gradient ~1500x the position gradient:

    pos_scale   |g|pos    |g|euler   ratio   fp32 error vs truth
         1e2      95.8     1.47e5    1537            154.27 um
         1e3       958     1.47e5     154              0.75 um
         1e4      9581     1.47e5    15.4              0.013 um
         1e5    9.58e4     1.47e5     1.5              0.004 um

Accuracy tracks the ratio monotonically. In fp64 the tiny position increment
is still resolvable; in fp32 it lands under the gradient's own ~1e-4 relative
rounding noise, the line search finds no further descent, and the grain keeps
its SEED position — silently (1-ID GE5 Au3, 2026-07-30: 20/20 grains, ~158 um
off the C reference `FitPosOrStrainsOMP`).

``refine_block`` now derives the scale from the entry gradient:
``s = |g_other| / |g_pos_um|``, since ``d/d(pos_scaled) = s * d/d(pos)``.
"""

from __future__ import annotations

import math

import pytest
import torch

from midas_fit_grain import FitConfig, refine_block
from midas_fit_grain.batch import MatchBatch, ObservedBatch, batch_residuals
from midas_fit_grain.refine_block import (
    _POS_SCALE_MAX, _POS_SCALE_MIN, _equilibrated_pos_scale,
)

from ._synthetic import fixture_to_observed, gt_match, make_synthetic

DEG2RAD = math.pi / 180.0


def _cfg(fix, solver="lbfgs"):
    return FitConfig(
        Lsd=fix.model.Lsd, px=fix.px, Wavelength=0.1729,
        LatticeConstant=tuple(fix.gt_lattice.tolist()), SpaceGroup=225,
        RingNumbers=fix.ring_numbers, RingRadii=[1.0] * len(fix.ring_numbers),
        OmegaRanges=[(-180.0, 180.0)], BoxSizes=[(-1e6, 1e6, -1e6, 1e6)],
        MarginEta=5.0, MarginOme=2.0, EtaBinSize=2.0, OmeBinSize=2.0,
        MinEta=6.0, solver=solver, mode="all_at_once", loss="full3d",
        max_iter=200, ftol=1e-5, xtol=1e-5, phase_steps=(8, 8, 8, 8),
        Rsample=2000.0, Hbeam=2000.0,
    )


def _setup(dtype, offset):
    dev = torch.device("cpu")
    fix = make_synthetic(device=dev, dtype=dtype)
    obs1 = fixture_to_observed(fix, device=dev, dtype=dtype)
    m1 = gt_match(fix, device=dev, dtype=dtype)
    pos = (fix.gt_position.clone() + torch.tensor(offset, dtype=dtype)).view(1, 3)
    eul = (fix.gt_euler.clone() + 0.05 * DEG2RAD).view(1, 3)
    lat = fix.gt_lattice.clone().view(1, 6)
    return fix, obs1, m1, pos, eul, lat


def _refine(dtype, offset, **kw):
    fix, obs1, m1, pos, eul, lat = _setup(dtype, offset)
    blk = refine_block(
        _cfg(fix), model=fix.model, grains_obs=[obs1],
        init_positions=pos, init_eulers=eul, init_lattices=lat,
        pred_ring_slot=fix.pred_ring_slot, precomputed_matches=[m1], **kw
    )
    err = float((blk.grains[0].position.double() - fix.gt_position.double()).norm())
    return blk, err


# ── the scale really does balance the blocks ─────────────────────────────

def test_derived_scale_equalises_the_gradient_blocks():
    """The whole point: after rescaling, |g_pos| ~ |g_other|."""
    dev = torch.device("cpu")
    fix, obs1, m1, pos, eul, lat = _setup(torch.float64, (90.0, -60.0, 110.0))
    obs = ObservedBatch.pack([obs1], device=dev, dtype=torch.float64)
    match = MatchBatch.pack([m1], s_max=obs.s_max, device=dev)
    cfg = _cfg(fix)

    s = _equilibrated_pos_scale(
        model=fix.model, obs=obs, match=match, cfg=cfg,
        init_positions=pos, init_eulers=eul, init_lattices=lat,
    )
    assert s > 1e3, f"derived scale {s:.4g} is barely above the old default"

    # Re-measure the blocks in the rescaled parameterisation.
    ps = (pos / s).clone().requires_grad_(True)
    eu = eul.clone().requires_grad_(True)
    la = lat.clone().requires_grad_(True)
    res = batch_residuals(
        fix.model, grain_position=ps * s, grain_euler=eu, grain_lattice=la,
        obs=obs, match=match, kind=cfg.loss,
        px=cfg.px, y_BC=fix.model.y_BC, z_BC=fix.model.z_BC,
    )
    (res * res).sum().backward()
    g_pos = float(ps.grad.norm())
    g_other = max(float(eu.grad.norm()), float(la.grad.norm()))
    ratio = g_other / g_pos
    assert 0.2 < ratio < 5.0, (
        f"blocks still unbalanced after rescale: |g_other|/|g_pos| = {ratio:.1f} "
        f"(the fixed pos_scale=100 gave ~1537)"
    )


def test_derived_scale_is_clamped_to_a_sane_range():
    dev = torch.device("cpu")
    fix, obs1, m1, pos, eul, lat = _setup(torch.float64, (90.0, -60.0, 110.0))
    obs = ObservedBatch.pack([obs1], device=dev, dtype=torch.float64)
    match = MatchBatch.pack([m1], s_max=obs.s_max, device=dev)
    s = _equilibrated_pos_scale(
        model=fix.model, obs=obs, match=match, cfg=_cfg(fix),
        init_positions=pos, init_eulers=eul, init_lattices=lat,
    )
    assert _POS_SCALE_MIN <= s <= _POS_SCALE_MAX


def test_a_degenerate_gradient_falls_back_instead_of_exploding():
    """A seed at a position stationary point would give |g_pos| -> 0 and an
    absurd scale; it must fall back, not divide by ~zero."""
    class _Boom:
        def __getattr__(self, _name):
            raise RuntimeError("forward model unavailable")

    s = _equilibrated_pos_scale(
        model=_Boom(), obs=None, match=None, cfg=_cfg(
            make_synthetic(device=torch.device("cpu"), dtype=torch.float64)),
        init_positions=torch.zeros(1, 3, dtype=torch.float64),
        init_eulers=torch.zeros(1, 3, dtype=torch.float64),
        init_lattices=torch.ones(1, 6, dtype=torch.float64),
    )
    assert s == _POS_SCALE_MIN


# ── it fixes the actual failure ──────────────────────────────────────────

@pytest.mark.parametrize("offset", [
    (90.0, -60.0, 110.0),
    (-200.0, 150.0, -80.0),
    (15.0, -5.0, 25.0),
])
def test_fp32_recovers_the_position_with_auto_scaling(offset):
    """The headline: fp32 went from 154 / 68 / 30 um of error to sub-0.01 um."""
    _, err_fixed = _refine(torch.float32, offset, pos_scale=100.0)
    _, err_auto = _refine(torch.float32, offset)          # auto
    assert err_fixed > 5.0, (
        f"fixture no longer reproduces the fp32 failure ({err_fixed:.4g} um)"
    )
    assert err_auto < 0.5, f"auto scaling still off by {err_auto:.4g} um"
    assert err_auto < err_fixed / 50.0


@pytest.mark.parametrize("offset", [
    (90.0, -60.0, 110.0),
    (-200.0, 150.0, -80.0),
    (15.0, -5.0, 25.0),
])
def test_float64_is_not_made_worse(offset):
    """Auto scaling must not regress the precision that already worked — it
    happens to improve it, but 'no worse' is the contract."""
    _, err_fixed = _refine(torch.float64, offset, pos_scale=100.0)
    _, err_auto = _refine(torch.float64, offset)
    assert err_auto <= max(err_fixed, 0.01), (
        f"fp64 regressed: {err_fixed:.4g} um -> {err_auto:.4g} um"
    )


def test_explicit_pos_scale_still_honoured():
    """The escape hatch must not be silently overridden."""
    dev = torch.device("cpu")
    fix, obs1, m1, pos, eul, lat = _setup(torch.float64, (90.0, -60.0, 110.0))
    blk = refine_block(
        _cfg(fix), model=fix.model, grains_obs=[obs1],
        init_positions=pos, init_eulers=eul, init_lattices=lat,
        pred_ring_slot=fix.pred_ring_slot, precomputed_matches=[m1],
        pos_scale=100.0,
    )
    _, err_auto = _refine(torch.float64, (90.0, -60.0, 110.0))
    err_fixed = float(
        (blk.grains[0].position.double() - fix.gt_position.double()).norm())
    # Different scales give measurably different iterates; if they were
    # identical the explicit value was being ignored.
    assert err_fixed != err_auto


def test_position_bounds_survive_the_rescale():
    """``_clamp_pos_to_sample`` divides the µm bounds by pos_scale, so the
    clamp must still bite at the same PHYSICAL position for any scale."""
    fix, obs1, m1, _, eul, lat = _setup(torch.float64, (0.0, 0.0, 0.0))
    cfg = _cfg(fix)
    cfg.Rsample = 50.0        # µm — far inside the seed below
    cfg.Hbeam = 100.0
    far = torch.tensor([[5000.0, -5000.0, 5000.0]], dtype=torch.float64)
    blk = refine_block(
        cfg, model=fix.model, grains_obs=[obs1],
        init_positions=far, init_eulers=eul, init_lattices=lat,
        pred_ring_slot=fix.pred_ring_slot, precomputed_matches=[m1],
    )
    p = blk.grains[0].position.double().abs()
    assert float(p[0]) <= cfg.Rsample + 1e-6, p
    assert float(p[1]) <= cfg.Rsample + 1e-6, p
    assert float(p[2]) <= cfg.Hbeam / 2.0 + 1e-6, p
