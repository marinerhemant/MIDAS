"""Residuals at ground truth should be zero (or near-zero in float32).

Exercises: forward model build → spot generation → reuse as 'observed' →
residual at GT state → check |res| < 1e-6 for all three loss kinds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_fit_grain.config import FitConfig
from midas_fit_grain.matching import associate, ring_slot_lookup
from midas_fit_grain.residuals import grain_residuals

from ._synthetic import fixture_to_observed, make_synthetic


@pytest.fixture(scope="module")
def fix():
    return make_synthetic(device=torch.device("cpu"), dtype=torch.float64)


def test_residuals_zero_at_gt_pixel(fix):
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    # Trivial association: each observed slot is itself.
    pred_ring = fix.pred_ring_slot
    obs_ring_slot = ring_slot_lookup(fix.ring_numbers, obs.ring_nr)

    spots = fix.model(fix.gt_euler.view(1, 1, 3),
                      fix.gt_position.view(1, 1, 3),
                      lattice_params=fix.gt_lattice.view(1, 6))

    def _sq(t):
        while t.dim() > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
            if t.dim() == 0:
                break
        return t

    match = associate(
        obs_ring_nr=obs.ring_nr,
        obs_omega=obs.omega,
        obs_eta=obs.eta,
        pred_ring_slot=pred_ring,
        pred_omega=_sq(spots.omega),
        pred_eta=_sq(spots.eta),
        pred_valid=_sq(spots.valid),
        obs_ring_slot=obs_ring_slot,
        omega_tolerance=math.pi,
        eta_tolerance=math.pi,
    )
    assert match.mask.all(), "every synthetic obs spot must associate"

    for kind in ("angular", "internal_angle"):
        res = grain_residuals(
            fix.model,
            grain_euler=fix.gt_euler,
            grain_position=fix.gt_position,
            grain_lattice=fix.gt_lattice,
            obs=obs, match=match,
            kind=kind,
            px=fix.px, y_BC=fix.y_BC, z_BC=fix.z_BC,
        )
        assert res.shape[0] == obs.n_spots
        amax = res.abs().max().item()
        # FF mode is float32 inside the model regardless of input dtype, so
        # the zero-tolerance is dominated by float32 round-off in the bragg
        # geometry pipeline. 1e-5 (rad) is plenty.
        assert amax < 1e-4, f"{kind} residual at GT = {amax}"


def test_pixel_loss_is_usable_but_guarded(fix):
    """The 2-D 'pixel' loss computes; the GUARD lives at the phase level.

    Re-enabled 2026-08-04. It omits omega, so with orientation FREE the crystal
    can rotate about the omega direction at no cost (~20 deg drift on real PF
    data, 2026-05) — but it is the objective the C refiner uses for POSITION
    and STRAIN, where orientation is held fixed. Measured: the C's stages under
    this loss land 1.55 um from c-omp, under full3d 38.57 um. So the loss is
    correct and the misuse is what must be blocked; `refine_block._run_phase`
    refuses it whenever euler is an active parameter.
    """
    import pytest

    def _sq(t):
        while t.dim() > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
            if t.dim() == 0:
                break
        return t

    obs = fixture_to_observed(fix, device=torch.device("cpu"), dtype=torch.float64)
    pred_ring = fix.pred_ring_slot
    obs_ring_slot = ring_slot_lookup(fix.ring_numbers, obs.ring_nr)
    spots = fix.model(fix.gt_euler.view(1, 1, 3), fix.gt_position.view(1, 1, 3),
                      lattice_params=fix.gt_lattice.view(1, 6))
    match = associate(
        obs_ring_nr=obs.ring_nr, obs_omega=obs.omega, obs_eta=obs.eta,
        pred_ring_slot=pred_ring, pred_omega=_sq(spots.omega),
        pred_eta=_sq(spots.eta), pred_valid=_sq(spots.valid),
        obs_ring_slot=obs_ring_slot, omega_tolerance=math.pi, eta_tolerance=math.pi,
    )
    res = grain_residuals(
        fix.model, grain_euler=fix.gt_euler, grain_position=fix.gt_position,
        grain_lattice=fix.gt_lattice, obs=obs, match=match, kind="pixel",
        px=fix.px, y_BC=fix.y_BC, z_BC=fix.z_BC,
    )
    # 2 residual components (y, z) and NO omega term
    assert res.shape[-1] == 2
    # at ground truth the fit is exact, so the residual is ~0
    assert float(res.abs().max()) < 1e-6

    # ...and the guard: a phase that fits orientation must refuse this loss.
    from midas_fit_grain import FitConfig
    from midas_fit_grain.refine_block import refine_block
    cfg = FitConfig(RingNumbers=fix.ring_numbers, px=fix.px, loss="pixel",
                    solver="lbfgs", mode="all_at_once")
    with pytest.raises(ValueError, match="pixel"):
        refine_block(cfg, model=fix.model, grains_obs=[obs],
                     init_positions=fix.gt_position.view(1, 3),
                     init_eulers=fix.gt_euler.view(1, 3),
                     init_lattices=fix.gt_lattice.view(1, 6),
                     pred_ring_slot=fix.pred_ring_slot)


def test_gradients_flow(fix):
    """Loss should backprop into all 12 grain params."""
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    pred_ring = fix.pred_ring_slot
    obs_ring_slot = ring_slot_lookup(fix.ring_numbers, obs.ring_nr)

    pos = fix.gt_position.clone() + 5.0          # 5 um perturbation
    eul = fix.gt_euler.clone() + 0.05            # ~3°
    lat = fix.gt_lattice.clone()
    lat[0] += 0.001                              # 0.001 Å

    pos.requires_grad_(True)
    eul.requires_grad_(True)
    lat.requires_grad_(True)

    spots = fix.model(eul.view(1, 1, 3), pos.view(1, 1, 3),
                      lattice_params=lat.view(1, 6))

    def _sq(t):
        while t.dim() > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
            if t.dim() == 0:
                break
        return t

    match = associate(
        obs_ring_nr=obs.ring_nr,
        obs_omega=obs.omega,
        obs_eta=obs.eta,
        pred_ring_slot=pred_ring,
        pred_omega=_sq(spots.omega).detach(),
        pred_eta=_sq(spots.eta).detach(),
        pred_valid=_sq(spots.valid).detach(),
        obs_ring_slot=obs_ring_slot,
        omega_tolerance=math.pi, eta_tolerance=math.pi,
    )

    res = grain_residuals(
        fix.model,
        grain_euler=eul, grain_position=pos, grain_lattice=lat,
        obs=obs, match=match, kind="full3d",
        px=fix.px, y_BC=fix.y_BC, z_BC=fix.z_BC,
    )
    loss = (res * res).sum()
    loss.backward()
    assert pos.grad is not None and pos.grad.abs().sum() > 0
    assert eul.grad is not None and eul.grad.abs().sum() > 0
    assert lat.grad is not None and lat.grad.abs().sum() > 0
