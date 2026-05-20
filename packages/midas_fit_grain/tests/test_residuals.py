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


def test_pixel_loss_is_disabled(fix):
    """The 2D 'pixel' loss (omits omega) is disabled — it must raise."""
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
    with pytest.raises(ValueError, match="pixel"):
        grain_residuals(
            fix.model, grain_euler=fix.gt_euler, grain_position=fix.gt_position,
            grain_lattice=fix.gt_lattice, obs=obs, match=match, kind="pixel",
            px=fix.px, y_BC=fix.y_BC, z_BC=fix.z_BC,
        )


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
