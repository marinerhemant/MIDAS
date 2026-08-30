"""A single-grain fit with too few matches must not look like a converged one.

Both early-outs in ``optimize_single_grain``'s closure used to return
``torch.tensor(1e6, requires_grad=True)`` -- a FRESH LEAF, disconnected from
``opt_euler``, on which ``.backward()`` was never called. So ``opt_euler.grad``
stayed None, L-BFGS took no step, and the function returned
``loss_history = [1e6, 1e6, 1e6]`` with the parameters moved exactly 0.0 and no
error raised. A constant sentinel is the worst possible signal: it reads as a
fit that plateaued.

Nothing can be restored -- with no matched spots there are no observations and
so no gradient. The contract (spec_autograd_classB_classC.md, C2) is that the
failure is legible instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_diffract import NoMatchError, optimize_single_grain
from midas_diffract.forward import HEDMForwardModel, HEDMGeometry


def _model():
    geom = HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0, wavelength=0.295,
    )
    a, wl = 2.87, 0.295
    hkls_int = torch.tensor([[1, 1, 0], [2, 0, 0], [2, 1, 1], [2, 2, 0]],
                            dtype=torch.float32)
    hkls_cart = (torch.eye(3) / a @ hkls_int.T).T
    thetas = torch.asin(torch.tensor(wl) / (2.0 * (1.0 / torch.norm(hkls_cart, dim=-1))))
    return HEDMForwardModel(hkls=hkls_cart, thetas=thetas, geometry=geom,
                            hkls_int=hkls_int, device=torch.device("cpu"))


def _run_with_unmatchable_observations():
    model = _model()
    init_euler = torch.tensor([0.4, 0.6, 0.8])
    init_lattice = torch.tensor([2.87, 2.87, 2.87, 90.0, 90.0, 90.0])
    # Observations nowhere near any prediction, and a tolerance too tight for
    # anything to associate.
    obs = torch.tensor([[3.0, 3.0, 3.0], [3.1, 3.1, 3.1]])
    return optimize_single_grain(
        model, observed_spots=obs, init_euler=init_euler,
        init_lattice=init_lattice, position=torch.zeros(3),
        max_match_distance=1e-12, min_matches=50,
        phase1_steps=2, phase2_steps=2, phase3_steps=2,
    )


def test_failure_is_reported_not_disguised_as_a_plateau():
    r = _run_with_unmatchable_observations()
    assert r["success"] is False
    assert "failure_reason" in r
    assert all(math.isinf(x) for x in r["loss_history"]), (
        f"loss_history={r['loss_history']} -- a finite constant is "
        f"indistinguishable from a fit that stopped improving"
    )


def test_failed_fit_never_beats_a_real_one_on_loss():
    r = _run_with_unmatchable_observations()
    assert min(r["loss_history"]) == float("inf")
    assert not any(x <= 1e6 for x in r["loss_history"])


def test_parameters_are_returned_as_the_caller_s_own_seed():
    """Not a defect -- there is nothing to move toward. Pinned so it is
    explicit that unmoved parameters come labelled as a failure."""
    init_euler = torch.tensor([0.4, 0.6, 0.8])
    r = _run_with_unmatchable_observations()
    assert torch.allclose(r["euler_rad"], init_euler, atol=1e-12)
    assert r["success"] is False


def test_n_matched_is_reported():
    r = _run_with_unmatchable_observations()
    assert "n_matched" in r
    assert r["n_matched"] < 50


def test_no_match_error_is_public_and_carries_the_count():
    assert issubclass(NoMatchError, RuntimeError)
    e = NoMatchError("x", n_matched=3)
    assert e.n_matched == 3
