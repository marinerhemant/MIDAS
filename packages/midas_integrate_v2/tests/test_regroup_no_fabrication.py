"""Regrouping must not invent intensity where nothing was measured.

``regroup_eta_R_E_to_Q_E`` resamples the (η, R, E) cube onto a Q grid with
:func:`numpy.interp`, which **clamps** outside the input range by default —
it returns the nearest endpoint value, not NaN. So asking for a Q grid wider
than the data returned a flat plateau of the first/last measured intensity,
with no NaN and no warning, indistinguishable from real signal.

Measured before the fix: a ramp from 10 to 50 over Q = 2–6, regrouped onto
Q = 0–10, came back as exactly 10.0 at every point below 2 and exactly 50.0 at
every point above 6.

Fixed by passing ``left``/``right`` explicitly; ``outside="clamp"`` restores
the old behaviour for anyone who wants it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate_v2.inelastic.regroup import regroup_eta_R_E_to_Q_E

N_ETA, N_R, N_E = 8, 40, 5
Q_LO, Q_HI = 2.0, 6.0
I_LO, I_HI = 10.0, 50.0


def _cube():
    eta = torch.linspace(-180.0, 180.0, N_ETA, dtype=torch.float64)
    Q = torch.linspace(Q_LO, Q_HI, N_R, dtype=torch.float64)
    E = torch.linspace(0.0, 100.0, N_E, dtype=torch.float64)
    ramp = torch.linspace(I_LO, I_HI, N_R, dtype=torch.float64)
    cube = ramp[None, :, None].expand(N_ETA, N_R, N_E).contiguous().clone()
    return cube, eta, Q, E


def test_unmeasured_q_is_nan_not_the_endpoint_value():
    cube, eta, Q, E = _cube()
    grid = torch.linspace(0.0, 10.0, 21, dtype=torch.float64)
    out = regroup_eta_R_E_to_Q_E(cube, eta, Q, E, Q_grid=grid).numpy()
    g = grid.numpy()
    below, above = g < Q_LO, g > Q_HI
    assert below.any() and above.any(), "test grid must extend past the data"
    assert np.isnan(out[below]).all(), "below the measured Q must be NaN"
    assert np.isnan(out[above]).all(), "above the measured Q must be NaN"
    # and specifically NOT the old fabricated plateau
    assert not np.allclose(np.nan_to_num(out[below], nan=-1.0), I_LO)
    assert not np.allclose(np.nan_to_num(out[above], nan=-1.0), I_HI)


def test_inside_the_measured_range_is_untouched():
    """The fix must not perturb the values that were always correct."""
    cube, eta, Q, E = _cube()
    grid = torch.linspace(Q_LO, Q_HI, 17, dtype=torch.float64)
    out = regroup_eta_R_E_to_Q_E(cube, eta, Q, E, Q_grid=grid).numpy()
    assert np.isfinite(out).all()
    expected = np.interp(grid.numpy(),
                         np.linspace(Q_LO, Q_HI, N_R),
                         np.linspace(I_LO, I_HI, N_R))
    assert np.allclose(out[:, 0], expected, rtol=1e-12)


def test_clamp_restores_the_old_behaviour_explicitly():
    cube, eta, Q, E = _cube()
    grid = torch.linspace(0.0, 10.0, 21, dtype=torch.float64)
    out = regroup_eta_R_E_to_Q_E(cube, eta, Q, E, Q_grid=grid,
                                 outside="clamp").numpy()
    g = grid.numpy()
    assert np.allclose(out[g < Q_LO, 0], I_LO)
    assert np.allclose(out[g > Q_HI, 0], I_HI)
    assert np.isfinite(out).all()


def test_unknown_outside_mode_is_rejected():
    cube, eta, Q, E = _cube()
    grid = torch.linspace(Q_LO, Q_HI, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="outside"):
        regroup_eta_R_E_to_Q_E(cube, eta, Q, E, Q_grid=grid, outside="hold")
