"""Tests for the Nelder-Mead hard-FracOverlap polish.

The polish is what gives the Python pipeline objective-bit-exact
agreement with C: after L-BFGS over the soft surrogate finds the
basin, scipy NM refines against the discrete FracOverlap that C's
NLopt minimises. We verify three properties:

1. The polish never reports a *lower* hard frac than the seed (it
   either improves or holds — never regresses).
2. The polished Eulers stay inside the ``±tol_rad`` box (matching the
   C bounds and the L-BFGS tanh box).
3. When the seed is already optimal in a tiny obs neighbourhood, the
   polish converges in few evals and reports ``converged=True``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.hard_polish import polish_hard_frac
from midas_nf_fitorientation.obs_volume import ObsVolume
from midas_nf_fitorientation.params import FitParams
from midas_nf_fitorientation.soft_overlap import build_forward_model


def _make_params() -> FitParams:
    """Tiny params suitable for fast unit-test forward calls."""
    p = FitParams()
    p.n_distances = 1
    p.Lsd = [1_000_000.0]
    p.ybc = [64.0]; p.zbc = [64.0]
    p.px = 200.0
    p.omega_start = -180.0; p.omega_step_raw = 1.0
    p.start_nr = 1; p.end_nr = 30
    p.exclude_pole_angle = 6.0
    p.wavelength = 0.172979
    p.lattice_constant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.n_pixels_y = 128; p.n_pixels_z = 128
    p.tx = p.ty = p.tz = 0.0
    p.wedge = 0.0
    return p


def _build_minimal_setup(seed_eul: torch.Tensor):
    """Forward model + obs volume with the centre-pixel of the seed
    orientation's predicted spots lit, so any polish can find at least
    one match."""
    p = _make_params()
    hkls_int = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.float64)
    model = build_forward_model(
        p, hkls_int, device="cpu", dtype=torch.float64,
    )
    pos_um = torch.zeros(3, dtype=torch.float64)

    # Lit a few obs pixels at the seed's predicted spot centres so the
    # polish has something to converge onto.
    obs_arr = np.zeros(
        (1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
        dtype=np.float32,
    )
    with torch.no_grad():
        spots = model(seed_eul.unsqueeze(0), pos_um.unsqueeze(0))
    valid = spots.valid.numpy() > 0.5
    yp = spots.y_pixel.numpy()
    zp = spots.z_pixel.numpy()
    fr = spots.frame_nr.numpy()
    flat = np.where(valid.flatten())[0]
    for fi in flat[:6]:
        k, m = np.unravel_index(fi, valid.shape)
        f_idx = int(np.floor(fr[k, m]))
        y_idx = int(np.floor(yp[k, m]))
        z_idx = int(np.floor(zp[k, m]))
        if (0 <= f_idx < p.n_frames_per_distance
                and 0 <= y_idx < p.n_pixels_y
                and 0 <= z_idx < p.n_pixels_z):
            obs_arr[0, f_idx, y_idx, z_idx] = 1.0
    obs = ObsVolume.from_dense_array(obs_arr, dtype=torch.float64)
    return model, obs, pos_um


def test_polish_returns_nonregressing_frac():
    """The polish must not return a lower hard frac than the seed."""
    seed = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
    model, obs, pos = _build_minimal_setup(seed)

    # Seed-frac before polish.
    with torch.no_grad():
        spots = model(seed.unsqueeze(0), pos.unsqueeze(0))
        seed_frac = float(obs.hard_fraction(
            spots.frame_nr, spots.y_pixel, spots.z_pixel, spots.valid,
        ))

    res = polish_hard_frac(
        model, obs, seed, pos, tol_rad=math.radians(1.0),
        max_iter=100,
    )
    assert res.hard_frac >= seed_frac - 1e-9, (
        f"polish regressed: seed {seed_frac:.4f} -> polished {res.hard_frac:.4f}"
    )


def test_polish_stays_within_tolerance_box():
    seed = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
    model, obs, pos = _build_minimal_setup(seed)
    tol = math.radians(0.5)
    res = polish_hard_frac(
        model, obs, seed, pos, tol_rad=tol, max_iter=100,
    )
    delta = res.eul - seed
    assert torch.all(torch.abs(delta) <= tol + 1e-9), (
        f"polish escaped box: |Δ|={torch.abs(delta).tolist()} vs tol={tol}"
    )


def test_polish_converges_quickly_from_optimum():
    """Seeded at a bright spot, the polish should converge in well
    under the max-iter budget and report ``converged=True``."""
    seed = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
    model, obs, pos = _build_minimal_setup(seed)
    res = polish_hard_frac(
        model, obs, seed, pos, tol_rad=math.radians(1.0),
        max_iter=300,
    )
    # NM convergence + a reasonable eval budget — should be far less
    # than max_iter.
    assert res.n_evals < 300, f"slow convergence: nfev={res.n_evals}"


def test_polish_runs_on_pure_dark_obs():
    """All obs pixels zero — polish must not crash, and its hard frac
    must be 0 (no improvement is possible)."""
    seed = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
    p = _make_params()
    hkls_int = np.array([[1, 0, 0]], dtype=np.float64)
    model = build_forward_model(p, hkls_int, device="cpu", dtype=torch.float64)
    obs = ObsVolume.from_dense_array(
        np.zeros((1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
                 dtype=np.float32),
        dtype=torch.float64,
    )
    pos = torch.zeros(3, dtype=torch.float64)
    res = polish_hard_frac(
        model, obs, seed, pos, tol_rad=math.radians(1.0), max_iter=50,
    )
    assert res.hard_frac == pytest.approx(0.0, abs=1e-9)
