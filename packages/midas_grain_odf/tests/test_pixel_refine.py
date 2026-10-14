"""pixel_refine: plant a grain with the SAME renderer, add Poisson noise, refit.

This is a self-consistency check of the optimiser and plumbing (inverse crime
by construction), not a validation of the method; the method's validation is
the independent known-truth simulation in the pre-registered Phase 0.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))
from conftest import make_model, random_orientation, standard_ff_geometry  # noqa: E402

from midas_grain_odf import pixel_refine as pr  # noqa: E402


def _plant(seed=3, spread_deg=0.06, scale=(300, 3000)):
    torch.manual_seed(seed)
    model = make_model()
    geo = standard_ff_geometry()
    R = random_orientation(seed).to(torch.float64)
    pos = torch.tensor([120.0, -80.0, 15.0], dtype=torch.float64)
    lat = torch.tensor([3.61, 3.61, 3.61, 90, 90, 90], dtype=torch.float64)
    eps = torch.tensor([4e-4, -1e-4, 2e-4, -3e-4, 1e-4, 5e-4], dtype=torch.float64)
    F, P = 7, 17
    refl, ay, az, af = pr.select_reflections(model, R, pos, lat, eps, n_frames=geo.n_frames,
                                             n_pixels_y=geo.n_pixels_y, n_pixels_z=geo.n_pixels_z,
                                             patch_F=F, patch_P=P)
    S = refl.numel()
    data = pr.GrainPixelData(torch.zeros(S, F, P, P, dtype=torch.float64), torch.ones(S, F, P, P, dtype=torch.float64),
                             ay, az, af, refl, F, P)
    cfg = pr.PixelFitConfig(psf0=1.3)
    gm = pr.GrainPixelModel(model, data, R, torch.zeros(3, dtype=torch.float64), lat, cfg)
    th = gm.theta0()
    th[pr.IDX_POS] = pos
    th[pr.IDX_EPS] = eps
    th[pr.IDX_SPREAD] = math.log(math.radians(spread_deg))
    with torch.no_grad():
        T = gm.template(th)
    a = torch.empty(S, dtype=torch.float64).uniform_(*scale)
    lam = a.reshape(-1, 1, 1, 1) * T + 4.0
    D = torch.poisson(lam)
    return model, lat, R, pos, eps, refl, ay, az, af, D, F, P


def test_plant_and_recover_same_renderer():
    model, lat, R, pos, eps, refl, ay, az, af, D, F, P = _plant()
    assert refl.numel() > 40
    data = pr.GrainPixelData(D, torch.ones_like(D), ay, az, af, refl, F, P)
    # seed: orientation off by 0.03 deg, position off by 15 um, no strain, wrong spread
    R0 = R @ pr.rot_exp(torch.tensor([3e-4, -4e-4, 2e-4], dtype=torch.float64))
    pos0 = pos + torch.tensor([10.0, -8.0, 9.0], dtype=torch.float64)
    cfg = pr.PixelFitConfig(psf0=1.3, spread0_deg=0.02)
    res = pr.fit_grain_pixels(model, data, R0, pos0, lat, cfg)
    assert res.cost < res.cost_init
    mis = torch.rad2deg(torch.arccos(((torch.trace(res.R.T @ R) - 1) / 2).clamp(-1, 1)))
    assert float(mis) < 0.003, float(mis)
    assert float((res.pos - pos).norm()) < 3.0, res.pos
    assert float((res.eps_sample - eps).abs().max()) < 6e-5, res.eps_sample
    assert abs(res.spread_deg - 0.06) < 0.012, res.spread_deg
    # reduced chi2 near 1 for Poisson data under the true model
    assert 0.8 < res.cost / res.n_pixels < 1.3


def test_select_reflections_without_strain_matches_zero_strain():
    """eps_sample=None must behave like explicit zero strain, on the model's own device
    (it was built on the CPU, which crashed a CUDA model)."""
    model, lat, R, pos, *_ = _plant()
    kw = dict(n_frames=1440, n_pixels_y=3888, n_pixels_z=3072, patch_F=7, patch_P=17)
    a = pr.select_reflections(model, R, pos, lat, None, **kw)
    b = pr.select_reflections(model, R, pos, lat, torch.zeros(6, dtype=torch.float64), **kw)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_heldout_cost_prefers_the_true_state_over_a_shifted_one():
    """The guard's score must rank the true grain state above a 15 um / 0.03 deg off one on held-out spots."""
    model, lat, R, pos, eps, refl, ay, az, af, D, F, P = _plant()
    data = pr.GrainPixelData(D, torch.ones_like(D), ay, az, af, refl, F, P)
    ho = pr.subset_data(data, list(range(0, refl.numel(), 5)))
    cfg = pr.PixelFitConfig(psf0=1.3)
    nz = pr.default_theta(cfg); nz[pr.IDX_SPREAD] = math.log(math.radians(0.06))
    c_true = pr.heldout_cost(model, ho, R, pos, eps, lat, cfg, nz)
    R_off = R @ pr.rot_exp(torch.tensor([3e-4, -4e-4, 2e-4], dtype=torch.float64))
    c_off = pr.heldout_cost(model, ho, R_off, pos + torch.tensor([10.0, -8.0, 9.0], dtype=torch.float64), eps, lat, cfg, nz)
    assert c_true < c_off


def test_convergence_flag_is_scale_free_and_true_at_the_optimum():
    """Started at the planted truth (an optimum up to noise), every stage must report converged, and the
    flag must not depend on the noise-level size of the gradient (it used to flip between identical runs)."""
    model, lat, R, pos, eps, refl, ay, az, af, D, F, P = _plant()
    data = pr.GrainPixelData(D, torch.ones_like(D), ay, az, af, refl, F, P)
    cfg = pr.PixelFitConfig(psf0=1.3, spread0_deg=0.06)
    th = pr.default_theta(cfg); th[pr.IDX_POS] = pos; th[pr.IDX_EPS] = eps
    res = pr.fit_grain_pixels(model, data, R, torch.zeros(3, dtype=torch.float64), lat, cfg, theta_init=th)
    assert all(res.converged.values()), res.converged
    res2 = pr.fit_grain_pixels(model, data, R, torch.zeros(3, dtype=torch.float64), lat, cfg, theta_init=th)
    assert res.converged == res2.converged


def test_readout_offset_matches_the_transforms_correction():
    """The model adds back exactly what midas_transforms subtracts: omega -= shift * (1 - |X - C| / (N/2))."""
    cfg = pr.PixelFitConfig(readout_shift_deg=-0.156, readout_axis=1, readout_shape=0, readout_npx=2880,
                            omega_step_deg=0.250584)
    y = torch.tensor([1440.0, 0.0, 720.0], dtype=torch.float64)
    off = pr.readout_frame_offset(cfg, y, torch.zeros(3, dtype=torch.float64))
    expect = torch.tensor([-0.156, 0.0, -0.078], dtype=torch.float64) / 0.250584
    assert torch.allclose(off, expect, atol=1e-12)
    assert torch.all(pr.readout_frame_offset(pr.PixelFitConfig(), y, y) == 0)
