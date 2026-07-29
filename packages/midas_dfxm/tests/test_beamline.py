"""Differentiable beamline: CRL verified vs Poulsen, differentiable, correct-by-gradient."""
import torch
import pytest
from midas_dfxm.beamline import (crl_abcd, crl_focal_length, crl_image, crl_na,
                                 Illumination, beamline_resolution)
torch.set_default_dtype(torch.float64)


@pytest.mark.unit
def test_crl_reproduces_poulsen():
    # single-lens f1 = R/(2 delta) = 21.19 m at N=69, R=50um, 17 keV (Poulsen 2017: 21.195)
    _, f1 = crl_abcd(69, 50.0, 17.0)
    assert abs(float(f1) - 21.195) < 0.05
    # thick-lens f_eff matches the analytic f1*phi/sin(N*phi)
    import math
    f_eff = float(crl_focal_length(69, 50.0, 17.0))
    phi = math.sqrt(1.6e-3 / float(f1)); f_analytic = float(f1) * phi / math.sin(69 * phi)
    assert abs(f_eff - f_analytic) / f_analytic < 0.02


@pytest.mark.unit
def test_crl_differentiable():
    R = torch.tensor(50.0, requires_grad=True); E = torch.tensor(17.0, requires_grad=True)
    crl_focal_length(69, R, E).backward()
    assert R.grad is not None and torch.isfinite(R.grad) and float(R.grad) != 0
    assert E.grad is not None and float(E.grad) != 0


@pytest.mark.unit
def test_correct_recovers_lens_radius():
    # differentiable auto-calibration: recover an unknown lens radius from a f_eff measurement
    f_meas = float(crl_focal_length(69, 46.5, 17.0))
    Rhat = torch.tensor(50.0, requires_grad=True)
    opt = torch.optim.Adam([Rhat], lr=5e-1)
    for _ in range(500):
        opt.zero_grad(); ((crl_focal_length(69, Rhat, 17.0) - f_meas) ** 2 * 1e4).backward(); opt.step()
    assert abs(float(Rhat) - 46.5) < 1e-2


@pytest.mark.unit
def test_optimize_hits_target_magnification():
    p = torch.tensor(0.30, requires_grad=True)
    opt = torch.optim.Adam([p], lr=1e-2)
    for _ in range(500):
        opt.zero_grad()
        _, mag = crl_image(69, 50.0, 17.0, p)
        ((mag.abs() - 15.0) ** 2).backward(); opt.step()
        with torch.no_grad(): p.clamp_(0.2, 0.45)
    _, mag = crl_image(69, 50.0, 17.0, p)
    assert abs(float(mag.abs()) - 15.0) < 0.2


@pytest.mark.unit
def test_beamline_resolution_couples_both_sides():
    # resolution depends on BOTH illumination (divergence/bandwidth) and objective (NA)
    r1 = beamline_resolution(Illumination(bandwidth=1e-4), 69, 50.0, 17.0, 3.0, 20.7)
    r2 = beamline_resolution(Illumination(bandwidth=1e-2), 69, 50.0, 17.0, 3.0, 20.7)
    assert r2["sigma_par"] > r1["sigma_par"]          # more bandwidth -> worse axial resolution
