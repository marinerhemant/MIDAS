"""Size / microstrain / mosaic separation from FF-HEDM spot widths.

Sibling to test_williamson_hall.py (which exercises the AsterismFit-driven
modified W-H + dislocation density); this tests the generic raw-(q, width)
peakshape module folded in from the standalone midas_peakshape prototype.
"""
import pytest
import torch

from midas_defect.peakshape import (
    azimuthal_width,
    crystallite_size_from_width,
    fit_size_strain_mosaic,
    radial_width,
    recover_size_distribution,
    size_broadened_profile,
    size_width_q,
    williamson_hall_qw,
)

DT = torch.float64


@pytest.mark.unit
def test_pure_size_is_constant_in_q():
    q = torch.linspace(2, 8, 10, dtype=DT)
    w = radial_width(q, w_size=size_width_q(300.0), eps=0.0)
    assert torch.allclose(w, w[0].expand_as(w), atol=1e-9)


@pytest.mark.unit
def test_pure_strain_grows_with_q():
    q = torch.linspace(2, 8, 10, dtype=DT)
    w = radial_width(q, w_size=0.0, eps=3e-3)
    assert torch.allclose(w, 3e-3 * q, atol=1e-9)


@pytest.mark.unit
def test_williamson_hall_qw_recovers_size_and_strain():
    D_true, eps_true, w_inst = 300.0, 3e-3, 0.008
    q = torch.linspace(2, 8, 14, dtype=DT)
    w = radial_width(q, w_size=size_width_q(D_true), eps=eps_true, w_inst=w_inst)
    out = williamson_hall_qw(q, w, w_inst=w_inst)
    assert abs(out["D"] - D_true) / D_true < 0.05
    assert abs(out["eps"] - eps_true) / eps_true < 0.10


@pytest.mark.unit
def test_williamson_hall_qw_robust_to_noise():
    torch.manual_seed(0)
    D_true, eps_true, w_inst = 250.0, 4e-3, 0.006
    q = torch.linspace(2, 9, 25, dtype=DT)
    w = radial_width(q, w_size=size_width_q(D_true), eps=eps_true, w_inst=w_inst)
    w = w * (1 + 0.02 * torch.randn_like(w))
    out = williamson_hall_qw(q, w, w_inst=w_inst)
    assert abs(out["D"] - D_true) / D_true < 0.15
    assert abs(out["eps"] - eps_true) / eps_true < 0.20


@pytest.mark.unit
def test_fit_size_strain_mosaic_recovers_all_three():
    D_true, eps_true, mosaic_true = 300.0, 3e-3, 2e-3
    w_inst, w_inst_az = 0.008, 1e-3
    q = torch.linspace(2, 8, 14, dtype=DT)
    w_rad = radial_width(q, w_size=size_width_q(D_true), eps=eps_true, w_inst=w_inst)
    w_az = azimuthal_width(torch.tensor(mosaic_true, dtype=DT),
                           w_inst_az=w_inst_az, n=len(q))
    out = fit_size_strain_mosaic(q, w_rad, w_az, w_inst=w_inst, w_inst_az=w_inst_az,
                                 steps=1500, lr=0.02)
    assert abs(out["D"] - D_true) / D_true < 0.06
    assert abs(out["eps"] - eps_true) / eps_true < 0.12
    assert abs(out["mosaic"] - mosaic_true) / mosaic_true < 0.15


@pytest.mark.unit
def test_fit_agrees_with_linear_williamson_hall_qw():
    D_true, eps_true, w_inst = 200.0, 5e-3, 0.005
    q = torch.linspace(2, 9, 16, dtype=DT)
    w = radial_width(q, w_size=size_width_q(D_true), eps=eps_true, w_inst=w_inst)
    wh = williamson_hall_qw(q, w, w_inst=w_inst)
    out = fit_size_strain_mosaic(q, w, w_inst=w_inst, steps=1200, lr=0.02)
    assert abs(out["D"] - wh["D"]) / wh["D"] < 0.05
    assert abs(out["eps"] - wh["eps"]) / wh["eps"] < 0.05


@pytest.mark.unit
def test_recover_size_distribution_peaks_correctly():
    dq = torch.linspace(-0.12, 0.12, 240, dtype=DT)
    D_grid = [120.0, 180.0, 250.0, 350.0, 500.0]
    w_true = torch.tensor([0.1, 0.25, 0.4, 0.18, 0.07], dtype=DT)
    obs = sum(wk * size_broadened_profile(dq, D) for wk, D in zip(w_true, D_grid))
    out = recover_size_distribution(dq, obs, D_grid, steps=2000, lr=0.05)
    grid = torch.tensor(D_grid, dtype=DT)
    mean_true = float((w_true / w_true.sum() * grid).sum())
    mean_rec = float((out["weights"] * grid).sum())
    assert abs(mean_rec - mean_true) / mean_true < 0.12
