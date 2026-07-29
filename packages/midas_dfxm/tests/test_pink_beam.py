"""Pink-beam DFXM: reciprocal-space broadening, intensity gain, resolution deconvolution."""
import numpy as np
import pytest
import torch

from midas_dfxm.mosaicity_fit import fit_orientation_mosaicity
from midas_dfxm.pink_beam import (
    EPS_MONO,
    EPS_PINK,
    axial_reciprocal_width,
    intensity_gain,
    pink_beam_res_cov,
    pink_beam_resolution,
    strain_resolution_ratio,
)

DT = torch.float64


def test_axial_width_bandwidth_dominates():
    q = torch.tensor(3.0, dtype=DT)
    mono = axial_reciprocal_width(q, eps=EPS_MONO, two_theta_deg=20.0)
    pink = axial_reciprocal_width(q, eps=EPS_PINK, two_theta_deg=20.0)
    assert pink > mono
    # with negligible divergence the pink width -> |Q0| * eps
    assert float(pink) == pytest.approx(float(q) * EPS_PINK, rel=1e-6)


def test_axial_width_differentiable_in_bandwidth():
    q = torch.tensor(3.0, dtype=DT)
    eps = torch.tensor(EPS_PINK, dtype=DT, requires_grad=True)
    axial_reciprocal_width(q, eps=eps, two_theta_deg=20.0).backward()
    assert eps.grad is not None and float(eps.grad) > 0


def test_intensity_gain_and_strain_penalty():
    assert intensity_gain() == pytest.approx(EPS_PINK / EPS_MONO, rel=1e-9)
    assert intensity_gain(cap=27.0) == pytest.approx(27.0)
    # axial-strain resolution degrades a few-fold at pink bandwidth
    ratio = strain_resolution_ratio(two_theta_deg=20.0)
    assert 2.0 < ratio < 8.0


def test_pink_beam_resolution_broadens():
    q_nom = torch.tensor([3.0, 0.0, 0.5], dtype=DT)
    res = pink_beam_resolution(q_nom, two_theta_deg=20.0, sigma_perp_mono=5e-3, rock_broaden=10.0)
    assert res.sigma_perp == pytest.approx(5e-2)         # 10x transverse
    assert res.sigma_par > res.sigma_perp * 0            # positive
    w = res.weight(q_nom)
    assert float(w) == pytest.approx(1.0)                # peak at q_nom


@pytest.mark.unit
def test_pink_beam_resolution_is_deconvolved_in_fit():
    # Simulate a pink-beam mosaicity scan (large instrument width), then confirm the
    # fit recovers the INTRINSIC mosaic, not the pink-broadened convolved width.
    rng = np.random.default_rng(0)
    P, m = 120, 41
    sig_m = 0.10                                          # intrinsic sample mosaic (deg)
    sigma_rock_mono, sigma_roll_mono = 0.02, 0.02
    res_cov = pink_beam_res_cov(sigma_rock_mono, sigma_roll_mono, rock_broaden=10.0, roll_broaden=1.2)
    R = np.array(res_cov)
    sig_r_eff = np.sqrt(0.5 * (R[0, 0] + R[1, 1]))       # ~0.14 deg, dominates sig_m

    ax = np.linspace(-1, 1, m)
    CH, PH = np.meshgrid(ax, ax, indexing="ij")
    chi = CH.reshape(-1); phi = PH.reshape(-1)
    c0 = rng.uniform(-0.3, 0.3, P); p0 = rng.uniform(-0.3, 0.3, P)
    cov_tot = R + np.eye(2) * sig_m**2
    inv = np.linalg.inv(cov_tot)
    dchi = chi[None] - c0[:, None]; dphi = phi[None] - p0[:, None]
    q = inv[0, 0]*dchi**2 + 2*inv[0, 1]*dchi*dphi + inv[1, 1]*dphi**2
    d = np.exp(-0.5 * q)
    d = rng.poisson(np.clip(d * 500, 0, None)) / 500.0

    out = fit_orientation_mosaicity(
        torch.tensor(d, dtype=DT), torch.tensor(chi, dtype=DT), torch.tensor(phi, dtype=DT),
        res_cov, steps=500, lr=0.03)
    mos = out["mosaic_cov"][:, 0].numpy()
    mos_sig = np.sqrt(0.5 * (mos[:, 0, 0] + mos[:, 1, 1])).mean()
    ori = out["orientation"][:, 0, :].numpy()

    assert np.abs(ori - np.stack([c0, p0], -1)).mean() < 0.03          # orientation recovered
    assert abs(mos_sig - sig_m) < 0.05                                 # intrinsic mosaic recovered
    assert mos_sig < sig_r_eff - 0.02                                  # genuinely below instrument width
