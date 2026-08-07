"""Takagi-Taupin dynamical-diffraction solver validation.

The core gate is the closed-form symmetric-Laue solution
``|Dh|^2 = sin^2(pi t/Lambda sqrt(1+y^2)) / (1+y^2)`` which the marcher must
reproduce to machine precision; plus Pendellosung, the kinematical thin-crystal
limit, asymmetric dislocation contrast, the 2x2 matrix exponential, and
differentiability in the displacement field. Susceptibilities for the solver gates
are synthetic (self-consistent); one gate exercises the midas_hkls path.
"""
import math

import pytest
import torch

from midas_dfxm.takagi_taupin import (
    solve_tt_laue, diffracted_intensity, extinction_length, laue_intensity_analytic,
    susceptibility_fourier, _expm2x2,
    solve_tt_bragg, bragg_reflectivity, darwin_width,
    pink_diffracted_intensity, pink_deviation_offsets,
)

WL = 0.729
TB = 10.0
CHI = complex(-1e-6, 0.0)          # real, non-absorbing, centrosymmetric
CHI0 = 0.0                          # drop common phase for the non-absorbing form
LAM = extinction_length(CHI, CHI, wavelength_A=WL, theta_B_deg=TB)


@pytest.mark.unit
def test_closed_form_laue():
    err = 0.0
    for t in (0.2 * LAM, 0.5 * LAM, 1.0 * LAM, 1.5 * LAM, 2.0 * LAM):
        for y in (0.0, 0.5, 1.0, 2.0, 4.0):
            num = float(diffracted_intensity(CHI0, CHI, CHI, wavelength_A=WL,
                                             theta_B_deg=TB, thickness_um=t, y=y, n_depth=150))
            ana = float(laue_intensity_analytic(t, y, LAM))
            err = max(err, abs(num - ana))
    assert err < 1e-9


@pytest.mark.unit
def test_pendellosung():
    at_lam = float(diffracted_intensity(CHI0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB,
                                        thickness_um=LAM, y=0.0))
    at_half = float(diffracted_intensity(CHI0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB,
                                         thickness_um=0.5 * LAM, y=0.0))
    assert at_lam < 1e-12          # first Pendellosung zero at t = Lambda
    assert abs(at_half - 1.0) < 1e-6   # full transfer at Lambda/2


@pytest.mark.unit
def test_kinematical_limit():
    # |Dh|^2 -> (pi t/Lambda)^2 as t -> 0; the relative error shrinks with thickness.
    t = 0.01 * LAM
    num = float(diffracted_intensity(CHI0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB,
                                     thickness_um=t, y=0.0))
    kin = (math.pi * t / LAM) ** 2
    assert abs(num - kin) / kin < 1e-3


@pytest.mark.unit
def test_dislocation_asymmetric_contrast():
    nx, nz = 96, 150
    t = 1.5 * LAM
    dx = 0.5
    xs = (torch.arange(nx) - nx / 2) * dx
    zs = (torch.arange(nz) + 0.5) * (t / nz)
    X = xs[None, :].expand(nz, nx)
    Z = zs[:, None].expand(nz, nx)
    hu = torch.atan2(Z - 0.5 * t, X - 0.0)      # screw-like winding phase
    _, Dh = solve_tt_laue(CHI0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB, thickness_um=t,
                          y=0.0, hu=hu, dx_um=dx, n_depth=nz)
    I = Dh.abs() ** 2
    I0 = float(diffracted_intensity(CHI0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB,
                                    thickness_um=t, y=0.0))
    assert float((I - torch.flip(I, [0])).abs().max()) > 1e-3     # asymmetric
    assert float((I - I0).abs().max()) > 1e-2                     # the defect shows


@pytest.mark.unit
def test_expm2x2_matches_torch():
    torch.manual_seed(0)
    M = torch.randn(5, 2, 2, dtype=torch.complex128)
    ours = _expm2x2(M)
    ref = torch.matrix_exp(M)
    assert torch.allclose(ours, ref, atol=1e-10)


@pytest.mark.autograd
def test_differentiable_in_displacement():
    hu = torch.linspace(0, 2 * math.pi, 8).repeat(60, 1).clone().requires_grad_(True)
    _, Dh = solve_tt_laue(CHI0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB, thickness_um=LAM,
                          y=0.0, hu=hu, dx_um=0.5, n_depth=60)
    (Dh.abs() ** 2).sum().backward()
    assert torch.isfinite(hu.grad).all() and float(hu.grad.norm()) > 0


@pytest.mark.unit
def test_susceptibility_from_midas_hkls():
    from midas_dfxm.io import fcc_reference_crystal
    chi0, chih, chihbar = susceptibility_fourier(fcc_reference_crystal(), (2, 2, 0),
                                                 wavelength_A=1.2398, absorption=False)
    assert 1e-7 < abs(chi0) < 1e-4        # physical for a metal at ~10 keV
    assert 1e-7 < abs(chih) < 1e-4
    assert abs(chi0.imag) < 1e-12         # absorption off -> real chi
    lam = extinction_length(chih, chihbar, wavelength_A=1.2398, theta_B_deg=22.0)
    assert 0.5 < lam < 50.0               # Cu 220 extinction length, micrometers


@pytest.mark.unit
def test_bragg_darwin_plateau():
    # thick perfect crystal: total-reflection plateau R~1 for |y|<1, dropping outside.
    T = 30 * LAM
    def R(y):
        return float(bragg_reflectivity(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=20.0,
                                        thickness_um=T, y=y, n_depth=1500))
    assert abs(R(0.0) - 1.0) < 1e-3
    assert abs(R(0.9) - 1.0) < 1e-2
    assert R(2.0) < 0.3                      # outside the total-reflection domain
    assert R(4.0) < 0.05


@pytest.mark.unit
def test_bragg_plateau_centered_with_refraction():
    # with chi0 != 0 the plateau is refraction-shifted; center=True puts it at y=0.
    from midas_dfxm.io import fcc_reference_crystal
    c0, ch, chb = susceptibility_fourier(fcc_reference_crystal(), (2, 2, 0),
                                         wavelength_A=1.2398, absorption=False)
    R0 = float(bragg_reflectivity(c0, ch, chb, wavelength_A=1.2398, theta_B_deg=22.0,
                                  thickness_um=50.0, y=0.0, n_depth=1500))
    assert abs(R0 - 1.0) < 1e-2              # centered peak reaches total reflection


@pytest.mark.unit
def test_bragg_kinematical_thin_grows():
    def R(t):
        return float(bragg_reflectivity(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=20.0,
                                        thickness_um=t * LAM, y=0.0, n_depth=400))
    assert R(0.2) > R(0.1) > R(0.05)         # reflectivity builds up with thickness


@pytest.mark.unit
def test_darwin_width_physical():
    w = darwin_width(CHI, CHI, wavelength_A=WL, theta_B_deg=20.0)
    assert 1e-7 < w < 1e-3                    # microradian-scale, physical


@pytest.mark.autograd
def test_bragg_differentiable_off_plateau():
    nz, nx = 200, 6
    z = torch.linspace(0, 1, nz)[:, None]
    hu = (z * torch.linspace(0, 2, nx)[None, :]).clone().requires_grad_(True)
    X = solve_tt_bragg(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=20.0,
                       thickness_um=0.3 * LAM, y=0.5, hu=hu, n_depth=nz)
    (X.abs() ** 2).sum().backward()
    assert torch.isfinite(hu.grad).all() and float(hu.grad.norm()) > 0


@pytest.mark.unit
def test_reduces_to_geometrical_mosaicity_com():
    # the shared observable is the mosaicity COM: the thin (kinematical) dynamical
    # rocking-curve COM recovers the planted orientation, matching the geometrical
    # symmetric acceptance. Thresholded COM (as real DFXM analysis does), since the
    # kinematical rocking curve has heavy 1/(1+y^2) tails.
    omega = torch.linspace(-8, 8, 121)

    def com(curve, thr=0.2):
        m = curve >= thr * curve.max()
        return float((omega[m] * curve[m]).sum() / curve[m].sum())

    for th in (-1.5, -0.5, 0.5, 1.5):
        R = torch.tensor([float(diffracted_intensity(CHI0, CHI, CHI, wavelength_A=WL,
                          theta_B_deg=20.0, thickness_um=0.15 * LAM, y=float(w - th), n_depth=50))
                          for w in omega])
        assert abs(com(R) - th) < 0.05          # recovers planted orientation


@pytest.mark.autograd
def test_deformation_sensitivity_nonzero():
    # dI/dA (the Fisher sensitivity used for design) is nonzero -> design is differentiable.
    from torch.autograd.functional import jvp
    nx, nz, dx = 24, 50, 0.4
    t = 0.8 * LAM
    zs = (torch.arange(nz, dtype=torch.float64) + 0.5) * (t / nz)
    xs = (torch.arange(nx, dtype=torch.float64) - nx / 2) * dx

    def intensity(A):
        hu = A * torch.atan2(zs[:, None] - 0.5 * t, xs[None, :] - 1.0)
        return solve_tt_laue(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=10.0, thickness_um=t,
                             y=0.0, hu=hu, dx_um=dx, n_depth=nz)[1].abs() ** 2

    _, dI = jvp(intensity, (torch.tensor(1.0, dtype=torch.float64),),
               (torch.ones((), dtype=torch.float64),))
    assert float((dI ** 2).sum()) > 0


@pytest.mark.autograd
def test_bragg_inverse_recovers_amplitude():
    # gradient descent through the differentiable Bragg (Riccati) solver recovers a deformation
    # amplitude from reflectivity contrast (near-surface, below the total-reflection plateau).
    nx, nz, dx, tb = 24, 60, 0.4, 20.0
    lam = extinction_length(CHI, CHI, wavelength_A=WL, theta_B_deg=tb)
    t = 0.6 * lam
    zs = (torch.arange(nz, dtype=torch.float64) + 0.5) * (t / nz)
    xs = (torch.arange(nx, dtype=torch.float64) - nx / 2) * dx
    z0 = 0.2 * t

    def fwd(A):
        hu = A * torch.atan2(zs[:, None] - z0, xs[None, :] - 1.0)
        return torch.stack([solve_tt_bragg(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=tb,
                            thickness_um=t, y=yi, hu=hu, n_depth=nz).abs() ** 2
                            for yi in (-0.7, -0.2, 0.3, 0.8)])

    with torch.no_grad():
        meas = fwd(torch.tensor(1.0, dtype=torch.float64))
    A = torch.tensor(0.85, dtype=torch.float64, requires_grad=True)     # warm start
    optimizer = torch.optim.Adam([A], lr=0.04)
    loss0 = None
    for _ in range(30):
        optimizer.zero_grad()
        loss = ((fwd(A) - meas) ** 2).mean()
        if loss0 is None:
            loss0 = float(loss)
        loss.backward()
        optimizer.step()
    assert float(loss) < 0.5 * loss0
    assert abs(float(A) - 1.0) < 0.2


@pytest.mark.unit
def test_kinematical_model_misfits_thick_dynamical_contrast():
    # a thick-crystal dynamical dislocation contrast is fit by the dynamical model (self) but
    # NOT by the weak-coupling (kinematical, single-scattering) model -- the "why dynamical" claim.
    nx, nz, dx = 24, 50, 0.4
    t = 1.5 * LAM
    zs = (torch.arange(nz, dtype=torch.float64) + 0.5) * (t / nz)
    xs = (torch.arange(nx, dtype=torch.float64) - nx / 2) * dx
    hu = torch.atan2(zs[:, None] - 0.5 * t, xs[None, :])

    def intensity(chih):
        _, Dh = solve_tt_laue(0.0, chih, chih, wavelength_A=WL, theta_B_deg=10.0, thickness_um=t,
                              y=0.0, hu=hu, dx_um=dx, n_depth=nz)
        return Dh.abs() ** 2

    data = intensity(CHI)                       # dynamical
    kin = intensity(CHI * 1e-3)                 # kinematical (weak coupling)

    def misfit(m):
        s = (m * data).sum() / (m * m).sum()
        return float((((s * m - data) ** 2).sum() ** 0.5) / ((data ** 2).sum() ** 0.5))

    assert misfit(data) < 1e-6                  # dynamical model fits itself
    assert misfit(kin) > 0.1                    # kinematical model cannot fit dynamical contrast


@pytest.mark.unit
def test_pink_mono_limit():
    # zero bandwidth (single wavelength) reduces to the monochromatic intensity.
    T = 0.5 * LAM
    mono = float(diffracted_intensity(CHI0, CHI, CHI, wavelength_A=WL, theta_B_deg=20.0,
                                      thickness_um=T, y=0.0))
    pink = float(pink_diffracted_intensity(CHI0, CHI, CHI, thickness_um=T, y0=0.0,
                                           bandwidth=0.0, theta_B_deg=20.0, wavelength_A=WL,
                                           n_lambda=1))
    assert abs(pink - mono) < 1e-9


@pytest.mark.unit
def test_pink_broadens_acceptance():
    # off the monochromatic peak, a pink band picks up in-band diffraction -> larger signal.
    T, y0 = 0.5 * LAM, 3.0
    mono = float(diffracted_intensity(CHI0, CHI, CHI, wavelength_A=WL, theta_B_deg=20.0,
                                      thickness_um=T, y=y0))
    pink = float(pink_diffracted_intensity(CHI0, CHI, CHI, thickness_um=T, y0=y0,
                                           bandwidth=3e-5, theta_B_deg=20.0, wavelength_A=WL,
                                           n_lambda=21))
    assert pink > mono


@pytest.mark.unit
def test_pink_offsets_scale_with_bandwidth():
    _, w = pink_deviation_offsets(1e-4, 20.0, CHI, n=21)
    assert abs(float(w.sum()) - 1.0) < 1e-9              # normalized spectrum
    ys_small, _ = pink_deviation_offsets(1e-5, 20.0, CHI, n=21)
    ys_big, _ = pink_deviation_offsets(1e-4, 20.0, CHI, n=21)
    assert float(ys_big.abs().max()) > float(ys_small.abs().max())  # wider band, wider offsets


@pytest.mark.unit
def test_absorption_decays_transmitted_power():
    from midas_dfxm.io import fcc_reference_crystal
    c0, ch, chb = susceptibility_fourier(fcc_reference_crystal(), (2, 2, 0),
                                         wavelength_A=1.2398, absorption=True)
    assert c0.imag > 0                    # positive imaginary part = absorption

    def power(t):
        D0, Dh = solve_tt_laue(c0, ch, chb, wavelength_A=1.2398, theta_B_deg=11.0,
                               thickness_um=t, y=0.0, n_depth=200)
        return float((D0.abs() ** 2 + Dh.abs() ** 2).squeeze())

    assert power(20.0) < power(2.0)       # transmitted power decays with thickness
