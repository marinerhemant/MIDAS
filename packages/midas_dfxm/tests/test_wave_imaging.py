"""Wave image formation (Stage B) and the composed dynamical DFXM forward (Stage D).

Gates: unit-norm amplitude PSF; incoherent imaging conserves intensity; coherent
differs from incoherent (interference); uniform exit -> uniform image;
differentiability; the composed forward gives a uniform image for a perfect crystal
and a structured, asymmetric image for a dislocation; and the whole
crystal->wave-imaging->detector chain is differentiable back to the h.u field.
"""
import math

import pytest
import torch

from midas_dfxm import (ObjectiveOptics, dfxm_image_wave, dfxm_image_dynamical,
                        dfxm_image_dynamical_pink, extinction_length)

CHI = complex(-1e-6, 0.0)
WL, TB = 0.729, 10.0
LAM = extinction_length(CHI, CHI, wavelength_A=WL, theta_B_deg=TB)


def _optics(det=(96, 96)):
    return ObjectiveOptics(two_theta_deg=2 * TB, magnification=10.0, pixel_um=0.3,
                           detector_shape=det, NA=3e-4, wavelength_A=WL)


def _structured_exit(n=96):
    x = torch.arange(n) - n / 2
    X, Y = torch.meshgrid(x, x, indexing="ij")
    amp = (X ** 2 + Y ** 2 < 20 ** 2).to(torch.float64)
    return (amp * torch.exp(1j * 0.3 * X)).to(torch.complex128)


@pytest.mark.unit
def test_amplitude_psf_unit_norm():
    h = _optics().amplitude_psf()
    assert abs(float((h.abs() ** 2).sum()) - 1.0) < 1e-10
    assert h.is_complex()


@pytest.mark.unit
def test_incoherent_conserves_intensity():
    ew = _structured_exit()
    Ii = dfxm_image_wave(ew, _optics(), coherent_fraction=0.0)
    assert abs(float(Ii.sum()) / float((ew.abs() ** 2).sum()) - 1.0) < 1e-9


@pytest.mark.unit
def test_coherent_differs_from_incoherent():
    ew = _structured_exit()
    Ic = dfxm_image_wave(ew, _optics(), coherent_fraction=1.0)
    Ii = dfxm_image_wave(ew, _optics(), coherent_fraction=0.0)
    assert float((Ic - Ii).abs().max()) > 1e-3


@pytest.mark.unit
def test_uniform_exit_uniform_image():
    u = torch.ones(96, 96, dtype=torch.complex128)
    I = dfxm_image_wave(u, _optics(), coherent_fraction=1.0)
    assert float(I.std() / I.mean()) < 1e-8


@pytest.mark.autograd
def test_wave_imaging_differentiable():
    ew = _structured_exit().clone().requires_grad_(True)
    c = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    dfxm_image_wave(ew, _optics(), coeffs=c, coherent_fraction=1.0).sum().backward()
    assert torch.isfinite(ew.grad).all() and float(c.grad.norm()) > 0


@pytest.mark.unit
def test_dynamical_perfect_crystal_uniform():
    I = dfxm_image_dynamical(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB,
                             thickness_um=0.5 * LAM, optics=_optics(), coherent_fraction=0.0,
                             ny=96)
    assert float(I.std() / I.mean()) < 1e-6


@pytest.mark.unit
def test_dynamical_dislocation_structured_asymmetric():
    nx, nz, t, dx = 96, 150, 1.5 * LAM, 0.4
    xs = (torch.arange(nx) - nx / 2) * dx
    zs = (torch.arange(nz) + 0.5) * (t / nz)
    X = xs[None, :].expand(nz, nx); Z = zs[:, None].expand(nz, nx)
    hu = torch.atan2(Z - 0.5 * t, X)
    I = dfxm_image_dynamical(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB, thickness_um=t,
                             optics=_optics(), hu=hu, dx_um=dx, n_depth=nz,
                             coherent_fraction=0.0, ny=48)
    assert float(I.std() / I.mean()) > 0.05                       # structured
    assert float((I - torch.flip(I, [1])).abs().max()) > 1e-3     # asymmetric


@pytest.mark.unit
def test_pink_image_mono_limit():
    # zero bandwidth pink image == the monochromatic dynamical image.
    kw = dict(wavelength_A=WL, theta_B_deg=TB, thickness_um=0.5 * LAM, optics=_optics(),
              coherent_fraction=0.0, ny=64)
    mono = dfxm_image_dynamical(0.0, CHI, CHI, **kw)
    pink = dfxm_image_dynamical_pink(0.0, CHI, CHI, bandwidth=0.0, n_lambda=1, **kw)
    assert torch.allclose(mono, pink, atol=1e-9)


@pytest.mark.autograd
def test_pink_image_differentiable():
    nx, nz, t, dx = 48, 80, 1.2 * LAM, 0.4
    xs = (torch.arange(nx) - nx / 2) * dx
    zs = (torch.arange(nz) + 0.5) * (t / nz)
    X = xs[None, :].expand(nz, nx); Z = zs[:, None].expand(nz, nx)
    hu = torch.atan2(Z - 0.5 * t, X).clone().requires_grad_(True)
    img = dfxm_image_dynamical_pink(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB,
                                    thickness_um=t, optics=_optics(), bandwidth=5e-5, hu=hu,
                                    dx_um=dx, n_depth=nz, coherent_fraction=0.0, ny=24,
                                    n_lambda=5)
    img.sum().backward()
    assert torch.isfinite(hu.grad).all() and float(hu.grad.norm()) > 0


@pytest.mark.autograd
def test_dynamical_inverse_recovers_amplitude():
    # gradient descent through the dynamical forward recovers a deformation amplitude from
    # dynamical DFXM contrast (the inverse demonstration, minimal version).
    nx, nz, t, dx = 24, 50, 1.5 * LAM, 0.4
    xs = (torch.arange(nx, dtype=torch.float64) - nx / 2) * dx
    zs = (torch.arange(nz, dtype=torch.float64) + 0.5) * (t / nz)
    X = xs[None, :].expand(nz, nx); Z = zs[:, None].expand(nz, nx)
    z0, x0 = 0.5 * t, 1.0
    opt = _optics((nx, nx))

    def fwd(A):
        hu = A * torch.atan2(Z - z0, X - x0)
        return dfxm_image_dynamical(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB, thickness_um=t,
                                    optics=opt, hu=hu, dx_um=dx, n_depth=nz,
                                    coherent_fraction=0.0, ny=12)

    with torch.no_grad():
        meas = fwd(torch.tensor(1.0, dtype=torch.float64))
    A = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([A], lr=0.08)
    loss0 = None
    for _ in range(40):
        optimizer.zero_grad()
        loss = ((fwd(A) - meas) ** 2).mean()
        if loss0 is None:
            loss0 = float(loss)
        loss.backward()
        optimizer.step()
    assert float(loss) < 0.1 * loss0          # loss reduced by >10x
    assert abs(float(A) - 1.0) < 0.1           # amplitude recovered near truth


@pytest.mark.autograd
def test_dynamical_end_to_end_differentiable():
    nx, nz, t, dx = 64, 100, 1.2 * LAM, 0.4
    xs = (torch.arange(nx) - nx / 2) * dx
    zs = (torch.arange(nz) + 0.5) * (t / nz)
    X = xs[None, :].expand(nz, nx); Z = zs[:, None].expand(nz, nx)
    hu = torch.atan2(Z - 0.5 * t, X).clone().requires_grad_(True)
    I = dfxm_image_dynamical(0.0, CHI, CHI, wavelength_A=WL, theta_B_deg=TB, thickness_um=t,
                             optics=_optics(), hu=hu, dx_um=dx, n_depth=nz,
                             coherent_fraction=0.0, ny=32)
    I.sum().backward()
    assert torch.isfinite(hu.grad).all() and float(hu.grad.norm()) > 0
