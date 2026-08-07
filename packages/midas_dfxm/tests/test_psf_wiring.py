"""Objective-PSF wiring in dfxm_image (Stage A of the wave-optics build).

The geometrical forward optionally convolves the rendered image with the
wave-optics pupil PSF (``ObjectiveOptics.psf``), sampled at the detector grid.
Gates: unit-sum PSF, intensity conservation, psf=None is a no-op, a symmetric PSF
does not move the mosaicity COM (orientation channel), an aberrated PSF is
asymmetric, and the PSF is differentiable in its Zernike coefficients.
"""
import dataclasses

import pytest
import torch

from midas_dfxm import (make_uniform_field, with_orientation_gradient, GoniometerSetting,
                        reference_q_nom, aligned_resolution, ObjectiveOptics,
                        bragg_two_theta_deg, dfxm_image, dfxm_stack)

DT = torch.float64


def _scene(dtype=DT):
    field = make_uniform_field(shape=(32, 32, 1), spacing_um=0.5)
    field = with_orientation_gradient(field, axis=(0, 0, 1), deg_per_um=0.02, along=0)
    hkl, g = (1, 1, 1), GoniometerSetting()
    q = reference_q_nom(field, hkl, g)
    res = aligned_resolution(q, sigma_par=5e-3, sigma_perp=5e-3)
    tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q)), wavelength_A=0.729)
    opt = ObjectiveOptics(two_theta_deg=tt, magnification=10.0, pixel_um=0.3,
                          detector_shape=(96, 96), NA=3e-4, wavelength_A=0.729)
    return field, hkl, g, res, opt


@pytest.mark.unit
def test_psf_requires_na_and_wavelength():
    with pytest.raises(ValueError):
        ObjectiveOptics(two_theta_deg=20.0).psf()


@pytest.mark.unit
def test_psf_is_unit_sum():
    *_, opt = _scene()
    p = opt.psf()
    assert p.ndim == 2
    assert abs(float(p.sum()) - 1.0) < 1e-10


@pytest.mark.unit
def test_psf_none_is_geometrical_noop():
    field, hkl, g, res, opt = _scene()
    a = dfxm_image(field, hkl, g, res, opt)
    b = dfxm_image(field, hkl, g, res, opt, psf=None)
    assert torch.equal(a, b)


@pytest.mark.unit
def test_psf_conserves_total_intensity():
    field, hkl, g, res, opt = _scene()
    base = dfxm_image(field, hkl, g, res, opt)
    blur = dfxm_image(field, hkl, g, res, opt, psf=opt.psf())
    assert abs(float(blur.sum()) / float(base.sum()) - 1.0) < 1e-10
    # and it actually blurs (not a no-op)
    assert float((blur - base).abs().max()) > 0.0


@pytest.mark.unit
def test_symmetric_psf_preserves_mosaicity_com():
    # uniform field: every pixel shares the same rocking curve, so a symmetric PSF
    # (which mixes identical neighbours) must leave the per-pixel COM unchanged.
    uf = make_uniform_field(shape=(32, 32, 1), spacing_um=0.5)
    uf = dataclasses.replace(uf, F=uf.F.to(DT))
    hkl, g0 = (1, 1, 1), GoniometerSetting()
    q = reference_q_nom(uf, hkl, g0)
    res = aligned_resolution(q, sigma_par=5e-3, sigma_perp=5e-3)
    tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q)), wavelength_A=0.729)
    opt = ObjectiveOptics(two_theta_deg=tt, magnification=10.0, pixel_um=0.3,
                          detector_shape=(96, 96), NA=3e-4, wavelength_A=0.729)
    angles = torch.linspace(-0.3, 0.3, 21)
    settings = [GoniometerSetting(chi=float(a)) for a in angles]
    st0 = dfxm_stack(uf, hkl, settings, res, opt)
    st1 = dfxm_stack(uf, hkl, settings, res, opt, psf=opt.psf())

    def com(stack):
        w = stack.clamp_min(0); tot = w.sum(0)
        m = tot > tot.max() * 0.05
        c = (w * angles[:, None, None]).sum(0) / tot.clamp_min(1e-30)
        return c[m].mean()

    assert abs(float(com(st1) - com(st0))) * 1e3 < 1e-3  # < 0.001 mdeg


@pytest.mark.unit
def test_aberrated_psf_is_asymmetric():
    *_, opt = _scene()
    p = opt.psf({"comaX": 2.0})
    assert float((p - torch.flip(p, [1])).abs().sum()) > 1e-3


@pytest.mark.autograd
def test_psf_differentiable_in_zernikes_and_defocus():
    field, hkl, g, res, opt = _scene()
    c = torch.zeros(8, dtype=DT, requires_grad=True)
    d = torch.tensor(0.3, dtype=DT, requires_grad=True)
    img = dfxm_image(field, hkl, g, res, opt, psf=opt.psf(c, defocus=d))
    (img ** 2).sum().backward()
    assert torch.isfinite(c.grad).all() and float(c.grad.norm()) > 0
    assert torch.isfinite(d.grad).all() and abs(float(d.grad)) > 0
