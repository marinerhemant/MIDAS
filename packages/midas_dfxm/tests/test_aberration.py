"""Wave-optics objective aberrations: PSF, differentiability, device portability, self-cal."""
import pytest
import torch

from midas_dfxm.aberration import (ABERRATIONS, aberrated_psf, coeffs_to_tensor,
                                    convolve_psf, fit_aberration, wiener_deconvolve)

torch.set_default_dtype(torch.float64)


def _devices():
    devs = [("cpu", torch.float64)]
    if torch.cuda.is_available():
        devs.append(("cuda", torch.float64))
    if torch.backends.mps.is_available():
        devs.append(("mps", torch.float32))   # MPS has no float64
    return devs


@pytest.mark.unit
def test_psf_normalized_and_shaped():
    psf = aberrated_psf({"astig0": 0.3, "comaX": 0.2}, defocus=0.5, grid_size=48)
    assert psf.shape == (48, 48)
    assert abs(float(psf.sum()) - 1.0) < 1e-10
    assert float(psf.min()) >= 0.0


@pytest.mark.unit
def test_delta_psf_is_identity():
    img = torch.randn(48, 48)
    delta = torch.zeros(48, 48); delta[24, 24] = 1.0
    assert float((convolve_psf(img, delta) - img).abs().max()) < 1e-12


@pytest.mark.unit
def test_zero_aberration_psf_is_symmetric():
    # with no aberrations the apodized-aperture PSF is 4-fold symmetric (Airy-like).
    # Exclude row/col 0: even-length fftshift leaves the DC at N//2, so index 0 (freq -N/2)
    # has no mirror partner -- a discrete-grid centering artifact, not an asymmetry.
    psf = aberrated_psf(torch.zeros(len(ABERRATIONS)), grid_size=64)[1:, 1:]
    assert float((psf - psf.flip(0)).abs().max()) < 1e-12
    assert float((psf - psf.flip(1)).abs().max()) < 1e-12


@pytest.mark.unit
def test_psf_differentiable_in_coeffs():
    c = torch.zeros(len(ABERRATIONS), requires_grad=True)
    psf = aberrated_psf(c, defocus=0.3, grid_size=32)
    # a functional of the PSF must carry gradient to every coefficient
    (psf ** 2).sum().backward()
    assert c.grad is not None and torch.isfinite(c.grad).all()
    assert float(c.grad.abs().sum()) > 0


@pytest.mark.unit
def test_convolve_differentiable_in_image():
    img = torch.randn(32, 32, requires_grad=True)
    psf = aberrated_psf({"comaX": 0.2}, grid_size=32)
    convolve_psf(img, psf).sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()


@pytest.mark.unit
@pytest.mark.parametrize("device,dtype", _devices())
def test_device_portable(device, dtype):
    c = coeffs_to_tensor({"astig0": 0.3, "comaX": 0.2}, device=device, dtype=dtype)
    psf = aberrated_psf(c, defocus=0.4, grid_size=32, device=device, dtype=dtype)
    assert psf.device.type == device and torch.isfinite(psf).all()
    img = torch.randn(32, 32, device=device, dtype=dtype)
    out = convolve_psf(img, psf)
    assert out.device.type == device and torch.isfinite(out).all()


@pytest.mark.integration
def test_self_calibration_recovers_aberrations_through_focal():
    # inject aberrations, image a known feature, recover by phase-diversity fit
    ctrue = coeffs_to_tensor({"astig0": 0.30, "astig45": -0.20, "comaX": 0.25,
                              "comaY": -0.15, "spher": 0.20})
    scene = torch.zeros(64, 64); scene[24, 22] = 1.0; scene[38, 41] = 0.7
    defocus = [-3.0, -1.5, 0.0, 1.5, 3.0]
    measured = torch.stack([convolve_psf(scene, aberrated_psf(ctrue, defocus=d, grid_size=64))
                            for d in defocus])
    c_focal = fit_aberration(measured, defocus, scene, steps=400, lr=0.03)
    c_single = fit_aberration(measured[2:3], [0.0], scene, steps=400, lr=0.03)
    err_focal = float((c_focal - ctrue).abs().max())
    err_single = float((c_single - ctrue).abs().max())
    assert err_focal < 0.05                       # through-focal recovers accurately
    assert err_single > 2 * err_focal             # single in-focus image is worse (identifiability)


@pytest.mark.integration
def test_wiener_deconvolve_recovers_blurred_scene():
    scene = torch.zeros(64, 64); scene[30, 28] = 1.0; scene[36, 40] = 0.6
    psf = aberrated_psf({"astig0": 0.2, "comaX": 0.15}, grid_size=64)
    blurred = convolve_psf(scene, psf)
    recon = wiener_deconvolve(blurred, psf, reg=1e-4)
    # deconvolution with the true PSF is closer to the sharp scene than the blurred image
    assert float((recon - scene).pow(2).mean()) < float((blurred - scene).pow(2).mean())
