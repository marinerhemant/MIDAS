"""Wave-optics aberrations of the DFXM objective, and self-calibration from data.

The beamline model (``beamline.py``) uses paraxial ABCD ray-transfer -- exact for focal
length and magnification, but *linear*, so it cannot represent aberrations, which are the
departure from the linear map. This module adds the missing wave-optics layer: the
objective's exit pupil carries a Zernike wavefront error on a Gaussian-apodized aperture
(the apodization is the CRL absorption taper), and the point-spread function is
``PSF = |FFT(pupil)|^2``. Everything is torch-differentiable in the Zernike coefficients.

Two capabilities follow, neither of which the non-differentiable optics codes (SHADOW/SRW,
and Henningsson's ABCD RayTransferMatrix) provide:

* **forward** -- an aberrated, possibly asymmetric PSF that blurs the reconstructed DFXM
  maps (the objective's contribution to the resolution function; Poulsen/Simons treat the
  resolution as blurring the reconstructed field -- here that blur is a fitted wavefront);
* **self-calibration** -- fit the objective's aberration coefficients *from the data* by
  gradient descent through this forward, using a through-focal (phase-diversity) series.
  The identifiability probe (``dev/paper/runs/aberration_selfcal/``) shows a single
  in-focus image is nearly blind to the even aberrations, so defocus diversity is required;
  that design prescription drops out of the Fisher tooling in ``field_inverse.py``.

Device/dtype-portable; the PSF and convolution carry gradients, so a fitted PSF can be
folded back into the reconstruction to remove the aberration-induced bias in F.
"""
from __future__ import annotations

import torch

from midas_invert.optimize import fit as _fit

# Fittable aberrations (canonical order). Defocus is handled separately: it is the KNOWN,
# controllable through-focal knob (phase diversity), not an unknown to recover.
ABERRATIONS = ("astig0", "astig45", "comaX", "comaY", "spher", "trefoilX", "trefoilY")


def _pupil_grid(grid_size, extent, *, device=None, dtype=torch.float64):
    """Sampled pupil coordinates over ``[-extent, extent]^2`` with a unit-disk aperture."""
    u = torch.linspace(-extent, extent, grid_size, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, u, indexing="ij")
    r2 = uu ** 2 + vv ** 2
    aperture = (r2 <= 1.0).to(dtype)
    return uu, vv, r2, aperture


def zernike_terms(uu, vv):
    """Low-order Zernike wavefront shapes over the unit disk (unnormalized, Cartesian)."""
    r2 = uu ** 2 + vv ** 2
    return {
        "defocus": 2 * r2 - 1,                       # Z_2^0  (through-focal knob shape)
        "astig0": uu ** 2 - vv ** 2,                  # Z_2^2
        "astig45": 2 * uu * vv,                       # Z_2^-2
        "comaX": (3 * r2 - 2) * uu,                   # Z_3^1
        "comaY": (3 * r2 - 2) * vv,                   # Z_3^-1
        "spher": 6 * r2 ** 2 - 6 * r2 + 1,            # Z_4^0
        "trefoilX": uu ** 3 - 3 * uu * vv ** 2,       # Z_3^3
        "trefoilY": 3 * uu ** 2 * vv - vv ** 3,       # Z_3^-3
    }


def coeffs_to_tensor(coeffs, *, device=None, dtype=torch.float64):
    """Accept a dict ``{name: value}`` or a length-``len(ABERRATIONS)`` tensor; return tensor."""
    if torch.is_tensor(coeffs):
        return coeffs.to(device=device, dtype=dtype) if device or coeffs.dtype != dtype else coeffs
    vec = [float(coeffs.get(k, 0.0)) for k in ABERRATIONS]
    return torch.tensor(vec, device=device, dtype=dtype)


def aberrated_psf(coeffs, *, defocus=0.0, grid_size=64, extent=1.4, apodization=1.5,
                  device=None, dtype=torch.float64):
    """Differentiable objective PSF for a Zernike-aberrated, Gaussian-apodized pupil.

    ``coeffs`` -- dict or length-``len(ABERRATIONS)`` tensor of wavefront amplitudes (rad);
    ``defocus`` -- known controllable defocus setting (the phase-diversity knob);
    ``apodization`` -- Gaussian taper strength modelling the CRL absorption profile.
    Returns a ``(grid_size, grid_size)`` PSF normalized to unit sum. Gradients flow to
    ``coeffs`` (and ``defocus``), so this PSF can be fit and back-propagated.
    """
    c = coeffs_to_tensor(coeffs, device=device, dtype=dtype)
    uu, vv, r2, aperture = _pupil_grid(grid_size, extent, device=c.device, dtype=c.dtype)
    apod = torch.exp(-apodization * r2) * aperture
    Z = zernike_terms(uu, vv)
    defocus = torch.as_tensor(defocus, device=c.device, dtype=c.dtype)
    phase = defocus * Z["defocus"]
    for i, name in enumerate(ABERRATIONS):
        phase = phase + c[i] * Z[name]
    pupil = apod * torch.exp(1j * phase.to(torch.complex128 if c.dtype == torch.float64 else torch.complex64))
    amp = torch.fft.fftshift(torch.fft.fft2(pupil))
    psf = amp.real ** 2 + amp.imag ** 2
    return psf / psf.sum()


def convolve_psf(image, psf):
    """FFT-based, same-size, centered convolution of ``image`` with ``psf`` (both ``(H, W)``).

    Differentiable in both arguments; a centered delta PSF returns ``image`` unchanged.
    ``psf`` is embedded/centered to the image size if smaller.
    """
    H, W = image.shape[-2:]
    if psf.shape[-2:] != (H, W):
        big = torch.zeros(H, W, device=psf.device, dtype=psf.dtype)
        ph, pw = psf.shape[-2:]
        r0, c0 = (H - ph) // 2, (W - pw) // 2
        big[r0:r0 + ph, c0:c0 + pw] = psf
        psf = big
    Fi = torch.fft.fft2(image.to(psf.dtype))
    Fp = torch.fft.fft2(torch.fft.ifftshift(psf))
    return torch.fft.ifft2(Fi * Fp).real


def wiener_deconvolve(image, psf, *, reg=1e-3):
    """Regularized (Wiener) deconvolution of ``image`` by a KNOWN ``psf``.

    Used to fold a *fitted* objective PSF back into the reconstruction: given the
    self-calibrated aberration, deblur the measured maps before recovering F. ``reg`` is
    the noise-to-signal regularization (larger = smoother). Not for gradients -- a
    correction step that consumes the fitted PSF.
    """
    H, W = image.shape[-2:]
    if psf.shape[-2:] != (H, W):
        image = image  # psf embedded below by convolve-style centering
        big = torch.zeros(H, W, device=psf.device, dtype=psf.dtype)
        ph, pw = psf.shape[-2:]
        r0, c0 = (H - ph) // 2, (W - pw) // 2
        big[r0:r0 + ph, c0:c0 + pw] = psf
        psf = big
    Fi = torch.fft.fft2(image.to(psf.dtype))
    Fp = torch.fft.fft2(torch.fft.ifftshift(psf))
    Wf = Fp.conj() / (Fp.conj() * Fp + reg)
    return torch.fft.ifft2(Fi * Wf).real


def fit_aberration(measured, defocus_list, scene, *, init=None, apodization=1.5,
                   grid_size=None, extent=1.4, steps=300, lr=0.05, dtype=torch.float64,
                   device=None):
    """Self-calibrate objective aberrations from a through-focal series (phase diversity).

    ``measured`` -- ``(K, H, W)`` stack, one image per defocus in ``defocus_list``;
    ``scene`` -- the known calibration object intensity ``(H, W)`` (e.g. a reference
    feature) that is imaged; ``init`` -- optional starting coefficients (dict/tensor).
    Fits the ``len(ABERRATIONS)`` Zernike amplitudes so ``scene (*) PSF(c, defocus_k)``
    matches each measured frame, by gradient descent through :func:`aberrated_psf`.
    Returns the fitted coefficient tensor. A single defocus is insufficient for the even
    aberrations -- pass a through-focal ``defocus_list`` (see module docstring).
    """
    measured = torch.as_tensor(measured, dtype=dtype, device=device)
    scene = torch.as_tensor(scene, dtype=dtype, device=device)
    H, W = scene.shape[-2:]
    gs = grid_size or H
    c0 = (coeffs_to_tensor(init, device=device, dtype=dtype) if init is not None
          else torch.zeros(len(ABERRATIONS), dtype=dtype, device=device))
    c = c0.clone().detach().requires_grad_(True)

    def loss_fn():
        total = scene.new_zeros(())
        for k, d in enumerate(defocus_list):
            psf = aberrated_psf(c, defocus=float(d), grid_size=gs, extent=extent,
                                apodization=apodization, device=device, dtype=dtype)
            pred = convolve_psf(scene, psf)
            total = total + ((pred - measured[k]) ** 2).mean()
        return total

    _fit([c], loss_fn, steps=steps, lr=lr)
    return c.detach()
