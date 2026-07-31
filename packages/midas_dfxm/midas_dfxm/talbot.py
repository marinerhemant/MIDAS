"""Talbot grating-interferometry wavefront sensor for the 6-ID-C microscope.

Qiao et al. (Rev. Sci. Instrum. 91, 113703, 2020) characterised their polymer optics with a
single-grating Talbot interferometer: a checkerboard pi/2 phase grating (4.8 um period, 45 deg
rotation) is placed after the optic, and the self-imaged grating pattern on the detector is
distorted by the optic's wavefront. Local fringe displacement is proportional to the wavefront
GRADIENT, so demodulating the interferogram recovers the differential phase; integrating it and
fitting Zernike polynomials gives the aberration -- the same coefficients the objective PSF in
:mod:`midas_dfxm.aberration` needs. This module is the differentiable forward (grating +
Fresnel propagation) and the matching inverse (Fourier demodulation + integration).

Physics. A ray at wavefront gradient ``dW/dx`` is deflected by that angle; propagating a
distance ``z`` shifts the grating self-image laterally by ``z*dW/dx``, i.e. by a fringe phase
``2*pi*(z/p)*dW/dx``. Recover that phase by isolating the grating's first harmonic in the FFT
of the interferogram (a reference, flat-wavefront interferogram fixes the carrier). Everything
is torch/complex128 and differentiable in the wavefront.
"""
from __future__ import annotations

import math

import torch

from .polymer import _HC_KEV_A


def wavelength_um(energy_keV) -> float:
    """X-ray wavelength in micrometres."""
    return (_HC_KEV_A / energy_keV) * 1e-4


def checkerboard_grating(n, dx_um, period_um, *, phase_rad=math.pi / 2, angle_deg=45.0,
                         dtype=torch.float64, device=None) -> torch.Tensor:
    """Complex transmission of a checkerboard phase grating (``n x n``, pixel ``dx_um``)."""
    xs = (torch.arange(n, dtype=dtype, device=device) - n // 2) * dx_um
    x, y = torch.meshgrid(xs, xs, indexing="xy")
    a = math.radians(angle_deg)
    xr = x * math.cos(a) + y * math.sin(a)
    yr = -x * math.sin(a) + y * math.cos(a)
    checker = (torch.sin(2 * math.pi * xr / period_um) *
               torch.sin(2 * math.pi * yr / period_um) > 0).to(dtype)   # 0/1 tiles
    cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
    return torch.exp(1j * phase_rad * checker.to(cdt))


def line_grating(n, dx_um, period_um, *, phase_rad=math.pi / 2, angle_deg=0.0,
                 dtype=torch.float64, device=None) -> torch.Tensor:
    """Complex transmission of a 1D line phase grating (modulation along ``angle_deg``).

    ``angle_deg=0`` modulates along x (carrier on the x-axis); ``90`` modulates along y. Two
    orthogonal line gratings give clean, separable x- and y-differential phase; the checkerboard
    (:func:`checkerboard_grating`) is their single-shot 2D combination.
    """
    xs = (torch.arange(n, dtype=dtype, device=device) - n // 2) * dx_um
    x, y = torch.meshgrid(xs, xs, indexing="xy")
    a = math.radians(angle_deg)
    xr = x * math.cos(a) + y * math.sin(a)
    g = (torch.sin(2 * math.pi * xr / period_um) > 0).to(dtype)
    cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
    return torch.exp(1j * phase_rad * g.to(cdt))


def fresnel_propagate(field, dx_um, z_um, lam_um) -> torch.Tensor:
    """Angular-spectrum (Fresnel) propagation of a complex field by ``z_um``."""
    n = field.shape[-1]
    f = torch.fft.fftfreq(n, d=dx_um, device=field.device)
    fx, fy = torch.meshgrid(f, f, indexing="xy")
    H = torch.exp(-1j * math.pi * lam_um * z_um * (fx ** 2 + fy ** 2)).to(field.dtype)
    return torch.fft.ifft2(torch.fft.fft2(field) * H)


def talbot_distance_um(period_um, lam_um) -> float:
    """Full Talbot distance ``d_T = 2 p^2 / lambda`` (self-image period along z)."""
    return 2.0 * period_um ** 2 / lam_um


def interferogram(wavefront_nm, grating, z_um, dx_um, lam_um) -> torch.Tensor:
    """Talbot interferogram intensity for an incident wavefront (nm) through ``grating``.

    ``wavefront_nm`` is an ``(n, n)`` optical-path map; a flat (zeros) input gives the
    reference interferogram. Differentiable in ``wavefront_nm``.
    """
    W_um = torch.as_tensor(wavefront_nm, dtype=torch.float64, device=grating.device) * 1e-3
    U0 = torch.exp(1j * (2 * math.pi / lam_um) * W_um.to(grating.dtype))
    Uz = fresnel_propagate(U0 * grating, dx_um, z_um, lam_um)
    return Uz.real ** 2 + Uz.imag ** 2


def _carrier_indices(ref, period_um, dx_um):
    """Pixel indices of the grating's first-harmonic peaks (x- and y-carrier) in the FFT."""
    n = ref.shape[-1]
    F = torch.fft.fftshift(torch.fft.fft2(ref - ref.mean())).abs()
    c = n // 2
    guard = max(2, int(round(n * dx_um / period_um / 4)))
    F[c - guard:c + guard + 1, c - guard:c + guard + 1] = 0        # kill the DC lobe
    # x-carrier: strongest peak in the right half-plane; y-carrier: upper half-plane
    right = F.clone(); right[:, :c] = 0
    upper = F.clone(); upper[:c, :] = 0
    ix = divmod(int(torch.argmax(right)), n)
    iy = divmod(int(torch.argmax(upper)), n)
    return ix, iy


def _demod_one(I, peak, n):
    """Complex demodulation of a single carrier: window the FFT around ``peak``, recenter, IFFT."""
    F = torch.fft.fftshift(torch.fft.fft2(I))
    c = n // 2
    win = torch.zeros_like(F)
    r = max(3, n // 16)
    py, px = peak
    win[py - r:py + r + 1, px - r:px + r + 1] = 1.0
    band = F * win
    band = torch.roll(band, shifts=(c - py, c - px), dims=(0, 1))   # recenter carrier to DC
    return torch.fft.ifft2(torch.fft.ifftshift(band))


def recover_gradient(I, I_ref, period_um, dx_um):
    """Wavefront gradients ``(dW/dx, dW/dy)`` (dimensionless slope) from the interferogram.

    Uses the reference interferogram to fix the carrier and the zero-phase baseline. The
    differential fringe phase is ``2*pi*(z/p)*dW`` -- but calibrated directly against a known
    tilt is unnecessary here: we return the *slope* by dividing the unwrapped phase by the
    carrier frequency and propagation is folded into the caller's ``z`` via :func:`slope_scale`.
    """
    n = I.shape[-1]
    ix, iy = _carrier_indices(I_ref, period_um, dx_um)
    gx = torch.angle(_demod_one(I, ix, n) * _demod_one(I_ref, ix, n).conj())
    gy = torch.angle(_demod_one(I, iy, n) * _demod_one(I_ref, iy, n).conj())
    return gx, gy


def differential_phase(I, I_ref, period_um, dx_um, axis="x") -> torch.Tensor:
    """Differential fringe phase along one axis (for a 1D line grating -> ``dW`` in that axis).

    Demodulates the single grating carrier (``axis='x'`` or ``'y'``) against the flat-wavefront
    reference. Multiply by the tilt-calibrated scale to get the dimensionless wavefront slope.
    """
    n = I.shape[-1]
    ix, iy = _carrier_indices(I_ref, period_um, dx_um)
    peak = ix if axis == "x" else iy
    return torch.angle(_demod_one(I, peak, n) * _demod_one(I_ref, peak, n).conj())


def integrate_gradient(gx, gy, dx_um) -> torch.Tensor:
    """Frankot-Chellappa least-squares integration of a gradient field -> surface (same units*dx)."""
    n = gx.shape[-1]
    f = torch.fft.fftfreq(n, d=dx_um, device=gx.device)
    fx, fy = torch.meshgrid(f, f, indexing="xy")
    denom = (fx ** 2 + fy ** 2)
    denom[0, 0] = 1.0
    num = (-1j * fx * torch.fft.fft2(gx) - 1j * fy * torch.fft.fft2(gy))
    W = torch.fft.ifft2(num / (2 * math.pi * denom)).real
    W[0, 0] = 0.0
    return W - W.mean()
