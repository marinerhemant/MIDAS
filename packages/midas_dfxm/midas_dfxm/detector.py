"""Realistic detector chain for DFXM images (R1.1 of the robustness plan).

Turns an ideal rendered intensity into a camera-realistic image, so the inverse is
tested against data like a real pco.edge sCMOS at ID03 rather than clean simulations.
Follows the detector models of Carlsen 2022 (Acta A78, section 2.5) and Henningsson
2025 (JMPS, Eqs. 31-34): point-spread blur, then Poisson shot noise (SNR = sqrt(counts)),
additive Gaussian thermal/read noise, and 16-bit binning of the finite dynamic range.

The point-spread step is differentiable (a separable Gaussian convolution), so the
noise-free path carries gradients for the forward/inverse; the Poisson/thermal draws are
stochastic corruption used to stress-test recovery. Device/dtype-portable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_psf_kernel(sigma_px: float, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """Normalized 2-D Gaussian PSF kernel; radius = ceil(3 sigma)."""
    r = max(1, int(3 * sigma_px + 0.5))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma_px) ** 2)
    k = torch.outer(g, g)
    return k / k.sum()


def apply_psf(image: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Convolve ``image`` ``(H, W)`` with a Gaussian point spread (reflect-padded).

    Differentiable in ``image``. Models the scintillator + visible-optics blur that sets
    the effective DFXM spatial resolution beyond the geometric pixel.
    """
    if sigma_px <= 0:
        return image
    k = gaussian_psf_kernel(sigma_px, device=image.device, dtype=image.dtype)
    r = k.shape[0] // 2
    x = image[None, None]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    return F.conv2d(x, k[None, None])[0, 0]


def add_detector_noise(
    image: torch.Tensor,
    *,
    peak_counts: float = 6e4,
    thermal_offset: float = 99.0,
    thermal_sigma: float = 2.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Scale to counts, add Poisson shot (SNR=sqrt(g)) + Gaussian thermal noise.

    ``peak_counts`` sets the exposure so the brightest pixel is near full well (real
    ID03 runs use close to the 16-bit range). ``thermal_offset``/``thermal_sigma`` are
    the pco camera's fixed offset and read-noise std (Henningsson 2025 uses ~99 / 2.3).
    Stochastic (not differentiable) -- corruption for inverse stress-tests.
    """
    scaled = image / (image.amax() + 1e-30) * peak_counts
    shot = torch.poisson(scaled.clamp_min(0), generator=generator)
    thermal = thermal_offset + thermal_sigma * torch.randn(
        image.shape, device=image.device, dtype=image.dtype, generator=generator)
    return shot + thermal


def quantize_16bit(image: torch.Tensor, *, full_well: int = 65535) -> torch.Tensor:
    """Round and clamp to the camera's 16-bit dynamic range ``[0, full_well]``."""
    return image.round().clamp(0, full_well)


def detector_model(
    image: torch.Tensor,
    *,
    psf_sigma_px: float = 1.0,
    peak_counts: float = 6e4,
    thermal_offset: float = 99.0,
    thermal_sigma: float = 2.3,
    full_well: int = 65535,
    noise: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full detector chain: PSF -> (Poisson + thermal) -> 16-bit binning.

    With ``noise=False`` returns the differentiable PSF-blurred, exposure-scaled image
    (for gradients); with ``noise=True`` returns a camera-realistic 16-bit frame.
    """
    img = apply_psf(image, psf_sigma_px)
    if not noise:
        return img / (img.amax() + 1e-30) * peak_counts
    img = add_detector_noise(img, peak_counts=peak_counts, thermal_offset=thermal_offset,
                             thermal_sigma=thermal_sigma, generator=generator)
    return quantize_16bit(img, full_well=full_well)
