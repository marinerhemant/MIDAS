"""Instrument realism: detector projection, corrections, noise, resolution.

Turns a scattering-vector intensity into something that looks like a real
measurement, so the same differentiable forwards can be fit against actual
detector data with a proper likelihood.

* :func:`project_to_detector` -- Ewald-correct mapping of Q-vectors (1/A) to
  pixel coordinates on a flat area detector (incident beam along +z).
* :func:`solid_angle_polarization` -- per-pixel geometric correction.
* :func:`poisson_nll` / :func:`add_poisson_noise` -- count statistics: the right
  loss for photon-limited ultrafast data, and a sampler to make synthetic data.
* :func:`resolution_convolve` -- Gaussian instrument-resolution blur.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "project_to_detector",
    "solid_angle_polarization",
    "poisson_nll",
    "add_poisson_noise",
    "resolution_convolve",
]


def project_to_detector(q_vec, *, wavelength_A, distance_mm, pixel_mm,
                        beam_center=(0.0, 0.0)):
    """Map Q-vectors (1/A) to detector pixel coordinates (Ewald-correct).

    Incident beam ``k_i = (0, 0, 2 pi / lambda)``; elastic scattering gives
    ``k_f = k_i + Q`` and the ray hits a flat detector at ``z = distance``.

    Parameters
    ----------
    q_vec : tensor (..., 3)  in 1/A
    wavelength_A, distance_mm, pixel_mm : floats
    beam_center : (px, py)

    Returns
    -------
    pix : tensor (..., 2)
        Pixel coordinates (x, y).
    valid : bool tensor (...)
        True where the ray scatters forward onto the detector (k_f_z > 0).
    """
    import torch
    q_vec = torch.as_tensor(q_vec)
    k = 2.0 * math.pi / float(wavelength_A)
    ki = torch.tensor([0.0, 0.0, k], dtype=q_vec.dtype, device=q_vec.device)
    kf = q_vec + ki                                          # (..., 3)
    kf_z = kf[..., 2]
    valid = kf_z > 1e-9
    safe_z = torch.where(valid, kf_z, torch.ones_like(kf_z))
    x_mm = float(distance_mm) * kf[..., 0] / safe_z
    y_mm = float(distance_mm) * kf[..., 1] / safe_z
    px = x_mm / float(pixel_mm) + float(beam_center[0])
    py = y_mm / float(pixel_mm) + float(beam_center[1])
    return torch.stack([px, py], dim=-1), valid


def solid_angle_polarization(q_vec, *, wavelength_A, polarization=0.5):
    """Combined inverse solid-angle (cos^3 of scattering angle) and
    polarisation correction factor for a flat detector."""
    import torch
    q_vec = torch.as_tensor(q_vec)
    k = 2.0 * math.pi / float(wavelength_A)
    qmag = torch.linalg.vector_norm(q_vec, dim=-1)
    sin_th = torch.clamp(qmag / (2.0 * k), max=1.0)
    two_th = 2.0 * torch.asin(sin_th)
    cos2 = torch.cos(two_th)
    K = float(polarization)
    pol = (1.0 - K) + K * cos2 * cos2
    solid = torch.cos(two_th).clamp(min=1e-6) ** 3           # ~ 1/r^2 obliquity
    return pol * solid


def poisson_nll(pred_counts, obs_counts, *, eps=1e-9):
    """Poisson negative log-likelihood (up to a data-only constant):
    ``sum(pred - obs * log(pred))``.  Differentiable in ``pred``."""
    import torch
    pred = torch.clamp(torch.as_tensor(pred_counts), min=eps)
    obs = torch.as_tensor(obs_counts)
    return (pred - obs * torch.log(pred)).sum()


def add_poisson_noise(intensity, *, photons_per_peak=1e4, generator=None):
    """Scale an intensity to a target peak photon count and sample Poisson
    counts (returns a float tensor of counts)."""
    import torch
    I = torch.as_tensor(intensity)
    scale = float(photons_per_peak) / torch.clamp(I.max(), min=1e-12)
    lam = I * scale
    return torch.poisson(lam, generator=generator)


def resolution_convolve(profile, sigma_pts):
    """1-D Gaussian instrument-resolution blur (reflect-padded)."""
    import torch
    import torch.nn.functional as F
    profile = torch.as_tensor(profile)
    sigma = float(sigma_pts)
    if sigma <= 0:
        return profile
    radius = max(1, int(round(4 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=profile.dtype, device=profile.device)
    kern = torch.exp(-0.5 * (x / sigma) ** 2)
    kern = kern / kern.sum()
    p = profile.reshape(1, 1, -1)
    p = F.pad(p, (radius, radius), mode="reflect")
    out = F.conv1d(p, kern.reshape(1, 1, -1))
    return out.reshape(-1)
