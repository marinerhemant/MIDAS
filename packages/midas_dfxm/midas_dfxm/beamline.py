"""Differentiable beamline digital twin: undulator -> illumination -> sample -> objective -> detector.

A differentiable model of the DFXM instrument. The synchrotron/optics community's tools
(SHADOW, SRW, McXtrace, OASYS) are not differentiable; making the beamline differentiable
lets us **optimize** the design, **correct** aberrations/misalignments from calibration
data, and **co-design** optics + acquisition -- by gradient descent.

Chain (both optical sides couple through the reciprocal-space resolution function):

    undulator/source --illumination optics--> sample (F, defects) --OBJECTIVE CRL--> detector

The compound refractive lens (CRL) OBJECTIVE sits on the *diffracted* beam (post-sample)
and images the illuminated volume onto the detector; the *illumination* optics (pre-sample)
set the incident-beam divergence and energy bandwidth. The CRL is modelled by exact
ray-transfer (ABCD) matrices through the lens stack -- verified against Poulsen 2017
(single-lens f1 = R/2delta = 21.19 m and thick-lens f_eff = f1 phi/sin(N phi) at N=69,
R=50 um, 17 keV). Everything is torch-differentiable in the design parameters
(lens count/radius/spacing, energy, distances, divergence, bandwidth).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .resolution import poulsen_resolution_widths

# Be refractive-index decrement, anchored to Poulsen 2017 (delta(17 keV)=1.18e-6 gives
# f1 = R/2delta = 21.19 m at R=50 um); delta ~ 1/E^2.
_DELTA_BE_17KEV = 1.18e-6


def delta_beryllium(energy_keV):
    """Refractive-index decrement of Be vs energy (``delta ~ 1/E^2``)."""
    E = torch.as_tensor(energy_keV, dtype=torch.get_default_dtype()) if not torch.is_tensor(energy_keV) else energy_keV
    return _DELTA_BE_17KEV * (17.0 / E) ** 2


def _lens(f):
    z = torch.zeros((), dtype=f.dtype, device=f.device); o = torch.ones((), dtype=f.dtype, device=f.device)
    return torch.stack([torch.stack([o, z]), torch.stack([-1.0 / f, o])])


def _drift(d):
    d = torch.as_tensor(d, dtype=torch.get_default_dtype()) if not torch.is_tensor(d) else d
    z = torch.zeros((), dtype=d.dtype, device=d.device); o = torch.ones((), dtype=d.dtype, device=d.device)
    return torch.stack([torch.stack([o, d]), torch.stack([z, o])])


def crl_abcd(n_lenses: int, radius_um, energy_keV, spacing_m=1.6e-3):
    """Ray-transfer ABCD matrix of an ``n_lenses`` Be CRL objective; also returns ``f1``.

    ``f1 = R/(2 delta)`` is the single-lens focal length; the stack matrix is the ordered
    product of (thin-lens, drift) per lenslet. Differentiable in radius/energy/spacing.
    """
    R = torch.as_tensor(radius_um, dtype=torch.float64) * 1e-6 if not torch.is_tensor(radius_um) else radius_um * 1e-6
    delta = delta_beryllium(energy_keV)
    f1 = R / (2.0 * delta)
    T = _drift(spacing_m); L = _lens(f1)
    M = torch.eye(2, dtype=f1.dtype, device=f1.device)
    step = T @ L
    for _ in range(int(n_lenses)):
        M = step @ M
    return M, f1


def crl_focal_length(n_lenses, radius_um, energy_keV, spacing_m=1.6e-3):
    """Effective (thick-lens) focal length of the CRL stack, ``f_eff = -1/C``."""
    M, _ = crl_abcd(n_lenses, radius_um, energy_keV, spacing_m)
    return -1.0 / M[1, 0]


def crl_image(n_lenses, radius_um, energy_keV, object_distance_m, spacing_m=1.6e-3):
    """Image distance and magnification for an object at ``object_distance_m`` before the CRL.

    Solves the imaging condition ``B_total = 0`` for the full system
    ``drift(q) @ ABCD @ drift(p)``; returns ``(image_distance q, magnification M)``.
    Differentiable -- the object of the ``optimize``/``correct`` demos.
    """
    M, _ = crl_abcd(n_lenses, radius_um, energy_keV, spacing_m)
    p = torch.as_tensor(object_distance_m, dtype=M.dtype, device=M.device)
    S0 = M @ _drift(p)                                  # ABCD @ drift(p)
    q = -S0[0, 1] / S0[1, 1]                            # require B_total=0
    mag = S0[0, 0] + q * S0[1, 0]
    return q, mag


def crl_na(n_lenses, radius_um, energy_keV, spacing_m=1.6e-3):
    """Geometric numerical aperture ``sigma_a ~ R / f_eff`` (rms) of the objective.

    A first-order (geometric-aperture) estimate; the absorption-limited effective aperture
    is a refinement. Larger for shorter ``f_eff`` (more/stronger lenses). Differentiable.
    """
    R = torch.as_tensor(radius_um, dtype=torch.float64) * 1e-6 if not torch.is_tensor(radius_um) else radius_um * 1e-6
    return R / crl_focal_length(n_lenses, radius_um, energy_keV, spacing_m)


@dataclass
class Illumination:
    """Pre-sample illumination (source + condenser): what reaches the sample."""
    divergence_v: float = 0.53e-3     # rms vertical divergence (rad)
    divergence_h: float = 0.53e-3     # rms horizontal divergence (rad)
    bandwidth: float = 1.4e-4         # rms energy spread dE/E


def beamline_resolution(illumination: Illumination, n_lenses, radius_um, energy_keV,
                        q_mag, two_theta_deg, spacing_m=1.6e-3) -> dict:
    """Reciprocal-space resolution widths from the FULL chain (illumination + objective).

    Couples the pre-sample divergence/bandwidth (``illumination``) with the post-sample
    objective NA (from the CRL) through the Poulsen 2017 covariance -- so the resolution
    is a differentiable function of every optical design parameter. Returns the
    ``{sigma_rock, sigma_roll, sigma_par}`` dict.
    """
    na = float(crl_na(n_lenses, radius_um, energy_keV, spacing_m))
    return poulsen_resolution_widths(q_mag, two_theta_deg=two_theta_deg,
                                     div_v=illumination.divergence_v, div_h=illumination.divergence_h,
                                     na=na, eps=illumination.bandwidth)
