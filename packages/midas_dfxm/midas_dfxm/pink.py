"""Spectrum-integrated (pink-beam) dynamical DFXM forward.

The monochromatic Takagi--Taupin forward (:mod:`midas_dfxm.takagi_taupin`) is extended to a
broad-bandwidth (pink) beam by integrating over the illumination spectrum ``S(E)``: each energy
sample is a *full* mono dynamical evaluation at its **own** wavelength, so the extinction length
``Lambda(lambda)``, the photoelectric absorption, and the Bragg condition all disperse across the
band; the pink intensity is the incoherent, spectrum-weighted sum

    I_pink = sum_i  w_i  R( chi(lambda_i), t, y_i ; lambda_i ) .

This is the dynamical (+imaging, see :mod:`midas_dfxm.beamline`) counterpart of the kinematical
spectrum-aware HEDM forward in ``midas-pink``; a ``midas_pink.ParameterisedSpectrum`` plugs in
directly as ``S(lambda)`` via :func:`pink_dynamical_reflectivity`'s ``spectrum`` argument.

Setting ``disperse=False`` freezes ``chi`` and ``Lambda`` at the centre wavelength and reproduces
the older "pure deviation blur" pink model -- so the difference between the two isolates exactly the
extinction-length / absorption dispersion the deviation blur omits.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .takagi_taupin import (
    susceptibility_fourier, bragg_reflectivity, diffracted_intensity,
)

H_C_KEV_A = 12.398419739  # lambda[A] = H_C_KEV_A / E[keV]


def spectrum_grid(E0_keV: float, bandwidth: float, n_lambda: int = 21,
                  shape: str = "gaussian", n_sigma: float = 3.0,
                  dtype=torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(energies_keV, lambdas_A, weights)`` for a pink spectrum centred on ``E0_keV``.

    ``bandwidth`` is the rms fractional energy spread ``dE/E`` for ``shape='gaussian'`` (the grid
    spans ``+-n_sigma*bandwidth``), or the half-width for ``shape='boxcar'``. ``weights`` sum to 1.
    ``bandwidth=0`` or ``n_lambda=1`` gives the monochromatic grid (single sample, weight 1).
    """
    if n_lambda <= 1 or bandwidth <= 0.0:
        E = torch.tensor([E0_keV], dtype=dtype)
        return E, H_C_KEV_A / E, torch.ones(1, dtype=dtype)
    if shape == "boxcar":
        rel = torch.linspace(-bandwidth, bandwidth, n_lambda, dtype=dtype)
        w = torch.ones(n_lambda, dtype=dtype)
    elif shape == "gaussian":
        rel = torch.linspace(-n_sigma * bandwidth, n_sigma * bandwidth, n_lambda, dtype=dtype)
        w = torch.exp(-0.5 * (rel / bandwidth) ** 2)
    else:
        raise ValueError("shape must be 'gaussian' or 'boxcar'")
    E = E0_keV * (1.0 + rel)
    return E, H_C_KEV_A / E, w / w.sum()


def pink_dynamical_reflectivity(
    crystal, hkl, *, thickness_um: float, theta_B_deg: float, E0_keV: float,
    bandwidth: float = 0.0, n_lambda: int = 21, y0: float = 0.0,
    geometry: str = "bragg", shape: str = "gaussian", C: float = 1.0,
    disperse: bool = True, absorption: bool = True, spectrum=None,
    y_cut: float = 60.0, **kw,
):
    """Pink-beam dynamical reflectivity/intensity at deviation ``y0`` (scan ``y0`` for a curve).

    Each spectral sample is evaluated with its own ``chi(lambda_i)`` (hence ``Lambda(lambda_i)`` and
    absorption) and its own geometric Bragg-shift deviation offset, then incoherently summed with
    ``S(lambda)``. ``geometry='bragg'`` returns ``|X|^2|gamma_h/gamma_0|``; ``'laue'`` returns
    ``|D_h|^2``. Pass ``spectrum`` (anything returning ``(E_keV, lambda_A, weights)`` from a no-arg
    call, e.g. ``midas_pink.ParameterisedSpectrum``) to override ``(bandwidth, n_lambda, shape)``.
    Differentiable in ``thickness_um``, ``y0``, and (through ``spectrum``) the spectral weights.
    """
    if spectrum is not None:
        energies, lambdas, weights = spectrum()
    else:
        energies, lambdas, weights = spectrum_grid(E0_keV, bandwidth, n_lambda, shape)
    lam0 = H_C_KEV_A / E0_keV
    s2 = math.sin(math.radians(theta_B_deg)) ** 2
    c0_0, ch_0, chb_0 = susceptibility_fourier(crystal, hkl, wavelength_A=lam0,
                                               absorption=absorption)
    total = None
    for lam, w in zip(lambdas.tolist(), weights):
        if disperse:
            c0, ch, chb = susceptibility_fourier(crystal, hkl, wavelength_A=float(lam),
                                                 absorption=absorption)
            lam_eval = float(lam)
        else:                                                # freeze Lambda/absorption at lam0
            c0, ch, chb, lam_eval = c0_0, ch_0, chb_0, lam0
        dy = 2.0 * s2 * ((lam - lam0) / lam0) / (C * abs(complex(ch)))   # geometric Bragg shift
        y = y0 + dy
        if abs(float(y)) > y_cut:      # this energy's Bragg condition is far from the setting: it
            continue                    # does not diffract here (R ~ 1/y^2 -> 0); skip (avoids the
            #                             large-|y| Riccati overflow, physically negligible).
        if geometry == "bragg":
            r = bragg_reflectivity(c0, ch, chb, thickness_um=thickness_um, y=y,
                                   theta_B_deg=theta_B_deg, wavelength_A=lam_eval, C=C, **kw)
        else:
            r = diffracted_intensity(c0, ch, chb, thickness_um=thickness_um, y=y,
                                     theta_B_deg=theta_B_deg, wavelength_A=lam_eval, C=C, **kw)
        term = w * r
        total = term if total is None else total + term
    if total is None:                  # no energy diffracts at this setting
        return torch.zeros((), dtype=torch.float64)
    return total
