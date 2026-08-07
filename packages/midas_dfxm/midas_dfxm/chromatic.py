"""Chromatic (pink-beam) objective imaging: the longitudinal chromatic aberration of the
refractive (CRL) objective, and the effective point-spread function it produces.

For a broad-bandwidth beam the CRL objective focuses at ``f ~ 1/E^2`` (:mod:`midas_dfxm.beamline`),
so with the detector fixed on the centre-energy image plane every other energy is defocused. Because
the crystal-side dispersion of the diffracted wave is negligible at DFXM bandwidths (extinction-length
and absorption dispersion are dominated by the deviation blur -- see :mod:`midas_dfxm.pink`), the
*binding* pink-beam effect on the image is this chromatic defocus of the objective. The chromatic
image is the incoherent, spectrum-weighted sum of the per-energy defocused images; for an incoherent
object it is a convolution with the **effective chromatic PSF**

    h_eff(x) = sum_i S(lambda_i) |PSF(x; defocus=coeff_i)|^2 ,

a sharp diffraction-limited core (the in-focus centre energy) on a broad defocused pedestal (the band
tails). Every operator is torch-differentiable, so ``S(lambda)`` can be *recovered* from the PSF.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .aberration import aberrated_psf
from .beamline import chromatic_defocus_coeffs, crl_na


def effective_chromatic_psf(
    defocus_coeffs, weights, *, NA: float, wavelength_A: float,
    grid_size: int = 256, extent: float = 1.4, apodization: float = 1.5,
    aberr_coeffs=None,
) -> Tuple[torch.Tensor, float]:
    """Effective chromatic PSF ``h_eff = sum_i w_i |PSF(defocus_i)|^2`` and its object-space pixel.

    ``defocus_coeffs`` (rad) and ``weights`` (``S(lambda)``, sum 1) are aligned per energy. Returns
    ``(h_eff, dx_um)`` with ``h_eff`` normalised to unit sum and ``dx_um`` the object-space micron
    pitch of a PSF pixel (``lambda / (2 extent NA)``). The <2% NA/wavelength variation across a
    DFXM band is neglected in the pixel scale (only the defocus disperses), which the validation
    tolerates. Differentiable in ``weights`` (hence invertible for the spectrum) and ``defocus_coeffs``.
    """
    w = weights if torch.is_tensor(weights) else torch.as_tensor(weights, dtype=torch.float64)
    w = w / w.sum()
    h = None
    for coeff, wi in zip(defocus_coeffs, w):
        psf = aberrated_psf(aberr_coeffs if aberr_coeffs is not None else {}, defocus=coeff,
                            grid_size=grid_size, extent=extent, apodization=apodization)
        term = wi * psf
        h = term if h is None else h + term
    h = h / h.sum()
    dx_um = (wavelength_A * 1e-4) / (2.0 * extent * NA)
    return h, dx_um


def chromatic_psf_from_spectrum(
    energies_keV, weights, *, E0_keV: float, n_lenses: int, radius_um: float,
    object_distance_m: float, spacing_m: float = 1.6e-3,
    grid_size: int = 256, extent: float = 1.4, apodization: float = 1.5, aberr_coeffs=None,
) -> Tuple[torch.Tensor, float]:
    """Effective chromatic PSF straight from a spectrum + CRL objective design.

    Computes the per-energy defocus (:func:`beamline.chromatic_defocus_coeffs`) from the CRL
    chromaticity, then the incoherent PSF sum (:func:`effective_chromatic_psf`). ``NA`` and the pixel
    scale use the centre energy. Differentiable in ``weights`` and the CRL design parameters.
    """
    coeffs = chromatic_defocus_coeffs(energies_keV, E0_keV=E0_keV, n_lenses=n_lenses,
                                      radius_um=radius_um, object_distance_m=object_distance_m,
                                      spacing_m=spacing_m)
    NA = float(crl_na(n_lenses, radius_um, E0_keV, spacing_m))
    lam0 = 12.398419739 / E0_keV
    return effective_chromatic_psf(coeffs, weights, NA=NA, wavelength_A=lam0,
                                   grid_size=grid_size, extent=extent, apodization=apodization,
                                   aberr_coeffs=aberr_coeffs)


def psf_fwhm_um(h: torch.Tensor, dx_um: float) -> float:
    """FWHM (microns) of a PSF from its central row."""
    import numpy as np
    line = h.detach().cpu().numpy()
    line = line[line.shape[0] // 2]
    line = line / line.max()
    idx = np.where(line >= 0.5)[0]
    return float((idx[-1] - idx[0]) * dx_um) if idx.size > 1 else 0.0
