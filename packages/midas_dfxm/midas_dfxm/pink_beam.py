"""Pink-beam DFXM: reciprocal-space broadening, intensity gain, deconvolvable resolution.

Implements the pink-beam DFXM resolution model of Quantifying Resolution in Pink
Beam Dark Field X-ray Microscopy (arXiv:2510.26665) and the 3D/4D pink-beam
imaging work (Comm. Mater. 2025, arXiv:2503.05921), both ESRF ID03. Replacing the
channel-cut monochromator (``eps = dE/E ~ 1e-4``) with a multilayer
monochromator (``eps ~ 1e-2``) buys ~1-2 orders of magnitude in flux (a reported
27x diffracted-intensity gain) at the cost of reciprocal-space resolution.

The longitudinal (axial-strain) reciprocal-space width follows

    dQ_par = |Q0|/2 * sqrt( (2*eps)^2 + cot(theta_B)^2 * (div^2 + na^2) )

so at pink-beam bandwidth the ``(2*eps)^2`` term dominates and dQ_par degrades by
~3-4x, while the rocking direction broadens by ~10x (chromatic blur through the
CRL stack). The published ID03 analysis Gaussian-fits mosaicity scans and reports
the **convolved** width; this module instead builds the pink-beam resolution
covariance so that :func:`midas_dfxm.fit_orientation_mosaicity` can **deconvolve**
it and return the intrinsic sample mosaic — the estimator upgrade the ID03
workflow does not perform.

Everything torch-differentiable and device/dtype-portable. Bandwidth ``eps`` is
the relative energy spread ``dE/E`` (dimensionless); angles in degrees; widths in
the same reciprocal units as the parent :class:`ResolutionFunction`.
"""
from __future__ import annotations

import math

import torch

from .resolution import ResolutionFunction, aligned_resolution

# Representative bandwidths (relative, dE/E) from the ID03 optics.
EPS_MONO = 1.4e-4   # Si channel-cut monochromator
EPS_PINK = 1.0e-2   # multilayer monochromator (pink beam)


def axial_reciprocal_width(
    q_mag,
    *,
    eps: float = EPS_PINK,
    two_theta_deg: float = 20.0,
    divergence: float = 0.0,
    na: float = 0.0,
) -> torch.Tensor:
    """Longitudinal (axial-strain) reciprocal-space width ``dQ_par`` for bandwidth ``eps``.

    ``dQ_par = |Q0|/2 * sqrt((2 eps)^2 + cot(theta_B)^2 (div^2 + na^2))`` from
    arXiv:2510.26665, with Bragg angle ``theta_B = two_theta/2``. ``q_mag`` may be a
    scalar or tensor. Differentiable in ``q_mag``, ``eps``, and the divergences —
    the pink-beam term that limits elastic-strain resolution.
    """
    q = torch.as_tensor(q_mag, dtype=torch.get_default_dtype()) if not torch.is_tensor(q_mag) else q_mag
    theta_b = torch.deg2rad(torch.as_tensor(two_theta_deg, dtype=q.dtype, device=q.device) / 2.0)
    cot = torch.cos(theta_b) / torch.sin(theta_b)
    ang = divergence ** 2 + na ** 2
    return 0.5 * torch.abs(q) * torch.sqrt((2.0 * eps) ** 2 + cot ** 2 * ang)


def intensity_gain(eps_pink: float = EPS_PINK, eps_mono: float = EPS_MONO, *, cap: float | None = None) -> float:
    """Approximate diffracted-intensity gain of pink vs monochromatic beam.

    To first order the integrated diffracted intensity scales with the accepted
    energy bandwidth, so the gain is ``eps_pink / eps_mono`` (the ID03 papers
    report ~27x in practice, lower than the raw bandwidth ratio because of optics
    throughput and the mosaic-limited acceptance — pass ``cap`` to clamp to a
    measured value). Returns a dimensionless factor.
    """
    g = float(eps_pink) / float(eps_mono)
    return min(g, cap) if cap is not None else g


def pink_beam_resolution(
    q_nom: torch.Tensor,
    *,
    eps: float = EPS_PINK,
    two_theta_deg: float = 20.0,
    sigma_perp_mono: float = 5e-3,
    rock_broaden: float = 10.0,
    divergence: float = 0.0,
    na: float = 0.0,
) -> ResolutionFunction:
    """Build a pink-beam :class:`ResolutionFunction` centred on ``q_nom``.

    The longitudinal width is set by :func:`axial_reciprocal_width` (bandwidth
    dominated); the transverse width is the monochromatic value scaled by
    ``rock_broaden`` (~10x chromatic blur through the objective, arXiv:2510.26665).
    Feed the result to :func:`midas_dfxm.dfxm_image` / the forward model to simulate
    pink-beam images, or use :func:`pink_beam_res_cov` to deconvolve it in a fit.
    """
    q_mag = torch.linalg.vector_norm(q_nom)
    sigma_par = float(axial_reciprocal_width(
        q_mag, eps=eps, two_theta_deg=two_theta_deg, divergence=divergence, na=na))
    return aligned_resolution(q_nom, sigma_par=sigma_par, sigma_perp=sigma_perp_mono * rock_broaden)


def pink_beam_res_cov(
    sigma_rock_deg: float,
    sigma_roll_deg: float,
    *,
    rock_broaden: float = 10.0,
    roll_broaden: float = 1.2,
):
    """Pink-beam angular resolution covariance ``(2, 2)`` for mosaicity-scan deconvolution.

    Scales the monochromatic rocking / rolling widths (degrees) by the pink-beam
    broadening factors (~10x rocking, ~1.2x rolling; arXiv:2510.26665) and returns
    the diagonal covariance ``diag(sigma_rock^2, sigma_roll^2)`` in deg^2, ready to
    pass as ``res_cov`` to :func:`midas_dfxm.fit_orientation_mosaicity`. Because the
    fit adds ``Sigma_mosaic + Sigma_res`` and solves for ``Sigma_mosaic``, this
    **deconvolves** the (large) pink-beam instrument width — recovering the intrinsic
    sample mosaic that the ID03 Gaussian-fit COM/FWHM analysis reports convolved.
    """
    sr = sigma_rock_deg * rock_broaden
    sp = sigma_roll_deg * roll_broaden
    return [[sr ** 2, 0.0], [0.0, sp ** 2]]


def strain_resolution_ratio(
    eps_pink: float = EPS_PINK,
    eps_mono: float = EPS_MONO,
    *,
    two_theta_deg: float = 20.0,
    divergence: float = 7e-4,
    na: float = 7e-4,
) -> float:
    """Ratio ``dQ_par(pink) / dQ_par(mono)`` — the axial-strain resolution penalty.

    Quantifies the trade the pink beam makes (elastic-strain resolution for flux):
    at ID03 bandwidths this is ~3-4x. Returned as a dimensionless factor for a
    single reflection geometry. The default divergence / NA reproduce the ID03
    box-beam numbers of arXiv:2510.26665 (mono ``q_par`` 0.00288 -> pink 0.01026,
    ~3.6x); the ratio is geometry-dependent through the angular terms.
    """
    kw = dict(two_theta_deg=two_theta_deg, divergence=divergence, na=na)
    pink = float(axial_reciprocal_width(1.0, eps=eps_pink, **kw))
    mono = float(axial_reciprocal_width(1.0, eps=eps_mono, **kw))
    return pink / mono
