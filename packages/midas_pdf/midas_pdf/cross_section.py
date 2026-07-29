"""Differential scattering cross-section per atom — the engine for first-principles
multiple scattering (Tier 3).

Multiple-scattering estimators (analytic slab/cylinder integrals *or* Monte
Carlo) all need one thing: the probability that a photon scatters from one
direction into another, i.e. the per-atom differential cross-section
``dσ/dΩ(ψ)``. In Thomson units (r_e^2 = 1) and azimuthally averaged it is

    dσ/dΩ(Q) = P(ψ) · [ <f(Q)^2> · S(Q) + <S_inc(Q)> ],
    cos ψ = 1 - 2 (Q λ / 4π)^2,

combining the coherent part (form-factor average times the structure factor;
S(Q)=1 in the independent-atom / Laue approximation, which is the usual choice
for the *smooth* MS background) and the Hubbell incoherent part already in
:mod:`midas_pdf.compton`. ``P(ψ)`` is the polarization factor.

Everything is a torch op and differentiable in Q, wavelength, and composition —
which is the whole point: it lets the eventual MS correction be differentiable,
the capability no existing total-scattering code provides. This module is the
shared foundation; the MS integrators build on it (see dev/PLAN.md, Tier 3).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .composition import Composition
from .compton import incoherent_scattering
from .corrections import cos_scattering_angle

__all__ = [
    "polarization_factor",
    "differential_cross_section",
    "total_cross_section",
]

_FOUR_PI = 4.0 * float(np.pi)


def polarization_factor(
    q: torch.Tensor | np.ndarray,
    *,
    wavelength_A: float,
    polarization_fraction: float = 0.0,
    plane_cos2: float = 0.5,
) -> torch.Tensor:
    """Azimuthally-averaged polarization factor P(ψ) as a function of Q.

    ``polarization_fraction = 0`` is unpolarized, ``P = (1 + cos²ψ)/2``.
    For a synchrotron, the horizontally-polarized beam gives an azimuth-
    dependent factor; ``plane_cos2`` is the azimuth average of ``cos²`` of the
    in-plane angle (1/2 after a full azimuthal integration), and
    ``polarization_fraction`` in [0, 1] interpolates toward fully polarized.
    """
    cos_psi = cos_scattering_angle(q, wavelength_A)
    cos2 = cos_psi * cos_psi
    unpol = 0.5 * (1.0 + cos2)
    # polarized correction (azimuth-averaged): blend by the polarization fraction
    pol = (1.0 - polarization_fraction) * unpol + polarization_fraction * (
        plane_cos2 + (1.0 - plane_cos2) * cos2
    )
    return pol


def differential_cross_section(
    q: torch.Tensor | np.ndarray,
    composition: Composition,
    *,
    wavelength_A: float,
    structure_factor: Optional[torch.Tensor | np.ndarray] = None,
    include_incoherent: bool = True,
    polarization_fraction: float = 0.0,
    fractions: Optional[torch.Tensor | Sequence[float]] = None,
) -> torch.Tensor:
    """Per-atom differential cross-section ``dσ/dΩ(Q)`` in Thomson units.

    ``structure_factor`` S(Q) modulates the coherent part; ``None`` uses the
    independent-atom approximation S(Q)=1 (the usual choice for the smooth MS
    background). ``include_incoherent`` adds the Hubbell Compton term.
    Differentiable in Q / wavelength / composition.
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    f_avg, f2_avg = composition.form_factor_averages(q_t, fractions=fractions)
    if structure_factor is None:
        S = torch.ones_like(q_t)
    else:
        S = torch.as_tensor(structure_factor, dtype=torch.float64)
    coherent = f2_avg * S
    if include_incoherent:
        inc = incoherent_scattering(
            q_t, composition.elements, wavelength_A=wavelength_A,
            fractions=(fractions if fractions is not None
                       else torch.as_tensor(composition.fractions, dtype=torch.float64)),
            breit_dirac=True,
        )
    else:
        inc = torch.zeros_like(q_t)
    P = polarization_factor(q_t, wavelength_A=wavelength_A,
                            polarization_fraction=polarization_fraction)
    return P * (coherent + inc)


def total_cross_section(
    composition: Composition,
    *,
    wavelength_A: float,
    n_theta: int = 512,
    include_incoherent: bool = True,
    polarization_fraction: float = 0.0,
) -> torch.Tensor:
    """Total scattering cross-section per atom, ``σ = ∫ dσ/dΩ dΩ`` (Thomson units).

    Integrated over the full sphere by Gauss-uniform quadrature in
    ``cos ψ ∈ [-1, 1]`` (azimuthally symmetric): ``σ = 2π ∫ (dσ/dΩ) d(cos ψ)``.
    Needed to convert the differential cross-section into scattering
    probabilities for the MS estimators. Differentiable.
    """
    lam = float(wavelength_A)
    # scattering angle psi in [0, psi_max]; cos psi spans [cos psi_max, 1].
    # Q from psi: Q = (4π/λ) sin(ψ/2).
    psi = torch.linspace(0.0, float(np.pi), n_theta, dtype=torch.float64)
    Q = (_FOUR_PI / lam) * torch.sin(0.5 * psi)
    dsig = differential_cross_section(
        Q, composition, wavelength_A=lam, include_incoherent=include_incoherent,
        polarization_fraction=polarization_fraction,
    )
    # σ = ∫ dσ/dΩ sin ψ dψ dφ = 2π ∫_0^π dσ/dΩ sin ψ dψ  (trapezoid in ψ)
    integrand = dsig * torch.sin(psi)
    return 2.0 * torch.pi * torch.trapz(integrand, psi)
