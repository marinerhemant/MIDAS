"""Core-shell and multi-shell SAXS form factors.

The single-density sphere / ellipsoid / cylinder in
:mod:`midas_pdf.saxs.form_factors` model a uniform-density particle.
Many real nanoparticles are structured:

  * Coated nanoparticles: metallic core + oxide / ligand shell.
  * Reverse micelles: solvent core + amphiphile shell.
  * Multi-shell nanocrystals: layered lattice + interface + surface region.

For these, the SAXS form factor is a difference of nested-sphere
amplitudes weighted by the contrast (electron-density difference)
in each layer.

For a two-shell (core + shell) spherical particle:

    F(Q) = 4π (ρ_core − ρ_shell) V_core · j₁(Q R_core) / (Q R_core)
         + 4π (ρ_shell − ρ_solvent) V_total · j₁(Q R_total) / (Q R_total)

where ``j₁`` is the spherical Bessel of order 1 and V is the volume of
each region. This generalises to N shells recursively.

All routines here return **|F(Q)|²** and are torch-differentiable in the
shell radii and contrast ratios.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

_FOUR_PI = 4.0 * float(np.pi)


def _sphere_amplitude(q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Complex spherical-shell amplitude at Q = 0 is V (the volume).

    Returns ``F(Q, R) = V · 3 [sin(QR) − QR cos(QR)] / (QR)³``
    with a stable Q → 0 limit.
    """
    V = _FOUR_PI * R ** 3 / 3.0
    x = q * R
    small = x.abs() < 1e-3
    # small-x: 1 − x²/10 + x⁴/280
    shape_small = 1.0 - x ** 2 / 10.0 + x ** 4 / 280.0
    x_safe = x.clamp(min=1e-9)
    shape_general = 3.0 * (torch.sin(x_safe) - x_safe * torch.cos(x_safe)) / x_safe ** 3
    return V * torch.where(small, shape_small, shape_general)


def core_shell_sphere_form_factor_squared(
    q: torch.Tensor,
    R_core_A: float | torch.Tensor,
    R_total_A: float | torch.Tensor,
    contrast_core: float | torch.Tensor = 1.0,
    contrast_shell: float | torch.Tensor = 0.5,
) -> torch.Tensor:
    """|F(Q)|² for a core-shell spherical particle.

    Parameters
    ----------
    R_core_A : radius of the core (Å).
    R_total_A : outer radius (core + shell, Å); shell thickness =
        R_total − R_core.
    contrast_core : electron-density difference (core − solvent), arbitrary
        units. Only the ratio (core − shell)/(shell − solvent) matters
        for the shape of I(Q); the absolute scale drops into the caller's
        overall scale factor.
    contrast_shell : electron-density difference (shell − solvent).

    Returns
    -------
    |F(Q)|² of the same units as (contrast · Å³)².
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    R_c = torch.as_tensor(R_core_A, dtype=torch.float64)
    R_t = torch.as_tensor(R_total_A, dtype=torch.float64)
    delta_core  = torch.as_tensor(contrast_core,  dtype=torch.float64) \
                  - torch.as_tensor(contrast_shell, dtype=torch.float64)
    delta_shell = torch.as_tensor(contrast_shell, dtype=torch.float64)
    F_core  = _sphere_amplitude(q_t, R_c)
    F_total = _sphere_amplitude(q_t, R_t)
    F = delta_core * F_core + delta_shell * F_total
    return F ** 2


def multi_shell_sphere_form_factor_squared(
    q: torch.Tensor,
    radii_A: Sequence[float | torch.Tensor],
    contrasts: Sequence[float | torch.Tensor],
) -> torch.Tensor:
    """|F(Q)|² for a multi-shell spherical particle.

    ``radii_A``: list of *outer* radii of each shell (monotonically
    increasing).  E.g. for a 3-shell particle, ``[R_1, R_2, R_3]``.

    ``contrasts``: contrast of *each* shell against the solvent (same
    length as ``radii_A``).  The interfacial contrast between successive
    shells drives the SAXS signal.
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    if len(radii_A) != len(contrasts):
        raise ValueError("len(radii_A) must equal len(contrasts)")
    if len(radii_A) < 1:
        raise ValueError("need at least one shell")
    # F = sum_i (ρ_i − ρ_{i+1}) F_i, with ρ_{N+1} = solvent ≡ 0
    F = torch.zeros_like(q_t)
    for i in range(len(radii_A)):
        rho_here = torch.as_tensor(contrasts[i], dtype=torch.float64)
        rho_next = (torch.as_tensor(contrasts[i + 1], dtype=torch.float64)
                    if i + 1 < len(contrasts) else torch.tensor(0.0, dtype=torch.float64))
        delta = rho_here - rho_next
        R = torch.as_tensor(radii_A[i], dtype=torch.float64)
        F = F + delta * _sphere_amplitude(q_t, R)
    return F ** 2


__all__ = [
    "core_shell_sphere_form_factor_squared",
    "multi_shell_sphere_form_factor_squared",
]
