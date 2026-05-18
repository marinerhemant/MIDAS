"""Compton (incoherent) scattering subtraction.

For a multi-element sample the Compton intensity per atom-mole is

    I_compton(Q) = Σ_z f_z S_inc(Q, Z) · (KN factor at θ)

where ``S_inc(Q, Z)`` is the per-element incoherent scattering function
(Hubbell 1975 tables, IT94 Vol C). For neutral atoms with high-Z this
is dominant at high Q and must be subtracted to recover the coherent
part of S(Q) for PDF analysis.

This stub uses the Klein-Nishina differential cross-section as the
angular factor and a simple analytic ``S_inc`` parameterisation
(``S_inc(Q, Z) = Z (1 - exp(-α Q))``, with ``α`` element-specific).
The exact tables are large; we ship the IT94 fitted-form for the
elements most commonly encountered in PDF analysis (H, C, N, O, Si, Ti,
Fe, Cu, Ce, La, B; expandable). Differentiable in a refinable scale
factor.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn


# IT94 / Hubbell-style fitted parameters: S_inc(Q, Z) ≈ Z (1 - exp(-α Q))
# Values are coarse approximations; suitable for first-pass subtraction
# only. Full PDFgetX3 / PDFgui usage uses tabulated S_inc(Q, Z).
_ALPHA_TABLE: Dict[str, float] = {
    "H": 5.0, "He": 4.5,
    "Li": 4.0, "Be": 3.8, "B": 3.5, "C": 3.2, "N": 3.0, "O": 2.9,
    "F": 2.8, "Ne": 2.7,
    "Na": 2.6, "Mg": 2.5, "Al": 2.4, "Si": 2.3, "P": 2.2, "S": 2.1,
    "Cl": 2.0, "Ar": 1.9,
    "K": 1.85, "Ca": 1.8, "Ti": 1.7, "V": 1.65, "Cr": 1.6, "Mn": 1.55,
    "Fe": 1.5, "Co": 1.45, "Ni": 1.4, "Cu": 1.35, "Zn": 1.3,
    "Y": 1.0, "Zr": 0.95,
    "La": 0.7, "Ce": 0.68, "Pr": 0.66, "Nd": 0.64, "Sm": 0.6,
    "Au": 0.45, "Pb": 0.4, "U": 0.35,
}

# Atomic numbers for the elements above
_Z_TABLE: Dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Ti": 22, "V": 23,
    "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Y": 39, "Zr": 40, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62,
    "Au": 79, "Pb": 82, "U": 92,
}


def _S_inc_per_element(Q_invA: torch.Tensor, element: str) -> torch.Tensor:
    """Incoherent scattering S_inc(Q) for a single element, simple param."""
    if element not in _ALPHA_TABLE:
        raise KeyError(
            f"element {element!r} not in Compton table; "
            f"add to _ALPHA_TABLE / _Z_TABLE"
        )
    alpha = _ALPHA_TABLE[element]
    Z = _Z_TABLE[element]
    return Z * (1.0 - torch.exp(-alpha * Q_invA))


def _klein_nishina(Q_invA: torch.Tensor, wavelength_A: float) -> torch.Tensor:
    """Klein-Nishina angular factor at the 2θ corresponding to Q.

        2θ = 2 arcsin(λ Q / 4π)

    Returns the differential cross-section ratio (dσ_KN/dΩ) /
    (dσ_Th/dΩ), useful for the scaling between the incoherent and
    elastic contributions.
    """
    arg = (wavelength_A * Q_invA / (4.0 * torch.pi)).clamp(max=0.999)
    two_theta = 2.0 * torch.arcsin(arg)
    cos2t = torch.cos(two_theta)
    return 0.5 * (1.0 + cos2t * cos2t)


class ComptonSubtraction(nn.Module):
    """Compton scattering for a composition-weighted sample.

    Parameters
    ----------
    composition :
        Mapping ``{element_symbol: mole_fraction}``. Mole fractions need
        not be normalised; the module renormalises to sum 1.
    wavelength_A :
        Source wavelength in Å (sets the angular factor scale).
    refinable_scale :
        If True, an overall multiplicative scale becomes refinable.
    """

    def __init__(
        self,
        composition: Mapping[str, float],
        *,
        wavelength_A: float,
        refinable_scale: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if not composition:
            raise ValueError("composition must be non-empty")
        total = float(sum(composition.values()))
        if total <= 0:
            raise ValueError("composition mole fractions must sum > 0")
        self._composition = {k: float(v) / total for k, v in composition.items()}
        self.wavelength_A = float(wavelength_A)
        scale_t = torch.tensor(1.0, dtype=dtype)
        if refinable_scale:
            self.scale = nn.Parameter(scale_t)
        else:
            self.register_buffer("scale", scale_t)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        q_t = torch.as_tensor(q, dtype=torch.float64)
        I_total = torch.zeros_like(q_t)
        for elem, frac in self._composition.items():
            I_total = I_total + frac * _S_inc_per_element(q_t, elem)
        kn = _klein_nishina(q_t, self.wavelength_A)
        return self.scale * I_total * kn


__all__ = ["ComptonSubtraction"]
