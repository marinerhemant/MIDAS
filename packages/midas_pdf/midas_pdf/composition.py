"""Sample composition → Faber-Ziman form-factor averages ⟨f²⟩(Q), ⟨f⟩²(Q).

This is the one piece that did not exist elsewhere in MIDAS. The existing
``midas_integrate_v2.pdf.normalize_to_S`` is *monoatomic* (divides by a single
⟨f²⟩). A polyatomic total-scattering sample needs the Faber-Ziman form, which
requires the number-fraction-weighted averages

    ⟨f⟩(Q)  = Σ_i c_i f_i(Q)
    ⟨f²⟩(Q) = Σ_i c_i f_i(Q)²

built from the per-element X-ray form factors f_i(Q). Those form factors come
straight from ``midas_hkls`` (Cromer-Mann IT92), so they are differentiable in
Q; here we additionally allow the mole fractions ``c_i`` to be a torch tensor,
making the averages differentiable in composition as well.

Form factors are tabulated with argument ``s² = (sinθ/λ)² = (Q/4π)²`` [Å⁻²].
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from midas_hkls import form_factor_batch

__all__ = ["Composition"]

_FOUR_PI = 4.0 * float(np.pi)


def _q_to_s2(q: torch.Tensor) -> torch.Tensor:
    """Q (Å⁻¹) → s² = (Q/4π)² (Å⁻²), the Cromer-Mann argument."""
    return (q / _FOUR_PI) ** 2


_ELEMENT_Z = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    "K":19, "Ca":20, "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26,
    "Co":27, "Ni":28, "Cu":29, "Zn":30, "Ga":31, "Ge":32, "As":33, "Se":34,
    "Br":35, "Kr":36, "Rb":37, "Sr":38, "Y":39, "Zr":40, "Nb":41, "Mo":42,
    "Tc":43, "Ru":44, "Rh":45, "Pd":46, "Ag":47, "Cd":48, "In":49, "Sn":50,
    "Sb":51, "Te":52, "I":53, "Xe":54, "Cs":55, "Ba":56, "La":57, "Ce":58,
    "Pr":59, "Nd":60, "Pm":61, "Sm":62, "Eu":63, "Gd":64, "Tb":65, "Dy":66,
    "Ho":67, "Er":68, "Tm":69, "Yb":70, "Lu":71, "Hf":72, "Ta":73, "W":74,
    "Re":75, "Os":76, "Ir":77, "Pt":78, "Au":79, "Hg":80, "Tl":81, "Pb":82,
    "Bi":83, "Po":84, "At":85, "Rn":86, "Fr":87, "Ra":88, "Ac":89, "Th":90,
    "Pa":91, "U":92,
}


def _anomalous_corrections(
    elements: Sequence[str], wavelength_A: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cromer-Liberman f'(E) and f''(E) for the given elements.

    Uses ``midas_hkls`` (MIDAS-native Cromer-Liberman table, already a
    midas-pdf dependency) — no runtime xraylib. Returns two 1-D arrays of
    shape (Nel,). f'' is returned as its absolute value (sign convention
    differs across tabulations; only |f''| enters the coherent |f|² used
    for S(Q)).
    """
    from midas_hkls.anomalous import anomalous_correction
    # accept ionic species by stripping the charge suffix ("Ni2+", "O2-")
    base = [
        el.rstrip("+-0123456789") if any(c in el for c in "+-") else el
        for el in elements
    ]
    for el in base:
        if el not in _ELEMENT_Z:
            raise ValueError(f"unknown element {el!r} for anomalous correction")
    fp, fpp = anomalous_correction(base, float(wavelength_A))
    f_prime  = np.asarray(fp, dtype=np.float64)
    f_dprime = np.abs(np.asarray(fpp, dtype=np.float64))
    return f_prime, f_dprime


class Composition:
    """A sample composition as number (mole) fractions of elements.

    Parameters
    ----------
    fractions :
        Mapping ``{element_symbol: number_fraction}``. Need not be normalised;
        stored fractions are renormalised to sum to 1.
    number_density :
        Optional atomic number density ρ₀ (atoms · Å⁻³). Only needed for the
        low-r slope check ``G(r→0) = -4π ρ₀ r`` / absolute normalisation; the
        S(Q) and G(r) computation itself does not require it.
    """

    def __init__(
        self,
        fractions: Mapping[str, float],
        *,
        number_density: Optional[float] = None,
    ):
        if not fractions:
            raise ValueError("composition must be non-empty")
        total = float(sum(fractions.values()))
        if total <= 0:
            raise ValueError("composition fractions must sum to > 0")
        self.elements: list[str] = list(fractions.keys())
        self._fractions = np.asarray(
            [float(v) / total for v in fractions.values()], dtype=np.float64
        )
        self.number_density = (
            None if number_density is None else float(number_density)
        )

    # -------------------------------------------------------------- accessors

    @property
    def fractions(self) -> np.ndarray:
        """Normalised number fractions, aligned with ``self.elements``."""
        return self._fractions.copy()

    def as_dict(self) -> dict[str, float]:
        return {e: float(c) for e, c in zip(self.elements, self._fractions)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        body = ", ".join(f"{e}:{c:.4g}" for e, c in self.as_dict().items())
        return f"Composition({body})"

    # ----------------------------------------------------------------- physics

    def _form_factor_batch_with_ions(
        self,
        s2: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Return atomic form factors f_i(Q) for each element in the composition,
        using ionic Cromer-Mann coefficients where available and falling back
        to neutral-atom form factors (via ``midas_hkls.form_factor_batch``)
        otherwise.

        Returns a tensor of shape ``(Nq, Nel)``.
        """
        from .ionic_form_factors import (
            is_ionic_species, ION_COEFFICIENTS, ionic_form_factor,
        )
        # First get the neutral-atom form factors for all elements as fallback
        # (midas_hkls silently maps "Ni2+" → "Ni" so this is safe).
        f_neutral = form_factor_batch(s2, self.elements)     # (Nq, Nel)
        # Now override columns where the species is a *registered* ion.
        f_out = f_neutral.clone()
        for i, el in enumerate(self.elements):
            if is_ionic_species(el) and el in ION_COEFFICIENTS:
                f_out[:, i] = ionic_form_factor(q, el)
        return f_out


    def _fraction_tensor(
        self,
        q: torch.Tensor,
        fractions: Optional[torch.Tensor | Sequence[float]],
    ) -> torch.Tensor:
        if fractions is None:
            c = torch.as_tensor(self._fractions, dtype=q.dtype, device=q.device)
        else:
            c = torch.as_tensor(fractions, dtype=q.dtype, device=q.device)
            if c.shape != (len(self.elements),):
                raise ValueError(
                    f"fractions must have shape ({len(self.elements)},), "
                    f"got {tuple(c.shape)}"
                )
        return c

    def form_factor_averages(
        self,
        q: torch.Tensor | np.ndarray,
        *,
        fractions: Optional[torch.Tensor | Sequence[float]] = None,
        wavelength_A: Optional[float] = None,
        anomalous: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(⟨f⟩(Q), ⟨f²⟩(Q))`` as torch tensors, same shape as ``q``.

        Differentiable in ``q`` and, if ``fractions`` is a tensor with
        ``requires_grad``, in composition. Pass ``fractions`` to override the
        stored fractions (e.g. for refinement) — it must be length
        ``len(self.elements)`` and need not be normalised here (caller owns
        the constraint).

        With ``anomalous=True`` and ``wavelength_A`` provided, the
        Cromer-Liberman anomalous dispersion corrections f'(E), f''(E)
        (xraylib.Fi / Fii) are added to the atomic form factor:

            f_i(Q, E) = f0_i(Q) + f'_i(E) + i f''_i(E)

        and the averages use ``|f_i|²`` for ``⟨f²⟩`` and the complex-
        magnitude average for ``⟨f⟩``. For elements far from any absorption
        edge at the incident energy this is a fraction-of-a-percent
        correction; for elements near an edge (or heavy elements at any
        energy) it can reach several percent.
        """
        q_t = torch.as_tensor(q, dtype=torch.float64)
        s2 = _q_to_s2(q_t)                                  # (Nq,)
        f0 = self._form_factor_batch_with_ions(s2, q_t)     # (Nq, Nel)
        c = self._fraction_tensor(q_t, fractions)           # (Nel,)

        if anomalous and wavelength_A is not None:
            f_prime, f_dprime = _anomalous_corrections(
                self.elements, float(wavelength_A))
            f_prime  = torch.as_tensor(f_prime,  dtype=q_t.dtype, device=q_t.device)
            f_dprime = torch.as_tensor(f_dprime, dtype=q_t.dtype, device=q_t.device)
            # Complex f_i(Q) = (f0 + f') + i f''; broadcast over Q
            f_real = f0 + f_prime                             # (Nq, Nel)
            f_imag = f_dprime.expand_as(f_real)               # (Nq, Nel)
            # |f_i|² for ⟨f²⟩
            f_abs2 = f_real ** 2 + f_imag ** 2
            f2_avg = (f_abs2 * c).sum(dim=-1)                 # ⟨|f|²⟩
            # ⟨f⟩² is well-defined only if we keep the complex average;
            # for the Faber-Ziman Laue term what matters is |⟨f⟩|² so we
            # sum real and imag components separately and take magnitude.
            f_avg_real = (f_real * c).sum(dim=-1)
            f_avg_imag = (f_imag * c).sum(dim=-1)
            f_avg = torch.sqrt(f_avg_real ** 2 + f_avg_imag ** 2)
            return f_avg, f2_avg

        # Standard (no anomalous)
        f_avg = (f0 * c).sum(dim=-1)                         # ⟨f⟩
        f2_avg = (f0 * f0 * c).sum(dim=-1)                   # ⟨f²⟩
        return f_avg, f2_avg

    def laue(
        self,
        q: torch.Tensor | np.ndarray,
        *,
        fractions: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> torch.Tensor:
        """Laue monotonic / self-scattering term ``⟨f²⟩ - ⟨f⟩²``.

        This is the polyatomic correction that vanishes for a monoatomic
        sample (⟨f²⟩ = ⟨f⟩²), recovering the monoatomic normalisation in
        ``midas_integrate_v2.pdf.normalize_to_S``.
        """
        f_avg, f2_avg = self.form_factor_averages(q, fractions=fractions)
        return f2_avg - f_avg * f_avg

    def compton(
        self,
        q: torch.Tensor | np.ndarray,
        *,
        wavelength_A: float,
        fractions: Optional[torch.Tensor | Sequence[float]] = None,
        breit_dirac: bool = True,
        k: int = 2,
        method: str = "hubbell",
    ) -> torch.Tensor:
        """Composition-weighted Compton (incoherent) intensity per atom.

        Default ``method='hubbell'`` uses the tabulated Hubbell incoherent
        scattering functions (:func:`midas_pdf.compton.incoherent_scattering`)
        with the Breit-Dirac recoil factor — the same data class GudrunX/PDFgetX
        use. ``method='it94'`` falls back to the coarse analytic form in
        ``midas_integrate_v2`` (kept for comparison). Differentiable in Q,
        wavelength, and (Hubbell path) composition.
        """
        q_t = torch.as_tensor(q, dtype=torch.float64)
        if fractions is None:
            fractions = torch.as_tensor(self._fractions, dtype=torch.float64)
        if method == "hubbell":
            from .compton import incoherent_scattering

            return incoherent_scattering(
                q_t, self.elements, wavelength_A=wavelength_A,
                fractions=fractions, breit_dirac=breit_dirac, k=k,
            )
        if method == "it94":
            from midas_integrate_v2.corrections.compton import ComptonSubtraction

            return ComptonSubtraction(self.as_dict(), wavelength_A=wavelength_A)(q_t)
        raise ValueError(f"unknown compton method {method!r}; use 'hubbell' or 'it94'")
