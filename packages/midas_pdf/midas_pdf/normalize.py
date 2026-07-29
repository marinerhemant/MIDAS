"""Polyatomic Faber-Ziman normalization: I(Q) → S(Q), with σ propagation.

Faber-Ziman total structure factor (X-ray, neutral-atom form factors)::

    S(Q) = [ I_coh(Q) - (⟨f²⟩ - ⟨f⟩²) ] / ⟨f⟩²
    I_coh(Q) = scale · I_meas(Q) - I_compton(Q)

``S(Q) → 1`` as ``Q → ∞``. For a *monoatomic* sample ⟨f²⟩ = ⟨f⟩² = f², the
Laue term vanishes and this reduces exactly to the existing monoatomic
``midas_integrate_v2.pdf.normalize_to_S`` (S = I_coh / f²). That equivalence is
pinned by a regression test.

σ propagation: form factors and the Compton model are treated as noiseless, so

    σ_S(Q) = scale · σ_I(Q) / ⟨f⟩²

This is the only normalization step that ``midas-pdf`` adds; the downstream
sine FT (S(Q) → G(r)) and its variance propagation are reused verbatim from
``midas_integrate_v2.pdf.fourier_sine_transform``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .composition import Composition

__all__ = ["faber_ziman_S"]


def faber_ziman_S(
    intensity: torch.Tensor | np.ndarray,
    q: torch.Tensor | np.ndarray,
    composition: Composition,
    *,
    wavelength_A: float,
    scale: Union[float, torch.Tensor] = 1.0,
    compton: Union[bool, torch.Tensor, np.ndarray, None] = True,
    background: Optional[torch.Tensor | np.ndarray] = None,
    sigma_intensity: Optional[torch.Tensor | np.ndarray] = None,
    fractions: Optional[torch.Tensor | Sequence[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Faber-Ziman S(Q) from a measured 1-D intensity, with σ propagation.

    The coherent single-scattering intensity is recovered as

        I_coh(Q) = scale · [ I_meas(Q) − background(Q) ] − I_compton(Q),

    so a lumped smooth ``background`` (Tier-1 multiple scattering + fluorescence
    + air/empty residual; see :mod:`midas_pdf.multiple_scattering`) is removed on
    the measured scale before normalization, while the modelled Compton term is
    in per-atom units.

    Parameters
    ----------
    intensity, q :
        1-D measured intensity I(Q) and its Q grid (Å⁻¹), same shape.
    composition :
        :class:`~midas_pdf.composition.Composition` for the sample.
    wavelength_A :
        Source wavelength (Å); sets the Compton angular factor.
    scale :
        Multiplicative normalization of the measured intensity onto the
        per-atom electron-unit scale. May be a tensor with ``requires_grad``
        for differentiable refinement.
    compton :
        ``True`` (default) to compute and subtract the composition-weighted
        Compton intensity; ``None``/``False`` to skip it; or a precomputed
        tensor/array shaped like ``q`` to subtract directly.
    background :
        Optional lumped smooth background ``b(Q)`` (same shape as ``q``)
        subtracted from the measured intensity before scaling. ``None`` = 0.
        May be a tensor with ``requires_grad`` (refinable).
    sigma_intensity :
        Optional 1σ on ``intensity``; propagated to ``σ_S``.
    fractions :
        Optional override of composition number fractions (length =
        ``len(composition.elements)``), e.g. for joint refinement; if a tensor
        with ``requires_grad`` the result is differentiable in composition.

    Returns
    -------
    (S, sigma_S) : tuple of torch.Tensor
        ``sigma_S`` is all-zeros if ``sigma_intensity`` is None.
    """
    I = torch.as_tensor(intensity, dtype=torch.float64)
    q_t = torch.as_tensor(q, dtype=torch.float64)
    if I.shape != q_t.shape:
        raise ValueError(
            f"intensity shape {tuple(I.shape)} != q shape {tuple(q_t.shape)}"
        )

    f_avg, f2_avg = composition.form_factor_averages(q_t, fractions=fractions)
    laue = f2_avg - f_avg * f_avg                       # ⟨f²⟩ - ⟨f⟩²
    f_avg2_safe = (f_avg * f_avg).clamp(min=1e-30)      # ⟨f⟩², division-safe

    scale_t = torch.as_tensor(scale, dtype=torch.float64)

    if compton is None or compton is False:
        cmp = torch.zeros_like(q_t)
    elif compton is True:
        cmp = composition.compton(q_t, wavelength_A=wavelength_A)
    else:
        cmp = torch.as_tensor(compton, dtype=torch.float64)
        if cmp.shape != q_t.shape:
            raise ValueError("precomputed compton must match q shape")

    if background is None:
        bg = torch.zeros_like(I)
    else:
        bg = torch.as_tensor(background, dtype=torch.float64)
        if bg.shape != q_t.shape:
            raise ValueError("background must match q shape")

    I_coh = scale_t * (I - bg) - cmp
    S = (I_coh - laue) / f_avg2_safe

    if sigma_intensity is None:
        sigma_S = torch.zeros_like(S)
    else:
        sig = torch.as_tensor(sigma_intensity, dtype=torch.float64)
        if sig.shape != q_t.shape:
            raise ValueError("sigma_intensity must match q shape")
        sigma_S = (scale_t.abs() * sig) / f_avg2_safe
    return S, sigma_S
