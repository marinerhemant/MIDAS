"""End-to-end orchestration: I(Q) → S(Q) → G(r), fully error-propagating.

This composes the one new step (polyatomic Faber-Ziman normalization,
:func:`midas_pdf.normalize.faber_ziman_S`) with the reused sine FT
(:func:`midas_integrate_v2.pdf.fourier_sine_transform`).

The pixels → I(Q) step is intentionally *not* duplicated here: that is
``midas_integrate_v2``'s job (``integrate_to_Gr_with_variance`` /
``integrate_*_with_variance``), and feeding its (I, σ_I) output into
:func:`i_of_q_to_Gr` gives the complete pixels → G(r) chain with σ carried
end to end.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .composition import Composition
from .gr import fourier_sine_transform
from .normalize import faber_ziman_S

__all__ = ["i_of_q_to_Gr"]


def i_of_q_to_Gr(
    q: torch.Tensor | np.ndarray,
    intensity: torch.Tensor | np.ndarray,
    composition: Composition,
    r_grid: torch.Tensor | np.ndarray,
    *,
    wavelength_A: float,
    scale: Union[float, torch.Tensor] = 1.0,
    compton: Union[bool, torch.Tensor, np.ndarray, None] = True,
    background: Optional[torch.Tensor | np.ndarray] = None,
    sigma_intensity: Optional[torch.Tensor | np.ndarray] = None,
    q_max: Optional[float] = None,
    window: str = "lorch",
    fractions: Optional[torch.Tensor | Sequence[float]] = None,
    return_S: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Measured I(Q) → G(r) with a propagated 1σ band.

    Returns ``(G, sigma_G, S)`` (``S`` is ``None`` if ``return_S=False``).
    All inputs may carry ``requires_grad``; gradients flow to ``scale``,
    ``fractions``, ``background``, and (through ``q``) the geometry/wavelength
    upstream. ``background`` is the lumped smooth term (Tier-1 multiple
    scattering + fluorescence + air); see :mod:`midas_pdf.multiple_scattering`.
    """
    S, sigma_S = faber_ziman_S(
        intensity, q, composition,
        wavelength_A=wavelength_A, scale=scale, compton=compton,
        background=background, sigma_intensity=sigma_intensity, fractions=fractions,
    )
    r_t = torch.as_tensor(r_grid, dtype=torch.float64)
    G, sigma_G = fourier_sine_transform(
        q, S, r_t, Q_max=q_max, window=window, sigma_S=sigma_S,
    )
    if sigma_G is None:
        sigma_G = torch.zeros_like(G)
    return G, sigma_G, (S if return_S else None)
