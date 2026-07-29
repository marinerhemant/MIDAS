"""Multiple-scattering handling.

Multiple scattering (MS) — a photon scattering more than once in the sample
before reaching the detector — adds a smooth, slowly-Q-varying background that
carries no sharp structural signal. Unlike the other corrections in this
package it has **no closed form**: the double-scattering intensity is an integral
over the sample volume and all intermediate directions, it depends on the sample
*shape*, and it is self-referential (the scattering probability depends on the
cross-section being measured). A first-principles MS module (Monte-Carlo, or an
analytic slab/cylinder double-scattering integral) is planned — see dev/PLAN.md.

**Tier 1 (this module): the lumped smooth-background treatment.** In the
high-energy, thin-sample regime typical of total-scattering beamlines, MS is
small *and* smooth, so it has no sharp Q-structure to alias into G(r). We model
it — together with the other smooth contaminants (fluorescence baseline, air /
empty-cell residual, Compton-model error) — as a single refinable smooth
background ``b(Q)`` subtracted from the measured intensity before normalization:

    I_coh,single(Q) ≈ scale · [ I_meas(Q) − b(Q) ] − I_compton(Q).

``b(Q)`` is a low-order polynomial in ``Q/Q_max``; its coefficients can be
supplied directly or fit by :func:`midas_pdf.refine.refine_normalization`
(``bg_order=...``). This is honest about what it is — a *lumped* correction, not
a first-principles MS calculation — and is adequate when MS is a small, smooth
fraction of the signal.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

__all__ = ["lumped_background", "polynomial_basis"]


def polynomial_basis(
    q: torch.Tensor | np.ndarray,
    order: int,
    *,
    q_max: Optional[float] = None,
) -> torch.Tensor:
    """Design matrix of powers ``(Q/Q_max)^j`` for ``j = 0..order``.

    Returns shape ``(len(q), order+1)``. ``Q/Q_max`` keeps the basis
    well-conditioned (entries in [0, 1]); ``q_max`` defaults to ``max(q)``.
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    if order < 0:
        raise ValueError("order must be >= 0")
    scale = float(q_max) if q_max is not None else float(q_t.max())
    x = q_t / scale
    return torch.stack([x ** j for j in range(order + 1)], dim=1)


def lumped_background(
    q: torch.Tensor | np.ndarray,
    coef: Sequence[float] | torch.Tensor,
    *,
    q_max: Optional[float] = None,
) -> torch.Tensor:
    """Evaluate the lumped smooth background ``b(Q) = Σ_j coef_j (Q/Q_max)^j``.

    ``coef`` may be a tensor with ``requires_grad`` (differentiable, refinable).
    The degree is ``len(coef) - 1``. This is the Tier-1 multiple-scattering /
    smooth-contaminant term subtracted from the measured intensity prior to
    Faber-Ziman normalization.
    """
    c = torch.as_tensor(coef, dtype=torch.float64)
    if c.ndim != 1:
        raise ValueError("coef must be 1-D")
    basis = polynomial_basis(q, c.shape[0] - 1, q_max=q_max)
    return basis @ c
