"""Differentiable refinement of the PDF normalization.

Total-scattering normalization has two parameters that are traditionally tuned
*by hand* (PDFgetX3 / Gudrun): the overall intensity ``scale`` onto the per-atom
electron scale, and a flat additive ``offset`` absorbing residual background.
Because the whole ``I(Q) → S(Q) → G(r)`` chain here is differentiable, those
parameters can instead be **fit by gradient descent** against two model-free
physical constraints:

  1. high-Q asymptote:  ⟨S(Q)⟩ → 1  over the top of the Q range;
  2. low-r behaviour:   G(r) = −4π ρ₀ r  below the closest interatomic distance
     (g(r)=0 there, so the reduced PDF is the exact straight line −4π ρ₀ r).

Both are differentiable in (scale, offset[, number_density]), so a few L-BFGS
steps replace the manual twiddling — and, because σ is propagated, the fit can
be reported with its parameter uncertainty downstream.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from .composition import Composition
from .gr import fourier_sine_transform
from .normalize import faber_ziman_S

__all__ = ["refine_normalization", "RefineResult"]

_FOUR_PI = 4.0 * float(np.pi)


class RefineResult(dict):
    """Plain dict of refinement outputs, attribute-accessible for convenience."""

    def __getattr__(self, name):
        try: return self[name]
        except KeyError: raise AttributeError(name) from None


def refine_normalization(
    q: torch.Tensor | np.ndarray,
    intensity: torch.Tensor | np.ndarray,
    composition: Composition,
    r_grid: torch.Tensor | np.ndarray,
    *,
    wavelength_A: float,
    number_density: float,
    sigma_intensity: Optional[torch.Tensor | np.ndarray] = None,
    compton: bool = True,
    q_max: Optional[float] = None,
    window: str = "lorch",
    fractions: Optional[torch.Tensor | Sequence[float]] = None,
    r_min_phys: float = 1.0,
    q_asymptote_frac: float = 0.25,
    init_scale: float = 1.0,
    fit_background: bool = True,
    bg_order: int = 0,
    fit_offset: Optional[bool] = None,
    fit_number_density: bool = False,
    steps: int = 60,
    lr: float = 0.2,
    w_lowr: float = 1.0,
    w_highq: float = 1.0,
) -> RefineResult:
    """Fit (scale, background[, ρ₀]) by gradient descent on the physical constraints.

    Parameters
    ----------
    r_min_phys :
        Upper bound (Å) of the unphysical low-r region where ``G = −4π ρ₀ r``
        must hold (just below the nearest-neighbour distance).
    q_asymptote_frac :
        Fraction of the high-Q tail over which ``⟨S⟩ → 1`` is enforced.
    fit_background :
        Refine a smooth background subtracted from I(Q) before normalization —
        a polynomial ``b(Q) = Σ_j c_j (Q/Q_max)^j`` of degree ``bg_order``. This
        absorbs the fluorescence baseline and air/Compton-tail residuals.
        ``bg_order=0`` (default) is the single additive constant.
    fit_offset :
        Deprecated alias for ``fit_background`` (kept for back-compat); if given,
        overrides it.
    fit_number_density :
        Refine ρ₀ as well (scale is always refined).
    steps, lr :
        L-BFGS iterations and learning rate.

    Returns
    -------
    RefineResult with keys: ``scale``, ``offset`` (= b at Q=0), ``bg_coef``,
    ``background`` (b(Q) on the input grid), ``number_density``, ``S``, ``G``,
    ``sigma_G``, ``loss``, ``history``.
    """
    if fit_offset is not None:
        fit_background = fit_offset
    if bg_order < 0:
        raise ValueError("bg_order must be >= 0")
    q_t = torch.as_tensor(q, dtype=torch.float64)
    I_t = torch.as_tensor(intensity, dtype=torch.float64)
    r_t = torch.as_tensor(r_grid, dtype=torch.float64)
    sig_I = (None if sigma_intensity is None
             else torch.as_tensor(sigma_intensity, dtype=torch.float64))

    # Region masks (constant across the optimization).
    q_hi = float(q_t.max())
    q_cut = q_hi - q_asymptote_frac * (q_hi - float(q_t.min()))
    highq_mask = q_t >= q_cut
    lowr_mask = (r_t > 0) & (r_t < r_min_phys)
    if not bool(highq_mask.any()):
        raise ValueError("no Q points in the high-Q asymptote window")
    if not bool(lowr_mask.any()):
        raise ValueError("no r points in the low-r window (check r_grid/r_min_phys)")

    log_scale = torch.tensor(np.log(init_scale), dtype=torch.float64, requires_grad=True)
    bg_coef = torch.zeros(bg_order + 1, dtype=torch.float64, requires_grad=fit_background)
    log_rho = torch.tensor(np.log(number_density), dtype=torch.float64,
                           requires_grad=fit_number_density)

    # Powers of (Q/Q_max) for the background polynomial (constant in the loop).
    q_scale = float(q_max) if q_max is not None else float(q_t.max())
    q_pows = torch.stack([(q_t / q_scale) ** j for j in range(bg_order + 1)], dim=1)

    params = [log_scale]
    if fit_background:
        params.append(bg_coef)
    if fit_number_density:
        params.append(log_rho)
    opt = torch.optim.LBFGS(params, lr=lr, max_iter=steps,
                            line_search_fn="strong_wolfe")

    history: list[float] = []

    def _background():
        return q_pows @ bg_coef                      # b(Q), shape == q

    def _forward():
        scale = torch.exp(log_scale)
        rho = torch.exp(log_rho)
        S, sigma_S = faber_ziman_S(
            I_t, q_t, composition, wavelength_A=wavelength_A,
            scale=scale, compton=compton, background=_background(),
            sigma_intensity=sig_I, fractions=fractions,
        )
        G, sigma_G = fourier_sine_transform(
            q_t, S, r_t, Q_max=q_max, window=window, sigma_S=sigma_S,
        )
        return scale, rho, S, G, sigma_G

    def closure():
        opt.zero_grad()
        _, rho, S, G, _ = _forward()
        loss_highq = ((S[highq_mask] - 1.0) ** 2).mean()
        target_lowr = -_FOUR_PI * rho * r_t[lowr_mask]
        loss_lowr = ((G[lowr_mask] - target_lowr) ** 2).mean()
        loss = w_highq * loss_highq + w_lowr * loss_lowr
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    opt.step(closure)

    with torch.no_grad():
        scale, rho, S, G, sigma_G = _forward()
        if sigma_G is None:
            sigma_G = torch.zeros_like(G)
        background = _background()
    return RefineResult(
        scale=float(scale.detach()),
        offset=float(bg_coef[0].detach()),
        bg_coef=[float(c) for c in bg_coef.detach()],
        background=background.detach(),
        number_density=float(rho.detach()),
        S=S.detach(),
        G=G.detach(),
        sigma_G=sigma_G.detach(),
        loss=history[-1] if history else float("nan"),
        history=history,
    )
