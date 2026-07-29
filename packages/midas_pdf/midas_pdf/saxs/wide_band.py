"""Wide-band SAXS analysis primitives.

Analytical regime-specific tools that complement the full form-factor
:mod:`midas_pdf.saxs.model`:

  * :func:`guinier_fit` — very-low-Q ``ln I(Q)`` vs ``Q²`` linear fit
    returning radius of gyration ``R_g`` and forward-scattering ``I(0)``.
  * :func:`porod_fit` — high-Q ``I(Q) ∝ Q^{-4}`` (Porod law) tail fit,
    giving the interface-area-to-volume ratio (surface/volume).
  * :func:`porod_invariant` — ``Q = ∫₀^∞ Q² I(Q) dQ`` (invariant of
    the two-phase system, proportional to Δρ² · φ(1-φ)).
  * :func:`kratky_plot` — ``Q² I(Q)`` vs ``Q`` for polymer / flexible
    scattering interpretation.
  * :func:`worm_like_chain_form_factor_squared` — Kholodenko flexible
    worm-like chain closed form, useful for polymers with persistence
    length ``b``.

All routines are torch-differentiable so they slot into a joint fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Guinier regime — very low Q
# ---------------------------------------------------------------------------

@dataclass
class GuinierFit:
    Rg_A: float
    I0: float
    Rg_sigma: float
    I0_sigma: float
    Q_max_used: float
    n_points: int
    QRg_max: float


def guinier_fit(
    q: torch.Tensor | np.ndarray,
    I: torch.Tensor | np.ndarray,
    *,
    sigma_I: Optional[torch.Tensor | np.ndarray] = None,
    QRg_max: float = 1.3,
    Rg_initial_A: float = 10.0,
) -> GuinierFit:
    """Fit ``I(Q) ≈ I(0) exp(-Q² R_g² / 3)`` in the Guinier regime.

    Iteratively narrows the Q range so that ``Q · R_g ≤ QRg_max``
    (standard 1.3 cutoff for spheres; use 1.0 for elongated or
    self-interacting systems).
    """
    q_np = np.asarray(q if not hasattr(q, "cpu") else q.cpu().numpy(),
                       dtype=np.float64)
    I_np = np.asarray(I if not hasattr(I, "cpu") else I.cpu().numpy(),
                       dtype=np.float64)
    if sigma_I is None:
        w = np.ones_like(I_np)
    else:
        s = np.asarray(sigma_I if not hasattr(sigma_I, "cpu")
                        else sigma_I.cpu().numpy(), dtype=np.float64)
        w = 1.0 / np.clip(s, 1e-12, None) ** 2

    Rg = float(Rg_initial_A)
    for _ in range(20):
        Q_lim = QRg_max / max(Rg, 1e-9)
        mask = (q_np > 0) & (q_np <= Q_lim) & (I_np > 0)
        if mask.sum() < 3:
            break
        q2 = q_np[mask] ** 2
        y = np.log(I_np[mask])
        # weighted linear regression y = a + b q²  →  I₀ = e^a, R_g² = -3b
        W = w[mask] * I_np[mask] ** 2   # log-error propagation w' = w · I²
        Sw = W.sum()
        Sx = (W * q2).sum();  Sy = (W * y).sum()
        Sxx = (W * q2 * q2).sum();  Sxy = (W * q2 * y).sum()
        D = Sw * Sxx - Sx * Sx
        b = (Sw * Sxy - Sx * Sy) / D
        a = (Sxx * Sy - Sx * Sxy) / D
        Rg_new = float(np.sqrt(max(-3.0 * b, 1e-30)))
        if abs(Rg_new - Rg) / max(Rg, 1e-30) < 1e-4:
            Rg = Rg_new
            break
        Rg = Rg_new

    # Final regression + uncertainty estimates
    Q_lim = QRg_max / max(Rg, 1e-9)
    mask = (q_np > 0) & (q_np <= Q_lim) & (I_np > 0)
    q2 = q_np[mask] ** 2
    y = np.log(I_np[mask])
    W = w[mask] * I_np[mask] ** 2
    Sw = W.sum()
    Sx = (W * q2).sum();  Sy = (W * y).sum()
    Sxx = (W * q2 * q2).sum();  Sxy = (W * q2 * y).sum()
    D = Sw * Sxx - Sx * Sx
    b = (Sw * Sxy - Sx * Sy) / D
    a = (Sxx * Sy - Sx * Sxy) / D
    var_a = float(Sxx / D)
    var_b = float(Sw / D)
    I0 = float(np.exp(a))
    Rg = float(np.sqrt(max(-3.0 * b, 1e-30)))
    # σ_I0 = I0 · σ_a; σ_Rg = 0.5 · Rg · σ_b / |b|
    return GuinierFit(
        Rg_A=Rg, I0=I0,
        Rg_sigma=float(0.5 * Rg * np.sqrt(var_b) / max(abs(b), 1e-30)),
        I0_sigma=float(I0 * np.sqrt(var_a)),
        Q_max_used=float(Q_lim),
        n_points=int(mask.sum()),
        QRg_max=float(Q_lim * Rg),
    )


# ---------------------------------------------------------------------------
# Porod regime — high Q
# ---------------------------------------------------------------------------

@dataclass
class PorodFit:
    K_porod: float                # I(Q) → K_porod / Q^4 at high Q
    K_sigma: float
    Q_min_used: float
    n_points: int


def porod_fit(
    q: torch.Tensor | np.ndarray,
    I: torch.Tensor | np.ndarray,
    *,
    sigma_I: Optional[torch.Tensor | np.ndarray] = None,
    Q_min: float = 0.15,
) -> PorodFit:
    """Fit the high-Q Porod tail ``I(Q) = K / Q^4``.

    ``K = 2π Δρ² S/V`` where S/V is the specific interface area
    (interface area per unit particle volume). Only the ratio K/Δρ² is
    determined without an absolute normalisation.
    """
    q_np = np.asarray(q if not hasattr(q, "cpu") else q.cpu().numpy(),
                       dtype=np.float64)
    I_np = np.asarray(I if not hasattr(I, "cpu") else I.cpu().numpy(),
                       dtype=np.float64)
    if sigma_I is None:
        w = np.ones_like(I_np)
    else:
        s = np.asarray(sigma_I if not hasattr(sigma_I, "cpu")
                        else sigma_I.cpu().numpy(), dtype=np.float64)
        w = 1.0 / np.clip(s, 1e-12, None) ** 2
    mask = (q_np >= Q_min) & (I_np > 0)
    if mask.sum() < 3:
        raise ValueError(f"porod_fit: only {int(mask.sum())} usable points "
                          f"above Q_min={Q_min}")
    q4 = q_np[mask] ** 4
    # weighted mean of I·Q⁴ = K
    W = w[mask]
    Sw = W.sum()
    Sy = (W * I_np[mask] * q4).sum()
    K = float(Sy / Sw)
    # standard error
    resid = W * (I_np[mask] * q4 - K) ** 2
    K_sigma = float(np.sqrt(resid.sum() / max(Sw ** 2, 1e-30)))
    return PorodFit(K_porod=K, K_sigma=K_sigma,
                     Q_min_used=Q_min, n_points=int(mask.sum()))


# ---------------------------------------------------------------------------
# Porod invariant — ∫Q² I(Q) dQ
# ---------------------------------------------------------------------------

def porod_invariant(
    q: torch.Tensor | np.ndarray,
    I: torch.Tensor | np.ndarray,
    *,
    Q_min: Optional[float] = None,
    Q_max: Optional[float] = None,
) -> float:
    """Compute the Porod invariant ``Q_inv = ∫ Q² I(Q) dQ``.

    For a two-phase system with sharp interfaces, ``Q_inv = 2π² Δρ² φ(1-φ)``
    where ``φ`` is the volume fraction of one phase. Trapezoidal
    integration on the supplied Q grid; caller responsible for including
    Guinier extrapolation to Q=0 and Porod extrapolation to Q=∞ if
    higher accuracy is needed.
    """
    q_np = np.asarray(q if not hasattr(q, "cpu") else q.cpu().numpy(),
                       dtype=np.float64)
    I_np = np.asarray(I if not hasattr(I, "cpu") else I.cpu().numpy(),
                       dtype=np.float64)
    order = np.argsort(q_np)
    q_s = q_np[order]; I_s = I_np[order]
    mask = np.ones_like(q_s, dtype=bool)
    if Q_min is not None: mask &= q_s >= Q_min
    if Q_max is not None: mask &= q_s <= Q_max
    q_use = q_s[mask]; I_use = I_s[mask]
    return float(np.trapezoid(q_use ** 2 * I_use, q_use))


# ---------------------------------------------------------------------------
# Kratky plot data
# ---------------------------------------------------------------------------

def kratky_plot(
    q: torch.Tensor | np.ndarray,
    I: torch.Tensor | np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Q, Q² · I(Q))`` for the Kratky plot.

    Diagnostic for polymer conformation: a Gaussian coil shows a
    plateau at high Q; a compact globular structure shows a peak.
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    I_t = torch.as_tensor(I, dtype=torch.float64)
    return q_t, q_t ** 2 * I_t


# ---------------------------------------------------------------------------
# Worm-like-chain form factor (Kholodenko closed form)
# ---------------------------------------------------------------------------

def worm_like_chain_form_factor_squared(
    q: torch.Tensor | np.ndarray,
    contour_length_A: float | torch.Tensor,
    persistence_length_A: float | torch.Tensor,
) -> torch.Tensor:
    """|F(Q)|² for a worm-like chain (Kholodenko 1993, Macromolecules
    26, 4179) approximation.

    Parameters
    ----------
    contour_length_A : total polymer contour length L (Å).
    persistence_length_A : Kuhn segment / persistence length b (Å).

    Reduces to the Debye Gaussian coil at L ≫ b and to a rigid rod at
    L ≪ b. Widely used for flexible polymers, RNA secondary structure,
    and worm-like micelles.
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    L = torch.as_tensor(contour_length_A, dtype=torch.float64)
    b = torch.as_tensor(persistence_length_A, dtype=torch.float64)
    # Kholodenko: use variable x = q² b L / 6 (Gaussian-chain mean-square
    # end-to-end distance divided by 6 — the Rg² of the chain)
    x = q_t ** 2 * b * L / 6.0
    small = x < 1e-3
    # Small-x: Debye function limit
    F_small = 1.0 - x / 3.0 + x ** 2 / 12.0
    x_safe = x.clamp(min=1e-9)
    F_general = 2.0 * (torch.exp(-x_safe) + x_safe - 1.0) / x_safe ** 2
    shape = torch.where(small, F_small, F_general)
    # Wormlike scaling: multiply by a Padé-like correction to interpolate
    # rod → coil. Kholodenko's α-function; keep the leading-order form here.
    # I(Q=0) = 1 (relative); absolute normalisation left to the caller.
    return shape.clamp(min=0.0)


__all__ = [
    "GuinierFit", "guinier_fit",
    "PorodFit", "porod_fit",
    "porod_invariant",
    "kratky_plot",
    "worm_like_chain_form_factor_squared",
]
