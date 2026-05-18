"""PDF helpers: pixel/R → Q, S(Q) normalisation, FT to G(r), σ propagation.

The novel contribution here is **rigorous σ propagation through the
sine Fourier transform** that takes S(Q) → G(r). PDFgetX3 and other
production PDF tools propagate intensity but not uncertainty; the
Bayesian / risk-aware downstream (peak fitting, refinement, model
selection) needs σ on every G(r) point.

Per-bin variance through the FT (assuming Q-bin independence — true
for shot-noise-only data):

    G(r) = (2/π) ∫ Q [S(Q) - 1] sin(Qr) W(Q) dQ
    σ²(G(r)) ≈ (2/π · ΔQ)² Σ_q [Q sin(Qr) W(Q)]² σ²(S(Q))

where W(Q) is the optional window function (Lorch by default; rectangular
if Q_max is None).

This module is the foundation for the PDF CLI (Item 11) and the
PDFgetX3 round-trip test/notebook (Items 9 + 31).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


_DEG2RAD = float(np.pi / 180.0)


def R_px_to_Q(
    R_px: torch.Tensor | np.ndarray,
    *,
    Lsd_um: float | torch.Tensor,
    px_um: float | torch.Tensor,
    lambda_A: float | torch.Tensor,
) -> torch.Tensor:
    """Map detector radius (pixels) to scattering vector Q (Å⁻¹).

    ``Q = (4π/λ) sin(2θ/2)`` with ``2θ = atan(R · px / Lsd)``.
    All inputs may be torch tensors with ``requires_grad`` for
    differentiable Q-grid construction.
    """
    R = torch.as_tensor(R_px, dtype=torch.float64)
    Lsd = torch.as_tensor(Lsd_um, dtype=torch.float64)
    px = torch.as_tensor(px_um, dtype=torch.float64)
    lam = torch.as_tensor(lambda_A, dtype=torch.float64)
    two_theta = torch.atan(R * px / Lsd)
    return (4.0 * torch.pi / lam) * torch.sin(0.5 * two_theta)


def estimate_background(
    profile: torch.Tensor | np.ndarray,
    *,
    window: int = 51,
    percentile: float = 10.0,
) -> torch.Tensor:
    """Rolling percentile background filter.

    For each point, take the ``percentile``th percentile of the
    surrounding ``window``-wide neighbourhood. ``percentile=10`` rejects
    Bragg peaks while preserving the diffuse baseline; ``window=51``
    is large enough to avoid wandering with sharp peaks.
    """
    p = np.asarray(profile, dtype=np.float64)
    n = p.shape[0]
    if window < 1:
        raise ValueError("window must be >= 1")
    half = window // 2
    bg = np.empty_like(p)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        bg[i] = np.percentile(p[lo:hi], percentile)
    return torch.as_tensor(bg, dtype=torch.float64)


def normalize_to_S(
    intensity: torch.Tensor | np.ndarray,
    *,
    q: torch.Tensor | np.ndarray,
    atomic_form_factor_squared: torch.Tensor | np.ndarray,
    background: Optional[torch.Tensor | np.ndarray] = None,
    compton: Optional[torch.Tensor | np.ndarray] = None,
    sigma_intensity: Optional[torch.Tensor | np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Form-factor normalisation: I(Q) → S(Q), with σ propagation.

    Monoatomic Faber-Ziman convention::

        S(Q) = (I(Q) - background - compton) / <f²>

    Equivalent to the more general ``(I - <f²> - compton)/<f²> + 1``
    when ``background = <f²>`` (the standard subtraction of incoherent
    self-scattering from the measured intensity). For PDFgetX3 / Egami-
    Billinge sign conventions this matches their default monoatomic
    output. ``<f²>`` is the average atomic form factor squared (input).
    σ propagates as ``σ_S = σ_I / <f²>`` (denominators are constants).
    Returns (S, σ_S). σ_S is None-equivalent (zeros) if no input σ.
    """
    I = torch.as_tensor(intensity, dtype=torch.float64)
    q_t = torch.as_tensor(q, dtype=torch.float64)
    if I.shape != q_t.shape:
        raise ValueError(
            f"intensity shape {tuple(I.shape)} != q shape {tuple(q_t.shape)}"
        )
    f2 = torch.as_tensor(atomic_form_factor_squared, dtype=torch.float64)
    bg = (torch.zeros_like(I)
          if background is None
          else torch.as_tensor(background, dtype=torch.float64))
    cmp = (torch.zeros_like(I)
           if compton is None
           else torch.as_tensor(compton, dtype=torch.float64))
    f2_safe = f2.clamp(min=1e-30)
    S = (I - bg - cmp) / f2_safe
    if sigma_intensity is None:
        sigma_S = torch.zeros_like(S)
    else:
        sig = torch.as_tensor(sigma_intensity, dtype=torch.float64)
        sigma_S = sig / f2_safe
    return S, sigma_S


def _lorch_window(q: torch.Tensor, Q_max: float) -> torch.Tensor:
    """Lorch window: ``W(Q) = sin(πQ/Q_max) / (πQ/Q_max)``.

    Reduces termination ripples in the FT at the cost of slight
    real-space broadening. Returns 1 at Q=0; values < 1 elsewhere.
    """
    arg = torch.pi * q / max(Q_max, 1e-30)
    return torch.where(arg.abs() < 1e-12,
                        torch.ones_like(arg),
                        torch.sin(arg) / arg)


def fourier_sine_transform(
    q: torch.Tensor | np.ndarray,
    S_q: torch.Tensor | np.ndarray,
    r_grid: torch.Tensor | np.ndarray,
    *,
    Q_max: Optional[float] = None,
    window: str = "lorch",
    sigma_S: Optional[torch.Tensor | np.ndarray] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute G(r) from S(Q) via the sine Fourier transform.

        G(r) = (2/π) ∫ Q [S(Q) - 1] sin(Qr) W(Q) dQ

    Discretised as a Riemann sum with step ΔQ. If ``sigma_S`` is
    provided, propagates Q-bin-independent variance:

        σ²(G(r)) = (2/π · ΔQ)² Σ_q [Q sin(Qr) W(Q)]² σ²(S(Q))

    (Q-bin independence is exact for Poisson-noise-only data integrated
    bin-by-bin; the soft-bin / polygon-bin schemes do introduce mild
    inter-bin covariance but it's tiny in practice and ignored here.)
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    S_t = torch.as_tensor(S_q, dtype=torch.float64)
    r_t = torch.as_tensor(r_grid, dtype=torch.float64)
    if q_t.shape != S_t.shape:
        raise ValueError("q and S_q shapes must match")
    if q_t.ndim != 1 or r_t.ndim != 1:
        raise ValueError("q, S_q, r_grid must be 1-D")
    n = int(q_t.shape[0])
    if n < 2:
        raise ValueError("FT needs >= 2 Q points")
    dQ = float(q_t[1] - q_t[0])

    Q_used = float(q_t[-1]) if Q_max is None else float(Q_max)
    if window == "lorch":
        W = _lorch_window(q_t, Q_used)
    elif window in ("rect", "rectangular", "none"):
        W = torch.ones_like(q_t)
    else:
        raise ValueError(f"unknown window {window!r}; use 'lorch' or 'rect'")
    # Optional Q_max truncation
    if Q_max is not None:
        W = W * (q_t <= Q_max).to(q_t.dtype)

    # Outer product Qr (n_q, n_r)
    Qr = q_t.unsqueeze(1) * r_t.unsqueeze(0)
    sinQr = torch.sin(Qr)
    # Q sin(Qr) W
    kernel = (q_t * W).unsqueeze(1) * sinQr           # (n_q, n_r)
    integrand = (S_t - 1.0).unsqueeze(1) * kernel
    G = (2.0 / torch.pi) * dQ * integrand.sum(dim=0)

    if sigma_S is None:
        return G, None
    sig_t = torch.as_tensor(sigma_S, dtype=torch.float64)
    if sig_t.shape != q_t.shape:
        raise ValueError("sigma_S shape must match q shape")
    var_kernel = kernel * kernel                        # (n_q, n_r)
    var_S = (sig_t * sig_t).unsqueeze(1)                # (n_q, 1)
    var_G = ((2.0 / torch.pi) ** 2) * (dQ ** 2) * (var_S * var_kernel).sum(dim=0)
    sigma_G = torch.sqrt(var_G.clamp(min=0.0))
    return G, sigma_G


def integrate_to_Gr_with_variance(
    image: torch.Tensor,
    spec,
    r_grid: torch.Tensor | np.ndarray,
    *,
    binning: str = "polygon",
    corrections: Optional[dict] = None,
    Q_max: Optional[float] = None,
    Q_min: float = 0.5,
    Q_step: float = 0.01,
    atomic_form_factor_squared: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
    compton: Optional[torch.Tensor] = None,
    window: str = "lorch",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """End-to-end pixel → G(r) with σ propagation.

    Returns ``(r_grid, G, sigma_G)``. Steps:

      1. Build geometry from ``spec``; integrate ``image`` to (mean, σ)
         on a Q-grid.
      2. Eta-average to a 1-D I(Q), √(Σ σ²/N²) for the σ.
      3. Normalise to S(Q) via ``normalize_to_S`` (default
         ``<f²> = 1``).
      4. FT to G(r) on ``r_grid`` with ``fourier_sine_transform``.

    Designed for the PDF CLI (Item 11) and the PDFgetX3 round-trip
    test (Item 9). For production work supply real form factors via
    ``atomic_form_factor_squared``.
    """
    from .binning import (
        PolygonBinGeometry, HardBinGeometry,
        integrate_polygon_with_variance, integrate_hard_with_variance,
    )

    if binning == "polygon":
        geom = PolygonBinGeometry.from_spec(spec)
        mean2d, sigma2d = integrate_polygon_with_variance(image, geom)
    elif binning == "hard":
        geom = HardBinGeometry.from_spec(spec)
        mean2d, sigma2d = integrate_hard_with_variance(image, geom)
    else:
        raise ValueError(f"binning must be 'polygon' or 'hard', got {binning!r}")

    # Empty / off-detector bins come back as NaN; collapse the eta axis
    # with NaN-aware reductions so a few masked slices don't poison
    # every R bin in the 1-D profile.
    valid = torch.isfinite(mean2d)
    n_valid = valid.sum(dim=0).clamp(min=1)
    I = torch.where(valid, mean2d, torch.zeros_like(mean2d)).sum(dim=0) / n_valid
    sig2 = torch.where(valid, sigma2d * sigma2d, torch.zeros_like(sigma2d))
    sigma_I = torch.sqrt(sig2.sum(dim=0)) / n_valid

    # R axis → Q axis for the integrated profile
    R_axis = (
        spec.RMin
        + (torch.arange(I.shape[0], dtype=torch.float64) + 0.5) * spec.RBinSize
    )
    Q_axis = R_px_to_Q(R_axis, Lsd_um=spec.Lsd, px_um=spec.pxY,
                       lambda_A=spec.Wavelength)

    # Resample (linear) onto a uniform Q-grid for FT-friendliness
    if Q_max is None:
        Q_max_val = float(Q_axis[-1])
    else:
        Q_max_val = Q_max
    n_q = int(np.ceil((Q_max_val - Q_min) / Q_step)) + 1
    q_uniform = torch.linspace(Q_min, Q_max_val, n_q, dtype=torch.float64)
    Q_axis_np = Q_axis.detach().cpu().numpy()
    sort_idx = np.argsort(Q_axis_np)
    I_uniform = torch.as_tensor(
        np.interp(q_uniform.numpy(), Q_axis_np[sort_idx], I.detach().cpu().numpy()[sort_idx]),
        dtype=torch.float64,
    )
    sigma_uniform = torch.as_tensor(
        np.interp(q_uniform.numpy(), Q_axis_np[sort_idx],
                  sigma_I.detach().cpu().numpy()[sort_idx]),
        dtype=torch.float64,
    )

    f2 = (atomic_form_factor_squared if atomic_form_factor_squared is not None
          else torch.ones_like(q_uniform))
    S, sigma_S = normalize_to_S(
        I_uniform, q=q_uniform,
        atomic_form_factor_squared=f2,
        background=background, compton=compton,
        sigma_intensity=sigma_uniform,
    )
    r_t = torch.as_tensor(r_grid, dtype=torch.float64)
    G, sigma_G = fourier_sine_transform(
        q_uniform, S, r_t, Q_max=Q_max_val, window=window, sigma_S=sigma_S,
    )
    if sigma_G is None:
        sigma_G = torch.zeros_like(G)
    return r_t, G, sigma_G


__all__ = [
    "R_px_to_Q",
    "estimate_background",
    "normalize_to_S",
    "fourier_sine_transform",
    "integrate_to_Gr_with_variance",
]
