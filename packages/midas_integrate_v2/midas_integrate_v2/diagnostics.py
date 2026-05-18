"""Per-ring quality diagnostics for cake (η, R) integrated images.

A well-aligned powder pattern has azimuthally uniform intensity around
each ring. Deviations from η-uniformity within a ring's capture radius
flag:

- Mis-centred beam (sinusoidal η dependence).
- Texture / preferred orientation (multi-modal η).
- Tilt residuals (azimuthal modulation).
- Sample stress (η-dependent ring shift).

:func:`per_ring_chi_squared` computes :math:`\\chi^2/\\mathrm{dof}` per
ring as the variance of ``I(η)`` divided by the per-bin variance estimate
inside a small ±capture-radius window. Returns one number per ring; a
clean isotropic powder gives values near 1.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def per_ring_chi_squared(
    int2d: torch.Tensor,
    eta_axis_deg: torch.Tensor,
    R_axis: torch.Tensor,
    ring_R_px: torch.Tensor,
    *,
    capture_radius_px: float = 5.0,
    sigma2d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-ring χ²/dof of I(η) deviation from the η-mean.

    Parameters
    ----------
    int2d :
        2D integrated array, shape ``(n_eta, n_r)``.
    eta_axis_deg :
        ``(n_eta,)`` η-axis values in degrees.
    R_axis :
        ``(n_r,)`` R-axis values (pixels or any monotone unit).
    ring_R_px :
        ``(n_rings,)`` ring centre values in the same unit as R_axis.
    capture_radius_px :
        Half-width (same unit as R_axis) of the radial window summed
        per η-bin to form the per-ring intensity profile.
    sigma2d :
        Optional ``(n_eta, n_r)`` per-bin σ. If provided, χ² uses these
        explicit weights; if not, Poisson σ²=I is assumed.

    Returns
    -------
    chi2 : ``(n_rings,)`` torch.Tensor.
    """
    int2d_t = torch.as_tensor(int2d, dtype=torch.float64)
    eta_t = torch.as_tensor(eta_axis_deg, dtype=torch.float64)
    R_t = torch.as_tensor(R_axis, dtype=torch.float64)
    rings = torch.as_tensor(ring_R_px, dtype=torch.float64).reshape(-1)
    if int2d_t.shape != (eta_t.shape[0], R_t.shape[0]):
        raise ValueError(
            f"int2d shape {tuple(int2d_t.shape)} != (n_eta={eta_t.shape[0]}, "
            f"n_r={R_t.shape[0]})"
        )
    if sigma2d is None:
        var2d = int2d_t.clamp(min=1.0)  # Poisson + min-variance floor
    else:
        sig_t = torch.as_tensor(sigma2d, dtype=torch.float64)
        var2d = sig_t * sig_t

    chi2 = torch.zeros(rings.shape[0], dtype=torch.float64)
    for k, R0 in enumerate(rings):
        mask = (R_t - R0).abs() <= capture_radius_px
        if not bool(mask.any()):
            chi2[k] = float("nan")
            continue
        I_in = int2d_t[:, mask]                  # (n_eta, m)
        var_in = var2d[:, mask]
        # Per-η aggregate intensity inside the capture window
        I_eta = I_in.sum(dim=1)
        var_eta = var_in.sum(dim=1).clamp(min=1e-30)
        # χ²/dof of I(η) vs constant η-mean
        I_mean = I_eta.mean()
        n_dof = max(I_eta.shape[0] - 1, 1)
        chi2[k] = ((I_eta - I_mean) ** 2 / var_eta).sum() / n_dof
    return chi2


__all__ = ["per_ring_chi_squared"]
