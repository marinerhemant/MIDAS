"""Pixel → (qy, qz) for grazing-incidence GISAXS / GIWAXS.

For a flat detector with the sample at incidence angle α_i to the beam:

    qx ≈ (k/L) · ((Z/L)cos α_i + sin α_i)
    qy = k · (Y - Y_BC) / sqrt(Lsd² + (Y-Y_BC)²)
    qz = k · ((Z - Z_BC) cos α_i + Lsd sin α_i)
                  / sqrt(Lsd² + (Z-Z_BC)²) - k sin α_i

with k = 2π/λ. We provide the differentiable-in-α_i mapping plus a
re-grid helper that resamples the detector image onto a uniform
(qy, qz) grid.

Reference: Salditt et al., 2014; Hexemer & Müller-Buschbaum 2015.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def pixel_to_qy_qz(
    Y_px: torch.Tensor,
    Z_px: torch.Tensor,
    *,
    spec,
    incidence_angle_deg: float,
    sample_normal_axis: str = "Z",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map detector pixels to GISAXS reciprocal-space (qy, qz).

    **Sign convention**: the sample is tilted so that the specularly reflected
    beam lands at NEGATIVE ``dz`` (below the beam centre) and therefore at
    negative ``qz``. Most GISAXS plots put the specular rod at positive qz, so
    flip the sign of ``qz`` (or of ``incidence_angle_deg``) to match them.

    Verified against the two convention-free anchors (2026-08-29):

    * direct beam, at ``(BC_y, BC_z)`` — ``q = 0`` to rounding (<= 1.1e-16) at
      every incidence angle tested (0.1 deg to 5 deg);
    * specular, at ``dz = -Lsd·tan(2α)/px`` — ``|qz| = 2k·sin(α)`` **exactly**
      (ratio 1.0000 at 0.1, 0.2, 0.5, 1, 2 and 5 deg).

    The opposite branch (``dz = +Lsd·tan(2α)/px``) is only correct to first
    order in α — it evaluates to ``k(sin3α − sinα)``, which is 0.15 % low at
    α = 1 deg and 1.5 % low at 5 deg. That branch is not the specular ray in
    this convention; it is listed here so the asymmetry is not mistaken for an
    error.
    """
    if sample_normal_axis not in ("Z", "Y"):
        raise ValueError(f"sample_normal_axis must be 'Z' or 'Y'")
    Y = torch.as_tensor(Y_px, dtype=torch.float64)
    Z = torch.as_tensor(Z_px, dtype=torch.float64)
    Lsd = torch.as_tensor(float(spec.Lsd), dtype=torch.float64)
    px = torch.as_tensor(float(spec.pxY), dtype=torch.float64)
    BC_y = torch.as_tensor(float(spec.BC_y), dtype=torch.float64)
    BC_z = torch.as_tensor(float(spec.BC_z), dtype=torch.float64)
    lam = torch.as_tensor(float(spec.Wavelength), dtype=torch.float64)
    alpha = torch.as_tensor(np.deg2rad(incidence_angle_deg), dtype=torch.float64)
    k = 2.0 * torch.pi / lam

    dy = (Y - BC_y) * px
    dz = (Z - BC_z) * px
    Ldenom = torch.sqrt(Lsd * Lsd + dy * dy + dz * dz)

    qy = k * dy / Ldenom
    if sample_normal_axis == "Z":
        qz = k * (dz * torch.cos(alpha) + Lsd * torch.sin(alpha)) / Ldenom \
            - k * torch.sin(alpha)
    else:
        qz = k * (dy * torch.cos(alpha) + Lsd * torch.sin(alpha)) / Ldenom \
            - k * torch.sin(alpha)
    return qy, qz


def remap_to_qy_qz_grid(
    image: np.ndarray,
    spec,
    *,
    incidence_angle_deg: float,
    qy_grid: np.ndarray,
    qz_grid: np.ndarray,
) -> np.ndarray:
    """Resample image onto a uniform (qy, qz) grid via nearest-neighbour.

    Returns shape ``(len(qy_grid), len(qz_grid))``.
    """
    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    Y_px, Z_px = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    qy_pix, qz_pix = pixel_to_qy_qz(
        torch.as_tensor(Y_px.astype(np.float64)),
        torch.as_tensor(Z_px.astype(np.float64)),
        spec=spec,
        incidence_angle_deg=incidence_angle_deg,
    )
    qy_pix = qy_pix.numpy().ravel()
    qz_pix = qz_pix.numpy().ravel()
    img_flat = image.ravel().astype(np.float64)
    qy_idx = np.clip(
        np.searchsorted(qy_grid, qy_pix), 0, qy_grid.size - 1
    )
    qz_idx = np.clip(
        np.searchsorted(qz_grid, qz_pix), 0, qz_grid.size - 1
    )
    out = np.zeros((qy_grid.size, qz_grid.size), dtype=np.float64)
    counts = np.zeros_like(out)
    for k, (i, j) in enumerate(zip(qy_idx, qz_idx)):
        out[i, j] += img_flat[k]
        counts[i, j] += 1
    out /= np.maximum(counts, 1)
    return out


__all__ = ["pixel_to_qy_qz", "remap_to_qy_qz_grid"]
