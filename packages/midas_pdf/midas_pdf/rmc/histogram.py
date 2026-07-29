"""Pair-distance → G(r) forward for a supercell configuration.

We match the normalisation of :func:`midas_pdf.structure.pdffit_gr`:

    G(r) = pair_term(r) / r  −  4π r ρ₀

with

    pair_term(r) = (1 / N_atoms)  Σ_{i≠j}  (1 / √(2π σ²)) exp( −(r − r_{ij})² / 2σ² )

Each unordered pair contributes twice (i,j and j,i) — implemented here by
scaling the unordered-pair sum by 2.  This matches ``pdffit_gr`` on a
crystalline supercell to sub-percent accuracy at the "Day 1" test.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

_TWO_PI = 2.0 * float(np.pi)
_FOUR_PI = 4.0 * float(np.pi)


def _resolve_sigma(u_iso: torch.Tensor | float,
                    q_broad: torch.Tensor | float,
                    r_ij: torch.Tensor) -> torch.Tensor:
    """σ_p per pair from thermal (u_iso) + q-broadening. Matches pdffit_gr."""
    u_pair = 2.0 * torch.as_tensor(u_iso, dtype=torch.float64)
    qb = torch.as_tensor(q_broad, dtype=torch.float64)
    sigma2 = (u_pair + (qb * r_ij) ** 2).clamp(min=1e-8)
    return torch.sqrt(sigma2)


def supercell_G_r(
    supercell,
    r_grid: torch.Tensor | np.ndarray,
    *,
    u_iso: float | torch.Tensor = 0.005,
    q_broad: float | torch.Tensor = 0.0,
    r_max: Optional[float] = None,
    min_image: bool = True,
) -> torch.Tensor:
    """Forward G(r) of a supercell — Gaussian-broadened pair sum.

    Parameters
    ----------
    supercell : :class:`Supercell`
    r_grid : (R,) real-space r-values (Å) at which to evaluate G(r).
    u_iso : isotropic displacement per atom (Å²). Contribution per pair is 2·u_iso.
    q_broad : Q-space broadening (Å⁻¹) — grows σ with r as (q_broad · r_ij).
    r_max : distance cut for pair evaluation. Default: ``r_grid.max() + 3σ_max``.
    min_image : use minimum-image PBC (recommended).

    Returns
    -------
    G(r) : (R,) tensor.
    """
    r_t = torch.as_tensor(r_grid, dtype=torch.float64)
    if r_max is None:
        # Include pairs a few σ beyond r_grid.max() so Gaussian tails don't
        # get chopped off.
        r_max = float(r_t.max()) + 5.0 * float(np.sqrt(2.0 * float(u_iso)))
    r_ij = supercell.pair_distances(r_max=r_max, min_image=min_image)
    if r_ij.numel() == 0:
        # No pairs → return the −4π r ρ₀ baseline
        return -_FOUR_PI * r_t * supercell.number_density

    sigma = _resolve_sigma(u_iso, q_broad, r_ij)                   # (P,)

    # Gaussian broadening: G_p(r) = (1/√(2π) σ) exp( −(r − r_p)² / 2σ² )
    rr = r_t[:, None]                                              # (R, 1)
    gauss = torch.exp(-0.5 * ((rr - r_ij[None, :]) / sigma[None, :]) ** 2) \
        / (np.sqrt(_TWO_PI) * sigma[None, :])                     # (R, P)

    # Sum with factor 2 (each unordered pair counted twice per pdffit_gr conv)
    pair_term = 2.0 * gauss.sum(dim=1) / supercell.n_atoms         # (R,)

    rho0 = supercell.number_density
    r_safe = r_t.clamp(min=1e-6)
    G = pair_term / r_safe - _FOUR_PI * r_t * rho0
    return G


def pair_distance_histogram(
    supercell,
    bins: int = 200,
    r_max: float = 10.0,
    *,
    min_image: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unweighted pair-distance histogram (returns bin_edges, counts).

    Useful for eyeballing the coordination shells before Gaussian broadening.
    """
    r_ij = supercell.pair_distances(r_max=r_max, min_image=min_image)
    hist = torch.histc(r_ij, bins=bins, min=0.0, max=r_max)
    edges = torch.linspace(0.0, r_max, bins + 1, dtype=torch.float64)
    return edges, hist
