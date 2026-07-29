"""Ensemble heterogeneity: recover a thickness *distribution*, not just a mean.

A colloidal nanoplatelet sample is polydisperse; the measured rod is an
incoherent sum over thicknesses, smearing the Laue fringes.  The generic
mixture-deconvolution core lives in ``midas_invert``; here we build the
thickness basis (one rod per candidate N) and call it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["polydisperse_rod", "recover_thickness_distribution"]


def polydisperse_rod(crystal_t, hk, n_grid, weights, l, *, wavelength_A=1.0,
                     n_inplane=1e4):
    """Incoherent mixture rod ``I(l) = sum_k w_k I(l; N_k)``."""
    import torch
    from .forward import rod_intensity

    l = torch.as_tensor(l)
    weights = torch.as_tensor(weights, dtype=l.dtype, device=l.device)
    w = weights / weights.sum()
    hkl = torch.stack([torch.full_like(l, float(hk[0])),
                       torch.full_like(l, float(hk[1])), l], dim=-1)
    total = torch.zeros_like(l)
    for nk, wk in zip(n_grid, w):
        N = torch.tensor([n_inplane, n_inplane, float(nk)], dtype=l.dtype, device=l.device)
        total = total + wk * rod_intensity(crystal_t, hkl, N, wavelength_A=wavelength_A,
                                           apply_lp=False)
    return total


def _thickness_basis(crystal_t, hk, n_grid, l, *, wavelength_A=1.0, n_inplane=1e4):
    import torch
    from .forward import rod_intensity
    l = torch.as_tensor(l)
    hkl = torch.stack([torch.full_like(l, float(hk[0])),
                       torch.full_like(l, float(hk[1])), l], dim=-1)
    rows = []
    for nk in n_grid:
        N = torch.tensor([n_inplane, n_inplane, float(nk)], dtype=l.dtype, device=l.device)
        rows.append(rod_intensity(crystal_t, hkl, N, wavelength_A=wavelength_A, apply_lp=False))
    return torch.stack(rows)                                  # (K, L)


def recover_thickness_distribution(obs, crystal_t, hk, n_grid, l, *,
                                   wavelength_A=1.0, steps=800, lr=0.05,
                                   entropy_weight=0.0):
    """Recover the thickness distribution (softmax weights over ``n_grid``) from
    the smeared rod, via the shared mixture-deconvolution core.

    Returns dict with ``n_grid`` and ``weights`` (normalised, recovered).
    """
    import torch
    from midas_invert.mixture import mixture_deconvolution

    basis = _thickness_basis(crystal_t, hk, n_grid, l, wavelength_A=wavelength_A)
    w = mixture_deconvolution(obs, basis, loss="cosine", steps=steps, lr=lr,
                              entropy_weight=entropy_weight)
    return {"n_grid": torch.as_tensor(n_grid, dtype=basis.dtype), "weights": w}
