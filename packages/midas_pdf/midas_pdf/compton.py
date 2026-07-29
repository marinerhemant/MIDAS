"""Hubbell-grade Compton (incoherent) scattering for total-scattering normalization.

Upgrades the coarse IT94 analytic form (``S_inc = Z(1 - e^{-alpha Q})`` in
``midas_integrate_v2.corrections.compton``) to the tabulated incoherent
scattering functions of Hubbell et al. (1975) — the same data GudrunX / PDFgetX
use — shipped as ``data/incoherent_hubbell.json`` (generated from xraylib at
build time; no xraylib runtime dependency).

The incoherent intensity per atom, composition-weighted, is

    I_inc(Q) = R(Q, lambda)^k * sum_i c_i * S_inc(q, Z_i),    q = Q / (4*pi)

where ``S_inc`` is the Hubbell incoherent scattering function (-> 0 at Q=0,
-> Z at high Q) and ``R`` is the **Breit-Dirac recoil factor** correcting for
the energy shift of Compton-scattered photons,

    R(Q, lambda) = 1 / [ 1 + (E0 / m_e c^2)(1 - cos psi) ],
    E0 = 12.398 / lambda[A]  keV,  m_e c^2 = 511 keV,
    (1 - cos psi) = 2 (Q lambda / 4 pi)^2,

with exponent ``k`` set by the detector (k=2 for a photon-counting area
detector — the default; k=3 for an energy-dispersive detector). Everything is a
torch op, differentiable in Q, wavelength, and composition fractions.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from midas_hkls.absorption import Z_for

__all__ = ["incoherent_scattering", "breit_dirac_factor"]

_FOUR_PI = 4.0 * float(np.pi)
_MEC2_KEV = 510.998950
_HC_KEV_A = 12.398419843320026  # h c in keV*Angstrom


@lru_cache(maxsize=1)
def _table() -> Tuple[np.ndarray, dict]:
    raw = json.loads(
        files("midas_pdf").joinpath("data/incoherent_hubbell.json").read_text()
    )
    q_grid = np.asarray(raw["q_grid"], dtype=np.float64)
    return q_grid, raw["S_inc"]


def _interp_columns(
    q_query: torch.Tensor, q_grid: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation, differentiable in ``q_query``.

    ``cols`` is (Ngrid, Ncol) on the uniform ``q_grid``; ``q_query`` may have
    any shape and the result is ``(*q_query.shape, Ncol)``. Uses the uniform
    spacing of the build-time grid for an O(1) index.
    """
    dq = q_grid[1] - q_grid[0]
    n = q_grid.shape[0]
    flat = q_query.reshape(-1)
    pos = ((flat - q_grid[0]) / dq).clamp(0.0, float(n - 1) - 1e-6)
    i0 = pos.floor().long()
    frac = (pos - i0.to(pos.dtype)).unsqueeze(-1)          # (Nflat, 1)
    lo = cols.index_select(0, i0)                          # (Nflat, Ncol)
    hi = cols.index_select(0, (i0 + 1).clamp(max=n - 1))
    out = lo * (1.0 - frac) + hi * frac
    return out.reshape(*q_query.shape, cols.shape[1])


def breit_dirac_factor(
    q: torch.Tensor,
    *,
    wavelength_A: float | torch.Tensor,
    k: int = 2,
) -> torch.Tensor:
    """Breit-Dirac recoil factor ``R^k`` (see module docstring)."""
    q_t = torch.as_tensor(q, dtype=torch.float64)
    lam = torch.as_tensor(wavelength_A, dtype=torch.float64)
    E0 = _HC_KEV_A / lam
    one_minus_cos = 2.0 * (q_t * lam / _FOUR_PI) ** 2
    R = 1.0 / (1.0 + (E0 / _MEC2_KEV) * one_minus_cos)
    return R ** k


def incoherent_scattering(
    q: torch.Tensor | np.ndarray,
    elements: Sequence[str],
    *,
    wavelength_A: float | torch.Tensor,
    fractions: Optional[torch.Tensor | Sequence[float]] = None,
    breit_dirac: bool = True,
    k: int = 2,
) -> torch.Tensor:
    """Composition-weighted Hubbell incoherent intensity per atom, I_inc(Q).

    Parameters
    ----------
    q : Q grid (Å^-1).
    elements : element symbols (one per species); fractions default to equal.
    wavelength_A : incident wavelength (Å), for the Breit-Dirac factor.
    fractions : number fractions aligned with ``elements`` (need not be
        normalized — renormalized here); a tensor with requires_grad makes the
        result differentiable in composition.
    breit_dirac : apply the recoil factor (default True).
    k : Breit-Dirac exponent (2 = photon-counting detector, 3 = energy-dispersive).

    Returns a tensor shaped like ``q``.
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    q_grid_np, S_inc = _table()
    q_grid = torch.as_tensor(q_grid_np, dtype=torch.float64)

    Zs = [Z_for(e) for e in elements]
    cols_np = np.stack([np.asarray(S_inc[str(Z)], dtype=np.float64) for Z in Zs], axis=1)
    cols = torch.as_tensor(cols_np, dtype=torch.float64)          # (Ngrid, Nel)

    q_for_table = q_t / _FOUR_PI                                  # Hubbell uses Q/4pi
    S_per_elem = _interp_columns(q_for_table, q_grid, cols)       # (Nq, Nel)

    if fractions is None:
        c = torch.full((len(elements),), 1.0 / len(elements), dtype=torch.float64)
    else:
        c = torch.as_tensor(fractions, dtype=torch.float64)
        c = c / c.sum()
    I_inc = (S_per_elem * c).sum(dim=-1)                          # (Nq,)

    if breit_dirac:
        I_inc = I_inc * breit_dirac_factor(q_t, wavelength_A=wavelength_A, k=k)
    return I_inc
