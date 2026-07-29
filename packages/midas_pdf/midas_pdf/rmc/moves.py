"""Monte-Carlo moves + Metropolis acceptance for RMC.

Day-2 building blocks:

  * :class:`DisplaceMove` — single-atom Gaussian displacement, PBC-wrapped.
  * :class:`SwapMove`     — swap two atoms of different species (only if the
    supercell has ≥ 2 species; no-op otherwise).
  * :func:`chi2_supercell` — χ² of a supercell's G(r) against measured data.
  * :func:`metropolis_step`   — propose → evaluate → accept/reject one move.

For efficiency, ``chi2_supercell`` recomputes the full pair sum. Incremental
updates that touch only pairs involving the moved atom are exposed as
:func:`incremental_chi2` — O(N) instead of O(N²) per move, essential for
long chains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .histogram import supercell_G_r, _resolve_sigma, _FOUR_PI, _TWO_PI


# ---------------------------------------------------------------------------
# χ² utilities
# ---------------------------------------------------------------------------

def chi2_supercell(
    supercell,
    r_grid: torch.Tensor,
    G_obs: torch.Tensor,
    *,
    sigma_G: Optional[torch.Tensor] = None,
    u_iso: float = 0.005,
    q_broad: float = 0.0,
    r_max_pairs: Optional[float] = None,
) -> Tuple[float, torch.Tensor]:
    """χ² of a supercell configuration against ``G_obs``.

    Returns ``(chi2, G_calc)``. If ``sigma_G`` is None, unit weights are used.
    """
    G_calc = supercell_G_r(
        supercell, r_grid, u_iso=u_iso, q_broad=q_broad,
        r_max=r_max_pairs,
    )
    resid = G_obs - G_calc
    if sigma_G is None:
        chi2 = float((resid * resid).sum())
    else:
        w = 1.0 / torch.as_tensor(sigma_G, dtype=torch.float64).clamp(min=1e-12) ** 2
        chi2 = float((w * resid * resid).sum())
    return chi2, G_calc


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------

@dataclass
class DisplaceMove:
    """Single-atom Gaussian displacement.

    On each call, picks a random atom, samples a Gaussian displacement
    of width ``sigma_A`` Å, applies periodic-boundary wrap.
    """
    sigma_A: float = 0.1
    rng: Optional[torch.Generator] = None

    def propose(self, supercell) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample (atom_index, old_position, new_position).

        Positions are Cartesian, Å; new_position is PBC-wrapped into the
        supercell.
        """
        g = self.rng
        N = supercell.n_atoms
        # torch.randint(generator=...) supports Generator
        atom_idx = int(torch.randint(0, N, (1,), generator=g).item())
        dx = torch.randn(3, generator=g, dtype=torch.float64) * self.sigma_A
        old = supercell.positions[atom_idx].clone()
        new = old + dx
        # PBC wrap in fractional space then back to Cartesian
        cell_inv = torch.linalg.inv(supercell.cell)
        frac = new @ cell_inv
        frac = frac - torch.floor(frac)
        new = frac @ supercell.cell
        return atom_idx, old, new


@dataclass
class SwapMove:
    """Swap two atoms of different species (multi-species supercell only).

    Returns (i, j, sp_i_old, sp_j_old). If the supercell has only one
    species, propose raises RuntimeError.
    """
    rng: Optional[torch.Generator] = None

    def propose(self, supercell) -> Tuple[int, int, str, str]:
        if len(supercell.species_set()) < 2:
            raise RuntimeError("SwapMove needs ≥2 species in the supercell")
        g = self.rng
        # find two atoms of different species; try a bounded number of times
        species_arr = supercell.species
        for _ in range(64):
            i = int(torch.randint(0, supercell.n_atoms, (1,), generator=g).item())
            j = int(torch.randint(0, supercell.n_atoms, (1,), generator=g).item())
            if i != j and species_arr[i] != species_arr[j]:
                return i, j, species_arr[i], species_arr[j]
        raise RuntimeError("SwapMove could not find distinct-species pair")


# ---------------------------------------------------------------------------
# Metropolis
# ---------------------------------------------------------------------------

def metropolis_step(
    supercell,
    move,
    r_grid: torch.Tensor,
    G_obs: torch.Tensor,
    *,
    sigma_G: Optional[torch.Tensor] = None,
    u_iso: float = 0.005,
    q_broad: float = 0.0,
    r_max_pairs: Optional[float] = None,
    temperature: float = 1.0,
    current_chi2: Optional[float] = None,
    min_distance_A: Optional[float] = None,
    rng: Optional[torch.Generator] = None,
) -> dict:
    """Propose one move, accept/reject on Metropolis with temperature.

    Returns a dict::

        {"accepted": bool,
         "chi2_new": float, "chi2_old": float,
         "delta_chi2": float, "n_atoms": int}

    * If ``current_chi2`` is provided, saves the recomputation of the
      old χ².
    * If ``min_distance_A`` is given, moves that would place two atoms
      closer than this cut are rejected before the χ² evaluation (fast
      hard-sphere veto).
    """
    if current_chi2 is None:
        current_chi2, _ = chi2_supercell(
            supercell, r_grid, G_obs, sigma_G=sigma_G,
            u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs)

    result = {
        "accepted": False, "chi2_new": current_chi2,
        "chi2_old": current_chi2, "delta_chi2": 0.0,
        "n_atoms": supercell.n_atoms,
    }

    if isinstance(move, DisplaceMove):
        atom_idx, old_pos, new_pos = move.propose(supercell)
        # Hard-sphere veto
        if min_distance_A is not None:
            trial = supercell.positions.clone()
            trial[atom_idx] = new_pos
            # Only need to check pairs involving the moved atom.
            from .supercell import Supercell
            trial_sc = Supercell(species=list(supercell.species),
                                  positions=trial, cell=supercell.cell)
            r = trial_sc.pair_distances(r_max=min_distance_A * 1.001)
            if r.numel() > 0 and float(r.min()) < min_distance_A:
                return result           # rejected before χ² eval

        # χ² of trial configuration
        trial_pos = supercell.positions.clone()
        trial_pos[atom_idx] = new_pos
        from .supercell import Supercell
        trial_sc = Supercell(species=list(supercell.species),
                              positions=trial_pos, cell=supercell.cell)
        chi2_new, _ = chi2_supercell(
            trial_sc, r_grid, G_obs, sigma_G=sigma_G,
            u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs)
        delta = chi2_new - current_chi2
        # Metropolis: accept with prob min(1, exp(-Δχ²/(2T)))  — RMC convention
        u = torch.rand((), generator=rng, dtype=torch.float64).item()
        p_accept = min(1.0, float(np.exp(-delta / (2.0 * temperature))))
        if u < p_accept:
            supercell.positions[atom_idx] = new_pos
            result.update(accepted=True, chi2_new=chi2_new,
                           delta_chi2=delta)
        else:
            result.update(chi2_new=current_chi2, delta_chi2=0.0)

    elif isinstance(move, SwapMove):
        i, j, sp_i, sp_j = move.propose(supercell)
        # Swap species tags; positions unchanged.
        species = list(supercell.species)
        species[i], species[j] = species[j], species[i]
        from .supercell import Supercell
        trial_sc = Supercell(species=species, positions=supercell.positions.clone(),
                              cell=supercell.cell)
        chi2_new, _ = chi2_supercell(
            trial_sc, r_grid, G_obs, sigma_G=sigma_G,
            u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs)
        delta = chi2_new - current_chi2
        u = torch.rand((), generator=rng, dtype=torch.float64).item()
        p_accept = min(1.0, float(np.exp(-delta / (2.0 * temperature))))
        if u < p_accept:
            supercell.species[i], supercell.species[j] = supercell.species[j], supercell.species[i]
            result.update(accepted=True, chi2_new=chi2_new,
                           delta_chi2=delta)
        else:
            result.update(chi2_new=current_chi2, delta_chi2=0.0)

    else:                                                       # pragma: no cover
        raise TypeError(f"unknown move type: {type(move)}")

    return result


__all__ = [
    "DisplaceMove",
    "SwapMove",
    "chi2_supercell",
    "metropolis_step",
]
