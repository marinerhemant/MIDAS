"""Grand-canonical Monte Carlo moves for RMC.

Rev 6 RMC keeps the atom count N fixed. That's the canonical ensemble;
it CANNOT reproduce a target G(r) with a different number density (the
``-4π r ρ₀`` term is locked to the initial N). For vacancy / defect
studies, we need variable-N moves.

This module adds:

  * :class:`InsertMove`  — insert an atom at a random position (with a
    hard-sphere veto so the insertion doesn't overlap an existing atom).
  * :class:`RemoveMove`  — remove a random atom.
  * :func:`grand_canonical_metropolis_step` — Metropolis acceptance
    with a chemical-potential term ``μ`` (favours higher / lower N).

Together with the Rev 6 canonical moves, these implement grand-canonical
RMC.  Physically ``μ`` is a Lagrange multiplier that controls the mean N
along the chain: μ > 0 biases toward more atoms, μ < 0 fewer.

Formal acceptance criterion (Frenkel & Smit ch. 5.6):

    Insert:   P_acc = min(1, (V / (N+1)) · exp((μ − ΔU) / kT))
    Remove:   P_acc = min(1, (N / V)   · exp((−μ − ΔU) / kT))

where ΔU = χ²_new − χ²_old plays the role of the potential energy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class InsertMove:
    """Propose inserting an atom of the given species at a random location.

    The trial position is drawn uniformly in the supercell (fractional in
    [0, 1)³). ``min_distance_A`` is enforced by the Metropolis step, not
    here.
    """
    species: str = "X"
    rng: Optional[torch.Generator] = None

    def propose(self, supercell) -> Tuple[str, torch.Tensor]:
        """Return (species, cartesian_position).  Position is inside the box."""
        g = self.rng
        frac = torch.rand(3, generator=g, dtype=torch.float64)
        cart = frac @ supercell.cell
        return self.species, cart


@dataclass
class RemoveMove:
    """Propose removing a random atom (any species)."""
    rng: Optional[torch.Generator] = None
    species: Optional[str] = None      # if given, only remove atoms of this species

    def propose(self, supercell) -> int:
        """Return the atom index to remove."""
        g = self.rng
        if self.species is None:
            i = int(torch.randint(0, supercell.n_atoms, (1,), generator=g).item())
            return i
        # Restricted to a specific species
        matching = [i for i, s in enumerate(supercell.species) if s == self.species]
        if not matching:
            raise RuntimeError(f"RemoveMove: no atoms of species {self.species!r}")
        j = int(torch.randint(0, len(matching), (1,), generator=g).item())
        return matching[j]


# ---------------------------------------------------------------------------
# Grand-canonical Metropolis
# ---------------------------------------------------------------------------

def grand_canonical_metropolis_step(
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
    chemical_potential: float = 0.0,
    current_chi2: Optional[float] = None,
    min_distance_A: Optional[float] = None,
    rng: Optional[torch.Generator] = None,
) -> dict:
    """Propose an insert/remove move and Metropolis-accept with the chemical
    potential.

    Same return contract as :func:`midas_pdf.rmc.moves.metropolis_step`,
    with an extra ``"delta_N"`` key.
    """
    from .moves import chi2_supercell
    from .supercell import Supercell

    if current_chi2 is None:
        current_chi2, _ = chi2_supercell(
            supercell, r_grid, G_obs, sigma_G=sigma_G,
            u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs)

    result = {
        "accepted": False, "chi2_new": current_chi2,
        "chi2_old": current_chi2, "delta_chi2": 0.0,
        "delta_N": 0, "n_atoms": supercell.n_atoms,
    }

    V = supercell.volume
    N = supercell.n_atoms
    T = float(temperature)
    mu = float(chemical_potential)

    if isinstance(move, InsertMove):
        species, pos = move.propose(supercell)

        # Hard-sphere veto (fast)
        if min_distance_A is not None:
            # distance from proposed position to all existing atoms (with PBC)
            cell_inv = torch.linalg.inv(supercell.cell)
            d_vec = supercell.positions - pos
            frac = d_vec @ cell_inv
            frac = frac - torch.round(frac)
            d_min_image = frac @ supercell.cell
            dists = torch.linalg.norm(d_min_image, dim=1)
            if float(dists.min()) < min_distance_A:
                return result           # rejected pre-χ²

        # Build the N+1 trial supercell
        trial_positions = torch.cat([supercell.positions, pos.unsqueeze(0)], dim=0)
        trial_species = list(supercell.species) + [species]
        trial_sc = Supercell(species=trial_species,
                              positions=trial_positions,
                              cell=supercell.cell)
        chi2_new, _ = chi2_supercell(
            trial_sc, r_grid, G_obs, sigma_G=sigma_G,
            u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs)
        delta_chi2 = chi2_new - current_chi2
        # P_acc = (V/(N+1)) · exp((μ − Δχ²/2) / T)
        log_p = np.log(V / (N + 1.0)) + (mu - delta_chi2 / 2.0) / T
        p_accept = min(1.0, float(np.exp(min(log_p, 0.0))))
        u_rand = torch.rand((), generator=rng, dtype=torch.float64).item()
        if u_rand < p_accept:
            supercell.species.append(species)
            supercell.positions = trial_positions
            result.update(accepted=True, chi2_new=chi2_new,
                           delta_chi2=delta_chi2, delta_N=+1)

    elif isinstance(move, RemoveMove):
        if N <= 1:
            return result                       # can't shrink below 1 atom
        try:
            i_remove = move.propose(supercell)
        except RuntimeError:
            return result

        keep = torch.ones(N, dtype=torch.bool)
        keep[i_remove] = False
        trial_positions = supercell.positions[keep]
        trial_species = [s for k, s in zip(keep.tolist(), supercell.species) if k]
        trial_sc = Supercell(species=trial_species,
                              positions=trial_positions,
                              cell=supercell.cell)
        chi2_new, _ = chi2_supercell(
            trial_sc, r_grid, G_obs, sigma_G=sigma_G,
            u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs)
        delta_chi2 = chi2_new - current_chi2
        # P_acc = (N/V) · exp((−μ − Δχ²/2) / T)
        log_p = np.log(N / V) + (-mu - delta_chi2 / 2.0) / T
        p_accept = min(1.0, float(np.exp(min(log_p, 0.0))))
        u_rand = torch.rand((), generator=rng, dtype=torch.float64).item()
        if u_rand < p_accept:
            supercell.positions = trial_positions
            supercell.species = trial_species
            result.update(accepted=True, chi2_new=chi2_new,
                           delta_chi2=delta_chi2, delta_N=-1)

    else:                                       # pragma: no cover
        raise TypeError(f"unknown GC move type: {type(move)}")

    return result


__all__ = [
    "InsertMove",
    "RemoveMove",
    "grand_canonical_metropolis_step",
]
