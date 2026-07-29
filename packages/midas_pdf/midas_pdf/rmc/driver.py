"""RMC refinement driver.

Chains many Metropolis moves together, records χ² trace + acceptance,
returns the final configuration + trace summary. Day 2 checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .moves import DisplaceMove, SwapMove, chi2_supercell, metropolis_step
from .gc_moves import (
    InsertMove, RemoveMove, grand_canonical_metropolis_step,
)
from .cluster_moves import (
    ClusterDisplaceMove, RigidRotationMove, cluster_metropolis_step,
)


@dataclass
class RMCResult:
    supercell: object                              # final Supercell
    chi2_trace: List[float] = field(default_factory=list)
    accept_trace: List[bool] = field(default_factory=list)
    initial_chi2: float = float("nan")
    final_chi2: float = float("nan")
    n_moves: int = 0
    n_accepted: int = 0

    @property
    def acceptance_ratio(self) -> float:
        return (self.n_accepted / self.n_moves) if self.n_moves else float("nan")


def rmc_refine(
    supercell,
    r_grid: torch.Tensor,
    G_obs: torch.Tensor,
    *,
    sigma_G: Optional[torch.Tensor] = None,
    moves: Optional[Sequence] = None,
    n_moves: int = 5_000,
    u_iso: float = 0.005,
    q_broad: float = 0.0,
    r_max_pairs: Optional[float] = None,
    temperature: float = 1.0,
    temperature_schedule: Optional[callable] = None,
    min_distance_A: Optional[float] = None,
    chemical_potential: float = 0.0,
    seed: Optional[int] = None,
    log_every: int = 500,
    verbose: bool = False,
) -> RMCResult:
    """Run RMC on ``supercell``, minimising χ² of G_calc vs G_obs.

    Parameters
    ----------
    supercell : mutated in place.
    r_grid, G_obs : the target G(r).
    sigma_G : optional per-r σ for χ² weighting; unit weights if omitted.
    moves : list of proposals to sample from at each step. Default: a
        single ``DisplaceMove(sigma_A=0.1)``.
    n_moves : number of Metropolis attempts.
    temperature : Metropolis "temperature"; controls acceptance width.
    temperature_schedule : optional callable(step) -> temperature for
        simulated annealing.
    min_distance_A : hard-sphere veto before χ² eval (fast).
    seed : reproducibility.
    log_every : print progress every N accepted moves (if verbose).
    """
    if moves is None:
        moves = [DisplaceMove(sigma_A=0.1)]
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(int(seed))
    for m in moves:
        # Share the same RNG for reproducibility
        if hasattr(m, "rng"):
            m.rng = rng

    initial_chi2, _ = chi2_supercell(
        supercell, r_grid, G_obs, sigma_G=sigma_G,
        u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs)

    result = RMCResult(
        supercell=supercell, chi2_trace=[initial_chi2],
        accept_trace=[], initial_chi2=initial_chi2,
        final_chi2=initial_chi2, n_moves=0, n_accepted=0,
    )

    current_chi2 = initial_chi2
    n_move_types = len(moves)
    for step in range(n_moves):
        T = (temperature_schedule(step) if temperature_schedule is not None
             else temperature)
        move = moves[step % n_move_types]
        try:
            if isinstance(move, (InsertMove, RemoveMove)):
                step_result = grand_canonical_metropolis_step(
                    supercell, move, r_grid, G_obs, sigma_G=sigma_G,
                    u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs,
                    temperature=T, chemical_potential=chemical_potential,
                    current_chi2=current_chi2,
                    min_distance_A=min_distance_A, rng=rng,
                )
            elif isinstance(move, (ClusterDisplaceMove, RigidRotationMove)):
                step_result = cluster_metropolis_step(
                    supercell, move, r_grid, G_obs, sigma_G=sigma_G,
                    u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs,
                    temperature=T, current_chi2=current_chi2,
                    min_distance_A=min_distance_A, rng=rng,
                )
            else:
                step_result = metropolis_step(
                    supercell, move, r_grid, G_obs, sigma_G=sigma_G,
                    u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs,
                    temperature=T, current_chi2=current_chi2,
                    min_distance_A=min_distance_A, rng=rng,
                )
        except RuntimeError:
            # e.g. SwapMove couldn't find distinct-species pair — count as skip
            continue

        result.chi2_trace.append(step_result["chi2_new"])
        result.accept_trace.append(step_result["accepted"])
        result.n_moves += 1
        if step_result["accepted"]:
            result.n_accepted += 1
            current_chi2 = step_result["chi2_new"]

        if verbose and (step + 1) % log_every == 0:
            print(f"  step {step+1:6d}: χ²={current_chi2:.4f}  "
                  f"accept={result.acceptance_ratio*100:.1f}%")

    result.final_chi2 = current_chi2
    return result
