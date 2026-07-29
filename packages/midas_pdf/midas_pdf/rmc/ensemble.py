"""Parallel-chain RMC ensembles.

Day-4 building blocks:

  * :func:`rmc_refine_ensemble` — run ``n_chains`` independent RMC chains
    from different random initialisations, return the final configurations
    + per-chain χ² traces.
  * :func:`ensemble_partial_g_r` — average g_{ij}(r) across the ensemble,
    with per-r uncertainty from chain-to-chain scatter.
  * :func:`ensemble_coordination` — mean Z ± σ across the ensemble.
  * :func:`ensemble_G_r` — average G(r) across the ensemble with σ.

The ensemble represents the RMC posterior over supercell configurations
consistent with the target G(r), given the chosen priors (min-distance,
coordination biases, initial jitter).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .analysis import coordination_number, partial_g_r
from .driver import rmc_refine, RMCResult
from .histogram import supercell_G_r
from .moves import DisplaceMove
from .supercell import Supercell


@dataclass
class RMCEnsembleResult:
    chains: List[RMCResult] = field(default_factory=list)

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    @property
    def acceptance_ratios(self) -> np.ndarray:
        return np.asarray([c.acceptance_ratio for c in self.chains])

    def final_chi2(self) -> np.ndarray:
        return np.asarray([c.final_chi2 for c in self.chains])

    def initial_chi2(self) -> np.ndarray:
        return np.asarray([c.initial_chi2 for c in self.chains])


def rmc_refine_ensemble(
    supercell_template,
    r_grid: torch.Tensor,
    G_obs: torch.Tensor,
    *,
    n_chains: int = 4,
    sigma_G: Optional[torch.Tensor] = None,
    moves: Optional[Sequence] = None,
    n_moves: int = 2_000,
    u_iso: float = 0.005,
    q_broad: float = 0.0,
    r_max_pairs: Optional[float] = None,
    temperature: float = 1.0,
    min_distance_A: Optional[float] = None,
    initial_jitter_A: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
) -> RMCEnsembleResult:
    """Run ``n_chains`` independent RMC chains from jittered copies of
    ``supercell_template``. Each chain gets its own RNG seed derived from
    ``seed``.

    Note: chains run **sequentially** in the base implementation. For
    real parallelism, wrap this in ``concurrent.futures.ProcessPoolExecutor``
    (Python-level GIL means torch operations can benefit from processes
    but the framework doesn't force it — we keep it simple in v1).
    """
    if moves is None:
        moves = [DisplaceMove(sigma_A=0.05)]

    chains: List[RMCResult] = []
    for chain_idx in range(n_chains):
        # Deep-copy the template so each chain has its own state
        chain_supercell = Supercell(
            species=list(supercell_template.species),
            positions=supercell_template.positions.clone(),
            cell=supercell_template.cell.clone(),
        )
        # Distinct jitter per chain
        rng = torch.Generator().manual_seed(int(seed) * 1000 + chain_idx)
        jitter = initial_jitter_A * torch.randn(
            chain_supercell.positions.shape, generator=rng,
            dtype=torch.float64)
        chain_supercell.positions = chain_supercell.positions + jitter
        cell_inv = torch.linalg.inv(chain_supercell.cell)
        frac = chain_supercell.positions @ cell_inv
        chain_supercell.positions = (frac - torch.floor(frac)) @ chain_supercell.cell

        # Fresh move instance per chain so RNG state is isolated
        chain_moves = [copy.copy(m) for m in moves]
        result = rmc_refine(
            chain_supercell, r_grid, G_obs, sigma_G=sigma_G,
            moves=chain_moves, n_moves=n_moves, u_iso=u_iso, q_broad=q_broad,
            r_max_pairs=r_max_pairs, temperature=temperature,
            min_distance_A=min_distance_A,
            seed=int(seed) * 1000 + chain_idx, verbose=False,
        )
        chains.append(result)
        if verbose:
            print(f"  chain {chain_idx}: χ² {result.initial_chi2:.2f} "
                   f"→ {result.final_chi2:.2f}, accept "
                   f"{result.acceptance_ratio*100:.1f}%")

    return RMCEnsembleResult(chains=chains)


# ---------------------------------------------------------------------------
# Ensemble analysis
# ---------------------------------------------------------------------------

def ensemble_partial_g_r(
    ensemble: RMCEnsembleResult,
    r_grid: torch.Tensor,
    *,
    bin_width: Optional[float] = None,
    r_max: Optional[float] = None,
) -> Dict[Tuple[str, str], Dict[str, torch.Tensor]]:
    """Ensemble-average partial g_{ij}(r) with chain-to-chain σ.

    Returns::

        {(species_i, species_j): {"mean": tensor, "std": tensor}}
    """
    if bin_width is None:
        bin_width = float(r_grid[1] - r_grid[0])
    per_chain: Dict[Tuple[str, str], List[torch.Tensor]] = {}
    for chain in ensemble.chains:
        g = partial_g_r(chain.supercell, r_grid,
                         bin_width=bin_width, r_max=r_max)
        for key, val in g.items():
            per_chain.setdefault(key, []).append(val)
    out: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
    for key, vals in per_chain.items():
        stack = torch.stack(vals)          # (n_chains, R)
        out[key] = {"mean": stack.mean(dim=0), "std": stack.std(dim=0)}
    return out


def ensemble_coordination(
    ensemble: RMCEnsembleResult,
    *,
    r_shell: Tuple[float, float],
    species_i: Optional[str] = None,
    species_j: Optional[str] = None,
) -> Dict[str, float]:
    """Ensemble-averaged coordination number ± σ across chains."""
    per_chain = np.asarray([
        coordination_number(c.supercell, r_shell=r_shell,
                              species_i=species_i, species_j=species_j)["Z_mean"]
        for c in ensemble.chains
    ], dtype=np.float64)
    return {
        "Z_mean_ensemble":       float(per_chain.mean()),
        "Z_std_ensemble":        float(per_chain.std()),
        "n_chains":              int(len(per_chain)),
        "Z_per_chain":           per_chain.tolist(),
    }


def ensemble_G_r(
    ensemble: RMCEnsembleResult,
    r_grid: torch.Tensor,
    *,
    u_iso: float = 0.005,
    q_broad: float = 0.0,
    r_max_pairs: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Ensemble-averaged G(r) with chain-to-chain σ (posterior band)."""
    stack = torch.stack([
        supercell_G_r(c.supercell, r_grid, u_iso=u_iso,
                       q_broad=q_broad, r_max=r_max_pairs)
        for c in ensemble.chains
    ])
    return {"mean": stack.mean(dim=0), "std": stack.std(dim=0),
             "samples": stack}


__all__ = [
    "RMCEnsembleResult",
    "rmc_refine_ensemble",
    "ensemble_partial_g_r",
    "ensemble_coordination",
    "ensemble_G_r",
]
