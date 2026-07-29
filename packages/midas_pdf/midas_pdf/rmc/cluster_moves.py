"""Cluster and rigid-body RMC moves for concerted rearrangements.

Rev 6 shipped single-atom :class:`DisplaceMove` and Rev 7 shipped
grand-canonical insert/remove. Both make single-atom moves. In dense
phases (glasses, close-packed liquids, framework materials) single-atom
moves are inefficient: the target G(r) may require several nearby atoms
to move *together*.

Rev 9 adds two concerted-move classes:

  * :class:`ClusterDisplaceMove` — pick an atom, find all its neighbours
    within a radius, translate the whole cluster by the same Gaussian
    displacement.
  * :class:`RigidRotationMove` — pick a cluster around a random anchor
    atom and rotate it rigidly about a random axis by a small angle.

Both work with the existing :func:`midas_pdf.rmc.moves.metropolis_step`
via ``isinstance(move, ClusterDisplaceMove)`` in the driver hook.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _pbc_min_image_distances(pos: torch.Tensor, anchor: torch.Tensor,
                              cell: torch.Tensor) -> torch.Tensor:
    """Return distances from every atom in ``pos`` (N×3) to ``anchor`` (3,)
    under minimum-image PBC."""
    cell_inv = torch.linalg.inv(cell)
    d = pos - anchor
    frac = d @ cell_inv
    frac_wrapped = frac - torch.round(frac)
    d_min = frac_wrapped @ cell
    return torch.linalg.norm(d_min, dim=1)


@dataclass
class ClusterDisplaceMove:
    """Translate a spatial cluster of atoms by the same Gaussian
    displacement.

    On propose(): pick a random anchor atom, collect all atoms within
    ``radius_A`` of it (minimum-image), draw one Gaussian displacement
    vector, and return the cluster indices + displacement.
    """
    radius_A: float = 3.0
    sigma_A: float = 0.05
    rng: Optional[torch.Generator] = None
    min_cluster_size: int = 2

    def propose(self, supercell):
        """Return (cluster_indices: LongTensor, delta: Tensor(3,))."""
        g = self.rng
        N = supercell.n_atoms
        for _ in range(16):
            i = int(torch.randint(0, N, (1,), generator=g).item())
            anchor = supercell.positions[i]
            dists = _pbc_min_image_distances(
                supercell.positions, anchor, supercell.cell)
            cluster = torch.where(dists <= self.radius_A)[0]
            if cluster.numel() >= self.min_cluster_size:
                delta = torch.randn(3, generator=g, dtype=torch.float64) * self.sigma_A
                return cluster, delta
        # Fallback: single-atom cluster
        return torch.tensor([i], dtype=torch.long), \
               torch.randn(3, generator=g, dtype=torch.float64) * self.sigma_A


@dataclass
class RigidRotationMove:
    """Rotate a spatial cluster rigidly about a random axis through the
    anchor atom, by a Gaussian angle centred at 0."""
    radius_A: float = 3.0
    sigma_rad: float = 0.05
    rng: Optional[torch.Generator] = None
    min_cluster_size: int = 3

    def propose(self, supercell):
        """Return (cluster_indices: LongTensor, R: 3×3 rotation matrix,
        anchor: 3-vector)."""
        g = self.rng
        N = supercell.n_atoms
        for _ in range(16):
            i = int(torch.randint(0, N, (1,), generator=g).item())
            anchor = supercell.positions[i]
            dists = _pbc_min_image_distances(
                supercell.positions, anchor, supercell.cell)
            cluster = torch.where(dists <= self.radius_A)[0]
            if cluster.numel() >= self.min_cluster_size:
                # Random rotation axis (uniform on sphere)
                axis = torch.randn(3, generator=g, dtype=torch.float64)
                axis = axis / torch.linalg.norm(axis).clamp(min=1e-12)
                angle = float(torch.randn((), generator=g,
                                          dtype=torch.float64).item()) * self.sigma_rad
                R = _rodrigues_rotation(axis, angle)
                return cluster, R, anchor
        # Fallback: identity rotation on single atom
        return (torch.tensor([i], dtype=torch.long),
                 torch.eye(3, dtype=torch.float64),
                 supercell.positions[i])


def _rodrigues_rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Rodrigues formula: 3×3 rotation matrix for axis · angle."""
    c = float(np.cos(angle)); s = float(np.sin(angle))
    K = torch.tensor([[0.0, -axis[2], axis[1]],
                       [axis[2], 0.0, -axis[0]],
                       [-axis[1], axis[0], 0.0]], dtype=torch.float64)
    I3 = torch.eye(3, dtype=torch.float64)
    return I3 + s * K + (1.0 - c) * (K @ K)


# ---------------------------------------------------------------------------
# Metropolis wrapper for cluster moves
# ---------------------------------------------------------------------------

def cluster_metropolis_step(
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
    """Propose a cluster move (translation or rotation) and accept/reject."""
    from .moves import chi2_supercell
    from .supercell import Supercell

    if current_chi2 is None:
        current_chi2, _ = chi2_supercell(
            supercell, r_grid, G_obs, sigma_G=sigma_G,
            u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs)

    result = {
        "accepted": False, "chi2_new": current_chi2,
        "chi2_old": current_chi2, "delta_chi2": 0.0,
        "cluster_size": 0, "move_type": type(move).__name__,
        "n_atoms": supercell.n_atoms,
    }

    trial_positions = supercell.positions.clone()

    if isinstance(move, ClusterDisplaceMove):
        cluster, delta = move.propose(supercell)
        trial_positions[cluster] = trial_positions[cluster] + delta
    elif isinstance(move, RigidRotationMove):
        cluster, R, anchor = move.propose(supercell)
        rel = trial_positions[cluster] - anchor            # (M, 3)
        rot_rel = rel @ R.T
        trial_positions[cluster] = anchor + rot_rel
    else:                                                   # pragma: no cover
        raise TypeError(f"unknown cluster move type: {type(move)}")

    # PBC-wrap the moved atoms
    cell_inv = torch.linalg.inv(supercell.cell)
    frac = trial_positions[cluster] @ cell_inv
    frac = frac - torch.floor(frac)
    trial_positions[cluster] = frac @ supercell.cell

    trial_sc = Supercell(species=list(supercell.species),
                          positions=trial_positions, cell=supercell.cell)

    # Hard-sphere veto
    if min_distance_A is not None:
        r_all = trial_sc.pair_distances(r_max=min_distance_A * 1.001)
        if r_all.numel() > 0 and float(r_all.min()) < min_distance_A:
            result["cluster_size"] = int(cluster.numel())
            return result

    chi2_new, _ = chi2_supercell(
        trial_sc, r_grid, G_obs, sigma_G=sigma_G,
        u_iso=u_iso, q_broad=q_broad, r_max_pairs=r_max_pairs)
    delta_chi2 = chi2_new - current_chi2
    p_accept = min(1.0, float(np.exp(-delta_chi2 / (2.0 * temperature))))
    u_rand = torch.rand((), generator=rng, dtype=torch.float64).item()
    if u_rand < p_accept:
        supercell.positions = trial_positions
        result.update(accepted=True, chi2_new=chi2_new,
                       delta_chi2=delta_chi2)
    result["cluster_size"] = int(cluster.numel())
    return result


__all__ = [
    "ClusterDisplaceMove",
    "RigidRotationMove",
    "cluster_metropolis_step",
]
