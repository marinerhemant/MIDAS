"""Supercell datastructure for RMC.

A supercell is:

  * a set of atomic species (``self.species``, length N)
  * Cartesian positions (``self.positions``, shape (N, 3), Å)
  * a 3×3 lattice matrix (``self.cell``, rows are lattice vectors, Å)

Periodic-boundary-condition minimum-image pair distances are exposed via
:meth:`pair_distances`. Everything is torch-tensor-backed so downstream
MC / gradient methods can differentiate through the forward model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Supercell:
    species: List[str]
    positions: torch.Tensor        # (N, 3) Cartesian, Å
    cell: torch.Tensor             # (3, 3) lattice vectors along rows, Å

    def __post_init__(self) -> None:
        if len(self.species) != self.positions.shape[0]:
            raise ValueError(
                f"species length {len(self.species)} != positions rows "
                f"{self.positions.shape[0]}")
        if self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.cell.shape != (3, 3):
            raise ValueError(f"cell must be 3×3, got {tuple(self.cell.shape)}")
        # Ensure torch.float64 everywhere for numerical consistency
        self.positions = self.positions.to(torch.float64)
        self.cell = self.cell.to(torch.float64)

    # ---------------------------------------------------------- properties
    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def volume(self) -> float:
        return float(torch.abs(torch.linalg.det(self.cell)))

    @property
    def number_density(self) -> float:
        """ρ₀ in atoms · Å⁻³."""
        return self.n_atoms / self.volume

    def species_set(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.species)))

    # ------------------------------------------------------- construction
    @classmethod
    def from_crystal(
        cls,
        crystal_tensor,
        size: Tuple[int, int, int] = (1, 1, 1),
    ) -> "Supercell":
        """Expand a MIDAS ``Crystal`` (or its torch form) to an (nx, ny, nz)
        supercell.

        Uses the crystal's fractional atomic positions and lattice, then
        replicates across ``size``.
        """
        nx, ny, nz = (int(s) for s in size)
        if min(nx, ny, nz) < 1:
            raise ValueError("size components must all be >= 1")

        lat = crystal_tensor.lattice_params.detach()          # (6,) a,b,c,α,β,γ
        a, b, c = float(lat[0]), float(lat[1]), float(lat[2])
        alpha, beta, gamma = (np.radians(float(lat[i])) for i in (3, 4, 5))

        # Build the 3×3 lattice matrix (rows are lattice vectors)
        cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
        sin_g = np.sin(gamma)
        av = np.array([a, 0.0, 0.0])
        bv = np.array([b * cos_g, b * sin_g, 0.0])
        cx = c * cos_b
        cy = c * (cos_a - cos_b * cos_g) / sin_g
        cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
        cv = np.array([cx, cy, cz])
        unit_cell = np.stack([av, bv, cv])                    # (3,3), unit cell

        # Grab fractional positions + species from the crystal's unit cell view
        _f, _occ, _B, _U = crystal_tensor.unit_cell_view()
        n_uc = int(_f.shape[0])
        if hasattr(crystal_tensor, "atomic_symbols"):
            elements_uc = list(crystal_tensor.atomic_symbols)
        else:
            # fall back to the source Crystal's atoms if the torch tensor lacks them
            elements_uc = ["X"] * n_uc

        fract_uc = _f.detach().cpu().numpy()                  # (n_uc, 3)

        species: List[str] = []
        cart_positions: List[np.ndarray] = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shift = np.array([ix, iy, iz], dtype=np.float64)
                    for k in range(n_uc):
                        f = fract_uc[k] + shift
                        cart = f @ unit_cell
                        species.append(elements_uc[k])
                        cart_positions.append(cart)

        big_cell = unit_cell * np.array([[nx, ny, nz]]).T     # rows scaled
        return cls(
            species=species,
            positions=torch.tensor(np.stack(cart_positions), dtype=torch.float64),
            cell=torch.tensor(big_cell, dtype=torch.float64),
        )

    # ------------------------------------------------------ pair distances
    def pair_distances(
        self,
        r_max: float,
        *,
        min_image: bool = True,
        return_indices: bool = False,
    ) -> torch.Tensor:
        """All pair distances ≤ ``r_max`` between distinct atoms.

        Uses the **minimum-image convention** (i.e. each pair is represented
        by its shortest image across the periodic boundaries).  For dense
        supercells and r_max less than half the shortest cell dimension this
        recovers the true bulk pair distribution; for r_max larger than that,
        supercells should be enlarged to avoid double-imaging artefacts.

        Returns a 1-D tensor of pair distances (each unordered pair (i, j)
        appears once).  With ``return_indices=True`` also returns
        ``(pair_i, pair_j)`` tensors.
        """
        if r_max <= 0:
            raise ValueError("r_max must be positive")

        pos = self.positions                                   # (N, 3)
        N = pos.shape[0]
        i_idx, j_idx = torch.triu_indices(N, N, offset=1)      # unordered i<j
        # Compute displacement vectors
        d = pos[j_idx] - pos[i_idx]                            # (P, 3)

        if min_image:
            # Minimum image: for each pair, subtract the nearest lattice
            # translation.  frac = d @ cell_inv; nearest integer round in
            # fractional space; then back to Cartesian.
            cell_inv = torch.linalg.inv(self.cell)
            frac = d @ cell_inv                                 # (P, 3)
            frac_wrapped = frac - torch.round(frac)
            d = frac_wrapped @ self.cell

        dist = torch.linalg.norm(d, dim=1)
        mask = dist <= r_max
        dist = dist[mask]
        if return_indices:
            return dist, i_idx[mask], j_idx[mask]
        return dist

    # ------------------------------------------------------ pretty repr
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Supercell(N={self.n_atoms}, "
                f"species={self.species_set()}, "
                f"volume={self.volume:.1f} Å³, "
                f"rho0={self.number_density:.4f} atoms/Å³)")
