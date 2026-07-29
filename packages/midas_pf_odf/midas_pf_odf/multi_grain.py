"""Multi-grain plant + per-grain dispatch.

The Phase-1 inversion is per-grain (assignment given). The
multi-grain layer adds:

  * ``MultiGrainPlant`` — per-voxel ``(R, ε)`` with grain labels and
    per-grain mean orientation/strain stats.
  * ``plant_multi_grain`` — Voronoi tessellation, random per-grain
    orientation, configurable per-grain ε.
  * ``split_into_grains`` — peel a multi-grain plant into independent
    ``SinglePhaseGrainPlant`` objects (per-grain voxel positions /
    orientations / strains).
  * ``simulate_multi_grain`` — calls the single-grain simulator
    independently per grain → ``Dict[grain_id, GrainPatchData]``.
  * ``fit_multi_grain`` — independent ``fit_grain_peakshape`` per grain.

Each grain's spots land at distinct detector locations because
orientations differ; the grains do not interfere in the splatter.
Real-data overlap (when grains' spots share pixels) is a Phase-1B
concern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from midas_diffract.forward import HEDMForwardModel
from midas_stress.orientation import axis_angle_to_orient_mat

from midas_pf_odf.simulate import (
    SinglePhaseGrainPlant,
    GrainPatchData,
    simulate_grain_patches,
)
from midas_pf_odf.inversion import (
    fit_grain_peakshape,
    GrainPeakFitResult,
    IdentifiabilityMode,
)


def _aa_to_R(axis_angle: torch.Tensor) -> torch.Tensor:
    eps = 1e-9
    norm = axis_angle.norm(dim=-1, keepdim=True).clamp_min(eps)
    axis = axis_angle / norm
    angle_deg = norm.squeeze(-1) * (180.0 / math.pi)
    R = axis_angle_to_orient_mat(axis, angle_deg)
    near_zero = (norm.squeeze(-1) < 10.0 * eps).unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return torch.where(near_zero, I.expand_as(R), R)


def _random_rotation(seed: int = 0, dtype: torch.dtype = torch.float64
                     ) -> torch.Tensor:
    """Uniformly-distributed (3, 3) rotation matrix on SO(3)."""
    g = torch.Generator().manual_seed(int(seed))
    q = torch.randn(4, generator=g, dtype=dtype)
    q = q / q.norm()
    w, x, y, z = q
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=dtype)


@dataclass
class MultiGrainPlant:
    """Multi-grain microstructure plant.

    Attributes
    ----------
    voxel_pos : (G_total, 3)
    R_voxel : (G_total, 3, 3)
    eps_voxel : (G_total, 6)
    grain_id : (G_total,) long
    R_avg_per_grain : (n_grains, 3, 3)
    eps_avg_per_grain : (n_grains, 6)
    lattice : (6,) — shared reference
    grid_shape : (G_x, G_y)
    grain_centroids : (n_grains, 2) — Voronoi seeds in (x, y)
    """
    voxel_pos: torch.Tensor
    R_voxel: torch.Tensor
    eps_voxel: torch.Tensor
    grain_id: torch.Tensor
    R_avg_per_grain: torch.Tensor
    eps_avg_per_grain: torch.Tensor
    lattice: torch.Tensor
    grid_shape: Tuple[int, int]
    grain_centroids: torch.Tensor
    metadata: dict = field(default_factory=dict)

    @property
    def n_voxels(self) -> int:
        return int(self.voxel_pos.shape[0])

    @property
    def n_grains(self) -> int:
        return int(self.R_avg_per_grain.shape[0])


def _voronoi_assign(
    grid_shape: Tuple[int, int], voxel_size_um: float, n_grains: int,
    seed: int, dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voronoi assignment of voxels to grains.

    Returns (voxel_pos (G,3), grain_id (G,), centroids (n_grains, 2)).
    """
    Gx, Gy = grid_shape
    G = Gx * Gy
    xs = (torch.arange(Gx, dtype=dtype) - 0.5 * (Gx - 1)) * voxel_size_um
    ys = (torch.arange(Gy, dtype=dtype) - 0.5 * (Gy - 1)) * voxel_size_um
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    pos2d = torch.stack([XX.flatten(), YY.flatten()], dim=-1)        # (G, 2)
    voxel_pos = torch.cat([pos2d, torch.zeros(G, 1, dtype=dtype)], dim=-1)

    g = torch.Generator().manual_seed(int(seed))
    half_x = 0.5 * Gx * voxel_size_um
    half_y = 0.5 * Gy * voxel_size_um
    centroids = torch.empty(n_grains, 2, dtype=dtype)
    centroids[:, 0] = torch.rand(n_grains, generator=g, dtype=dtype) * 2 * half_x - half_x
    centroids[:, 1] = torch.rand(n_grains, generator=g, dtype=dtype) * 2 * half_y - half_y

    d2 = ((pos2d.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(-1)  # (G, n_grains)
    grain_id = d2.argmin(dim=-1).to(torch.long)
    return voxel_pos, grain_id, centroids


def plant_multi_grain(
    grid_shape: Tuple[int, int] = (16, 16),
    n_grains: int = 4,
    *,
    voxel_size_um: float = 2.0,
    lattice: Tuple[float, ...] = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0),
    eps_per_grain_amp: float = 1.0e-3,
    intra_grain_spread_deg: float = 0.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> MultiGrainPlant:
    """Plant a multi-grain microstructure.

    - Voronoi tessellation of an `(Gx, Gy)` grid into ``n_grains``
      regions by random centroids (seeded).
    - Each grain gets a random ``R_avg`` (uniform on SO(3)) and a
      random Voigt strain with magnitude up to ``eps_per_grain_amp``.
    - Optional small intra-grain orientation spread (Gaussian
      axis-angle perturbation, std = ``intra_grain_spread_deg``).
    """
    voxel_pos, grain_id, centroids = _voronoi_assign(
        grid_shape, voxel_size_um, n_grains, seed, dtype,
    )

    R_avg_per_grain = torch.stack(
        [_random_rotation(seed=seed * 131 + g, dtype=dtype)
         for g in range(n_grains)], dim=0,
    )                                                                # (n_grains, 3, 3)

    rng = torch.Generator().manual_seed(int(seed) + 7919)
    eps_avg_per_grain = (torch.rand(n_grains, 6, generator=rng, dtype=dtype)
                         * 2 - 1) * eps_per_grain_amp
    # Symmetrize: Voigt is already 6 free components; nothing to do.

    R_voxel = R_avg_per_grain[grain_id]                              # (G, 3, 3)
    if intra_grain_spread_deg > 0.0:
        spread_rad = math.radians(intra_grain_spread_deg)
        rng2 = torch.Generator().manual_seed(int(seed) + 911)
        aa_perturb = torch.randn(R_voxel.shape[0], 3, generator=rng2,
                                  dtype=dtype) * spread_rad
        R_voxel = R_voxel @ _aa_to_R(aa_perturb)

    eps_voxel = eps_avg_per_grain[grain_id]                          # (G, 6)

    return MultiGrainPlant(
        voxel_pos=voxel_pos,
        R_voxel=R_voxel,
        eps_voxel=eps_voxel,
        grain_id=grain_id,
        R_avg_per_grain=R_avg_per_grain,
        eps_avg_per_grain=eps_avg_per_grain,
        lattice=torch.tensor(lattice, dtype=dtype),
        grid_shape=grid_shape,
        grain_centroids=centroids,
        metadata={
            "voxel_size_um": float(voxel_size_um),
            "n_grains": int(n_grains),
            "eps_per_grain_amp": float(eps_per_grain_amp),
            "intra_grain_spread_deg": float(intra_grain_spread_deg),
            "seed": int(seed),
        },
    )


def split_into_grains(
    plant: MultiGrainPlant,
) -> Dict[int, SinglePhaseGrainPlant]:
    """Build a per-grain :class:`SinglePhaseGrainPlant` for each grain.

    Each sub-plant carries that grain's voxels with their per-voxel
    ``(R, ε)`` and the SHARED reference lattice. Grain-id is recorded
    in ``metadata["grain_id"]``.
    """
    sub_plants: Dict[int, SinglePhaseGrainPlant] = {}
    grain_ids = torch.unique(plant.grain_id).tolist()
    for g in grain_ids:
        mask = plant.grain_id == g
        # 1D plant — grid_shape no longer 2D-meaningful for the subgrain;
        # keep the sub_plant.grid_shape = (n_voxels_in_grain, 1) as a
        # default placeholder. Plotting helpers should special-case.
        sub_n = int(mask.sum().item())
        sub = SinglePhaseGrainPlant(
            voxel_pos=plant.voxel_pos[mask],
            R_voxel=plant.R_voxel[mask],
            eps_voxel=plant.eps_voxel[mask],
            lattice=plant.lattice,
            R_avg=plant.R_avg_per_grain[g],
            grid_shape=(sub_n, 1),
            metadata={
                "grain_id": int(g),
                "parent_grid_shape": plant.grid_shape,
                "voxel_indices_in_parent": mask.nonzero(as_tuple=False).flatten(),
            },
        )
        sub_plants[int(g)] = sub
    return sub_plants


def simulate_multi_grain(
    plant: MultiGrainPlant,
    model: HEDMForwardModel,
    *,
    patch_F: int = 5,
    patch_P: int = 15,
    sigma_yz: float = 1.0,
    sigma_f: float = 0.6,
    gate_tau_um: float = 0.5,
    add_noise_sigma: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[int, GrainPatchData]:
    """Per-grain forward simulation. Each grain's spots are independent
    in detector space (different R_avg → different (y, z, ω))."""
    sub_plants = split_into_grains(plant)
    out: Dict[int, GrainPatchData] = {}
    for g, sp in sub_plants.items():
        out[g] = simulate_grain_patches(
            sp, model,
            patch_F=patch_F, patch_P=patch_P,
            sigma_yz=sigma_yz, sigma_f=sigma_f,
            gate_tau_um=gate_tau_um,
            add_noise_sigma=add_noise_sigma,
            seed=None if seed is None else int(seed) + g * 1009,
        )
    return out


def fit_multi_grain(
    per_grain_data: Dict[int, GrainPatchData],
    sub_plants: Dict[int, SinglePhaseGrainPlant],
    model: HEDMForwardModel,
    *,
    eps_init_per_grain: Optional[Dict[int, torch.Tensor]] = None,
    R_init_per_grain: Optional[Dict[int, torch.Tensor]] = None,
    identifiability: IdentifiabilityMode = IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
    optimizer: Optional[str] = None,
    inner_steps: int = 100,
    lr_aa: float = 1e-4,
    lr_eps: float = 1e-3,
    lr_lat: float = 1e-5,
    gate_tau_um: float = 0.5,
    verbose: bool = False,
) -> Dict[int, GrainPeakFitResult]:
    """Fit each grain independently. The grain assignment is taken as
    given; this driver does not refit assignments.
    """
    results: Dict[int, GrainPeakFitResult] = {}
    for g, sp in sub_plants.items():
        eps_init = (eps_init_per_grain or {}).get(
            g, torch.zeros_like(sp.eps_voxel),
        )
        R_init = (R_init_per_grain or {}).get(g, sp.R_voxel)
        result = fit_grain_peakshape(
            per_grain_data[g], model,
            voxel_pos=sp.voxel_pos,
            R_init=R_init, eps_init=eps_init,
            lattice_init=sp.lattice,
            identifiability=identifiability,
            optimizer=optimizer,
            inner_steps=inner_steps,
            lr_aa=lr_aa, lr_eps=lr_eps, lr_lat=lr_lat,
            gate_tau_um=gate_tau_um,
            verbose=verbose,
        )
        results[g] = result
    return results
