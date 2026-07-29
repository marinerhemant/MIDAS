"""Sample generation: grain populations and high-Z fiducial markers.

Everything is seeded and deterministic (no wall-clock / global RNG) so a sweep
row is reproducible and two configs can be compared on the *same* sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import XAFConfig


@dataclass
class GrainPopulation:
    """A set of grains and (optionally) fiducial markers.

    Angles in radians (Bunge ZXZ Euler, matching the forward model), positions
    in micrometres, strains in crystal-frame plain-Voigt
    ``[e11, e12, e13, e22, e23, e33]``.
    """
    euler: torch.Tensor        # (N, 3) rad
    position: torch.Tensor     # (N, 3) um
    strain: torch.Tensor       # (N, 6) Voigt
    fiducial_euler: Optional[torch.Tensor] = None      # (F, 3) rad
    fiducial_position: Optional[torch.Tensor] = None    # (F, 3) um

    @property
    def n_grains(self) -> int:
        return self.euler.shape[0]

    @property
    def n_fiducials(self) -> int:
        return 0 if self.fiducial_euler is None else self.fiducial_euler.shape[0]


def _random_euler(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform random orientations on SO(3) as Bunge ZXZ Euler angles (rad)."""
    phi1 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi2 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    # Uniform in cos(Phi) gives uniform sampling of the sphere.
    Phi = np.arccos(rng.uniform(-1.0, 1.0, size=n))
    return np.stack([phi1, Phi, phi2], axis=1)


def _random_positions_in_sphere(
    rng: np.random.Generator, n: int, radius: float
) -> np.ndarray:
    """Uniformly fill a sphere of ``radius`` (um)."""
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    r = radius * np.cbrt(rng.uniform(0.0, 1.0, size=(n, 1)))
    return pts * r


def make_sample(cfg: XAFConfig) -> GrainPopulation:
    """Build a seeded grain population + fiducials for a config."""
    rng = np.random.default_rng(cfg.seed)
    dtype = torch.float64 if cfg.dtype_double else torch.float32

    euler = _random_euler(rng, cfg.n_grains)
    pos = _random_positions_in_sphere(rng, cfg.n_grains, cfg.sample_radius_um)
    strain = rng.normal(scale=cfg.strain_rms, size=(cfg.n_grains, 6))

    fid_euler = fid_pos = None
    if cfg.n_fiducials > 0:
        # Separate RNG stream so adding/removing fiducials never perturbs the
        # grain population (keeps sweeps comparable).
        frng = np.random.default_rng(cfg.seed + 10_007)
        fid_euler = _random_euler(frng, cfg.n_fiducials)
        # Fiducials are placed a little inside the sample volume.
        fid_pos = _random_positions_in_sphere(
            frng, cfg.n_fiducials, 0.8 * cfg.sample_radius_um)

    def T(a):
        return None if a is None else torch.as_tensor(a, dtype=dtype)

    return GrainPopulation(
        euler=T(euler), position=T(pos), strain=T(strain),
        fiducial_euler=T(fid_euler), fiducial_position=T(fid_pos),
    )
