"""Resampling primitives for bootstrap UQ.

Each sampler returns *indices* (not resampled values) so the same draw can be
applied to multiple parallel arrays of values, weights, masks, etc.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def voxel_resample(n_voxels: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Indices into a voxel-field array, sampled with replacement."""
    return rng.integers(0, n_voxels, size=n_voxels)


def grain_resample(n_grains: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Grain-indices sampled with replacement."""
    return rng.integers(0, n_grains, size=n_grains)


def pair_resample(n_pairs: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """Grain-pair indices sampled with replacement."""
    return rng.integers(0, n_pairs, size=n_pairs)


def reflection_within_grain_resample(
    per_grain_reflection_indices: list[NDArray[np.intp]],
    rng: np.random.Generator,
) -> list[NDArray[np.intp]]:
    """Resample reflections independently within each grain.

    Preserves the per-grain count: grain g with k_g reflections returns k_g
    resampled indices into ``per_grain_reflection_indices[g]``.
    """
    out: list[NDArray[np.intp]] = []
    for refl_idx in per_grain_reflection_indices:
        k = len(refl_idx)
        if k == 0:
            out.append(np.empty(0, dtype=np.intp))
        else:
            out.append(np.asarray(refl_idx)[rng.integers(0, k, size=k)])
    return out


__all__ = [
    "voxel_resample",
    "grain_resample",
    "pair_resample",
    "reflection_within_grain_resample",
]
