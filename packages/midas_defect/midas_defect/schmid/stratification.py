"""Stratify matched pairs by max(Schmid_matrix, Schmid_twin)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stratify_pairs_by_schmid_max(
    pair_indices: NDArray[np.intp],
    schmid_per_grain: NDArray[np.floating],
    quantiles: tuple[float, ...] = (0.0, 33.0, 67.0, 100.0),
) -> dict:
    """Split matched pairs into Schmid terciles (or any quantile partition).

    Per-pair Schmid is taken as the maximum of the two grains' Schmid factors,
    matching the convention that the higher-Schmid grain dominates the twin
    activation.

    Parameters
    ----------
    pair_indices : (n_pairs, 2)
    schmid_per_grain : (n_grains,)
    quantiles : percentile edges in [0, 100]; defaults to terciles.

    Returns
    -------
    dict with
        per_pair_schmid_max : (n_pairs,)
        tier : (n_pairs,) int in [0, n_tiers)
        tier_edges : (n_tiers + 1,) the Schmid edges used
        tier_pair_indices : list of length n_tiers; each entry is an array of
                            pair indices in that tier
    """
    pairs = np.asarray(pair_indices, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pair_indices must be (n_pairs, 2); got {pairs.shape}")
    schmid = np.asarray(schmid_per_grain, dtype=float)
    pair_schmid = np.maximum(schmid[pairs[:, 0]], schmid[pairs[:, 1]])

    edges = np.percentile(pair_schmid, quantiles)
    # Bump the upper edge by an epsilon so the maximum point falls in the
    # last tier rather than off the end.
    edges_for_digitize = edges.copy()
    edges_for_digitize[-1] = edges_for_digitize[-1] + 1e-12
    tier = np.digitize(pair_schmid, edges_for_digitize[1:-1], right=False)

    n_tiers = len(quantiles) - 1
    tier_pair_indices = [np.where(tier == t)[0] for t in range(n_tiers)]

    return {
        "per_pair_schmid_max": pair_schmid,
        "tier": tier,
        "tier_edges": edges,
        "tier_pair_indices": tier_pair_indices,
    }


__all__ = ["stratify_pairs_by_schmid_max"]
