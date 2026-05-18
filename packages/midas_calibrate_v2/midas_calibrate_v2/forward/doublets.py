"""Doublet ring detection — pair rings whose ideal radii are within
``min_separation_px`` so the per-region peak fitter co-fits them as a
shared 2-peak model instead of an independent 1-peak per ring.

This matches v1 C's ``DoubletSeparation`` behaviour (default 25 px) and
prevents label-swap degeneracies when adjacent rings overlap inside the
fit window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DoubletGroup:
    """One ring pair flagged as a doublet."""
    i: int
    j: int
    R_i: float
    R_j: float
    separation_px: float


def detect_doublets(
    ring_R_ideal_px: np.ndarray,
    *,
    min_separation_px: float = 25.0,
) -> List[DoubletGroup]:
    """Pair adjacent rings whose ideal radii are within ``min_separation_px``.

    Returns
    -------
    list of :class:`DoubletGroup` for each adjacent pair below the
    separation threshold.  Sorted by R_i.
    """
    R = np.asarray(ring_R_ideal_px, dtype=np.float64)
    order = np.argsort(R)
    R_sorted = R[order]
    pairs: List[DoubletGroup] = []
    for k in range(len(R_sorted) - 1):
        sep = R_sorted[k + 1] - R_sorted[k]
        if sep < min_separation_px:
            pairs.append(DoubletGroup(
                i=int(order[k]), j=int(order[k + 1]),
                R_i=float(R_sorted[k]), R_j=float(R_sorted[k + 1]),
                separation_px=float(sep),
            ))
    return pairs


def doublet_index_map(
    ring_R_ideal_px: np.ndarray,
    *,
    min_separation_px: float = 25.0,
) -> Tuple[np.ndarray, List[DoubletGroup]]:
    """Return a per-ring "doublet partner" index array.

    ``partner[i] == j`` means ring i is part of a doublet with ring j;
    ``-1`` means ring i is a singleton.  When two rings form a doublet,
    only the lower-radius ring stores the partner; the higher-radius
    ring is left as ``-1`` so the caller fits the doublet once per pair.
    """
    pairs = detect_doublets(ring_R_ideal_px,
                             min_separation_px=min_separation_px)
    n = len(ring_R_ideal_px)
    partner = np.full(n, -1, dtype=np.int64)
    for g in pairs:
        partner[g.i] = g.j
    return partner, pairs


__all__ = ["DoubletGroup", "detect_doublets", "doublet_index_map"]
