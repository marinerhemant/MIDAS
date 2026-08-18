"""Ring-blend grouping for the peak fitter.

Rings closer in radius than ``min_separation_px`` cannot be fitted
independently: the fit windows overlap and the two centres trade against each
other.  ``fit_doublet_pairs`` co-fits such a pair as a shared 2-peak model,
matching v1 C's ``DoubletSeparation`` behaviour (default 25 px).

Clusters of THREE OR MORE
-------------------------
Adjacent-pair logic silently breaks down once three rings fall inside the
window.  Rings (k, k+1) and (k+1, k+2) are both "doublets", so ring k+1 gets
fitted twice — once in each pair — and both fits enter the residual with
different centres.  This is not hypothetical: merging CeO2 and LaB6 ring tables
for a 1-ID geometry at 71.7 keV produces 18 close pairs, of which **10 are
chains of >= 3 rings inside 25 px**.

The 2-peak co-fitter has an analytic Jacobian written for exactly two centres,
so it cannot represent an n-ary blend.  Rather than mis-fit them, this module
groups rings into CLUSTERS and reports n>=3 clusters separately so the caller
can exclude those rings (see
:func:`midas_calibrate.rings.drop_blended_rings`).  Only genuine 2-clusters are
handed to the co-fitter, and no ring ever appears in more than one pair.
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


@dataclass
class BlendClusters:
    """Result of grouping rings by radial proximity.

    ``pairs`` are the 2-member clusters, safe for the 2-peak co-fitter and
    guaranteed disjoint.  ``n_ary`` are the >=3-member clusters, given as
    lists of ring indices sorted by radius; the co-fitter cannot model them,
    so exclude those rings instead.  ``singletons`` fit independently.
    """
    pairs: List[DoubletGroup]
    n_ary: List[List[int]]
    singletons: List[int]

    @property
    def n_ary_ring_indices(self) -> List[int]:
        """Flat list of every ring inside a >=3-member blend."""
        return [i for c in self.n_ary for i in c]


def cluster_rings(
    ring_R_ideal_px: np.ndarray,
    *,
    min_separation_px: float = 25.0,
) -> BlendClusters:
    """Single-linkage grouping of rings by radial gap.

    Two rings join the same cluster when their radii differ by less than
    ``min_separation_px``; the relation is transitive, so a chain of rings each
    within the cut of the next forms ONE cluster rather than a set of
    overlapping pairs.
    """
    R = np.asarray(ring_R_ideal_px, dtype=np.float64)
    n = R.size
    out = BlendClusters(pairs=[], n_ary=[], singletons=[])
    if n == 0:
        return out
    order = np.argsort(R)
    R_sorted = R[order]

    start = 0
    for k in range(1, n + 1):
        contiguous = k < n and (R_sorted[k] - R_sorted[k - 1]) < min_separation_px
        if contiguous:
            continue
        members = [int(order[t]) for t in range(start, k)]
        if len(members) == 1:
            out.singletons.append(members[0])
        elif len(members) == 2:
            a, b = members
            out.pairs.append(DoubletGroup(
                i=a, j=b, R_i=float(R[a]), R_j=float(R[b]),
                separation_px=float(abs(R[b] - R[a]))))
        else:
            out.n_ary.append(members)
        start = k
    out.pairs.sort(key=lambda g: g.R_i)
    return out


def detect_doublets(
    ring_R_ideal_px: np.ndarray,
    *,
    min_separation_px: float = 25.0,
    include_n_ary: bool = False,
) -> List[DoubletGroup]:
    """Pair adjacent rings whose ideal radii are within ``min_separation_px``.

    Returns disjoint 2-member blends only, sorted by R_i.  Rings that belong to
    a >=3-member chain are NOT returned as pairs (they cannot be co-fitted as a
    doublet); use :func:`cluster_rings` to see them.  Set ``include_n_ary`` to
    additionally emit the consecutive pairs inside n-ary clusters, which
    restores the historical behaviour — including its double-counting of the
    interior rings — and is provided only for comparison.
    """
    clusters = cluster_rings(ring_R_ideal_px,
                             min_separation_px=min_separation_px)
    pairs = list(clusters.pairs)
    if include_n_ary:
        R = np.asarray(ring_R_ideal_px, dtype=np.float64)
        for members in clusters.n_ary:
            ordered = sorted(members, key=lambda i: R[i])
            for a, b in zip(ordered[:-1], ordered[1:]):
                pairs.append(DoubletGroup(
                    i=int(a), j=int(b), R_i=float(R[a]), R_j=float(R[b]),
                    separation_px=float(abs(R[b] - R[a]))))
        pairs.sort(key=lambda g: g.R_i)
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

    Rings inside a >=3-member blend get ``-2``: they are neither singletons
    (fitting them independently is wrong) nor representable as a pair.  Callers
    should drop them; :func:`cluster_rings` gives the full grouping.
    """
    clusters = cluster_rings(ring_R_ideal_px,
                             min_separation_px=min_separation_px)
    n = len(np.asarray(ring_R_ideal_px))
    partner = np.full(n, -1, dtype=np.int64)
    for g in clusters.pairs:
        lo, hi = (g.i, g.j) if g.R_i <= g.R_j else (g.j, g.i)
        partner[lo] = hi
    for i in clusters.n_ary_ring_indices:
        partner[i] = -2
    return partner, clusters.pairs


__all__ = ["DoubletGroup", "BlendClusters", "cluster_rings",
           "detect_doublets", "doublet_index_map"]
