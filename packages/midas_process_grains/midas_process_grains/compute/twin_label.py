"""Stage 5: twin + sub-grain post-hoc labeling.

After v4 emits the final grain table (Stages 1+2+3+4), this module
adds two pieces of information per grain:

1. ``twin_partner_id`` + ``twin_type`` — for each grain that has a
   partner in the same dataset whose misorientation matches a known
   twin relation (Σ3, Σ9, Σ27 for cubic; basal/prismatic for hex)
   within ``tol_deg``. Twins are kept as distinct grains in the leaf
   table; a parent ``twin_family_id`` is provided for downstream
   collapse if desired.

2. ``subgrain_partner_id`` — for each grain whose nearest spatial
   neighbour has a misorientation below ``subgrain_max_deg`` AND
   matched-spot Jaccard above ``subgrain_min_jaccard``. Indicates a
   low-angle boundary (sub-grain of a parent).

Both are bucket-prefilter accelerated to handle 50,000+ grain
populations in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass
class TwinLabelResult:
    """Per-grain twin + sub-grain labels."""

    twin_partner_id:    np.ndarray   # (N,) int64 — -1 if no partner
    twin_family_id:     np.ndarray   # (N,) int64 — index into family table
    twin_type:          List[str]    # length N (str, "" if not a twin)
    subgrain_partner_id: np.ndarray  # (N,) int64 — -1 if no partner
    n_twin_pairs:       int
    n_subgrain_pairs:   int


# ---------------------------------------------------------------------------
# Reusable bucket prefilter
# ---------------------------------------------------------------------------


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _bucket_pairs(quats: np.ndarray, cell_size: float) -> np.ndarray:
    """Return candidate (i, j) pairs whose quats land in the same 4D bucket.

    Used for both close-orientation and twin-relation searches: for twin
    search, the caller pre-multiplies one half of the population by the
    twin operator, then asks for same-bucket pairs.
    """
    sgn = np.where(quats[:, 0] >= 0, 1.0, -1.0)
    qc = quats * sgn[:, None]
    cell = np.floor(qc / cell_size).astype(np.int64)
    order = np.lexsort((cell[:, 3], cell[:, 2], cell[:, 1], cell[:, 0]))
    sc = cell[order]
    indices = np.argsort(order).argsort()    # not used; keeping order array
    diff = np.any(np.diff(sc, axis=0) != 0, axis=1)
    breaks = np.concatenate([[0], np.flatnonzero(diff) + 1, [sc.shape[0]]])
    pairs = []
    for k in range(len(breaks) - 1):
        lo, hi = int(breaks[k]), int(breaks[k+1])
        if hi - lo < 2:
            continue
        members = order[lo:hi]
        ii, jj = np.triu_indices(len(members), k=1)
        pairs.append(np.stack([members[ii], members[jj]], axis=1))
    return np.concatenate(pairs, axis=0) if pairs else np.empty((0, 2), dtype=np.int64)


# ---------------------------------------------------------------------------
# Twin labeling
# ---------------------------------------------------------------------------


def label_twins(
    *,
    grain_quats:    np.ndarray,        # (N, 4) per-grain orientation quats (FZ-canonical)
    grain_positions: Optional[np.ndarray] = None,  # (N, 3) µm; optional spatial filter
    twin_relations: Optional[List["TwinRelation"]] = None,
    space_group:    int = 225,
    tol_deg:        float = 0.5,
    spatial_max_um: Optional[float] = None,
    c_over_a:       Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """For every grain, find its closest twin partner (if any).

    Returns (twin_partner_id, twin_family_id, twin_type, n_pairs):
    - ``twin_partner_id[i]`` is the partner index (or -1)
    - ``twin_family_id[i]`` groups twin-related grains into families
      (transitive closure of the twin relation)
    - ``twin_type[i]`` is the human-readable name of the matched relation
    - ``n_pairs`` is the number of twin pairs detected
    """
    from .twins import default_twin_relations_for
    from midas_stress.orientation import misorientation_quat_batch
    import torch

    N = grain_quats.shape[0]
    twin_partner = np.full(N, -1, dtype=np.int64)
    twin_family  = np.full(N, -1, dtype=np.int64)
    twin_type    = [""] * N

    if N < 2:
        return twin_partner, twin_family, twin_type, 0

    if twin_relations is None:
        twin_relations = default_twin_relations_for(
            space_group, c_over_a=c_over_a,
        )

    if not twin_relations:
        # No standard twin operators for this crystal system — return clean
        # (twins will all be unlabelled but the function should still
        # succeed).
        return twin_partner, twin_family, twin_type, 0

    tol_rad = math.radians(tol_deg)

    # Search: for each twin operator T, find pairs (i, j) where
    # rotate(quats[i], T) is bucket-close to quats[j].
    # cell size ~ chord of 2·tol_deg
    bucket_deg = max(2.0, 4.0 * tol_deg)
    cell_size = 2.0 * math.sin(math.radians(bucket_deg) / 2.0)

    found_pairs: List[Tuple[int, int, str, float]] = []

    for tw in twin_relations:
        T = tw.quaternion
        rotated = _qmul(grain_quats, T[None, :])
        # Search rotated vs original via bucket: stack both halves
        stack = np.vstack([grain_quats, rotated])
        labels = np.concatenate([np.zeros(N, dtype=np.int8), np.ones(N, dtype=np.int8)])
        pairs_all = _bucket_pairs(stack, cell_size)
        if len(pairs_all) == 0:
            continue
        # Keep only mixed-population pairs (one from each half)
        mixed = labels[pairs_all[:, 0]] != labels[pairs_all[:, 1]]
        pairs = pairs_all[mixed]
        if len(pairs) == 0:
            continue
        # Map back to original grain indices: pop-0 → i; pop-1 → j
        first = pairs[:, 0]; second = pairs[:, 1]
        gi = np.where(first < N, first, first - N)
        gj = np.where(second < N, second, second - N)
        # Don't self-pair
        keep = gi != gj
        gi = gi[keep]; gj = gj[keep]
        if len(gi) == 0:
            continue
        # Spatial filter
        if grain_positions is not None and spatial_max_um is not None:
            d = np.linalg.norm(
                grain_positions[gi] - grain_positions[gj], axis=1
            )
            ok = d < spatial_max_um
            gi = gi[ok]; gj = gj[ok]
            if len(gi) == 0:
                continue
        # Exact misorientation check
        qa = torch.from_numpy(grain_quats[gi])
        qb = torch.from_numpy(_qmul(grain_quats[gj], T[None, :]))
        miso = misorientation_quat_batch(qa, qb, space_group).numpy()
        ok = miso < tol_rad
        for k in np.flatnonzero(ok):
            found_pairs.append((int(gi[k]), int(gj[k]), tw.name, float(miso[k])))

    # ── Deduplicate: each unordered pair (i,j) is counted ONCE, keeping
    # the lowest-misori operator's name. Without this, HCP symmetry-
    # equivalent operators or cubic Σ3-of-Σ3≡Σ9 ambiguities can
    # double-count a single physical pair.
    best_pair: Dict[Tuple[int, int], Tuple[str, float]] = {}
    for i, j, name, m in found_pairs:
        key = (i, j) if i < j else (j, i)
        cur = best_pair.get(key)
        if cur is None or m < cur[1]:
            best_pair[key] = (name, m)
    dedup_pairs = [(k[0], k[1], v[0], v[1]) for k, v in best_pair.items()]
    n_pairs_unique = len(dedup_pairs)

    # Per-grain: keep the best (lowest-misori) twin partner
    best_by_grain: Dict[int, Tuple[int, str, float]] = {}
    for i, j, name, m in dedup_pairs:
        for a, b in ((i, j), (j, i)):
            cur = best_by_grain.get(a)
            if cur is None or m < cur[2]:
                best_by_grain[a] = (b, name, m)

    for g, (p, name, _) in best_by_grain.items():
        twin_partner[g] = p
        twin_type[g] = name

    # Build twin families via union-find on the deduplicated pairs.
    # This is the transitive closure across ALL twin operators (Σ3, Σ9,
    # Σ27a, Σ27b on cubic; the 5 HCP twin systems on hex). Two grains
    # related by a Σ3-chain (Σ3 then Σ9 cousin) end up in the SAME
    # twin family.
    parent = np.arange(N, dtype=np.int64)
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j, _, _ in dedup_pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj: parent[ri] = rj
    roots = np.array([find(i) for i in range(N)])
    in_family = np.zeros(N, dtype=bool)
    for i, j, _, _ in dedup_pairs:
        in_family[i] = in_family[j] = True
    if in_family.any():
        unique_roots = np.unique(roots[in_family])
        root_to_fam = {int(r): f for f, r in enumerate(unique_roots)}
        for g in np.flatnonzero(in_family):
            twin_family[g] = root_to_fam[int(roots[g])]

    return twin_partner, twin_family, twin_type, n_pairs_unique


# ---------------------------------------------------------------------------
# Sub-grain detection (low-angle boundary)
# ---------------------------------------------------------------------------


def label_subgrains(
    *,
    grain_quats:        np.ndarray,    # (N, 4) FZ-canonical quats
    grain_positions:    np.ndarray,    # (N, 3) µm
    grain_spot_sets:    List[set],     # length N
    space_group:        int = 225,
    subgrain_max_deg:   float = 1.0,
    subgrain_min_jaccard: float = 0.5,
    spatial_max_um:     float = 100.0,
) -> Tuple[np.ndarray, int]:
    """Return ``subgrain_partner_id`` per grain.

    A grain g is labeled a sub-grain partner of g' if:
    - misori(g, g') < subgrain_max_deg
    - position distance(g, g') < spatial_max_um
    - jaccard(spots_g, spots_g') > subgrain_min_jaccard

    The closest such partner (lowest misori) is reported.
    """
    from midas_stress.orientation import misorientation_quat_batch
    import torch

    N = grain_quats.shape[0]
    out = np.full(N, -1, dtype=np.int64)
    if N < 2:
        return out, 0

    bucket_deg = max(2.0, 4.0 * subgrain_max_deg)
    cell_size = 2.0 * math.sin(math.radians(bucket_deg) / 2.0)
    pairs = _bucket_pairs(grain_quats, cell_size)
    if len(pairs) == 0:
        return out, 0
    gi = pairs[:, 0]; gj = pairs[:, 1]

    # Spatial filter
    d = np.linalg.norm(grain_positions[gi] - grain_positions[gj], axis=1)
    ok = d < spatial_max_um
    gi = gi[ok]; gj = gj[ok]
    if len(gi) == 0:
        return out, 0

    # Exact misori
    qa = torch.from_numpy(grain_quats[gi])
    qb = torch.from_numpy(grain_quats[gj])
    miso = misorientation_quat_batch(qa, qb, space_group).numpy()
    ok = miso < math.radians(subgrain_max_deg)
    gi = gi[ok]; gj = gj[ok]; miso = miso[ok]
    if len(gi) == 0:
        return out, 0

    # Jaccard
    n_pairs = 0
    best: Dict[int, Tuple[int, float]] = {}
    for k in range(len(gi)):
        i, j = int(gi[k]), int(gj[k])
        sa = grain_spot_sets[i]; sb = grain_spot_sets[j]
        if not sa or not sb: continue
        jac = len(sa & sb) / len(sa | sb)
        if jac < subgrain_min_jaccard: continue
        n_pairs += 1
        m = float(miso[k])
        for a, b in ((i, j), (j, i)):
            cur = best.get(a)
            if cur is None or m < cur[1]:
                best[a] = (b, m)

    for g, (p, _) in best.items():
        out[g] = p
    return out, n_pairs
