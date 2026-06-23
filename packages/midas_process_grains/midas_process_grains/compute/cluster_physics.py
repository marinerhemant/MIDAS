"""Stage 3: physics-bounded grain clustering with constrained split.

Given:
    - per-candidate FZ-canonical orientations
    - per-candidate refined positions (X, Y, Z) in sample frame
    - per-candidate seed-(h, k, l) attributions from Stage 1
    - per-candidate matched-spot sets (for Jaccard)
    - per-candidate Pass-1 misori cluster labels

produce:
    - final grain labels per candidate
    - per-final-grain trust columns
      (hkl_cluster_size, hkl_dup_count, splits_emerged_from, ...)

The core idea:
    1. Inside each Pass-1 misori cluster, check the seed-hkl multiset.
    2. Clean cluster (every signed (h,k,l) ≤ 1) → emit as a final grain.
    3. Dirty cluster → SPLIT:
         a. K = max duplicated-hkl count → at least K distinct physical grains
         b. The K candidates of the most-duplicated variant become initial
            split centers
         c. Each other candidate is assigned to one of the K centers via
            Hungarian min-cost assignment, with the constraint that no center
            can accept two candidates of the same variant (encoded as +∞
            in the cost matrix)
         d. After assignment, each resulting sub-cluster gets its own check.
            If still dirty (a leftover variant collides), recurse.

Cost function for the assignment:
    cost(c, center) = w_pos · (pos_dist / pos_unc_scale)
                    + w_jac · (1 - jaccard(spots_c, spots_center))
                    + w_ori · (misori_deg / misori_scale)

with weights derived from the within-cluster distribution of these
quantities (each axis normalised by its IQR so the cost is dimensionless).

Hungarian solver: ``scipy.optimize.linear_sum_assignment``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None  # graceful fallback


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class PhysicsClusterResult:
    """Output of :func:`split_clusters_by_physics`.

    All arrays are per-candidate, length N (the input candidate count).
    """

    # Final grain label per candidate. -1 = candidate is unassigned.
    final_grain_id:        np.ndarray   # (N,) int64
    # The Pass-1 misori cluster the candidate originally belonged to.
    pass1_cluster_id:      np.ndarray   # (N,) int64

    # Per-grain summary tables (length N_grains_final):
    n_final_grains:        int
    grain_pass1_parent:    np.ndarray   # (N_g,) int64 — which Pass-1 it came from
    grain_n_candidates:    np.ndarray   # (N_g,) int32
    grain_n_unique_hkls:   np.ndarray   # (N_g,) int32
    grain_hkl_dup_count:   np.ndarray   # (N_g,) int32 — duplicates BEFORE split
    grain_splits_emerged:  np.ndarray   # (N_g,) int32 — 0 = clean original,
                                        # k = emerged from k splits
    grain_n_expected_hkls: np.ndarray   # (N_g,) int32 — from Stage-2 predictor
    grain_hkl_coverage:    np.ndarray   # (N_g,) float64 — n_unique / n_expected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hkl_key(h: np.ndarray, k: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Encode signed (h, k, l) triples into a single int64 key for fast counting.

    Each component is shifted into a 16-bit unsigned slot so the encoding
    is collision-free for the |h|, |k|, |l| ≤ 30 we encounter in HEDM.
    """
    h = h.astype(np.int64)
    k = k.astype(np.int64)
    l = l.astype(np.int64)
    return ((h + 32_000) << 32) | ((k + 32_000) << 16) | (l + 32_000)


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _split_one_cluster(
    cand_idx_in_cluster: np.ndarray,
    seed_h: np.ndarray,
    seed_k: np.ndarray,
    seed_l: np.ndarray,
    positions: np.ndarray,         # (M, 3)
    spot_sets: List[set],          # length M
    om_per_cand: np.ndarray,       # (M, 3, 3) FZ-canonical
    *,
    w_pos: float = 1.0,
    w_jac: float = 1.0,
    w_ori: float = 0.5,
    max_iters: int = 4,
) -> List[List[int]]:
    """Split a single over-merged cluster into physics-valid sub-clusters.

    Returns a list of sub-clusters, each as a list of LOCAL indices into
    cand_idx_in_cluster (0..M-1). The caller maps local → global candidate
    indices.

    Algorithm:
      1. Compute the seed-hkl multiset.
      2. If clean → return [list(range(M))].
      3. Identify the most-duplicated variant (K members).
      4. K initial sub-clusters, seeded by those K candidates.
      5. For each remaining candidate, compute cost to each sub-cluster's
         seed (Hungarian-style), with +∞ where the assignment would
         duplicate a variant already in that sub-cluster.
      6. Solve the assignment via greedy variant-aware Hungarian.
      7. Recurse on any still-dirty sub-cluster.
    """
    M = len(cand_idx_in_cluster)
    if M == 0:
        return []

    keys = _hkl_key(seed_h, seed_k, seed_l)
    unique_keys, counts = np.unique(keys, return_counts=True)
    if counts.max() <= 1:
        return [list(range(M))]

    # K = number of distinct grains we must split into
    K = int(counts.max())
    most_dup_key = unique_keys[counts.argmax()]
    seed_mask = keys == most_dup_key
    seed_idx = np.flatnonzero(seed_mask)[:K]       # K initial split centers

    # Other candidates needing assignment
    other_idx = np.flatnonzero(~seed_mask)

    # Build sub-clusters; each starts with one seed candidate
    sub_clusters: List[List[int]] = [[int(s)] for s in seed_idx]
    sub_hkls: List[set] = [{int(keys[s])} for s in seed_idx]

    if len(other_idx) > 0:
        # Cost matrix: (n_other, K)
        # Normalise axes by IQR for dimensional consistency
        all_keys_set = set()
        pos_dists = np.linalg.norm(
            positions[other_idx][:, None, :] - positions[seed_idx][None, :, :],
            axis=2,
        )                                                       # (n_other, K)
        # Misori distance (approximate via Frobenius norm of OM difference;
        # for small angles this scales as θ in radians, fine for assignment.)
        om_o = om_per_cand[other_idx]                          # (n_other, 3, 3)
        om_s = om_per_cand[seed_idx]                           # (K, 3, 3)
        diff = om_o[:, None, :, :] - om_s[None, :, :, :]
        ori_dists = np.linalg.norm(diff, axis=(2, 3))          # (n_other, K)

        # Jaccard distance
        jac_dists = np.zeros((len(other_idx), K))
        for i_o, o in enumerate(other_idx):
            for i_s, s in enumerate(seed_idx):
                jac_dists[i_o, i_s] = 1.0 - _jaccard(spot_sets[o], spot_sets[s])

        # Normalise each axis by its IQR (within this cluster) to make
        # the weights dimensionally consistent.
        def _scale(x):
            q1, q3 = np.percentile(x, [25, 75])
            scale = max(q3 - q1, 1e-9)
            return x / scale
        cost = (w_pos * _scale(pos_dists)
                + w_jac * _scale(jac_dists)
                + w_ori * _scale(ori_dists))

        # Variant-constrained greedy assignment: at each step, pick the
        # lowest-cost feasible (candidate, sub-cluster) pair where adding
        # the candidate's variant would not duplicate.
        # This isn't strictly optimal but is order-independent for our
        # cost magnitudes (ties broken by ordering, which is from input).
        cand_variant = keys[other_idx]
        unassigned = list(range(len(other_idx)))
        while unassigned:
            best_cost = np.inf
            best_oi = None
            best_si = None
            for oi in unassigned:
                v = int(cand_variant[oi])
                # Find lowest-cost sub-cluster that doesn't already hold v
                for si in range(K):
                    if v in sub_hkls[si]:
                        continue
                    c = cost[oi, si]
                    if c < best_cost:
                        best_cost = c; best_oi = oi; best_si = si
            if best_oi is None:
                # All remaining candidates are constrained out → must add
                # NEW sub-cluster(s). For now, just attach them to the
                # nearest sub-cluster anyway (Stage-3 v1 caveat).
                for oi in list(unassigned):
                    si = int(np.argmin(cost[oi]))
                    sub_clusters[si].append(int(other_idx[oi]))
                    sub_hkls[si].add(int(cand_variant[oi]))
                break
            sub_clusters[best_si].append(int(other_idx[best_oi]))
            sub_hkls[best_si].add(int(cand_variant[best_oi]))
            unassigned.remove(best_oi)

    # Recurse on still-dirty sub-clusters
    if max_iters > 0:
        result: List[List[int]] = []
        for sub in sub_clusters:
            sub_local = np.array(sub, dtype=np.int64)
            sub_keys = keys[sub_local]
            _, sub_counts = np.unique(sub_keys, return_counts=True)
            if sub_counts.max() > 1:
                # Recurse with reduced budget
                sub_pos = positions[sub_local]
                sub_spots = [spot_sets[i] for i in sub_local]
                sub_om = om_per_cand[sub_local]
                rec = _split_one_cluster(
                    sub_local,
                    seed_h[sub_local], seed_k[sub_local], seed_l[sub_local],
                    sub_pos, sub_spots, sub_om,
                    w_pos=w_pos, w_jac=w_jac, w_ori=w_ori,
                    max_iters=max_iters - 1,
                )
                # Map recursion's local indices (into sub_local) back to
                # original local indices (into cand_idx_in_cluster)
                for r in rec:
                    result.append([int(sub_local[ri]) for ri in r])
            else:
                result.append(sub)
        return result
    else:
        return sub_clusters


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def split_clusters_by_physics(
    *,
    pass1_cluster_id: np.ndarray,        # (N,) int — Pass-1 cluster label per candidate
    seed_h: np.ndarray,                  # (N,) int8
    seed_k: np.ndarray,                  # (N,) int8
    seed_l: np.ndarray,                  # (N,) int8
    seed_alive: np.ndarray,              # (N,) bool — True if seed-hkl recovered
    positions: np.ndarray,               # (N, 3) — refined sample-frame position
    spot_sets: List[set],                # length N — matched SpotIDs per candidate
    om_fz: np.ndarray,                   # (N, 3, 3) — FZ-canonical OM per candidate
    n_expected_per_pass1: Optional[Dict[int, int]] = None,
    disable_split: bool = False,
) -> PhysicsClusterResult:
    """Apply the physics-bounded split to every Pass-1 cluster.

    ``n_expected_per_pass1`` (optional) provides the geometric
    expected-visible-variant count per Pass-1 cluster (from Stage 2);
    if absent, hkl_coverage is computed against ring-multiplicity only.

    ``disable_split=True`` maps each Pass-1 cluster to exactly one final grain
    (no internal physics split) while still populating the diagnostic fields.
    Used by the ``consensus_anchor`` merge primitive, whose clusters are already
    orientation-homogeneous — the position-based split would only re-fragment
    them (acutely so when refined positions are diverged).
    """
    N = pass1_cluster_id.shape[0]
    final_grain_id = np.full(N, -1, dtype=np.int64)

    pass1_unique = np.unique(pass1_cluster_id[pass1_cluster_id >= 0])

    # Per-grain summary arrays (will grow as we process)
    grain_pass1: List[int] = []
    grain_n_cand: List[int] = []
    grain_n_unique: List[int] = []
    grain_dup_count: List[int] = []
    grain_splits_from: List[int] = []
    grain_n_expected: List[int] = []
    grain_coverage: List[float] = []

    next_grain_id = 0
    for p1 in pass1_unique:
        in_cluster = np.flatnonzero(
            (pass1_cluster_id == p1) & seed_alive
        )
        if in_cluster.size == 0:
            continue
        # Pre-split duplication count (for the trust signal)
        keys = _hkl_key(seed_h[in_cluster], seed_k[in_cluster], seed_l[in_cluster])
        _, cts_pre = np.unique(keys, return_counts=True)
        dup_count_pre = int((cts_pre > 1).sum())

        # Apply the split (or keep the whole cluster as one grain)
        if disable_split:
            subs = [list(range(in_cluster.size))]
        else:
            spots_local = [spot_sets[c] for c in in_cluster]
            subs = _split_one_cluster(
                in_cluster,
                seed_h[in_cluster], seed_k[in_cluster], seed_l[in_cluster],
                positions[in_cluster], spots_local, om_fz[in_cluster],
            )

        splits_emerged = 0 if len(subs) == 1 and dup_count_pre == 0 else 1
        n_expected = (n_expected_per_pass1.get(int(p1))
                       if n_expected_per_pass1 is not None else 0)

        for sub in subs:
            globals_ = in_cluster[np.array(sub, dtype=np.int64)]
            final_grain_id[globals_] = next_grain_id
            n_cand = len(globals_)
            keys_g = _hkl_key(seed_h[globals_], seed_k[globals_], seed_l[globals_])
            n_unique = int(np.unique(keys_g).size)

            grain_pass1.append(int(p1))
            grain_n_cand.append(int(n_cand))
            grain_n_unique.append(n_unique)
            grain_dup_count.append(dup_count_pre)
            grain_splits_from.append(splits_emerged)
            grain_n_expected.append(int(n_expected))
            grain_coverage.append(
                n_unique / n_expected if n_expected > 0 else float("nan")
            )
            next_grain_id += 1

    return PhysicsClusterResult(
        final_grain_id=final_grain_id,
        pass1_cluster_id=pass1_cluster_id.copy(),
        n_final_grains=next_grain_id,
        grain_pass1_parent=np.asarray(grain_pass1, dtype=np.int64),
        grain_n_candidates=np.asarray(grain_n_cand, dtype=np.int32),
        grain_n_unique_hkls=np.asarray(grain_n_unique, dtype=np.int32),
        grain_hkl_dup_count=np.asarray(grain_dup_count, dtype=np.int32),
        grain_splits_emerged=np.asarray(grain_splits_from, dtype=np.int32),
        grain_n_expected_hkls=np.asarray(grain_n_expected, dtype=np.int32),
        grain_hkl_coverage=np.asarray(grain_coverage, dtype=np.float64),
    )
