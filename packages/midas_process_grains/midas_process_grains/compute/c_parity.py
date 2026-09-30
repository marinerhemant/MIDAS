"""C ProcessGrains exact replica.

Goal: bit-level parity with ``FF_HEDM/src/ProcessGrains.c`` for everything
except the Kenesei strain tensor (which uses NLOPT in C and SciPy in Python
and is allowed to differ within ~few hundred microstrain).

This module is the canonical implementation. Earlier modules
(``cluster.py`` 4-D quat bucketing, ``refine_cluster.py`` Phase-2 split,
``pass_a.py`` spot-overlap merge) are now opt-in experimental modes; the
default ``legacy`` mode of the pipeline calls into here.

Algorithm summary (mirroring ProcessGrains.c)
---------------------------------------------
1. **Stage 1 — FindInternalAngles** (ProcessGrains.c:154-207).
   For each unclaimed alive seed `Pos`:
     a. Loop k=0..NrIDsPerID[Pos]-1, look up candidate position
        ``j = pos_by_id[IDsPerGrain[Pos, k]]``.
     b. Skip if `j < 0` or already claimed.
     c. Compute symmetry-aware misorientation between ``Pos`` and ``j``.
     d. If `< 0.4°` and atomic-claim succeeds, recurse into `j`.
   Members are visited in pre-order DFS; min-IA member is the cluster rep.
   We pre-compute every ``(Pos, j)`` misorientation in one batched torch call
   before DFS, then walk the resulting filtered adjacency in pure Python —
   functionally identical to C's recursive version, an order of magnitude
   faster than per-edge misori calls.

2. **Pass A — position+orientation dedup** (ProcessGrains.c:836-874).
   All-pairs over Stage-1 cluster reps; mark `j > i` as duplicate iff
   `misori(i, j) < 0.1°` AND `|Δposition(i, j)| < 5 µm`. C runs this at
   O(N²) over OpenMP; we use a 5 µm spatial hash to limit pairs to those
   spatially close, dropping the work to O(N) at peakfit scales.

3. **Confidence filter**: ``OPs[i][22] >= 0.05`` (ProcessGrains.c:893).

4. **Per-grain emit**: 47-column Grains.csv, 12-column SpotMatrix.csv,
   GrainIDsKey.csv — formatted line-for-line as C does it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from midas_stress.orientation import (
    fundamental_zone,
    misorientation_quat_batch,
    orient_mat_to_quat,
)


# --------------------------------------------------------------------------
# OPF / OPs column conventions (C ProcessGrains.c, see comment at L43)
# --------------------------------------------------------------------------
# C builds OPs[i][0..22] from OPF[i][0..26] by dropping cols 0, 10, 14, 21.
# Below we always work in OPF coordinates (0..26) for clarity, never OPs.
#
#  OPF col  meaning              OPs idx (for cross-reference with C)
#   0       SpotID/GrainID       (dropped)
#   1..9    OM (3×3 row-major)   0..8
#   10      pad                  (dropped)
#   11..13  X, Y, Z              9, 10, 11
#   14      pad                  (dropped)
#   15..20  a, b, c, α, β, γ     12..17
#   21      pad                  (dropped)
#   22      DiffPos              18
#   23      DiffOme              19
#   24      DiffAngle (= IA)     20  ← IAColNr in C
#   25      meanRadius           21
#   26      completeness         22
OPF_OM = slice(1, 10)
OPF_POS = slice(11, 14)
OPF_LATTICE = slice(15, 21)
OPF_DIFF_POS = 22
OPF_DIFF_OME = 23
OPF_IA = 24                # ← matches C's IAColNr=20 in OPs space
OPF_RADIUS = 25
OPF_CONFIDENCE = 26


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass
class Stage1Cluster:
    """One cluster from FindInternalAngles, in DFS visit order."""
    rep_pos: int                       # min-IA member's position
    rep_id: int                        # SpotID at rep_pos
    member_positions: np.ndarray       # int64, in DFS order; rep is somewhere inside
    member_ids: np.ndarray             # int64, same order


@dataclass
class Stage1Result:
    clusters: List[Stage1Cluster]
    grain_positions: np.ndarray        # rep_pos for each cluster, in cluster order
    cluster_label_per_pos: np.ndarray  # (n_seeds,) int64; -1 for non-alive


# --------------------------------------------------------------------------
# Stage 1: FindInternalAngles equivalent
# --------------------------------------------------------------------------


def _build_pos_by_id(ids: np.ndarray) -> Tuple[np.ndarray, int]:
    """Build the ID→position lookup. ``ids[pos]`` is the SpotID at pos.
    Returns (pos_by_id, max_id). Mirrors C ProcessGrains.c:472-498.
    First-wins on duplicate IDs.
    """
    if ids.size == 0:
        return np.empty(0, dtype=np.int64), 0
    max_id = int(ids.max())
    pos_by_id = np.full(max_id + 1, -1, dtype=np.int64)
    valid = ids >= 0
    valid_ids = ids[valid]
    valid_pos = np.flatnonzero(valid)
    # First-wins: assign in ascending pos order, but np indexing overwrites,
    # so reverse to make the LOWEST pos the one that wins.
    pos_by_id[valid_ids[::-1]] = valid_pos[::-1]
    return pos_by_id, max_id


def _build_misori_filtered_edges(
    *,
    opf: np.ndarray,                   # (n_seeds, 27) float64
    process_key: np.ndarray,           # (n_seeds, 5000) int32 — IDsPerGrain
    nr_ids_per_id: np.ndarray,         # (n_seeds,) int32
    pos_by_id: np.ndarray,             # (max_id+1,) int64
    space_group: int,
    misori_tol_rad: float,
    device: str = "cpu",
) -> Dict[int, np.ndarray]:
    """Return adjacency dict: for each Pos, the array of j's (in ProcessKey k-order)
    where misori(Pos, j) < tol. Misori computed in one batched torch call.
    """
    n_seeds = opf.shape[0]

    # Flatten the candidate (Pos, k, j_id) edges into source/dest pos arrays.
    # We keep (src, dst, k_in_pos) so we can later re-sort by k_in_pos for
    # DFS-order parity with C.
    src_list = []
    dst_list = []
    src_k_list = []
    for pos in range(n_seeds):
        n = int(nr_ids_per_id[pos])
        if n <= 0:
            continue
        cand_ids = process_key[pos, :n]
        # Map to positions
        valid_mask = (cand_ids >= 0) & (cand_ids <= pos_by_id.size - 1)
        if not valid_mask.any():
            continue
        cand_pos = np.full(n, -1, dtype=np.int64)
        valid_idx = np.flatnonzero(valid_mask)
        cand_pos[valid_idx] = pos_by_id[cand_ids[valid_idx]]
        ok = cand_pos >= 0
        if not ok.any():
            continue
        ok_idx = np.flatnonzero(ok)
        src_list.append(np.full(ok_idx.size, pos, dtype=np.int64))
        dst_list.append(cand_pos[ok_idx])
        src_k_list.append(ok_idx.astype(np.int64))

    if not src_list:
        return {}

    src_arr = np.concatenate(src_list)
    dst_arr = np.concatenate(dst_list)
    k_arr = np.concatenate(src_k_list)
    n_edges = src_arr.size
    print(f"[c-parity] Stage1: {n_edges:,} candidate (pos, j) edges from ProcessKey",
          flush=True)

    # OM → quat → FZ for all alive seeds (on requested device).
    om = np.ascontiguousarray(opf[:, OPF_OM], dtype=np.float64)
    om_t = torch.from_numpy(om).to(device)
    q = orient_mat_to_quat(om_t)
    q_fz_t = fundamental_zone(q, space_group)               # stays on device

    # Batched misorientation for all edges (on device).
    src_t = torch.from_numpy(src_arr).to(device)
    dst_t = torch.from_numpy(dst_arr).to(device)
    qa = q_fz_t.index_select(0, src_t)
    qb = q_fz_t.index_select(0, dst_t)
    misori_t = misorientation_quat_batch(qa, qb, space_group)
    misori = misori_t.detach().cpu().numpy().astype(np.float64)

    keep = misori < misori_tol_rad
    print(f"[c-parity] Stage1: {keep.sum():,} edges pass misori < "
          f"{math.degrees(misori_tol_rad):.2f}°", flush=True)

    src_arr = src_arr[keep]
    dst_arr = dst_arr[keep]
    k_arr = k_arr[keep]

    # Build adjacency: for each Pos, the list of (k, j) pairs sorted by k.
    # We need k-order so DFS matches C's recursive visit order.
    order = np.lexsort((k_arr, src_arr))
    src_arr = src_arr[order]
    dst_arr = dst_arr[order]

    adj: Dict[int, np.ndarray] = {}
    if src_arr.size > 0:
        # Group by src.
        diff = np.diff(src_arr) != 0
        breaks = np.concatenate(
            [[0], np.flatnonzero(diff) + 1, [src_arr.size]]
        )
        for i in range(breaks.size - 1):
            lo, hi = int(breaks[i]), int(breaks[i + 1])
            adj[int(src_arr[lo])] = dst_arr[lo:hi]
    return adj


def stage1_find_internal_angles(
    *,
    opf: np.ndarray,                   # (n_seeds, 27) float64
    ids: np.ndarray,                   # (n_seeds,) int64 — IDs[pos]
    keep_flag: np.ndarray,             # (n_seeds,) bool — IDsToKeep
    nr_ids_per_id: np.ndarray,         # (n_seeds,) int32
    process_key: np.ndarray,           # (n_seeds, 5000) int32
    space_group: int,
    misori_tol_rad: float = math.radians(0.4),
    min_nr_spots: int = 1,
    device: str = "cpu",
) -> Stage1Result:
    """Cluster seeds via the C-parity recursive DFS.

    The misori-filtered adjacency graph is precomputed in one batched torch
    call; DFS itself is a pure-Python loop using an explicit stack that
    mirrors C's recursion order (push k-children onto the stack in reverse
    so pop yields k=0 first).

    Clusters with fewer than ``min_nr_spots`` members are *dropped* from
    the output, mirroring C ProcessGrains.c:688
    (``if (counten_l < MinNrSpots) continue;``). Their seed positions still
    have ``IDsChecked = True`` (so they don't become roots later) but they
    receive ``cluster_label_per_pos = -1``.
    """
    n_seeds = opf.shape[0]
    print(f"[c-parity] Stage1: {n_seeds:,} seeds, "
          f"{int(keep_flag.sum()):,} alive, "
          f"misori_tol = {math.degrees(misori_tol_rad):.3f}°", flush=True)

    pos_by_id, max_id = _build_pos_by_id(ids)
    adj = _build_misori_filtered_edges(
        opf=opf,
        process_key=process_key,
        nr_ids_per_id=nr_ids_per_id,
        pos_by_id=pos_by_id,
        space_group=space_group,
        misori_tol_rad=misori_tol_rad,
        device=device,
    )

    ia = opf[:, OPF_IA]

    # DFS — visit order matches C's recursive FindInternalAngles.
    ids_checked = np.zeros(n_seeds, dtype=bool)
    ids_checked[~keep_flag] = True   # mark non-alive as checked (skip)

    cluster_label_per_pos = np.full(n_seeds, -1, dtype=np.int64)
    clusters: List[Stage1Cluster] = []
    grain_positions_list: List[int] = []

    t0 = time.time()
    next_progress = 0

    for ii in range(n_seeds):
        if ids_checked[ii]:
            continue
        # Claim ii and start a new cluster.
        ids_checked[ii] = True
        members = [ii]
        stack = [ii]
        while stack:
            pos = stack.pop()
            cands = adj.get(pos)
            if cands is None or cands.size == 0:
                continue
            # Push children in REVERSE so pop yields k=0 first → pre-order DFS.
            for j in reversed(cands.tolist()):
                if not ids_checked[j]:
                    ids_checked[j] = True
                    members.append(j)
                    stack.append(j)

        # MinNrSpots filter (ProcessGrains.c:688). Drop clusters too small
        # to be physical grains; their seeds stay claimed (ids_checked=True)
        # so they don't become future roots, but they receive label=-1.
        if len(members) < min_nr_spots:
            continue

        # Pick min-IA rep (matches C's loop at ProcessGrains.c:691-697).
        members_arr = np.asarray(members, dtype=np.int64)
        rep_local = int(np.argmin(ia[members_arr]))
        rep_pos = int(members_arr[rep_local])

        cluster_id = len(clusters)
        cluster_label_per_pos[members_arr] = cluster_id
        grain_positions_list.append(rep_pos)
        clusters.append(Stage1Cluster(
            rep_pos=rep_pos,
            rep_id=int(ids[rep_pos]),
            member_positions=members_arr,
            member_ids=ids[members_arr].astype(np.int64),
        ))

        if cluster_id >= next_progress:
            elapsed = time.time() - t0
            rate = cluster_id / elapsed if elapsed > 0 else 0.0
            print(f"[c-parity] Stage1: {cluster_id:,} clusters built  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.0f}/s", flush=True)
            next_progress += max(1, n_seeds // 100)

    elapsed = time.time() - t0
    print(f"[c-parity] Stage1 done: {len(clusters):,} clusters in {elapsed:.1f}s",
          flush=True)
    return Stage1Result(
        clusters=clusters,
        grain_positions=np.asarray(grain_positions_list, dtype=np.int64),
        cluster_label_per_pos=cluster_label_per_pos,
    )


# --------------------------------------------------------------------------
# Pass A: position + misorientation dedup
# --------------------------------------------------------------------------


def pass_a_position_dedup(
    *,
    grain_positions: np.ndarray,       # (n_kept_stage1,) int64 — rep pos per cluster
    opf: np.ndarray,                   # (n_seeds, 27) float64
    space_group: int,
    misori_tol_rad: float = math.radians(0.1),
    pos_tol_um: float = 5.0,
    device: str = "cpu",
) -> np.ndarray:
    """Mark j > i as duplicate iff misori<tol AND |Δpos|<pos_tol_um.

    Outer-serial / inner-parallel-equivalent: returns ``isDup`` array
    indexed by Stage-1 cluster order (= grain_positions order).

    Optimised to O(N) with a 5 µm spatial hash on rep positions, since
    most rep pairs are far apart and don't need to be checked.
    """
    n = grain_positions.size
    is_dup = np.zeros(n, dtype=bool)
    if n < 2:
        return is_dup

    print(f"[c-parity] PassA: {n:,} grain positions, "
          f"misori<{math.degrees(misori_tol_rad):.3f}° AND |Δpos|<{pos_tol_um} µm",
          flush=True)

    t0 = time.time()
    positions = opf[grain_positions, OPF_POS]               # (n, 3) float64

    # Spatial hash — bucket positions into pos_tol_um cubes. Two reps must be
    # in the same or an adjacent cube to satisfy |Δpos| < pos_tol_um.
    cell = pos_tol_um
    cell_idx = np.floor(positions / cell).astype(np.int64)  # (n, 3)
    bucket: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(n):
        key = (int(cell_idx[i, 0]), int(cell_idx[i, 1]), int(cell_idx[i, 2]))
        bucket.setdefault(key, []).append(i)

    print(f"[c-parity] PassA: {len(bucket):,} non-empty cells "
          f"[{time.time()-t0:.1f}s]", flush=True)

    # Pre-compute quats for misori batches (on requested device).
    om_grain = np.ascontiguousarray(opf[grain_positions, :][:, OPF_OM])
    q_grain_t = orient_mat_to_quat(torch.from_numpy(om_grain).to(device))
    q_grain_t = fundamental_zone(q_grain_t, space_group)               # stays on device

    # For each cube, generate i<j candidate pairs from same cube + 27-cube
    # neighbourhood. With cell == pos_tol_um, any pair with |Δpos| < cell
    # falls in the same or adjacent cube.
    pair_chunks: List[np.ndarray] = []
    keys = list(bucket.keys())
    for kx, ky, kz in keys:
        members_a = bucket.get((kx, ky, kz), [])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    other = bucket.get((kx + dx, ky + dy, kz + dz))
                    if other is None:
                        continue
                    if (dx, dy, dz) == (0, 0, 0):
                        # Internal pairs i<j only.
                        if len(members_a) < 2:
                            continue
                        a = np.asarray(members_a, dtype=np.int64)
                        ii, jj = np.triu_indices(a.size, k=1)
                        chunk = np.stack([a[ii], a[jj]], axis=1)
                    else:
                        # Cross-cell pairs; only emit (i, j) with i < j to
                        # match C's serial outer-loop order. EVERY offset must
                        # be walked: the i < j mask already emits each unordered
                        # pair exactly once, from the side holding the lower
                        # index. Walking only lex-positive offsets as well
                        # dropped every cross-cell pair whose lower index sat in
                        # the lex-greater cell -- about half of them -- so pairs
                        # C merges survived (two identical orientations 2 um
                        # apart across a cell face were never compared).
                        a = np.asarray(members_a, dtype=np.int64)
                        b = np.asarray(other, dtype=np.int64)
                        ai, bj = np.meshgrid(a, b, indexing="ij")
                        ai = ai.ravel()
                        bj = bj.ravel()
                        mask = ai < bj
                        if not mask.any():
                            continue
                        chunk = np.stack([ai[mask], bj[mask]], axis=1)
                    pair_chunks.append(chunk)

    if not pair_chunks:
        print(f"[c-parity] PassA: 0 candidate pairs from spatial hash",
              flush=True)
        return is_dup

    pairs = np.concatenate(pair_chunks, axis=0)
    pairs = np.unique(
        pairs[:, 0] * np.int64(2**31) + pairs[:, 1]
    )
    a = pairs // np.int64(2**31)
    b = pairs % np.int64(2**31)
    pairs = np.stack([a, b], axis=1)
    print(f"[c-parity] PassA: {pairs.shape[0]:,} candidate pairs "
          f"[{time.time()-t0:.1f}s]", flush=True)

    # |Δpos|² < pos_tol_um² check (vectorised).
    dpos = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    d2 = (dpos * dpos).sum(axis=1)
    pos_ok = d2 < (pos_tol_um * pos_tol_um)
    pairs = pairs[pos_ok]
    print(f"[c-parity] PassA: {pairs.shape[0]:,} pairs pass |Δpos|<{pos_tol_um} "
          f"[{time.time()-t0:.1f}s]", flush=True)
    if pairs.shape[0] == 0:
        return is_dup

    # Misori check (vectorised, on device).
    pairs_t0 = torch.from_numpy(np.ascontiguousarray(pairs[:, 0])).to(device)
    pairs_t1 = torch.from_numpy(np.ascontiguousarray(pairs[:, 1])).to(device)
    qa = q_grain_t.index_select(0, pairs_t0)
    qb = q_grain_t.index_select(0, pairs_t1)
    misori_t = misorientation_quat_batch(qa, qb, space_group)
    misori = misori_t.detach().cpu().numpy().astype(np.float64)
    misori_ok = misori < misori_tol_rad
    final = pairs[misori_ok]
    print(f"[c-parity] PassA: {final.shape[0]:,} pairs pass misori "
          f"[{time.time()-t0:.1f}s]", flush=True)

    # Greedy outer-serial dedup: walk pairs in (i, j) lex order; if neither
    # endpoint is already marked, mark j. Matches C: outer i ascending,
    # inner j ascending; we never re-check a pair that involves a now-dup
    # endpoint (C: `if (isDup[ii]) continue;` and `if (isDup[jj]) continue`).
    # Sort final pairs by (i, j).
    order = np.lexsort((final[:, 1], final[:, 0]))
    final = final[order]
    for k in range(final.shape[0]):
        i, j = int(final[k, 0]), int(final[k, 1])
        if is_dup[i] or is_dup[j]:
            continue
        is_dup[j] = True

    n_dup = int(is_dup.sum())
    print(f"[c-parity] PassA done: {n_dup:,} dups; {n - n_dup:,} survive "
          f"[{time.time()-t0:.1f}s]", flush=True)
    return is_dup


# --------------------------------------------------------------------------
# kept-list assembly (Stage1 + PassA + confidence filter)
# --------------------------------------------------------------------------


def build_kept_list(
    *,
    grain_positions: np.ndarray,
    is_dup: np.ndarray,
    opf: np.ndarray,
    confidence_min: float = 0.05,
) -> np.ndarray:
    """Return the indices into ``grain_positions`` that survive both PassA
    dedup and the ``OPs[ri][22] < 0.05`` confidence filter.
    """
    survives_dup = ~is_dup
    confidences = opf[grain_positions, OPF_CONFIDENCE]
    survives_conf = confidences >= confidence_min
    keep = survives_dup & survives_conf
    return np.flatnonzero(keep)
