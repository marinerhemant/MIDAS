"""Pass-1.5 twin-aware cluster merge.

For heavily-twinned materials (LMO, MnO₂, hexagonal alloys), forward-predict
Pass-1 cannot merge twin variants because their predicted spot positions
are physically distinct (different ω rotations of the same parent atoms).
The result is N twin variants of one parent grain end up in N disjoint
Pass-1 clusters.

This module identifies cluster pairs that are likely twin variants (or
alt-indexings) of the same physical parent, and returns a remap that
collapses them into a single parent cluster.

Two merge modes:

* ``direct``: merge clusters with raw misorientation < ``tol_misori_deg``
  AND positions within ``tol_position_um``. Catches alt-indexings — same
  parent indexed multiple times with slightly different orientations.

* ``twin_aware``: also try all twin operators for the space group; merge
  if ANY operator yields misorientation < ``tol_misori_deg``. Catches
  twin variants of the same parent.

Use ``mode="combined"`` (default) for both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, List

import numpy as np


def _orient_mat_to_quat(om: np.ndarray) -> np.ndarray:
    """(...,3,3) → (...,4) unit quaternion (w,x,y,z), via Cayley/Shepperd."""
    om = np.asarray(om, dtype=np.float64)
    *batch, _, _ = om.shape
    flat = om.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4), dtype=np.float64)
    for k in range(flat.shape[0]):
        m = flat[k]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (m[2, 1] - m[1, 2]) / S
            y = (m[0, 2] - m[2, 0]) / S
            z = (m[1, 0] - m[0, 1]) / S
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
        n = np.sqrt(w * w + x * x + y * y + z * z)
        out[k] = (w / n, x / n, y / n, z / n)
    return out.reshape(*batch, 4)


def _q_misori_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    """Quaternion misorientation angle in radians (smallest 2·acos|<q1,q2>|)."""
    d = abs(float(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]))
    d = min(d, 1.0)
    return 2.0 * np.arccos(d)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion multiply (w,x,y,z)."""
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])


def _sym_misori_rad(qa: np.ndarray, qb: np.ndarray, space_group: int,
                    chunk: int = 200_000) -> np.ndarray:
    """Crystal-symmetry-reduced misorientation (rad) of paired quaternion rows, chunked.

    ``midas_stress.orientation.misorientation_quat_batch`` reduces over the point group of
    ``space_group`` on the crystal side, so two orientations that are the same grain under
    different symmetry representatives (or on opposite sides of a fundamental-zone boundary)
    compare as ~0, which a raw quaternion dot product does not.
    """
    import torch
    from midas_stress.orientation import misorientation_quat_batch
    n = qa.shape[0]
    out = np.empty(n, dtype=np.float64)
    for lo in range(0, n, chunk):
        sl = slice(lo, lo + chunk)
        out[sl] = misorientation_quat_batch(
            torch.from_numpy(np.ascontiguousarray(qa[sl])),
            torch.from_numpy(np.ascontiguousarray(qb[sl])),
            space_group).numpy()
    return out


def _qmul_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion multiply for arrays. ``a`` and ``b`` are (...,4) and
    broadcast against each other. Returns (...,4)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class ClusterMergeResult:
    """Result of :func:`compute_cluster_merges`.

    Attributes
    ----------
    parent_cluster_id : (N_clusters,) int
        New parent cluster IDs after merging (densely renumbered 0..M-1).
    n_in_clusters : int
        Number of input clusters.
    n_out_parents : int
        Number of distinct merged parents.
    n_merges_direct : int
        Number of cluster pairs merged by direct misori.
    n_merges_twin : int
        Number of cluster pairs merged via twin operators.
    """

    parent_cluster_id: np.ndarray
    n_in_clusters: int
    n_out_parents: int
    n_merges_direct: int
    n_merges_twin: int


def compute_cluster_merges(
    *,
    cluster_orientation_matrices: np.ndarray,    # (N_c, 3, 3) — mean OM per cluster
    cluster_positions_um: np.ndarray,            # (N_c, 3) — mean centroid per cluster
    space_group: int,
    c_over_a: Optional[float] = None,
    tol_misori_deg: float = 2.0,
    tol_position_um: float = 200.0,
    mode: Literal["direct", "twin", "combined"] = "combined",
) -> ClusterMergeResult:
    """Compute a parent-cluster remap that merges alt-indexings and twin
    variants of the same physical parent grain.

    Parameters
    ----------
    cluster_orientation_matrices : (N_c, 3, 3) float
        Per-cluster mean orientation matrix.
    cluster_positions_um : (N_c, 3) float
        Per-cluster mean centroid in µm.
    space_group : int
        IT space group number; used to fetch twin operators.
    c_over_a : float, optional
        Required for HCP / tetragonal twin operators.
    tol_misori_deg : float, default 2.0
        Maximum misorientation to consider a merge candidate.
    tol_position_um : float, default 200.0
        Maximum spatial distance to consider a merge candidate.
    mode : "direct"|"twin"|"combined"
        ``direct``: only direct-misori merges (alt-indexings).
        ``twin``:   only twin-operator-mediated merges.
        ``combined``: both (recommended).

    Returns
    -------
    ClusterMergeResult
    """
    OMs = np.asarray(cluster_orientation_matrices, dtype=np.float64)
    pos = np.asarray(cluster_positions_um, dtype=np.float64)
    N_c = OMs.shape[0]
    if N_c == 0:
        return ClusterMergeResult(
            parent_cluster_id=np.zeros(0, dtype=np.int64),
            n_in_clusters=0, n_out_parents=0,
            n_merges_direct=0, n_merges_twin=0,
        )

    qs = _orient_mat_to_quat(OMs)        # (N_c, 4)
    tol_rad = float(np.radians(tol_misori_deg))
    tol_pos_sq = float(tol_position_um) ** 2

    # Twin operators
    twin_qs: List[np.ndarray] = []
    if mode in ("twin", "combined"):
        from .twins import default_twin_relations_for
        try:
            relations = default_twin_relations_for(
                space_group=space_group, c_over_a=c_over_a,
            )
            twin_qs = [np.asarray(r.quaternion, dtype=np.float64) for r in relations]
        except Exception:
            twin_qs = []

    uf = _UnionFind(N_c)
    n_direct = 0
    n_twin = 0

    # Spatial pair finding: KD-tree query gives all pairs (i,j) with
    # ‖pos_i - pos_j‖ ≤ tol_position_um. Returns sorted (i, j) with i < j.
    try:
        from scipy.spatial import cKDTree as KDTree
        tree = KDTree(pos)
        pair_list = tree.query_pairs(r=tol_position_um, output_type="ndarray")
    except ImportError:
        # Fallback to O(N²); only fine for small N.
        ii, jj = np.where(
            np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) <= tol_position_um
        )
        mask = ii < jj
        pair_list = np.stack([ii[mask], jj[mask]], axis=1)
    if pair_list.size == 0:
        roots = list(range(N_c))
    else:
        I, J = pair_list[:, 0], pair_list[:, 1]
        # Direct misori, symmetry-reduced (the raw quaternion dot product this replaced
        # missed alt-indexings that sit in different symmetry representatives / across a
        # fundamental-zone boundary).
        qI = qs[I]; qJ = qs[J]
        misori_direct_rad = _sym_misori_rad(qI, qJ, space_group)

        if mode in ("direct", "combined"):
            mask_dir = misori_direct_rad < tol_rad
            for k in np.flatnonzero(mask_dir):
                uf.union(int(I[k]), int(J[k]))
            n_direct = int(mask_dir.sum())
            # Remaining unmatched pairs (for the twin path)
            remaining = ~mask_dir
        else:
            remaining = np.ones(pair_list.shape[0], dtype=bool)

        # Twin-operator misori, SYMMETRY-AWARE, with the operator on the CRYSTAL side.
        # A twin relation is O_J = O_I . T . S (T in the crystal frame, S any crystal
        # symmetry operator), so the test is  misorientation(q_I, q_J . T)  reduced over
        # the point group, in both directions (T need not be an involution, and neither
        # cluster is the "parent"). The earlier version tested |<q_I, T.q_J>| with T on the
        # LAB side and no symmetry reduction: it found a twin only when the parent was the
        # identity (the only test) and merged 0 of 40 random FCC Sigma3 twin pairs.
        # Loop over T and chunk the pairs to keep peak memory flat.
        if mode in ("twin", "combined") and twin_qs and remaining.any():
            I_r = I[remaining]; J_r = J[remaining]
            qI_r = qs[I_r]; qJ_r = qs[J_r]
            n_r = I_r.shape[0]
            misori_best = np.full(n_r, np.inf, dtype=np.float64)
            for T in twin_qs:
                T_b = np.asarray(T, dtype=np.float64)[None, :]
                m_a = _sym_misori_rad(qI_r, _qmul_batch(qJ_r, T_b), space_group)
                m_b = _sym_misori_rad(qJ_r, _qmul_batch(qI_r, T_b), space_group)
                np.minimum(misori_best, np.minimum(m_a, m_b), out=misori_best)
            mask_twin = misori_best < tol_rad
            for k in np.flatnonzero(mask_twin):
                uf.union(int(I_r[k]), int(J_r[k]))
            n_twin = int(mask_twin.sum())

    # Renumber roots densely
    if pair_list.size == 0:
        roots = list(range(N_c))
    else:
        roots = [uf.find(i) for i in range(N_c)]
    unique_roots = sorted(set(roots))
    root_to_dense = {r: k for k, r in enumerate(unique_roots)}
    parent_id = np.array([root_to_dense[r] for r in roots], dtype=np.int64)

    return ClusterMergeResult(
        parent_cluster_id=parent_id,
        n_in_clusters=N_c,
        n_out_parents=len(unique_roots),
        n_merges_direct=n_direct,
        n_merges_twin=n_twin,
    )
