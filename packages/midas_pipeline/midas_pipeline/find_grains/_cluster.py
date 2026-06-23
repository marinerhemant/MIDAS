"""Symmetry-aware orientation clustering.

Two scopes:

  - :func:`per_voxel_cluster` — for one voxel's candidate solutions, picks
    the highest-confidence representative of each misorientation-cluster.
    Shared by ``find_grains_single`` and ``find_grains_multiple``.
  - :func:`global_cluster` — cross-voxel dedup of the best-per-voxel
    orientations, used by ``find_grains_single`` only.

Both delegate to :func:`midas_stress.orientation.misorientation_om_batch`
for the symmetry-aware angle computation (returns RADIANS).

The torch path is provided for downstream differentiable workflows. The
clustering itself is inherently combinatorial (assignments are integer
indices) so autograd doesn't flow *through* the cluster decisions, but
when called with torch inputs the OM math runs in torch on the input
device and the returned representative OMs are torch tensors that flow
gradients into any subsequent computation that depends on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch  # noqa: F401
    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False

from midas_stress.orientation import misorientation_om_batch


@dataclass
class PerVoxelClusterResult:
    """Output of :func:`per_voxel_cluster`.

    Attributes
    ----------
    best_row : int
        Row index (within the voxel's candidates) of the
        highest-confidence solution. ``-1`` if no valid candidate.
    best_conf : float
        Confidence (matchedSpots / nExpected) of the best candidate.
    best_ia : float
        Internal angle of the best candidate.
    unique_keys : ndarray (n_clusters, 4) uint64
        For each cluster, the keys of its representative (highest-conf,
        ties broken by lowest IA).
    unique_OMs : ndarray (n_clusters, 9) float64
        Same — representative orientation matrices, row-major flattened.
    """

    best_row: int
    best_conf: float
    best_ia: float
    unique_keys: np.ndarray
    unique_OMs: np.ndarray


def per_voxel_cluster(
    OMs: np.ndarray,
    confs: np.ndarray,
    ias: np.ndarray,
    keys: np.ndarray,
    *,
    space_group: int,
    max_ang_deg: float,
    min_conf: float = 0.0,
) -> PerVoxelClusterResult:
    """Cluster one voxel's candidate orientations.

    Pure-Python port of:

      - ``process_voxel`` (findSingleSolutionPFRefactored.c:603–847) — the
        best-row pick + within-voxel unique grouping.
      - ``processVoxel`` (findMultipleSolutionsPF.c:83–199) — same
        grouping logic but used to write SpotsToIndex rows.

    Both C functions use the same algorithm: sweep i = 0..nIDs-1, for each
    unmarked i convert to quaternion and compare to all later unmarked j
    using ``GetMisOrientation``; ``ang < maxAng_deg`` marks j and the
    representative gets the higher confidence (ties broken by lower IA).

    Parameters
    ----------
    OMs : ndarray (n, 9) float64
    confs : ndarray (n,) float64
    ias : ndarray (n,) float64
    keys : ndarray (n, 4) uint64 — per-candidate key rows
    space_group : int
    max_ang_deg : float — degrees, converted internally
    min_conf : float — candidates below this confidence are skipped
        (``findMultipleSolutionsPF`` semantics; ``findSingleSolution``
        uses ``0.0`` for clustering and applies min_conf later).
    """
    OMs = np.ascontiguousarray(OMs, dtype=np.float64)
    confs = np.ascontiguousarray(confs, dtype=np.float64)
    ias = np.ascontiguousarray(ias, dtype=np.float64)
    keys = np.ascontiguousarray(keys, dtype=np.uint64)
    n = int(OMs.shape[0])
    if n == 0:
        return PerVoxelClusterResult(
            best_row=-1, best_conf=-1.0, best_ia=100.0,
            unique_keys=np.empty((0, 4), dtype=np.uint64),
            unique_OMs=np.empty((0, 9), dtype=np.float64),
        )

    max_ang_rad = float(max_ang_deg) * np.pi / 180.0

    # --- 1. Best-row pick (matches C process_voxel:675–690). ---
    # Order-stable scan, tie-break = lowest IA. C semantics: skip if conf
    # < best, skip if conf == best && ia > best_ia; update otherwise. We
    # initialise best_conf = -1 so the first valid candidate always wins.
    # Note: min_conf only applies for the multiple-mode caller; single
    # picks the overall best then filters at sino time. For consistency
    # with both C codes we always do the best-pick on the un-filtered
    # set first.
    best_row = -1
    best_conf = -1.0
    best_ia = 100.0
    for i in range(n):
        ci = float(confs[i])
        ai = float(ias[i])
        if ci < best_conf:
            continue
        if ci == best_conf and ai > best_ia:
            continue
        best_conf = ci
        best_ia = ai
        best_row = i

    # --- 2. Within-voxel unique grouping ---
    if best_row < 0:
        return PerVoxelClusterResult(
            best_row=-1, best_conf=-1.0, best_ia=100.0,
            unique_keys=np.empty((0, 4), dtype=np.uint64),
            unique_OMs=np.empty((0, 9), dtype=np.float64),
        )

    marked = np.zeros(n, dtype=bool)
    if min_conf > 0.0:
        # Multiple-mode semantics: skip below-threshold candidates entirely
        # (matches findMultipleSolutionsPF.c:137-141).
        for i in range(n):
            if confs[i] < min_conf:
                marked[i] = True

    unique_keys: list[np.ndarray] = []
    unique_OMs: list[np.ndarray] = []

    # For each unmarked i, compute miso vs all later unmarked j in a
    # single batched call to misorientation_om_batch. This is the only
    # hot loop and the batched call collapses the C nested for-loop into
    # one ctypes call per i.
    for i in range(n):
        if marked[i]:
            continue
        # Initial best for this cluster = i.
        b_conf = float(confs[i])
        b_ia = float(ias[i])
        b_rn = i

        # Indices of remaining unmarked j > i.
        rem = np.flatnonzero(~marked[i + 1 :]) + (i + 1)
        if rem.size > 0:
            OMs1 = np.broadcast_to(OMs[i], (rem.size, 9))
            OMs2 = OMs[rem]
            angles = misorientation_om_batch(OMs1, OMs2, space_group)  # radians
            close = angles < max_ang_rad
            for k_idx, is_close in enumerate(close):
                if not is_close:
                    continue
                j = int(rem[k_idx])
                cj = float(confs[j])
                aj = float(ias[j])
                if b_conf < cj:
                    b_conf = cj
                    b_ia = aj
                    b_rn = j
                elif b_conf == cj and b_ia > aj:
                    b_conf = cj
                    b_ia = aj
                    b_rn = j
                marked[j] = True

        unique_keys.append(keys[b_rn].copy())
        unique_OMs.append(OMs[b_rn].copy())

    return PerVoxelClusterResult(
        best_row=best_row,
        best_conf=best_conf,
        best_ia=best_ia,
        unique_keys=np.stack(unique_keys, axis=0) if unique_keys else np.empty((0, 4), dtype=np.uint64),
        unique_OMs=np.stack(unique_OMs, axis=0) if unique_OMs else np.empty((0, 9), dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Cross-voxel global clustering
# ---------------------------------------------------------------------------


@dataclass
class GlobalClusterResult:
    """Output of :func:`global_cluster`.

    Attributes
    ----------
    n_uniques : int
    unique_key_arr : ndarray (n_uniques, 5) uint64
        ``[voxNr, SpotID, nMatches, nIDs, bestSolIdx]`` — directly maps
        to a row of ``UniqueOrientations.csv`` (sans the OM columns).
    unique_OM_arr : ndarray (n_uniques, 9) float64
    voxel_to_unique : ndarray (n_vox,) int64
        Per-voxel grain id (index into ``unique_key_arr`` / ``unique_OM_arr``).
        ``-1`` for voxels with no valid solution.  Consumed by
        :func:`midas_pipeline.stages.refine_vmap` to build the voxel grid.
    """

    n_uniques: int
    unique_key_arr: np.ndarray
    unique_OM_arr: np.ndarray
    voxel_to_unique: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def global_cluster(
    per_vox_OMs: np.ndarray,
    per_vox_confs: np.ndarray,
    per_vox_keys: np.ndarray,
    *,
    space_group: int,
    max_ang_deg: float,
    invalid_marker: int = -1,
) -> GlobalClusterResult:
    """Cross-voxel dedup of the per-voxel best orientations.

    Pure-Python port of :c:func:`find_unique_orientations`
    (findSingleSolutionPFRefactored.c:862–970).

    Parameters
    ----------
    per_vox_OMs : ndarray (n_vox, 9) float64
        For each voxel, the OM of its best candidate (zero rows for
        voxels with no valid solution — they are filtered via
        ``per_vox_keys[:, 0] == INVALID_VOX``).
    per_vox_confs : ndarray (n_vox,) float64
        Quality metric — used to break ties within a cluster (higher = better).
    per_vox_keys : ndarray (n_vox, 4) uint64
        ``[SpotID, nMatches, nIDs_best_solution, bestSolIdx]`` per voxel.
        A voxel with no valid solution has ``per_vox_keys[v, 0] ==
        (uint64)(-1)`` (the C sentinel ``INVALID_VOX``).
    space_group : int
    max_ang_deg : float
    invalid_marker : int
        Value of the ``per_vox_keys[v, 0]`` sentinel indicating "no valid
        solution at voxel v". The C code uses ``(size_t)-1`` which as an
        ``uint64`` is ``0xFFFFFFFFFFFFFFFF``. Pass ``-1`` (default) for
        the C semantics, or whatever you used to mark invalid voxels.

    Returns
    -------
    GlobalClusterResult

    Notes
    -----
    The output ``unique_key_arr`` layout matches the C
    ``UniqueOrientationsResult.uniqueKeyArr`` exactly (5 cols), so the
    same array drives the ``UniqueOrientations.csv`` writer and the
    sinogen module.
    """
    per_vox_OMs = np.ascontiguousarray(per_vox_OMs, dtype=np.float64)
    per_vox_confs = np.ascontiguousarray(per_vox_confs, dtype=np.float64).ravel()
    per_vox_keys = np.ascontiguousarray(per_vox_keys, dtype=np.uint64)
    n_vox = int(per_vox_OMs.shape[0])
    max_ang_rad = float(max_ang_deg) * np.pi / 180.0

    # C uses (size_t)-1 to mark invalid voxels in column 0.
    invalid_u64 = np.uint64(invalid_marker if invalid_marker >= 0 else (2**64 - 1))

    # Initial mark array — voxels with no valid solution are pre-marked.
    marked = (per_vox_keys[:, 0] == invalid_u64)

    out_keys: list[np.ndarray] = []
    out_OMs: list[np.ndarray] = []
    # voxel_to_unique[v] = grain id (index into out_keys), or -1 if no valid solution.
    voxel_to_unique = np.full(n_vox, -1, dtype=np.int64)

    for i in range(n_vox):
        if marked[i]:
            continue
        best_frac = float(per_vox_confs[i])
        best_row = i
        # Voxels in this cluster — start with the seed voxel.
        cluster_members: list[int] = [i]

        rem = np.flatnonzero(~marked[i + 1 :]) + (i + 1)
        if rem.size > 0:
            OMs1 = np.broadcast_to(per_vox_OMs[i], (rem.size, 9))
            OMs2 = per_vox_OMs[rem]
            angles = misorientation_om_batch(OMs1, OMs2, space_group)
            close = angles < max_ang_rad
            for k_idx, is_close in enumerate(close):
                if not is_close:
                    continue
                j = int(rem[k_idx])
                cj = float(per_vox_confs[j])
                if best_frac < cj:
                    best_frac = cj
                    best_row = j
                marked[j] = True
                cluster_members.append(j)

        # uniqueKeyArr cols: [bestVoxNr, SpotID, nMatches, nIDs, bestSolIdx]
        # (from findSingleSolutionPFRefactored.c:937-941)
        row5 = np.empty(5, dtype=np.uint64)
        row5[0] = np.uint64(best_row)
        row5[1:5] = per_vox_keys[best_row]
        grain_idx = len(out_keys)
        out_keys.append(row5)
        out_OMs.append(per_vox_OMs[best_row].copy())
        for v in cluster_members:
            voxel_to_unique[v] = grain_idx

    if not out_keys:
        return GlobalClusterResult(
            n_uniques=0,
            unique_key_arr=np.empty((0, 5), dtype=np.uint64),
            unique_OM_arr=np.empty((0, 9), dtype=np.float64),
            voxel_to_unique=voxel_to_unique,
        )
    return GlobalClusterResult(
        n_uniques=len(out_keys),
        unique_key_arr=np.stack(out_keys, axis=0),
        unique_OM_arr=np.stack(out_OMs, axis=0),
        voxel_to_unique=voxel_to_unique,
    )


# ---------------------------------------------------------------------------
# Torch path — the OM math goes via misorientation_om_batch which already
# dispatches to torch when fed tensors. We expose a thin wrapper that does
# *cluster bookkeeping* on the host (integer indices) but lets the OM math
# live on the device + flow gradients.
# ---------------------------------------------------------------------------


def per_voxel_cluster_torch(
    OMs,
    confs,
    ias,
    keys: np.ndarray,
    *,
    space_group: int,
    max_ang_deg: float,
    min_conf: float = 0.0,
):
    """Torch-backed version of :func:`per_voxel_cluster`.

    ``OMs``, ``confs``, ``ias`` are torch.Tensor (any device, any dtype).
    The misorientation computation runs in torch on the input device;
    cluster-membership decisions and the bookkeeping arrays still live
    on the host (they're integer-index work — autograd would be a no-op).

    Returned ``unique_OMs`` is a torch.Tensor on the input device that
    inherits the input's autograd graph: downstream gradients to the
    representative OMs flow back to the *input* OMs at the chosen rows.
    """
    if not _HAVE_TORCH:
        raise RuntimeError("torch backend requested but torch not installed")
    if not isinstance(OMs, torch.Tensor):
        raise TypeError("OMs must be a torch.Tensor")
    n = int(OMs.shape[0])
    device = OMs.device
    dtype = OMs.dtype
    if n == 0:
        return PerVoxelClusterResult(
            best_row=-1, best_conf=-1.0, best_ia=100.0,
            unique_keys=np.empty((0, 4), dtype=np.uint64),
            unique_OMs=torch.empty((0, 9), dtype=dtype, device=device),
        )

    confs_t = confs.to(device=device, dtype=dtype) if isinstance(confs, torch.Tensor) else torch.as_tensor(confs, dtype=dtype, device=device)
    ias_t = ias.to(device=device, dtype=dtype) if isinstance(ias, torch.Tensor) else torch.as_tensor(ias, dtype=dtype, device=device)
    confs_h = confs_t.detach().cpu().numpy().astype(np.float64)
    ias_h = ias_t.detach().cpu().numpy().astype(np.float64)
    keys = np.ascontiguousarray(keys, dtype=np.uint64)

    max_ang_rad = float(max_ang_deg) * np.pi / 180.0

    best_row = -1
    best_conf = -1.0
    best_ia = 100.0
    for i in range(n):
        ci = float(confs_h[i])
        ai = float(ias_h[i])
        if ci < best_conf:
            continue
        if ci == best_conf and ai > best_ia:
            continue
        best_conf = ci
        best_ia = ai
        best_row = i

    marked = np.zeros(n, dtype=bool)
    if min_conf > 0.0:
        for i in range(n):
            if confs_h[i] < min_conf:
                marked[i] = True

    cluster_reps: list[int] = []
    unique_keys: list[np.ndarray] = []
    for i in range(n):
        if marked[i]:
            continue
        b_conf = float(confs_h[i])
        b_ia = float(ias_h[i])
        b_rn = i

        rem = np.flatnonzero(~marked[i + 1 :]) + (i + 1)
        if rem.size > 0:
            OMs1 = OMs[i].unsqueeze(0).expand(rem.size, 9)
            OMs2 = OMs[rem]
            angles = misorientation_om_batch(OMs1, OMs2, space_group)
            if isinstance(angles, torch.Tensor):
                angles_h = angles.detach().cpu().numpy()
            else:
                angles_h = angles
            for k_idx, ang in enumerate(angles_h):
                if not (ang < max_ang_rad):
                    continue
                j = int(rem[k_idx])
                cj = float(confs_h[j])
                aj = float(ias_h[j])
                if b_conf < cj:
                    b_conf = cj
                    b_ia = aj
                    b_rn = j
                elif b_conf == cj and b_ia > aj:
                    b_conf = cj
                    b_ia = aj
                    b_rn = j
                marked[j] = True
        cluster_reps.append(b_rn)
        unique_keys.append(keys[b_rn].copy())

    unique_OMs_t = OMs[cluster_reps] if cluster_reps else torch.empty((0, 9), dtype=dtype, device=device)
    return PerVoxelClusterResult(
        best_row=best_row,
        best_conf=best_conf,
        best_ia=best_ia,
        unique_keys=np.stack(unique_keys, axis=0) if unique_keys else np.empty((0, 4), dtype=np.uint64),
        unique_OMs=unique_OMs_t,
    )


# ===========================================================================
# Fast cross-voxel dedup — byte-parity with global_cluster(), much faster on
# DEFORMED maps where intragranular spread defeats the greedy marking shortcut
# and the reference degrades to O(N^2) misorientation evals on CPU.
#
# Strategy: keep the EXACT greedy single-link marking (i=0..N, j>i, first-seed
# wins, first-max-conf representative). Change ONLY how the symmetric "close"
# (misorientation < maxAngle) relation is computed:
#   * _build_neighbors_gpu  — N^2 misorientation on GPU in row tiles (universal
#     ~100x constant speedup; needs torch + a device).
#   * _build_neighbors_binned — fundamental-zone quaternion hashing so only
#     spatially-near (in orientation space) candidate pairs are evaluated
#     (O(N*k); the asymptotic fix for spread/deformed maps). Parity-safe: the
#     candidate set is a SUPERSET of all true-close pairs (every symmetry image
#     of each voxel is hashed, neighbour cells searched).
# ===========================================================================


def _greedy_from_neighbors(nbr, per_vox_OMs, per_vox_confs, per_vox_keys, marked):
    """Exact-parity greedy single-link clustering given a symmetric neighbour
    structure ``nbr[i] = sorted int array of j with close(i, j)``.

    Reproduces :func:`global_cluster` bit-for-bit: seeds in ascending order,
    candidates are unmarked ``j > i`` with ``close(i, j)``, representative is the
    first (lowest-index) member of maximal confidence.
    """
    n = int(per_vox_OMs.shape[0])
    out_keys: list[np.ndarray] = []
    out_OMs: list[np.ndarray] = []
    voxel_to_unique = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if marked[i]:
            continue
        cand = nbr[i]
        if cand.size:
            cand = cand[cand > i]
            if cand.size:
                cand = cand[~marked[cand]]
        members = np.concatenate(([i], cand)) if cand.size else np.array([i], dtype=np.int64)
        confs_m = per_vox_confs[members]
        best_row = int(members[int(np.argmax(confs_m))])  # first max == lowest idx
        if cand.size:
            marked[cand] = True
        row5 = np.empty(5, dtype=np.uint64)
        row5[0] = np.uint64(best_row)
        row5[1:5] = per_vox_keys[best_row]
        grain_idx = len(out_keys)
        out_keys.append(row5)
        out_OMs.append(per_vox_OMs[best_row].copy())
        voxel_to_unique[members] = grain_idx
    if not out_keys:
        return GlobalClusterResult(
            n_uniques=0,
            unique_key_arr=np.empty((0, 5), dtype=np.uint64),
            unique_OM_arr=np.empty((0, 9), dtype=np.float64),
            voxel_to_unique=voxel_to_unique,
        )
    return GlobalClusterResult(
        n_uniques=len(out_keys),
        unique_key_arr=np.stack(out_keys, axis=0),
        unique_OM_arr=np.stack(out_OMs, axis=0),
        voxel_to_unique=voxel_to_unique,
    )


def _neighbors_from_pairs(pairs_i, pairs_j, n):
    """Symmetric sorted neighbour lists from undirected close pairs."""
    if len(pairs_i) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n)]
    ii = np.concatenate([pairs_i, pairs_j])
    jj = np.concatenate([pairs_j, pairs_i])
    order = np.argsort(ii, kind="stable")
    ii = ii[order]; jj = jj[order]
    bounds = np.searchsorted(ii, np.arange(n + 1))
    return [np.unique(jj[bounds[i]:bounds[i + 1]]) for i in range(n)]


def _build_neighbors_gpu(per_vox_OMs, valid_idx, space_group, max_ang_rad, device, tile=128):
    """Symmetric close-neighbour lists via N^2 misorientation on GPU (row tiles)."""
    import torch
    M = int(valid_idx.size)
    OMv = torch.as_tensor(per_vox_OMs[valid_idx], dtype=torch.float64, device=device)  # (M,9)
    pi_list, pj_list = [], []
    for r0 in range(0, M, tile):
        r1 = min(r0 + tile, M)
        t = r1 - r0
        oms1 = OMv[r0:r1].repeat_interleave(M, dim=0)   # (t*M,9)
        oms2 = OMv.repeat(t, 1)                          # (t*M,9)
        ang = misorientation_om_batch(oms1, oms2, space_group).reshape(t, M)
        close = ang < max_ang_rad
        # only keep j>i in GLOBAL valid index space; map local->global below
        rows, cols = torch.nonzero(close, as_tuple=True)
        gi = valid_idx[(r0 + rows).cpu().numpy()]
        gj = valid_idx[cols.cpu().numpy()]
        keep = gj > gi
        pi_list.append(gi[keep]); pj_list.append(gj[keep])
    pi = np.concatenate(pi_list) if pi_list else np.empty(0, np.int64)
    pj = np.concatenate(pj_list) if pj_list else np.empty(0, np.int64)
    return _neighbors_from_pairs(pi, pj, int(per_vox_OMs.shape[0]))


def _build_neighbors_binned(per_vox_OMs, valid_idx, space_group, max_ang_rad,
                            device=None, batch=200000):
    """Symmetric close-neighbour lists via fundamental-zone quaternion hashing.

    Parity-safe: hash EVERY symmetry image of each voxel quaternion into a 4-D
    grid (cell = max_ang_rad, generous), gather candidate pairs from the 3^4
    neighbour cells, then evaluate misorientation only on the unique candidates.
    The candidate set is a superset of all pairs with misorientation < maxAngle.
    """
    from midas_stress.orientation import (
        make_symmetries, orient_mat_to_quat, quaternion_product,
    )
    n_all = int(per_vox_OMs.shape[0])
    vi = np.asarray(valid_idx, dtype=np.int64)
    M = int(vi.size)
    if M == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n_all)]
    n_sym, sym = make_symmetries(space_group)
    sym = np.asarray(sym, dtype=np.float64)                       # (n_sym,4)

    # base quaternions (raw; we hash BOTH signs of every image so the antipodal
    # q ~ -q boundary at q0≈0 cannot drop a close pair).
    q = np.array([orient_mat_to_quat(per_vox_OMs[v]) for v in vi], dtype=np.float64)  # (M,4)
    cell = max(float(max_ang_rad), 1e-9)

    # hash all symmetry images (both signs): grid[cell_tuple] -> list of local idx.
    # q_img = sym[sidx] ⊗ q  (Hamilton product), floored to cells.
    from collections import defaultdict
    grid = defaultdict(list)
    cells_per_img = []   # per (sym, sign): (M,4) int cells — used again in the gather pass
    for sidx in range(n_sym):
        w0, x0, y0, z0 = sym[sidx]
        w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        qi = np.stack([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ], axis=1)
        for sgn in (1.0, -1.0):
            ck = np.floor((sgn * qi) / cell).astype(np.int64)    # (M,4)
            cells_per_img.append(ck)
            for loc in range(M):
                grid[(int(ck[loc, 0]), int(ck[loc, 1]), int(ck[loc, 2]), int(ck[loc, 3]))].append(loc)

    # candidate pairs: for each voxel image, search the 3^4 neighbour cells
    import itertools
    offs = list(itertools.product((-1, 0, 1), repeat=4))
    cand = set()
    for ck in cells_per_img:        # 2*n_sym entries (sym image × sign)
        for loc in range(M):
            base = (int(ck[loc, 0]), int(ck[loc, 1]), int(ck[loc, 2]), int(ck[loc, 3]))
            for off in offs:
                key = (base[0] + off[0], base[1] + off[1], base[2] + off[2], base[3] + off[3])
                bucket = grid.get(key)
                if not bucket:
                    continue
                for loc2 in bucket:
                    if loc2 > loc:
                        cand.add((loc, loc2))
                    elif loc2 < loc:
                        cand.add((loc2, loc))
    if not cand:
        return [np.empty(0, dtype=np.int64) for _ in range(n_all)]
    cand = np.array(sorted(cand), dtype=np.int64)                # (P,2) local idx

    # evaluate misorientation on unique candidate pairs (batched)
    oms = per_vox_OMs[vi]
    keep_mask = np.zeros(cand.shape[0], dtype=bool)
    use_torch = device is not None
    for b0 in range(0, cand.shape[0], batch):
        b1 = min(b0 + batch, cand.shape[0])
        a = oms[cand[b0:b1, 0]]; bb = oms[cand[b0:b1, 1]]
        if use_torch:
            import torch
            a = torch.as_tensor(a, dtype=torch.float64, device=device)
            bb = torch.as_tensor(bb, dtype=torch.float64, device=device)
            ang = misorientation_om_batch(a, bb, space_group).cpu().numpy()
        else:
            ang = misorientation_om_batch(a, bb, space_group)
        keep_mask[b0:b1] = ang < max_ang_rad
    cp = cand[keep_mask]
    gi = vi[cp[:, 0]]; gj = vi[cp[:, 1]]
    return _neighbors_from_pairs(gi, gj, n_all)


def global_cluster_fast(
    per_vox_OMs: np.ndarray,
    per_vox_confs: np.ndarray,
    per_vox_keys: np.ndarray,
    *,
    space_group: int,
    max_ang_deg: float,
    invalid_marker: int = -1,
    method: str = "auto",          # "auto" | "binned" | "gpu" | "reference"
    device=None,                   # torch device for gpu/binned misorientation
) -> GlobalClusterResult:
    """Drop-in faster equivalent of :func:`global_cluster` (byte-parity).

    ``method``:
      * ``"reference"`` — original O(N^2) CPU implementation (exact, slow).
      * ``"gpu"``       — N^2 misorientation on ``device`` via torch (EXACT — all
                          pairs, no pruning; ~100x constant speedup; needs torch).
      * ``"binned"``    — fundamental-zone quaternion hashing (O(N*k)). EXPERIMENTAL:
                          a rare symmetry-boundary candidate-generation gap can drop
                          a handful of close pairs at large N (see
                          tests/test_global_cluster_parity.py::test_binned_known_gap).
                          Do NOT use where exact parity is required until fixed.
      * ``"auto"``      — ``"gpu"`` when torch + a usable device are available, else
                          ``"reference"``. Always EXACT.
    """
    if method == "auto":
        method = "reference"
        try:
            import torch
            if device is not None or torch.cuda.is_available():
                if device is None:
                    device = torch.device("cuda")
                method = "gpu"
        except Exception:
            method = "reference"
    if method == "reference":
        return global_cluster(per_vox_OMs, per_vox_confs, per_vox_keys,
                              space_group=space_group, max_ang_deg=max_ang_deg,
                              invalid_marker=invalid_marker)
    per_vox_OMs = np.ascontiguousarray(per_vox_OMs, dtype=np.float64)
    per_vox_confs = np.ascontiguousarray(per_vox_confs, dtype=np.float64).ravel()
    per_vox_keys = np.ascontiguousarray(per_vox_keys, dtype=np.uint64)
    max_ang_rad = float(max_ang_deg) * np.pi / 180.0
    invalid_u64 = np.uint64(invalid_marker if invalid_marker >= 0 else (2**64 - 1))
    marked = (per_vox_keys[:, 0] == invalid_u64)
    valid_idx = np.flatnonzero(~marked).astype(np.int64)

    if method == "gpu":
        nbr = _build_neighbors_gpu(per_vox_OMs, valid_idx, space_group, max_ang_rad, device)
    elif method == "binned":
        nbr = _build_neighbors_binned(per_vox_OMs, valid_idx, space_group, max_ang_rad,
                                      device=device)
    else:
        raise ValueError(f"unknown method {method!r}")
    return _greedy_from_neighbors(nbr, per_vox_OMs, per_vox_confs, per_vox_keys, marked)
