"""Volume-budget drop policy — Stage 6.5 of the v4 pipeline.

Combines two complementary diagnostics:

1. **Per-grain NNLS recovery** (``recovery_i = V_NNLS_i / V_naive_i``):
   a local ranking signal. Grains whose NNLS deflation absorbed most of
   their intensity into neighbors are flagged as redundant.

2. **Global volume budget**: the sum ``ΣV_NNLS`` should not exceed the
   illuminated sample volume ``V_sample_true``. If it does, double-counting
   must persist somewhere — drop grains until the total fits.

Algorithm: sort grains by ``recovery`` ascending (worst first), drop
greedily until ``ΣV_kept ≤ V_sample_true``.

If ``ΣV_NNLS ≤ V_sample_true`` already, no drops are issued.

This module gives the v4 pipeline a principled, non-hand-tuned drop list
anchored to a physical reference. See [PR-mode] details in the docstring
of :func:`compute_volume_budget_drops`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class VolumeBudgetDropResult:
    """Result of :func:`compute_volume_budget_drops`.

    Attributes
    ----------
    drop_by_budget : (N,) bool
        ``True`` for grains that should be dropped to satisfy the global
        ΣV ≤ V_sample_true budget.
    recovery : (N,) float
        Per-grain ``V_NNLS / V_naive``. NaN for grains with V_naive=0.
    sum_V_NNLS_um3 : float
        ``Σ V_NNLS_i`` over the input population (µm³).
    sum_V_kept_um3 : float
        ``Σ V_NNLS_i`` over the kept population (≤ ``v_sample_true_um3``).
    n_dropped : int
    v_sample_true_um3 : float
        The reference budget that was used.
    overcounting_ratio : float
        ``sum_V_NNLS / v_sample_true``. > 1 means double-counting was
        present; the drop list brings it back to ≤ 1.
    """

    drop_by_budget: np.ndarray
    recovery: np.ndarray
    sum_V_NNLS_um3: float
    sum_V_kept_um3: float
    n_dropped: int
    v_sample_true_um3: float
    overcounting_ratio: float


def compute_volume_budget_drops(
    *,
    volume_NNLS_um3: np.ndarray,
    volume_naive_um3: np.ndarray,
    v_sample_true_um3: float,
    recovery_floor: float = 0.0,
    tolerance: float = 1.0,
    quality_score: Optional[np.ndarray] = None,
) -> VolumeBudgetDropResult:
    """Issue a drop mask satisfying the global volume budget.

    Parameters
    ----------
    volume_NNLS_um3 : (N,) float
        Per-grain NNLS-attributed volume (µm³). Must be in **physical**
        units (after Vsample correction).
    volume_naive_um3 : (N,) float
        Per-grain naive (legacy) volume (µm³), in the same physical units.
    v_sample_true_um3 : float
        Illuminated sample volume (µm³). The drop loop runs until
        ``ΣV_kept ≤ v_sample_true_um3``.
    recovery_floor : float, default 0.0
        If > 0, immediately drop any grain with ``recovery < recovery_floor``
        regardless of the global budget. The budget loop then continues
        from there. Use this to cull obviously-redundant grains even when
        the global ΣV is already within budget.
    tolerance : float, default 1.0
        Multiplicative slack on the budget (drop until
        ``ΣV ≤ tolerance·V_sample``). Set to e.g. 1.05 to allow a 5 %
        overshoot before dropping.
    """
    Vn = np.asarray(volume_NNLS_um3, dtype=np.float64)
    V0 = np.asarray(volume_naive_um3, dtype=np.float64)
    N = Vn.shape[0]
    if N == 0:
        return VolumeBudgetDropResult(
            drop_by_budget=np.zeros(0, dtype=bool),
            recovery=np.zeros(0, dtype=np.float64),
            sum_V_NNLS_um3=0.0, sum_V_kept_um3=0.0, n_dropped=0,
            v_sample_true_um3=v_sample_true_um3, overcounting_ratio=0.0,
        )

    recovery = np.where(V0 > 0, Vn / np.maximum(V0, 1e-30), np.nan)

    # Stage A: per-grain floor (immediate drops on low recovery)
    drop = np.zeros(N, dtype=bool)
    if recovery_floor > 0:
        drop |= (recovery < recovery_floor) & np.isfinite(recovery)

    sum_total = float(Vn.sum())
    budget = float(v_sample_true_um3) * float(tolerance)
    sum_kept = float(Vn[~drop].sum())

    # Stage B: global-budget greedy drop. Rank ascending by quality (worst
    # first) if quality_score supplied — keeps the BEST grains that fit the
    # budget. Otherwise rank by NNLS recovery (lower recovery = duplicate-ish).
    if sum_kept > budget and budget > 0:
        kept_idx = np.flatnonzero(~drop)
        if quality_score is not None:
            qs = np.asarray(quality_score, dtype=np.float64)
            rank_metric = np.where(np.isfinite(qs[kept_idx]),
                                    qs[kept_idx], -np.inf)
        else:
            rank_metric = np.where(np.isnan(recovery[kept_idx]),
                                    np.inf, recovery[kept_idx])
        order = np.argsort(rank_metric)   # ascending: worst first
        for i in order:
            g = kept_idx[i]
            drop[g] = True
            sum_kept -= float(Vn[g])
            if sum_kept <= budget:
                break

    return VolumeBudgetDropResult(
        drop_by_budget=drop,
        recovery=recovery,
        sum_V_NNLS_um3=sum_total,
        sum_V_kept_um3=sum_kept,
        n_dropped=int(drop.sum()),
        v_sample_true_um3=float(v_sample_true_um3),
        overcounting_ratio=(sum_total / float(v_sample_true_um3))
                           if v_sample_true_um3 > 0 else float("inf"),
    )


@dataclass
class FamilyBudgetDropResult:
    """Result of :func:`compute_volume_budget_drops_family`.

    Attributes
    ----------
    drop_by_budget : (N_grain,) bool
        Per-grain mask; True iff its family was dropped.
    family_id_per_grain : (N_grain,) int64
        Dense family index assigned to each grain. Singleton variants
        (input twin_family_id < 0 or NaN) each get a unique family.
    family_V_um3 : (N_families,) float
        Aggregated V per family (per ``family_aggregation``).
    family_quality : (N_families,) float
        Aggregated quality score per family (max across variants).
    family_n_members : (N_families,) int
    family_drop : (N_families,) bool
    sum_V_family_um3 : float
    sum_V_kept_family_um3 : float
    n_families_dropped : int
    n_grains_dropped : int
    v_sample_true_um3 : float
    overcounting_ratio_family : float
        ``Σ V_family / V_sample_true``. ≤ 1 means no drops needed.
    """

    drop_by_budget: np.ndarray
    family_id_per_grain: np.ndarray
    family_V_um3: np.ndarray
    family_quality: np.ndarray
    family_n_members: np.ndarray
    family_drop: np.ndarray
    sum_V_family_um3: float
    sum_V_kept_family_um3: float
    n_families_dropped: int
    n_grains_dropped: int
    v_sample_true_um3: float
    overcounting_ratio_family: float


def compute_volume_budget_drops_family(
    *,
    volume_NNLS_um3: np.ndarray,
    twin_family_id: np.ndarray,
    quality_score: np.ndarray,
    v_sample_true_um3: float,
    family_aggregation: Literal["max", "mean", "median", "sum"] = "max",
    tolerance: float = 1.0,
) -> FamilyBudgetDropResult:
    """Family-aware drop policy. Twin variants of the same parent grain
    are collapsed into a single budget-entry.

    Why this is needed: the v4 NNLS only deconvolves shared spots WITHIN a
    forward-predict cluster. Twin variants of the same parent grain often
    land in disjoint clusters (their orientations differ by the full twin
    misorientation, separating them at Stage 2), so NNLS never sees that
    V_variant_1 and V_variant_2 are both claims for the same physical
    parent volume. This function collapses them.

    Parameters
    ----------
    volume_NNLS_um3 : (N_grain,) float
        Per-grain NNLS-attributed volume (µm³), in physical units.
    twin_family_id : (N_grain,) int
        Family ID from Stage 5 (twin labeling). Values < 0 or NaN are
        treated as singleton (each grain its own family).
    quality_score : (N_grain,) float
        Per-grain quality score for ranking; higher = better. A sensible
        default is ``Confidence × hkl_coverage / max(sigma_Z_um, 1)``.
        NaN scores rank as -inf (drop first).
    v_sample_true_um3 : float
    family_aggregation : str
        How to aggregate V across variants of the same family. ``"max"``
        (default) treats variants as redundant claims for one parent;
        ``"sum"`` treats them as physically distinct sub-volumes; ``"mean"``
        and ``"median"`` are robust middle-ground choices.
    tolerance : float
        Multiplicative slack on the budget.
    """
    V = np.asarray(volume_NNLS_um3, dtype=np.float64)
    fam_raw = np.asarray(twin_family_id, dtype=np.float64)
    Q = np.asarray(quality_score, dtype=np.float64)
    N = V.shape[0]
    if N == 0:
        return FamilyBudgetDropResult(
            drop_by_budget=np.zeros(0, dtype=bool),
            family_id_per_grain=np.zeros(0, dtype=np.int64),
            family_V_um3=np.zeros(0, dtype=np.float64),
            family_quality=np.zeros(0, dtype=np.float64),
            family_n_members=np.zeros(0, dtype=np.int64),
            family_drop=np.zeros(0, dtype=bool),
            sum_V_family_um3=0.0, sum_V_kept_family_um3=0.0,
            n_families_dropped=0, n_grains_dropped=0,
            v_sample_true_um3=float(v_sample_true_um3),
            overcounting_ratio_family=0.0,
        )

    # Singletons: anything with fam_raw < 0 or NaN → unique family ID.
    is_single = ~np.isfinite(fam_raw) | (fam_raw < 0)
    # Dense family IDs: keep multi-member families with their original IDs
    # (offset to start at 0), then append per-singleton unique families.
    multi_ids = np.unique(fam_raw[~is_single].astype(np.int64))
    id_map: dict[int, int] = {int(g): k for k, g in enumerate(multi_ids)}
    n_multi = len(multi_ids)
    fam_dense = np.empty(N, dtype=np.int64)
    next_single = n_multi
    for i in range(N):
        if is_single[i]:
            fam_dense[i] = next_single
            next_single += 1
        else:
            fam_dense[i] = id_map[int(fam_raw[i])]
    n_families = next_single

    # Per-family aggregation
    family_V = np.zeros(n_families, dtype=np.float64)
    family_Q = np.full(n_families, -np.inf, dtype=np.float64)
    family_n = np.zeros(n_families, dtype=np.int64)

    if family_aggregation == "max":
        # Per-family V = max(V_members)
        for i in range(N):
            f = fam_dense[i]
            if V[i] > family_V[f]:
                family_V[f] = V[i]
            family_n[f] += 1
            q = Q[i] if np.isfinite(Q[i]) else -np.inf
            if q > family_Q[f]:
                family_Q[f] = q
    elif family_aggregation == "sum":
        for i in range(N):
            f = fam_dense[i]
            family_V[f] += V[i]
            family_n[f] += 1
            q = Q[i] if np.isfinite(Q[i]) else -np.inf
            if q > family_Q[f]:
                family_Q[f] = q
    elif family_aggregation in ("mean", "median"):
        # Accumulate into lists, then aggregate
        members: list[list[float]] = [[] for _ in range(n_families)]
        for i in range(N):
            f = fam_dense[i]
            members[f].append(V[i])
            family_n[f] += 1
            q = Q[i] if np.isfinite(Q[i]) else -np.inf
            if q > family_Q[f]:
                family_Q[f] = q
        agg = np.mean if family_aggregation == "mean" else np.median
        for f in range(n_families):
            family_V[f] = float(agg(members[f])) if members[f] else 0.0
    else:
        raise ValueError(f"unknown family_aggregation: {family_aggregation!r}")

    sum_V_total = float(family_V.sum())
    budget = float(v_sample_true_um3) * float(tolerance)
    family_drop = np.zeros(n_families, dtype=bool)
    sum_V_kept = sum_V_total

    if sum_V_kept > budget and budget > 0:
        # Rank families by quality ascending (worst first); ties broken by family idx.
        # NaN/-inf-quality families drop first.
        order = np.argsort(family_Q, kind="stable")
        for f in order:
            family_drop[f] = True
            sum_V_kept -= float(family_V[f])
            if sum_V_kept <= budget:
                break

    drop_per_grain = family_drop[fam_dense]

    return FamilyBudgetDropResult(
        drop_by_budget=drop_per_grain,
        family_id_per_grain=fam_dense,
        family_V_um3=family_V,
        family_quality=family_Q,
        family_n_members=family_n,
        family_drop=family_drop,
        sum_V_family_um3=sum_V_total,
        sum_V_kept_family_um3=sum_V_kept,
        n_families_dropped=int(family_drop.sum()),
        n_grains_dropped=int(drop_per_grain.sum()),
        v_sample_true_um3=float(v_sample_true_um3),
        overcounting_ratio_family=(sum_V_total / float(v_sample_true_um3))
                                   if v_sample_true_um3 > 0 else float("inf"),
    )


# ===========================================================================
# Path 2: Force-keep "distinct" candidates the budget-drop wrongly removed.
# ===========================================================================

import math as _math

def _cubic_24_quats() -> np.ndarray:
    """24 quaternions for cubic point group 432 (FCC/BCC)."""
    ops = [[1, 0, 0, 0]]
    for axis in ([1,0,0], [0,1,0], [0,0,1]):
        for ang in (_math.pi/2, _math.pi, 3*_math.pi/2):
            c = _math.cos(ang/2); s = _math.sin(ang/2)
            ops.append([c, s*axis[0], s*axis[1], s*axis[2]])
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                ax = np.array([sx, sy, sz]) / _math.sqrt(3)
                for ang in (2*_math.pi/3, 4*_math.pi/3):
                    c = _math.cos(ang/2); s = _math.sin(ang/2)
                    ops.append([c, s*ax[0], s*ax[1], s*ax[2]])
    for ax in ([1,1,0], [1,-1,0], [1,0,1], [1,0,-1], [0,1,1], [0,1,-1]):
        ax_n = np.array(ax) / _math.sqrt(2)
        ops.append([0, ax_n[0], ax_n[1], ax_n[2]])
    return np.array(ops, dtype=np.float64)


def _hcp_12_quats() -> np.ndarray:
    """12 quaternions for hexagonal point group 622 (HCP)."""
    ops = []
    for k in range(6):
        a = k * _math.pi / 3.0
        c = _math.cos(a/2); s = _math.sin(a/2)
        ops.append([c, 0, 0, s])
    for k in range(6):
        a = k * _math.pi / 6.0
        ax_n = np.array([_math.cos(a), _math.sin(a), 0])
        ops.append([0, ax_n[0], ax_n[1], 0])
    return np.array(ops, dtype=np.float64)


def _symmetry_quats(space_group: int) -> np.ndarray:
    """Return point-group quaternions appropriate for a given space group."""
    if 195 <= space_group <= 230:
        return _cubic_24_quats()
    if 168 <= space_group <= 194:
        return _hcp_12_quats()
    # Other crystal systems: return identity only (no symmetry)
    return np.array([[1, 0, 0, 0]], dtype=np.float64)


def _om_to_quat(om: np.ndarray) -> np.ndarray:
    """(N, 3, 3) → (N, 4) quaternion via Shepperd."""
    N = om.shape[0]
    out = np.empty((N, 4), dtype=np.float64)
    for k in range(N):
        m = om[k]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S0 = _math.sqrt(tr + 1) * 2
            out[k] = [0.25*S0, (m[2,1]-m[1,2])/S0, (m[0,2]-m[2,0])/S0, (m[1,0]-m[0,1])/S0]
        elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            S0 = _math.sqrt(1 + m[0,0] - m[1,1] - m[2,2]) * 2
            out[k] = [(m[2,1]-m[1,2])/S0, 0.25*S0, (m[0,1]+m[1,0])/S0, (m[0,2]+m[2,0])/S0]
        elif m[1,1] > m[2,2]:
            S0 = _math.sqrt(1 + m[1,1] - m[0,0] - m[2,2]) * 2
            out[k] = [(m[0,2]-m[2,0])/S0, (m[0,1]+m[1,0])/S0, 0.25*S0, (m[1,2]+m[2,1])/S0]
        else:
            S0 = _math.sqrt(1 + m[2,2] - m[0,0] - m[1,1]) * 2
            out[k] = [(m[1,0]-m[0,1])/S0, (m[0,2]+m[2,0])/S0, (m[1,2]+m[2,1])/S0, 0.25*S0]
        out[k] /= np.linalg.norm(out[k])
    return out


def _qmul_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


@dataclass
class ForceKeepDistinctResult:
    """Result of :func:`compute_force_keep_distinct`."""

    new_drop_mask: np.ndarray         # (N,) bool — updated drop mask
    force_kept_mask: np.ndarray       # (N,) bool — which grains were force-kept
    min_misori_deg: np.ndarray        # (N,) float — per-grain min misori to nearest kept
    sigma_distance: np.ndarray        # (N,) float — σ-distance to that nearest kept
    n_force_kept: int                 # number of grains recovered from drop


def compute_force_keep_distinct(
    *,
    grain_OMs: np.ndarray,                  # (N, 3, 3)
    grain_positions_um: np.ndarray,         # (N, 3)
    grain_sigma_xyz_um: np.ndarray,         # (N, 3) — per-axis σ; NaN-fallback caller's job
    drop_mask: np.ndarray,                  # (N,) bool — current drop state
    space_group: int,
    misori_deg_threshold: float = 1.0,
    sigma_distance_threshold: float = 3.0,
) -> ForceKeepDistinctResult:
    """For each dropped grain, find the kept grain it's closest to in (FZ-
    symmetry-aware) orientation space. If that nearest-by-OM kept is also
    far in σ-normalised position (≥ ``sigma_distance_threshold``), AND the
    misorientation is itself ≥ ``misori_deg_threshold``, the dropped grain
    is **distinct** — force-keep it.

    The default threshold 1° matches typical FF-HEDM peak-search resolution
    (3-4 px detector + ~3 ω frames → ~0.5–1° OM-distinguishability).

    σ-distance uses combined per-axis σ in quadrature.
    """
    N = grain_OMs.shape[0]
    if N == 0 or drop_mask.sum() == 0 or (~drop_mask).sum() == 0:
        return ForceKeepDistinctResult(
            new_drop_mask=drop_mask.copy(),
            force_kept_mask=np.zeros(N, dtype=bool),
            min_misori_deg=np.zeros(N, dtype=np.float64),
            sigma_distance=np.zeros(N, dtype=np.float64),
            n_force_kept=0,
        )
    sym = _symmetry_quats(int(space_group))
    quats = _om_to_quat(grain_OMs)
    kept_idx = np.flatnonzero(~drop_mask)
    drop_idx = np.flatnonzero(drop_mask)

    # Build the symmetry-replicated kept-quaternion KDTree (±antipodal).
    q_kept = quats[kept_idx]
    SQ_kept = _qmul_batch(sym[:, None, :], q_kept[None, :, :])     # (S, K, 4)
    SQ_kept_flat = SQ_kept.reshape(-1, 4)                          # (S·K, 4)
    sym_replicate_idx = np.tile(np.arange(len(kept_idx)), len(sym))
    SQ_flat_pm = np.concatenate([SQ_kept_flat, -SQ_kept_flat], axis=0)
    sym_replicate_idx_pm = np.tile(sym_replicate_idx, 2)
    try:
        from scipy.spatial import cKDTree as _KDTree
        tree = _KDTree(SQ_flat_pm)
        d4, nearest_in_pm = tree.query(quats[drop_idx], k=1)
    except ImportError:
        # Fallback: brute-force (slow but works)
        dot_all = np.abs(quats[drop_idx] @ SQ_flat_pm.T)            # (D, 2·S·K)
        nearest_in_pm = dot_all.argmax(axis=1)
        d4 = np.sqrt(np.maximum(0, 2 - 2 * dot_all[np.arange(len(drop_idx)), nearest_in_pm]))

    inner = np.clip(np.abs(1.0 - d4**2 / 2), 0, 1)
    min_misori_deg_drop = np.degrees(2 * np.arccos(inner))

    # Nearest kept grain index (in 0..N-1 space)
    nearest_k_idx = kept_idx[sym_replicate_idx_pm[nearest_in_pm]]
    delta_pos = grain_positions_um[drop_idx] - grain_positions_um[nearest_k_idx]
    sig_comb = np.sqrt(grain_sigma_xyz_um[drop_idx]**2 + grain_sigma_xyz_um[nearest_k_idx]**2)
    sig_comb = np.maximum(sig_comb, 1e-3)
    sigma_dist_drop = np.sqrt(np.sum((delta_pos / sig_comb)**2, axis=-1))

    is_distinct = (
        (min_misori_deg_drop >= misori_deg_threshold)
        & (sigma_dist_drop   >= sigma_distance_threshold)
    )
    force_kept_mask = np.zeros(N, dtype=bool)
    force_kept_mask[drop_idx[is_distinct]] = True
    new_drop_mask = drop_mask.copy()
    new_drop_mask[drop_idx[is_distinct]] = False

    # Per-grain arrays (full size N, filled with NaN for kept; values for dropped)
    misori_full = np.full(N, np.nan, dtype=np.float64)
    sigdist_full = np.full(N, np.nan, dtype=np.float64)
    misori_full[drop_idx] = min_misori_deg_drop
    sigdist_full[drop_idx] = sigma_dist_drop

    return ForceKeepDistinctResult(
        new_drop_mask=new_drop_mask,
        force_kept_mask=force_kept_mask,
        min_misori_deg=misori_full,
        sigma_distance=sigdist_full,
        n_force_kept=int(force_kept_mask.sum()),
    )


# ===========================================================================
# Path 3: Orphan-greedy reclaim. Recover dropped grains whose spot-sets
# uniquely cover spots not in the current kept population.
# ===========================================================================

@dataclass
class OrphanReclaimResult:
    """Result of :func:`compute_orphan_greedy_reclaim`."""
    new_drop_mask: np.ndarray          # (N,) bool
    reclaimed_mask: np.ndarray         # (N,) bool — grains added back via reclaim
    n_reclaimed: int                   # count
    n_orphan_spots_before: int
    n_orphan_spots_after: int
    n_unique_spots_per_reclaim: np.ndarray   # debug: unique spots each reclaim contributed


def compute_orphan_greedy_reclaim(
    *,
    drop_mask: np.ndarray,
    spot_sets: list,                # length N, each is a set[int] of SpotIDs
    quality_score: np.ndarray,      # (N,) per-grain quality
    min_unique_spots: int = 5,
) -> OrphanReclaimResult:
    """Greedily reclaim DROPPED grains that uniquely cover spots not in the
    current kept set.

    Iterate dropped grains in **descending quality**. For each, compute how
    many spots it claims that are NOT yet covered by any kept grain. If that
    unique-contribution exceeds ``min_unique_spots``, reclaim the grain.

    This reduces the orphan rate of the kept population at the cost of
    keeping lower-quality grains. Use ``min_unique_spots`` to tune the
    permissiveness (higher = more conservative).
    """
    N = len(drop_mask)
    if N == 0:
        empty = np.zeros(0, dtype=bool); empty_int = np.zeros(0, dtype=np.int64)
        return OrphanReclaimResult(
            new_drop_mask=empty, reclaimed_mask=empty,
            n_reclaimed=0, n_orphan_spots_before=0, n_orphan_spots_after=0,
            n_unique_spots_per_reclaim=empty_int,
        )
    # Currently covered spots
    covered: set = set()
    for i in range(N):
        if not drop_mask[i]:
            covered.update(spot_sets[i])
    # All spots claimed by ANY candidate (kept or dropped)
    all_claimed: set = set()
    for s in spot_sets:
        all_claimed.update(s)
    n_orphan_before = len(all_claimed - covered)

    dropped_idx = np.flatnonzero(drop_mask)
    if len(dropped_idx) == 0:
        return OrphanReclaimResult(
            new_drop_mask=drop_mask.copy(),
            reclaimed_mask=np.zeros(N, dtype=bool),
            n_reclaimed=0,
            n_orphan_spots_before=n_orphan_before, n_orphan_spots_after=n_orphan_before,
            n_unique_spots_per_reclaim=np.zeros(0, dtype=np.int64),
        )
    # Sort dropped indices by descending quality
    Q_dropped = quality_score[dropped_idx]
    order = np.argsort(-Q_dropped)

    new_drop = drop_mask.copy()
    reclaimed = np.zeros(N, dtype=bool)
    unique_per_reclaim = []
    for i in order:
        g = int(dropped_idx[i])
        spots_g = spot_sets[g]
        if not spots_g:
            continue
        unique_contrib = spots_g - covered
        if len(unique_contrib) >= min_unique_spots:
            new_drop[g] = False
            reclaimed[g] = True
            covered.update(spots_g)
            unique_per_reclaim.append(len(unique_contrib))
    n_orphan_after = len(all_claimed - covered)

    return OrphanReclaimResult(
        new_drop_mask=new_drop,
        reclaimed_mask=reclaimed,
        n_reclaimed=int(reclaimed.sum()),
        n_orphan_spots_before=n_orphan_before,
        n_orphan_spots_after=n_orphan_after,
        n_unique_spots_per_reclaim=np.asarray(unique_per_reclaim, dtype=np.int64),
    )
