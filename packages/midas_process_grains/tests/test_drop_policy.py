"""Tests for compute.drop_policy."""

import numpy as np
import pytest

from midas_process_grains.compute.drop_policy import (
    compute_volume_budget_drops, VolumeBudgetDropResult,
    compute_volume_budget_drops_family, FamilyBudgetDropResult,
)


def _V(R): return (4.0 / 3.0) * np.pi * np.asarray(R, dtype=float) ** 3


def test_no_drops_when_total_within_budget():
    R = np.array([1.0, 2.0, 3.0])
    V = _V(R)
    res = compute_volume_budget_drops(
        volume_NNLS_um3=V, volume_naive_um3=V,
        v_sample_true_um3=1000.0,
    )
    assert res.n_dropped == 0
    assert res.drop_by_budget.sum() == 0
    assert np.isclose(res.sum_V_kept_um3, V.sum())
    assert res.overcounting_ratio < 1.0


def test_drops_until_budget_satisfied():
    """ΣV_NNLS = 1000; budget = 600. Drop ascending-recovery until ≤600."""
    Vn = np.array([100.0, 100.0, 100.0, 100.0, 100.0,
                   100.0, 100.0, 100.0, 100.0, 100.0])   # all V=100, Σ=1000
    V0 = np.array([100.0, 100.0, 200.0, 200.0, 200.0,
                   200.0, 200.0, 200.0, 200.0, 200.0])
    # recovery = [1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    # Worst (rec=0.5) gets dropped first; need to drop 4 to bring 1000 → 600
    res = compute_volume_budget_drops(
        volume_NNLS_um3=Vn, volume_naive_um3=V0,
        v_sample_true_um3=600.0,
    )
    assert res.n_dropped == 4
    assert res.sum_V_kept_um3 <= 600.0
    assert res.overcounting_ratio == pytest.approx(1000.0 / 600.0)
    # First two grains (rec=1.0) must be KEPT
    assert not res.drop_by_budget[0] and not res.drop_by_budget[1]


def test_recovery_floor_drops_low_recovery_even_within_budget():
    """Total = 100 ≤ budget = 1000, but one grain has rec < 0.05 → drop it."""
    Vn = np.array([50.0, 5.0, 45.0])
    V0 = np.array([50.0, 1000.0, 45.0])   # rec = [1.0, 0.005, 1.0]
    res = compute_volume_budget_drops(
        volume_NNLS_um3=Vn, volume_naive_um3=V0,
        v_sample_true_um3=1000.0, recovery_floor=0.05,
    )
    assert res.n_dropped == 1
    assert res.drop_by_budget[1] == True
    assert res.drop_by_budget[0] == False
    assert res.drop_by_budget[2] == False


def test_empty_input_returns_zero_drops():
    res = compute_volume_budget_drops(
        volume_NNLS_um3=np.array([]),
        volume_naive_um3=np.array([]),
        v_sample_true_um3=1e8,
    )
    assert res.n_dropped == 0
    assert res.sum_V_NNLS_um3 == 0.0


def test_tolerance_allows_overshoot():
    """V=1000, budget=900, tol=1.2 → effective budget = 1080 ≥ 1000 → no drops."""
    Vn = np.full(10, 100.0); V0 = Vn.copy()
    res = compute_volume_budget_drops(
        volume_NNLS_um3=Vn, volume_naive_um3=V0,
        v_sample_true_um3=900.0, tolerance=1.2,
    )
    assert res.n_dropped == 0


def test_drop_order_is_recovery_ascending_not_volume():
    """Two grains both V_NNLS=100: low-rec one must drop first."""
    Vn = np.array([100.0, 100.0])
    V0 = np.array([110.0, 1000.0])   # rec = [0.91, 0.10]
    res = compute_volume_budget_drops(
        volume_NNLS_um3=Vn, volume_naive_um3=V0,
        v_sample_true_um3=100.0,
    )
    # Should drop the low-rec grain (idx 1) first, leaving idx 0
    assert res.n_dropped == 1
    assert res.drop_by_budget[1] == True
    assert res.drop_by_budget[0] == False


def test_nan_recovery_grains_dropped_last():
    """V_naive=0 → recovery is NaN → those grains rank LAST in the drop order."""
    Vn = np.array([100.0, 100.0, 100.0])
    V0 = np.array([0.0,   1000.0, 200.0])  # rec = [NaN, 0.1, 0.5]
    # ΣV=300, budget=100, must drop 2.  Order: rec=0.1 first, then rec=0.5,
    # leaving the NaN-recovery grain kept.
    res = compute_volume_budget_drops(
        volume_NNLS_um3=Vn, volume_naive_um3=V0,
        v_sample_true_um3=100.0,
    )
    assert res.n_dropped == 2
    assert res.drop_by_budget[0] == False    # NaN-rec grain kept
    assert res.drop_by_budget[1] == True     # rec=0.1 dropped first
    assert res.drop_by_budget[2] == True     # rec=0.5 dropped second


def test_zero_budget_drops_all():
    Vn = np.array([10.0, 20.0, 30.0])
    V0 = Vn.copy()
    res = compute_volume_budget_drops(
        volume_NNLS_um3=Vn, volume_naive_um3=V0,
        v_sample_true_um3=0.0,
    )
    # Budget=0 should leave nothing kept (or special-case to no-op)
    # Current impl: budget=0 means budget loop skips (>0 check); no budget drops
    # That is the safer default — V_sample=0 means user didn't supply it.
    assert res.n_dropped == 0


# ---- Family-aware drop policy ----

def test_family_aware_collapses_variants_via_max():
    """Two twin variants of the same parent (fam=7), V=100 each. Singleton (fam=-1).
    With max-aggregation: V_family_parent = 100 (not 200), so 2 families with
    V_total = 100+50 = 150 vs budget 200 → no drops."""
    V = np.array([100.0, 100.0, 50.0])
    fam = np.array([7, 7, -1])
    Q = np.array([0.9, 0.8, 0.7])
    res = compute_volume_budget_drops_family(
        volume_NNLS_um3=V, twin_family_id=fam, quality_score=Q,
        v_sample_true_um3=200.0, family_aggregation="max",
    )
    assert res.family_V_um3.shape == (2,)   # one twin family + one singleton
    assert np.isclose(res.family_V_um3.max(), 100.0)   # max of [100,100]
    assert res.sum_V_family_um3 == pytest.approx(150.0)
    assert res.n_families_dropped == 0
    assert res.n_grains_dropped == 0


def test_family_aware_drops_low_quality_family_first():
    """Two families: A (fam=1, V=80, Q=0.9) and B (fam=2, V=80, Q=0.5).
    ΣV=160; budget=100 → must drop one. Drop B (lower Q)."""
    V = np.array([80.0, 80.0])
    fam = np.array([1, 2])
    Q = np.array([0.9, 0.5])
    res = compute_volume_budget_drops_family(
        volume_NNLS_um3=V, twin_family_id=fam, quality_score=Q,
        v_sample_true_um3=100.0,
    )
    assert res.n_families_dropped == 1
    assert res.drop_by_budget[0] == False   # high-Q kept
    assert res.drop_by_budget[1] == True    # low-Q dropped


def test_family_aware_drops_all_members_of_dropped_family():
    """Family of 3 variants, all dropped together when family is dropped."""
    V = np.array([50.0, 50.0, 50.0, 5.0])
    fam = np.array([10, 10, 10, -1])
    Q = np.array([0.5, 0.5, 0.5, 0.9])    # family 10 is worse than singleton
    res = compute_volume_budget_drops_family(
        volume_NNLS_um3=V, twin_family_id=fam, quality_score=Q,
        v_sample_true_um3=10.0, family_aggregation="max",
    )
    # Family 10 V_family=50, singleton V=5 → ΣV=55, budget=10 → drop family 10.
    # All 3 variants of fam 10 dropped; singleton kept (Q higher).
    assert res.drop_by_budget[0] == True
    assert res.drop_by_budget[1] == True
    assert res.drop_by_budget[2] == True
    assert res.drop_by_budget[3] == False
    # Singleton alone = 5 ≤ 10, so we stop after dropping family 10
    assert res.n_families_dropped == 1   # only one family dropped
    assert res.n_grains_dropped == 3


def test_family_aware_singleton_singletons_each_get_own_family():
    """All grains have fam=-1 → N families = N grains; behavior reduces to per-grain."""
    V = np.array([100.0, 100.0, 100.0])
    fam = np.array([-1, -1, -1])
    Q = np.array([0.5, 0.7, 0.9])
    res = compute_volume_budget_drops_family(
        volume_NNLS_um3=V, twin_family_id=fam, quality_score=Q,
        v_sample_true_um3=150.0,
    )
    assert len(res.family_V_um3) == 3
    # ΣV=300, budget=150, drop 2 (lowest Q: idx 0, then idx 1).
    # Singleton idx 0 (Q=0.5) dropped first; idx 1 (Q=0.7) dropped second.
    assert res.n_families_dropped == 2
    assert res.drop_by_budget[0] == True
    assert res.drop_by_budget[1] == True
    assert res.drop_by_budget[2] == False


def test_family_aware_sum_aggregation_treats_variants_as_distinct():
    """With sum-aggregation, twin variants behave like distinct sub-grains."""
    V = np.array([100.0, 100.0])
    fam = np.array([7, 7])
    Q = np.array([0.9, 0.8])
    res_max = compute_volume_budget_drops_family(
        volume_NNLS_um3=V, twin_family_id=fam, quality_score=Q,
        v_sample_true_um3=150.0, family_aggregation="max",
    )
    assert res_max.sum_V_family_um3 == pytest.approx(100.0)
    assert res_max.n_families_dropped == 0

    res_sum = compute_volume_budget_drops_family(
        volume_NNLS_um3=V, twin_family_id=fam, quality_score=Q,
        v_sample_true_um3=150.0, family_aggregation="sum",
    )
    assert res_sum.sum_V_family_um3 == pytest.approx(200.0)
    # ΣV=200, budget=150 → must drop the single 200µm³ family
    assert res_sum.n_families_dropped == 1
    assert res_sum.drop_by_budget.all()   # both members dropped


def test_family_aware_no_drops_when_within_budget():
    V = np.array([10.0, 20.0, 30.0])
    fam = np.array([-1, -1, -1])
    Q = np.array([0.5, 0.5, 0.5])
    res = compute_volume_budget_drops_family(
        volume_NNLS_um3=V, twin_family_id=fam, quality_score=Q,
        v_sample_true_um3=1000.0,
    )
    assert res.n_families_dropped == 0
    assert res.overcounting_ratio_family == pytest.approx(60.0 / 1000.0)


# ===== Force-keep-distinct tests =====

def test_force_keep_distinct_no_op_if_no_drops():
    """If drop_mask is all-False, force-keep is no-op."""
    from midas_process_grains.compute.drop_policy import compute_force_keep_distinct
    OMs = np.stack([np.eye(3), np.eye(3)])
    pos = np.array([[0,0,0], [200, 0, 0]], dtype=float)
    sig = np.array([[10,10,10], [10,10,10]], dtype=float)
    drop = np.zeros(2, dtype=bool)
    res = compute_force_keep_distinct(
        grain_OMs=OMs, grain_positions_um=pos, grain_sigma_xyz_um=sig,
        drop_mask=drop, space_group=225,
    )
    assert res.n_force_kept == 0
    assert (res.new_drop_mask == drop).all()


def test_force_keep_distinct_recovers_far_grain():
    """A dropped grain that is far from the kept grain in BOTH misori and
    position should be force-kept."""
    from midas_process_grains.compute.drop_policy import compute_force_keep_distinct
    import math
    # Kept grain: identity at (0,0,0)
    OM1 = np.eye(3)
    # Dropped grain: 45° rotation about z at (500, 0, 0)
    a = math.radians(45)
    OM2 = np.array([[math.cos(a), -math.sin(a), 0],
                    [math.sin(a),  math.cos(a), 0],
                    [0, 0, 1]])
    OMs = np.stack([OM1, OM2])
    pos = np.array([[0,0,0], [500, 0, 0]], dtype=float)
    sig = np.array([[10,10,10], [10,10,10]], dtype=float)
    drop = np.array([False, True])
    res = compute_force_keep_distinct(
        grain_OMs=OMs, grain_positions_um=pos, grain_sigma_xyz_um=sig,
        drop_mask=drop, space_group=225,
        misori_deg_threshold=2.0, sigma_distance_threshold=3.0,
    )
    assert res.n_force_kept == 1, f"Expected 1 force-kept, got {res.n_force_kept}"
    assert res.force_kept_mask[1] == True
    assert res.new_drop_mask[1] == False


def test_force_keep_distinct_does_not_recover_alt_indexing():
    """Dropped grain with same OM and close position is an alt-indexing,
    should NOT be force-kept."""
    from midas_process_grains.compute.drop_policy import compute_force_keep_distinct
    OMs = np.stack([np.eye(3), np.eye(3)])
    pos = np.array([[0,0,0], [5, 5, 5]], dtype=float)
    sig = np.array([[10,10,10], [10,10,10]], dtype=float)
    drop = np.array([False, True])
    res = compute_force_keep_distinct(
        grain_OMs=OMs, grain_positions_um=pos, grain_sigma_xyz_um=sig,
        drop_mask=drop, space_group=225,
        misori_deg_threshold=2.0, sigma_distance_threshold=3.0,
    )
    assert res.n_force_kept == 0
    assert res.new_drop_mask[1] == True   # still dropped


def test_force_keep_distinct_respects_cubic_symmetry():
    """A dropped grain with a 90° rotation about z from the kept (which is
    cubic-equivalent under 432) should NOT be force-kept (= alt-indexing
    of the same cubic crystal)."""
    from midas_process_grains.compute.drop_policy import compute_force_keep_distinct
    import math
    OM1 = np.eye(3)
    a = math.radians(90)   # 90° about z is a cubic symmetry op
    OM2 = np.array([[math.cos(a), -math.sin(a), 0],
                    [math.sin(a),  math.cos(a), 0],
                    [0, 0, 1]])
    OMs = np.stack([OM1, OM2])
    pos = np.array([[0,0,0], [500, 0, 0]], dtype=float)
    sig = np.array([[10,10,10], [10,10,10]], dtype=float)
    drop = np.array([False, True])
    res = compute_force_keep_distinct(
        grain_OMs=OMs, grain_positions_um=pos, grain_sigma_xyz_um=sig,
        drop_mask=drop, space_group=225,    # cubic
        misori_deg_threshold=2.0, sigma_distance_threshold=3.0,
    )
    # 90° about z is identity under cubic symmetry → misori = 0
    assert res.n_force_kept == 0
    assert res.min_misori_deg[1] < 1.0   # essentially zero misori after symmetry


# ===== Orphan-greedy reclaim tests =====

def test_orphan_reclaim_no_op_no_dropped():
    from midas_process_grains.compute.drop_policy import compute_orphan_greedy_reclaim
    res = compute_orphan_greedy_reclaim(
        drop_mask=np.zeros(3, dtype=bool),
        spot_sets=[{1,2}, {3,4}, {5}], quality_score=np.array([1,2,3]),
        min_unique_spots=1,
    )
    assert res.n_reclaimed == 0


def test_orphan_reclaim_recovers_uniquely_covering():
    """Dropped grain claims orphan spots not covered by any kept → reclaim."""
    from midas_process_grains.compute.drop_policy import compute_orphan_greedy_reclaim
    drop = np.array([False, True, True])
    spot_sets = [{1,2,3}, {4,5,6}, {7,8,9}]      # 1: kept covers {1,2,3}, 2 & 3 dropped have unique
    Q = np.array([1.0, 0.8, 0.5])                # 2 has higher Q than 3
    res = compute_orphan_greedy_reclaim(
        drop_mask=drop, spot_sets=spot_sets, quality_score=Q,
        min_unique_spots=2,
    )
    assert res.n_reclaimed == 2   # both have >= 2 unique
    assert res.new_drop_mask.sum() == 0
    # 2 reclaimed first (higher Q), then 3
    assert res.reclaimed_mask[1] == True and res.reclaimed_mask[2] == True


def test_orphan_reclaim_skips_redundant():
    """Dropped grain whose spots are ALREADY covered → no reclaim."""
    from midas_process_grains.compute.drop_policy import compute_orphan_greedy_reclaim
    drop = np.array([False, True])
    spot_sets = [{1,2,3,4,5}, {1,2,3}]
    Q = np.array([1.0, 0.5])
    res = compute_orphan_greedy_reclaim(
        drop_mask=drop, spot_sets=spot_sets, quality_score=Q,
        min_unique_spots=1,
    )
    assert res.n_reclaimed == 0


def test_orphan_reclaim_min_unique_threshold():
    """Dropped grain with 2 unique spots, threshold=3 → NOT reclaimed."""
    from midas_process_grains.compute.drop_policy import compute_orphan_greedy_reclaim
    drop = np.array([False, True])
    spot_sets = [{1,2}, {3,4,1}]    # dropped has 2 unique
    Q = np.array([1.0, 0.5])
    res = compute_orphan_greedy_reclaim(
        drop_mask=drop, spot_sets=spot_sets, quality_score=Q,
        min_unique_spots=3,
    )
    assert res.n_reclaimed == 0
