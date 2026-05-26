"""Stage-3 tests: physics-bounded split of synthetic over-merged clusters."""
from __future__ import annotations

import numpy as np

from midas_process_grains.compute.cluster_physics import (
    split_clusters_by_physics, PhysicsClusterResult, _hkl_key,
)


def test_clean_cluster_passes_through_unchanged():
    """A cluster of 6 candidates with 6 distinct {002} variants should
    emit as 1 grain with no splits."""
    # 6 candidates, all in Pass-1 cluster 0, distinct (h, k, l)
    h = np.array([2, -2, 0,  0, 0,  0], dtype=np.int8)
    k = np.array([0,  0, 2, -2, 0,  0], dtype=np.int8)
    l = np.array([0,  0, 0,  0, 2, -2], dtype=np.int8)
    p1 = np.zeros(6, dtype=np.int64)
    alive = np.ones(6, dtype=bool)
    pos = np.tile(np.array([100., 200., 300.]), (6, 1))
    spots = [set([i*10 + j for j in range(50)]) for i in range(6)]
    om = np.tile(np.eye(3), (6, 1, 1))

    res = split_clusters_by_physics(
        pass1_cluster_id=p1, seed_h=h, seed_k=k, seed_l=l, seed_alive=alive,
        positions=pos, spot_sets=spots, om_fz=om,
    )
    assert res.n_final_grains == 1
    assert res.grain_n_candidates[0] == 6
    assert res.grain_n_unique_hkls[0] == 6
    assert res.grain_hkl_dup_count[0] == 0
    assert res.grain_splits_emerged[0] == 0
    # All 6 candidates land in grain 0
    assert (res.final_grain_id == 0).sum() == 6


def test_over_merged_cluster_splits_into_two():
    """A cluster of 4 candidates with (2,0,0) appearing twice and
    (0,2,0) appearing twice should split into 2 sub-clusters."""
    # Two grains, each with 2 variants
    # Grain A: (2,0,0) and (0,2,0) at position (100, 0, 0)
    # Grain B: (2,0,0) and (0,2,0) at position (500, 0, 0)
    h = np.array([2, 2, 0, 0], dtype=np.int8)
    k = np.array([0, 0, 2, 2], dtype=np.int8)
    l = np.array([0, 0, 0, 0], dtype=np.int8)
    p1 = np.zeros(4, dtype=np.int64)
    alive = np.ones(4, dtype=bool)
    pos = np.array([
        [100., 0., 0.],  # A's (2,0,0)
        [500., 0., 0.],  # B's (2,0,0)
        [102., 0., 0.],  # A's (0,2,0) — close to grain A
        [498., 0., 0.],  # B's (0,2,0) — close to grain B
    ])
    # Spot sets: A's pair share spots, B's pair share spots
    spots = [
        set(range(100, 150)),   # A's spots
        set(range(500, 550)),   # B's spots
        set(range(100, 150)) | set(range(150, 175)),  # mostly A's
        set(range(500, 550)) | set(range(550, 575)),  # mostly B's
    ]
    om = np.tile(np.eye(3), (4, 1, 1))

    res = split_clusters_by_physics(
        pass1_cluster_id=p1, seed_h=h, seed_k=k, seed_l=l, seed_alive=alive,
        positions=pos, spot_sets=spots, om_fz=om,
    )
    assert res.n_final_grains == 2, f"expected 2 grains, got {res.n_final_grains}"

    # Each final grain should have unique hkls (no duplicates within)
    for g in range(2):
        in_g = np.flatnonzero(res.final_grain_id == g)
        assert len(in_g) == 2
        keys = _hkl_key(h[in_g], k[in_g], l[in_g])
        assert len(np.unique(keys)) == 2

    # The split was correctly attributed: candidates 0,2 → one grain; 1,3 → other
    g0 = res.final_grain_id[0]
    g1 = res.final_grain_id[1]
    g2 = res.final_grain_id[2]
    g3 = res.final_grain_id[3]
    assert g0 != g1, "A's (2,0,0) and B's (2,0,0) should be in different grains"
    assert g0 == g2, "A's two candidates should be in the same grain"
    assert g1 == g3, "B's two candidates should be in the same grain"


def test_splits_emerged_flag_set_on_split_products():
    """The grain emitted from a split should have splits_emerged_from > 0."""
    h = np.array([2, 2], dtype=np.int8)
    k = np.array([0, 0], dtype=np.int8)
    l = np.array([0, 0], dtype=np.int8)
    p1 = np.zeros(2, dtype=np.int64)
    alive = np.ones(2, dtype=bool)
    pos = np.array([[0., 0., 0.], [1000., 0., 0.]])
    spots = [set(range(0, 10)), set(range(20, 30))]
    om = np.tile(np.eye(3), (2, 1, 1))
    res = split_clusters_by_physics(
        pass1_cluster_id=p1, seed_h=h, seed_k=k, seed_l=l, seed_alive=alive,
        positions=pos, spot_sets=spots, om_fz=om,
    )
    assert res.n_final_grains == 2
    assert (res.grain_splits_emerged == 1).all()


def test_dead_seeds_are_dropped():
    """Candidates with seed_alive=False should not appear in any grain."""
    h = np.array([2, 2], dtype=np.int8)
    k = np.array([0, 0], dtype=np.int8)
    l = np.array([0, 0], dtype=np.int8)
    p1 = np.zeros(2, dtype=np.int64)
    alive = np.array([True, False])
    pos = np.array([[0., 0., 0.], [1.0, 0., 0.]])
    spots = [set(range(0, 5)), set(range(5, 10))]
    om = np.tile(np.eye(3), (2, 1, 1))
    res = split_clusters_by_physics(
        pass1_cluster_id=p1, seed_h=h, seed_k=k, seed_l=l, seed_alive=alive,
        positions=pos, spot_sets=spots, om_fz=om,
    )
    assert res.n_final_grains == 1
    assert res.final_grain_id[0] == 0
    assert res.final_grain_id[1] == -1
