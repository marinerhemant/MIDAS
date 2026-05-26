"""Unit tests for compute.forward_predict_merge."""

from __future__ import annotations

import math
import numpy as np
import pytest

from midas_process_grains.compute.forward_predict_merge import (
    ForwardPredictAttribution,
    ForwardPredictGraph,
    build_forward_predict_graph,
    forward_predict_merge_components,
    select_k_agree_auto,
    split_components_by_om_spread,
)


def _make_attribution(rows):
    """Helper: build a ForwardPredictAttribution from a list of (cand, spot, variant)."""
    if not rows:
        return ForwardPredictAttribution(
            cand_idx=np.zeros(0, dtype=np.int32),
            spot_idx=np.zeros(0, dtype=np.int32),
            variant_idx=np.zeros(0, dtype=np.int8),
            spot_id=np.zeros(0, dtype=np.int32),
            n_alive=0, n_variants=0, snap_rate=0.0,
        )
    arr = np.asarray(rows, dtype=np.int64)
    return ForwardPredictAttribution(
        cand_idx=arr[:, 0].astype(np.int32),
        spot_idx=arr[:, 1].astype(np.int32),
        variant_idx=arr[:, 2].astype(np.int8),
        spot_id=arr[:, 1].astype(np.int32),  # use spot_idx for spot_id in tests
        n_alive=int(arr[:, 0].max() + 1),
        n_variants=int(arr[:, 2].max() + 1),
        snap_rate=1.0,
    )


# ---------------------------------------------------------------------------
# build_forward_predict_graph
# ---------------------------------------------------------------------------


def test_graph_empty_attribution_returns_empty_graph():
    attrib = _make_attribution([])
    g = build_forward_predict_graph(attrib)
    assert g.pair_a.size == 0
    assert g.pair_b.size == 0


def test_graph_single_attribution_has_no_pairs():
    # One candidate claims one spot — no pair to score
    attrib = _make_attribution([[0, 100, 5]])
    g = build_forward_predict_graph(attrib)
    assert g.pair_a.size == 0


def test_graph_two_cands_same_variant_agrees():
    # cands 0 and 1 both attribute spot 100 to variant 5 → agree=1
    attrib = _make_attribution([
        [0, 100, 5],
        [1, 100, 5],
    ])
    g = build_forward_predict_graph(attrib)
    assert g.pair_a.size == 1
    assert g.agree_count[0] == 1
    assert g.disagree_count[0] == 0


def test_graph_two_cands_different_variants_disagree():
    attrib = _make_attribution([
        [0, 100, 5],
        [1, 100, 7],
    ])
    g = build_forward_predict_graph(attrib)
    assert g.agree_count[0] == 0
    assert g.disagree_count[0] == 1


def test_graph_pair_keys_are_canonical_unordered():
    # cand_idx larger appears first; key should still be (0, 1)
    attrib = _make_attribution([
        [3, 100, 5],
        [1, 100, 5],
    ])
    g = build_forward_predict_graph(attrib)
    assert g.pair_a[0] == 1
    assert g.pair_b[0] == 3


def test_graph_counts_agree_and_disagree_across_spots():
    # cands 0,1: agree on spot 100 (variant 5), disagree on spot 101 (5 vs 6)
    attrib = _make_attribution([
        [0, 100, 5], [1, 100, 5],
        [0, 101, 5], [1, 101, 6],
    ])
    g = build_forward_predict_graph(attrib)
    assert g.pair_a.size == 1
    assert g.agree_count[0] == 1
    assert g.disagree_count[0] == 1


# ---------------------------------------------------------------------------
# forward_predict_merge_components
# ---------------------------------------------------------------------------


def test_components_no_edges_returns_singletons():
    graph = ForwardPredictGraph(
        pair_a=np.zeros(0, dtype=np.int32),
        pair_b=np.zeros(0, dtype=np.int32),
        agree_count=np.zeros(0, dtype=np.int16),
        disagree_count=np.zeros(0, dtype=np.int16),
    )
    labels, twin_edges = forward_predict_merge_components(
        graph, n_alive=5, k_agree=3,
    )
    assert labels.tolist() == [0, 1, 2, 3, 4]
    assert twin_edges.shape == (0, 2)


def test_components_clean_merge_threshold_at_k_agree():
    # Three candidates: 0-1 agree=3 disagree=0 (merge), 1-2 agree=2 disagree=0 (below K)
    graph = ForwardPredictGraph(
        pair_a=np.array([0, 1], dtype=np.int32),
        pair_b=np.array([1, 2], dtype=np.int32),
        agree_count=np.array([3, 2], dtype=np.int16),
        disagree_count=np.array([0, 0], dtype=np.int16),
    )
    labels, _ = forward_predict_merge_components(graph, n_alive=3, k_agree=3)
    # 0 and 1 merge, 2 is a singleton
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_components_merge_blocked_by_disagree():
    # 0-1 agree=10 disagree=0 (would merge), but ALSO disagree=2 (blocks merge)
    graph = ForwardPredictGraph(
        pair_a=np.array([0], dtype=np.int32),
        pair_b=np.array([1], dtype=np.int32),
        agree_count=np.array([10], dtype=np.int16),
        disagree_count=np.array([2], dtype=np.int16),
    )
    labels, _ = forward_predict_merge_components(graph, n_alive=2, k_agree=3)
    # 0 and 1 stay separate because any disagreement blocks merge
    assert labels[0] != labels[1]


def test_components_twin_edges_emitted_between_clusters():
    # cand 0,1 merge (agree=5 disagree=0); cand 2 stays separate (twin to 0 via disagree=5)
    graph = ForwardPredictGraph(
        pair_a=np.array([0, 0], dtype=np.int32),
        pair_b=np.array([1, 2], dtype=np.int32),
        agree_count=np.array([5, 0], dtype=np.int16),
        disagree_count=np.array([0, 5], dtype=np.int16),
    )
    labels, twin_edges = forward_predict_merge_components(
        graph, n_alive=3, k_agree=3,
    )
    # cluster 0 has {0,1}; cluster 1 = {2}
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    # Twin edge connects the two cluster labels
    assert twin_edges.shape == (1, 2)
    assert set(twin_edges[0].tolist()) == set([labels[0], labels[2]])


def test_components_self_twin_edges_dropped():
    # 0-1 merge AND 0-1 disagree → after union into one cluster, twin self-loop dropped
    graph = ForwardPredictGraph(
        pair_a=np.array([0, 0], dtype=np.int32),
        pair_b=np.array([1, 1], dtype=np.int32),
        agree_count=np.array([10, 0], dtype=np.int16),
        disagree_count=np.array([0, 10], dtype=np.int16),
    )
    # Wait, those are the same pair — disambiguate with distinct pairs:
    # 0-1 merge agree=10; 0-2 merge agree=10; 1-2 disagree=10 (self-loop after merge)
    graph = ForwardPredictGraph(
        pair_a=np.array([0, 0, 1], dtype=np.int32),
        pair_b=np.array([1, 2, 2], dtype=np.int32),
        agree_count=np.array([10, 10, 0], dtype=np.int16),
        disagree_count=np.array([0, 0, 10], dtype=np.int16),
    )
    labels, twin_edges = forward_predict_merge_components(
        graph, n_alive=3, k_agree=3,
    )
    # 0, 1, 2 all in one cluster (chain merge); twin self-loop dropped
    assert labels[0] == labels[1] == labels[2]
    assert twin_edges.shape == (0, 2)


# ---------------------------------------------------------------------------
# select_k_agree_auto
# ---------------------------------------------------------------------------


def test_auto_k_picks_smallest_k_with_no_giant_component():
    # Construct a graph where K=3 gives biggest=500 (above 100), K=4 gives biggest=50
    # n_alive=1000, giant_floor=100, giant_fraction=0.01 → threshold=max(100, 10)=100
    n_alive = 1000
    # K=3 merges all 0..500 into a chain via weak agree=3 edges
    pa_3 = np.arange(0, 500, dtype=np.int32)
    pb_3 = np.arange(1, 501, dtype=np.int32)
    # K=4 only merges (0,1), (2,3), ... pairs — each pair becomes a 2-cluster
    pa_4 = np.arange(0, 100, 2, dtype=np.int32)
    pb_4 = np.arange(1, 101, 2, dtype=np.int32)
    pair_a = np.concatenate([pa_3, pa_4])
    pair_b = np.concatenate([pb_3, pb_4])
    agree = np.concatenate([
        np.full(500, 3, dtype=np.int16),  # K=3 weak edges
        np.full(50, 5, dtype=np.int16),   # K>=4 strong edges
    ])
    disagree = np.zeros(550, dtype=np.int16)

    graph = ForwardPredictGraph(
        pair_a=pair_a, pair_b=pair_b,
        agree_count=agree, disagree_count=disagree,
    )
    K = select_k_agree_auto(graph, n_alive=n_alive, k_min=3, k_max=8)
    # K=3 builds a 501-member chain (above giant_floor=100) → rejected.
    # K=4 only fires the 50 strong edges → biggest=2 → accepted.
    assert K == 4


def test_om_split_separates_two_grains_in_one_component():
    """Five candidates pre-merged into one component; the OM-spread splitter
    must separate them into the two underlying physical grains.

    Layout: three identity-like OMs (within 0.2°) + two 30°-rotated OMs.
    With ``om_tol_deg=1.0``, expect labels [0,0,0,1,1] (or any equivalent
    relabelling that gives the same 3-2 partition).
    """
    import torch
    from midas_stress.orientation import orient_mat_to_quat

    def _rot_z(deg):
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    OMs = np.stack([
        _rot_z(0.0), _rot_z(0.1), _rot_z(0.2),  # grain A
        _rot_z(30.0), _rot_z(30.1),             # grain B
    ])
    qs = orient_mat_to_quat(torch.from_numpy(OMs)).numpy()
    # All five start in the same merge component (simulating chain-fusion):
    labels = np.zeros(5, dtype=np.int64)
    new = split_components_by_om_spread(
        labels, om_fz_quat=qs, space_group=225, om_tol_deg=1.0,
    )
    # Expect 2 final labels with members partitioned 3 / 2.
    counts = np.bincount(new)
    assert sorted(counts.tolist()) == [2, 3]
    # And the partition is exactly the OM-coherent groups:
    assert new[0] == new[1] == new[2]
    assert new[3] == new[4]
    assert new[0] != new[3]


def test_om_split_leaves_singletons_alone():
    """Singleton components must not be touched even with a tight tolerance."""
    qs = np.array([
        [1.0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 0, 0, 0],
    ], dtype=np.float64)
    labels = np.array([0, 1, 2], dtype=np.int64)
    new = split_components_by_om_spread(
        labels, om_fz_quat=qs, space_group=225, om_tol_deg=0.01,
    )
    # No split possible: 3 singletons → 3 labels
    assert len(np.unique(new)) == 3


def test_om_split_keeps_coherent_component_intact():
    """Component with all members within tolerance must not be split."""
    import torch
    from midas_stress.orientation import orient_mat_to_quat

    def _rot_z(deg):
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    OMs = np.stack([_rot_z(0.0), _rot_z(0.1), _rot_z(0.2), _rot_z(0.3)])
    qs = orient_mat_to_quat(torch.from_numpy(OMs)).numpy()
    labels = np.zeros(4, dtype=np.int64)
    new = split_components_by_om_spread(
        labels, om_fz_quat=qs, space_group=225, om_tol_deg=1.0,
    )
    assert len(np.unique(new)) == 1


def test_auto_k_returns_k_max_if_giant_persists():
    # All edges agree=15 → giant component for all K in range → returns k_max
    pair_a = np.arange(0, 9, dtype=np.int32)
    pair_b = np.arange(1, 10, dtype=np.int32)
    graph = ForwardPredictGraph(
        pair_a=pair_a, pair_b=pair_b,
        agree_count=np.full(9, 15, dtype=np.int16),
        disagree_count=np.zeros(9, dtype=np.int16),
    )
    # n_alive=10, giant_floor=5; K=15 would break, but k_max=8 → giant
    # persists → returns 8. Bypass the small-dataset guard rail so we
    # actually exercise the persistence path (not the fallback).
    K = select_k_agree_auto(
        graph, n_alive=10, k_min=3, k_max=8, giant_floor=5,
        min_alive_for_auto=0, min_pairs_for_auto=0,
    )
    assert K == 8


def test_auto_k_guard_rail_small_dataset_returns_fallback():
    """n_alive < min_alive_for_auto → returns fallback_k without sweeping."""
    graph = ForwardPredictGraph(
        pair_a=np.array([0], dtype=np.int32),
        pair_b=np.array([1], dtype=np.int32),
        agree_count=np.array([5], dtype=np.int16),
        disagree_count=np.zeros(1, dtype=np.int16),
    )
    # n_alive=50 (< default min_alive_for_auto=100) → fallback_k=4
    K = select_k_agree_auto(graph, n_alive=50)
    assert K == 4


def test_auto_k_guard_rail_sparse_graph_returns_fallback():
    """Too few pairs in the graph → fallback regardless of n_alive."""
    graph = ForwardPredictGraph(
        pair_a=np.zeros(0, dtype=np.int32),
        pair_b=np.zeros(0, dtype=np.int32),
        agree_count=np.zeros(0, dtype=np.int16),
        disagree_count=np.zeros(0, dtype=np.int16),
    )
    K = select_k_agree_auto(graph, n_alive=10_000, fallback_k=7)
    assert K == 7


def test_auto_om_spread_tol_unimodal_floor():
    """When the misori histogram is unimodal (no chain-fusion 2nd mode),
    the auto-tolerance picker should return the floor."""
    from midas_process_grains.compute.forward_predict_merge import (
        select_om_spread_tol_auto,
    )
    import torch
    from midas_stress.orientation import orient_mat_to_quat

    def _rot_z(deg):
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    OMs = np.stack([_rot_z(d) for d in np.linspace(0, 0.5, 50)])
    qs = orient_mat_to_quat(torch.from_numpy(OMs)).numpy()
    labels = np.zeros(50, dtype=np.int64)
    tol = select_om_spread_tol_auto(
        labels, om_fz_quat=qs, space_group=225, floor_deg=0.5, ceil_deg=5.0,
    )
    # Unimodal → floor returned
    assert tol == 0.5
