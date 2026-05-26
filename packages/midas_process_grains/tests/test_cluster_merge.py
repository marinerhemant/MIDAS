"""Tests for compute.cluster_merge — Pass-1.5 twin-aware cluster merge."""

import math
import numpy as np
import pytest

from midas_process_grains.compute.cluster_merge import (
    compute_cluster_merges, ClusterMergeResult,
)


def _axis_angle_to_om(axis, angle_deg):
    """Rodrigues rotation."""
    a = np.array(axis, dtype=np.float64); a /= np.linalg.norm(a)
    t = math.radians(angle_deg)
    K = np.array([[ 0,    -a[2],  a[1]],
                  [ a[2],  0,    -a[0]],
                  [-a[1],  a[0],  0  ]])
    return np.eye(3) + math.sin(t)*K + (1-math.cos(t))*K@K


def test_no_merges_when_far_apart():
    """Two clusters 5000µm apart should never merge."""
    OMs = np.stack([np.eye(3), _axis_angle_to_om([0,0,1], 0.5)])
    pos = np.array([[0,0,0], [5000,0,0]], dtype=float)
    res = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=225, tol_misori_deg=2.0, tol_position_um=100.0,
    )
    assert res.n_out_parents == 2
    assert res.n_merges_direct + res.n_merges_twin == 0


def test_direct_misori_merge_for_alt_indexings():
    """Two clusters at the same position with 1° misori → merge (direct)."""
    OMs = np.stack([np.eye(3), _axis_angle_to_om([0,0,1], 1.0)])
    pos = np.array([[0,0,0], [50,0,0]], dtype=float)
    res = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=225, tol_misori_deg=2.0, tol_position_um=200.0,
        mode="direct",
    )
    assert res.n_out_parents == 1
    assert res.n_merges_direct == 1
    assert res.parent_cluster_id[0] == res.parent_cluster_id[1]


def test_twin_aware_merge_for_fcc_sigma3():
    """Two FCC grains in Σ3 twin relationship (60° about <111>) at same
    position should merge via the twin operator (not direct)."""
    OM_parent = np.eye(3)
    OM_twin   = _axis_angle_to_om([1, 1, 1], 60.0)   # FCC Σ3
    OMs = np.stack([OM_parent, OM_twin])
    pos = np.array([[0,0,0], [40,0,0]], dtype=float)
    # Direct misori is ~60° → too big for tol=2°; only twin-mode should merge
    res_direct = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=225, tol_misori_deg=2.0, tol_position_um=200.0,
        mode="direct",
    )
    assert res_direct.n_out_parents == 2   # no direct merge

    res_twin = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=225, tol_misori_deg=2.0, tol_position_um=200.0,
        mode="twin",
    )
    assert res_twin.n_out_parents == 1     # twin op catches it
    assert res_twin.n_merges_twin == 1


def test_combined_mode_catches_both():
    """Three clusters at same site: A=identity, B=A+1° (alt-indexing), C=A·Σ3 (twin).
    Combined mode must merge all three."""
    OMs = np.stack([
        np.eye(3),
        _axis_angle_to_om([0, 0, 1], 1.0),
        _axis_angle_to_om([1, 1, 1], 60.0),
    ])
    pos = np.zeros((3, 3))
    res = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=225, tol_misori_deg=2.0, tol_position_um=100.0,
        mode="combined",
    )
    assert res.n_out_parents == 1
    assert res.n_merges_direct >= 1
    assert res.n_merges_twin >= 1


def test_singleton_input_no_merge():
    OMs = np.eye(3)[None, :, :]
    pos = np.zeros((1, 3))
    res = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=225,
    )
    assert res.n_out_parents == 1
    assert res.parent_cluster_id[0] == 0


def test_empty_input():
    OMs = np.zeros((0, 3, 3))
    pos = np.zeros((0, 3))
    res = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=225,
    )
    assert res.n_out_parents == 0
    assert res.n_in_clusters == 0


def test_transitive_closure_via_union_find():
    """Chain: A-B by direct, B-C by direct, A-C by neither → still merged together."""
    OMs = np.stack([
        np.eye(3),
        _axis_angle_to_om([0, 0, 1], 1.5),    # A-B = 1.5° (direct merge)
        _axis_angle_to_om([0, 0, 1], 3.0),    # B-C = 1.5° (direct merge); A-C = 3° (NOT direct)
    ])
    pos = np.zeros((3, 3))
    res = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=225, tol_misori_deg=2.0, tol_position_um=100.0,
        mode="direct",
    )
    # Pairs (0,1) and (1,2) both merged via direct; transitive closure
    # makes them all one parent even though (0,2) wouldn't directly merge.
    assert res.n_out_parents == 1


def test_hcp_twin_requires_c_over_a():
    """Hexagonal SG without c_over_a → twin mode has no operators →
    no merges in pure twin mode. Combined mode still does direct merges."""
    OMs = np.stack([np.eye(3), np.eye(3)])
    pos = np.zeros((2, 3))
    res_twin = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=194, c_over_a=None,
        mode="twin",
    )
    # Pure twin mode + no ops = no merges, even for identical OMs
    assert res_twin.n_out_parents == 2
    assert res_twin.n_merges_twin == 0

    res_combined = compute_cluster_merges(
        cluster_orientation_matrices=OMs, cluster_positions_um=pos,
        space_group=194, c_over_a=None,
        mode="combined",
    )
    # Combined mode still catches identity-identity via direct path
    assert res_combined.n_out_parents == 1
