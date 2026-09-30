"""Cross-voxel global clustering."""

from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.find_grains import (
    global_cluster,
    write_unique_orientations_csv,
)


def test_4vox_2grains_dedup_to_2_uniques(axis_angle_om, tmp_path):
    """4 voxels, 2 underlying grains → global_cluster yields 2 uniques."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 30.0)
    per_vox_OMs = np.vstack([g1, g1, g2, g2])
    per_vox_confs = np.array([0.9, 0.7, 0.8, 0.6])
    # Layout: [SpotID, nMatches, nIDs_best_solution, bestSolIdx] per voxel.
    per_vox_keys = np.array([
        [10, 5, 4, 0],
        [11, 5, 4, 1],
        [20, 6, 5, 0],
        [21, 6, 5, 1],
    ], dtype=np.uint64)
    glob = global_cluster(
        per_vox_OMs, per_vox_confs, per_vox_keys,
        space_group=225, max_ang_deg=1.0,
    )
    assert glob.n_uniques == 2
    # The representative voxel of each cluster should be the highest-conf one.
    # Voxel 0 wins for g1 (conf 0.9), voxel 2 wins for g2 (conf 0.8).
    rep_voxels = sorted(int(r[0]) for r in glob.unique_key_arr)
    assert rep_voxels == [0, 2]


def test_global_cluster_skips_invalid_voxels(axis_angle_om):
    """Voxels with sentinel ``per_vox_keys[v, 0] == (uint64)-1`` are skipped."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    per_vox_OMs = np.vstack([g1, g1, g1])
    per_vox_confs = np.array([0.5, 0.5, 0.5])
    per_vox_keys = np.array([
        [2**64 - 1, 0, 0, 0],   # invalid
        [11, 5, 4, 0],
        [12, 5, 4, 0],
    ], dtype=np.uint64)
    glob = global_cluster(
        per_vox_OMs, per_vox_confs, per_vox_keys,
        space_group=225, max_ang_deg=1.0,
    )
    assert glob.n_uniques == 1
    # The rep voxel is 1 (first non-invalid; ties broken by lower index).
    assert int(glob.unique_key_arr[0, 0]) == 1


def test_unique_orientations_csv_14col_format(axis_angle_om, tmp_path):
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 30.0)
    per_vox_OMs = np.vstack([g1, g2])
    per_vox_confs = np.array([0.9, 0.7])
    per_vox_keys = np.array([
        [10, 5, 4, 0],
        [20, 6, 5, 1],
    ], dtype=np.uint64)
    glob = global_cluster(
        per_vox_OMs, per_vox_confs, per_vox_keys,
        space_group=225, max_ang_deg=1.0,
    )
    csv_path = tmp_path / "UniqueOrientations.csv"
    write_unique_orientations_csv(csv_path, glob.unique_key_arr, glob.unique_OM_arr)
    text = csv_path.read_text().strip().splitlines()
    assert text[0].startswith("#")
    # Header tokens after '#' should mention 5 key cols + 9 OM cols = 14.
    header_tokens = [t for t in text[0].lstrip("#").split() if t]
    assert len(header_tokens) == 14, header_tokens
    # Body: 2 rows, each row has 14 columns.
    for body_line in text[1:]:
        toks = body_line.split()
        assert len(toks) == 14


# --- opt-in sibling merge (midas_pipeline.find_grains.merge_sibling_grains) -------------------------

def _three_grain_result(axis_angle_om):
    """A, B (0.4 deg from A, a sibling of A) and C (30 deg away). Voxels: A A B B C."""
    from midas_pipeline.find_grains import GlobalClusterResult
    a = axis_angle_om(np.array([0, 0, 1]), 0.0)
    b = axis_angle_om(np.array([0, 0, 1]), 0.4)
    c = axis_angle_om(np.array([0, 0, 1]), 30.0)
    keys = np.array([[0, 10, 5, 4, 0], [2, 20, 5, 4, 0], [4, 30, 6, 5, 0]], dtype=np.uint64)   # rep voxels 0, 2, 4
    glob = GlobalClusterResult(3, keys, np.vstack([a, b, c]), np.array([0, 0, 1, 1, 2], dtype=np.int64))
    confs = np.array([0.6, 0.6, 0.9, 0.9, 0.8])          # B's representative voxel has the higher confidence
    return glob, confs


def test_sibling_merge_default_off_is_identity(axis_angle_om):
    from midas_pipeline.find_grains import merge_sibling_grains
    glob, confs = _three_grain_result(axis_angle_om)
    assert merge_sibling_grains(glob, confs, space_group=225, merge_deg=0.0) is glob


def test_sibling_merge_joins_close_grains_and_remaps_voxels(axis_angle_om):
    from midas_pipeline.find_grains import merge_sibling_grains
    glob, confs = _three_grain_result(axis_angle_om)
    out = merge_sibling_grains(glob, confs, space_group=225, merge_deg=1.0)
    assert out.n_uniques == 2
    # the merged grain keeps the higher-confidence member's row (B, rep voxel 2) and stays first
    assert [int(r[0]) for r in out.unique_key_arr] == [2, 4]
    assert out.voxel_to_unique.tolist() == [0, 0, 0, 0, 1]
    np.testing.assert_array_equal(out.unique_OM_arr[0], glob.unique_OM_arr[1])


def test_sibling_merge_below_threshold_keeps_grains_apart(axis_angle_om):
    from midas_pipeline.find_grains import merge_sibling_grains
    glob, confs = _three_grain_result(axis_angle_om)
    assert merge_sibling_grains(glob, confs, space_group=225, merge_deg=0.2) is glob


def test_sibling_merge_keeps_invalid_voxels_invalid(axis_angle_om):
    from midas_pipeline.find_grains import merge_sibling_grains, GlobalClusterResult
    glob, confs = _three_grain_result(axis_angle_om)
    glob2 = GlobalClusterResult(3, glob.unique_key_arr, glob.unique_OM_arr, np.array([0, -1, 1, 1, 2], dtype=np.int64))
    out = merge_sibling_grains(glob2, confs, space_group=225, merge_deg=1.0)
    assert out.voxel_to_unique.tolist() == [0, -1, 0, 0, 1]


def test_sibling_merge_warns_when_a_chain_spans_more_than_twice_the_threshold(axis_angle_om, caplog):
    import logging
    from midas_pipeline.find_grains import merge_sibling_grains, GlobalClusterResult
    oms = np.vstack([axis_angle_om(np.array([0, 0, 1]), d) for d in (0.0, 0.9, 1.8, 2.7)])
    keys = np.array([[i, 10 + i, 5, 4, 0] for i in range(4)], dtype=np.uint64)
    glob = GlobalClusterResult(4, keys, oms, np.arange(4, dtype=np.int64))
    with caplog.at_level(logging.WARNING):
        out = merge_sibling_grains(glob, np.full(4, 0.5), space_group=225, merge_deg=1.0)
    assert out.n_uniques == 1
    assert any("chained" in r.message for r in caplog.records)


def test_fusion_config_sibling_merge_defaults_off():
    from midas_pipeline.config import FusionConfig
    assert FusionConfig().sibling_merge_deg == 0.0
