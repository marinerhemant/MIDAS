"""Per-voxel clustering — splits two grains, collapses symmetry variants."""

from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.find_grains import per_voxel_cluster


def make_inputs(OMs, confs, ias):
    """Build the (OMs, confs, ias, keys) tuple for per_voxel_cluster."""
    OMs = np.asarray(OMs, dtype=np.float64)
    n = OMs.shape[0]
    keys = np.zeros((n, 4), dtype=np.uint64)
    # Seed keys with row index so we can recover which row was picked.
    keys[:, 0] = np.arange(n, dtype=np.uint64)
    return OMs, np.asarray(confs, dtype=np.float64), np.asarray(ias, dtype=np.float64), keys


def test_two_grains_30deg_apart_split(axis_angle_om):
    """Two grain orientations 30° apart in cubic crystal: should yield 2 clusters."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 30.0)
    OMs = np.vstack([g1, g1, g2, g2])
    confs = np.array([0.9, 0.8, 0.7, 0.6])
    ias = np.array([0.1, 0.2, 0.3, 0.4])
    OMs, confs, ias, keys = make_inputs(OMs, confs, ias)
    result = per_voxel_cluster(
        OMs, confs, ias, keys,
        space_group=225,  # FCC cubic
        max_ang_deg=1.0,
    )
    # Two distinct grains → 2 clusters.
    assert result.unique_keys.shape[0] == 2
    # The reps should be the highest-conf of each cluster: row 0 (0.9) and row 2 (0.7).
    rep_ids = sorted(int(r[0]) for r in result.unique_keys)
    assert rep_ids == [0, 2]
    # best_row points at the overall best (row 0, conf 0.9).
    assert result.best_row == 0


def test_two_grains_within_maxang_collapse(axis_angle_om):
    """If grains are within max_ang_deg they fold into one cluster."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 0.2)   # 0.2° rotation, sub-threshold
    OMs = np.vstack([g1, g2])
    confs = np.array([0.5, 0.9])
    ias = np.array([0.5, 0.1])
    OMs, confs, ias, keys = make_inputs(OMs, confs, ias)
    result = per_voxel_cluster(
        OMs, confs, ias, keys,
        space_group=225,
        max_ang_deg=1.0,
    )
    # One cluster — the higher-conf representative wins.
    assert result.unique_keys.shape[0] == 1
    assert int(result.unique_keys[0, 0]) == 1  # row 1 = higher conf


def test_symmetry_variant_collapses_with_cubic(axis_angle_om):
    """A 90° rotation about z is a symmetry-equivalent OM in cubic; must collapse."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 90.0)  # cubic-symmetric to g1
    OMs = np.vstack([g1, g2])
    confs = np.array([0.7, 0.7])
    ias = np.array([0.5, 0.5])
    OMs, confs, ias, keys = make_inputs(OMs, confs, ias)
    result = per_voxel_cluster(
        OMs, confs, ias, keys,
        space_group=225,
        max_ang_deg=1.0,
    )
    # Cubic symmetry should fold these to one cluster.
    assert result.unique_keys.shape[0] == 1


def test_empty_input_returns_no_best():
    OMs = np.empty((0, 9))
    confs = np.empty(0)
    ias = np.empty(0)
    keys = np.empty((0, 4), dtype=np.uint64)
    result = per_voxel_cluster(OMs, confs, ias, keys, space_group=225, max_ang_deg=1.0)
    assert result.best_row == -1
    assert result.unique_keys.shape[0] == 0


def test_tie_break_lower_ia_wins(axis_angle_om):
    """Best-row: equal conf, lower IA wins."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    OMs = np.vstack([g1, g1, g1])
    confs = np.array([0.5, 0.5, 0.4])
    ias = np.array([0.3, 0.1, 0.05])    # row 1 has lowest IA among the conf-0.5 pair
    OMs, confs, ias, keys = make_inputs(OMs, confs, ias)
    result = per_voxel_cluster(
        OMs, confs, ias, keys,
        space_group=225, max_ang_deg=1.0,
    )
    # Best row = 1 (conf 0.5, IA 0.1).
    assert result.best_row == 1


def test_min_conf_filters_rows(axis_angle_om):
    """When min_conf > 0, below-threshold rows are skipped from clustering."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 30.0)
    OMs = np.vstack([g1, g1, g2])
    confs = np.array([0.9, 0.05, 0.8])   # row 1 is below 0.5 threshold
    ias = np.array([0.1, 0.1, 0.1])
    OMs, confs, ias, keys = make_inputs(OMs, confs, ias)
    result = per_voxel_cluster(
        OMs, confs, ias, keys,
        space_group=225, max_ang_deg=1.0,
        min_conf=0.5,
    )
    # Two clusters (row 1 was filtered out, but g1 still has row 0 and g2 has row 2).
    assert result.unique_keys.shape[0] == 2


def test_torch_per_voxel_matches_numpy(axis_angle_om):
    """Torch path produces same cluster assignments as numpy."""
    torch = pytest.importorskip("torch")
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 30.0)
    OMs_np = np.vstack([g1, g1, g2])
    confs = np.array([0.9, 0.8, 0.7])
    ias = np.array([0.1, 0.2, 0.3])
    keys = np.array([[i, 0, 0, 0] for i in range(3)], dtype=np.uint64)

    res_np = per_voxel_cluster(OMs_np, confs, ias, keys, space_group=225, max_ang_deg=1.0)
    from midas_pipeline.find_grains import per_voxel_cluster_torch
    OMs_t = torch.tensor(OMs_np, dtype=torch.float64)
    res_t = per_voxel_cluster_torch(
        OMs_t, torch.tensor(confs, dtype=torch.float64),
        torch.tensor(ias, dtype=torch.float64),
        keys, space_group=225, max_ang_deg=1.0,
    )
    # Compare keys arrays (host-side identical).
    np.testing.assert_array_equal(res_np.unique_keys, res_t.unique_keys)
    # OMs in torch path should be tensors.
    assert isinstance(res_t.unique_OMs, torch.Tensor)


# ---------------------------------------------------------------------------
# need_uniques=False — the fast path find_grains_single takes
# ---------------------------------------------------------------------------


def _random_voxel(rng, n):
    """A voxel's worth of candidates: random OMs via QR, random conf/IA."""
    A = rng.normal(size=(n, 3, 3))
    OMs = np.empty((n, 9), dtype=np.float64)
    for i in range(n):
        q, r = np.linalg.qr(A[i])
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        OMs[i] = q.ravel()
    confs = rng.random(n)
    ias = rng.random(n)
    keys = np.zeros((n, 4), dtype=np.uint64)
    keys[:, 0] = np.arange(n, dtype=np.uint64)
    return OMs, confs, ias, keys


@pytest.mark.parametrize("n", [1, 2, 7, 50, 200])
def test_need_uniques_false_gives_identical_best_row(n):
    """The skipped grouping cannot move best_row/best_conf/best_ia.

    This is the whole safety argument for the fast path: best_row comes from
    an O(n) scan that runs BEFORE the O(n_sol^2) grouping, so dropping the
    grouping must leave it bit-identical.
    """
    rng = np.random.default_rng(1234 + n)
    OMs, confs, ias, keys = _random_voxel(rng, n)
    full = per_voxel_cluster(OMs, confs, ias, keys,
                             space_group=166, max_ang_deg=1.0)
    fast = per_voxel_cluster(OMs, confs, ias, keys,
                             space_group=166, max_ang_deg=1.0,
                             need_uniques=False)
    assert fast.best_row == full.best_row
    assert fast.best_conf == full.best_conf
    assert fast.best_ia == full.best_ia


def test_need_uniques_false_holds_under_confidence_ties():
    """Ties are where best_row selection is subtle — IA breaks them.

    Random floats almost never tie, so force exact ties and duplicate
    orientations, which is also what a saturated dense voxel looks like.
    """
    rng = np.random.default_rng(7)
    OMs, _, _, keys = _random_voxel(rng, 12)
    OMs[1] = OMs[0]
    OMs[2] = OMs[0]
    confs = np.array([0.5, 0.9, 0.9, 0.9, 0.5, 0.5,
                      0.9, 0.2, 0.2, 0.5, 0.9, 0.5])
    ias = np.array([0.4, 0.3, 0.1, 0.7, 0.2, 0.9,
                    0.1, 0.5, 0.5, 0.8, 0.6, 0.3])
    full = per_voxel_cluster(OMs, confs, ias, keys,
                             space_group=166, max_ang_deg=1.0)
    fast = per_voxel_cluster(OMs, confs, ias, keys,
                             space_group=166, max_ang_deg=1.0,
                             need_uniques=False)
    assert fast.best_row == full.best_row
    assert fast.best_conf == full.best_conf
    assert fast.best_ia == full.best_ia


def test_need_uniques_false_returns_empty_unique_arrays(axis_angle_om):
    """Shapes stay valid so a pass-through caller does not crash."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    OMs, confs, ias, keys = make_inputs(
        np.vstack([g1, g1]), [0.9, 0.8], [0.1, 0.2])
    fast = per_voxel_cluster(OMs, confs, ias, keys, space_group=225,
                             max_ang_deg=1.0, need_uniques=False)
    assert fast.unique_keys.shape == (0, 4)
    assert fast.unique_OMs.shape == (0, 9)
    assert fast.unique_keys.dtype == np.uint64
    assert fast.unique_OMs.dtype == np.float64


def test_default_still_groups(axis_angle_om):
    """Guard the DEFAULT: find_grains_multiple depends on the unique arrays.

    If need_uniques ever defaults to False, the multiple-solutions path
    silently returns no orientations, so pin the default explicitly.
    """
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 30.0)
    OMs, confs, ias, keys = make_inputs(
        np.vstack([g1, g1, g2]), [0.9, 0.8, 0.7], [0.1, 0.2, 0.3])
    res = per_voxel_cluster(OMs, confs, ias, keys, space_group=225,
                            max_ang_deg=1.0)
    assert res.unique_keys.shape[0] == 2
    assert res.unique_OMs.shape[0] == 2
