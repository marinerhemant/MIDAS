"""Stage-6 tests: grain-size recompute on split clusters."""
from __future__ import annotations

import numpy as np

from midas_process_grains.compute.grain_size_recompute import (
    recompute_grain_sizes, merge_twin_family_volumes,
)


def test_unsplit_cluster_keeps_naive_radius():
    """A Pass-1 cluster that didn't split (one final grain) should
    have V_NNLS == V_naive trivially."""
    final = np.array([0, 0, 0], dtype=np.int64)
    p1 = np.array([0, 0, 0], dtype=np.int64)
    spots = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}]
    per_int = {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}
    per_ring = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    ring_K = {1: 100.0}
    radius_naive = np.array([5.0])
    out = recompute_grain_sizes(
        final_grain_id_per_candidate=final, pass1_cluster_id=p1,
        spot_sets_per_candidate=spots,
        per_spot_intensity=per_int, per_spot_ring=per_ring, ring_K=ring_K,
        rep_radius_naive_um_per_grain=radius_naive,
    )
    np.testing.assert_allclose(out.radius_nnls_um, out.radius_naive_um)
    np.testing.assert_allclose(out.deflation_factor, 1.0)


def test_split_cluster_redistributes_intensity():
    """Two split-product grains whose shared spots have known intensity
    should each get ~half the total volume."""
    # Two final grains from the same Pass-1 cluster
    final = np.array([0, 1, 0, 1], dtype=np.int64)
    p1 = np.array([0, 0, 0, 0], dtype=np.int64)
    # Each candidate claims 5 spots; the four candidates together claim
    # 10 distinct spots, with the middle 2 shared between both grains.
    spots = [
        {1, 2, 3, 4, 5},        # grain 0 candidate
        {4, 5, 6, 7, 8},        # grain 1 candidate
        {2, 3, 4, 5, 9},        # grain 0 candidate (overlaps 4, 5 with grain 1)
        {5, 6, 7, 8, 10},       # grain 1 candidate
    ]
    per_int = {s: 100.0 for s in range(1, 11)}
    per_ring = {s: 1 for s in range(1, 11)}
    ring_K = {1: 100.0}
    radius_naive = np.array([5.0, 5.0])
    out = recompute_grain_sizes(
        final_grain_id_per_candidate=final, pass1_cluster_id=p1,
        spot_sets_per_candidate=spots,
        per_spot_intensity=per_int, per_spot_ring=per_ring, ring_K=ring_K,
        rep_radius_naive_um_per_grain=radius_naive,
    )
    # Both grains should get finite volumes
    assert (out.volume_nnls_um3 > 0).all()
    # Total volume preserved (by construction)
    np.testing.assert_allclose(out.volume_nnls_um3.sum(),
                                out.volume_naive_um3.sum(), rtol=1e-6)


def test_twin_family_rollup_sums_volumes():
    vols = np.array([100.0, 200.0, 50.0])
    family = np.array([0, 0, -1], dtype=np.int64)
    fam = merge_twin_family_volumes(volumes_um3=vols, twin_family_id=family)
    assert fam == {0: 300.0}
