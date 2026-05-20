"""Unit tests for midas_pipeline.voxel_cleanup (core, forward-model-injected)."""
from __future__ import annotations

import numpy as np

from midas_pipeline.voxel_cleanup import cleanup_voxel_grid


def _scenario():
    """7x7 grid, pitch 1. Grain 0 = compact 3x3 centre block; plus ONE
    over-extended corner voxel at (3,3) — two pitches beyond the block, clearly
    outside the occupancy dilation margin — wrongly assigned to grain 0 and
    spatially isolated. Observed spots cover only the true block's sinogram, so
    the corner's predicted spots are unmatched in empty-sinogram directions ->
    high directional score; block voxels are fully matched."""
    pos = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    n = 7
    vx = np.empty(n * n); vy = np.empty(n * n)
    for i in range(n):
        for j in range(n):
            v = i * n + j
            vx[v] = pos[i]; vy[v] = pos[j]
    grain = np.full(n * n, -1, np.int64)
    block = []
    for i in (2, 3, 4):
        for j in (2, 3, 4):
            grain[i * n + j] = 0
            block.append(i * n + j)
    corner = 6 * n + 6          # (3, 3), isolated, 2 pitches beyond block
    grain[corner] = 0

    omg_deg = np.array([0.0, 45.0, 90.0, 135.0])

    def predict_fn(g, vox_ids):
        k = len(vox_ids); m = omg_deg.size
        om = np.tile(omg_deg, (k, 1))
        et = np.zeros((k, m)); rg = np.ones((k, m), int)
        valid = np.ones((k, m), bool)
        return om, et, rg, valid

    # observed spots: block voxels' projections only (ring 1, eta 0)
    oo, oe, osp = [], [], []
    for b in block:
        for od in omg_deg:
            r = np.deg2rad(od)
            oo.append(od); oe.append(0.0)
            osp.append(vx[b] * np.sin(r) + vy[b] * np.cos(r))
    obs_by_ring = {1: (np.array(oo), np.array(oe), np.array(osp))}
    return vx, vy, grain, predict_fn, obs_by_ring, corner, block


def test_flags_overextended_voxel_not_block():
    vx, vy, grain, predict_fn, obs_by_ring, corner, block = _scenario()
    res = cleanup_voxel_grid(
        predict_fn=predict_fn, vx=vx, vy=vy, grain=grain, grains=[0],
        obs_by_ring=obs_by_ring, pitch=1.0,
        margin_ome=1.0, margin_eta=1.0, scan_tol=0.5,
        score_threshold=0.3, max_same_neighbours=1, max_iters=5,
        action="remove",
    )
    assert res.flagged[corner], "isolated over-extended voxel must be flagged"
    assert res.new_grain[corner] == -1
    # the compact block must be untouched
    for b in block:
        assert not res.flagged[b], f"block voxel {b} wrongly flagged"
        assert res.new_grain[b] == 0


def test_directional_score_high_for_corner_zero_for_block():
    vx, vy, grain, predict_fn, obs_by_ring, corner, block = _scenario()
    res = cleanup_voxel_grid(
        predict_fn=predict_fn, vx=vx, vy=vy, grain=grain, grains=[0],
        obs_by_ring=obs_by_ring, pitch=1.0,
        margin_ome=1.0, margin_eta=1.0, scan_tol=0.5,
        score_threshold=2.0,           # so nothing is acted on; just inspect scores
        max_same_neighbours=4, max_iters=1,
    )
    assert res.directional[corner] > 0.3
    assert np.allclose(res.directional[block], 0.0)


def test_connectivity_gate_protects_connected_voxels():
    """With max_same_neighbours=0, only fully-isolated voxels are eligible; the
    corner (0 same-grain neighbours) qualifies, a block-interior voxel never."""
    vx, vy, grain, predict_fn, obs_by_ring, corner, block = _scenario()
    res = cleanup_voxel_grid(
        predict_fn=predict_fn, vx=vx, vy=vy, grain=grain, grains=[0],
        obs_by_ring=obs_by_ring, pitch=1.0,
        margin_ome=1.0, margin_eta=1.0, scan_tol=0.5,
        score_threshold=0.3, max_same_neighbours=0, max_iters=5,
        action="remove",
    )
    centre = 3 * 7 + 3
    assert not res.flagged[centre]


def test_noop_when_all_matched():
    """If observed spots cover everything (incl. the corner), nothing is
    flagged — the safe behaviour on well-supported maps."""
    vx, vy, grain, predict_fn, obs_by_ring, corner, block = _scenario()
    # extend observed coverage to the corner too
    omg = np.array([0.0, 45.0, 90.0, 135.0])
    oo, oe, osp = list(obs_by_ring[1][0]), list(obs_by_ring[1][1]), list(obs_by_ring[1][2])
    for od in omg:
        r = np.deg2rad(od)
        oo.append(od); oe.append(0.0)
        osp.append(vx[corner] * np.sin(r) + vy[corner] * np.cos(r))
    obs2 = {1: (np.array(oo), np.array(oe), np.array(osp))}
    res = cleanup_voxel_grid(
        predict_fn=predict_fn, vx=vx, vy=vy, grain=grain, grains=[0],
        obs_by_ring=obs2, pitch=1.0,
        margin_ome=1.0, margin_eta=1.0, scan_tol=0.5,
        score_threshold=0.3, max_same_neighbours=4, max_iters=5,
        action="remove",
    )
    assert res.flagged.sum() == 0
