"""P2-8 + P2-9 tests: sparse-grain smoothness + spread warm-start pooling."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from midas_pf_odf import fit_grain_peakshape
from midas_pf_odf.inversion import (
    _pool_init_by_region, neighbor_edges_from_grid_ij,
)
from midas_pf_odf.simulate import plant_single_grain, simulate_grain_patches
from tests.conftest import build_model, make_fcc_hkls, small_scan_config

DT = torch.float64


def test_edges_from_sparse_grid_ij():
    # L-shaped grain: (0,0) (0,1) (1,0) — two 4-neighbour edges.
    ij = np.array([[0, 0], [0, 1], [1, 0]])
    e = neighbor_edges_from_grid_ij(ij)
    got = {tuple(sorted(p)) for p in e.tolist()}
    assert got == {(0, 1), (0, 2)}
    # Diagonal-only pair: no edges.
    assert neighbor_edges_from_grid_ij(np.array([[0, 0], [1, 1]])).shape == (0, 2)


def _planted():
    G_cart, thetas, hkls_int = make_fcc_hkls()
    scan = small_scan_config(sample_size_um=12.0, n_scans=5, beam_size_um=4.0)
    model = build_model(scan, hkls_int, G_cart, thetas)
    plant = plant_single_grain(
        grid_shape=(3, 3), voxel_size_um=4.0,
        lattice=(3.61, 3.61, 3.61, 90.0, 90.0, 90.0),
        eps_gradient_voigt=0, eps_gradient_amp=1e-3, eps_gradient_dir="y",
    )
    data = simulate_grain_patches(plant, model, patch_F=5, patch_P=15)
    return model, plant, data


def _kw(plant, **over):
    kw = dict(
        voxel_pos=plant.voxel_pos, R_init=plant.R_voxel,
        eps_init=torch.zeros(plant.n_voxels, 6, dtype=DT),
        lattice_init=plant.lattice,
        optimizer="adam", inner_steps=2,
        lr_aa=1e-4, lr_eps=1e-5, lr_lat=0.0,
    )
    kw.update(over)
    return kw


def test_sparse_grid_shape_raises_helpfully():
    """The old crash ('shape [41,41,3] invalid for size 3936') is now an
    actionable error pointing at neighbor_edges_from_grid_ij."""
    model, plant, data = _planted()
    with pytest.raises(ValueError, match="neighbor_edges_from_grid_ij"):
        fit_grain_peakshape(
            data, model,
            **_kw(plant, grid_shape=(41, 41)),
            lambda_smooth=1e-3,
        )


def test_sparse_smoothness_via_edges_runs():
    model, plant, data = _planted()
    ij = np.stack(np.meshgrid(np.arange(3), np.arange(3), indexing="ij"),
                  -1).reshape(-1, 2)
    edges = neighbor_edges_from_grid_ij(ij)
    assert edges.shape[0] == 12                 # 3x3 grid: 2*3*2 = 12 edges
    res = fit_grain_peakshape(
        data, model,
        **_kw(plant, neighbor_edges=edges),
        lambda_smooth=1e-3,
    )
    assert np.isfinite(res.losses).all()


def test_pool_init_by_region():
    reg = torch.tensor([0, 0, 1, 1, 1])
    per_voxel = torch.tensor([1.0, 3.0, 2.0, 4.0, 6.0])
    pooled = _pool_init_by_region(per_voxel, reg, 2, 5, what="spread_init")
    torch.testing.assert_close(pooled, torch.tensor([2.0, 4.0]))
    # Per-region passes through unchanged.
    per_region = torch.tensor([0.7, 0.9])
    torch.testing.assert_close(
        _pool_init_by_region(per_region, reg, 2, 5, what="spread_init"),
        per_region)
    # Wrong size: actionable error, not a reshape crash.
    with pytest.raises(ValueError, match="spread_init has 3 entries"):
        _pool_init_by_region(torch.ones(3), reg, 2, 5, what="spread_init")


def test_stage2_to_stage3_per_voxel_spread_handoff():
    """The exact P2-9 failure: per-voxel (G,) spread_fit from Stage 2 fed
    as spread_init to a region-pooled Stage 3 (n_spread << G)."""
    model, plant, data = _planted()
    G = plant.n_voxels
    per_voxel_spread = torch.full((G,), 0.6, dtype=DT)
    reg_map = torch.arange(G) % 2               # 2 regions
    res = fit_grain_peakshape(
        data, model,
        **_kw(plant),
        refine_spread=True,
        spread_init=per_voxel_spread,
        spread_region_map=reg_map,
    )
    assert res.spread_fit is not None
    assert np.isfinite(res.losses).all()
