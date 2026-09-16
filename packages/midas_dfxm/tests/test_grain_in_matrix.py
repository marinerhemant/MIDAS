"""Tests for :func:`midas_dfxm.io.make_grain_in_matrix_field` (Component-2 B1)."""
from __future__ import annotations

import math

import torch

from midas_dfxm.field import polar_decomposition
from midas_dfxm.io import ellipsoid_mask, make_grain_in_matrix_field


def test_ellipsoid_mask_sphere_matches_analytic_count():
    # A regular grid over a box big enough to contain the sphere; count inside
    # voxels against the discretised analytic volume ratio.
    n, half, spacing = 41, 4.0, 4.0 * 2 / 40
    xs = torch.linspace(-half, half, n, dtype=torch.float64)
    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    radius = 2.0
    mask = ellipsoid_mask(positions, (0.0, 0.0, 0.0), radius)
    voxel_vol = spacing ** 3
    counted_vol = float(mask.sum()) * voxel_vol
    sphere_vol = 4.0 / 3.0 * math.pi * radius ** 3
    assert abs(counted_vol - sphere_vol) / sphere_vol < 0.05


def test_grain_orientation_recovered_inside_identity_outside():
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    angle = math.radians(7.0)
    c, s = math.cos(angle), math.sin(angle)
    grain_orientation = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    field = make_grain_in_matrix_field(
        shape=(20, 20, 6), spacing_um=0.2,
        grain_center_um=(0.0, 0.0, 0.0), grain_radius_um=1.0,
        grain_orientation=grain_orientation,
    )
    mask = ellipsoid_mask(field.positions, (0.0, 0.0, 0.0), 1.0)
    assert bool(mask.any()) and bool((~mask).any())

    R, U = polar_decomposition(field.F)
    eye = torch.eye(3, dtype=torch.float64)
    # Pure rotation everywhere: no elastic stretch planted.
    assert float((U - eye).abs().max()) < 1e-10
    # Outside the grain, F is exactly the reference orientation (identity here).
    assert torch.allclose(field.F[~mask], eye.expand((~mask).sum(), 3, 3))
    # Inside the grain, reference_orientation @ R recovers grain_orientation.
    physical = field.reference_orientation @ R[mask]
    assert torch.allclose(physical, grain_orientation.expand_as(physical), atol=1e-10)


def test_no_grain_orientation_gives_uniform_field():
    field = make_grain_in_matrix_field(
        shape=(10, 10, 4), spacing_um=0.3,
        grain_center_um=(0.0, 0.0, 0.0), grain_radius_um=1.0,
    )
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(field.F, eye.expand(field.n_voxels, 3, 3))


def test_bad_grain_shape_rejected():
    try:
        make_grain_in_matrix_field(grain_shape="cube")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsupported grain_shape")
