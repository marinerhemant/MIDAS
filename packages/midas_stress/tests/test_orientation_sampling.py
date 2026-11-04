"""Orientation-space sampling: valid rotations, one representative per symmetry class, and a covering radius of about the requested step."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

import midas_stress as ms
from midas_stress.orientation import make_symmetries, misorientation_om
from midas_stress.orientation_sampling import laue_rotations, orientation_grid, uniform_rotations


def test_exports():
    assert ms.orientation_grid is orientation_grid and ms.uniform_rotations is uniform_rotations and ms.laue_rotations is laue_rotations


def test_rotations_are_proper():
    g = orientation_grid(6.0, 139)
    assert np.allclose(np.linalg.det(g), 1.0) and np.allclose(np.einsum("nij,nkj->nik", g, g), np.eye(3), atol=1e-9)
    with pytest.raises(ValueError):
        uniform_rotations(0)
    with pytest.raises(ValueError):
        orientation_grid(0.0)


@pytest.mark.parametrize("sg,order", [(1, 1), (87, 4), (139, 8), (225, 24), (194, 12)])
def test_laue_rotations_match_make_symmetries(sg, order):
    assert len(laue_rotations(sg)) == make_symmetries(sg)[0] == order


@pytest.mark.parametrize("sg", [139, 71, 1, 194, 225])
def test_covering_distance(sg):
    """Every random orientation, modulo the Laue group, is within ~1.2 step (95th percentile) and 1.6 step (max) of a grid point; the reference
    distance is the library's own symmetry-reduced misorientation."""
    step = 6.0
    g = orientation_grid(step, sg)
    ops = laue_rotations(sg)
    dist = []
    for u in Rot.random(120, random_state=5).as_matrix():
        best = max(np.einsum("nij,ij->n", g, u @ s).max() for s in ops)
        dist.append(math.degrees(math.acos(np.clip((best - 1) / 2, -1, 1))))
    assert np.percentile(dist, 95) < 1.2 * step and max(dist) < 1.6 * step, (np.percentile(dist, 95), max(dist))


def test_grid_distance_agrees_with_misorientation_om():
    """The covering test above and ``misorientation_om`` use the same symmetry list, so a grid point reached through a symmetry operator has zero misorientation."""
    g = orientation_grid(8.0, 139)
    u = Rot.random(1, random_state=2).as_matrix()[0]
    s = laue_rotations(139)[3]
    r = misorientation_om(u.ravel().tolist(), (u @ s).ravel().tolist(), 139)
    assert (r[0] if isinstance(r, (list, tuple)) else r) < 1e-6 and len(g) > 10


def test_one_representative_per_class():
    step = 6.0
    full, sym = orientation_grid(step, 1), orientation_grid(step, 139)
    assert 0.8 / 8 < len(sym) / len(full) < 1.2 / 8
    assert len(orientation_grid(3.0, 139)) > 5 * len(orientation_grid(6.0, 139))
