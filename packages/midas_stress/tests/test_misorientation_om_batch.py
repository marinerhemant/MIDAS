"""``misorientation_om_batch`` must equal the scalar ``misorientation_om``.

The batch function is the hot path of the pf-HEDM per-voxel clustering
(``midas_pipeline.find_grains._cluster.per_voxel_cluster``), which is
O(n_sol^2) in the number of indexer solutions per voxel. It used to be a
Python ``for`` loop over the scalar function, which rebuilt the symmetry
table for every pair and computed the fundamental zone five times per pair
(twice for an axis the batch function then threw away). These tests pin the
vectorized replacement to the scalar reference, element for element.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_stress.orientation import (
    make_symmetries,
    misorientation_om,
    misorientation_om_batch,
)

# One from each symmetry table branch of make_symmetries.
SPACE_GROUPS = [1, 5, 20, 90, 148, 166, 194, 200, 225]


def _scalar_reference(oms1, oms2, sg):
    """What the old implementation computed, pair by pair."""
    return np.array([
        misorientation_om(list(a), list(b), sg)[0]
        for a, b in zip(oms1, oms2)
    ])


@pytest.mark.parametrize("sg", SPACE_GROUPS)
def test_matches_scalar_on_random_orientations(sg):
    n = 200
    a = Rotation.random(n, random_state=11).as_matrix().reshape(n, 9)
    b = Rotation.random(n, random_state=22).as_matrix().reshape(n, 9)

    got = misorientation_om_batch(a, b, sg)
    want = _scalar_reference(a, b, sg)

    assert np.allclose(got, want, rtol=0, atol=1e-12), (
        f"sg={sg}: max |diff| = {np.max(np.abs(got - want)):.3e}"
    )


@pytest.mark.parametrize("sg", SPACE_GROUPS)
def test_identical_orientations_give_zero(sg):
    n = 50
    a = Rotation.random(n, random_state=7).as_matrix().reshape(n, 9)
    got = misorientation_om_batch(a, a, sg)
    assert np.all(got < 1e-6), f"sg={sg}: max {got.max():.3e} rad"


@pytest.mark.parametrize("sg", [166, 194, 225])
def test_symmetry_equivalent_orientations_give_zero(sg):
    """Applying a symmetry operator must not change the orientation."""
    n_sym, sym = make_symmetries(sg)
    base = Rotation.random(40, random_state=3)
    for k in range(n_sym):
        w, x, y, z = sym[k]
        # scipy wants (x, y, z, w)
        op = Rotation.from_quat([x, y, z, w])
        a = base.as_matrix().reshape(-1, 9)
        b = (base * op).as_matrix().reshape(-1, 9)
        got = misorientation_om_batch(a, b, sg)
        assert np.all(got < 1e-6), f"sg={sg} op={k}: max {got.max():.3e} rad"


def test_hits_every_orient_mat_to_quat_branch():
    """The four-branch quaternion conversion must match the scalar branch-for-branch.

    trace > 0 takes branch 0; the three 180-degree rotations about x, y, z each
    have trace = -1 and select branches 1, 2, 3 respectively.
    """
    mats = [np.eye(3)]
    for axis in ("x", "y", "z"):
        mats.append(Rotation.from_euler(axis, 180, degrees=True).as_matrix())
    # plus a near-degenerate case just past the trace boundary
    mats.append(Rotation.from_euler("x", 179.999, degrees=True).as_matrix())
    a = np.array([m.reshape(9) for m in mats])
    b = Rotation.random(len(mats), random_state=99).as_matrix().reshape(-1, 9)

    for sg in (166, 225):
        got = misorientation_om_batch(a, b, sg)
        want = _scalar_reference(a, b, sg)
        assert np.allclose(got, want, rtol=0, atol=1e-12), (
            f"sg={sg}: {np.max(np.abs(got - want)):.3e}"
        )


def test_accepts_3x3_stack():
    """The docstring has always promised (n, 3, 3); the scalar loop never honoured it."""
    n = 30
    r1 = Rotation.random(n, random_state=1).as_matrix()
    r2 = Rotation.random(n, random_state=2).as_matrix()
    flat = misorientation_om_batch(r1.reshape(n, 9), r2.reshape(n, 9), 166)
    stacked = misorientation_om_batch(r1, r2, 166)
    assert np.allclose(flat, stacked, rtol=0, atol=0)


def test_empty_input():
    out = misorientation_om_batch(np.empty((0, 9)), np.empty((0, 9)), 166)
    assert out.shape == (0,)


def test_angles_are_within_the_fundamental_zone():
    """A hexagonal misorientation angle cannot exceed ~93.8 degrees."""
    n = 500
    a = Rotation.random(n, random_state=41).as_matrix().reshape(n, 9)
    b = Rotation.random(n, random_state=42).as_matrix().reshape(n, 9)
    got = np.degrees(misorientation_om_batch(a, b, 194))
    assert got.min() >= 0.0
    assert got.max() <= 93.9, f"max {got.max():.3f} deg"


def test_is_actually_faster_than_the_scalar_loop():
    """Guard against a silent regression back to the per-pair Python path."""
    import time

    n = 2000
    a = Rotation.random(n, random_state=5).as_matrix().reshape(n, 9)
    b = Rotation.random(n, random_state=6).as_matrix().reshape(n, 9)

    t0 = time.perf_counter()
    misorientation_om_batch(a, b, 166)
    batched = time.perf_counter() - t0

    m = 100
    t0 = time.perf_counter()
    _scalar_reference(a[:m], b[:m], 166)
    per_pair_scalar = (time.perf_counter() - t0) / m

    per_pair_batched = batched / n
    speedup = per_pair_scalar / per_pair_batched
    assert speedup > 20.0, f"only {speedup:.1f}x faster than the scalar loop"
