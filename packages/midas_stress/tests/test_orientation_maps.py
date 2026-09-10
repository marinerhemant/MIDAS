"""Map-level orientation statistics: KAM, GROD, medoid, axis mean."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from midas_stress import (mean_axis, axis_spread, medoid_orientation, grod,
                          kam, grid_neighbours)

SG = 225


def _tilted(n, spread_deg, seed=0):
    """n orientations scattered by `spread_deg` about a common one."""
    rng = np.random.default_rng(seed)
    U0 = R.random(random_state=seed).as_matrix()
    out = []
    for _ in range(n):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        out.append(R.from_rotvec(ax*np.radians(rng.normal(0, spread_deg))).as_matrix() @ U0)
    return np.array(out), U0


class TestMeanAxis:
    def test_sign_flips_do_not_move_the_mean(self):
        """The failure this function exists to prevent.

        A crystal direction has no sign. Flipping half the vectors must not
        change the answer; for a vector mean it changes it completely (measured
        on real data: 22.6 deg of 'spread' that was pure sign bookkeeping).
        """
        rng = np.random.default_rng(0)
        v = np.array([0.1, 0.2, 1.0]) + rng.normal(0, 0.02, (200, 3))
        flipped = v.copy(); flipped[::2] *= -1.0
        a, b = mean_axis(v), mean_axis(flipped)
        # clip: |a.b| can land a few ulp above 1 and arccos then gives NaN,
        # which silently passes a naive '< tol' comparison as False
        assert np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))) < 1e-6
        # and the naive vector mean does NOT survive the same flip
        m1 = v.mean(0)/np.linalg.norm(v.mean(0))
        m2 = flipped.mean(0)/np.linalg.norm(flipped.mean(0))
        assert np.degrees(np.arccos(np.clip(abs(m1 @ m2), 0, 1))) > 5.0

    def test_spread_is_sign_insensitive_and_bounded(self):
        rng = np.random.default_rng(1)
        v = np.array([0., 0., 1.]) + rng.normal(0, 0.05, (100, 3))
        v[::3] *= -1.0
        s = axis_spread(v)
        assert s.min() >= 0.0 and s.max() <= np.pi/2 + 1e-12
        assert np.degrees(np.median(s)) < 6.0


class TestGrodMedoid:
    def test_medoid_is_central_and_grod_is_zero_there(self):
        oms, _ = _tilted(40, 1.0, seed=3)
        i = medoid_orientation(oms, SG)
        g = grod(oms, SG)
        assert g[i] == pytest.approx(0.0, abs=1e-9)
        assert np.argmin(g) == i

    def test_grod_scales_with_the_planted_spread(self):
        for spread in (0.5, 2.0, 8.0):
            oms, _ = _tilted(40, spread, seed=7)
            med = np.degrees(np.median(grod(oms, SG)))
            assert 0.3*spread < med < 3.0*spread, (spread, med)

    def test_explicit_reference_is_honoured(self):
        oms, U0 = _tilted(20, 1.0, seed=5)
        g = grod(oms, SG, reference=U0)
        assert not np.allclose(g, grod(oms, SG))
        assert np.degrees(g).max() < 10.0


class TestKam:
    def test_cross_boundary_neighbours_swamp_kam_unless_excluded(self):
        """Why `max_angle` exists.

        Two domains 30 deg apart, each internally 0.2 deg. Without max_angle the
        sites at the wall report the WALL, not local curvature -- on real S5
        data that was 11-14 deg against a 0.44 deg interior.
        """
        a, _ = _tilted(8, 0.2, seed=1)
        b0 = R.from_rotvec(np.array([0., 0., 1.])*np.radians(30.0)).as_matrix()
        b = np.array([b0 @ m for m in _tilted(8, 0.2, seed=2)[0]])
        oms = np.concatenate([a, b])
        idx = list(range(16))                      # 1 x 16 raster: wall at 7|8
        nb = grid_neighbours(idx, (1, 16))
        loose = kam(oms, nb, SG)
        tight = kam(oms, nb, SG, max_angle=np.radians(5.0))
        assert np.degrees(loose[7]) > 5.0, "the wall must show up when not excluded"
        assert np.degrees(tight[7]) < 1.0, "max_angle must remove the wall pair"
        assert np.degrees(np.nanmedian(tight)) < 1.0

    def test_isolated_site_is_nan_not_zero(self):
        """Zero is a legitimate KAM; a missing neighbour must not manufacture it."""
        oms, _ = _tilted(3, 0.5, seed=4)
        nb = grid_neighbours([0, 1, 50], (10, 10))   # site 50 is far away
        out = kam(oms, nb, SG)
        assert np.isnan(out[2])
        assert not np.isnan(out[0])

    def test_neighbour_shape_is_validated(self):
        oms, _ = _tilted(4, 0.5, seed=6)
        with pytest.raises(ValueError, match=r"\(n, k\)"):
            kam(oms, np.zeros((3, 4), int), SG)


class TestGridNeighbours:
    def test_unmeasured_sites_are_not_neighbours(self):
        """A hole in the map must break the kernel, not be treated as coincident."""
        nb = grid_neighbours([0, 1, 3], (1, 4))      # site 2 missing
        assert (nb[1] == 2).sum() == 0
        assert set(nb[1][nb[1] >= 0].tolist()) == {0}

    def test_connectivity_8_adds_the_diagonals(self):
        idx = list(range(9))
        n4 = grid_neighbours(idx, (3, 3), connectivity=4)
        n8 = grid_neighbours(idx, (3, 3), connectivity=8)
        assert (n4[4] >= 0).sum() == 4 and (n8[4] >= 0).sum() == 8

    def test_rejects_bad_connectivity(self):
        with pytest.raises(ValueError, match="4 or 8"):
            grid_neighbours([0], (1, 1), connectivity=6)


class TestGroupOrientations:
    def test_finds_the_planted_groups(self):
        from midas_stress import group_orientations
        a, _ = _tilted(12, 0.3, seed=11)
        b0 = R.from_rotvec(np.array([0., 1., 0.])*np.radians(25.0)).as_matrix()
        b = np.array([b0 @ m for m in _tilted(7, 0.3, seed=12)[0]])
        lab, reps = group_orientations(np.concatenate([a, b]), SG, np.radians(5.0))
        assert len(reps) == 2
        assert sorted(np.bincount(lab).tolist(), reverse=True) == [12, 7]
        assert lab[0] == 0 and lab[-1] == 1        # label 0 is the LARGEST

    def test_priority_changes_the_representatives(self):
        """Documents the order-dependence rather than pretending it away.

        A CHAIN of orientations 4 deg apart spanning 32 deg, grouped at 5 deg.
        There is no natural partition: walking the chain forwards and backwards
        opens representatives at different links. This is exactly the situation
        that made a real group come out at 236 members one way and 234 another,
        and it is why the docstring refuses to call the sizes stable.
        """
        from midas_stress import group_orientations
        U0 = R.random(random_state=13).as_matrix()
        oms = np.array([R.from_rotvec(np.array([0., 0., 1.])
                                      * np.radians(4.0*i)).as_matrix() @ U0
                        for i in range(9)])
        l1, r1 = group_orientations(oms, SG, np.radians(5.0))
        l2, r2 = group_orientations(oms, SG, np.radians(5.0),
                                    priority=np.arange(len(oms)))
        assert not (np.array_equal(r1, r2) and np.array_equal(l1, l2)), \
            "priority must actually steer the greedy picks"

    def test_single_group_when_tolerance_is_wide(self):
        from midas_stress import group_orientations
        oms, _ = _tilted(15, 1.0, seed=14)
        lab, reps = group_orientations(oms, SG, np.radians(60.0))
        assert len(reps) == 1 and (lab == 0).all()

    def test_empty_input(self):
        from midas_stress import group_orientations
        lab, reps = group_orientations(np.zeros((0, 3, 3)), SG, 0.1)
        assert len(lab) == 0 and len(reps) == 0
