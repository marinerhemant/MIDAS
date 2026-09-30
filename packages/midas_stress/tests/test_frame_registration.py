"""find_frame_rotation: recover a continuous, off-grid remount rotation; refuse unrelated sets."""
import numpy as np
import pytest

from midas_stress.frame_registration import find_frame_rotation, frame_match_fraction, _sym_mats
from midas_stress.orientation import axis_angle_to_orient_mat


def _rand_R(n, rng):
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z),
                     2*(y*z-x*w), 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)


def _small_noise(n, sigma_deg, rng):
    ax = rng.normal(size=(n, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    ang = rng.normal(scale=sigma_deg, size=n)
    return np.array([np.asarray(axis_angle_to_orient_mat(a, t), float).reshape(3, 3) for a, t in zip(ax, ang)])


# Test-sized searches: the defaults (n_seeds 40, n_score 300, n_null 8) cost minutes per call on set sizes
# that only need a few votes to show a true rotation. The stated capability is checked at the defaults elsewhere
# (verify rounds: 18/18 recoveries, 0/195 false positives); these keep the suite to about 2 minutes.
# NOTE: the tests deliberately pass the SAME seed to the data generator and to the search (bin-edge seeds 3 and 7);
# that collided with the search's null stream before `_stream` decorrelated them.
FAST = dict(n_seeds=16, n_score=120, n_null=3)


def _make_sets(R_true, rng, n_b=200, n_shared=120, n_extra_a=80, sigma_deg=0.2):
    S = _sym_mats(225)
    B = _rand_R(n_b, rng)
    shared = rng.choice(n_b, n_shared, replace=False)
    # O_a = R^T O_b S_k (a random symmetry-equivalent), plus measurement noise
    Sk = S[rng.integers(0, len(S), n_shared)]
    A_shared = _small_noise(n_shared, sigma_deg, rng) @ (R_true.T[None] @ B[shared] @ Sk)
    A = np.concatenate([A_shared, _rand_R(n_extra_a, rng)])
    return A[rng.permutation(len(A))], B


def _angle_between(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)))


def test_recovers_continuous_offgrid_rotation():
    rng = np.random.default_rng(7)
    axis = np.array([0.31, -0.72, 0.62]); axis /= np.linalg.norm(axis)
    R_true = np.asarray(axis_angle_to_orient_mat(axis, 37.37), float).reshape(3, 3)
    A, B = _make_sets(R_true, rng)
    res = find_frame_rotation(A, B, 225, tol_deg=1.0, seed=1, **FAST)
    assert res.significant, res.summary()
    assert res.score > 0.5, res.summary()                # 180 of 300 A grains are shared
    assert res.identity_score < 0.1
    # R is defined up to nothing else here (sample-side), so compare directly
    assert _angle_between(res.R, R_true) < 0.3, res.summary()


def test_identity_frame_is_recovered_as_identity():
    rng = np.random.default_rng(11)
    A, B = _make_sets(np.eye(3), rng)
    res = find_frame_rotation(A, B, 225, tol_deg=1.0, seed=2, **FAST)
    assert res.significant
    assert res.angle_deg < 0.3


def test_unrelated_sets_are_not_significant():
    rng = np.random.default_rng(3)
    A, B = _rand_R(300, rng), _rand_R(300, rng)
    res = find_frame_rotation(A, B, 225, tol_deg=1.0, seed=3, **FAST)
    assert not res.significant, res.summary()


def test_match_fraction_agrees_with_result():
    rng = np.random.default_rng(5)
    R_true = np.asarray(axis_angle_to_orient_mat([0, 0, 1], 12.5), float).reshape(3, 3)
    A, B = _make_sets(R_true, rng)
    f = frame_match_fraction(R_true, A, B, 225, tol_deg=1.0)
    assert 0.5 < f < 0.7                                    # 120 / 200 shared


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        find_frame_rotation(np.zeros((5, 4)), np.zeros((5, 9)), 225)


@pytest.mark.parametrize("sd", [3, 7])
def test_rotation_on_quaternion_bin_edges_is_recovered(sd):
    """Regression (verify artifact lens, 2026-09-28): a rotation whose quaternion sits on voting-bin
    edges in two components (w = 5 cells, x ~ 57 cells; 169.99 deg about x) fragmented its votes over
    ~15 bins and was reported as 'no significant rotation' for these seeds."""
    from midas_stress.orientation import quat_to_orient_mat
    cell = np.sin(np.radians(1.0) / 2.0) * 2.0
    q = np.array([5 * cell, np.sqrt(1 - (5 * cell) ** 2), 0.0, 0.0])
    R_edge = np.asarray(quat_to_orient_mat(q / np.linalg.norm(q)), float).reshape(3, 3)
    A, B = _make_sets(R_edge, np.random.default_rng(sd))
    res = find_frame_rotation(A, B, 225, tol_deg=1.0, seed=sd, **FAST)
    assert res.significant, res.summary()
    assert _angle_between(res.R, R_edge) < 0.3, res.summary()


def test_clumped_voxel_orientations_do_not_fake_significance():
    """Regression (HAO NF vs FF, 2026-09-28): voxel maps repeat each grain's orientation dozens of
    times. A rotation matching ONE grain then matches all its voxels, while the random-orientation
    null has no clumps -> unrelated sets scored 12-15 % against a ~4.5 % null and one was called
    significant. Clumped UNRELATED sets must not be significant."""
    rng = np.random.default_rng(21)
    grains = _rand_R(25, rng)
    A = np.concatenate([_small_noise(30, 0.1, rng) @ g[None] for g in grains])   # 25 grains x 30 voxels
    B = _rand_R(2000, rng)
    res = find_frame_rotation(A, B, 225, tol_deg=1.0, seed=4, **FAST)
    assert not res.significant, res.summary()
