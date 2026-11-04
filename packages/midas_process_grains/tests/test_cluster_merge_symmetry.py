"""cluster_merge's twin path must use the crystal-side operator and reduce by symmetry.

The old code tested |<q_I, T.q_J>| (T on the lab side, no symmetry): it merged 0 of 40
random FCC Sigma3 twin pairs; its one test used an identity parent, where left- and
right-multiplication coincide. Twin pairs here have a random parent and a random
symmetry representative of the twin, built from rotation matrices independent of the
code under test.
"""
import math

import numpy as np
import pytest

from midas_process_grains.compute.cluster_merge import compute_cluster_merges
from midas_stress.orientation import make_symmetries, quat_to_orient_mat


def _rr(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _axis_angle(ax, deg):
    a = np.asarray(ax, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def _sym(sg):
    n, sym = make_symmetries(sg)
    return [np.asarray(quat_to_orient_mat(list(q)), float).reshape(3, 3) for q in sym]


POS = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])


def _merged(gp, gt, sg, **kw):
    r = compute_cluster_merges(
        cluster_orientation_matrices=np.stack([gp, gt]), cluster_positions_um=POS,
        space_group=sg, tol_misori_deg=2.0, tol_position_um=100.0, mode="twin", **kw)
    return r.n_out_parents == 1


def test_fcc_sigma3_with_random_parent_and_symmetry_representative():
    rng = np.random.default_rng(11)
    S = _sym(225)
    T = _axis_angle([1, 1, 1], 60.0)
    ok = sum(_merged(gp, gp @ T @ S[rng.integers(len(S))], 225)
             for gp in (_rr(rng) for _ in range(40)))
    assert ok == 40, f"{ok}/40"


def test_hcp_twins_built_from_the_midas_b_matrix_merge():
    hk = pytest.importorskip("midas_hkls")
    from midas_hkls.nf_hkls import _b_matrix
    ca = 1.587
    B = _b_matrix(hk.Lattice(1.0, 1.0, ca, 90.0, 90.0, 120.0))
    S = _sym(194)
    rng = np.random.default_rng(12)
    for hkl in ((1, 0, 2), (1, 0, 1), (1, 1, 1), (1, 1, 2)):      # {10-12} {10-11} {11-21} {11-22}
        n0 = B @ np.array(hkl, float)
        n0 /= np.linalg.norm(n0)
        ok = 0
        for _ in range(20):
            gp = _rr(rng)
            nv = S[rng.integers(12)] @ n0
            T = 2.0 * np.outer(nv, nv) - np.eye(3)
            ok += _merged(gp, gp @ T @ S[rng.integers(12)], 194, c_over_a=ca)
        assert ok == 20, f"{hkl}: {ok}/20"


def test_unrelated_orientations_do_not_merge_much():
    rng = np.random.default_rng(13)
    n = 80
    OM = np.stack([_rr(rng) for _ in range(n)])
    pos = np.zeros((n, 3))
    r = compute_cluster_merges(cluster_orientation_matrices=OM, cluster_positions_um=pos,
                               space_group=225, tol_misori_deg=2.0, tol_position_um=100.0,
                               mode="twin")
    # 3160 random pairs x 2 directions x the cubic twin set at 2 deg: measured 1 chance merge
    assert r.n_merges_twin <= 6, r.n_merges_twin
    assert r.n_out_parents >= n - 6


# ---- the direct (alt-indexing) branch -------------------------------------------------

def _small_rot(rng, deg):
    ax = rng.normal(size=3)
    return _axis_angle(ax, deg)


def _direct(gp, gq, sg, tol=2.0):
    r = compute_cluster_merges(
        cluster_orientation_matrices=np.stack([gp, gq]), cluster_positions_um=POS,
        space_group=sg, tol_misori_deg=tol, tol_position_um=100.0, mode="direct")
    return r.n_out_parents == 1


@pytest.mark.parametrize("sg", [225, 194, 14])
def test_direct_merges_the_same_grain_in_different_symmetry_representatives(sg):
    """An alt-indexing is the same crystal written with another symmetry representative
    (and ~1 deg of fit noise). The old raw-quaternion dot product merged only the lucky
    representatives (about 1 in n_sym)."""
    rng = np.random.default_rng(21)
    S = _sym(sg)
    ok = sum(_direct(gp, gp @ S[rng.integers(len(S))] @ _small_rot(rng, 1.0), sg)
             for gp in (_rr(rng) for _ in range(40)))
    assert ok == 40, f"SG {sg}: {ok}/40"


@pytest.mark.parametrize("sg", [225, 194])
def test_direct_does_not_merge_orientations_3_deg_apart(sg):
    rng = np.random.default_rng(22)
    S = _sym(sg)
    ok = sum(_direct(gp, gp @ S[rng.integers(len(S))] @ _small_rot(rng, 3.0), sg)
             for gp in (_rr(rng) for _ in range(30)))
    assert ok == 0, f"SG {sg}: {ok}/30"
