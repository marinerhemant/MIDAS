"""Twin identification, and the adjacency control that makes it meaningful.

The control is the point. Any large grain list contains chance CSL pairs, so a
twin fraction quoted without an adjacency null measures the size of the list,
not the microstructure. These tests pin both directions: a planted adjacent
twin population must be found AND flagged trustworthy, and a population of
correctly-oriented but randomly-placed pairs must be flagged NOT trustworthy.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_process_grains.compute.twins import default_fcc_twin_relations
from midas_process_grains.twins_posthoc import find_twins, tolerance_sweep


def _rot(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K @ K)


def _random_orients(n, rng):
    out = np.empty((n, 3, 3))
    for i in range(n):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        out[i] = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
        ])
    return out


def _planted(n_matrix=260, n_twin=40, *, adjacent=True, seed=0):
    """Matrix grains plus n_twin Σ3 partners, placed adjacent or at random."""
    rng = np.random.default_rng(seed)
    O = _random_orients(n_matrix, rng)
    P = rng.uniform(-150, 150, size=(n_matrix, 3))   # dense enough
                                                     # to give >=50
                                                     # neighbour pairs
    tw = _rot((1, 1, 1), 60.0)
    Ot, Pt = [], []
    for k in range(n_twin):
        parent = k % n_matrix
        # The twin rotation acts in the CRYSTAL frame, so it post-multiplies:
        # O @ tw gives exactly 60 deg about <111>, while tw @ O (sample frame)
        # reduces to 41.36 deg about an irrational axis and is not a twin at all.
        Ot.append(O[parent] @ tw)
        if adjacent:
            step = rng.normal(size=3)
            step = 12.0 * step / np.linalg.norm(step)      # touching
            Pt.append(P[parent] + step)
        else:
            Pt.append(rng.uniform(-150, 150, size=3))       # same orientations,
                                                            # unrelated positions
    return np.concatenate([O, np.array(Ot)]), np.concatenate([P, np.array(Pt)])


def test_finds_planted_adjacent_twins():
    O, P = _planted(adjacent=True)
    r = find_twins(O, P, 225, max_distance_um=45.0, n_far_samples=1500)
    assert len(r.pairs) >= 30, f"planted 40 adjacent twins, found {len(r.pairs)}"
    assert r.enrichment > 3.0
    assert r.trustworthy
    for i, j in r.pairs:
        assert np.linalg.norm(P[i] - P[j]) <= 45.0


def test_scattered_twins_are_flagged_untrustworthy():
    """Correct orientations, random positions ⇒ not twins, and the tool says so.

    This is the case a bare CSL count gets wrong.
    """
    O, P = _planted(adjacent=False, seed=3)
    r = find_twins(O, P, 225, max_distance_um=45.0, n_far_samples=1500)
    assert not r.trustworthy, (
        f"enrichment {r.enrichment:.2f} on randomly-placed pairs — the "
        "adjacency control failed to fire")


def test_no_twins_in_a_random_population():
    rng = np.random.default_rng(11)
    O = _random_orients(300, rng)
    P = rng.uniform(-500, 500, size=(300, 3))
    r = find_twins(O, P, 225, max_distance_um=45.0, n_far_samples=1500)
    assert r.twinned_fraction < 0.10
    assert not r.trustworthy


def test_partner_counts_and_pair_bookkeeping():
    O, P = _planted(n_matrix=120, n_twin=20, adjacent=True, seed=5)
    r = find_twins(O, P, 225, max_distance_um=45.0, n_far_samples=800)
    assert r.n_partners.sum() == 2 * len(r.pairs), "each pair counts for both grains"
    assert r.n_grains == len(O)
    assert r.residual_deg.shape[0] == len(r.pairs)
    assert len(r.variants) == len(r.pairs)
    assert np.all(r.residual_deg <= 1.0)     # residual from the ideal OR
    assert np.all(r.distance_um <= 45.0)


def test_variant_is_resolved_and_correct():
    """The reported variant must name the {111} plane actually planted."""
    rng = np.random.default_rng(4)
    O = _random_orients(150, rng)
    P = rng.uniform(-500, 500, size=(150, 3))
    rels = {r.name: r for r in default_fcc_twin_relations()}
    from midas_stress.orientation import quat_to_orient_mat
    want = "FCC_Sigma3_<-111>"
    T = np.asarray(quat_to_orient_mat(list(rels[want].quaternion)), float).reshape(3, 3)
    Ot = np.einsum("nij,jk->nik", O[:25], T)
    Pt = P[:25] + rng.normal(size=(25, 3)) * 4.0
    r = find_twins(np.concatenate([O, Ot]), np.concatenate([P, Pt]), 225,
                   max_distance_um=45.0, n_far_samples=800)
    assert len(r.pairs) >= 18, f"planted 25, found {len(r.pairs)}"
    from collections import Counter
    got = Counter(r.variants).most_common(1)[0][0]
    assert got == want, f"planted {want}, reported {got}"


def test_all_four_variants_are_found_and_balanced():
    """A population spread over all four {111} planes must come back balanced."""
    rng = np.random.default_rng(6)
    O = _random_orients(200, rng)
    P = rng.uniform(-600, 600, size=(200, 3))
    from midas_stress.orientation import quat_to_orient_mat
    Ot, Pt = [], []
    for k, rel in enumerate(default_fcc_twin_relations()):
        T = np.asarray(quat_to_orient_mat(list(rel.quaternion)), float).reshape(3, 3)
        for m in range(15):
            parent = k * 15 + m
            Ot.append(O[parent] @ T)
            Pt.append(P[parent] + rng.normal(size=3) * 4.0)
    r = find_twins(np.concatenate([O, np.array(Ot)]),
                   np.concatenate([P, np.array(Pt)]), 225,
                   max_distance_um=45.0, n_far_samples=800)
    assert len(r.variant_counts) == 4
    assert sum(v > 0 for v in r.variant_counts.values()) == 4, r.variant_counts
    assert r.variant_balance > 0.4, f"unbalanced: {r.variant_counts}"


def test_tolerance_sweep_reports_enrichment():
    """A real population is not sharply tolerance-sensitive."""
    O, P = _planted(adjacent=True)
    sw = tolerance_sweep(O, P, 225, tols=(0.5, 1.0, 2.0),
                         max_distance_um=45.0, n_far_samples=600)
    assert len(sw) == 3
    counts = [n for _, n, _, _ in sw]
    assert counts == sorted(counts), "a looser tolerance cannot find fewer pairs"
    assert all(e > 3.0 for *_, e in sw)


def test_rejects_mismatched_inputs():
    rng = np.random.default_rng(1)
    O = _random_orients(10, rng)
    with pytest.raises(ValueError):
        find_twins(O, rng.uniform(size=(9, 3)), 225)
    with pytest.raises(ValueError):
        find_twins(O, rng.uniform(size=(10, 3)), 225, relations=[])
