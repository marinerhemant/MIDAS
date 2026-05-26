"""Stage-5 tests: twin + sub-grain labeling."""
from __future__ import annotations

import math

import numpy as np

from midas_process_grains.compute.twin_label import (
    label_twins, label_subgrains, _qmul,
)
from midas_process_grains.compute.twins import default_fcc_twin_relations


def test_no_twin_pairs_in_random_population():
    """A small population of random-ish orientations should have no
    twin partners under the 0.5° tolerance."""
    rng = np.random.default_rng(0)
    # 5 random unit quats
    q = rng.normal(size=(5, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = q * np.sign(q[:, :1])
    partner, family, ttype, npairs = label_twins(
        grain_quats=q, tol_deg=0.5,
    )
    assert npairs == 0
    assert (partner == -1).all()


def test_known_sigma3_pair_is_detected():
    """Construct two orientations differing by an exact Σ3 twin
    operator; the labeler should pair them."""
    twins = default_fcc_twin_relations()
    sigma3 = twins[0]
    q_a = np.array([1.0, 0.0, 0.0, 0.0])
    q_b = _qmul(q_a, sigma3.quaternion)
    q_b = q_b / np.linalg.norm(q_b)
    if q_b[0] < 0: q_b = -q_b
    q = np.stack([q_a, q_b])
    partner, family, ttype, npairs = label_twins(
        grain_quats=q, tol_deg=1.0,
    )
    # At least the pair (0, 1) is detected
    assert npairs >= 1
    assert partner[0] == 1 or partner[1] == 0
    # And they go into the same twin family
    assert family[0] == family[1]
    assert family[0] != -1


def test_subgrain_pair_within_thresholds():
    """Two near-identical orientations at nearby positions, sharing
    most spots, should be paired as sub-grains."""
    # Quat-A and a tiny rotation (~0.3°) about z
    q_a = np.array([1.0, 0.0, 0.0, 0.0])
    eps = math.radians(0.3) / 2.0
    q_b = np.array([math.cos(eps), 0.0, 0.0, math.sin(eps)])
    q = np.stack([q_a, q_b])
    pos = np.array([[0., 0., 0.], [10., 0., 0.]])
    spots = [set(range(100)), set(range(50, 150))]   # heavy overlap
    out, n_pairs = label_subgrains(
        grain_quats=q, grain_positions=pos, grain_spot_sets=spots,
        subgrain_max_deg=1.0, subgrain_min_jaccard=0.3,
        spatial_max_um=50.0,
    )
    assert n_pairs >= 1
    assert out[0] == 1 or out[1] == 0
