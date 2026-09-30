"""Pass A must compare every pair within tolerance, as C ProcessGrains does.

Regression: the spatial hash walked only lex-positive neighbour offsets AND kept
a pair only when i < j, so every cross-cell pair whose lower index sat in the
lex-greater cell was never compared -- about half of them. Two identical
orientations 2 um apart across a 5 um cell face both survived; C merges them.
On bt_20id_jul26b nf_sampleF layer 6 the default output kept 10 pairs < 5 um / < 0.1 deg,
and a 50 um Pass A left 68 pairs < 50 um / < 0.1 deg.

The reference below is the C loop itself (ProcessGrains.c:836-874): outer i
ascending, skip if dup; inner j ascending, skip if dup; mark j when both gates pass.
"""
import math

import numpy as np
import pytest

from midas_process_grains.compute.c_parity import pass_a_position_dedup
from test_c_parity import _identity_om, _make_opf, _rot_om


def _run(om, pos, tol_um=5.0, tol_deg=0.1):
    opf, _ = _make_opf(np.asarray(om), np.asarray(pos, float))
    return pass_a_position_dedup(grain_positions=np.arange(len(pos), dtype=np.int64),
                                 opf=opf, space_group=225,
                                 misori_tol_rad=math.radians(tol_deg), pos_tol_um=tol_um)


@pytest.mark.parametrize("pos", [
    [[1, 0, 0], [3, 0, 0]],      # same cell
    [[4, 0, 0], [6, 0, 0]],      # adjacent cell, lower index in the lower cell
    [[6, 0, 0], [4, 0, 0]],      # adjacent cell, lower index in the HIGHER cell (was missed)
    [[4, 6, 0], [6, 4, 0]],      # diagonal neighbour
    [[6, 4, 0], [4, 6, 0]],      # diagonal neighbour, reversed (was missed)
    [[4, 4, 6], [6, 6, 4]],      # 3-D diagonal
    [[6, 6, 4], [4, 4, 6]],      # 3-D diagonal, reversed
])
def test_close_identical_pair_is_merged_whatever_the_cell_layout(pos):
    om = [_identity_om(), _identity_om()]
    assert _run(om, pos).tolist() == [False, True]


def _c_reference(om, pos, tol_um, tol_deg):
    from midas_stress.orientation import misorientation_om_batch   # radians
    n = len(pos); dup = np.zeros(n, bool)
    for i in range(n):
        if dup[i]:
            continue
        for j in range(i + 1, n):
            if dup[j]:
                continue
            if np.linalg.norm(pos[i] - pos[j]) < tol_um and \
               misorientation_om_batch(om[i:i + 1], om[j:j + 1], 225)[0] < math.radians(tol_deg):
                dup[j] = True
    return dup


@pytest.mark.parametrize("seed,tol_um", [(0, 5.0), (1, 5.0), (2, 20.0), (3, 50.0)])
def test_matches_the_c_loop_on_random_clusters(seed, tol_um):
    """Clusters of near-identical grains scattered across cell boundaries."""
    rng = np.random.default_rng(seed)
    oms, poss = [], []
    for c in range(40):
        base = _rot_om(rng.normal(size=3), rng.uniform(0.2, 1.0))     # distinct grains
        centre = rng.uniform(-200, 200, size=3)
        for _ in range(rng.integers(1, 5)):
            if rng.random() < 0.7:     # a duplicate: 0-0.05 deg away, well inside 0.1
                oms.append(base @ _rot_om(rng.normal(size=3), math.radians(rng.uniform(0, 0.05))))
            else:                      # a genuinely different grain: >= 1 deg away
                oms.append(base @ _rot_om(rng.normal(size=3), math.radians(rng.uniform(1, 5))))
            poss.append(centre + rng.normal(scale=tol_um * 0.6, size=3))
    om, pos = np.array(oms), np.array(poss)
    order = rng.permutation(len(pos))              # indices unrelated to cell layout
    om, pos = om[order], pos[order]
    got = _run(om, pos, tol_um=tol_um)
    ref = _c_reference(om, pos, tol_um, 0.1)
    assert got.tolist() == ref.tolist()
    assert ref.sum() > 0, "the fixture must contain merges to be a test"
