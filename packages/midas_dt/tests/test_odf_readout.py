"""Known-answer gates for the ODF readout and the near-duplicate node merge (orientation math is midas_stress's; these check the wiring)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("midas_stress")
from scipy.spatial.transform import Rotation  # noqa: E402

from midas_dt.odf_readout import VoxelODF, misorientation_matrix_deg, readout_table, voxel_odf  # noqa: E402
from midas_dt.texture_nodes import NodeBasis, merge_nodes, _laue_matrices  # noqa: E402

SG = 225
AXIS = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)


def rot(deg, base=None):
    r = Rotation.from_rotvec(AXIS * np.radians(deg)).as_matrix()
    return r if base is None else r @ base


@pytest.fixture(scope="module")
def base():
    return Rotation.random(1, random_state=2).as_matrix()[0]


def make_nodes(base, angles):
    return np.array([rot(a, base) for a in angles])


def test_misorientation_wiring(base):
    d = misorientation_matrix_deg(make_nodes(base, [0, 10, 30]), make_nodes(base, [0]), SG)[:, 0]
    np.testing.assert_allclose(d, [0, 10, 30], atol=1e-6)


def test_perfect_delta(base):
    nodes = make_nodes(base, [0, 10, 25, 40])
    c = np.zeros(4)
    c[2] = 3.0
    odf = voxel_odf(c, nodes, space_group=SG, floor=0.0)
    assert len(odf) == 1 and odf.index[0] == 2 and odf.weight[0] == pytest.approx(1.0)
    assert odf.medoid() == 2
    U, mass = odf.mean_orientation(radius_deg=5.0)
    assert mass == pytest.approx(1.0)
    assert misorientation_matrix_deg(U[None], nodes[2:3], SG)[0, 0] < 1e-6
    assert odf.mass_within(nodes[2:3], 0.1) == pytest.approx(1.0)
    assert odf.w1(nodes[2:3]) == pytest.approx(0.0, abs=1e-9)
    assert odf.top_weight == pytest.approx(1.0)


def test_w1_of_two_atoms_ten_degrees_apart(base):
    """0.5 / 0.5 on two atoms 10 deg apart against one of them: half the mass moves 10 deg, W1 = 5 deg."""
    nodes = make_nodes(base, [0, 10])
    odf = voxel_odf(np.array([1.0, 1.0]), nodes, space_group=SG, floor=0.0)
    assert odf.w1(nodes[0:1]) == pytest.approx(5.0, abs=1e-5)
    assert odf.w1(nodes[1:2]) == pytest.approx(5.0, abs=1e-5)
    # two-point reference, uniform: zero when it is the same set, and the transport can pair across
    assert odf.w1(nodes) == pytest.approx(0.0, abs=1e-6)
    far = make_nodes(base, [0, 20])
    assert odf.w1(far) == pytest.approx(5.0, abs=1e-5)           # 0.5*0 + 0.5*10: the 10 deg atom pairs with the 20 deg reference


def test_mass_within_tolerance_and_floor(base):
    nodes = make_nodes(base, [0, 10, 30, 50])
    c = np.array([0.5, 0.3, 0.17, 0.03])
    odf = voxel_odf(c, nodes, space_group=SG, floor=0.0)
    ref = nodes[0:1]
    assert odf.mass_within(ref, 1.5) == pytest.approx(0.5)
    assert odf.mass_within(ref, 12.0) == pytest.approx(0.8)
    assert odf.mass_within(ref, 35.0) == pytest.approx(0.97)
    assert odf.mass_within(ref, 60.0) == pytest.approx(1.0)
    # the floor drops the 3 % atom and renormalises; W1 depends on it
    f5 = voxel_odf(c, nodes, space_group=SG, floor=0.05)
    assert len(f5) == 3 and f5.weight.sum() == pytest.approx(1.0) and f5.dropped == pytest.approx(0.03)
    assert f5.mass_within(ref, 12.0) == pytest.approx(0.8 / 0.97)
    assert odf.w1(ref) != pytest.approx(f5.w1(ref))
    top2 = voxel_odf(c, nodes, space_group=SG, floor=0.0, top=2)
    assert top2.index.tolist() == [0, 1] and top2.weight.sum() == pytest.approx(1.0)
    assert odf.top(2)[0].tolist() == [0, 1]


def test_floor_is_required(base):
    with pytest.raises(TypeError):
        voxel_odf(np.array([1.0, 1.0]), make_nodes(base, [0, 1]), space_group=SG)          # no floor given
    with pytest.raises(ValueError):
        voxel_odf(np.array([1.0, 1.0]), make_nodes(base, [0, 1]), space_group=SG, floor=1.5)


def test_medoid_and_weighted_mean(base):
    nodes = make_nodes(base, [0, 10, 20])
    odf = voxel_odf(np.array([0.2, 0.5, 0.3]), nodes, space_group=SG, floor=0.0)
    assert odf.medoid() == 1                                    # weighted costs: 11, 5, 9
    U, mass = odf.mean_orientation(radius_deg=12.0)             # the 0 and 10 deg atoms (the 20 deg one is 10 deg from the medoid -> in)
    assert mass == pytest.approx(1.0)
    # weighted mean angle ~ 0.2*0 + 0.5*10 + 0.3*20 = 11 deg (small angles about one axis)
    assert misorientation_matrix_deg(U[None], nodes[0:1], SG)[0, 0] == pytest.approx(11.0, abs=0.1)
    U2, mass2 = odf.mean_orientation(radius_deg=5.0)            # only the medoid itself
    assert mass2 == pytest.approx(0.5)
    assert misorientation_matrix_deg(U2[None], nodes[1:2], SG)[0, 0] < 1e-6


def test_symmetry_equivalent_atoms_are_the_same_orientation(base):
    """U and U.S are one orientation: zero misorientation, mean reduces to it (a raw quaternion mean of the pair would not)."""
    sym = _laue_matrices(SG)
    nodes = np.array([base, base @ sym[7]])
    assert misorientation_matrix_deg(nodes[0:1], nodes[1:2], SG)[0, 0] < 1e-5
    odf = voxel_odf(np.array([1.0, 1.0]), nodes, space_group=SG, floor=0.0)
    U, mass = odf.mean_orientation(radius_deg=1.0)
    assert mass == pytest.approx(1.0)
    assert misorientation_matrix_deg(U[None], base[None], SG)[0, 0] < 1e-5
    assert odf.w1(base[None]) == pytest.approx(0.0, abs=1e-6)


def test_merge_nodes_matches_group_orientations_and_conserves_weight(base):
    from midas_stress import group_orientations

    rng = np.random.default_rng(0)
    sym = _laue_matrices(SG)
    centres = Rotation.random(12, random_state=8).as_matrix()
    oms = []
    for c in centres:
        for _ in range(4):                                       # four near-duplicates (0.1 deg scatter), some written as symmetric variants
            d = Rotation.from_rotvec(rng.normal(scale=np.radians(0.1) / np.sqrt(3), size=3)).as_matrix()
            oms.append(d @ c @ sym[rng.integers(len(sym))])
    oms = np.array(oms)
    pr = rng.random(len(oms))
    m = merge_nodes(oms, 0.5, SG, priority=pr)
    labels, reps = group_orientations(oms, SG, np.radians(0.5), priority=pr)
    assert sorted(m.reps.tolist()) == sorted(reps.tolist())
    assert len(m.reps) == 12
    coef = rng.random((3, len(oms)))
    merged = m.merge_coefficients(coef)
    assert merged.shape == (3, 12)
    np.testing.assert_allclose(merged.sum(axis=1), coef.sum(axis=1))
    # tolerance zero merges nothing, huge tolerance merges everything
    assert len(merge_nodes(oms, 1e-6, SG).reps) == len(oms)
    assert len(merge_nodes(oms, 90.0, SG).reps) == 1


def test_node_basis_validation_merge_and_voxel_merge(base):
    nodes = make_nodes(base, [0, 0.2, 10, 10.1, 30])
    with pytest.raises(ValueError):
        NodeBasis(nodes * 1.1, 1.0, SG)
    with pytest.raises(ValueError):
        NodeBasis(nodes, -1.0, SG)
    nb = NodeBasis(nodes, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), SG)
    small, mg = nb.merge(0.5, priority=np.array([5, 1, 4, 2, 3.0]))
    assert len(small) == 3 and mg.reps.tolist() == [0, 2, 4]
    np.testing.assert_allclose(small.sigma_deg, [1.0, 3.0, 5.0])
    odf = voxel_odf(np.array([0.3, 0.2, 0.2, 0.2, 0.1]), nodes, space_group=SG, floor=0.0)
    mo = odf.merged(0.5)
    assert len(mo) == 3 and mo.weight.sum() == pytest.approx(1.0)
    assert mo.weight[0] == pytest.approx(0.5) and mo.weight[1] == pytest.approx(0.4)
    assert mo.index.tolist() == [0, 2, 4]


def test_readout_table_reports_every_floor(base):
    nodes = make_nodes(base, [0, 10, 30])
    coef = np.array([[1.0, 0.0, 0.0], [0.6, 0.35, 0.05], [0.0, 0.0, 0.0]])
    truth = [nodes[0:1], nodes[0:1], nodes[0:1]]
    tab = readout_table(coef, nodes, truth, space_group=SG, floors=(0.0, 0.1), tol_deg=1.5)
    assert set(tab) == {0.0, 0.1}
    assert tab[0.0]["mass"][0] == pytest.approx(1.0) and tab[0.0]["mass"][1] == pytest.approx(0.6)
    assert tab[0.0]["w1"][1] == pytest.approx(0.35 * 10 + 0.05 * 30)
    assert tab[0.1]["w1"][1] == pytest.approx(0.35 / 0.95 * 10)
    assert np.isnan(tab[0.0]["w1"][2])
    assert tab[0.1]["n_atoms"][1] == 2


# ------------------------------------------- gates added after a mutation check
def test_mass_within_uses_the_nearest_reference_member(base):
    """A reference SET: an atom counts when it is near ANY member, not when it is near all of them."""
    nodes = make_nodes(base, [0, 40])
    odf = voxel_odf(np.array([0.3, 0.7]), nodes, space_group=SG, floor=0.0)
    ref = make_nodes(base, [0, 100])
    assert odf.mass_within(ref, 1.5) == pytest.approx(0.3)
    assert odf.mass_within(make_nodes(base, [0, 40]), 1.5) == pytest.approx(1.0)


def test_w1_honours_reference_weights(base):
    """Non-uniform reference weights, both the single-atom shortcut and the transport problem; swapping the weights changes the answer."""
    one = voxel_odf(np.array([1.0, 0.0]), make_nodes(base, [0, 5]), space_group=SG, floor=0.0)
    assert len(one) == 1
    ref = make_nodes(base, [10, 30])
    assert one.w1(ref, [0.8, 0.2]) == pytest.approx(0.8 * 10 + 0.2 * 30, abs=1e-5)
    assert one.w1(ref, [0.2, 0.8]) == pytest.approx(0.2 * 10 + 0.8 * 30, abs=1e-5)
    two = voxel_odf(np.array([0.8, 0.2]), make_nodes(base, [0, 20]), space_group=SG, floor=0.0)
    ref2 = make_nodes(base, [0, 20])
    assert two.w1(ref2, [0.9, 0.1]) == pytest.approx(0.1 * 20, abs=1e-5)        # only 0.1 of the mass moves, 20 deg
    assert two.w1(ref2, [0.1, 0.9]) == pytest.approx(0.7 * 20, abs=1e-5)        # 0.7 moves
    assert two.w1(ref2, [0.8, 0.2]) == pytest.approx(0.0, abs=1e-6)


def test_merge_keeps_the_heaviest_atom_and_respects_the_tolerance(base):
    """The representative of a cluster is its highest-priority member (not the first by index); the tolerance is the one asked for."""
    nodes = make_nodes(base, [0, 0.2])
    odf = voxel_odf(np.array([0.2, 0.8]), nodes, space_group=SG, floor=0.0)
    mo = odf.merged(0.5)
    assert len(mo) == 1 and mo.index.tolist() == [1] and mo.weight[0] == pytest.approx(1.0)
    # voxel_odf always hands over atoms heaviest first, so the priority passed by merged() only matters for a hand-built, unsorted VoxelODF
    unsorted = VoxelODF(index=np.array([0, 1]), weight=np.array([0.2, 0.8]), orientations=nodes, space_group=SG, floor=0.0, total=1.0,
                        dropped=0.0)
    assert unsorted.merged(0.5).index.tolist() == [1]
    m = merge_nodes(nodes, 0.5, SG, priority=np.array([1.0, 2.0]))
    assert m.reps.tolist() == [1]
    # three nodes 0.7 deg apart: below the tolerance two merge, at 0.65 none (a doubled tolerance would merge more)
    chain = make_nodes(base, [0, 0.7, 1.4])
    assert len(merge_nodes(chain, 0.65, SG).reps) == 3
    assert len(merge_nodes(chain, 0.75, SG).reps) == 2
