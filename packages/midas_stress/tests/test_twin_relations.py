"""Twin / Sigma3 orientation relationships (midas_stress.twin_relations).

Orientation matrices are crystal -> lab; a twin is ``O_B = O_A . T . S``. For a hexagonal cell the
crystal frame is a choice: ``"midas"`` (a* along x, the frame of Grains.csv and of the
``midas_hkls`` B matrix; default) or ``"crystallographic"`` (a1 along x), 30 deg about c apart.
The 12 hexagonal operators and every disorientation ANGLE are identical in both, so only axes
(and tests that look at them) can see a frame error. The truth below is built from the B matrix
written out inline (hexagonal, a = 1).
"""
import math

import numpy as np
import pytest

import midas_stress.orientation as o
from midas_stress import (
    TwinRelation, build_twin_pair, default_fcc_twin_relations, default_hcp_twin_relations,
    default_twin_relations_for, find_twin_partners, relation_axis_family,
    relation_disorientation, relation_rotation_matrix,
)
from midas_stress.twin_relations import HCP_FRAMES, hcp_hkil_to_cartesian_normal

CA = 1.587
S3 = math.sqrt(3.0)
A_STAR = np.array([2.0 / S3, 0.0, 0.0])
B_STAR = np.array([1.0 / S3, 1.0, 0.0])
C_STAR = np.array([0.0, 0.0, 1.0 / CA])
B = np.stack([A_STAR, B_STAR, C_STAR], axis=1)
SYSTEMS = {                      # name: (3-index K1, disorientation deg for c/a = 1.587, axis family)
    "tension_1012": ((1, 0, 2), 85.00, "a"),
    "compression_1011": ((1, 0, 1), 57.24, "a"),
    "tension_1121": ((1, 1, 1), 34.98, "astar"),
    "compression_1122": ((1, 1, 2), 64.43, "astar"),
}


def _unit(v):
    return v / np.linalg.norm(v)


def _a_axes():
    a1 = _unit(np.cross(B_STAR, C_STAR))
    a2 = _unit(np.cross(C_STAR, A_STAR))
    return [a1, a2, _unit(-(a1 + a2))]


def _parallel(v, targets, tol=1e-9):
    return any(abs(abs(_unit(v) @ _unit(t)) - 1.0) < tol for t in targets)


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _sym(sg):
    n, sym = o.make_symmetries(sg)
    return [np.asarray(o.quat_to_orient_mat(list(q)), float).reshape(3, 3) for q in sym]


def _rot180(n):
    return 2.0 * np.outer(n, n) - np.eye(3)


def _random_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _axis_angle(ax, deg):
    a = _unit(np.asarray(ax, float))
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


# ---- operators ---------------------------------------------------------------------------

@pytest.mark.parametrize("hkil", [(1, 0, -1, 2), (1, 0, -1, 1), (1, 1, -2, 1), (1, 1, -2, 2),
                                  (2, -1, -1, 2), (0, 1, -1, 3)])
def test_hcp_normal_equals_the_b_matrix_direction(hkil):
    h, k, i, l = hkil
    np.testing.assert_allclose(hcp_hkil_to_cartesian_normal(h, k, i, l, CA),
                               _unit(B @ np.array([h, k, l], float)), atol=1e-12)


def test_frames_differ_by_30_degrees_about_c():
    assert HCP_FRAMES == ("midas", "crystallographic")
    m = hcp_hkil_to_cartesian_normal(1, 0, -1, 2, CA)
    t = hcp_hkil_to_cartesian_normal(1, 0, -1, 2, CA, frame="crystallographic")
    np.testing.assert_allclose(_rz(-30.0) @ t, m, atol=1e-12)
    with pytest.raises(ValueError):
        hcp_hkil_to_cartesian_normal(1, 0, -1, 2, CA, frame="bogus")


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_hcp_disorientation_angle_and_axis_from_the_b_matrix(name):
    (hkl, angle, fam) = SYSTEMS[name]
    targets = _a_axes() if fam == "a" else [A_STAR, B_STAR, A_STAR - B_STAR]
    rels = default_hcp_twin_relations(CA, systems=(name,))
    assert len(rels) == 6
    for rel in rels:
        a, ax = relation_disorientation(rel, 194)
        assert a == pytest.approx(angle, abs=0.05)
        assert _parallel(ax, targets)
    fam_axes = relation_axis_family(rels[0], 194)
    assert len(fam_axes) == 3 and all(_parallel(v, targets) for v in fam_axes)


def test_default_list_is_four_systems_and_compression_2112_is_an_alias_of_1122():
    assert len(default_hcp_twin_relations(CA)) == 24
    assert not any("compression_2112" in r.name for r in default_hcp_twin_relations(CA))
    a, _ = relation_disorientation(default_hcp_twin_relations(CA, systems=("compression_2112",))[0], 194)
    assert a == pytest.approx(64.43, abs=0.05)         # (2 -1 -1 2) = {11-22}


def test_fcc_sigma3_is_60_degrees_about_111_with_four_axes():
    rel = default_fcc_twin_relations()[0]
    a, ax = relation_disorientation(rel, 225)
    assert a == pytest.approx(60.0, abs=1e-9)
    assert _parallel(ax, [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    assert len(relation_axis_family(rel, 225)) == 4
    np.testing.assert_allclose(relation_rotation_matrix(rel), _axis_angle(rel.axis, 60.0), atol=1e-12)


def test_dispatcher_counts():
    assert len(default_twin_relations_for(194, c_over_a=CA)) == 24
    assert len(default_twin_relations_for(194, c_over_a=CA, hcp_frame="crystallographic")) == 24
    assert len(default_twin_relations_for(225)) == 4 + 3
    assert isinstance(default_twin_relations_for(225)[0], TwinRelation)


# ---- pair building and finding ------------------------------------------------------------

def _pairs(rel_sg_frame, name, n=40, seed=3, frame_of_data="midas"):
    """n ideal twin pairs built in the MIDAS B frame (random variant, random symmetry op), written
    in ``frame_of_data``; matrix grains labelled 0, twins 1, 5 um apart, pairs 50 um apart."""
    (hkl, _a, _f) = SYSTEMS[name]
    n0 = _unit(B @ np.array(hkl, float))
    S = _sym(194)
    rng = np.random.default_rng(seed)
    OM = []
    for _ in range(n):
        gp = _random_rot(rng)
        OM += [gp, gp @ _rot180(S[rng.integers(12)] @ n0) @ S[rng.integers(12)]]
    OM = np.array(OM)
    if frame_of_data == "crystallographic":
        OM = OM @ _rz(-30.0).T
    pos = np.repeat(np.arange(n, dtype=float)[:, None] * 50.0, 2, axis=0) * np.array([1, 0, 0])
    pos[1::2] += [5.0, 0.0, 0.0]
    return OM, pos, np.tile([0, 1], n)


def _found(OM, pos, lab, name, frame, **kw):
    rel = default_hcp_twin_relations(CA, systems=(name,), frame=frame)[0]
    return len(find_twin_partners(OM, pos, lab, rel, 194, k_NN=3, **kw)["pairs"])


def _true_found(OM, pos, lab, name, frame, **kw):
    """Planted twins (matrix g, twin g + 1) among the accepted pairs."""
    rel = default_hcp_twin_relations(CA, systems=(name,), frame=frame)[0]
    out = find_twin_partners(OM, pos, lab, rel, 194, k_NN=3, **kw)
    return sum(1 for g, tw in out["pairs"] if tw == g + 1)


# Unrelated neighbours that pass the weak in-plane axis gate are not bounded by the relation:
# 1000 seeds of this generator gave a maximum of 22 of 40 ({10-12}, wrong frame).
CHANCE_MAX = 25


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_find_twin_partners_hcp_frame_switch(name):
    OM, pos, lab = _pairs(None, name)
    assert _found(OM, pos, lab, name, "midas") == 40
    assert _true_found(OM, pos, lab, name, "crystallographic") == 0   # axes 30 deg off: |dot| = 0.866 < 0.9
    assert _found(OM, pos, lab, name, "crystallographic") <= CHANCE_MAX
    OMt, pos, lab = _pairs(None, name, frame_of_data="crystallographic")
    assert _found(OMt, pos, lab, name, "crystallographic") == 40
    assert _true_found(OMt, pos, lab, name, "midas") == 0
    assert _found(OMt, pos, lab, name, "midas") <= CHANCE_MAX


def test_find_twin_partners_fcc_sigma3_and_default_window():
    rng = np.random.default_rng(8)
    S = _sym(225)
    rel = default_fcc_twin_relations()[0]
    OM, n = [], 40
    for _ in range(n):
        gp = _random_rot(rng)
        OM += list(build_twin_pair(gp, rel))
    OM = np.array(OM)
    OM[1::2] = np.array([g @ S[rng.integers(24)] for g in OM[1::2]])      # random symmetry representative
    pos = np.repeat(np.arange(n, dtype=float)[:, None] * 50.0, 2, axis=0) * np.array([1, 0, 0])
    pos[1::2] += [5.0, 0.0, 0.0]
    lab = np.tile([0, 1], n)
    out = find_twin_partners(OM, pos, lab, rel, 225, k_NN=3, misori_low=55.0, misori_high=65.0)
    assert len(out["pairs"]) == n
    assert np.allclose(out["pair_misori"], 60.0, atol=1e-6)
    out_default = find_twin_partners(OM, pos, lab, rel, 225, k_NN=3)           # 60 +- 5
    assert len(out_default["pairs"]) == n


def test_build_twin_pair_is_the_exact_relation_and_validates_shape():
    rel = default_hcp_twin_relations(CA, systems=("tension_1012",))[0]
    R = _random_rot(np.random.default_rng(2))
    R0, Rt = build_twin_pair(R, rel)
    np.testing.assert_array_equal(R0, R)
    ang, ax = o.misorientation_om(R.ravel(), Rt.ravel(), space_group=194)
    a_ref, _ = relation_disorientation(rel, 194)
    assert math.degrees(ang) == pytest.approx(a_ref, abs=1e-9)
    with pytest.raises(ValueError):
        build_twin_pair(np.eye(4), rel)


def test_find_twin_partners_handles_empty_and_length_mismatch():
    rel = default_fcc_twin_relations()[0]
    OM = np.stack([np.eye(3)] * 3)
    pos = np.zeros((3, 3))
    out = find_twin_partners(OM, pos, np.array([0, 0, 0]), rel, 225)
    assert out["pairs"].shape == (0, 2)
    with pytest.raises(ValueError):
        find_twin_partners(OM, pos[:2], np.array([0, 0, 0]), rel, 225)


def test_hcp_frames_single_definition():
    from midas_stress import plasticity, twin_relations

    assert plasticity.HCP_FRAMES == twin_relations.HCP_FRAMES == ("midas", "crystallographic")
