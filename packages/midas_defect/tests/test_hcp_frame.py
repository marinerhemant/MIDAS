"""HCP tables and the Sigma3 gate must be in the crystal frame the orientations live in.

MIDAS ``Grains.csv`` orientation matrices (crystal -> lab) live in the ``midas_hkls``
B-matrix frame: ``g = B (h, k, l)``, a* along x, b* at 60 deg in the xy plane, c* along
z, so real-space a is at -30 deg (150 deg) and b at 90 deg. The textbook frame (a1
along x) is 30 deg about c from it. Disorientation angles and the 12 hexagonal
operators cannot tell them apart, so the old tests (angles, unit norms, a1 = e_x) never
noticed. The truth below is built from the B matrix, written out inline (hexagonal,
a = 1), so nothing here depends on the code under test.
"""
import math

import numpy as np
import pytest

import midas_stress.orientation as o
from midas_defect.phases.hcp import (
    HCP_FRAMES, direction_bravais_to_cart, hcp_systems, plane_normal_bravais_to_cart,
)
from midas_defect.types import CrystalPhase
from midas_defect.variants import build_sigma3_pair, find_sigma3_partners

CA = 1.587
S3 = math.sqrt(3.0)
A_STAR = np.array([2.0 / S3, 0.0, 0.0])
B_STAR = np.array([1.0 / S3, 1.0, 0.0])
C_STAR = np.array([0.0, 0.0, 1.0 / CA])
B = np.stack([A_STAR, B_STAR, C_STAR], axis=1)


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


def _d6():
    g = [_rz(60 * k) for k in range(6)]
    return np.array(g + [m @ np.diag([1.0, -1.0, -1.0]) for m in g])


def _rot180(n):
    return 2.0 * np.outer(n, n) - np.eye(3)


def _random_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_frames_constant_and_default():
    assert HCP_FRAMES == ("midas", "crystallographic")
    np.testing.assert_allclose(
        hcp_systems("basal_a", CA), hcp_systems("basal_a", CA, frame="midas"))
    with pytest.raises(ValueError):
        hcp_systems("basal_a", CA, frame="bogus")


@pytest.mark.parametrize("hkil", [(1, 0, -1, 2), (1, 0, -1, 1), (1, 1, -2, 1),
                                  (1, 1, -2, 2), (2, -1, -1, 2), (0, 1, -1, 3)])
def test_plane_normal_equals_b_matrix_direction(hkil):
    h, k, i, l = hkil
    np.testing.assert_allclose(
        plane_normal_bravais_to_cart(h, k, i, l, c_over_a=CA),
        _unit(B @ np.array([h, k, l], float)), atol=1e-12)


def test_a_directions_are_the_real_space_a_axes_in_the_midas_frame():
    a_axes = _a_axes()
    for fam in ("basal_a", "prismatic_a", "pyramidal_a"):
        sysm = hcp_systems(fam, CA)
        assert all(_parallel(b, a_axes) for b in sysm[:, 1]), fam
    # the textbook frame is the one with a1 along x, i.e. NOT the real a axes of the B frame
    textbook = hcp_systems("basal_a", CA, frame="crystallographic")
    assert not all(_parallel(b, a_axes) for b in textbook[:, 1])


def test_prismatic_normals_are_a_star_axes_and_systems_are_orthogonal_in_both_frames():
    n = hcp_systems("prismatic_a", CA)[:, 0]
    assert all(_parallel(v, [A_STAR, B_STAR, A_STAR - B_STAR]) for v in n)
    for fr in HCP_FRAMES:
        for fam in ("basal_a", "prismatic_a", "pyramidal_a", "pyramidal_ca",
                    "twin_tensile", "twin_compressive"):
            sysm = hcp_systems(fam, CA, frame=fr)
            assert np.allclose((sysm[:, 0] * sysm[:, 1]).sum(1), 0.0, atol=1e-9)


def test_frames_differ_by_a_rotation_of_30_degrees_about_c():
    for fam in ("basal_a", "pyramidal_a", "twin_tensile"):
        m = hcp_systems(fam, CA)
        t = hcp_systems(fam, CA, frame="crystallographic")
        np.testing.assert_allclose(m, t @ _rz(-30.0).T, atol=1e-12)


def test_agrees_with_midas_hkls_frame_rotation_when_available():
    fr = pytest.importorskip("midas_hkls")
    if not hasattr(fr, "frame_rotation"):
        pytest.skip("midas_hkls without frame_rotation")
    R = fr.frame_rotation((1.0, 1.0, CA, 90, 90, 120))
    t = hcp_systems("pyramidal_ca", CA, frame="crystallographic")
    np.testing.assert_allclose(hcp_systems("pyramidal_ca", CA), t @ R.T, atol=1e-12)


# ---- the twin gate ------------------------------------------------------------------
# The orientation relationships live in midas_stress.twin_relations; midas_defect's
# find_sigma3_partners / build_sigma3_pair select one by phase (HCP: ``hcp_twin``, ``c_over_a``,
# ``frame``) and the gate's axis family is derived from that relation, so it is always in the
# same crystal frame as the operator.

TWINS = {  # hcp_twin: 3-index (h, k, l) of K1
    "tension_1012": (1, 0, 2),
    "compression_1011": (1, 0, 1),
    "tension_1121": (1, 1, 1),
    "compression_1122": (1, 1, 2),
}


def _pairs(frame, name, n=40, seed=3):
    """n ideal twin pairs built in the MIDAS B frame (random variant and symmetry op),
    re-expressed in ``frame`` if it is the textbook one."""
    n0 = _unit(B @ np.array(TWINS[name], float))
    rng = np.random.default_rng(seed)
    G = _d6()
    OM = []
    for _ in range(n):
        gp = _random_rot(rng)
        gt = gp @ _rot180(G[rng.integers(12)] @ n0) @ G[rng.integers(12)]
        OM += [gp, gt]
    OM = np.array(OM)
    if frame == "crystallographic":     # same crystals, orientations written in the textbook frame
        OM = OM @ _rz(-30.0).T
    pos = np.repeat(np.arange(n, dtype=float)[:, None] * 50.0, 2, axis=0) * np.array([1, 0, 0])
    pos[1::2] += [5.0, 0.0, 0.0]
    return OM, pos, np.tile([0, 1], n)


def _accepted(OM, pos, labels, name, **kw):
    out = find_sigma3_partners(OM, pos, labels, k_NN=3, phase=CrystalPhase.HCP,
                               c_over_a=CA, hcp_twin=name, **kw)
    return len(out["pairs"])


def _true_found(OM, pos, labels, name, **kw):
    """How many of the planted twins (matrix g, twin g + 1) are among the accepted pairs."""
    out = find_sigma3_partners(OM, pos, labels, k_NN=3, phase=CrystalPhase.HCP,
                               c_over_a=CA, hcp_twin=name, **kw)
    return sum(1 for g, tw in out["pairs"] if tw == g + 1)


# Chance partners (unrelated neighbours that pass the weak in-plane axis gate) are not bounded
# by the relation: 1000 seeds of this generator gave a maximum of 22 of 40 ({10-12}, wrong frame).
CHANCE_MAX = 25


@pytest.mark.parametrize("name", list(TWINS))
def test_gate_accepts_each_hcp_twin_in_the_matching_frame_and_mostly_rejects_the_other(name):
    """0 of 40 {11-21}/{11-22} pairs were accepted for MIDAS-frame data before the frame option
    existed (alignment 0.866 < 0.9 against a hand-written textbook <1-100> family)."""
    OM, pos, lab = _pairs("midas", name)
    assert _accepted(OM, pos, lab, name) == 40                                     # default = midas
    # wrong frame: the planted twin is never the accepted partner (its axis is 30 deg off, |dot| =
    # 0.866 < 0.9); what passes are chance neighbours, which the gate does not bound
    assert _true_found(OM, pos, lab, name, frame="crystallographic") == 0
    assert _accepted(OM, pos, lab, name, frame="crystallographic") <= CHANCE_MAX
    OMt, pos, lab = _pairs("crystallographic", name)
    assert _accepted(OMt, pos, lab, name, frame="crystallographic") == 40
    assert _true_found(OMt, pos, lab, name) == 0                                   # and the reverse
    assert _accepted(OMt, pos, lab, name) <= CHANCE_MAX


def test_the_gate_axis_family_is_the_twin_axis_in_the_b_frame():
    from midas_stress.twin_relations import default_hcp_twin_relations, relation_axis_family
    a_axes = _a_axes()
    star = [A_STAR, B_STAR, A_STAR - B_STAR]
    for name, expect in (("tension_1012", a_axes), ("compression_1011", a_axes),
                         ("tension_1121", star), ("compression_1122", star)):
        rel = default_hcp_twin_relations(CA, systems=(name,))[0]
        fam = relation_axis_family(rel, 194)
        assert all(_parallel(v, expect) for v in fam), name


def test_default_hcp_window_is_the_relations_own_angle_not_55_to_65():
    """Before: window 55-65 for every phase, so HCP {10-12} (85 deg) and {11-21} (35 deg) could never pass."""
    OM, pos, lab = _pairs("midas", "tension_1012")
    assert _accepted(OM, pos, lab, "tension_1012") == 40
    out = find_sigma3_partners(OM, pos, lab, k_NN=3, phase=CrystalPhase.HCP, c_over_a=CA,
                               hcp_twin="tension_1012", misori_low=55.0, misori_high=65.0)
    # the true 85 deg twins are never selected by the 55-65 window; what passes are chance pairs
    # of neighbouring unrelated grains (the axis gate is weak for in-plane axes: three lines 60 deg
    # apart at |dot| >= 0.9 cover ~86% of in-plane directions, so the ANGLE window does the work)
    assert all(tw != g + 1 for g, tw in out["pairs"])
    assert len(out["pairs"]) <= CHANCE_MAX


@pytest.mark.parametrize("name", list(TWINS))
def test_build_sigma3_pair_is_the_exact_twin_in_the_requested_frame(name):
    """build_sigma3_pair(HCP) now builds the real deformation twin (Mg 86.3 deg {10-12} was on
    the wrong axis family before) and its axis follows the frame."""
    from midas_stress.twin_relations import default_hcp_twin_relations, relation_disorientation
    for fr in HCP_FRAMES:
        R = _random_rot(np.random.default_rng(4))
        Rm, twin = build_sigma3_pair(R, CrystalPhase.HCP, frame=fr, c_over_a=CA, hcp_twin=name)
        ang, ax = o.misorientation_om(Rm.ravel(), twin.ravel(), space_group=194)
        ref, _ = relation_disorientation(
            default_hcp_twin_relations(CA, systems=(name,), frame=fr)[0], 194)
        assert math.degrees(ang) == pytest.approx(ref, abs=1e-6)


def test_build_sigma3_pair_fcc_is_unchanged():
    R = _random_rot(np.random.default_rng(5))
    _, twin = build_sigma3_pair(R, CrystalPhase.FCC)
    ang, _ = o.misorientation_om(R.ravel(), twin.ravel(), space_group=225)
    assert math.degrees(ang) == pytest.approx(60.0, abs=1e-6)
