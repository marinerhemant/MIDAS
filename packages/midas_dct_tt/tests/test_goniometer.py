"""Goniometer tilt solution and stage reachability.

The tilt formula is easy to test tautologically -- recompute it and compare. The
tests here assert the *defining property* instead: after the stage moves, ``G``
lies along the tomographic rotation axis. A sign slip or a transposed frame
still satisfies the formula and fails this.
"""
import math

import numpy as np
import pytest

from midas_dct_tt import (INSTRUMENT_OFFSETS_ID11, best_reachable_pair,
                          instrument_transformation, reachable_reflections,
                          reciprocal_basis, rodrigues_to_crystal_to_sample,
                          rotation_conditioning, tilt_branches, topotomo_tilts)

LAM_A = 0.32626          # 38 keV, the ID11 Ti-7Al campaign
A_A, C_A = 2.931, 4.694  # Ti-7Al hexagonal


def _rx(t):
    t = math.radians(t)
    return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)],
                     [0, math.sin(t), math.cos(t)]])


def _ry(t):
    t = math.radians(t)
    return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0],
                     [-math.sin(t), 0, math.cos(t)]])


def _aligned(G, T, up, low):
    """Component of the moved ``G`` off the rotation axis, normalised."""
    v = _ry(low) @ _rx(up) @ (np.asarray(T).T @ np.asarray(G))
    v = v / np.linalg.norm(v)
    return math.hypot(v[0], v[1])


@pytest.fixture
def T():
    return instrument_transformation(*INSTRUMENT_OFFSETS_ID11)


@pytest.fixture
def B():
    return reciprocal_basis(A_A, A_A, C_A, 90.0, 90.0, 120.0)


# --- the tilt solution ------------------------------------------------------

def test_tilts_put_G_on_the_rotation_axis(T):
    rng = np.random.default_rng(0)
    for _ in range(50):
        G = rng.normal(size=3)
        up, low = topotomo_tilts(G, T)
        assert _aligned(G, T, up, low) < 1e-9


def test_both_branches_align_equally_well(T):
    rng = np.random.default_rng(1)
    for _ in range(20):
        G = rng.normal(size=3)
        for up, low in tilt_branches(G, T):
            assert _aligned(G, T, up, low) < 1e-9


def test_sibling_branch_is_the_180_degree_partner(T):
    rng = np.random.default_rng(2)
    for _ in range(20):
        G = rng.normal(size=3)
        (u0, l0), (u1, l1) = tilt_branches(G, T)
        assert abs(abs(u1 - u0) - 180.0) < 1e-9
        assert l1 == pytest.approx(-l0)


def test_principal_branch_stays_inside_ninety_degrees(T):
    """So the sibling is always the one a sub-90 stage cannot reach."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        up, _ = topotomo_tilts(rng.normal(size=3), T)
        assert abs(up) <= 90.0 + 1e-9


def test_friedel_partners_need_identical_tilts(T):
    """``G`` and ``-G`` are the same TT scan; enumeration must not disagree."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        G = rng.normal(size=3)
        assert topotomo_tilts(G, T) == pytest.approx(topotomo_tilts(-G, T))


def test_G_along_the_beam_is_rejected_not_silently_wrong(T):
    G = T @ np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="degenerate"):
        topotomo_tilts(G, T)


def test_instrument_transformation_order_is_not_symmetric():
    a = instrument_transformation(-1.2, 0.7, 90.0)
    b = instrument_transformation(0.7, -1.2, 90.0)
    assert not np.allclose(a, b)


# --- reciprocal basis (delegated to midas_hkls) -----------------------------

def test_reciprocal_basis_reproduces_the_hexagonal_d_spacing(B):
    """``|B @ hkl| == 1/d``, with ``d`` from the closed form for a hexagonal cell."""
    for h, k, l in [(1, 0, 0), (0, 0, 2), (1, 0, 1), (2, -1, 3), (1, 1, 0)]:
        inv_d = math.sqrt(4.0 / 3.0 * (h * h + h * k + k * k) / A_A ** 2
                          + l * l / C_A ** 2)
        assert np.linalg.norm(B @ np.array([h, k, l], float)) == pytest.approx(inv_d)


# --- reachability -----------------------------------------------------------

def test_reachable_reflections_respects_the_envelope(T, B):
    U = rodrigues_to_crystal_to_sample([0.1, -0.2, 0.3])
    for env in (5.0, 15.0, 30.0):
        for hkl, up, low in reachable_reflections(U, B, LAM_A, envelope=env, T=T):
            assert abs(up) <= env and abs(low) <= env


def test_envelope_accepts_an_asymmetric_pair(T, B):
    U = rodrigues_to_crystal_to_sample([0.1, -0.2, 0.3])
    for hkl, up, low in reachable_reflections(U, B, LAM_A, envelope=(18.7, 14.3), T=T):
        assert abs(up) <= 18.7 and abs(low) <= 14.3


def test_a_wider_stage_can_only_reach_more(T, B):
    U = rodrigues_to_crystal_to_sample([0.05, 0.4, -0.1])
    seen = [{h for h, _, _ in reachable_reflections(U, B, LAM_A, envelope=e, T=T)}
            for e in (10.0, 20.0, 30.0, 45.0)]
    for a, b in zip(seen, seen[1:]):
        assert a <= b


def test_non_diffracting_reflections_are_dropped(T, B):
    """``d < lambda/2`` cannot satisfy Bragg at any angle."""
    U = np.eye(3)
    for hkl, _, _ in reachable_reflections(U, B, LAM_A, envelope=90.0, T=T,
                                           hkl_max=6):
        assert 1.0 / np.linalg.norm(B @ np.array(hkl, float)) >= LAM_A / 2.0


def test_friedel_deduplication_removes_the_partner(T, B):
    U = rodrigues_to_crystal_to_sample([0.1, -0.2, 0.3])
    kw = dict(envelope=45.0, T=T)
    uniq = reachable_reflections(U, B, LAM_A, unique=True, **kw)
    both = reachable_reflections(U, B, LAM_A, unique=False, **kw)
    assert len(uniq) < len(both)
    for hkl, _, _ in uniq:
        assert tuple(-x for x in hkl) not in {h for h, _, _ in uniq}


# --- conditioning + reachability composed -----------------------------------

def test_best_reachable_pair_agrees_with_the_conditioning_law(T, B):
    U = rodrigues_to_crystal_to_sample([0.1, -0.2, 0.3])
    got = best_reachable_pair(U, B, LAM_A, envelope=30.0, T=T)
    assert got is not None
    ha, hb, gamma, ratio = got
    ga = U @ (B @ np.array(ha, float))
    gb = U @ (B @ np.array(hb, float))
    _, expect = rotation_conditioning([ga, gb])
    assert ratio == pytest.approx(expect)
    ca = abs(float(ga @ gb) / (np.linalg.norm(ga) * np.linalg.norm(gb)))
    assert gamma == pytest.approx(math.degrees(math.acos(min(1.0, ca))))


def test_best_reachable_pair_is_actually_the_best(T, B):
    import itertools
    U = rodrigues_to_crystal_to_sample([0.02, 0.31, -0.12])
    reach = reachable_reflections(U, B, LAM_A, envelope=25.0, T=T)
    gs = [U @ (B @ np.array(h, float)) for h, _, _ in reach]
    brute = max(rotation_conditioning([gs[i], gs[j]])[1]
                for i, j in itertools.combinations(range(len(gs)), 2))
    assert best_reachable_pair(U, B, LAM_A, envelope=25.0, T=T)[3] == pytest.approx(brute)


def test_a_stage_too_tight_for_two_reflections_returns_none(T, B):
    U = rodrigues_to_crystal_to_sample([0.1, -0.2, 0.3])
    assert best_reachable_pair(U, B, LAM_A, envelope=0.01, T=T) is None


def test_wider_envelope_never_worsens_conditioning(T, B):
    """The paper's Sec. 8 claim in miniature: opening the stage cannot hurt."""
    U = rodrigues_to_crystal_to_sample([0.1, -0.2, 0.3])
    best = -1.0
    for env in (15.0, 20.0, 25.0, 30.0, 45.0, 90.0):
        got = best_reachable_pair(U, B, LAM_A, envelope=env, T=T)
        if got is not None:
            assert got[3] >= best - 1e-12
            best = got[3]
    # A PAIR saturates at 0.5, not 1: the largest eigenvalue is always 1/2.
    assert best == pytest.approx(0.5, abs=1e-3)


def test_pair_ratio_saturates_at_one_half_not_one(T, B):
    """Guards the normalisation the Sec. 8 table does NOT use."""
    for deg in (30.0, 60.0, 90.0):
        r = math.radians(deg)
        g = [np.array([1.0, 0.0, 0.0]),
             np.array([math.cos(r), math.sin(r), 0.0])]
        _, ratio = rotation_conditioning(g)
        assert ratio == pytest.approx((1 - math.cos(r)) / 2)
    assert rotation_conditioning([[1, 0, 0], [0, 1, 0]])[1] == pytest.approx(0.5)


# --- the pymicro Rodrigues convention ---------------------------------------

def test_rodrigues_is_a_proper_rotation():
    rng = np.random.default_rng(5)
    for _ in range(20):
        R = rodrigues_to_crystal_to_sample(rng.normal(size=3) * 0.4)
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_rodrigues_angle_is_exact_at_large_rotations():
    """The property midas_stress's converter fails; see esrf.py for why."""
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    for deg in (5.0, 30.0, 60.0, 90.0, 120.0):
        R = rodrigues_to_crystal_to_sample(axis * math.tan(math.radians(deg) / 2))
        got = math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
        assert got == pytest.approx(deg, abs=1e-9)


def test_rodrigues_recovers_the_axis():
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    R = rodrigues_to_crystal_to_sample(axis * math.tan(math.radians(70.0) / 2))
    assert np.allclose(R @ axis, axis, atol=1e-12)


def test_zero_rodrigues_is_the_identity():
    assert np.allclose(rodrigues_to_crystal_to_sample([0, 0, 0]), np.eye(3))
