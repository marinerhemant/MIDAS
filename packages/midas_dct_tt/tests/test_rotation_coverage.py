"""Rotation-field conditioning: one reflection is exactly null, pairs follow
(1 - cos gamma)/4, and the law matches an explicit view-by-view simulation."""
import math

import numpy as np
import pytest

from midas_dct_tt.rotation_coverage import (best_reflection_pair,
                                            rotation_conditioning,
                                            sensitivity_moment,
                                            separation_for_conditioning)


def _pair(gamma_deg):
    g = math.radians(gamma_deg)
    return np.array([[0.0, 0.0, 1.0],
                     [math.sin(g), 0.0, math.cos(g)]])


def test_single_reflection_has_an_exact_null():
    ev, ratio = rotation_conditioning([[0.0, 0.0, 1.0]])
    assert ev[0] == pytest.approx(0.0, abs=1e-14)
    assert ev[1] == pytest.approx(0.5) and ev[2] == pytest.approx(0.5)
    assert ratio == 0.0


@pytest.mark.parametrize("gamma", [0.0, 5.0, 13.31, 30.0, 60.0, 90.0])
def test_pair_follows_the_analytic_law(gamma):
    ev, _ = rotation_conditioning(_pair(gamma))
    c = math.cos(math.radians(gamma))
    assert np.allclose(sorted(ev), sorted([(1 - c) / 4, (1 + c) / 4, 0.5]), atol=1e-12)


def test_matches_an_explicit_view_simulation():
    """Build the actual sensitivity axes over a psi scan and compare moments."""
    gamma = 13.31
    g = _pair(gamma)
    axes = []
    for u in g:
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, u)
        n = np.linalg.norm(v)
        if n < 1e-12:
            R = np.eye(3)
        else:
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + K + K @ K / (1.0 + float(z @ u))
        for k in range(90):
            t = math.radians(4.0 * k)
            axes.append(R @ np.array([math.cos(t), math.sin(t), 0.0]))
    axes = np.array(axes)
    ev_sim = np.linalg.eigvalsh(axes.T @ axes / len(axes))
    ev_law, _ = rotation_conditioning(g)
    assert np.allclose(ev_sim, ev_law, atol=1e-4)


def test_the_esrf_experiment_is_flagged_as_underdetermined():
    """Grain 605's two published reflections: 13.3 deg, third component 75x worse."""
    g1 = np.array([0.228, -0.017, 0.974])
    g2 = np.array([0.084, 0.165, 0.983])
    ev, ratio = rotation_conditioning([g1, g2])
    assert np.allclose(ev, [0.0067, 0.4933, 0.5], atol=5e-4)
    assert ratio < 0.02
    assert ev[-1] / ev[0] > 50


def test_orthogonal_pair_is_the_best_a_pair_can_do():
    ev, ratio = rotation_conditioning(_pair(90.0))
    assert ev[0] == pytest.approx(0.25)
    assert ratio == pytest.approx(0.5)


def test_separation_thresholds_round_trip():
    assert separation_for_conditioning(0.0) == pytest.approx(0.0)
    assert separation_for_conditioning(0.5) == pytest.approx(90.0)
    for r in (0.05, 0.25, 0.5):
        gam = separation_for_conditioning(r)
        _, back = rotation_conditioning(_pair(gam))
        assert back == pytest.approx(r, abs=1e-9)
    with pytest.raises(ValueError):
        separation_for_conditioning(1.5)


def test_close_reflections_cannot_be_rescued_by_taking_more():
    """Many nearly-parallel reflections stay ill-conditioned."""
    rng = np.random.default_rng(0)
    g = np.array([0.0, 0.0, 1.0]) + 0.05 * rng.normal(size=(12, 3))
    _, ratio = rotation_conditioning(g)
    assert ratio < 0.02


def test_best_pair_picks_the_most_orthogonal():
    g = np.array([[0.0, 0.0, 1.0],
                  [math.sin(math.radians(10)), 0.0, math.cos(math.radians(10))],
                  [1.0, 0.0, 0.0]])
    i, j, gam, ratio = best_reflection_pair(g)
    assert {i, j} == {0, 2}
    assert gam == pytest.approx(90.0, abs=1e-6)
    assert ratio == pytest.approx(0.5)
    with pytest.raises(ValueError):
        best_reflection_pair(g[:1])


def test_sign_of_g_is_irrelevant():
    """A reflection and its Friedel mate constrain identically."""
    a, _ = rotation_conditioning(_pair(30.0))
    g = _pair(30.0)
    g[1] *= -1.0
    b, _ = rotation_conditioning(g)
    assert np.allclose(a, b)


def test_moment_is_symmetric_and_trace_one():
    M = sensitivity_moment(_pair(37.0))
    assert np.allclose(M, M.T)
    assert np.trace(M) == pytest.approx(1.0)


def test_single_plane_multi_scan_schemes_do_not_break_the_roll_null():
    """The claim made in Sec. 5: no combination of scans on ONE lattice plane
    recovers the component about G.

    Joining a scan to its antipodal partner (-h-k-l), to a higher diffraction
    order (2h 2k 2l), or to any repeat of itself leaves the sensitivity moment
    identical, because ``g g^T`` is invariant under ``g -> -g`` and a higher
    order shares the direction. Only a genuinely different plane normal helps.
    """
    g = np.array([0.0, -2.0, 2.0])
    g = g / np.linalg.norm(g)
    single = sensitivity_moment([g])

    for label, extra in (("antipode", -g),
                         ("second order", 2.0 * g),
                         ("antipodal second order", -2.0 * g),
                         ("a repeat", g)):
        M = sensitivity_moment([g, extra])
        assert np.allclose(M, single), label
        # the null stays exactly along G
        assert np.allclose(M @ g, 0.0, atol=1e-15), label
        assert rotation_conditioning([g, extra])[1] == 0.0, label

    # ...while a different plane normal does break it, per eq. (cond)
    for deg, expect in ((60.0, (1 - np.cos(np.radians(60.0))) / 2),
                        (90.0, 0.5)):
        _, ratio = rotation_conditioning(_pair(deg))
        assert ratio == pytest.approx(expect)
        assert ratio > 0.0
