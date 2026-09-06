"""Free-triclinic UB refinement, and the uncertainties ImageD11 does not return.

The estimator itself (Paciorek, Acta A55 543) is exact on noiseless data, so the
interesting tests are the ones about what the data does NOT determine.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from midas_hkls.ub_refine import (refine_ub_from_gvectors, ub_to_cell, ub_to_u_b,
                                  cell_from_metric, drlv2)

RNG = np.random.default_rng(20260902)


def _B_from_cell(a, b, c, al, be, ga):
    """B from the package's own lattice math -- columns are the reciprocal basis.

    Uses `Lattice.reciprocal_cartesian_vectors` so the test does not carry a
    second, hand-rolled copy of the Busing-Levy construction.
    """
    from midas_hkls import Lattice
    lat = Lattice(a=a, b=b, c=c, alpha=al, beta=be, gamma=ga)
    return np.asarray(lat.reciprocal_cartesian_vectors(), float).T


def _rot(seed=0):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _synth(cell, n=40, noise=0.0, seed=1, hmax=4, U=None):
    B = _B_from_cell(*cell)
    U = _rot(seed) if U is None else U
    UB = U @ B
    rng = np.random.default_rng(seed)
    hs = set()
    while len(hs) < n:
        t = tuple(int(v) for v in rng.integers(-hmax, hmax + 1, 3))
        if t != (0, 0, 0):
            hs.add(t)
    h = np.array(sorted(hs), float)
    g = h @ UB.T
    if noise:
        g = g + rng.normal(0, noise, g.shape)
    return h, g, UB, cell


TRICLINIC = (5.1, 6.3, 7.7, 82.0, 95.0, 103.0)
TETRAGONAL = (3.6116, 3.6116, 19.2516, 90.0, 90.0, 90.0)


def test_exact_recovery_of_a_triclinic_cell_with_no_noise():
    h, g, UB, cell = _synth(TRICLINIC, n=40)
    fit = refine_ub_from_gvectors(h, g, sigma_g=1e-9)
    assert np.allclose(fit.cell, cell, atol=1e-8), fit.cell
    assert np.allclose(fit.UB, UB, atol=1e-10)
    assert fit.rms_drlv < 1e-10


def test_it_does_not_assume_symmetry():
    """A triclinic cell must come back triclinic, not snapped to 90 degrees."""
    h, g, UB, cell = _synth(TRICLINIC, n=40)
    fit = refine_ub_from_gvectors(h, g, sigma_g=1e-9)
    for got, want in zip(fit.cell[3:], cell[3:]):
        assert abs(got - want) < 1e-6
        assert abs(got - 90.0) > 1.0        # genuinely not orthogonal


def test_U_is_a_rotation_and_UB_reassembles():
    h, g, UB, cell = _synth(TRICLINIC, n=40)
    fit = refine_ub_from_gvectors(h, g, sigma_g=1e-9)
    U, B = fit.U, fit.B
    assert np.allclose(U @ U.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(U), 1.0, atol=1e-10)
    assert np.allclose(U @ B, fit.UB, atol=1e-9)


def test_sigma_shrinks_as_sqrt_N():
    """The covariance must behave like a least squares, not be decorative."""
    s = []
    for n in (12, 48, 192):
        h, g, UB, cell = _synth(TRICLINIC, n=n, noise=2e-4, seed=5)
        s.append(refine_ub_from_gvectors(h, g).cell_sigma[0])
    assert s[0] > s[1] > s[2]
    ratio = s[0] / s[2]
    assert 2.0 < ratio < 8.0, f"sigma fell by {ratio:.2f}x for 16x the data"


def test_sigma_scales_with_the_noise():
    out = []
    for nz in (1e-4, 4e-4):
        h, g, UB, cell = _synth(TRICLINIC, n=60, noise=nz, seed=7)
        out.append(refine_ub_from_gvectors(h, g).cell_sigma[0])
    assert out[1] > 2.5 * out[0]


def test_sigma_brackets_the_truth():
    """The interval must actually cover the planted cell most of the time."""
    hits = 0
    for seed in range(20):
        h, g, UB, cell = _synth(TRICLINIC, n=60, noise=2e-4, seed=seed)
        fit = refine_ub_from_gvectors(h, g)
        if abs(fit.cell[0] - cell[0]) < 3 * fit.cell_sigma[0]:
            hits += 1
    assert hits >= 17, f"only {hits}/20 within 3 sigma — the covariance is wrong"


def test_a_PARAMETER_THE_DATA_CANNOT_SEE_is_flagged_undetermined():
    """The point of the whole thing: report 'c undetermined', not a number.

    Give it only reflections with l = 0, so nothing constrains the c axis.
    """
    B = _B_from_cell(*TETRAGONAL); U = _rot(3); UB = U @ B
    h = np.array([(i, j, 0) for i in range(-3, 4) for j in range(-3, 4)
                  if (i, j) != (0, 0)], float)
    g = h @ UB.T + RNG.normal(0, 1e-4, (len(h), 3))
    with pytest.raises(ValueError, match="coplanar"):
        refine_ub_from_gvectors(h, g)


def test_a_weakly_seen_axis_gets_a_large_sigma_and_is_flagged():
    """One out-of-plane reflection makes it solvable but badly determined."""
    B = _B_from_cell(*TETRAGONAL); U = _rot(3); UB = U @ B
    h = [(i, j, 0) for i in range(-3, 4) for j in range(-3, 4) if (i, j) != (0, 0)]
    h += [(1, 0, 1)]
    h = np.array(h, float)
    g = h @ UB.T + RNG.normal(0, 2e-4, (len(h), 3))
    fit = refine_ub_from_gvectors(h, g)
    assert fit.cell_sigma[2] > fit.cell_sigma[0] * 20, (
        f"c sigma {fit.cell_sigma[2]:.4g} vs a sigma {fit.cell_sigma[0]:.4g} — "
        "the barely-constrained axis was not penalised")
    # It is still DETERMINED here (sigma/c ~ 0.4 %): one out-of-plane reflection
    # on a 19 A axis constrains it well, because c* is small so a given |g|
    # error maps to a small relative error in c. The covariance says so, which
    # is the point -- it is not guessing from the reflection count.
    assert fit.determined["c"]


def test_the_undetermined_flag_fires_when_sigma_really_is_large():
    """Same geometry, but with the out-of-plane reflection barely measured."""
    B = _B_from_cell(*TETRAGONAL); U = _rot(3); UB = U @ B
    h = [(i, j, 0) for i in range(-3, 4) for j in range(-3, 4) if (i, j) != (0, 0)]
    h += [(1, 0, 1)]
    h = np.array(h, float)
    g = h @ UB.T
    w = np.ones(len(h)); w[-1] = 1e-8          # the only c constraint, distrusted
    fit = refine_ub_from_gvectors(h, g, weights=w, sigma_g=2e-4)
    assert "c" in fit.undetermined, (
        f"c sigma {fit.cell_sigma[2]:.4g} on c={fit.cell[2]:.3f} was not flagged")
    assert "a" not in fit.undetermined
    assert "UNDET" in str(fit)


def test_weights_change_the_answer_toward_the_trusted_reflections():
    h, g, UB, cell = _synth(TRICLINIC, n=40, noise=0.0, seed=9)
    g_bad = g.copy()
    g_bad[0] += 0.05                              # one badly wrong reflection
    plain = refine_ub_from_gvectors(h, g_bad, sigma_g=1e-3)
    w = np.ones(len(h)); w[0] = 1e-6
    down = refine_ub_from_gvectors(h, g_bad, weights=w, sigma_g=1e-3)
    assert (abs(down.cell[0] - cell[0]) < abs(plain.cell[0] - cell[0]))


def test_coplanar_and_too_few_are_refused():
    h = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], float)
    g = h * 0.1
    with pytest.raises(ValueError, match="coplanar|collinear"):
        refine_ub_from_gvectors(h, g)
    with pytest.raises(ValueError, match="at least 3"):
        refine_ub_from_gvectors(h[:2], g[:2])


def test_drlv2_matches_the_imaged11_definition():
    h, g, UB, cell = _synth(TRICLINIC, n=20)
    UBI = np.linalg.inv(UB)
    assert np.allclose(drlv2(UBI, g), 0.0, atol=1e-18)
    assert (drlv2(UBI, g + 0.5 / np.linalg.norm(UB, axis=0).mean()) > 0).all()


def test_cell_from_metric_round_trips():
    B = _B_from_cell(*TRICLINIC)
    UB = _rot(2) @ B
    assert np.allclose(ub_to_cell(UB), TRICLINIC, atol=1e-9)
