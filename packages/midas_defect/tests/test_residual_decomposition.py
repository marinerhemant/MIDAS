"""Radial vs angular indexing residuals.

The verdicts are the product here, so each is driven by a planted, unambiguous
case: a pure cell error, a pure orientation error, and the mixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_defect.residual_decomposition import decompose_residuals

RNG = np.random.default_rng(7)


def _random_q(n=200, qmin=0.15, qmax=0.6):
    v = RNG.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * RNG.uniform(qmin, qmax, (n, 1))


def _rotate(q, deg):
    """Tilt every vector by EXACTLY `deg`, about its own perpendicular.

    Rotating about a fixed axis would tilt each vector by a different amount
    (sin(theta/2) = sin(alpha/2) sin(phi)), so the planted angular error would
    not be the one asserted.
    """
    q = np.asarray(q, float)
    r = RNG.normal(size=q.shape)
    perp = r - (np.einsum("ij,ij->i", r, q) / np.einsum("ij,ij->i", q, q))[:, None] * q
    perp /= np.linalg.norm(perp, axis=1, keepdims=True)
    t = np.radians(deg)
    return np.cos(t) * q + np.sin(t) * perp * np.linalg.norm(q, axis=1, keepdims=True)


def test_a_pure_cell_error_is_RADIAL():
    qp = _random_q()
    qo = qp * 1.004                                   # 0.4 % too large, no rotation
    d = decompose_residuals(qo, qp)
    assert d.median_radial_pct == pytest.approx(0.4, abs=0.01)
    assert d.median_angular_deg < 1e-9
    assert d.angular_over_radial < 0.33
    assert "RADIAL-dominated" in d.verdict


def test_a_pure_ORIENTATION_error_is_ANGULAR():
    qp = _random_q()
    qo = _rotate(qp, 1.5)                             # 1.5 deg, |q| untouched
    d = decompose_residuals(qo, qp)
    assert d.median_angular_deg == pytest.approx(1.5, abs=1e-6)
    assert d.median_radial_pct < 1e-9
    assert "ANGULAR-dominated" in d.verdict


def test_THE_REAL_CASE_small_radial_large_angular():
    """0.24 % radial with 1.5 deg angular -> the cell is right, the grain is not."""
    qp = _random_q()
    qo = _rotate(qp, 1.5) * 1.0024
    d = decompose_residuals(qo, qp)
    assert d.median_radial_pct == pytest.approx(0.24, abs=0.02)
    assert d.median_angular_deg == pytest.approx(1.5, abs=0.05)
    assert d.angular_over_radial == pytest.approx(np.radians(1.5) / 0.0024,
                                                  rel=0.1)
    assert d.angular_over_radial > 3.0
    assert "multiple grains" in d.verdict


def test_the_ratio_is_dimensionless_and_independent_of_q():
    """Percent and degrees are not comparable; the ratio must be."""
    small = _random_q(qmin=0.10, qmax=0.15)
    large = _random_q(qmin=0.50, qmax=0.60)
    r_small = decompose_residuals(_rotate(small, 1.0) * 1.002, small)
    r_large = decompose_residuals(_rotate(large, 1.0) * 1.002, large)
    assert r_small.angular_over_radial == pytest.approx(
        r_large.angular_over_radial, rel=0.05)


def test_displacements_are_reported_in_commensurate_units():
    qp = _random_q()
    qo = _rotate(qp, 2.0) * 1.001
    d = decompose_residuals(qo, qp)
    assert np.allclose(d.displacement_angular, d.q_mag * np.radians(2.0), rtol=1e-4)
    assert np.allclose(d.displacement_radial, d.q_mag * 0.001, rtol=1e-6)


def test_a_mixed_case_says_so_rather_than_picking_a_side():
    qp = _random_q()
    qo = _rotate(qp, 0.1146) * 1.002       # ratio ~1
    d = decompose_residuals(qo, qp)
    assert 0.33 <= d.angular_over_radial <= 3.0
    assert "mixed" in d.verdict


def test_a_perfect_fit_reports_zero_radial_not_a_crash():
    qp = _random_q()
    d = decompose_residuals(_rotate(qp, 0.5), qp)
    assert np.isinf(d.angular_over_radial)
    assert "cell is exact" in d.verdict
    assert "ANGULAR-dominated" in d.verdict   # same label in both branches


def test_bad_input_is_refused_not_turned_into_NaN():
    with pytest.raises(ValueError, match="matching"):
        decompose_residuals(np.zeros((5, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="no matched"):
        decompose_residuals(np.zeros((0, 3)), np.zeros((0, 3)))
    with pytest.raises(ValueError, match="zero-length"):
        decompose_residuals(np.ones((2, 3)), np.zeros((2, 3)))
