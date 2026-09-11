"""Tests for classify_q_pair: telling Friedel mates from Ewald crossings by q, not by quadrant."""

from __future__ import annotations

import numpy as np
import pytest

from midas_defect.distributions import classify_q_pair


def _rot(v, axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    v = np.asarray(v, float)
    return (v * np.cos(t) + np.cross(axis, v) * np.sin(t)
            + axis * np.dot(axis, v) * (1 - np.cos(t)))


def test_antipodal_equal_magnitude_is_friedel():
    q = np.array([0.3, -0.8, 0.4]) * 2.99
    out = classify_q_pair(q, -q)
    assert out["kind"] == "friedel"
    assert out["dot"] == pytest.approx(-1.0)
    assert out["axis_angle_deg"] == pytest.approx(0.0, abs=1e-6)


def test_parallel_equal_magnitude_is_crossing():
    q = np.array([0.3, -0.8, 0.4]) * 2.99
    assert classify_q_pair(q, _rot(q, [0, 0, 1], 1.0))["kind"] == "crossing"


def test_friedel_survives_centroid_scatter():
    q = np.array([0.3, -0.8, 0.4]) * 2.99
    assert classify_q_pair(q, -_rot(q, [1, 0, 0], 1.0))["kind"] == "friedel"


def test_beyond_the_tolerance_cone_is_unrelated():
    q = np.array([0.3, -0.8, 0.4]) * 2.99
    perp = np.cross(q, [0, 0, 1])          # rotate about an axis normal to q so the tilt is exactly 5 deg
    out = classify_q_pair(q, _rot(q, perp, 5.0))
    assert out["kind"] == "unrelated"
    assert out["axis_angle_deg"] == pytest.approx(5.0, abs=1e-6)


def test_two_220_variants_of_one_grain_are_unrelated():
    """The mis-pairing that voided a Bragg control: same |q|, 60 deg apart, NOT a pair."""
    g = 2 * np.pi * np.sqrt(8) / 3.6356
    q1 = g * np.array([1, 1, 0]) / np.sqrt(2)
    q2 = g * np.array([1, 0, 1]) / np.sqrt(2)
    out = classify_q_pair(q1, q2)
    assert out["kind"] == "unrelated"
    assert out["axis_angle_deg"] == pytest.approx(60.0)


def test_magnitude_mismatch_is_unrelated_even_when_parallel():
    q = np.array([0.0, 0.6, 0.8])
    assert classify_q_pair(q, 1.05 * q)["kind"] == "unrelated"
    assert classify_q_pair(q, 1.005 * q)["kind"] == "crossing"


def test_demk_l5_label_centroids_classify_as_verified():
    """Unit vectors and |q| of demk L5 label centroids (cc3d/L5_gapfix_annot.csv through the
    corrected transform, 2026-09-09): 34/40 are Friedel mates, 77/84 two crossings of one q,
    and 77/130 two different <220> variants."""
    q34 = 2.9079 * np.array([+0.2907, +0.8701, +0.3980])
    q40 = 2.8970 * np.array([-0.2910, -0.8700, -0.3980])
    q77 = 4.8980 * np.array([+0.5709, +0.1720, -0.8028])
    q84 = 4.9009 * np.array([+0.5743, +0.1722, -0.8003])
    q130 = 4.8922 * np.array([-0.3706, +0.4783, -0.7962])
    assert classify_q_pair(q34, q40)["kind"] == "friedel"
    assert classify_q_pair(q77, q84)["kind"] == "crossing"
    assert classify_q_pair(q77, q130)["kind"] == "unrelated"


@pytest.mark.parametrize("a,b", [([0, 0, 0], [1, 0, 0]), ([1, 0], [1, 0]), ([np.nan, 0, 0], [1, 0, 0])])
def test_rejects_degenerate_vectors(a, b):
    with pytest.raises(ValueError):
        classify_q_pair(a, b)


@pytest.mark.parametrize("cos_tol", [0.0, 1.0, -0.1])
def test_rejects_bad_tolerance(cos_tol):
    with pytest.raises(ValueError):
        classify_q_pair([1, 0, 0], [1, 0, 0], cos_tol=cos_tol)
