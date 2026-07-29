"""Tests for the family-level, symmetry-safe, isolated-peak asterism arc."""

import math

import numpy as np
import pytest

from midas_defect.asterism import family_asterism_arc, reflection_directions


def _cone(center, sigma_deg, n, rng, qmag=3.46):
    """n unit-ish q-vectors scattered by ~sigma_deg (Gaussian) about `center`."""
    center = np.asarray(center, float); center /= np.linalg.norm(center)
    # build a local frame
    a = np.array([1.0, 0, 0]) if abs(center[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(center, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(center, e1)
    ang = np.radians(rng.normal(0, sigma_deg, n))
    phi = rng.uniform(0, 2 * math.pi, n)
    dirs = (np.cos(ang)[:, None] * center
            + np.sin(ang)[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
    return dirs * qmag


def test_arc_recovers_injected_spread():
    rng = np.random.default_rng(0)
    c = np.array([1.0, 0.3, -0.2])
    q = _cone(c, 3.0, 4000, rng)
    r = family_asterism_arc(q, np.ones(len(q)), c[None, :], max_angle_deg=20)
    # arc = intensity-weighted RMS angular deviation ~ the injected sigma (3 deg)
    assert 2.5 < r["arc_deg"] < 4.0
    assert r["n_reflections_used"] == 1


def test_more_spread_gives_larger_arc():
    rng = np.random.default_rng(1)
    c = np.array([1.0, 0.0, 0.0])
    tight = _cone(c, 2.0, 4000, rng)
    broad = _cone(c, 4.0, 4000, rng)
    a_t = family_asterism_arc(tight, np.ones(len(tight)), c[None, :])["arc_deg"]
    a_b = family_asterism_arc(broad, np.ones(len(broad)), c[None, :])["arc_deg"]
    assert a_b > 1.5 * a_t


def test_symmetry_safe_multiple_directions_not_mixed():
    """Two reflection directions 90 deg apart: pooling must give the per-direction arc
    (~few deg), NOT the 90 deg separation (the crystallographic-hkl variant-mixing bug)."""
    rng = np.random.default_rng(2)
    c1 = np.array([1.0, 0, 0]); c2 = np.array([0, 1.0, 0])
    q = np.vstack([_cone(c1, 3.0, 3000, rng), _cone(c2, 3.0, 3000, rng)])
    r = family_asterism_arc(q, np.ones(len(q)), np.vstack([c1, c2]), max_angle_deg=20)
    assert r["n_reflections_used"] == 2
    assert r["arc_deg"] < 8.0            # not ~90
    assert np.all(r["per_reflection"][:2] < 8.0)


def test_far_voxels_excluded():
    rng = np.random.default_rng(3)
    c = np.array([1.0, 0, 0])
    on = _cone(c, 2.0, 3000, rng)
    off = _cone(np.array([0, 0, 1.0]), 2.0, 3000, rng)   # a different reflection, far
    q = np.vstack([on, off])
    # only give the on-center; the off voxels are >20 deg away and must be dropped
    r = family_asterism_arc(q, np.ones(len(q)), c[None, :], max_angle_deg=20)
    assert r["n_voxels"][0] == pytest.approx(len(on), rel=0.05)
    assert r["arc_deg"] < 4.0


def test_min_voxels_skips_sparse_reflection():
    rng = np.random.default_rng(4)
    c = np.array([1.0, 0, 0])
    q = _cone(c, 2.0, 10, rng)
    r = family_asterism_arc(q, np.ones(len(q)), c[None, :], min_voxels_per_reflection=50)
    assert math.isnan(r["arc_deg"])
    assert r["n_reflections_used"] == 0


def test_reflection_directions_sample_frame():
    OM = np.eye(3)
    d = reflection_directions(OM, np.array([[2, 0, 0], [0, 0, 2]]))
    assert np.allclose(d[0], [1, 0, 0])
    assert np.allclose(d[1], [0, 0, 1])
    # rotation carries the direction into the sample frame
    th = math.radians(90)
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    d2 = reflection_directions(Rz, np.array([[2, 0, 0]]))
    assert np.allclose(d2[0], [0, 1, 0], atol=1e-9)


def test_empty_input():
    r = family_asterism_arc(np.zeros((0, 3)), np.zeros(0), np.eye(3))
    assert math.isnan(r["arc_deg"]) and r["n_reflections_used"] == 0
