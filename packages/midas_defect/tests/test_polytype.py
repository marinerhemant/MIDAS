import numpy as np
import pytest

from midas_defect.polytype import (
    detect_activated_111_axis,
    per_grain_lamella_thickness,
    polytype_satellite_enhancement,
)


def _plant_satellite_along(axis: np.ndarray, q_target: float, n_satellite: int, n_bg: int, rng):
    axis = axis / np.linalg.norm(axis)
    sat_pos = axis * q_target
    sat_qs = rng.normal(scale=0.02, size=(n_satellite, 3)) + sat_pos
    # Isotropic background voxels on a sphere of similar radius
    bg = rng.normal(size=(n_bg, 3))
    bg = bg / np.linalg.norm(bg, axis=1, keepdims=True) * q_target
    bg_qs = bg + rng.normal(scale=0.04, size=(n_bg, 3))
    qs = np.concatenate([sat_qs, bg_qs], axis=0)
    vals = np.concatenate([np.ones(n_satellite) * 10.0, np.ones(n_bg)], axis=0)
    return qs, vals


# -- activated axis ---------------------------------------------------------

def test_detect_activated_111_picks_the_planted_axis():
    rng = np.random.default_rng(0)
    G_111 = 3.0
    # Plant satellites along (1,1,1)/sqrt 3 at q = G/3.
    truth_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    qs, vals = _plant_satellite_along(truth_axis, G_111 / 3.0, 200, 300, rng)
    out = detect_activated_111_axis(qs, vals, G_111_magnitude=G_111)
    np.testing.assert_allclose(out["a_sample"], truth_axis, atol=1e-12)
    # Strongest among the four <111> directions
    assert out["argmax_index"] == 0


def test_detect_activated_with_crystal_to_sample_OM_rotates_candidate_axes():
    rng = np.random.default_rng(0)
    G_111 = 3.0
    # Plant along sample [0, 0, 1] (= rotated crystal [1, 1, 1]/sqrt 3 via OM)
    truth_sample = np.array([0.0, 0.0, 1.0])
    qs, vals = _plant_satellite_along(truth_sample, G_111 / 3.0, 200, 300, rng)
    # Rotation that sends [1,1,1]/sqrt 3 onto [0,0,1].
    a = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s * s))
    out = detect_activated_111_axis(qs, vals, G_111_magnitude=G_111, crystal_to_sample_OM=R)
    # First candidate after rotation should be ~ (0, 0, 1); chosen index 0.
    assert out["argmax_index"] == 0
    np.testing.assert_allclose(out["a_sample"], truth_sample, atol=1e-9)


# -- satellite enhancement (DEPRECATED — see test_geometry_honest_2026.py) ---
# These two functions produced artifacts (AUDIT_2026-06-23.md) and now refuse by
# default; the geometry-honest replacements are tested in test_geometry_honest_2026.py.

def test_polytype_satellite_enhancement_is_deprecated():
    rng = np.random.default_rng(1)
    axis = np.array([0.0, 0.0, 1.0])
    qs, vals = _plant_satellite_along(axis, 1.0, 300, 100, rng)
    with pytest.raises(RuntimeError):
        polytype_satellite_enhancement(qs, vals, axis, G_magnitude=3.0)
    # the historical (artifact) path is still reachable for reproduction only
    out = polytype_satellite_enhancement(qs, vals, axis, G_magnitude=3.0,
                                         allow_deprecated=True)
    assert "enhancement_G_over_3" in out


def test_per_grain_lamella_thickness_is_deprecated():
    qs = np.zeros((5, 3)); vals = np.ones(5); OM = np.eye(3)[None]
    with pytest.raises(RuntimeError):
        per_grain_lamella_thickness(
            qs, vals, OM, a_crystal=np.array([0.0, 0.0, 1.0]),
            G_3_magnitude=1.0, grain_of_voxel=np.zeros(5, dtype=int), n_grains=1)
    # historical path reachable only with the explicit escape hatch
    L = per_grain_lamella_thickness(
        qs, vals, OM, a_crystal=np.array([0.0, 0.0, 1.0]), G_3_magnitude=1.0,
        grain_of_voxel=np.zeros(5, dtype=int), n_grains=1, allow_deprecated=True)
    assert np.isnan(L[0])
