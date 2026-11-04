"""HCP slip systems must be in the crystal frame the MIDAS orientations live in.

MIDAS ``Grains.csv`` orientation matrices are crystal->lab matrices whose crystal
Cartesian frame is the ``midas_hkls`` B-matrix frame: ``G_cart = B (h, k, l)``
with a* along x, b* in the xy plane (60 deg from a* for hexagonal), c* along z.
Real-space a then sits at -30 deg (150 deg) and b at 90 deg. The textbook frame
(a1 along x) is rotated 30 deg about c from it. Disorientation angles and the 12
hexagonal operators cannot tell the two apart, which is why the old code, and
its tests, never noticed. The truth below is built independently from the B
matrix (written out inline: hexagonal, a = 1).
"""
import math

import numpy as np
import pytest

from midas_stress.plasticity import (
    HCP_FRAMES, HCP_RATIOS, get_slip_systems, get_slip_systems_for_material,
)

CA = HCP_RATIOS["Ti"]
S3 = math.sqrt(3.0)
A_STAR = np.array([2.0 / S3, 0.0, 0.0])
B_STAR = np.array([1.0 / S3, 1.0, 0.0])
C_STAR = np.array([0.0, 0.0, 1.0 / CA])
B = np.stack([A_STAR, B_STAR, C_STAR], axis=1)       # G_cart = B (h, k, l)


def _unit(v):
    return v / np.linalg.norm(v)


def _a_axes():
    a1 = _unit(np.cross(B_STAR, C_STAR))
    a2 = _unit(np.cross(C_STAR, A_STAR))
    a3 = _unit(-(np.cross(B_STAR, C_STAR) / np.linalg.norm(np.cross(B_STAR, C_STAR))
                 + np.cross(C_STAR, A_STAR) / np.linalg.norm(np.cross(C_STAR, A_STAR))))
    return [a1, a2, a3]


def _parallel(v, targets, tol=1e-9):
    v = _unit(v)
    return any(abs(abs(v @ _unit(t)) - 1.0) < tol for t in targets)


def _planes(hk_list, l_list):
    return [B @ np.array([h, k, l], float) for (h, k) in hk_list for l in l_list]


# (h, k) of the six in-basal-plane equivalents of (10-1x) and (11-2x)
_HK_10 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
_HK_11 = [(1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)]


def test_a_axes_are_at_150_and_90_degrees():
    """Sanity of the truth itself: real a at -30/150 deg, b at 90 deg."""
    a1, a2, _ = _a_axes()
    ang = lambda v: math.degrees(math.atan2(v[1], v[0])) % 180.0
    assert abs(ang(a1) - 150.0) < 1e-9
    assert abs(ang(a2) - 90.0) < 1e-9


def test_basal_directions_are_the_a_axes():
    n, b = get_slip_systems("hcp_basal", c_over_a=CA)
    assert all(abs(x[2]) > 1 - 1e-12 for x in n)
    assert all(_parallel(v, _a_axes()) for v in b)
    assert len({round(abs(float(v @ _a_axes()[0])), 6) for v in b}) >= 2     # all three used


def test_prismatic_planes_are_10m10_and_directions_a():
    n, b = get_slip_systems("hcp_prismatic", c_over_a=CA)
    prism = [A_STAR, B_STAR, A_STAR - B_STAR]
    assert all(_parallel(v, prism) for v in n)
    assert all(_parallel(v, _a_axes()) for v in b)
    assert np.allclose((n * b).sum(1), 0.0, atol=1e-12)


def test_pyramidal_a_planes_are_10m11():
    """Regression: these used to be the prismatic planes twice (normal at 90 deg
    from c, 6 systems = 3 distinct). {10-11} normals sit at 61.4 deg from c."""
    n, b = get_slip_systems("hcp_pyramidal_a", c_over_a=CA)
    planes = _planes(_HK_10, (1, -1))
    assert all(_parallel(v, planes) for v in n)
    polar = np.degrees(np.arccos(np.abs(n[:, 2])))
    assert np.allclose(polar, math.degrees(math.atan(2 * CA / S3)), atol=1e-9)
    assert len({tuple(np.round(v, 9)) for v in n}) == 6        # 6 DISTINCT planes
    assert len({tuple(np.round(abs(v), 9)) for v in n}) >= 2
    assert all(_parallel(v, _a_axes()) for v in b)
    assert np.allclose((n * b).sum(1), 0.0, atol=1e-12)


def test_pyramidal_ca_planes_are_10m11_and_directions_c_plus_a():
    """12 first-order <c+a> systems: {10-11}<11-23> (the 6 second-order
    {11-22}<11-23> systems are a different set; the old docstring said {11-22})."""
    n, b = get_slip_systems("hcp_pyramidal_ca", c_over_a=CA)
    planes = _planes(_HK_10, (1, -1))
    assert len(n) == 12
    assert all(_parallel(v, planes) for v in n)
    c_plus_a = [s * a + np.array([0.0, 0.0, CA]) for a in _a_axes() for s in (1, -1)]
    assert all(_parallel(v, c_plus_a) for v in b)
    assert np.allclose((n * b).sum(1), 0.0, atol=1e-12)


def test_default_frame_is_midas_and_crystallographic_is_30deg_off():
    assert HCP_FRAMES == ("midas", "crystallographic")
    n_m, b_m = get_slip_systems("hcp_prismatic", c_over_a=CA)
    n_t, b_t = get_slip_systems("hcp_prismatic", c_over_a=CA, frame="crystallographic")
    # textbook: <a> along x; MIDAS: no <a> along x, but the set is the 30-deg rotated one
    assert any(np.allclose(abs(v), [1, 0, 0], atol=1e-12) for v in b_t)
    assert not any(np.allclose(abs(v), [1, 0, 0], atol=1e-6) for v in b_m)
    # the rotation maps one onto the other (-30 deg about z, row vectors)
    c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    np.testing.assert_allclose(b_m, b_t @ R.T, atol=1e-12)
    np.testing.assert_allclose(n_m, n_t @ R.T, atol=1e-12)


def test_frame_does_not_touch_cubic_and_is_validated():
    n0, b0 = get_slip_systems("fcc")
    n1, b1 = get_slip_systems("fcc", frame="crystallographic")
    np.testing.assert_array_equal(n0, n1)
    np.testing.assert_array_equal(b0, b1)
    with pytest.raises(ValueError):
        get_slip_systems("hcp_basal", c_over_a=CA, frame="bogus")
    n_a, _ = get_slip_systems_for_material("Ti")
    n_b, _ = get_slip_systems("hcp", c_over_a=CA)
    np.testing.assert_allclose(n_a, n_b)
