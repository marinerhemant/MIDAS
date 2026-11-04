"""frame_rotation: textbook (Lattice) frame -> MIDAS (B matrix, a* along x) frame."""
import math

import numpy as np
import pytest

from midas_hkls import Lattice, frame_rotation, midas_b_matrix

CELLS = {
    "cubic": (3.6, 3.6, 3.6, 90, 90, 90),
    "tetragonal": (3.0, 3.0, 4.5, 90, 90, 90),
    "orthorhombic": (3.0, 4.0, 5.0, 90, 90, 90),
    "hexagonal": (2.95, 2.95, 4.68, 90, 90, 120),
    "monoclinic-b-105": (4.0, 5.0, 6.0, 90, 105, 90),
    "monoclinic-b-95": (4.0, 5.0, 6.0, 90, 95, 90),
    "monoclinic-a-105": (4.0, 5.0, 6.0, 105, 90, 90),
    "monoclinic-c-105": (4.0, 5.0, 6.0, 90, 90, 105),
    "rhombohedral-axes": (5.0, 5.0, 5.0, 55, 55, 55),
    "triclinic": (4.0, 5.0, 6.0, 80, 95, 100),
}


def _angle_axis(R):
    ang = math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, np.argmin(abs(w - 1))])
    return ang, ax / np.linalg.norm(ax)


@pytest.mark.parametrize("name", list(CELLS))
def test_is_a_proper_rotation(name):
    R = frame_rotation(CELLS[name])
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["cubic", "tetragonal", "orthorhombic"])
def test_identity_when_all_angles_are_90(name):
    np.testing.assert_allclose(frame_rotation(CELLS[name]), np.eye(3), atol=1e-12)


def test_hexagonal_is_30_degrees_about_c():
    ang, ax = _angle_axis(frame_rotation(CELLS["hexagonal"]))
    assert ang == pytest.approx(30.0, abs=1e-9)
    assert abs(ax[2]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name,expect,axis", [
    ("monoclinic-b-105", 15.0, 1), ("monoclinic-b-95", 5.0, 1),
    ("monoclinic-a-105", 15.0, 0), ("monoclinic-c-105", 15.0, 2),
])
def test_monoclinic_is_angle_minus_90_about_the_unique_axis(name, expect, axis):
    ang, ax = _angle_axis(frame_rotation(CELLS[name]))
    assert ang == pytest.approx(expect, abs=1e-9)
    assert abs(ax[axis]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", list(CELLS))
def test_maps_the_textbook_reciprocal_vectors_onto_the_midas_ones(name):
    """The defining property, checked on the reciprocal lattice (independent of
    how R is assembled): R · (Lattice reciprocal vectors) = B · (h, k, l)."""
    lat = Lattice(*CELLS[name])
    g_text = np.asarray(lat.reciprocal_cartesian_vectors(), float).T     # columns a*, b*, c*
    B = midas_b_matrix(lat)
    np.testing.assert_allclose(frame_rotation(lat) @ g_text, B, atol=1e-12)
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [2, -1, 3], [-1, 2, 1]]).T
    np.testing.assert_allclose(frame_rotation(lat) @ (g_text @ hkl), B @ hkl, atol=1e-12)


def test_midas_frame_puts_a_star_along_x_and_c_star_along_z_for_hexagonal():
    B = midas_b_matrix(CELLS["hexagonal"])
    a_star, b_star, c_star = B[:, 0], B[:, 1], B[:, 2]
    assert a_star[1] == pytest.approx(0, abs=1e-12) and a_star[2] == pytest.approx(0, abs=1e-12)
    assert c_star[0] == pytest.approx(0, abs=1e-12) and c_star[1] == pytest.approx(0, abs=1e-12)
    assert math.degrees(math.atan2(b_star[1], b_star[0])) == pytest.approx(60.0)
    # real-space a at -30 deg, b at 90 deg
    A = np.linalg.inv(B).T
    az = lambda v: math.degrees(math.atan2(v[1], v[0]))
    assert az(A[:, 0]) == pytest.approx(-30.0)
    assert az(A[:, 1]) == pytest.approx(90.0)


def test_accepts_a_lattice_or_a_six_tuple():
    np.testing.assert_allclose(frame_rotation(Lattice(*CELLS["hexagonal"])),
                               frame_rotation(CELLS["hexagonal"]))
