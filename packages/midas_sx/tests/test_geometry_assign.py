"""Geometry and hkl assignment, built from independent numpy so a convention error cannot cancel."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from midas_hkls import Lattice
from midas_sx.assign import assign_hkl, qsample_of_hkl, reciprocal_basis
from midas_sx.geometry import Scan, make_geometry, spot_directions

SCAN = Scan(omega_start=-40.0, omega_step=0.25, n_frames=320)
GEOM = make_geometry(wavelength_A=0.4133, dist_mm=100.0, pixel_mm=0.3, centre_rc=(255.5, 255.5),
                     shape=(512, 512), scan=SCAN)


def test_scan_frame_centres():
    assert SCAN.omega_of(0) == pytest.approx(-40.0 + 0.125)
    assert SCAN.omega_of(319.0) == pytest.approx(40.0 - 0.125)
    assert SCAN.omega_end == pytest.approx(40.0)


def test_pixel_at_the_beam_centre_is_straight_through():
    s, tth, eta = spot_directions([255.5], [255.5], GEOM)
    assert np.allclose(s[0], [1, 0, 0], atol=1e-12) and tth[0] == pytest.approx(0.0, abs=1e-9)


def test_midas_pixel_axes():
    """Y = -(col - bcy) px and Z = (row - bcz) px: +col -> -y, +row -> +z (midas_defect convention)."""
    d = 100.0
    px = 0.3
    # 20 px to higher column, and 20 px to higher row
    s, tth, eta = spot_directions([255.5, 275.5], [275.5, 255.5], GEOM)
    assert s[0, 1] < 0 and abs(s[0, 2]) < 1e-12                    # +col -> -y
    assert s[1, 2] > 0 and abs(s[1, 1]) < 1e-12                    # +row -> +z
    assert tth[0] == pytest.approx(np.degrees(np.arctan(20 * px / d)), rel=1e-9)
    assert eta[0] == pytest.approx(90.0) and eta[1] == pytest.approx(0.0, abs=1e-9)   # eta = atan2(-y, z)


def _rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _crossing_spots(lattice, U, hkls, lam, sign=1.0):
    """Independent Ewald crossings: q_lab = Rz(+omega) U B h, s = (k s0 + q)/k at f(omega) = 0."""
    B = reciprocal_basis(lattice)
    k = 1.0 / lam
    out_s, out_om, out_h = [], [], []
    for h in hkls:
        q0 = U @ B @ np.asarray(h, float)
        f = lambda om: np.linalg.norm(np.array([k, 0, 0]) + _rz(om) @ q0) - k
        grid = np.linspace(-40, 40, 801)
        v = np.array([f(g) for g in grid])
        for i in np.flatnonzero(v[:-1] * v[1:] < 0):
            om = brentq(f, grid[i], grid[i + 1], xtol=1e-12)
            s = (np.array([k, 0, 0]) + _rz(om) @ q0) / k
            out_s.append(s); out_om.append(sign * om); out_h.append(h)
    return np.array(out_s), np.array(out_om), np.array(out_h)


@pytest.mark.parametrize("cell", [(3.6, 3.6, 3.6, 90, 90, 90), (3.83, 3.83, 20.5, 90, 90, 90),
                                  (5.1, 6.3, 7.7, 83.0, 101.0, 112.0)])
def test_assignment_recovers_hkl_for_any_cell_and_flags_a_wrong_omega_sign(cell):
    lat = Lattice(*cell)
    U = Rotation.random(random_state=np.random.RandomState(5)).as_matrix()
    hkls = [(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-4, 5) if (h, k, l) != (0, 0, 0)]
    s, om, h = _crossing_spots(lat, U, hkls, 0.4133)
    assert len(s) > 40
    hi, hf, res = assign_hkl(s, om, lat, U, 0.4133)
    assert np.array_equal(hi, h) and res.max() < 1e-9
    # positive control: the same spots read with the opposite omega sign must NOT assign
    _, _, res_bad = assign_hkl(s, -om, lat, U, 0.4133)
    assert np.median(res_bad) > 0.15


def test_reciprocal_basis_matches_gemmi_for_a_triclinic_cell():
    gemmi = pytest.importorskip("gemmi")
    cell = (5.1, 6.3, 7.7, 83.0, 101.0, 112.0)
    A = np.array(gemmi.UnitCell(*cell).orth.mat.tolist())
    assert np.allclose(reciprocal_basis(Lattice(*cell)), np.linalg.inv(A).T, atol=1e-12)


def test_qsample_of_hkl_has_length_one_over_d():
    lat = Lattice(3.6, 3.6, 3.6, 90, 90, 90)
    U = Rotation.random(random_state=np.random.RandomState(1)).as_matrix()
    q = qsample_of_hkl([(1, 1, 1), (2, 0, 0)], lat, U)
    assert np.linalg.norm(q[0]) == pytest.approx(1.0 / lat.d_spacing(1, 1, 1))
    assert np.linalg.norm(q[1]) == pytest.approx(1.0 / lat.d_spacing(2, 0, 0))


def test_calculated_crossing_matches_an_independent_root_finder_and_declines_when_there_is_none():
    from midas_sx.assign import calculated_crossing
    lat = Lattice(3.83, 3.83, 20.5, 90, 90, 90)
    U = Rotation.random(random_state=np.random.RandomState(8)).as_matrix()
    hkls = [(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-9, 10) if (h, k, l) != (0, 0, 0)]
    s, om, h = _crossing_spots(lat, U, hkls, 0.4133)
    assert len(s) > 40
    q = qsample_of_hkl(h, lat, U)
    for i in range(0, len(s), 7):
        w, s_star = calculated_crossing(q[i], om[i] + 0.37, 0.4133)          # start 0.37 deg off the true crossing
        assert w == pytest.approx(om[i], abs=1e-9)
        assert np.allclose(s_star, s[i], atol=1e-12)
    # nothing within the window -> None
    assert calculated_crossing(q[0], om[0] + 30.0, 0.4133, window_deg=1.0) is None


def test_calculated_crossing_finds_a_root_that_sits_exactly_on_a_grid_node():
    """Started AT the exact Ewald crossing (a predicted target), |v| - k rounds to exactly 0.0 at the middle grid node in some cases; a strict product < 0 bracket then returned None."""
    from midas_sx.assign import calculated_crossing, ewald_crossings
    rng = np.random.default_rng(7)
    n_none = 0
    for _ in range(300):
        q = rng.normal(size=3)
        q *= rng.uniform(0.3, 1.2) / np.linalg.norm(q)
        for w in ewald_crossings(2.0 * np.pi * q, 0.7107):                       # ewald_crossings: q WITH the 2 pi; calculated_crossing: without
            c = calculated_crossing(q, float(np.degrees(w)), 0.7107)
            n_none += c is None
            if c is not None:
                assert abs(c[0] - np.degrees(w)) < 1e-6
    assert n_none == 0
