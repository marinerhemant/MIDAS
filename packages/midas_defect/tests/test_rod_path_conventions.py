"""rod_path conventions that were undocumented until 2026-09-10 -- pinned so a change is deliberate."""
import math
import numpy as np
from midas_defect.rod_profile import rod_path, DROP_NO_OMEGA

A, C = 3.6116, 19.2516
GEO = dict(wavelength_A=0.42459, lsd_um=349640.6, pixel_um=172.0, bc_row=810.0, bc_col=737.0,
           n_rows=1679, n_cols=1475, omega_lo_deg=-180.0, omega_hi_deg=180.0)
B1 = np.diag([1 / A, 1 / A, 1 / C])
L = np.arange(-12.0, 12.01, 0.25)


def _U():
    ax, az = math.radians(60.0), math.radians(10.0)
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return rx @ rz


def test_omega_sign_changes_only_the_reported_omega():
    p = rod_path(_U(), B1, 1, 0, L, omega_sign=+1, q_convention="1/d", **GEO)
    m = rod_path(_U(), B1, 1, 0, L, omega_sign=-1, q_convention="1/d", **GEO)
    assert len(p.L) > 0
    np.testing.assert_array_equal(p.L, m.L)
    np.testing.assert_allclose(p.row, m.row)
    np.testing.assert_allclose(p.col, m.col)
    np.testing.assert_allclose((p.omega_deg + m.omega_deg + 180.0) % 360.0 - 180.0, 0.0, atol=1e-9)


def test_a_two_pi_B_on_the_default_convention_drops_points_instead_of_raising():
    ok = rod_path(_U(), 2 * math.pi * B1, 1, 0, L, q_convention="2pi/d", **GEO)
    bad = rod_path(_U(), 2 * math.pi * B1, 1, 0, L, **GEO)            # default "1/d"
    assert len(ok.L) > 0
    assert len(bad.L) < len(ok.L)
    assert bad.dropped[DROP_NO_OMEGA] > ok.dropped[DROP_NO_OMEGA]
