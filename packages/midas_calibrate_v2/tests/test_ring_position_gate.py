"""The ring-position gate, tested against an INDEPENDENT renderer.

The image is drawn by exact ray / plane geometry in this file (no MIDAS forward model), so a
convention error in the package cannot hide here. The 2theta map handed to the gate is the
renderer's own exact map for the true geometry, then deliberately damaged:

* true map                -> PASS, median far below a pixel
* map shifted by 1 px     -> the gate reads about 2/pi px (the radial projection of a shift)
* map shifted by 8 px     -> FAIL, and it fails because the rings are no longer found (cells
                             lost), not because the survivors' residual is large
* map scaled by 1.001     -> still PASS: wavelength / distance are degenerate with a global
                             scale, the gate removes it and reports it
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_calibrate_v2.pipelines.tilted import calibrate_tilted, clean_frame
from midas_calibrate_v2.ring_position_gate import ring_position_gate

N = 800
PX_UM = 200.0
L_PERP_UM = 500_000.0
RINGS = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])           # degrees 2theta


def _frame(alpha_deg=20.0, phi_deg=35.0, shift_px=(0.0, 0.0), scale=1.0):
    """Return (image, tth_map_deg, lsd_along_beam_um) for a plane tilted by alpha about phi.

    Beam along +z. The point where the beam hits the plane (2theta = 0) sits at the image centre.
    ``shift_px`` moves the pixel grid used for the MAP only (the image stays true).
    """
    a, p = np.radians(alpha_deg), np.radians(phi_deg)
    n = np.array([np.sin(a) * np.cos(p), np.sin(a) * np.sin(p), np.cos(a)])   # plane normal
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - n * (ref @ n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    c = np.array([0.0, 0.0, L_PERP_UM / np.cos(a)])             # where the beam hits the plane

    def tth_of(dx, dy):
        j, i = np.meshgrid(np.arange(N) + dx - (N - 1) / 2, np.arange(N) + dy - (N - 1) / 2)
        pts = c[None, None, :] + (j[..., None] * PX_UM) * e1 + (i[..., None] * PX_UM) * e2
        return np.degrees(np.arccos(pts[..., 2] / np.linalg.norm(pts, axis=-1)))

    true = tth_of(0.0, 0.0)
    img = np.full((N, N), 50.0)
    for k, t in enumerate(RINGS):
        img += (800.0 + 100 * k) * np.exp(-0.5 * ((true - t) / 0.03) ** 2)
    m = tth_of(*shift_px)
    if scale != 1.0:
        m = np.degrees(np.arctan(np.tan(np.radians(m)) * scale))
    lsd_along_beam = L_PERP_UM / np.cos(a)
    return img, m, lsd_along_beam


def test_true_geometry_passes_with_subpixel_residual():
    img, m, lsd = _frame()
    g = ring_position_gate(img, m, RINGS, lsd_um=lsd, px_um=PX_UM)
    assert g.verdict == "PASS", str(g)
    assert g.median_px < 0.05 and g.p95_px < 0.2, str(g)
    assert g.n_rings_used >= 5
    assert g.n_fits >= 0.9 * g.n_cells_expected


def test_one_pixel_shift_is_read_as_a_fraction_of_a_pixel_projection():
    img, m, lsd = _frame(shift_px=(1.0, 0.0))
    g = ring_position_gate(img, m, RINGS, lsd_um=lsd, px_um=PX_UM)
    assert 0.4 < g.median_px < 0.9, str(g)           # mean |radial component| of a 1 px shift = 2/pi
    assert g.verdict in ("WARN", "FAIL"), str(g)


def test_large_shift_fails_because_cells_are_lost():
    img, m, lsd = _frame(shift_px=(8.0, 5.0))
    g = ring_position_gate(img, m, RINGS, lsd_um=lsd, px_um=PX_UM)
    assert g.verdict == "FAIL", str(g)
    assert g.n_fits < 0.5 * g.n_cells_expected or g.median_px > 1.0, str(g)


def test_global_scale_is_removed_and_reported_not_penalised():
    img, m, lsd = _frame(scale=1.001)
    g = ring_position_gate(img, m, RINGS, lsd_um=lsd, px_um=PX_UM)
    assert g.median_px < 0.1, str(g)
    assert g.scale == pytest.approx(1.001, abs=2e-4), str(g)


def test_mask_excludes_bad_pixels():
    img, m, lsd = _frame()
    img = img.copy()
    img[300:340, :] = 0.0                              # a dead band; without the mask these fits break
    bad = np.zeros(img.shape, bool)
    bad[300:340, :] = True
    g = ring_position_gate(img, m, RINGS, lsd_um=lsd, px_um=PX_UM, mask=bad)
    assert g.verdict == "PASS", str(g)


def test_shape_mismatch_is_an_error():
    img, m, lsd = _frame()
    with pytest.raises(ValueError):
        ring_position_gate(img, m[:-1], RINGS, lsd_um=lsd, px_um=PX_UM)


def test_clean_frame_zeroes_negatives_and_dead_columns():
    raw = np.full((60, 80), 100.0)
    raw[10, 10] = -5.0
    raw[:, 40] = 1.0                                   # dead column (< 20 % of the median column)
    img, bad = clean_frame(raw)
    assert bad[10, 10] and bad[:, 40].all()
    assert (img[bad] == 0.0).all()
    assert bad[10, 12] and not bad[30, 20]             # dilated by 2 px, interior untouched


def test_calibrate_tilted_rejects_bad_input():
    with pytest.raises(ValueError):
        calibrate_tilted(np.zeros((4, 4, 4)), wavelength_A=0.5, px_um=200.0)
    with pytest.raises(ValueError):
        calibrate_tilted(np.zeros((64, 64)), wavelength_A=-1.0, px_um=200.0)


def test_calibrate_tilted_rejects_nonpositive_distance_hint():
    with pytest.raises(ValueError):
        calibrate_tilted(np.ones((64, 64)), wavelength_A=0.5, px_um=200.0, lsd_um=-5.0)


def test_pixel_unit_is_the_local_radial_gradient_not_a_distance_formula():
    """At 45 deg tilt the old Lsd/px/cos^2 conversion was off by up to +-30 % with azimuth. The unit
    is now the map's own gradient, so the reading must not depend on lsd_um and must equal the true
    radial component of the shift, n.s with n the unit normal of the level sets of the map."""
    img, m, lsd = _frame(alpha_deg=45.0, phi_deg=35.0, shift_px=(1.0, 0.0))
    g1 = ring_position_gate(img, m, RINGS, lsd_um=lsd, px_um=PX_UM)
    g2 = ring_position_gate(img, m, RINGS, lsd_um=3.0 * lsd, px_um=PX_UM)
    g3 = ring_position_gate(img, m, RINGS)
    assert g1.median_px == pytest.approx(g2.median_px, rel=1e-9) == pytest.approx(g3.median_px, rel=1e-9)
    gy, gx = np.gradient(m)
    n_dot_s = np.abs(gx) / np.hypot(gx, gy)                  # shift is (1, 0) px along columns
    near = np.zeros(m.shape, bool)
    for t in RINGS:
        near |= np.abs(m - t) < 0.05
    expected = float(np.median(n_dot_s[near]))
    assert g3.median_px == pytest.approx(expected, rel=0.15), (g3.median_px, expected)


def test_duplicate_ring_entries_do_not_drop_rings():
    img, m, lsd = _frame()
    g1 = ring_position_gate(img, m, RINGS)
    g2 = ring_position_gate(img, m, np.concatenate([RINGS, RINGS]))
    assert g2.n_rings_used == g1.n_rings_used and g2.n_fits == g1.n_fits, (str(g1), str(g2))


def test_localized_error_is_flagged_not_hidden_by_the_survivors():
    """A 4 px error over 40 % of the azimuth used to PASS (median 0.003 px, 57 % of cells found)."""
    img, m_true, lsd = _frame(alpha_deg=20.0, phi_deg=35.0)
    _, m_bad, _ = _frame(alpha_deg=20.0, phi_deg=35.0, shift_px=(4.0, 0.0))
    yy, xx = np.indices(m_true.shape)
    az = np.arctan2(yy - (N - 1) / 2, xx - (N - 1) / 2)
    m = np.where(np.abs(az) < 0.4 * np.pi, m_bad, m_true)
    g = ring_position_gate(img, m, RINGS)
    assert g.verdict in ("WARN", "FAIL"), str(g)
    assert len(g.bad_sectors) >= 2 or g.n_fits < 0.8 * g.n_cells_expected, str(g)


def test_ring_list_with_wrong_lattice_constant_is_warned_not_passed():
    img, m, lsd = _frame()
    wrong = np.degrees(np.arctan(np.tan(np.radians(RINGS)) * 1.002))      # lattice constant 0.2 % off
    g = ring_position_gate(img, m, wrong)
    assert g.verdict != "PASS", str(g)
    assert abs(g.scale - 1.0) > 1e-3, str(g)
