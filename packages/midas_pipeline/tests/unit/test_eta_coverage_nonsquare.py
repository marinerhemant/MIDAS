"""Panel eta coverage for a non-square detector (#75 follow-up).

A real 1028 x 512 Eiger panel was enumerated as 1028 x 1028, so rings were
credited with arcs on pixels that do not exist (``global_powder`` logged
"non-square panel ... will use 1028" and went on).
"""
from __future__ import annotations

import numpy as np

from midas_pipeline.eta_coverage import (
    _pixel_rt_eta, compute_panel_eta_coverage,
)
from midas_pipeline.stages.global_powder import _coverage_from_geometry

PX = 75.0
LSD = 400_000.0
# 200 x 80 panel, beam centre mid-panel: a ring of radius 60 px runs off the
# short (Z) axis, so a square 200 x 200 enumeration over-covers it.
N_Y, N_Z = 200, 80
Y_BC, Z_BC = 100.0, 40.0
R_PX = 60.0


def _arcs(**kw):
    return compute_panel_eta_coverage(
        n_pixels=N_Y, px_um=PX, lsd_um=LSD, y_bc_px=Y_BC, z_bc_px=Z_BC,
        tx_deg=0.0, ty_deg=0.0, tz_deg=0.0,
        ring_radii_um=[(1, R_PX * PX)], width_um=2 * PX, **kw)


def test_rectangular_grid_is_the_corner_of_the_square_one():
    sq_r, sq_eta = _pixel_rt_eta(N_Y, PX, LSD, Y_BC, Z_BC, 0, 0, 0)
    r, eta = _pixel_rt_eta(N_Y, PX, LSD, Y_BC, Z_BC, 0, 0, 0, n_pixels_z=N_Z)
    assert r.shape == (N_Y, N_Z)
    np.testing.assert_array_equal(r, sq_r[:, :N_Z])
    np.testing.assert_array_equal(eta, sq_eta[:, :N_Z])


def test_square_default_is_unchanged():
    r, _ = _pixel_rt_eta(N_Y, PX, LSD, Y_BC, Z_BC, 0, 0, 0)
    assert r.shape == (N_Y, N_Y)


def test_rect_arcs_cover_only_pixels_that_exist():
    square = _arcs()
    rect = _arcs(n_pixels_z=N_Z)

    def width(arcs):
        return sum(a.eta_hi_deg - a.eta_lo_deg for a in arcs)

    assert width(rect) < width(square) - 30.0      # the ring leaves the panel
    # Every pixel of the real panel in the ring band lies inside a rect arc.
    r, eta = _pixel_rt_eta(N_Y, PX, LSD, Y_BC, Z_BC, 0, 0, 0, n_pixels_z=N_Z)
    band = np.abs(r - R_PX) <= 2.0
    assert band.any()
    for e in eta[band]:
        assert any(a.eta_lo_deg - 1.0 <= e <= a.eta_hi_deg + 1.0 for a in rect)
    # ... and the square enumeration credits eta values no real pixel has.
    r_sq, eta_sq = _pixel_rt_eta(N_Y, PX, LSD, Y_BC, Z_BC, 0, 0, 0)
    ghost = np.abs(r_sq - R_PX) <= 2.0
    ghost[:, :N_Z] = False                       # keep only the non-existent pixels
    n_ghost = sum(not any(a.eta_lo_deg - 1.0 <= e <= a.eta_hi_deg + 1.0 for a in rect)
                  for e in eta_sq[ghost])
    assert n_ghost > 0


def test_global_powder_reads_nrpixels_y_and_z(caplog):
    text = (f"RingNumbers 1\nRingRadii {R_PX * PX}\nLsd {LSD}\nYBC {Y_BC}\nZBC {Z_BC}\n"
            f"tx 0\nty 0\ntz 0\npx {PX}\nNrPixelsY {N_Y}\nNrPixelsZ {N_Z}\nWidth {2 * PX}\n")
    with caplog.at_level("WARNING"):
        got = _coverage_from_geometry(text)
    assert "non-square" not in caplog.text
    want = _arcs(n_pixels_z=N_Z)
    assert [(a.ring_nr, a.eta_lo_deg, a.eta_hi_deg) for a in got[1]] == \
           [(a.ring_nr, a.eta_lo_deg, a.eta_hi_deg) for a in want]
