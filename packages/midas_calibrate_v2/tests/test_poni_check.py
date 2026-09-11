"""check_poni_against_friedel -- the data decide which reading of a PONI is right (2026-09-10)."""
import math
import numpy as np
from midas_calibrate_v2.poni_check import check_poni_against_friedel, poni_readings

NR, NC = 1679, 1475


def _friedel_points(r0, c0, n=90, seed=0):
    rng = np.random.default_rng(seed)
    rad = rng.uniform(120.0, 600.0, n); ang = rng.uniform(0.0, 2 * math.pi, n)
    a = np.c_[r0 + rad * np.sin(ang), c0 + rad * np.cos(ang)]
    b = np.c_[2 * r0 - a[:, 0], 2 * c0 - a[:, 1]] + rng.normal(0.0, 0.3, (n, 2))
    junk = np.c_[rng.uniform(0, NR, 40), rng.uniform(0, NC, 40)]
    pts = np.vstack([a, b, junk])
    keep = (pts[:, 0] > 0) & (pts[:, 0] < NR) & (pts[:, 1] > 0) & (pts[:, 1] < NC)
    return pts[keep]


GEO = dict(n_rows=NR, n_cols=NC, lsd_um=349621.8, pixel_um=172.0, rot1_rad=0.0063, rot2_rad=0.0035,
           n_null=40)


def test_there_are_eight_readings():
    assert len(poni_readings(860.8, 751.4, NR, NC)) == 8


def test_a_row_flipped_poni_is_caught_and_the_column_flip_reported_undecidable():
    r0, c0 = 810.3, 737.4                      # centre ON the central column, as on the La3Ni2O7 Pilatus
    pts = _friedel_points(r0, c0)
    chk = check_poni_against_friedel(pts, poni1_px=NR - 1 - (r0 + 7.0), poni2_px=c0 + 14.0, **GEO)
    assert chk.axes == "as given" and chk.row_flip == "flipped" and chk.col_flip == "undecidable"
    assert abs(chk.friedel_row - r0) < 1.0 and abs(chk.friedel_col - c0) < 1.0 and chk.within_expected


def test_a_correct_poni_is_left_alone_when_the_centre_is_off_the_middle():
    r0, c0 = 610.0, 500.0
    pts = _friedel_points(r0, c0, seed=1)
    chk = check_poni_against_friedel(pts, poni1_px=r0 + 7.0, poni2_px=c0 + 14.0, **GEO)
    assert (chk.axes, chk.row_flip, chk.col_flip) == ("as given", "as given", "as given")


def test_swapped_axes_are_caught():
    r0, c0 = 610.0, 500.0
    pts = _friedel_points(r0, c0, seed=2)
    chk = check_poni_against_friedel(pts, poni1_px=c0 + 14.0, poni2_px=r0 + 7.0, **GEO)
    assert chk.axes == "swapped" and chk.row_flip == "as given" and chk.col_flip == "as given"
