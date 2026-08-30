"""Area and intensity conservation in the pixel/bin intersection.

Two properties, which fail independently:

**Conservation.** The (R, eta) bins tile the plane, so the areas one pixel
contributes to every bin it touches must sum to that pixel's own area. This is
an identity — it needs no reference implementation, and any deviation means
area, and therefore intensity, is being created or destroyed.

**Absolute accuracy.** Each individual (pixel, bin) area must be right, not
merely right in aggregate. Checked against closed-form cases and against an
independent brute-force occupancy computation.

Both were broken. ``QUAD_ORDER = (0, 1, 3, 2)`` is a closed boundary walk only
for Z-ordered corners, but ``PIXEL_CORNER_OFFSETS`` stored them in boundary
order, so the walk traced two sides and two diagonals — a bowtie. That made
``point_in_quad`` answer incorrectly and ``pixel_bin_intersect`` search the
wrong segments for crossings. Interior pixels retained a MEAN of 0.84 of their
area, worst case 0.50. See ``dev/verify_intersection_accuracy.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from midas_integrate.geometry import (
    pixel_bin_intersect, REta_to_YZ, calc_eta_angle,
    PIXEL_CORNER_OFFSETS, QUAD_ORDER,
)

RMIN, RBIN, ETAMIN, ETABIN = 2.0, 1.0, -180.0, 30.0
N_R, N_ETA = 60, 12


def _bin_corners(R0, R1, E0, E1):
    return np.array([REta_to_YZ(R0, E0), REta_to_YZ(R0, E1),
                     REta_to_YZ(R1, E0), REta_to_YZ(R1, E1)], dtype=float)


def _pixel(cy, cz):
    return np.array([[cy + dy, cz + dz] for dy, dz in PIXEL_CORNER_OFFSETS])


def _quad_area(pc):
    idx = list(QUAD_ORDER)
    Y, Z = pc[idx, 0], pc[idx, 1]
    return 0.5 * abs(float((Y * np.roll(Z, -1) - np.roll(Y, -1) * Z).sum()))


def _sum_over_bins(pc):
    R = np.hypot(pc[:, 0], pc[:, 1])
    E = np.array([calc_eta_angle(*p) for p in pc])
    rl = max(int(np.floor((R.min() - RMIN) / RBIN)), 0)
    rh = min(int(np.floor((R.max() - RMIN) / RBIN)), N_R - 1)
    el = max(int(np.floor((E.min() - ETAMIN) / ETABIN)), 0)
    eh = min(int(np.floor((E.max() - ETAMIN) / ETABIN)), N_ETA - 1)
    tot = 0.0
    for r in range(rl, rh + 1):
        for e in range(el, eh + 1):
            R0, R1 = RMIN + r * RBIN, RMIN + (r + 1) * RBIN
            E0, E1 = ETAMIN + e * ETABIN, ETAMIN + (e + 1) * ETABIN
            tot += pixel_bin_intersect(pc, R0, R1, E0, E1,
                                       bin_corners=_bin_corners(R0, R1, E0, E1))
    return tot


# ------------------------------------------------------------- the walk

def test_quad_order_traverses_a_closed_boundary_not_a_bowtie():
    """The invariant everything else here depends on.

    Guards the corner order directly: if someone reorders
    ``PIXEL_CORNER_OFFSETS`` without reordering ``QUAD_ORDER``, the walk
    silently becomes a bowtie again and every area goes subtly wrong.
    """
    offs = PIXEL_CORNER_OFFSETS
    for e in range(4):
        p = offs[QUAD_ORDER[e]]
        q = offs[QUAD_ORDER[(e + 1) % 4]]
        shares_y = abs(p[0] - q[0]) < 1e-15
        shares_z = abs(p[1] - q[1]) < 1e-15
        assert shares_y != shares_z, (
            f"edge {tuple(p)}->{tuple(q)} is a diagonal, not a side; "
            f"QUAD_ORDER {QUAD_ORDER} does not traverse these offsets"
        )


# ------------------------------------------------------- conservation

@pytest.mark.parametrize("cy,cz", [
    (0.0, 27.5), (0.3, 27.0), (-23.13 + 0.5, 12.91 + 0.5),   # straddles R and eta
    (12.0, 12.0), (-15.0, 8.0), (0.0, 10.0), (7.0, -19.0),
])
def test_single_pixel_conserves(cy, cz):
    pc = _pixel(cy, cz)
    assert abs(_sum_over_bins(pc) - _quad_area(pc)) < 1e-12


def test_conservation_over_many_random_pixels():
    """The mean deficit before the fix was 16%, worst case 50%."""
    rng = np.random.default_rng(0)
    worst = 0.0
    n = 0
    for _ in range(600):
        ang = rng.uniform(-math.pi, math.pi)
        rad = rng.uniform(8.0, 45.0)
        pc = _pixel(rad * math.cos(ang), rad * math.sin(ang))
        R = np.hypot(pc[:, 0], pc[:, 1])
        if R.min() < RMIN + 1.5 or R.max() > RMIN + N_R * RBIN - 1.5:
            continue
        n += 1
        worst = max(worst, abs(_sum_over_bins(pc) - _quad_area(pc)))
    assert n > 300, "not enough pixels exercised"
    assert worst < 1e-12, f"worst area non-conservation {worst:.3e}"


def test_the_regression_case_from_the_bowtie():
    """The pixel that returned zero for a cell it demonstrably overlaps.

    Pre-fix: r=25/eta=7 gave exactly 0.000000 and the four cells summed to
    0.495611 of a pixel area of 1.0.
    """
    pc = _pixel(-23.63, 13.41)
    cells = {}
    for r, e in ((24, 7), (24, 8), (25, 7), (25, 8)):
        R0, R1 = RMIN + r, RMIN + r + 1
        E0, E1 = ETAMIN + e * ETABIN, ETAMIN + (e + 1) * ETABIN
        cells[(r, e)] = pixel_bin_intersect(
            pc, R0, R1, E0, E1, bin_corners=_bin_corners(R0, R1, E0, E1))
    assert cells[(25, 7)] > 1e-3, "the cell that used to come back empty"
    assert abs(sum(cells.values()) - _quad_area(pc)) < 1e-12


# ---------------------------------------------------------- accuracy

def test_pixel_wholly_inside_one_bin_is_exact():
    pc = _pixel(0.0, 100.0)
    a = pixel_bin_intersect(pc, 95.0, 105.0, -30.0, 30.0,
                            bin_corners=_bin_corners(95., 105., -30., 30.))
    assert abs(a - 1.0) < 1e-12


def test_eta_ray_through_the_centre_splits_exactly_in_half():
    pc = _pixel(0.0, 100.0)                 # centred on eta = 0
    lo = pixel_bin_intersect(pc, 95., 105., -30., 0.,
                             bin_corners=_bin_corners(95., 105., -30., 0.))
    hi = pixel_bin_intersect(pc, 95., 105., 0., 30.,
                             bin_corners=_bin_corners(95., 105., 0., 30.))
    assert abs(lo - 0.5) < 1e-9
    assert abs(hi - 0.5) < 1e-9


def test_r_boundary_through_the_centre_splits_almost_in_half():
    """Almost, not exactly: the arc is curved, so the inner piece is smaller.
    The SUM must still be exact."""
    pc = _pixel(0.0, 1000.0)
    i = pixel_bin_intersect(pc, 999., 1000., -5., 5.,
                            bin_corners=_bin_corners(999., 1000., -5., 5.))
    o = pixel_bin_intersect(pc, 1000., 1001., -5., 5.,
                            bin_corners=_bin_corners(1000., 1001., -5., 5.))
    assert abs(i + o - 1.0) < 1e-12
    assert i < o                              # inner piece is the smaller one
    assert abs(i - 0.5) < 1e-3


def _brute_force(pc, R0, R1, E0, E1, N=1200):
    y0, y1 = pc[:, 0].min(), pc[:, 0].max()
    z0, z1 = pc[:, 1].min(), pc[:, 1].max()
    gy = np.linspace(y0, y1, N + 1)
    gz = np.linspace(z0, z1, N + 1)
    YY, ZZ = np.meshgrid(0.5 * (gy[:-1] + gy[1:]), 0.5 * (gz[:-1] + gz[1:]),
                         indexing="xy")
    R = np.hypot(YY, ZZ)
    E = np.degrees(np.arctan2(-YY, ZZ))
    inside = (R >= R0) & (R < R1) & (E >= E0) & (E < E1)
    return float(inside.sum()) * (gy[1] - gy[0]) * (gz[1] - gz[0])


@pytest.mark.parametrize("cy,cz", [(0.0, 27.4), (-19.0, 19.5), (24.0, -12.0)])
def test_individual_cell_areas_match_brute_force(cy, cz):
    """Aggregate conservation can hold while individual cells are wrong;
    this pins the cells themselves against an independent computation."""
    pc = _pixel(cy, cz)
    R = np.hypot(pc[:, 0], pc[:, 1])
    E = np.array([calc_eta_angle(*p) for p in pc])
    rl = int(np.floor((R.min() - RMIN) / RBIN))
    rh = int(np.floor((R.max() - RMIN) / RBIN))
    el = int(np.floor((E.min() - ETAMIN) / ETABIN))
    eh = int(np.floor((E.max() - ETAMIN) / ETABIN))
    for r in range(rl, rh + 1):
        for e in range(el, eh + 1):
            R0, R1 = RMIN + r * RBIN, RMIN + (r + 1) * RBIN
            E0, E1 = ETAMIN + e * ETABIN, ETAMIN + (e + 1) * ETABIN
            a = pixel_bin_intersect(pc, R0, R1, E0, E1,
                                    bin_corners=_bin_corners(R0, R1, E0, E1))
            ref = _brute_force(pc, R0, R1, E0, E1)
            assert abs(a - ref) < 5e-3, (
                f"cell r={r} e={e}: kernel {a:.8f} vs brute force {ref:.8f}")
