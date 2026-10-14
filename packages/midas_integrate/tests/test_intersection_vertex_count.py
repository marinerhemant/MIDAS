"""Pixel/bin intersection: conservation where the old vertex enumeration failed.

``pixel_bin_intersect`` used to build the intersection polygon from (1) pixel corners
inside the bin, (2) bin corners inside the pixel, and (3) edge crossings, sort the
vertices, and integrate any edge joining two vertices on a bin circle as an arc. It
is now the exact area of (pixel clipped to the bin's wedge) inside a disk, at R_max
minus at R_min. The old scheme failed in three ways:

* Step (3) ran only ``if len(edges) < 4``, on the idea that four vertices from (1)
  and (2) already close the polygon. They do not. A bin no larger than the pixel can
  put ONE pixel corner in the bin and THREE bin
  corners in the pixel -- four vertices, but the polygon also has two edge
  crossings, so the area came back short. Seen with bins of a pixel or less, i.e.
  the fine-binning regime (SAXS / PDF-like) where exact splitting matters most.
* Also step (3): when a pixel edge lies EXACTLY on a bin ray (a beam centre at x.5, as in every
  synthetic benchmark geometry), a pixel corner and a bin corner coincide, the
  count reaches four on duplicates, the arc crossings are skipped, and about half
  the pixel was lost (0.495 of its area).
* A straight pixel edge crossing a bin's circular boundary twice (near-tangent, as along
  the cardinal axes) was integrated as an arc between the two crossings, which over- or
  under-counted the circular cap (up to ~0.3 percent of a pixel), and returned ZERO when
  only those two vertices existed.

The area is an identity -- the (R, eta) bins tile the plane -- so these need no
reference implementation: summed over every bin the pixel touches, the areas must
equal the pixel's own quad area.

The fixture quads are random jittered unit squares collected from a 30,000-quad
stress run on the unmodified code; each one failed there (``deficit_on_head``) and
conserves to ~1e-15 now.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_integrate.geometry import (
    PIXEL_CORNER_OFFSETS, QUAD_ORDER, REta_to_YZ, calc_eta_angle, pixel_bin_intersect,
)


def _bin_corners(R0, R1, E0, E1):
    return np.array([REta_to_YZ(R0, E0), REta_to_YZ(R0, E1),
                     REta_to_YZ(R1, E0), REta_to_YZ(R1, E1)], dtype=float)


def _quad_area(pc):
    idx = list(QUAD_ORDER)
    y, z = pc[idx, 0], pc[idx, 1]
    return 0.5 * abs(float((y * np.roll(z, -1) - np.roll(y, -1) * z).sum()))


def _sum_over_bins(pc, rmin, rbin, etabin):
    """Sum of pixel_bin_intersect over every bin the pixel can touch.

    The candidate range is widened by one bin each way: a straight pixel edge
    reaches below its corners' minimum radius (chord sagitta), so a range taken
    from the corners alone can miss a bin the pixel genuinely overlaps.
    """
    R = np.hypot(pc[:, 0], pc[:, 1])
    E = np.array([calc_eta_angle(*q) for q in pc])
    r_lo = int(math.floor((R.min() - rmin) / rbin)); r_hi = int(math.floor((R.max() - rmin) / rbin))
    e_lo = int(math.floor((E.min() + 180.0) / etabin)); e_hi = int(math.floor((E.max() + 180.0) / etabin))
    tot = 0.0
    for r in range(max(r_lo - 1, 0), r_hi + 2):
        for e in range(e_lo - 1, e_hi + 2):
            R0, R1 = rmin + r * rbin, rmin + (r + 1) * rbin
            E0, E1 = -180.0 + e * etabin, -180.0 + (e + 1) * etabin
            tot += pixel_bin_intersect(pc, R0, R1, E0, E1, bin_corners=_bin_corners(R0, R1, E0, E1))
    return tot


FAILING_GENERIC = [
    dict(corners=[[-1.6269453385056436, 7.730787608616701], [-1.597088215741059, 8.72575901936856], [-0.6084352016837804, 7.7347179124791365], [-0.7507335498250504, 8.501066868654801]],
         RMIN=1.3835826268814606, RBIN=0.25, ETABIN=5.0,
         deficit_on_head=3.445e-02),
    dict(corners=[[7.316842875611456, -0.17231342920885706], [7.549307304770043, 0.7409126612040308], [8.47381605431307, -0.12698886820658994], [8.378807892780872, 0.7315931077633893]],
         RMIN=1.1681410558132612, RBIN=0.25, ETABIN=5.0,
         deficit_on_head=6.740e-03),
    dict(corners=[[-22.267252542658667, 0.9296404016856971], [-22.19632020687351, 1.9958089719356817], [-21.296479718331717, 0.9412578042970791], [-21.267107439772357, 1.840728527892052]],
         RMIN=1.199301230937657, RBIN=0.5, ETABIN=2.5,
         deficit_on_head=2.128e-03),
    dict(corners=[[3.2577664283309833, 37.48611937807797], [3.356947463118204, 38.301797191495005], [4.254108634091105, 37.31643687781452], [4.3031202230056955, 38.50625549557581]],
         RMIN=1.0145317029055383, RBIN=0.5, ETABIN=1.0,
         deficit_on_head=1.685e-02),
]

FAILING_ALIGNED = [
    dict(corners=[[9.954160138092103, 0.0], [9.954160138092103, 1.0], [10.954160138092103, 0.0], [10.954160138092103, 1.0]],
         RMIN=0.0, RBIN=1.0, ETABIN=30.0,
         deficit_on_head=6.321e-03),
    dict(corners=[[7.488841129967406, 0.0], [7.488841129967406, 1.0], [8.488841129967405, 0.0], [8.488841129967405, 1.0]],
         RMIN=0.0, RBIN=0.5, ETABIN=5.0,
         deficit_on_head=4.894e-03),
    dict(corners=[[0.0, 25.981012485345122], [0.0, 26.981012485345122], [1.0, 25.981012485345122], [1.0, 26.981012485345122]],
         RMIN=0.0, RBIN=2.0, ETABIN=30.0,
         deficit_on_head=3.083e-03),
    dict(corners=[[3.9750455241382925, 0.0], [3.9750455241382925, 1.0], [4.9750455241382925, 0.0], [4.9750455241382925, 1.0]],
         RMIN=0.0, RBIN=0.25, ETABIN=10.0,
         deficit_on_head=1.359e-02),
    dict(corners=[[-1.0, 17.997767019935427], [-1.0, 18.997767019935427], [0.0, 17.997767019935427], [0.0, 18.997767019935427]],
         RMIN=0.0, RBIN=0.5, ETABIN=2.5,
         deficit_on_head=2.991e-03),
    dict(corners=[[0.0, 4.197988359422072], [0.0, 5.197988359422072], [1.0, 4.197988359422072], [1.0, 5.197988359422072]],
         RMIN=0.0, RBIN=0.25, ETABIN=10.0,
         deficit_on_head=1.031e-02),
]


FAILING_CHORD_ARC = [
    dict(corners=[[5.174133054382679, -0.6059031122927486], [5.122683791571482, 0.5024085842133936], [6.107673986306073, -0.5873565164931812], [6.158771740650235, 0.4407693200262107]],
         RMIN=0.13229670373561597, RBIN=0.25, ETABIN=10.0,
         deficit_on_head=7.280e-04),
    dict(corners=[[-13.89505682079805, 0.48555749687299515], [-13.888633763777609, 1.4935100273594712], [-13.023334254840432, 0.5487328309027432], [-12.923430206602276, 1.6620155449485492]],
         RMIN=1.0238108810341309, RBIN=0.25, ETABIN=30.0,
         deficit_on_head=1.490e-03),
    dict(corners=[[-19.70879894919854, -3.1409281748106377], [-19.848393118831428, -2.119275731840155], [-18.719465865141352, -3.2104241957314947], [-18.789542289931997, -2.2338760692910116]],
         RMIN=1.9572827476431724, RBIN=0.25, ETABIN=10.0,
         deficit_on_head=2.687e-03),
    dict(corners=[[3.2607774049404328, 0.2792003372845837], [3.042202653055811, 1.3153163009883735], [4.17492399488934, 0.40804768669698477], [4.111643988332363, 1.376791467130077]],
         RMIN=0.2563437701795268, RBIN=1.0, ETABIN=30.0,
         deficit_on_head=2.487e-03),
    dict(corners=[[3.6569490962243574, -1.2649250036619963], [3.5842251952307187, -0.21611397313025837], [4.576401891289038, -1.214904265249844], [4.75745140279607, -0.24692245472944496]],
         RMIN=0.9745517668650112, RBIN=0.25, ETABIN=30.0,
         deficit_on_head=5.750e-04),
]


@pytest.mark.parametrize("q", FAILING_GENERIC + FAILING_CHORD_ARC)
def test_jittered_quad_with_fine_bins_conserves(q):
    pc = np.array(q["corners"])
    assert abs(_sum_over_bins(pc, q["RMIN"], q["RBIN"], q["ETABIN"]) - _quad_area(pc)) < 1e-9 * _quad_area(pc)


@pytest.mark.parametrize("h,over", [(20.0, 0.002), (20.0, 0.005), (6.0, 0.003), (6.0, 0.015)])
def test_edge_cutting_a_bin_circle_twice_gives_the_circular_cap(h, over):
    """Analytic check: a pixel whose lower edge z = h is a chord of the circle R = h + over.

    The pixel lies above the chord, so pixel intersect disk(R) is exactly the circular
    cap: R^2 acos(h/R) - h sqrt(R^2 - h^2). The parameters keep the half-chord
    sqrt(R^2 - h^2) below the pixel half-width 0.5, so the pixel's side edges never clip
    the cap. The old code integrated the chord as an arc (overcounting) or, with only
    the two crossings as vertices, returned zero.
    """
    R = h + over
    pc = np.array([[cy + dy, cz + dz] for dy, dz in PIXEL_CORNER_OFFSETS
                   for cy, cz in [(0.0, h + 0.5)]])           # y in [-0.5, 0.5], z in [h, h+1]
    cap = R * R * math.acos(h / R) - h * math.sqrt(R * R - h * h)
    wide = 20.0                                              # eta wedge wide enough to hold the whole cap
    area = pixel_bin_intersect(pc, 0.5 * h, R, -wide, wide)
    assert math.sqrt(R * R - h * h) < 0.5
    assert abs(area - cap) < 1e-11, (area, cap)


@pytest.mark.parametrize("q", FAILING_ALIGNED)
def test_pixel_edge_exactly_on_a_bin_ray_conserves(q):
    pc = np.array(q["corners"])
    assert abs(_sum_over_bins(pc, q["RMIN"], q["RBIN"], q["ETABIN"]) - _quad_area(pc)) < 1e-9 * _quad_area(pc)


@pytest.mark.parametrize("radius", [8.5, 14.5, 22.5, 30.5])
@pytest.mark.parametrize("axis", ["eta0_left", "eta0_right", "eta-90"])
def test_pixel_adjacent_to_a_beam_centre_at_a_pixel_edge_conserves(axis, radius):
    """The benchmark geometry: BC at x.5 puts pixel edges exactly on the axes.

    On unmodified code this lost 0.34-0.49 of the pixel on these three axes.
    """
    cy, cz = {"eta0_left": (-0.5, radius), "eta0_right": (0.5, radius), "eta-90": (radius, 0.5)}[axis]
    pc = np.array([[cy + dy, cz + dz] for dy, dz in PIXEL_CORNER_OFFSETS])
    assert abs(_sum_over_bins(pc, 2.0, 1.0, 5.0) - _quad_area(pc)) < 1e-9
