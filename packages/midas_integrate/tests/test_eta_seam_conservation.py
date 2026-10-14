"""Map-build area conservation across the eta = +-180 seam.

``test_area_conservation`` checks ``pixel_bin_intersect`` for ONE pixel against
every bin it is told about. It cannot see which bins the map builders choose to
ask about, and that is where the seam bug lived.

A pixel that crosses eta = +-180 has corners on both sides, so min/max of its
corner etas spans > 180 deg. The builders then unwrapped that to
[max corner, 360 + min corner], which is the INNER part of the pixel's true
extent. A bin that only the outermost corner reaches was never visited, and the
area the pixel has in it was silently dropped: 0.9 % of one pixel on a flat
detector, up to 31 % of one at R ~ 4 px with 10 deg bins under tilt. The area is
an identity -- the (R, eta) bins tile the plane, so a pixel's mapped areas must
sum to its own mapped-quad area -- so this needs no reference implementation.

A beam centre sitting exactly on a pixel edge is covered as well: it puts pixel edges
exactly on bin rays, where the earlier vertex enumeration in the intersection lost
about half of every pixel along the axes (see test_intersection_vertex_count).

Checked on all three builders: the plain numba kernel, the sub-pixel numba
kernel (forced here by passing zero distortion arrays, the way a real panel or
distortion correction does), and the pure-Python fallback.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_integrate import detector_mapper as dm
from midas_integrate.geometry import QUAD_ORDER, build_tilt_matrix
from midas_integrate.params import IntegrationParams

NY = NZ = 40
PX = 172.0
RMIN, RMAX, RBIN, ETABIN = 2.0, 26.0, 1.0, 10.0


def _params(**over):
    kw = dict(NrPixelsY=NY, NrPixelsZ=NZ, pxY=PX, pxZ=PX, BC_y=19.37, BC_z=21.11,
              RhoD=NY * PX, RMin=RMIN, RMax=RMAX, RBinSize=RBIN,
              EtaMin=-180.0, EtaMax=180.0, EtaBinSize=ETABIN)
    kw.update(over)
    return IntegrationParams(**kw)


def _mapped_quads(p):
    TRs = build_tilt_matrix(p.tx, p.ty, p.tz)
    pp = tuple(getattr(p, f"p{i}") for i in range(15))
    R, E, _, _ = dm._per_pixel_corners_REta(
        NrPixelsY=NY, NrPixelsZ=NZ, Ycen=p.BC_y, Zcen=p.BC_z, Lsd=p.Lsd,
        RhoD=p.RhoD, px=PX, TRs=TRs, p=pp, parallax=p.Parallax)
    return dm._build_pixel_quads(R, E), R


def _quad_area(q):
    idx = list(QUAD_ORDER)
    y, z = q[idx, 0], q[idx, 1]
    return 0.5 * abs(float((y * np.roll(z, -1) - np.roll(y, -1) * z).sum()))


def _per_pixel_sums(pm):
    px = pm.pxList
    bins_n = len(pm.counts)
    del bins_n
    # Sub-pixel entries carry the sub-pixel CENTRE (y + sp_cy), so round.
    key = np.rint(px["z"]).astype(np.int64) * NY + np.rint(px["y"]).astype(np.int64)
    return np.bincount(key, weights=px["frac"], minlength=NY * NZ).reshape(NZ, NY)


CASES = [
    ("flat", dict(Lsd=300_000.0)),
    ("tilted", dict(Lsd=50_000.0, tx=5.0, ty=20.0, tz=-8.0)),
    # Beam centre exactly on a pixel edge, as in every synthetic benchmark geometry:
    # pixel edges then lie exactly on the eta = 0 / +-90 / +-180 rays.
    ("bc_on_pixel_edge", dict(Lsd=300_000.0, BC_y=19.5, BC_z=21.5)),
    # Beam centre on a pixel CENTRE, first radial boundary 0.004 px beyond a pixel edge:
    # on every pixel along the cardinal axes the edge's tangent foot is closer to the
    # origin than either corner, by L^2/(8R) ~ 0.0125 px at R = 10, so the bin below the
    # boundary overlaps the pixel in a sliver that a corners-only radial range never visits.
    ("sagitta_window", dict(Lsd=300_000.0, BC_y=19.0, BC_z=21.5, RMin=2.004)),
]
# Conservation tolerance per case (relative to the pixel's mapped-quad area).
TOL = {"flat": 1e-9, "tilted": 1e-9, "bc_on_pixel_edge": 1e-9, "sagitta_window": 1e-9}
PATHS = ["fast", "subpixel", "python"]


@pytest.mark.parametrize("name,over", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("path", PATHS)
def test_every_interior_pixel_conserves_area_including_the_seam(name, over, path):
    p = _params(**over)
    kw = dict(auto_load=False, verbose=False, use_numba=(path != "python"))
    if path == "subpixel":
        kw["distortion_y"] = np.zeros((NZ, NY))
        kw["distortion_z"] = np.zeros((NZ, NY))
    sums = _per_pixel_sums(dm.build_map(p, **kw).as_pixel_map())
    quads, R = _mapped_quads(p)

    worst, n, n_seam = 0.0, 0, 0
    for j in range(NZ):
        for i in range(NY):
            if R[j, i].min() <= RMIN + 1.0 or R[j, i].max() >= RMAX - 1.0:
                continue
            A = _quad_area(quads[j, i])
            n += 1
            eta = np.degrees(np.arctan2(-quads[j, i, :, 0], quads[j, i, :, 1]))
            if eta.max() - eta.min() > 180.0:
                n_seam += 1
            worst = max(worst, abs(sums[j, i] - A) / A)
    assert n > 500, "not enough interior pixels exercised"
    assert n_seam >= 5, "the test geometry must include seam-crossing pixels"
    assert worst < TOL[name], (
        f"{path}/{name}: worst per-pixel area non-conservation {worst:.3e} "
        f"over {n} interior pixels ({n_seam} cross the eta seam)")
