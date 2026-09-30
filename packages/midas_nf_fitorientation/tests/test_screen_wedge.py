"""Issue #11/#17 for the screen kernel: voxels turn with G about the wedge axis.

``screen`` projects the precomputed ``(yl, zl, omega)`` rows through each
voxel's triangle vertices (``DisplacementSpots``). MIDAS Wedge convention
(``midas_diffract.forward`` module doc): the axis is
``n = (-sin W, 0, cos W)`` and the voxel lives in the rotation-stage frame,
at ``S = R_y(-W)`` from the lab at omega = 0, so a vertex ``v = (x, y, 0)``
goes to ``R_n(omega) S v`` -- which has a lab z -- not to the about-z
rotation. The reference builds ``R_n`` and ``S`` by Rodrigues' formula,
independently of the ``R_y R_z`` factorisation in the code.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.io import GridTable, OrientationData
from midas_nf_fitorientation.obs_volume import ObsVolume
from midas_nf_fitorientation.params import FitParams
from midas_nf_fitorientation.screen import _rotate_vertices, _wedge_cos_sin, screen

WEDGE = 3.0          # deg; lab-z of a 600 um off-axis vertex ~ 31 um ~ 3 px
PX = 10.0
NPIX = 256
LSD = 100_000.0
SPOTS = np.array([   # (yl, zl, omega_deg), arbitrary ring positions
    [300.0, 700.0, 30.0],
    [-650.0, 250.0, 100.0],
    [420.0, -560.0, -120.0],
    [-380.0, -610.0, -45.0],
    [720.0, 120.0, 160.0],
], dtype=np.float64)


def _params(wedge):
    p = FitParams()
    p.n_distances = 1
    p.Lsd = [LSD]
    p.ybc = [NPIX / 2.0]
    p.zbc = [NPIX / 2.0]
    p.px = PX
    p.omega_start = -180.0
    p.omega_step_raw = 1.0
    p.start_nr = 1
    p.end_nr = 360
    p.exclude_pole_angle = 0.0
    p.wavelength = 0.172979
    p.lattice_constant = (4.08, 4.08, 4.08, 90, 90, 90)
    p.n_pixels_y = NPIX
    p.n_pixels_z = NPIX
    p.tx = p.ty = p.tz = 0.0
    p.wedge = wedge
    p.min_frac_accept = 0.0
    return p


def _rod(n, a):
    K = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def _rodrigues(W_deg, omega_rad):
    """Stage frame -> lab at ``omega``: ``R_n(omega) @ S``."""
    W = W_deg * math.pi / 180.0
    n = np.array([-math.sin(W), 0.0, math.cos(W)])
    S = _rod(np.array([0.0, 1.0, 0.0]), -W)
    return _rod(n, omega_rad) @ S


def _expected_pixels(verts_xy, wedge_deg):
    """Rigid-body (frame, y_px, z_px) of each SPOTS row for a sub-pixel voxel
    (the screen's single rounded-centroid path)."""
    bc = NPIX / 2.0
    out = []
    for yl, zl, om in SPOTS:
        Rn = _rodrigues(wedge_deg, om * math.pi / 180.0)
        vy, vz = [], []
        for x, y in verts_xy:
            pl = Rn @ np.array([x, y, 0.0])
            t = 1.0 - pl[0] / LSD          # ray to the plane x = Lsd along (Lsd, yl, zl)
            vy.append((pl[1] + yl * t) / PX + bc)
            vz.append((pl[2] + zl * t) / PX + bc)
        cy, cz = yl / PX + bc, zl / PX + bc
        ry = int(np.round(np.mean(np.array(vy) - cy)))
        rz = int(np.round(np.mean(np.array(vz) - cz)))
        frame = int(om - (-180.0))
        out.append((frame, math.floor(cy) + ry, math.floor(cz) + rz))
    return out


def _od():
    n = SPOTS.shape[0]
    return OrientationData(
        matrices=np.eye(3)[None, ...].copy(),
        n_spots=np.array([n], dtype=np.int64),
        starts=np.array([0], dtype=np.int64),
        spots=SPOTS.copy(),
    )


def _obs_at(pixels):
    arr = np.zeros((1, 360, NPIX, NPIX), dtype=np.float32)
    for f, y, z in pixels:
        arr[0, f, y, z] = 1.0
    return ObsVolume.from_dense_array(arr)


# Small voxel far off-axis: 2*gs <= px -> single-centroid path.
_SMALL = dict(xs=600.0, ys=-350.0, gs=4.0, y1=2.0, y2=4.0)
# Super-pixel voxel: forces the mixed-gs per-voxel fallback when paired.
_BIG = dict(xs=-200.0, ys=100.0, gs=40.0, y1=20.0, y2=40.0)


def _grid(*vox):
    return GridTable(
        y1=np.array([v["y1"] for v in vox]), y2=np.array([v["y2"] for v in vox]),
        xs=np.array([v["xs"] for v in vox]), ys=np.array([v["ys"] for v in vox]),
        gs=np.array([v["gs"] for v in vox]),
        ud=np.array([1] * len(vox), dtype=np.int8),
    )


def _frac(result, voxel):
    ws = [w for w in result.winners if w.voxel_idx == voxel and w.orient_idx == 0]
    assert len(ws) == 1
    return ws[0].frac_overlap


def test_rotate_vertices_w0_is_the_about_z_formula():
    XG = torch.tensor([[600.0, -500.0, 12.5]], dtype=torch.float64)
    YG = torch.tensor([[-350.0, 40.0, 700.0]], dtype=torch.float64)
    om = torch.tensor([[0.3], [2.1], [-1.4]], dtype=torch.float64)
    c, s = torch.cos(om), torch.sin(om)
    assert _wedge_cos_sin(_params(0.0)) is None
    xa, ya, za = _rotate_vertices(XG, YG, c, s, None)
    assert za is None
    assert torch.equal(xa, XG * c - YG * s)
    assert torch.equal(ya, XG * s + YG * c)


def test_rotate_vertices_matches_rodrigues():
    cs = _wedge_cos_sin(_params(WEDGE))
    XG = torch.tensor([[600.0, -500.0, 12.5]], dtype=torch.float64)
    YG = torch.tensor([[-350.0, 40.0, 700.0]], dtype=torch.float64)
    oms = [0.3, 2.1, -1.4]
    om = torch.tensor([[o] for o in oms], dtype=torch.float64)
    xa, ya, za = _rotate_vertices(XG, YG, torch.cos(om), torch.sin(om), cs)
    for i, o in enumerate(oms):
        Rn = _rodrigues(WEDGE, o)
        for j in range(3):
            ref = Rn @ np.array([float(XG[0, j]), float(YG[0, j]), 0.0])
            got = np.array([float(xa[i, j]), float(ya[i, j]), float(za[i, j])])
            assert np.max(np.abs(got - ref)) < 1e-9


def _screen_small_voxel_frac(wedge_in_params, *, mixed):
    verts = _grid(_SMALL).triangle_vertices(0)
    pix = _expected_pixels(verts, WEDGE)
    grid = _grid(_SMALL, _BIG) if mixed else _grid(_SMALL)
    res = screen(grid, _od(), _obs_at(pix), _params(wedge_in_params),
                 dtype=torch.float64)
    return _frac(res, 0), pix, verts


@pytest.mark.parametrize("mixed", [False, True],
                         ids=["vectorised", "per_voxel_fallback"])
def test_screen_hits_rigid_body_pixels_with_wedge(mixed):
    frac, _, _ = _screen_small_voxel_frac(WEDGE, mixed=mixed)
    assert frac == pytest.approx(1.0)


@pytest.mark.parametrize("mixed", [False, True],
                         ids=["vectorised", "per_voxel_fallback"])
def test_screen_about_z_rotation_misses_them(mixed):
    """Null control: screening the same obs with the wedge dropped from the
    vertex rotation (the pre-fix behaviour) must miss spots."""
    frac_old, pix, verts = _screen_small_voxel_frac(0.0, mixed=mixed)
    # Sanity: the about-z pixels really differ for this fixture.
    assert _expected_pixels(verts, 0.0) != pix
    assert frac_old < 1.0
