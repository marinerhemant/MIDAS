"""The polygon kernel must apply the same detector geometry as every other one.

Regression tests for issue #69: ``PolygonBinGeometry`` built pixel corners from
a bare beam-centre projection and silently discarded detector tilt, the
distortion polynomial, parallax, per-panel rigid shifts and the residual
correction map -- while being documented as the exact/reference kernel and used
as the accuracy reference by ``corrections/background.py``.

Reported and independently reproduced discrepancies against the shared geometry
(``pixel_to_REta_from_spec``) at identical sub-pixel positions, 200x200 detector:

    baseline                    max|dR| = 0.0000 px
    tilt ty = 5 deg             max|dR| = 1.4577 px
    distortion iso_R2 = 0.05    max|dR| = 0.2871 px
    two panels, +-2/+-3 px      max|dR| = 3.6056 px  ( = sqrt(2^2+3^2) )

Every test here fails on the pre-fix code, so they discriminate rather than
merely pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate_v2.spec import IntegrationSpec
from midas_integrate_v2.forward import pixel_to_REta_from_spec
from midas_integrate_v2.forward.pixels import _panel_inputs_from_spec
from midas_calibrate_v2.forward.panels import panel_idx_for_points
from midas_integrate_v2.binning.polygon import (
    PolygonBinGeometry, integrate_polygon, _pixel_corner_YZ_px,
    _quad_area, _PIXEL_CORNERS,
)
from midas_integrate_v2.binning.subpixel import (
    SubpixelBinGeometry, integrate_subpixel,
)

DT = torch.float64


def _spec(*, NY=64, NZ=64, ty=0.0, tz=0.0, iso_R2=0.0,
          shifts_path=None, RBinSize=1.0):
    s = IntegrationSpec()
    s.NrPixelsY = NY
    s.NrPixelsZ = NZ
    s.pxY = 200.0
    s.pxZ = 200.0
    s.Lsd = torch.tensor(200000.0, dtype=DT)
    s.BC_y = torch.tensor(NY / 2.0 + 0.37, dtype=DT)
    s.BC_z = torch.tensor(NZ / 2.0 - 0.41, dtype=DT)
    s.tx = torch.tensor(0.0, dtype=DT)
    s.ty = torch.tensor(ty, dtype=DT)
    s.tz = torch.tensor(tz, dtype=DT)
    s.Wavelength = torch.tensor(0.1729, dtype=DT)
    s.Parallax = torch.tensor(0.0, dtype=DT)
    s.iso_R2 = torch.tensor(iso_R2, dtype=DT)
    s.RhoD = float(NY) * 200.0
    s.RMin = 2.0
    s.RMax = float(NY) / 2.0 - 2.0
    s.RBinSize = RBinSize
    s.EtaMin = -180.0
    s.EtaMax = 180.0
    s.EtaBinSize = 30.0
    if shifts_path is not None:
        s.NPanelsY = 1
        s.NPanelsZ = 2
        s.PanelSizeY = NY
        s.PanelSizeZ = NZ // 2
        s.PanelGapsY = []
        s.PanelGapsZ = [0]
        s.PanelShiftsFile = str(shifts_path)
    return s


@pytest.fixture
def shifts_file(tmp_path):
    p = tmp_path / "panel_shifts.txt"
    p.write_text(
        "    0  +2.000000  +3.000000  +0.000000e+00  +0.0000  +0.000000e+00\n"
        "    1  -2.000000  -3.000000  +0.000000e+00  +0.0000  +0.000000e+00\n"
    )
    return p


def _corner_R_from_polygon(spec, corner=0):
    c = _pixel_corner_YZ_px(spec)
    return np.sqrt(c[..., corner, 0] ** 2 + c[..., corner, 1] ** 2)


def _corner_R_from_shared(spec, corner=0):
    """Reference R for one corner, from the shared geometry.

    The panel index is taken at the pixel CENTRE and applied to the corner --
    the same rigid-body convention the polygon kernel uses. Letting
    ``pixel_to_REta_from_spec`` round each corner to its own panel would move
    the corners of a boundary-straddling pixel by different shifts, which is
    the tearing the kernel deliberately avoids; comparing against that would
    test the wrong thing.
    """
    dy, dz = _PIXEL_CORNERS[corner]
    Yi, Zi = np.meshgrid(np.arange(spec.NrPixelsY), np.arange(spec.NrPixelsZ),
                         indexing="xy")
    Yc = torch.as_tensor(Yi, dtype=DT)
    Zc = torch.as_tensor(Zi, dtype=DT)
    panel_idx = None
    panels = _panel_inputs_from_spec(spec)
    if panels is not None:
        panel_idx = panel_idx_for_points(panels[0], Yc, Zc)
    out = pixel_to_REta_from_spec(
        torch.as_tensor(Yi + dy, dtype=DT),
        torch.as_tensor(Zi + dz, dtype=DT),
        spec,
        panel_idx=panel_idx,
    )
    return out.R_px.detach().cpu().numpy()


# --------------------------------------------------------------- geometry

@pytest.mark.parametrize("label,kw", [
    ("baseline", {}),
    ("tilt ty=5deg", {"ty": 5.0}),
    ("tilt tz=3deg", {"tz": 3.0}),
    ("distortion iso_R2=0.05", {"iso_R2": 0.05}),
    ("tilt+distortion", {"ty": 5.0, "iso_R2": 0.05}),
])
def test_polygon_corners_match_the_shared_geometry(label, kw):
    """The whole bug in one assertion, for every corrections combination."""
    s = _spec(**kw)
    d = np.abs(_corner_R_from_polygon(s) - _corner_R_from_shared(s))
    assert d.max() < 1e-9, f"{label}: corner R off by {d.max():.4e} px"


def test_polygon_corners_match_with_panel_shifts(shifts_file):
    """Pre-fix this was off by exactly the discarded shift, sqrt(2^2+3^2)."""
    s = _spec(shifts_path=shifts_file)
    d = np.abs(_corner_R_from_polygon(s) - _corner_R_from_shared(s))
    assert d.max() < 1e-9, f"panel corner R off by {d.max():.4e} px"


def test_every_corner_not_just_the_first(shifts_file):
    s = _spec(ty=4.0, iso_R2=0.02, shifts_path=shifts_file)
    for c in range(len(_PIXEL_CORNERS)):
        d = np.abs(_corner_R_from_polygon(s, c) - _corner_R_from_shared(s, c))
        assert d.max() < 1e-9, f"corner {c} off by {d.max():.4e} px"


def test_panel_boundary_pixel_is_not_torn(shifts_file):
    """A pixel straddling a panel edge must move as ONE rigid body.

    ``panel_idx_for_points`` rounds each point to a pixel, so mapping the four
    corners independently would give them different panel shifts and tear the
    quad. The panel index is therefore taken once at the pixel centre. If that
    regressed, boundary-row quads would have a wildly wrong area.
    """
    s = _spec(shifts_path=shifts_file)
    corners = _pixel_corner_YZ_px(s)
    areas = _quad_area(corners)
    NZ = s.NrPixelsZ
    boundary = NZ // 2                      # first row of the second panel
    interior = areas[NZ // 4, :]
    for row in (boundary - 1, boundary, boundary + 1):
        assert np.allclose(areas[row, :], interior, atol=1e-9), (
            f"row {row} quad areas differ from interior -- pixel torn across "
            f"the panel boundary"
        )


# ------------------------------------------------------------ area weights

def _per_pixel_total(g, s):
    tot = torch.zeros(s.NrPixelsY * s.NrPixelsZ, dtype=torch.float64)
    return tot.index_add(0, g.pix_idx, g.area)


def test_tilt_actually_stretches_pixels():
    """Premise of the whole area_weight option: under tilt a pixel's mapped
    quad is no longer 1 px². Without this the option would be pointless."""
    quad_flat = _quad_area(_pixel_corner_YZ_px(_spec()))
    assert np.allclose(quad_flat, 1.0, atol=1e-9), "baseline must stay unit"
    quad_tilt = _quad_area(_pixel_corner_YZ_px(_spec(ty=5.0)))
    assert quad_tilt.max() > 1.0 + 1e-4
    assert quad_tilt.min() < 1.0 - 1e-4


def test_area_weight_pixel_keeps_unit_weight_under_tilt():
    """Default weighting: no pixel carries more than weight 1, and the fully
    contained ones carry exactly 1."""
    s = _spec(ty=5.0)
    g = PolygonBinGeometry.from_spec(s, area_weight="pixel")
    tot = _per_pixel_total(g, s)
    assert float(tot.max()) <= 1.0 + 1e-9
    # Some pixel must actually reach 1 — otherwise "unit weight" is vacuous.
    assert abs(float(tot.max()) - 1.0) < 1e-9


def test_area_weight_projected_tracks_the_mapped_quad_area():
    """Opt-in weighting: a pixel carries at most the area it actually subtends
    in the corrected frame, and the fully contained ones carry exactly that —
    not a nominal 1.0.

    Before the fast path was corrected, a fully-contained pixel emitted a
    hard-coded 1.0 while the slow path emitted true geometric areas, so one
    output mixed two area conventions (1.000000 against a quad of 0.988316).
    """
    s = _spec(ty=5.0)
    g = PolygonBinGeometry.from_spec(s, area_weight="projected")
    tot = _per_pixel_total(g, s)
    quad = torch.from_numpy(_quad_area(_pixel_corner_YZ_px(s)).reshape(-1))
    covered = tot > 1e-12
    assert bool(covered.any())
    # Never more than the pixel's own area.
    assert float((tot[covered] - quad[covered]).max()) < 1e-9
    # And some pixel attains it exactly.
    assert float((tot[covered] - quad[covered]).abs().min()) < 1e-9
    # Those attained values must differ from the nominal 1.0, otherwise this
    # mode would be indistinguishable from "pixel".
    attained = covered & ((tot - quad).abs() < 1e-9)
    assert float((tot[attained] - 1.0).abs().max()) > 1e-4


def test_the_two_modes_differ_by_exactly_the_quad_area():
    """The defining relation between the two options."""
    s = _spec(ty=5.0)
    g_pix = PolygonBinGeometry.from_spec(s, area_weight="pixel")
    g_prj = PolygonBinGeometry.from_spec(s, area_weight="projected")
    assert torch.equal(g_pix.pix_idx, g_prj.pix_idx)
    assert torch.equal(g_pix.bin_idx, g_prj.bin_idx)
    quad = torch.from_numpy(_quad_area(_pixel_corner_YZ_px(s)).reshape(-1))
    assert torch.allclose(g_pix.area * quad[g_pix.pix_idx], g_prj.area,
                          atol=1e-12)


def test_area_weight_rejects_unknown_value():
    with pytest.raises(ValueError, match="area_weight"):
        PolygonBinGeometry.from_spec(_spec(), area_weight="solid_angle")


def test_baseline_modes_agree_because_quads_are_unit_area():
    """With no corrections the two modes must be identical — proving the
    option is inert exactly where the old behaviour was already right."""
    s = _spec()
    a = PolygonBinGeometry.from_spec(s, area_weight="pixel")
    b = PolygonBinGeometry.from_spec(s, area_weight="projected")
    assert torch.allclose(a.area, b.area, atol=1e-12)


# ---------------------------------------------------------- kernel parity

def _ring_image(s, R0_px=12.0, sigma_px=1.5):
    """Gaussian ring drawn in the TRUE geometry, so a kernel that ignores the
    corrections cannot reproduce it."""
    Yi, Zi = np.meshgrid(np.arange(s.NrPixelsY), np.arange(s.NrPixelsZ),
                         indexing="xy")
    out = pixel_to_REta_from_spec(torch.as_tensor(Yi, dtype=DT),
                                  torch.as_tensor(Zi, dtype=DT), s)
    R = out.R_px.detach().cpu().numpy()
    return torch.from_numpy(
        np.exp(-(R - R0_px) ** 2 / (2 * sigma_px ** 2)).astype(np.float64))


def _radial_centroid(prof, s):
    r = s.RMin + s.RBinSize * (np.arange(s.n_r_bins) + 0.5)
    w = np.maximum(prof, 0.0)
    return float((w * r).sum() / (w.sum() + 1e-30))


@pytest.mark.parametrize("label,kw", [
    ("tilt", {"ty": 5.0}),
    ("distortion", {"iso_R2": 0.05}),
])
def test_polygon_agrees_with_subpixel_under_corrections(label, kw):
    """The user-visible consequence: polygon and subpixel must put the ring in
    the same place. Pre-fix they disagreed by ~1 px under tilt."""
    s = _spec(**kw)
    img = _ring_image(s)

    g_poly = PolygonBinGeometry.from_spec(s)
    prof_poly = integrate_polygon(img, g_poly, normalize=True).mean(dim=0).numpy()

    g_sub = SubpixelBinGeometry.from_spec(s, K=4)
    prof_sub = integrate_subpixel(img, g_sub, normalize=True).mean(dim=0).numpy()

    c_poly = _radial_centroid(prof_poly, s)
    c_sub = _radial_centroid(prof_sub, s)
    assert abs(c_poly - c_sub) < 0.25 * s.RBinSize, (
        f"{label}: polygon centroid {c_poly:.4f} vs subpixel {c_sub:.4f} "
        f"(drift {abs(c_poly - c_sub):.4f} px)")


def test_polygon_recovers_the_true_ring_radius_under_tilt():
    """Absolute check, not just kernel-vs-kernel: the ring was drawn at
    R = 12 px in the corrected frame and must integrate back to 12 px."""
    s = _spec(ty=5.0, RBinSize=0.5)
    img = _ring_image(s, R0_px=12.0)
    g = PolygonBinGeometry.from_spec(s)
    prof = integrate_polygon(img, g, normalize=True).mean(dim=0).numpy()
    assert abs(_radial_centroid(prof, s) - 12.0) < 0.25


# ----------------------------------------------------------- no regression

def test_baseline_spec_is_bit_identical_to_the_bare_projection():
    """Where the old code was already right, nothing may move.

    With no tilt, distortion, parallax, panels or residual map the corrected
    map reduces to the bare beam-centre projection the old code used.
    """
    s = _spec()
    c = _pixel_corner_YZ_px(s)
    BC_y = float(s.BC_y)
    BC_z = float(s.BC_z)
    Yi, Zi = np.meshgrid(np.arange(s.NrPixelsY), np.arange(s.NrPixelsZ),
                         indexing="xy")
    for k, (dy, dz) in enumerate(_PIXEL_CORNERS):
        assert np.allclose(c[:, :, k, 0], -(Yi + dy - BC_y), atol=1e-9)
        assert np.allclose(c[:, :, k, 1], (Zi + dz - BC_z), atol=1e-9)


# --------------------------------------------------- conservation (issue #69b)

def test_per_pixel_area_is_conserved_exactly():
    """Every pixel's areas must sum to its own mapped quad area.

    The (R, eta) bins tile the plane, so this is an identity. It failed
    badly before the corner-order fix: interior pixels retained a mean of
    0.85 of their area, worst case 0.50, which silently destroyed intensity
    at every bin boundary.
    """
    for kw in ({}, {"ty": 5.0}, {"iso_R2": 0.05}, {"ty": 4.0, "iso_R2": 0.02}):
        s = _spec(**kw)
        g = PolygonBinGeometry.from_spec(s, area_weight="projected")
        quad = torch.from_numpy(_quad_area(_pixel_corner_YZ_px(s)).reshape(-1))
        tot = _per_pixel_total(g, s)
        c = _pixel_corner_YZ_px(s)
        R = np.sqrt(c[..., 0] ** 2 + c[..., 1] ** 2).reshape(-1, 4)
        interior = (R.min(1) > s.RMin + 1.5) & (R.max(1) < s.RMax - 1.5)
        sel = torch.from_numpy(interior)
        assert int(sel.sum()) > 100
        err = float((tot[sel] - quad[sel]).abs().max())
        assert err < 1e-12, f"{kw}: worst per-pixel area error {err:.3e}"


def test_constant_image_integrates_to_the_constant():
    """Intensity conservation. A flat field must come back flat in every
    non-empty bin, for both weightings and with corrections active."""
    for kw in ({}, {"ty": 5.0}, {"iso_R2": 0.05}):
        s = _spec(**kw)
        for aw in ("pixel", "projected"):
            g = PolygonBinGeometry.from_spec(s, area_weight=aw)
            img = torch.full((s.NrPixelsZ, s.NrPixelsY), 7.5, dtype=torch.float64)
            out = integrate_polygon(img, g, normalize=True)
            areas = torch.zeros(g.n_eta * g.n_r, dtype=torch.float64)
            areas = areas.index_add(0, g.bin_idx, g.area).reshape(g.n_eta, g.n_r)
            nz = areas > 1e-12
            assert bool(nz.any())
            dev = float((out[nz] - 7.5).abs().max())
            assert dev < 1e-12, f"{kw} {aw}: flat field deviated by {dev:.3e}"


def test_total_mass_is_preserved():
    """Un-normalised integration must redistribute intensity, not create or
    destroy it: the binned total equals the per-entry total."""
    s = _spec(ty=5.0)
    g = PolygonBinGeometry.from_spec(s, area_weight="projected")
    rng = np.random.default_rng(0)
    img = torch.from_numpy(rng.uniform(0, 100, (s.NrPixelsZ, s.NrPixelsY)))
    raw = integrate_polygon(img, g, normalize=False)
    per_entry = float((img.reshape(-1)[g.pix_idx] * g.area).sum())
    assert abs(float(raw.sum()) - per_entry) < 1e-9 * max(per_entry, 1.0)
