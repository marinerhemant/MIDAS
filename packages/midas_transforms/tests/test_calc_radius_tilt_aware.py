"""calc_radius assigns spots to rings with the tilt-corrected radius.

Before this, ``calc_radius`` compared the FLAT distance of a spot from the beam
centre (what the peak fitter reports as ``Radius``) with the ``hkls.csv`` ring
radius, inside ``Width``. On a tilted detector the flat distance of a point on a
ring is off by about ``|tilt| * R**2 / Lsd``, so real, correctly thresholded
spots on the outer rings fell outside ``Width`` and were dropped with no message
(the peak-search band, the eta-coverage table and ``global_powder`` all use the
tilt-corrected radius with the same ``Width``, so the stages disagreed).

Geometry below is the reported case: one 1028 x 512 Eiger panel, beam centre
off the frame, ty = 0.397 deg, tz = -1.216 deg, Lsd = 136.7 mm, Width = 600 um.
On-ring spots are placed with the forward kernel ``fit_setup`` uses
(``pixel_to_REta_torch``), by bisection along each flat azimuth, so every spot
is exactly on its ring by construction and only the matching rule is under test.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_calibrate.geometry_torch import pixel_to_REta_torch
from midas_transforms.fit_setup import fit_setup
from midas_transforms.params import ZarrParams
from midas_transforms.radius import calc_radius
from midas_transforms.radius import core as radius_core
from midas_transforms.radius import match

LSD = 136744.701
PX = 75.0
NY, NZ = 1028, 512
BCY, BCZ = -268.5130, 385.7461
# Ring number -> radius on the plane at Lsd (um), as in the report.
RINGS = {2: 23321.0, 3: 28719.0, 4: 33344.0, 6: 41296.0, 8: 51447.0,
         10: 57543.0, 13: 66086.0, 15: 74172.0, 19: 84549.0, 23: 97236.0}
OUTER = (13, 15, 19, 23)
REPORT_TILT = (0.397495, -1.216155)
SIBLING_TILT = (0.275, 0.253)


def _T(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float64)


def _zarr_params(ty: float, tz: float, width: float = 600.0, **kw) -> ZarrParams:
    zp = ZarrParams(
        Lsd=LSD, PixelSize=PX, YCen=BCY, ZCen=BCZ, tx=0.0, ty=ty, tz=tz,
        RhoD=3000.0 * PX, NrPixelsY=NY, NrPixelsZ=NZ, NrPixels=NY, EndNr=592,
        OmegaStep=0.25, Width=width, Hbeam=1.0, Rsample=1.0, Wavelength=0.2,
        RingThresh=[(r, 10.0) for r in RINGS],
    )
    for k, v in kw.items():
        setattr(zp, k, v)
    return zp


def _corrected_radius_px(Y, Z, ty, tz):
    R, _ = pixel_to_REta_torch(
        Y, Z, Lsd=_T(LSD), BC_y=_T(BCY), BC_z=_T(BCZ), tx=_T(0.0),
        ty=_T(ty), tz=_T(tz), p_coeffs=torch.zeros(15, dtype=torch.float64),
        parallax=_T(0.0), px=_T(PX), rho_d=_T(3000.0))
    return R


def _ring_pixels(R_um: float, ty: float, tz: float, n_phi: int = 720):
    """Pixels (Y, Z) inside the frame whose corrected radius is ``R_um``."""
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, dtype=torch.float64)[:-1]
    lo = torch.full_like(phi, 0.5 * R_um / PX)
    hi = torch.full_like(phi, 1.5 * R_um / PX)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        Y, Z = BCY + mid * torch.sin(phi), BCZ + mid * torch.cos(phi)
        too_far = _corrected_radius_px(Y, Z, ty, tz) > R_um / PX
        hi = torch.where(too_far, mid, hi)
        lo = torch.where(too_far, lo, mid)
    rf = 0.5 * (lo + hi)
    Y, Z = BCY + rf * torch.sin(phi), BCZ + rf * torch.cos(phi)
    keep = (Y >= 0) & (Y < NY) & (Z >= 0) & (Z < NZ)
    return Y[keep].numpy(), Z[keep].numpy()


def _result_array(per_ring_yz) -> tuple[np.ndarray, np.ndarray]:
    """18-column Result array; Radius/Eta are what the peak fitter reports
    (plain distance from the beam centre). Returns (array, true ring per row)."""
    rows, ring_of = [], []
    sid = 1
    for ring, (Y, Z) in per_ring_yz.items():
        for y, z in zip(Y, Z):
            dY, dZ = y - BCY, z - BCZ
            r_flat = math.hypot(dY, dZ)
            eta = math.degrees(math.atan2(dY, dZ))
            rows.append([sid, 1000.0, 0.0, y, z, 100.0, 0.0, 0.0, 1.0, 1.0,
                         10.0, 10.0, r_flat, eta, 1000.0, 0.0, 0.1, 0.0])
            ring_of.append(ring)
            sid += 1
    return np.array(rows, dtype=np.float64), np.array(ring_of)


def _hkls(path: Path) -> Path:
    with open(path, "w") as f:
        f.write("h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\n")
        for rn, R in RINGS.items():
            f.write(f"1 0 0 1.0 {rn} 0 0 0 0 0 {R}\n")
    return path


@pytest.fixture(scope="module")
def report_spots():
    per_ring = {rn: _ring_pixels(R, *REPORT_TILT) for rn, R in RINGS.items()}
    return _result_array(per_ring)


def _run(arr, zp, tmp_path):
    return calc_radius(
        tmp_path, zarr_params=zp, result_array=arr,
        hkls_path=_hkls(tmp_path / "hkls.csv"), write=False,
        device="cpu", dtype="float64")


def _kept_per_ring(res):
    ring = res.spots[:, 13].numpy().astype(int)
    return {rn: int((ring == rn).sum()) for rn in RINGS}


# --------------------------------------------------------------------------
# The reported failure, and the controls that make it a test
# --------------------------------------------------------------------------


def test_every_on_ring_spot_is_kept_on_the_tilted_panel(report_spots, tmp_path):
    arr, ring_of = report_spots
    res = _run(arr, _zarr_params(*REPORT_TILT, width=600.0), tmp_path)
    n_in = {rn: int((ring_of == rn).sum()) for rn in RINGS}
    assert _kept_per_ring(res) == n_in
    # ...and each is assigned to the ring it lies on, not a neighbour.
    out_ring = res.spots[:, 13].numpy().astype(int)
    out_orig = res.spots[:, 24].numpy().astype(int)
    assert len(out_orig) == len(arr), "a spot matched two rings or none"
    np.testing.assert_array_equal(out_ring, ring_of[out_orig - 1])


def test_the_geometry_really_defeats_a_flat_radius_match(report_spots, tmp_path):
    """Negative control: the same spots through the flat-radius rule (no
    ``R_match_um``) lose rings 13/15/19/23 entirely and keep the inner rings.
    Without this the test above could pass for any geometry."""
    arr, ring_of = report_spots
    t = torch.from_numpy(arr)
    ring_numbers = torch.tensor(list(RINGS), dtype=torch.int64)
    radii = torch.tensor(list(RINGS.values()), dtype=torch.float64)
    out, _, _ = radius_core._filter_and_compute_radius(
        t, ring_numbers, radii, width_px=600.0, px_um=PX, Lsd_um=LSD,
        OmegaStep=0.25, Hbeam=1.0, Rsample=1.0, Vsample=0.0, DiscModel=0,
        DiscArea=0.0, n_frames=592)
    ring = out[:, 13].numpy().astype(int)
    for rn in OUTER:
        assert (ring == rn).sum() == 0
    for rn in (2, 3, 4, 6, 8, 10):
        assert (ring == rn).sum() == (ring_of == rn).sum()


def test_zero_tilt_is_unchanged(tmp_path):
    per_ring = {rn: _ring_pixels(R, 0.0, 0.0) for rn, R in RINGS.items()}
    arr, ring_of = _result_array(per_ring)
    zp = _zarr_params(0.0, 0.0, width=600.0)
    R_flat = torch.from_numpy(arr[:, 12]) * PX
    R_corr = match.corrected_radius_um(zp, torch.from_numpy(arr[:, 3]),
                                       torch.from_numpy(arr[:, 4]))
    assert torch.allclose(R_corr, R_flat, atol=1e-6, rtol=0.0)
    res = _run(arr, zp, tmp_path)
    assert _kept_per_ring(res) == {rn: int((ring_of == rn).sum()) for rn in RINGS}
    assert res.match_stats["n_recovered_vs_flat"] == 0
    assert res.match_stats["n_rejected_vs_flat"] == 0


def test_smaller_tilt_panel_keeps_everything(tmp_path):
    per_ring = {rn: _ring_pixels(R, *SIBLING_TILT) for rn, R in RINGS.items()}
    arr, ring_of = _result_array(per_ring)
    res = _run(arr, _zarr_params(*SIBLING_TILT, width=600.0), tmp_path)
    assert _kept_per_ring(res) == {rn: int((ring_of == rn).sum()) for rn in RINGS}


def test_spots_off_the_ring_are_still_rejected(tmp_path):
    """The correction must not widen the window: spots whose CORRECTED radius
    is 3 Widths off the ring match nothing, on the same tilted panel."""
    width = 600.0
    per_ring = {}
    for rn in (10, 13):
        for sign in (+1, -1):
            Y, Z = _ring_pixels(RINGS[rn] + sign * 3 * width, *REPORT_TILT)
            per_ring.setdefault(rn, ([], []))[0].append(Y)
            per_ring[rn][1].append(Z)
    per_ring = {rn: (np.concatenate(v[0]), np.concatenate(v[1]))
                for rn, v in per_ring.items()}
    arr, _ = _result_array(per_ring)
    assert len(arr) > 0
    res = _run(arr, _zarr_params(*REPORT_TILT, width=width), tmp_path)
    assert res.spots.shape[0] == 0
    assert res.match_stats["n_unmatched"] == len(arr)


# --------------------------------------------------------------------------
# The stages agree on what "radius" means
# --------------------------------------------------------------------------


def test_matching_radius_is_the_one_fit_setup_corrects_to(report_spots, tmp_path):
    """Drift guard: the radius used for matching equals the radius of the lab
    position ``fit_setup`` writes for the same spots."""
    arr, _ = report_spots
    zp = _zarr_params(*REPORT_TILT, width=1500.0)
    res = _run(arr, zp, tmp_path)
    fs = fit_setup(
        result_folder=tmp_path, zarr_params=zp,
        radius_array=res.spots.numpy(), hkls_path=tmp_path / "hkls.csv",
        write=False, do_fit=False, device="cpu", dtype="float64")
    ia = fs.spots_inputall.numpy()
    R_fs = np.hypot(ia[:, 0], ia[:, 1])
    live = R_fs > 0
    R_match = match.corrected_radius_um(
        zp, res.spots[:, 3].double(), res.spots[:, 4].double()).numpy()
    assert live.sum() > 0.9 * len(R_fs)
    np.testing.assert_allclose(R_fs[live], R_match[live], rtol=0.0, atol=1e-6)


def test_matching_radius_is_the_one_peak_search_thresholds_on(tmp_path):
    """The peak-search band (midas_peakfit compute_rt_eta) and calc_radius
    must define the radius of a pixel identically."""
    pf_geom = pytest.importorskip("midas_peakfit.geometry")
    pf_params = pytest.importorskip("midas_peakfit.params")
    p = pf_params.ZarrParams()
    p.NrPixels = NY
    p.NrPixelsY, p.NrPixelsZ = NY, NZ
    p.Lsd, p.px, p.Ycen, p.Zcen = LSD, PX, BCY, BCZ
    p.tx, p.ty, p.tz = 0.0, *REPORT_TILT
    p.RhoD = 3000.0 * PX
    Rt, _ = pf_geom.compute_rt_eta(p, panels=[])
    rng = np.random.default_rng(0)
    a = rng.integers(0, NY, 400)
    b = rng.integers(0, NZ, 400)
    zp = _zarr_params(*REPORT_TILT)
    mine = match.corrected_radius_um(
        zp, torch.from_numpy(a.astype(np.float64)),
        torch.from_numpy(b.astype(np.float64))).numpy() / PX
    np.testing.assert_allclose(mine, Rt[a, b], rtol=0.0, atol=1e-6)


# --------------------------------------------------------------------------
# Which Width applies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kw,expected,source", [
    (dict(Width=600.0), 600.0, "Width"),
    (dict(Width=-1.0, WidthOrig=900.0), 900.0, "WidthOrig"),
    (dict(Width=600.0, WidthTthPx=2500.0), 2500.0, "WidthTthPx"),
    (dict(Width=-1.0, WidthOrig=-1.0, WidthTthPx=-1.0),
     match.DEFAULT_WIDTH_UM, "default"),
])
def test_resolve_width(kw, expected, source, caplog):
    zp = ZarrParams()
    for k, v in kw.items():
        setattr(zp, k, v)
    with caplog.at_level(logging.WARNING, logger=match.LOG.name):
        width, src = match.resolve_width_um(zp)
    assert (width, src) == (expected, source)
    assert ("default" in caplog.text) == (source == "default")


def test_missing_width_no_longer_drops_every_spot(report_spots, tmp_path, caplog):
    """An archive with no Width reached the kernel as -1 and matched nothing."""
    arr, ring_of = report_spots
    zp = _zarr_params(*REPORT_TILT, width=-1.0)
    assert zp.Width == -1.0 and zp.WidthOrig == -1.0
    with caplog.at_level(logging.WARNING, logger=match.LOG.name):
        res = _run(arr, zp, tmp_path)
    assert res.spots.shape[0] == len(arr)
    assert "default" in caplog.text


def test_width_tth_px_widens_the_assignment_window(tmp_path):
    """Spots 300 um off ring 23 (corrected radius): a 100 um Width rejects
    them, and the same Width with WidthTthPx = 600 um accepts them."""
    per_ring = {23: _ring_pixels(RINGS[23] + 300.0, *REPORT_TILT)}
    arr, _ = _result_array(per_ring)
    narrow = _run(arr, _zarr_params(*REPORT_TILT, width=100.0), tmp_path)
    wide = _run(arr, _zarr_params(*REPORT_TILT, width=100.0, WidthTthPx=600.0), tmp_path)
    assert narrow.spots.shape[0] == 0
    assert wide.spots.shape[0] == len(arr)


# --------------------------------------------------------------------------
# The report of what matched nothing
# --------------------------------------------------------------------------


def test_match_stats_count_matched_unmatched_and_near_misses(tmp_path):
    width = 600.0
    per_ring = {
        10: _ring_pixels(RINGS[10], *REPORT_TILT),                 # on ring
        13: _ring_pixels(RINGS[13] + 1.5 * width, *REPORT_TILT),   # near miss
        15: _ring_pixels(RINGS[15] + 6 * width, *REPORT_TILT),     # far from any ring
    }
    arr, ring_of = _result_array(per_ring)
    res = _run(arr, _zarr_params(*REPORT_TILT, width=width), tmp_path)
    s = res.match_stats
    n10, n13, n15 = (int((ring_of == r).sum()) for r in (10, 13, 15))
    assert s["n_input"] == len(arr)
    assert s["n_matched"] == n10
    assert s["n_unmatched"] == n13 + n15
    assert s["n_near_miss"] == n13
    assert s["per_ring"][10]["matched"] == n10
    assert s["per_ring"][13]["near_miss"] == n13
    assert s["per_ring"][15]["near_miss"] == 0
    assert "matched no ring" in s["summary"]


def test_starved_ring_with_near_misses_warns(tmp_path, caplog):
    width = 600.0
    arr, _ = _result_array({13: _ring_pixels(RINGS[13] + 1.5 * width, *REPORT_TILT),
                            10: _ring_pixels(RINGS[10], *REPORT_TILT)})
    with caplog.at_level(logging.INFO, logger=match.LOG.name):
        _run(arr, _zarr_params(*REPORT_TILT, width=width), tmp_path)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("ring 13 got no peaks" in r.getMessage() for r in warnings)
    assert any("matched a ring" in r.getMessage() for r in caplog.records
               if r.levelno == logging.INFO)


def test_nothing_matching_anything_warns(tmp_path, caplog):
    arr, _ = _result_array({10: _ring_pixels(RINGS[10] + 300.0, *REPORT_TILT)})
    with caplog.at_level(logging.WARNING, logger=match.LOG.name):
        _run(arr, _zarr_params(*REPORT_TILT, width=100.0), tmp_path)
    assert "no peak matched any configured ring" in caplog.text


def test_archive_without_geometry_falls_back_to_flat(tmp_path):
    """A ZarrParams with no Lsd/PixelSize cannot be tilt-corrected; the
    flat radius is used, as before."""
    arr, _ = _result_array({10: _ring_pixels(RINGS[10], 0.0, 0.0)})
    zp = _zarr_params(0.0, 0.0, width=600.0, Lsd=0.0)
    assert match.corrected_radius_um(zp, torch.zeros(3), torch.zeros(3)) is None
