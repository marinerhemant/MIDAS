"""Tests for polytype/gap_fraction.py, on a 900x900 synthetic detector carrying the same
real-like tilt/distortion values as test_rod_profile.py's TILTED_GEOM.

Each test would fail on the mistake it guards against: a wrong voxel -> L map or weight
(planted f not recovered), a static background leaking into G (ring test), a bias from
starved gap bins (masked stripe), a silently accepted nonlinear/masked node, or the specific
real-data failure mode `bg_mode="local_t"` exists to fix (a background level that drifts
across the frame/scan index, uniform over the detector at each frame -- `bg_mode="temporal"`
reads it from the wrong frames and gets it wrong; `local_t` reads the same frames the signal
sits in, at a different position, and does not).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import numpy as np
import pytest
import torch

from midas_defect.forward_sim import hendricks_teller
from midas_defect.geometry import Geometry, detector_angle_maps
from midas_defect.polytype.gap_fraction import (PERIOD_L, empty_site_sum, f_from_gap_fraction,
                                                find_empty_site, gap_fraction_from_f,
                                                gap_fraction_numeric, gap_fraction_window,
                                                gaussian_unit_total, ht_density_fn, inject_gaussian,
                                                plant_rod)
from midas_defect.rod_profile import rod_path_geometry

LAM, LSD, PX = 0.42459, 349_622.0, 172.0
A_CELL, C_CELL = 3.6116, 19.2516
BMAT = 2 * math.pi * np.diag([1 / A_CELL, 1 / A_CELL, 1 / C_CELL])       # "2pi/d" convention
GEOM = Geometry(lsd_um=LSD, bcy_px=450.0, bcz_px=450.0, px_um=PX, wavelength_A=LAM,
                n_pix_y=900, n_pix_z=900, omega_first_deg=-18.5, omega_step_deg=1.0, n_frames=38,
                tx_deg=0.05, ty_deg=-0.22, tz_deg=-0.31,
                p_coeffs=(0.0002, 0.0011, 0.0002, 30.0) + (0.0,) * 11, rho_d_um=196_000.0)
SIGMA_L = 0.0195          # a plausible matched instrumental L floor: FWHM 0.046 / 2.3548
L0 = 8


def _U():
    """An orientation whose (0,0,L0) node's Ewald crossing lands mid-scan, so a +/-K frame
    window and background frames on both sides fit inside n_frames."""

    def _rot(ax, deg):
        t = math.radians(deg); c, s = math.cos(t), math.sin(t)
        if ax == "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    U0 = _rot("x", -78.0)
    best = None
    for phi in np.radians(np.arange(-60.0, 60.5, 1.0)):
        U = _rot("z", phi) @ U0
        p = rod_path_geometry(U, BMAT, 0, 0, np.arange(L0 - 1.1, L0 + 1.11, 0.1), GEOM, omega_sign=1)
        if len(p) < 23:
            continue
        k = (p.omega_deg - GEOM.omega_first_deg) / GEOM.omega_step_deg
        edge = min(k.min(), GEOM.n_frames - 1 - k.max())
        mid = abs(0.5 * (k.min() + k.max()) - (GEOM.n_frames - 1) / 2)
        if edge >= 12 and (best is None or mid < best[0]):
            best = (mid, U)
    if best is None:
        raise RuntimeError("no orientation puts the test rod mid-scan")
    return best[1]


U = _U()


@pytest.fixture(scope="module")
def setup():
    path = rod_path_geometry(U, BMAT, 0, 0, [L0 - 1.1, L0, L0 + 1.1], GEOM, omega_sign=1)
    if len(path) < 3:
        pytest.skip("rod not on the test detector")
    return path


def _frames(level=50.0):
    return np.full((GEOM.n_frames, GEOM.n_pix_z, GEOM.n_pix_y), level, np.float32)


def _plant(frames, f, scale=4000.0, rng=None, poisson=False):
    """Plants through the package's own ht_density_fn (Fourier-coefficient construction,
    handles f=0 -- a true delta comb -- exactly, unlike a discretely-sampled real-space
    hendricks_teller would). The closed-form cross-check against forward_sim.hendricks_teller
    below tests gap_fraction_from_f/gap_fraction_numeric directly and does not need planting."""
    fn = ht_density_fn(f, SIGMA_L, L0=L0, scale=scale)
    plant_rod(frames, GEOM, U, BMAT, 1, (0, 0), L0 - 1.3, L0 + 1.3, fn, T_px=14.0, sigma_t_px=2.0,
             sigma_k_frames=1.0, K=4, rng=rng, poisson=poisson)


def _run(frames, mask=None, **kw):
    if mask is None:
        mask = np.zeros(frames.shape[1:], bool)
    return gap_fraction_window(frames, mask, GEOM, U, BMAT, 1, L0, T_px=14.0, K=4, **kw)


# --------------------------------------------------------------------------------------------
# The closed-form Hendricks-Teller gap fraction, cross-checked against forward_sim.hendricks_teller
# --------------------------------------------------------------------------------------------

def _numeric_G(f, half_L):
    L = np.linspace(-1.0, 1.0, 2_000_001)[:-1]
    I = hendricks_teller(torch.as_tensor(L / PERIOD_L, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64),
                         torch.tensor(1.0 - f, dtype=torch.float64)).numpy()
    return float(I[np.abs(L) > half_L].sum() / I.sum())


@pytest.mark.parametrize("f", [0.01, 0.05, 0.2, 0.6])
@pytest.mark.parametrize("half_L", [0.15, 0.25, 0.4])
def test_closed_form_matches_forward_sim_hendricks_teller(f, half_L):
    assert float(gap_fraction_from_f(f, half_L)) == pytest.approx(_numeric_G(f, half_L), abs=2e-4)


def test_gap_fraction_limits():
    assert float(gap_fraction_from_f(0.0, 0.25)) == pytest.approx(0.0, abs=1e-12)       # perfect stack
    assert float(gap_fraction_from_f(1.0 - 1e-9, 0.25)) == pytest.approx(0.75, abs=1e-6)  # uniform rod


@pytest.mark.parametrize("f", [0.003, 0.03, 0.14, 0.5])
def test_f_from_gap_fraction_round_trip(f):
    assert f_from_gap_fraction(float(gap_fraction_from_f(f, 0.25)), 0.25) == pytest.approx(f, rel=1e-6)


def test_f_from_gap_fraction_clamps():
    assert f_from_gap_fraction(-0.01, 0.25) == 0.0
    assert f_from_gap_fraction(0.9, 0.25) == 1.0


# --------------------------------------------------------------------------------------------
# gap_fraction_window on real voxels
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("f", [0.02, 0.08, 0.2])
def test_planted_fault_probability_is_recovered_noise_free(setup, f):
    fr = _frames()
    _plant(fr, f)
    w = _run(fr)
    assert w.ok, w.reason
    assert w.G == pytest.approx(gap_fraction_numeric(f, SIGMA_L, 0.25), abs=0.006)


def test_zero_fault_node_gives_zero_gap_fraction(setup):
    fr = _frames()
    _plant(fr, 0.0)
    w = _run(fr)
    assert w.ok, w.reason
    assert abs(w.G) < 0.003


def test_static_ring_is_removed_by_the_temporal_median(setup):
    fr = _frames()
    _plant(fr, 0.08)
    tth, _ = detector_angle_maps(GEOM)
    ring_tth = float(tth[int(round(setup.row[0])), int(round(setup.col[0]))])
    ring = 3000.0 * np.exp(-0.5 * ((tth - ring_tth) / 0.03) ** 2)
    fr += ring[None].astype(np.float32)
    w = _run(fr)
    assert w.ok, w.reason
    assert w.G == pytest.approx(gap_fraction_numeric(0.08, SIGMA_L, 0.25), abs=0.008)


def test_masked_stripe_across_the_gap_is_interpolated_not_biased(setup):
    fr = _frames()
    _plant(fr, 0.08)
    path = rod_path_geometry(U, BMAT, 0, 0, [L0 + 0.65, L0 + 0.75], GEOM, omega_sign=1)
    tr, tc = path.row[1] - path.row[0], path.col[1] - path.col[0]
    n = np.hypot(tr, tc); tr, tc = tr / n, tc / n
    rr, cc = np.mgrid[0:fr.shape[1], 0:fr.shape[2]]
    s = (rr - 0.5 * (path.row[0] + path.row[1])) * tr + (cc - 0.5 * (path.col[0] + path.col[1])) * tc
    mask = np.abs(s) <= 2.5
    w = _run(fr, mask=mask)
    assert w.ok, w.reason
    assert w.interpolated.any() or (w.coverage < 1).any()
    assert w.G == pytest.approx(gap_fraction_numeric(0.08, SIGMA_L, 0.25), abs=0.012)


def test_bright_node_is_flagged_nonlinear_and_a_faint_one_is_not(setup):
    fr = _frames()
    _plant(fr, 0.02, scale=4.0e6)
    assert _run(fr).n_node_vox_over_censor > 0
    fr = _frames()
    _plant(fr, 0.02, scale=4000.0)
    assert fr.max() < 119000.0
    assert _run(fr).n_node_vox_over_censor == 0


def test_masked_node_is_rejected(setup):
    fr = _frames()
    _plant(fr, 0.08)
    path = rod_path_geometry(U, BMAT, 0, 0, [L0], GEOM, omega_sign=1)
    mask = np.zeros(fr.shape[1:], bool)
    r, c = int(round(path.row[0])), int(round(path.col[0]))
    mask[r - 20:r + 21, c - 20:c + 21] = True
    w = _run(fr, mask=mask)
    assert not w.ok and "node coverage" in w.reason


def test_bg_mode_rejects_unknown_value(setup):
    fr = _frames()
    _plant(fr, 0.08)
    with pytest.raises(ValueError):
        _run(fr, bg_mode="nonsense")


# --------------------------------------------------------------------------------------------
# local_t background: the specific real-data failure mode it exists to fix
# --------------------------------------------------------------------------------------------

def test_local_t_background_tracks_a_frame_dependent_drift_that_temporal_background_misses(setup):
    """The mechanism `local_t` fixes: a background level that ramps across the SCAN (frame
    index), uniform over the detector at each frame. `temporal` reads its background from
    frames far away in the scan, at the WRONG point on the ramp; `local_t` reads it from the
    SAME frames the signal sits in, at a different transverse offset, and is unaffected."""
    fr = _frames(level=0.0)
    n_f = fr.shape[0]
    fr += (400.0 * np.arange(n_f) / n_f)[:, None, None].astype(np.float32)     # ramps 0 -> 400 counts
    _plant(fr, 0.08)
    truth = gap_fraction_numeric(0.08, SIGMA_L, 0.25)
    w_local = _run(fr, bg_mode="local_t", bg_margin=2.0, bg_width=8.0)
    assert w_local.ok, w_local.reason
    assert abs(w_local.G - truth) < 0.012
    w_temporal = _run(fr, bg_mode="temporal")
    assert w_temporal.ok, w_temporal.reason
    assert abs(w_temporal.G - truth) > abs(w_local.G - truth) + 0.02     # temporal is measurably worse


def test_local_t_background_still_removes_a_static_ring(setup):
    fr = _frames()
    _plant(fr, 0.08)
    tth, _ = detector_angle_maps(GEOM)
    ring_tth = float(tth[int(round(setup.row[0])), int(round(setup.col[0]))])
    ring = 3000.0 * np.exp(-0.5 * ((tth - ring_tth) / 0.03) ** 2)
    fr += ring[None].astype(np.float32)
    w = _run(fr, bg_mode="local_t", bg_margin=2.0, bg_width=8.0)
    assert w.ok, w.reason
    assert w.G == pytest.approx(gap_fraction_numeric(0.08, SIGMA_L, 0.25), abs=0.012)


def test_local_t_background_rejects_frames_with_too_few_valid_annulus_pixels(setup):
    fr = _frames()
    _plant(fr, 0.08)
    mask = np.ones(fr.shape[1:], bool)
    mask[int(round(setup.row[1])) - 2:int(round(setup.row[1])) + 3,
         int(round(setup.col[1])) - 2:int(round(setup.col[1])) + 3] = False     # only the core unmasked
    w = _run(fr, mask=mask, bg_mode="local_t", bg_margin=2.0, bg_width=8.0)
    assert not w.ok


# --------------------------------------------------------------------------------------------
# empty_site_sum / find_empty_site / inject_gaussian: standalone, no domain/geometry needed --
# these operate directly on detector pixel coordinates.
# --------------------------------------------------------------------------------------------

SHAPE2 = (9, 200, 200)
BC = (100.0, 100.0)


def _blank2(noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    return (np.full(SHAPE2, 500.0) + rng.normal(0.0, noise, SHAPE2)).astype(np.float32)


def _inject(frames, r0, c0, kf, amp, sigma=2.0, K=4):
    for off in range(-K, K + 1):
        kk = kf + off
        if not (0 <= kk < frames.shape[0]):
            continue
        rr, cc = np.mgrid[int(r0) - 8:int(r0) + 9, int(c0) - 8:int(c0) + 9]
        frames[kk, rr, cc] += amp * np.exp(-0.5 * (((rr - r0) ** 2 + (cc - c0) ** 2) / sigma ** 2))


def test_empty_site_sum_is_near_zero_on_blank_frames():
    fr = _blank2()
    mask = np.zeros(fr.shape[1:], bool)
    s, reason = empty_site_sum(fr, mask, 60.0, 140.0, 4, T_px=6.0, bg_mode="local_t", bg_margin=2.0, bg_width=8.0)
    assert reason == ""
    assert abs(s) < 500.0


def test_empty_site_sum_detects_a_planted_feature():
    fr = _blank2()
    _inject(fr, 60.0, 140.0, 4, amp=4000.0)
    mask = np.zeros(fr.shape[1:], bool)
    s, reason = empty_site_sum(fr, mask, 60.0, 140.0, 4, T_px=6.0, bg_mode="local_t", bg_margin=2.0, bg_width=8.0)
    assert reason == ""
    assert s > 3000.0


def test_find_empty_site_avoids_a_bright_neighbour_and_finds_the_blank_ones():
    fr = _blank2()
    r0, c0, kf = 100.0, 180.0, 4                       # radius 80 from BC, angle 0
    bright_theta = math.radians(360.0 / 24.0 * 3)       # the 4th candidate angle (i=3)
    br = BC[0] + 80.0 * math.sin(bright_theta); bc = BC[1] + 80.0 * math.cos(bright_theta)
    _inject(fr, br, bc, kf, amp=5000.0)
    mask = np.zeros(fr.shape[1:], bool)
    best, results = find_empty_site(fr, mask, r0, c0, kf, BC[0], BC[1], T_px=6.0, bg_mode="local_t",
                                    bg_margin=2.0, bg_width=8.0)
    assert best is not None
    assert abs(best["sum"]) < 1000.0
    bright_result = min(results, key=lambda x: abs(x["theta_deg"] - math.degrees(bright_theta)))
    assert bright_result["sum"] is not None and bright_result["sum"] > 3000.0
    assert abs(best["sum"]) < bright_result["sum"]


def test_find_empty_site_excludes_the_real_site_by_min_angle():
    fr = _blank2()
    mask = np.zeros(fr.shape[1:], bool)
    _, results = find_empty_site(fr, mask, 100.0, 180.0, 4, BC[0], BC[1], n_angles=24, min_angle_deg=20.0,
                                 T_px=6.0, bg_mode="local_t", bg_margin=2.0, bg_width=8.0)
    thetas = [r["theta_deg"] % 360.0 for r in results]
    assert all(min(abs(t - 0.0), 360.0 - abs(t - 0.0)) >= 20.0 - 1e-6 for t in thetas)


def test_inject_gaussian_recovered_by_empty_site_sum_noise_free_and_exactly_undoable():
    fr = _blank2(noise=0.0)
    mask = np.zeros(fr.shape[1:], bool)
    base, _ = empty_site_sum(fr, mask, 60.0, 140.0, 4, T_px=8.0, bg_mode="local_t", bg_margin=2.0, bg_width=8.0)
    added, applied = inject_gaussian(fr, 60.0, 140.0, 4, amp=2000.0, sigma_t_px=1.5, K=4, poisson=False)
    after, reason = empty_site_sum(fr, mask, 60.0, 140.0, 4, T_px=8.0, bg_mode="local_t", bg_margin=2.0, bg_width=8.0)
    assert reason == ""
    assert (after - base) == pytest.approx(added, rel=0.01)
    for kk, rr, cc, add in applied:
        fr[kk, rr, cc] -= add
    restored, _ = empty_site_sum(fr, mask, 60.0, 140.0, 4, T_px=8.0, bg_mode="local_t", bg_margin=2.0, bg_width=8.0)
    assert restored == pytest.approx(base, abs=1e-3)     # float32 frames: rounding, not a real residual


def test_gaussian_unit_total_matches_a_noise_free_injection():
    fr = _blank2(noise=0.0)
    total, _ = inject_gaussian(fr, 60.0, 140.0, 4, amp=3.7, sigma_t_px=2.2, K=4, sigma_k_frames=1.3, poisson=False)
    assert total == pytest.approx(3.7 * gaussian_unit_total(2.2, 1.3, K=4), rel=1e-5)
