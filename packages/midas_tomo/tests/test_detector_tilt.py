"""Measuring the detector roll, and refusing to when the premise fails.

The headline gate is recovery of a *planted* angle: rotate a synthetic beam box
by a known amount and require it back to a hundredth of a degree. Everything
else here is about the two nulls, because without them the edge fitter returns
a confident angle for a shape that is not a rectangle.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_tomo.detector_tilt import (
    compare_tilt_estimates,
    locate_edge_subpixel,
    tilt_from_beam_box,
    tilt_from_rotation_axis,
)

scipy_rotate = pytest.importorskip("scipy.ndimage").rotate


def _box(ny=600, nx=700, angle_deg=0.0, x0=180, x1=520, y0=140, y1=460,
         level=10000.0, floor=200.0, noise=0.0, seed=0):
    """A rectangular beam footprint, optionally rolled by ``angle_deg``."""
    img = np.full((ny, nx), floor, dtype=np.float64)
    img[y0:y1, x0:x1] = level
    if angle_deg:
        img = scipy_rotate(img, angle_deg, reshape=False, order=1,
                           mode="nearest")
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return img


# --------------------------------------------------------- THE gate

@pytest.mark.parametrize("planted", [0.0, 0.05, -0.3, 1.0, -2.0])
def test_a_planted_roll_is_recovered(planted):
    """Plant a known angle, get its magnitude back exactly.

    The returned value is the CORRECTING angle, so it is the negative of the
    roll that was applied. That sign is not asserted here on its own -- see
    test_applying_the_measured_angle_squares_the_box, which checks the
    definition operationally instead of by convention.
    """
    r = tilt_from_beam_box(_box(angle_deg=planted))
    assert r.trustworthy, r.reason
    assert abs(r.angle_deg) == pytest.approx(abs(planted), abs=0.02), r.summary()


@pytest.mark.parametrize("planted", [0.05, -0.3, 1.0, -2.0])
def test_applying_the_measured_angle_squares_the_box(planted):
    """The definition that matters: rotating by the returned angle removes the
    roll. This is what makes the number safe to drop into RotationAngle, which
    hdf5.py:181 feeds straight to _rotate_stack."""
    rolled = _box(angle_deg=planted)
    measured = tilt_from_beam_box(rolled).angle_deg
    corrected = scipy_rotate(rolled, measured, reshape=False, order=1,
                             mode="nearest")
    residual = tilt_from_beam_box(corrected).angle_deg
    assert abs(residual) < 0.02, f"{planted} -> {measured} left {residual}"


def test_all_four_edges_are_fitted_and_agree():
    r = tilt_from_beam_box(_box(angle_deg=0.4))
    assert {e.name for e in r.edges} == {"left", "right", "top", "bottom"}
    angles = [e.angle_deg for e in r.edges]
    assert max(angles) - min(angles) < 0.03, r.summary()


def test_it_survives_realistic_noise():
    r = tilt_from_beam_box(_box(angle_deg=-0.25, noise=120.0))
    assert r.trustworthy, r.reason
    assert abs(r.angle_deg) == pytest.approx(0.25, abs=0.03)


def test_a_dark_field_is_subtracted_when_given():
    dark = np.full((600, 700), 500.0)
    r = tilt_from_beam_box(_box(angle_deg=0.2, floor=500.0), dark)
    assert r.trustworthy and abs(r.angle_deg) == pytest.approx(0.2, abs=0.02)


def test_the_uncertainty_shrinks_with_a_longer_lever_arm():
    """Precision comes from the span of the edge, so a taller box must do
    better. If it does not, the fit is dominated by something else."""
    short = tilt_from_beam_box(_box(ny=600, y0=230, y1=380, noise=120.0))
    tall = tilt_from_beam_box(_box(ny=600, y0=60, y1=540, noise=120.0))
    assert tall.edges[0].span_px > short.edges[0].span_px
    assert tall.edges[0].rms_px / max(tall.edges[0].span_px, 1) < \
        short.edges[0].rms_px / max(short.edges[0].span_px, 1)


# ------------------------------------------------------------ the nulls

def test_a_NON_RECTANGULAR_aperture_is_refused_by_the_orthogonality_null():
    """The null that stops a confident angle being reported for a trapezoid.

    Built by shearing the box: opposite edges stay parallel, so the
    parallelism check passes and only orthogonality can catch it.
    """
    ny, nx = 600, 700
    img = np.full((ny, nx), 200.0)
    for y in range(140, 460):
        dx = int(0.25 * (y - 140))          # shear: verticals lean, tops do not
        img[y, 180 + dx:520 + dx] = 10000.0
    r = tilt_from_beam_box(img)
    assert not r.trustworthy
    assert "not a rectangle" in r.reason
    assert abs(r.detail["orthogonality_error_deg"]) > 0.05


def test_a_NON_PARALLEL_pair_is_refused():
    """One 'edge' that is not an aperture edge -- here the right-hand boundary
    fans out, as the shadow of something else would."""
    ny, nx = 600, 700
    img = np.full((ny, nx), 200.0)
    for y in range(140, 460):
        right = 520 + int(0.30 * (y - 300))
        img[y, 180:right] = 10000.0
    r = tilt_from_beam_box(img)
    assert not r.trustworthy
    assert "not parallel" in r.reason or "not a rectangle" in r.reason


def test_a_beam_that_overfills_the_detector_is_refused():
    """No aperture edges in view means no power, not an answer of zero."""
    img = np.full((400, 400), 10000.0)
    img += np.random.default_rng(0).normal(0, 5.0, img.shape)
    with pytest.raises(ValueError, match="no illuminated region|reaches the edge"):
        tilt_from_beam_box(img)


def test_an_aperture_running_off_the_frame_is_refused():
    img = np.full((600, 700), 200.0)
    img[140:460, 180:700] = 10000.0          # right edge outside the frame
    with pytest.raises(ValueError, match="reaches the edge of the detector"):
        tilt_from_beam_box(img)


def test_a_tiny_illuminated_box_is_refused():
    img = np.full((600, 700), 200.0)
    img[300:320, 300:320] = 10000.0
    with pytest.raises(ValueError, match="too small to fit edges"):
        tilt_from_beam_box(img)


# ---------------------------------------------------------- the edge finder

def test_the_subpixel_edge_finder_is_unbiased():
    """A step placed between samples must come back between samples."""
    for true_pos in (40.0, 40.25, 40.5, 40.75):
        x = np.arange(80, dtype=float)
        prof = 100.0 + 900.0 / (1.0 + np.exp(-(x - true_pos) / 0.7))
        got = locate_edge_subpixel(prof, rising=True)
        assert got == pytest.approx(true_pos, abs=0.15), f"{true_pos} -> {got}"


def test_the_edge_finder_returns_None_with_no_edge():
    assert locate_edge_subpixel(np.full(80, 5.0), rising=True) is None


def test_a_falling_edge_is_found_too():
    x = np.arange(80, dtype=float)
    prof = 1000.0 - 900.0 / (1.0 + np.exp(-(x - 52.0) / 0.7))
    assert locate_edge_subpixel(prof, rising=False) == pytest.approx(52.0, abs=0.2)


# ------------------------------------------------- the rotation-axis route

def _sino_stack(ny=160, nx=160, n_ang=180, roll_deg=0.0, axis0=80.0,
                radius=18.0, offset=25.0, mu=0.02):
    """Projections of an off-axis rod, with the detector rolled.

    The rod sits ``offset`` from the rotation axis so its centre of mass
    oscillates; that oscillation must average away over the full turn, leaving
    only the axis position.
    """
    ang = np.linspace(-180.0, 180.0, n_ang)
    x = np.arange(nx, dtype=np.float64)
    data = np.zeros((n_ang, ny, nx), dtype=np.float64)
    for i, a in enumerate(ang):
        cx_base = axis0 + offset * math.cos(math.radians(a))
        for r in range(ny):
            # A rolled detector makes the projected axis drift with row.
            cx = cx_base + math.tan(math.radians(roll_deg)) * r
            t = np.clip(radius ** 2 - (x - cx) ** 2, 0.0, None)
            data[i, r] = np.exp(-mu * 2.0 * np.sqrt(t))
    dark = np.zeros((ny, nx))
    whites = np.ones((2, ny, nx))
    return (data * 1000.0), (dark), (whites * 1000.0), ang


@pytest.mark.parametrize("planted", [0.0, 0.5, -1.0])
def test_the_rotation_axis_route_recovers_a_planted_roll(planted):
    data, dark, whites, ang = _sino_stack(roll_deg=planted)
    r = tilt_from_rotation_axis(data, dark, whites, ang)
    assert r.trustworthy, r.reason
    assert r.angle_deg == pytest.approx(planted, abs=0.05), r.summary()


def test_the_specimen_offset_averages_away():
    """The premise of the method: over 360 degrees the centre-of-mass
    oscillation cancels, so a badly off-axis rod gives the same answer."""
    a = tilt_from_rotation_axis(*_sino_stack(roll_deg=0.5, offset=5.0)[:3],
                                _sino_stack(roll_deg=0.5, offset=5.0)[3])
    b = tilt_from_rotation_axis(*_sino_stack(roll_deg=0.5, offset=40.0)[:3],
                                _sino_stack(roll_deg=0.5, offset=40.0)[3])
    assert a.angle_deg == pytest.approx(b.angle_deg, abs=0.05)


def test_a_180_degree_scan_is_REFUSED_not_biased():
    """Half a turn leaves the oscillation uncancelled, so the answer would
    depend on the specimen's asymmetry rather than the geometry."""
    data, dark, whites, _ = _sino_stack(roll_deg=0.5)
    half = np.linspace(0.0, 180.0, data.shape[0])
    with pytest.raises(ValueError, match="does not cancel"):
        tilt_from_rotation_axis(data, dark, whites, half)


# -------------------------------------------------------- the cross-check

def _res(angle, trust=True, method="m", reason="reason"):
    from midas_tomo.detector_tilt import TiltResult
    return TiltResult(angle_deg=angle, uncertainty_deg=0.01, method=method,
                      trustworthy=trust, reason=reason)


def test_agreement_between_the_two_routes_is_reported_as_established():
    c = compare_tilt_estimates(_res(0.31), _res(0.33))
    assert c["verdict"] == "AGREE"
    assert c["recommended_deg"] == pytest.approx(0.32)


def test_disagreement_says_which_number_to_use_and_why():
    """They measure against different references, so the gap is informative:
    the slits are not square to the rotation axis, and it is the axis the
    reconstruction geometry depends on. Requires BOTH to be valid."""
    c = compare_tilt_estimates(_res(0.31), _res(0.90))
    assert c["verdict"] == "DISAGREE"
    assert c["recommended_deg"] == 0.90
    assert "not square to the rotation axis" in c["note"]


def test_a_disagreement_with_an_INVALID_input_is_not_a_recommendation():
    """The bug this caught on real data, 2026-08-23.

    On the Ce scan the rotation-axis route returned -2.5172 deg with 40 px of
    residual scatter against a 2 px limit -- it flagged itself invalid. The
    comparison recommended it anyway, because the disagreement branch never
    looked at ``trustworthy``. The earlier test used two valid inputs and so
    could never have caught it.

    A disagreement is only interpretable as "the slits are not square to the
    rotation axis" when both numbers are real measurements.
    """
    box = _res(0.0181, trust=True, method="beam-box edges")
    axis = _res(-2.5172, trust=False, method="rotation-axis drift",
                reason="scatter 40.28 px about a straight line")
    c = compare_tilt_estimates(box, axis)
    assert c["verdict"] != "DISAGREE"
    assert c["recommended_deg"] == pytest.approx(0.0181)
    assert "flagged itself invalid" in c["note"]
    assert "not square to the rotation axis" not in c["note"] or \
        "says nothing about" in c["note"]


def test_two_invalid_inputs_give_no_measurement_at_all():
    c = compare_tilt_estimates(_res(0.3, trust=False), _res(1.7, trust=False))
    assert c["verdict"] == "NO MEASUREMENT"
    assert math.isnan(c["recommended_deg"])


def test_numerical_agreement_with_an_untrustworthy_input_is_not_promoted():
    c = compare_tilt_estimates(_res(0.31, trust=False), _res(0.32))
    assert c["verdict"] == "UNCERTAIN"
    assert c["recommended_deg"] == pytest.approx(0.32)   # the valid one only
    assert "says nothing about" in c["note"]


# --------------------------------------- route 3: per-slice best shift

def _tilted_cube(n_shifts=21, n_slices=40, x=64, slope_px_per_row=0.05,
                 base_idx=10):
    """A sweep whose optimum index drifts linearly with slice.

    That drift is exactly what a rolled detector produces: the rotation axis
    is not vertical in detector coordinates, so each row's best shift differs.
    """
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    r = np.hypot(ix - (x - 1) / 2, iy - (x - 1) / 2)
    disc = 1.0 / (1.0 + np.exp((r - x * 0.25) / 0.8))
    streaks = np.sin(ix * 2.1) * np.cos(iy * 1.7)
    cube = np.empty((n_shifts, n_slices, x, x), dtype=np.float32)
    for s in range(n_slices):
        best = base_idx + slope_px_per_row * s          # in shift-index units
        for i in range(n_shifts):
            k = abs(i - best)
            cube[i, s] = disc * (1.0 - 0.03 * k) + 0.04 * k * streaks
    return cube


def test_the_per_slice_shift_route_recovers_a_planted_slope():
    from midas_tomo.detector_tilt import tilt_from_slice_shifts

    # sweep step 1.0 px/index, so slope in index == slope in px per row
    sweep = (-10.0, 10.0, 1.0)
    slope = 0.05
    r = tilt_from_slice_shifts(_tilted_cube(slope_px_per_row=slope), sweep)
    assert r.detail["slope_px_per_row"] == pytest.approx(slope, abs=0.02)
    assert r.angle_deg == pytest.approx(math.degrees(math.atan(slope)), abs=1.5)


def test_a_slope_below_the_sweep_RESOLUTION_is_an_upper_bound_not_a_value():
    """A slope fitted through quantised picks always returns some number. With
    a 1 px step over 40 rows nothing finer than ~1.4 deg is visible."""
    from midas_tomo.detector_tilt import tilt_from_slice_shifts

    sweep = (-10.0, 10.0, 1.0)
    r = tilt_from_slice_shifts(_tilted_cube(slope_px_per_row=0.0), sweep)
    assert not r.trustworthy
    assert "UPPER BOUND" in r.reason
    assert r.detail["resolution_deg"] > 0


def test_slices_with_no_interior_optimum_are_dropped():
    from midas_tomo.detector_tilt import tilt_from_slice_shifts

    cube = _tilted_cube(slope_px_per_row=0.05)
    cube[:, :6] = 1.0                    # first six slices carry nothing
    r = tilt_from_slice_shifts(cube, (-10.0, 10.0, 1.0))
    assert r.detail["dropped_slices"], "an empty slice must not be fitted"


def test_too_few_usable_slices_raises():
    from midas_tomo.detector_tilt import tilt_from_slice_shifts

    cube = np.ones((21, 40, 64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="need at least 3"):
        tilt_from_slice_shifts(cube, (-10.0, 10.0, 1.0))
