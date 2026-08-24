"""Registration checks, and the meta-null that says whether they had power.

The organising worry: both of these checks can be run in a form that cannot
fail, and then reported as a pass. Half of these tests exist to prove the
refusals fire on exactly those cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_transforms.geometry.registration import (
    centroid_containment_check,
    meta_null,
    sinogram_check,
)
from midas_transforms.geometry.sample_shape import SampleShape


def _box(**kw):
    kw.setdefault("size_x_um", 300.0)
    kw.setdefault("size_y_um", 80.0)
    kw.setdefault("height_um", 40.0)
    kw.setdefault("pixel_size_um", 5.0)
    return SampleShape.box(**kw)


OMEGAS = np.arange(0.0, 360.0, 10.0)


# ------------------------------------------------------------------ V1

def test_V1_passes_when_the_measured_curve_tracks_the_prediction():
    s = _box()
    pred = s.illuminated_volume_sinogram(OMEGAS, beam_width_um=100.0)
    measured = 3.7 * pred + 1000.0                 # arbitrary gain and offset
    r = sinogram_check(s, OMEGAS, measured, beam_width_um=100.0)
    assert r.verdict == "PASS" and bool(r)
    assert r.statistic > 0.999


def test_V1_survives_realistic_noise():
    s = _box()
    pred = s.illuminated_volume_sinogram(OMEGAS, beam_width_um=100.0)
    rng = np.random.default_rng(0)
    measured = pred * (1.0 + 0.05 * rng.standard_normal(pred.size))
    assert sinogram_check(s, OMEGAS, measured, beam_width_um=100.0).verdict == "PASS"


def test_V1_fails_on_an_unrelated_curve():
    s = _box()
    r = sinogram_check(s, OMEGAS, np.cos(np.radians(OMEGAS)),
                       beam_width_um=100.0)
    assert r.verdict == "FAIL"


def test_V1_REFUSES_on_a_cylinder_because_it_has_no_power():
    """The headline case. A cylinder's lit volume does not vary with omega, so
    ANY measured curve 'agrees' with the flat prediction. This must not be a
    pass, and the message has to say why."""
    s = SampleShape.cylinder(diameter_um=200.0, height_um=40.0,
                             pixel_size_um=5.0)
    rng = np.random.default_rng(1)
    r = sinogram_check(s, OMEGAS, rng.uniform(1.0, 2.0, OMEGAS.size),
                       beam_width_um=100.0)
    assert r.verdict == "NO_POWER"
    assert not bool(r), "NO_POWER must not be truthy"
    assert "cannot distinguish" in r.message
    assert "NOT a pass" in r.message


def test_V1_reports_the_one_cycle_component_that_measures_an_axis_offset():
    s = _box()
    pred = s.illuminated_volume_sinogram(OMEGAS, beam_width_um=100.0)
    r = sinogram_check(s, OMEGAS, pred, beam_width_um=100.0)
    assert "measured_1cycle_amplitude" in r.detail
    assert "measured_1cycle_phase_deg" in r.detail


def test_V1_fails_loudly_when_the_beam_misses_the_sample_in_z():
    """The commonest real failure: slice0_z_um does not match the stage."""
    s = _box(centre_z_um=0.0)
    r = sinogram_check(s, OMEGAS, np.ones(OMEGAS.size), beam_width_um=100.0,
                       beam_height_um=10.0, beam_centre_z_um=5000.0)
    assert r.verdict == "FAIL" and "slice0_z_um" in r.message


def test_V1_needs_enough_omega_samples_to_fit_a_phase():
    s = _box()
    with pytest.raises(ValueError, match="too few"):
        sinogram_check(s, np.arange(4.0), np.ones(4), beam_width_um=100.0)


# ------------------------------------------------------------------ V2

def _centroids_in(shape, n, rng, jitter_um=0.0):
    pos = shape.voxel_positions_um()[shape.occupancy >= 0.5]
    pts = pos[rng.choice(pos.shape[0], size=n, replace=False)]
    return pts + jitter_um * rng.standard_normal(pts.shape)


def test_V2_passes_on_centroids_drawn_from_inside_the_mask():
    s = _box()
    rng = np.random.default_rng(0)
    r = centroid_containment_check(s, _centroids_in(s, 200, rng))
    assert r.verdict == "PASS"
    assert r.detail["held_out_contained"] >= 0.98


def test_V2_is_BLIND_to_a_pure_translation_by_construction():
    """A limit worth pinning, because it is easy to read a PASS as more than
    it is. The fit starts from the difference of centroids, so any rigid
    offset of the whole cloud is absorbed exactly. V2 tests the *shape* of the
    registration, never its origin.
    """
    s = _box()
    rng = np.random.default_rng(0)
    pts = _centroids_in(s, 200, rng) + np.array([0.0, 400.0, 0.0])
    r = centroid_containment_check(s, pts, search_px=3)
    assert r.verdict == "PASS"
    assert abs(r.detail["translation_um"][1] + 400.0) < s.pixel_size_um


def test_V2_fails_on_a_scale_error_which_no_translation_can_absorb():
    """The failure V2 does catch: a wrong pixel size. The mask is then the
    wrong size for the grain cloud at every offset."""
    s = _box()
    rng = np.random.default_rng(0)
    pts = _centroids_in(s, 200, rng)
    small = SampleShape(
        occupancy=s.occupancy, pixel_size_um=s.pixel_size_um / 3.0,
        slice_pitch_um=s.slice_pitch_um / 3.0,
        rot_axis_ix=s.rot_axis_ix, rot_axis_iy=s.rot_axis_iy,
        in_plane=s.in_plane,
    )
    assert centroid_containment_check(small, pts).verdict == "FAIL"


def test_V2_reports_held_out_separately_from_in_sample():
    """Fitting and scoring on the same grains reports ~100 % for any mask big
    enough to swallow the cloud, so the gap has to be visible."""
    s = _box()
    rng = np.random.default_rng(2)
    r = centroid_containment_check(s, _centroids_in(s, 120, rng))
    assert "in_sample_contained" in r.detail and "held_out_contained" in r.detail
    assert r.statistic == r.detail["held_out_contained"]


def test_V2_needs_enough_grains_to_split():
    s = _box()
    with pytest.raises(ValueError, match="too few"):
        centroid_containment_check(s, np.zeros((5, 3)))


def test_V2_refuses_an_empty_mask_rather_than_dividing_by_zero():
    s = _box()
    s.occupancy[...] = 0.0
    r = centroid_containment_check(s, np.zeros((20, 3)))
    assert r.verdict == "FAIL" and "empty" in r.message


# ------------------------------------------------------------- the meta-null

def test_mirroring_flips_the_handedness_and_nothing_else():
    s = _box()
    m = s.mirrored()
    assert m.in_plane != s.in_plane
    assert m.total_volume_um3() == s.total_volume_um3()
    assert "META-NULL" in m.provenance["source"]
    assert m.mirrored().in_plane == s.in_plane      # involution


def test_the_meta_null_says_NO_POWER_when_the_shape_is_symmetric():
    """A box centred on the axis is symmetric under a y flip, so V1 on it
    cannot see handedness however good its correlation was. This is the
    scenario the plan warns about: a check that passes and proves nothing."""
    s = _box()
    pred = s.illuminated_volume_sinogram(OMEGAS, beam_width_um=100.0)
    r = meta_null(sinogram_check, s, OMEGAS, pred, beam_width_um=100.0)
    assert r.verdict == "NO_POWER"
    assert "no power over handedness" in r.message


def test_the_meta_null_PASSES_on_a_shape_that_is_actually_chiral():
    """The companion: an L-shaped cross-section is not mirror-symmetric, so
    the same check does have power there. Without this the test above could be
    satisfied by a meta-null that always says NO_POWER."""
    occ = np.zeros((4, 40, 40))
    occ[:, 8:32, 8:16] = 1.0            # the upright of the L
    occ[:, 24:32, 8:34] = 1.0           # the foot
    s = SampleShape(occupancy=occ, pixel_size_um=5.0, slice_pitch_um=5.0,
                    rot_axis_ix=19.5, rot_axis_iy=19.5, in_plane="xy")
    pred = s.illuminated_volume_sinogram(OMEGAS, beam_width_um=40.0)
    r = meta_null(sinogram_check, s, OMEGAS, pred, beam_width_um=40.0)
    assert r.verdict == "PASS", r.message
    assert r.detail["real_statistic"] > r.detail["mirror_statistic"]


def test_the_meta_null_FAILS_when_the_mask_really_is_mirrored():
    """Feed the check a measurement generated from the mirror image. The
    mirrored mask then scores better, and that is the diagnosis, not a pass."""
    occ = np.zeros((4, 40, 40))
    occ[:, 8:32, 8:16] = 1.0
    occ[:, 24:32, 8:34] = 1.0
    s = SampleShape(occupancy=occ, pixel_size_um=5.0, slice_pitch_um=5.0,
                    rot_axis_ix=19.5, rot_axis_iy=19.5, in_plane="xy")
    truth = s.mirrored().illuminated_volume_sinogram(OMEGAS, beam_width_um=40.0)
    r = meta_null(sinogram_check, s, OMEGAS, truth, beam_width_um=40.0)
    assert r.verdict == "FAIL"
    assert "handedness is inverted" in r.message


def test_the_meta_null_works_on_V2_too():
    s = _box()
    rng = np.random.default_rng(3)
    pts = _centroids_in(s, 200, rng)
    r = meta_null(centroid_containment_check, s, pts)
    # A centred box is symmetric under the flip, so V2 cannot see it either.
    # That is the documented limit of containment, asserted rather than assumed.
    assert r.verdict == "NO_POWER"


# ------------------------------------------------------ the inverse affine

def test_contains_is_the_exact_inverse_of_voxel_positions():
    s = _box()
    pos = s.voxel_positions_um()
    inside = s.contains(pos[s.occupancy >= 0.5])
    assert inside.all()
    outside = s.contains(pos[s.occupancy < 0.5]) if (s.occupancy < 0.5).any() \
        else np.array([False])
    assert not outside.any()


def test_contains_round_trips_under_every_handedness():
    """The inverse is a transpose, so it must hold for all eight signed
    permutations, not just the identity one."""
    from midas_stress.frames import TOMO_IN_PLANE

    occ = np.zeros((2, 10, 14))
    occ[:, 3:7, 4:10] = 1.0
    for ip in sorted(TOMO_IN_PLANE):
        s = SampleShape(occupancy=occ, pixel_size_um=3.0, slice_pitch_um=3.0,
                        rot_axis_ix=6.5, rot_axis_iy=4.5, in_plane=ip)
        pos = s.voxel_positions_um()
        assert s.contains(pos[s.occupancy >= 0.5]).all(), ip
        assert not s.contains(pos[s.occupancy < 0.5]).any(), ip


def test_contains_does_not_clamp_points_outside_the_volume():
    """A point a millimetre above the tomogram is outside the sample, not in
    the nearest edge voxel."""
    s = _box()
    far = np.array([[0.0, 0.0, 1.0e4], [0.0, 1.0e4, 0.0], [1.0e4, 0.0, 0.0]])
    assert not s.contains(far).any()
