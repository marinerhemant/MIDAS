"""Setting Vsample from a tomogram — the integration that never existed.

Most of these are refusals. Writing a measured-looking `Vsample` that is
actually threshold-driven is worse than leaving the template constant in place,
because a constant is at least obviously a constant.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_transforms.geometry import SampleShape
from midas_transforms.radius.vsample import (
    VsampleResult,
    vsample_from_shape,
    write_vsample,
)

GOOD_THRESHOLD = {"stationary": True, "fractional_spread": 0.01}
BAD_THRESHOLD = {"stationary": False, "fractional_spread": 0.72}
SOFT_THRESHOLD = {"stationary": False, "fractional_spread": 0.157}


def _rod(diameter_um=1000.0, height_um=140.0, px=10.0):
    return SampleShape.cylinder(diameter_um=diameter_um, height_um=height_um,
                                pixel_size_um=px)


# ------------------------------------------------------------ the arithmetic

def test_vsample_is_the_cross_section_times_the_beam_height():
    s = _rod(diameter_um=1000.0)
    r = vsample_from_shape(s, beam_height_um=13.0,
                           threshold_report=GOOD_THRESHOLD)
    assert r.usable, r.reasons
    want = math.pi * 500.0 ** 2 * 13.0
    assert r.vsample_um3 == pytest.approx(want, rel=0.02)
    assert r.equivalent_diameter_um == pytest.approx(1000.0, rel=0.02)


def test_it_reproduces_the_measured_Ce_numbers():
    """The real case: Ce ht525 s2 measured 772 241 um^2, and the FF file's
    Vsample 1e7 implies a 12.95 um beam height."""
    s = _rod(diameter_um=991.6, px=5.0)
    r = vsample_from_shape(s, beam_height_um=12.95,
                           threshold_report=GOOD_THRESHOLD)
    assert r.cross_section_um2 == pytest.approx(772241.0, rel=0.02)
    assert r.vsample_um3 == pytest.approx(1.0e7, rel=0.03)


def test_the_scale_factors_are_reported_against_the_old_value(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text("SpaceGroup 225;\nVsample 50000000;\nHbeam 2000;\n"
                 "Rsample 2000;\n")
    s = _rod(diameter_um=1000.0)
    r = vsample_from_shape(s, beam_height_um=13.0, param_file=p,
                           threshold_report=GOOD_THRESHOLD)
    assert r.previous_vsample_um3 == 5.0e7
    assert "TEMPLATE DEFAULT" in r.previous_source
    assert r.volume_scale == pytest.approx(r.vsample_um3 / 5.0e7)
    assert r.radius_scale == pytest.approx(r.volume_scale ** (1 / 3))


def test_a_missing_Vsample_line_reports_the_search_bound_fallback(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text("Hbeam 2000;\nRsample 2000;\n")
    r = vsample_from_shape(_rod(), beam_height_um=13.0, param_file=p,
                           threshold_report=GOOD_THRESHOLD)
    assert "SEARCH BOUNDS" in r.previous_source
    assert r.previous_vsample_um3 == pytest.approx(2000 * math.pi * 2000 ** 2)


# --------------------------------------------------------------- the refusals

def test_the_beam_height_is_never_guessed():
    with pytest.raises(ValueError, match="beam_height_um must be > 0"):
        vsample_from_shape(_rod(), beam_height_um=0.0)
    with pytest.raises(ValueError, match="search bound"):
        vsample_from_shape(_rod(), beam_height_um=-1.0)


def test_a_wildly_threshold_driven_volume_is_REFUSED():
    """72 % across the sweep: the number means nothing."""
    r = vsample_from_shape(_rod(), beam_height_um=13.0,
                           threshold_report=BAD_THRESHOLD)
    assert not r.usable
    assert any("too much" in x for x in r.reasons)


def test_a_MODERATELY_soft_boundary_is_accepted_WITH_its_uncertainty():
    """The trade that matters. Ce ht525 s2 measured 16 % across the sweep --
    real, and caused by the specimen edge sitting in the beam penumbra. But it
    replaces a template constant that can be wrong by orders of magnitude, so
    refusing it would be the worse error. The uncertainty travels with the
    value instead."""
    r = vsample_from_shape(_rod(), beam_height_um=13.0,
                           threshold_report=SOFT_THRESHOLD)
    assert r.usable, r.reasons
    assert "+/-8 %" in r.detail["volume_uncertainty"]
    prov = "\n".join(r.provenance_lines())
    assert "MEASUREMENT WITH AN UNCERTAINTY" in prov
    assert "threshold_spread" in prov


def test_the_spread_limit_is_adjustable():
    r = vsample_from_shape(_rod(), beam_height_um=13.0, max_spread=0.10,
                           threshold_report=SOFT_THRESHOLD)
    assert not r.usable


def test_no_threshold_report_at_all_is_also_refused():
    """Silence is not evidence of stationarity."""
    r = vsample_from_shape(_rod(), beam_height_um=13.0)
    assert not r.usable
    assert any("not known whether" in x for x in r.reasons)


def test_a_height_varying_cross_section_needs_the_registration():
    """If the specimen tapers, which slab the beam lit matters."""
    occ = np.zeros((40, 60, 60))
    iy, ix = np.mgrid[0:60, 0:60].astype(float)
    rr = np.hypot(ix - 29.5, iy - 29.5)
    for k in range(40):
        occ[k] = rr <= (8.0 + 0.5 * k)          # a cone
    s = SampleShape(occupancy=occ, pixel_size_um=10.0, slice_pitch_um=10.0,
                    rot_axis_ix=29.5, rot_axis_iy=29.5, in_plane="xy")
    r = vsample_from_shape(s, beam_height_um=13.0,
                           threshold_report=GOOD_THRESHOLD)
    assert not r.usable
    assert any("which slab the beam lit matters" in x for x in r.reasons)
    assert r.cross_section_cv > 0.05


def test_a_uniform_rod_passes_the_uniformity_check():
    r = vsample_from_shape(_rod(), beam_height_um=13.0,
                           threshold_report=GOOD_THRESHOLD)
    assert r.cross_section_cv < 0.01
    assert r.usable


def test_an_omega_varying_illuminated_volume_is_REFUSED():
    """Vsample is a SCALAR. A narrow beam on a non-cylindrical specimen makes
    V_illum depend on omega, and then no single value is correct."""
    s = SampleShape.box(size_x_um=900.0, size_y_um=200.0, height_um=140.0,
                        pixel_size_um=10.0)
    r = vsample_from_shape(s, beam_height_um=13.0, beam_width_um=250.0,
                           threshold_report=GOOD_THRESHOLD)
    assert not r.usable
    assert any("SCALAR" in x for x in r.reasons)
    assert r.omega_modulation > 0.05


def test_a_beam_wider_than_a_round_specimen_does_not_modulate():
    """The companion: the usual FF case must not trip the omega refusal, or it
    would refuse everything."""
    s = _rod(diameter_um=400.0)
    r = vsample_from_shape(s, beam_height_um=13.0, beam_width_um=2000.0,
                           threshold_report=GOOD_THRESHOLD)
    assert r.omega_modulation < 0.02
    assert r.usable, r.reasons


def test_the_beam_height_source_is_recorded(tmp_path):
    """The Ce case: the height came from a knife-edge measurement in the
    beamline log, not from an operator and emphatically not from the slits
    (which would have been 100x too large). The file has to say which."""
    r = vsample_from_shape(
        _rod(), beam_height_um=1.0, threshold_report=GOOD_THRESHOLD,
        beam_height_source="FullLog.log knife-edge measurement, command 131",
    )
    prov = "\n".join(r.provenance_lines())
    assert "knife-edge" in prov
    assert "operator-supplied" not in prov
    assert "beam height      1.0 um [FullLog" in r.summary()


def test_an_empty_mask_raises():
    s = _rod()
    s.occupancy[...] = 0.0
    with pytest.raises(ValueError, match="mask is empty"):
        vsample_from_shape(s, beam_height_um=13.0)


# ---------------------------------------------------------------- the writer

def test_writing_patches_the_file_and_keeps_the_old_value_visible(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text("SpaceGroup 225;\nVsample 50000000;\nLsd 1666219.6;\n")
    r = vsample_from_shape(_rod(), beam_height_um=13.0, param_file=p,
                           threshold_report=GOOD_THRESHOLD)
    write_vsample(p, r)

    txt = p.read_text()
    assert "# superseded by the measured value below: Vsample 50000000;" in txt
    assert f"Vsample {r.vsample_um3:.6f}" in txt
    assert "MEASURED from a tomographic reconstruction" in txt
    assert "operator-supplied" in txt          # the beam-height source is named
    assert "Lsd 1666219.6;" in txt             # nothing else disturbed
    # and exactly one live Vsample line survives
    live = [l for l in txt.splitlines()
            if l.strip().lower().startswith("vsample")]
    assert len(live) == 1


def test_the_original_is_backed_up(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text("Vsample 50000000;\n")
    r = vsample_from_shape(_rod(), beam_height_um=13.0,
                           threshold_report=GOOD_THRESHOLD)
    write_vsample(p, r)
    bak = p.with_suffix(p.suffix + ".before_vsample")
    assert bak.is_file() and "Vsample 50000000" in bak.read_text()


def test_writing_an_UNUSABLE_result_is_refused(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text("Vsample 50000000;\n")
    r = vsample_from_shape(_rod(), beam_height_um=13.0,
                           threshold_report=BAD_THRESHOLD)
    with pytest.raises(ValueError, match="refusing to write an unusable"):
        write_vsample(p, r)
    assert "Vsample 50000000" in p.read_text()       # untouched


def test_force_writes_it_anyway_and_says_so(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text("Vsample 50000000;\n")
    r = vsample_from_shape(_rod(), beam_height_um=13.0,
                           threshold_report=BAD_THRESHOLD)
    write_vsample(p, r, force=True)
    assert f"Vsample {r.vsample_um3:.6f}" in p.read_text()


def test_a_file_with_no_Vsample_records_what_it_was_using(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text("Hbeam 2000;\nRsample 2000;\n")
    r = vsample_from_shape(_rod(), beam_height_um=13.0, param_file=p,
                           threshold_report=GOOD_THRESHOLD)
    write_vsample(p, r)
    txt = p.read_text()
    assert "no Vsample line existed before" in txt
    assert "search bounds" in txt


def test_the_written_file_still_parses_as_a_gauge_volume(tmp_path):
    """The round trip that matters: what we wrote is what the pipeline reads."""
    from midas_transforms.radius.shape_correction import GaugeVolume

    p = tmp_path / "Parameters.txt"
    p.write_text("Hbeam 2000;\nRsample 2000;\nVsample 50000000;\n")
    r = vsample_from_shape(_rod(), beam_height_um=13.0,
                           threshold_report=GOOD_THRESHOLD)
    write_vsample(p, r)
    g = GaugeVolume.from_param_file(p)
    assert g.value_um3 == pytest.approx(r.vsample_um3, rel=1e-6)
    assert g.source == "Vsample"
    assert not g.is_template_default
