"""A geometry seeded by the last resort must not look like one that was not.

`calibrate()` has three seed paths. When the validated seeder raises, the
exception was caught, printed only under `verbose=True`, and a last-resort
chord-only arc seed silently took over — leaving no trace on the result.

That matters because the refiner tends to stay where it is put, so
`basin_check` then reports zero seed-to-MAP drift as "within safe basin". On the
1-ID archive a frame went through with a beam centre 103 px off the edge of a
4148-wide detector and collected green ticks from four of five gates.
"""
import warnings

import pytest

from midas_calibrate_v2.pipelines.auto import SeedFallbackWarning
from midas_calibrate_v2.pipelines.diagnostics import seed_provenance_gate

EIGER = dict(NrPixelsY=4148, NrPixelsZ=4362)


def test_fallback_is_a_failure_not_a_footnote():
    r = seed_provenance_gate(
        seed_method="fallback", seed_note="make_seed failed: no arcs detected",
        seed_BC_y=-103.4, seed_BC_z=2611.2, **EIGER)
    assert r.severity == "fail"
    assert "LAST-RESORT" in r.message
    # the reason must survive to the record, not be swallowed
    assert "no arcs detected" in r.message


def test_fallback_names_the_off_panel_beam_centre():
    off = seed_provenance_gate(seed_method="fallback", seed_BC_y=-103.4,
                               seed_BC_z=2611.2, **EIGER)
    on = seed_provenance_gate(seed_method="fallback", seed_BC_y=2074.0,
                              seed_BC_z=2181.0, **EIGER)
    assert off.metrics["off_panel"] == 1.0
    assert on.metrics["off_panel"] == 0.0
    assert "OFF THE PANEL" in off.message
    assert "OFF THE PANEL" not in on.message
    assert on.severity == "fail"          # still a fallback, still a failure


def test_validated_seeder_passes():
    r = seed_provenance_gate(seed_method="make_seed", seed_note="19 rings, rms=0.31 px",
                             seed_BC_y=1020.0, seed_BC_z=984.0,
                             NrPixelsY=2048, NrPixelsZ=2048)
    assert r.severity == "ok"
    assert "make_seed" in r.message


def test_user_supplied_seed_passes():
    r = seed_provenance_gate(seed_method="user", seed_BC_y=1000.0, seed_BC_z=1000.0,
                             NrPixelsY=2048, NrPixelsZ=2048)
    assert r.severity == "ok"


def test_off_panel_from_a_good_seeder_is_only_a_warning():
    """A wedge geometry legitimately has its beam centre off the panel, so this
    cannot be an error — but it must not be silent either."""
    r = seed_provenance_gate(seed_method="make_seed", seed_BC_y=-50.0,
                             seed_BC_z=984.0, NrPixelsY=2048, NrPixelsZ=2048)
    assert r.severity == "warn"
    assert "wedge" in r.message


def test_unrecorded_provenance_is_not_a_pass():
    r = seed_provenance_gate(seed_method="unknown")
    assert r.severity == "warn"


def test_missing_geometry_does_not_claim_off_panel():
    r = seed_provenance_gate(seed_method="make_seed")
    assert r.metrics["off_panel"] == 0.0
    assert r.severity == "ok"


def test_warning_class_is_exported_and_is_a_warning():
    assert issubclass(SeedFallbackWarning, UserWarning)
    with pytest.warns(SeedFallbackWarning):
        warnings.warn("x", SeedFallbackWarning)
