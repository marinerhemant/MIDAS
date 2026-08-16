"""Stages must be observable WHILE they run, not only after they finish.

Before this, the provenance ledger recorded a stage only on completion, so
during the single stage that dominates a long FF run (peakfit is ~83 % of a
beta run's wall time) nothing outside the log said what was happening.
"""

from __future__ import annotations

import time

from midas_pipeline.provenance import (PROGRESS_FILENAME, ProvenanceStore,
                                       write_progress)

STAGES = ["zip_convert", "hkl", "peakfit", "indexing", "refinement"]


def test_running_status_is_not_complete(tmp_path):
    """Resume must re-run a stage that was interrupted mid-flight."""
    store = ProvenanceStore(tmp_path)
    store.record("peakfit", status="running", started_at=time.time())
    assert store.read("peakfit")["status"] == "running"
    assert not store.is_complete("peakfit")


def test_progress_file_marks_the_live_stage(tmp_path):
    store = ProvenanceStore(tmp_path)
    store.record("zip_convert", status="complete", started_at=1.0,
                 finished_at=101.0, duration_s=100.0)
    store.record("hkl", status="complete", started_at=101.0,
                 finished_at=101.6, duration_s=0.6)
    store.record("peakfit", status="running", started_at=time.time() - 630)

    write_progress(tmp_path, layer_nr=1, scan_mode="ff",
                   stage_names=STAGES, stages=store.all_stages())
    txt = (tmp_path / PROGRESS_FILENAME).read_text()

    assert "2/5 stages complete" in txt
    assert "RUNNING: peakfit" in txt
    assert "<-- in progress" in txt
    assert "10.5m" in txt                     # elapsed, formatted
    assert "indexing" in txt and "pending" in txt


def test_progress_file_is_rewritten_not_appended(tmp_path):
    store = ProvenanceStore(tmp_path)
    store.record("peakfit", status="running", started_at=time.time())
    write_progress(tmp_path, layer_nr=1, scan_mode="ff",
                   stage_names=STAGES, stages=store.all_stages())
    first = (tmp_path / PROGRESS_FILENAME).read_text()

    store.record("peakfit", status="complete", started_at=1.0,
                 finished_at=2.0, duration_s=1.0)
    write_progress(tmp_path, layer_nr=1, scan_mode="ff",
                   stage_names=STAGES, stages=store.all_stages())
    second = (tmp_path / PROGRESS_FILENAME).read_text()

    assert "RUNNING" not in second
    assert len(second) < len(first) + len(first)      # rewritten, not appended
    assert second.count("MIDAS pipeline progress") == 1


def test_progress_file_never_raises_on_unwritable_dir(tmp_path):
    """Reporting must not be able to kill a run."""
    bad = tmp_path / "does" / "not" / "exist"
    write_progress(bad, layer_nr=1, scan_mode="ff",
                   stage_names=STAGES, stages={})       # must not raise


def test_failed_status_is_recorded_and_shown(tmp_path):
    """A failed stage must READ as failed, not as one that never started.

    Asserting only that the name appears is not enough -- it did appear while
    the stage was rendering as "pending", because `failed` had no branch of its
    own and fell through the else. This file is what a user cats after a run
    stops, so the distinction is the entire point.
    """
    store = ProvenanceStore(tmp_path)
    store.record("indexing", status="failed", started_at=time.time())
    write_progress(tmp_path, layer_nr=1, scan_mode="ff",
                   stage_names=STAGES, stages=store.all_stages())
    assert not store.is_complete("indexing")
    txt = (tmp_path / PROGRESS_FILENAME).read_text()
    # Must be the stage ROW: the header also names failed stages, so matching
    # on the name alone would assert against the header and pass even if the
    # row still said "pending" -- the exact bug this test exists to catch.
    line = next(l for l in txt.splitlines()
                if "indexing" in l and l.lstrip().startswith("["))
    assert "FAILED" in line, f"failed stage rendered as: {line!r}"
    assert "pending" not in line


def test_failure_is_visible_in_the_header(tmp_path):
    """"3/12 complete" alone cannot be told apart from a run still going."""
    store = ProvenanceStore(tmp_path)
    store.record("zip_convert", status="complete", duration_s=1.0)
    store.record("peakfit", status="failed", started_at=time.time())
    write_progress(tmp_path, layer_nr=1, scan_mode="ff",
                   stage_names=STAGES, stages=store.all_stages())
    head = (tmp_path / PROGRESS_FILENAME).read_text().split("\n\n")[1]
    assert "FAILED: peakfit" in head


def test_failed_stage_reports_how_long_it_ran(tmp_path):
    """started_at must be the stage's START, not the moment it died.

    Stamping the failed record with time.time() reports every failure as
    instantaneous, which throws away the one number worth having after a
    multi-hour stage falls over.
    """
    store = ProvenanceStore(tmp_path)
    store.record("peakfit", status="failed",
                 started_at=time.time() - 1200.0, duration_s=1200.0)
    write_progress(tmp_path, layer_nr=1, scan_mode="ff",
                   stage_names=STAGES, stages=store.all_stages())
    # The stage ROW, not the header -- the header now names failed stages too,
    # so a bare `"peakfit" in l` matches it first.
    line = next(l for l in (tmp_path / PROGRESS_FILENAME).read_text().splitlines()
                if "peakfit" in l and l.lstrip().startswith("["))
    assert "20.0m" in line, f"expected ~20 minutes of elapsed time, got: {line!r}"


def test_outputs_appear_as_stages_complete(tmp_path):
    store = ProvenanceStore(tmp_path)
    store.record("peakfit", status="complete", started_at=1.0, finished_at=2.0,
                 duration_s=1.0, outputs={"InputAll.csv": "/tmp/InputAll.csv"})
    write_progress(tmp_path, layer_nr=1, scan_mode="ff",
                   stage_names=STAGES, stages=store.all_stages())
    txt = (tmp_path / PROGRESS_FILENAME).read_text()
    assert "outputs so far" in txt
    assert "InputAll.csv" in txt
