"""Intra-stage progress: how far through the LIVE stage a run is.

peakfit was 88.5 % of a 2652-grain gamma reconstruction (62 min of 70), so
"peakfit, RUNNING, 41m" is most of what progress.txt ever said. These cover the
two sources that fix that: the in-process peakfit callback, and the c-omp
progress lines streamed off indexing/refinement stdout.
"""

from __future__ import annotations

import time

from midas_pipeline.progress import (StageProgress, format_sub,
                                     parse_progress_line)
from midas_pipeline.provenance import (PROGRESS_FILENAME, ProvenanceStore,
                                       write_progress)

STAGES = ["zip_convert", "hkl", "peakfit", "indexing", "refinement"]


# --- parsing the c-omp lines ------------------------------------------------

def test_parses_the_indexer_line():
    p = parse_progress_line("  progress: 18000/36163 voxels, 142.7 vox/s, elapsed 126.1s")
    assert p == {"done": 18000, "total": 36163, "unit": "voxels", "rate": 142.7}


def test_parses_the_refiner_line():
    p = parse_progress_line("  progress: 500/26662 seeds, 88.2 seeds/s, elapsed 5.7s")
    assert p["done"] == 500 and p["total"] == 26662 and p["unit"] == "seeds"


def test_ignores_unrelated_output():
    for line in ("Reading parameters from file: x.txt",
                 "nSpots = 590844",
                 "Finished. Mode=FF nVoxels=36163 block=[0,36163). Time: 264.9s.",
                 ""):
        assert parse_progress_line(line) is None


def test_zero_total_is_not_progress():
    """A divide-by-zero here would kill a run for the sake of a status line."""
    assert parse_progress_line("  progress: 0/0 voxels") is None


# --- the reporter -----------------------------------------------------------

def test_update_throttles_but_always_fires_on_the_last_tick():
    hits = []
    sp = StageProgress(on_change=lambda: hits.append(1), min_interval_s=60.0)
    sp.reset("peakfit")
    sp.update(10, 3600, "frames", 5.0)          # first: fires
    sp.update(20, 3600, "frames", 5.0)          # throttled
    sp.update(30, 3600, "frames", 5.0)          # throttled
    assert len(hits) == 1
    sp.update(3600, 3600, "frames", 5.0)        # final: must fire regardless
    assert len(hits) == 2


def test_reset_clears_the_previous_stages_counts():
    sp = StageProgress()
    sp.reset("peakfit")
    sp.update(1800, 3600, "frames")
    assert sp.snapshot()["done"] == 1800
    sp.reset("indexing")
    assert sp.snapshot() is None                # not peakfit's stale 1800


def test_a_raising_callback_cannot_kill_the_run():
    def boom():
        raise RuntimeError("progress sink exploded")
    sp = StageProgress(on_change=boom, min_interval_s=0.0)
    sp.reset("peakfit")
    sp.update(10, 100, "frames")                # must not raise


def test_feed_line_drives_updates():
    sp = StageProgress(min_interval_s=0.0)
    sp.reset("indexing")
    sp.feed_line("  progress: 9000/36163 voxels, 100.0 vox/s, elapsed 90.0s")
    snap = sp.snapshot()
    assert snap["done"] == 9000 and snap["unit"] == "voxels"


# --- rendering --------------------------------------------------------------

def test_format_includes_percent_and_eta():
    out = format_sub({"stage": "peakfit", "done": 900, "total": 3600,
                      "unit": "frames", "rate": 10.0})
    assert "900/3600 frames" in out and "25%" in out
    assert "10.0/s" in out
    assert "eta 4.5m" in out                    # 2700 remaining / 10 per s


def test_format_survives_a_missing_rate():
    out = format_sub({"stage": "x", "done": 1, "total": 2, "unit": "f", "rate": None})
    assert "50%" in out and "eta" not in out


# --- end to end into progress.txt -------------------------------------------

def test_sub_progress_reaches_the_progress_file(tmp_path):
    store = ProvenanceStore(tmp_path)
    store.record("peakfit", status="running", started_at=time.time() - 300)
    sub = {"stage": "peakfit", "done": 1200, "total": 3600,
           "unit": "frames", "rate": 4.0}
    write_progress(tmp_path, layer_nr=1, scan_mode="ff", stage_names=STAGES,
                   stages=store.all_stages(), sub=sub)
    txt = (tmp_path / PROGRESS_FILENAME).read_text()
    assert "1200/3600 frames" in txt
    assert "33%" in txt
    assert "eta" in txt


def test_sub_progress_for_a_different_stage_is_not_shown(tmp_path):
    """A stale snapshot must not be painted onto whatever is running now."""
    store = ProvenanceStore(tmp_path)
    store.record("indexing", status="running", started_at=time.time())
    write_progress(tmp_path, layer_nr=1, scan_mode="ff", stage_names=STAGES,
                   stages=store.all_stages(),
                   sub={"stage": "peakfit", "done": 5, "total": 10,
                        "unit": "frames", "rate": 1.0})
    assert "5/10" not in (tmp_path / PROGRESS_FILENAME).read_text()


def test_progress_file_is_fine_with_no_sub_progress(tmp_path):
    store = ProvenanceStore(tmp_path)
    store.record("peakfit", status="running", started_at=time.time())
    write_progress(tmp_path, layer_nr=1, scan_mode="ff", stage_names=STAGES,
                   stages=store.all_stages(), sub=None)
    assert "RUNNING" in (tmp_path / PROGRESS_FILENAME).read_text()
