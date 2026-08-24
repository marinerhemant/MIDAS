"""PF peakfit reports FRAMES aggregated across concurrent scans, not scans.

A scan count is too coarse to be useful: 19 scans against 12 workers left the
bar reading ``1/19`` for forty minutes and then jumping, the same
"everything in flight, nothing finished" failure the c-omp indexer has with
voxels. Each scan reports its own frames every 10, so the sum across scans is
both monotone and fine-grained (27360 frames for a 19-scan layer).
"""

from __future__ import annotations

from midas_pipeline.stages.peakfit import _aggregate_frames


def test_none_until_a_scan_reports():
    # Nothing has reported, so the frame denominator is unknowable and the
    # caller must fall back to counting scans.
    assert _aggregate_frames({}, {}, 19) is None


def test_denominator_extrapolates_from_the_first_report():
    # One scan reporting 10/1440 fixes the total at 19 * 1440 immediately,
    # rather than growing as scans join.
    assert _aggregate_frames({1: 10}, {1: 1440}, 19) == (10, 19 * 1440)


def test_sums_across_concurrent_scans():
    done = {1: 700, 2: 300, 3: 20}
    total = {1: 1440, 2: 1440, 3: 1440}
    assert _aggregate_frames(done, total, 19) == (1020, 27360)


def test_denominator_is_stable_as_more_scans_start():
    """The bar must not go backwards when a new scan joins.

    A denominator built from `sum(reported totals)` would grow as scans start,
    so a fixed done/total could shrink as a percentage. Anchoring on
    n_scans * per keeps it fixed.
    """
    _, t1 = _aggregate_frames({1: 700}, {1: 1440}, 19)
    _, t2 = _aggregate_frames({1: 700, 2: 5}, {1: 1440, 2: 1440}, 19)
    assert t1 == t2 == 27360


def test_completed_scans_are_charged_full_frames():
    # How the finally-block marks a finished or CACHED scan: without it a
    # resumed layer would show fresh scans diluting the total.
    done = {1: 1440, 2: 1440, 3: 60}
    total = {1: 1440, 2: 1440, 3: 1440}
    d, t = _aggregate_frames(done, total, 4)
    assert (d, t) == (2940, 5760)


def test_reaches_one_hundred_percent():
    done = {i: 1440 for i in range(1, 20)}
    total = {i: 1440 for i in range(1, 20)}
    d, t = _aggregate_frames(done, total, 19)
    assert d == t == 27360


def test_zero_frame_total_is_rejected():
    # A degenerate report must not produce a zero denominator downstream.
    assert _aggregate_frames({1: 0}, {1: 0}, 19) is None
