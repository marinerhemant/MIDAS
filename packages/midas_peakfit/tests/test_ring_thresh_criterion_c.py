"""Criterion C: a threshold must not be so low that distinct spots merge.

A and B are detection criteria — is this blob real? Neither can see a blob that
is real but is several spots fused into one connected component, because
merging changes neither the blob's SNR nor the noise statistics, only what the
blob contains. On bt_1id_jun25b s1 that blind spot produced a recommendation of
20-30 and regions holding >=400 peaks (the maxNPeaks cap), against a healthy
1-15 on the other three samples of the same experiment.

These tests pin the picker logic and the combination rule. They deliberately do
NOT need a detector file: the sweep points are synthesised, so the tests state
what the criterion means rather than re-measuring one dataset.
"""

import pytest

from midas_peakfit.ring_thresh import (
    DEFAULT_P99_PEAKS_MAX,
    RingRecommendation,
    RingSweepPoint,
    _pick_best_resolved,
    _pick_merge,
)


def _pt(thr, p99, n_resolved=0.0, max_peaks=None):
    return RingSweepPoint(
        threshold=thr, n_blobs=0.0, n_kept=0.0, largest=0.0,
        median_snr=0.0, frac_snr_ok=0.0, expected_false_positives=0.0,
        n_resolved=n_resolved, frac_merged=0.0, p99_peaks=p99,
        max_peaks=p99 if max_peaks is None else max_peaks,
    )


def test_does_not_bind_on_uncrowded_data():
    """The no-regression test: on data that never merges, C must stay silent.

    s5/s2/s4 sit at 1-3 peaks per region at every threshold. If C bound there
    it would raise thresholds that were already correct.
    """
    sweep = [_pt(t, p99) for t, p99 in
             [(5, 1), (10, 1), (20, 2), (50, 1), (100, 1)]]
    assert _pick_merge(sweep, DEFAULT_P99_PEAKS_MAX) is None


def test_binds_where_regions_merge():
    """s1's shape: fine at high threshold, percolating at low."""
    sweep = [_pt(t, p99) for t, p99 in
             [(20, 390), (30, 120), (50, 8), (75, 2), (100, 1)]]
    assert _pick_merge(sweep, DEFAULT_P99_PEAKS_MAX) == 75


def test_requires_the_whole_upper_tail_to_be_clean():
    """One clean point below a dirty one must NOT be read as the floor.

    A single sweep point can come out clean by sampling luck; the floor is only
    meaningful if every HIGHER threshold is clean too.
    """
    sweep = [_pt(t, p99) for t, p99 in
             [(10, 2), (20, 300), (50, 2), (75, 1)]]
    # 10 looks clean in isolation, but 20 above it merges.
    assert _pick_merge(sweep, DEFAULT_P99_PEAKS_MAX) == 50


def test_returns_none_when_nothing_is_ever_clean():
    sweep = [_pt(t, 200) for t in (5, 10, 20)]
    assert _pick_merge(sweep, DEFAULT_P99_PEAKS_MAX) is None


def test_best_resolved_finds_the_interior_maximum():
    """Lower threshold gains real spots until percolation, then loses them."""
    sweep = [_pt(5, 300, n_resolved=10.0), _pt(20, 50, n_resolved=90.0),
             _pt(50, 3, n_resolved=140.0), _pt(75, 2, n_resolved=120.0),
             _pt(150, 1, n_resolved=40.0)]
    assert _pick_best_resolved(sweep) == 50


def test_best_resolved_none_when_nothing_resolves():
    assert _pick_best_resolved([_pt(5, 1), _pt(10, 1)]) is None


@pytest.mark.parametrize("a,b,c,want", [
    (30.0, 20.0, 75.0, 75.0),     # C is strictest -> C wins (the s1 case)
    (100.0, 20.0, 75.0, 100.0),   # A strictest    -> unchanged behaviour
    (30.0, 20.0, None, 30.0),     # C silent       -> old two-criterion answer
    (None, None, 50.0, 50.0),     # only C
    (None, None, None, None),     # nothing
])
def test_recommendation_is_the_strictest_of_the_three(a, b, c, want):
    rec = RingRecommendation(ring_nr=1, radius_px=100.0,
                             thresh_snr=a, thresh_fp=b, thresh_merge=c)
    assert rec.recommended == want


def test_c_can_only_raise_a_recommendation_never_lower_it():
    """C is a lower bound like the others, so adding it must never relax."""
    for a, b, c in [(30.0, 20.0, 5.0), (30.0, 20.0, 75.0), (30.0, 20.0, None)]:
        two = max(v for v in (a, b) if v is not None)
        rec = RingRecommendation(ring_nr=1, radius_px=1.0,
                                 thresh_snr=a, thresh_fp=b, thresh_merge=c)
        assert rec.recommended >= two


# --- C railed at the sweep ceiling (bt_20id_sep26, 2026-09-29) --------------------------------------
from midas_peakfit.ring_thresh import _merge_floor, format_recommendations

SWEEP = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)


def _railed_sweep():
    """p99 peaks per region stays above the limit at EVERY threshold up to 300 and is clean only at 500."""
    return [_pt(t, 40 if t < 500 else 1, n_resolved={30: 20.0, 50: 60.0, 75: 40.0}.get(t, 5.0)) for t in SWEEP]


def test_floor_supported_only_by_the_last_point_is_a_rail_not_a_floor():
    assert _merge_floor(_railed_sweep(), DEFAULT_P99_PEAKS_MAX) == (None, "railed")
    assert _pick_merge(_railed_sweep(), DEFAULT_P99_PEAKS_MAX) is None


def test_a_confirmed_floor_next_to_the_ceiling_is_still_a_floor():
    """Clean from 300 up (300 AND 500): the tail confirms 300, so it is a real floor."""
    sweep = [_pt(t, 40 if t < 300 else 1) for t in SWEEP]
    assert _merge_floor(sweep, DEFAULT_P99_PEAKS_MAX) == (300.0, "floor")


def test_status_of_the_other_cases_is_unchanged():
    assert _merge_floor([_pt(t, 1) for t in SWEEP], DEFAULT_P99_PEAKS_MAX) == (None, "none")
    assert _merge_floor([_pt(t, 200) for t in SWEEP], DEFAULT_P99_PEAKS_MAX) == (None, "never_clean")
    assert _merge_floor([_pt(t, p) for t, p in [(20, 390), (30, 120), (50, 8), (75, 2), (100, 1)]],
                        DEFAULT_P99_PEAKS_MAX) == (75.0, "floor")


def test_railed_recommendation_uses_best_resolved_never_the_ceiling():
    """The H5 50 um ring 2 case: A = B = 30, C railed at 500, best resolved 50 -> 50, not 500."""
    rec = RingRecommendation(ring_nr=2, radius_px=1.0, sweep=_railed_sweep(),
                             thresh_snr=30.0, thresh_fp=30.0, thresh_merge=None,
                             thresh_best_resolved=50.0, merge_railed=True)
    assert rec.recommended == 50.0


def test_railed_never_lowers_a_stricter_a_or_b():
    rec = RingRecommendation(ring_nr=2, radius_px=1.0, sweep=_railed_sweep(),
                             thresh_snr=100.0, thresh_fp=30.0, thresh_best_resolved=50.0, merge_railed=True)
    assert rec.recommended == 100.0


def test_railed_without_a_resolved_optimum_falls_back_to_a_and_b():
    rec = RingRecommendation(ring_nr=2, radius_px=1.0, sweep=_railed_sweep(),
                             thresh_snr=30.0, thresh_fp=20.0, thresh_best_resolved=None, merge_railed=True)
    assert rec.recommended == 30.0


def test_paste_block_says_railed_and_does_not_print_the_ceiling():
    rec = RingRecommendation(ring_nr=2, radius_px=1.0, sweep=_railed_sweep(),
                             thresh_snr=30.0, thresh_fp=30.0, thresh_best_resolved=50.0, merge_railed=True)
    text = format_recommendations([rec])
    assert "RAILED" in text
    paste = text.split("Paste into the parameter file:")[1]
    assert "RingThresh 2 50" in paste and "RingThresh 2 500" not in paste
    assert "railed" in paste.lower()


def test_unrailed_recommendation_is_untouched_by_the_flag():
    rec = RingRecommendation(ring_nr=1, radius_px=1.0, thresh_snr=30.0, thresh_fp=20.0,
                             thresh_merge=75.0, thresh_best_resolved=50.0, merge_railed=False)
    assert rec.recommended == 75.0
