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
