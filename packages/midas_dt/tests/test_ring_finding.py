"""Ring finding: the rolling baseline, and what a global one misses."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from midas_dt.rings import find_rings, rolling_baseline  # noqa: E402


def _profile(n=500, peaks=((100, 50.0), (200, 30.0), (300, 12.0), (400, 6.0)),
             seed=0):
    """Peaks on a background that FALLS STEEPLY with radius.

    That decay is the whole point: it is what makes a global-median threshold
    find the inner rings and miss the outer ones.
    """
    rng = np.random.default_rng(seed)
    r = np.arange(n, dtype=np.float64) + 60.0
    bg = 300.0 * np.exp(-(r - 60.0) / 120.0) + 2.0
    prof = bg.copy()
    for centre, amp in peaks:
        prof += amp * np.exp(-0.5 * ((r - (centre + 60.0)) / 2.5) ** 2)
    return r, prof + rng.normal(0, 0.4, n), bg


def test_rolling_baseline_tracks_a_falling_background():
    r, prof, bg = _profile()
    base = rolling_baseline(prof, 51)
    # Within a few percent of the true background almost everywhere.
    err = np.abs(base - bg) / np.maximum(bg, 1e-9)
    assert np.median(err) < 0.05, float(np.median(err))


def test_rolling_baseline_rejects_a_tiny_window():
    with pytest.raises(ValueError, match="window must be"):
        rolling_baseline(np.zeros(10), 2)


def test_finds_every_ring_including_the_faint_outer_ones():
    r, prof, _ = _profile()
    rings = find_rings(r, prof, min_snr=3.0)
    found = sorted(x.radius_px for x in rings)
    for expected in (160.0, 260.0, 360.0, 460.0):
        assert any(abs(f - expected) < 3.0 for f in found), (
            f"missed the ring at {expected}; found {found}"
        )


def test_a_global_median_baseline_would_miss_the_faint_ones():
    """The failure this module exists to prevent.

    With a global median the threshold sits far above the background at low R
    and below it at high R: the strong inner rings are found and the faint
    outer ones are lost, producing a short list that looks reasonable.
    """
    r, prof, _ = _profile()
    good = find_rings(r, prof, min_snr=3.0)

    globally = prof - np.median(prof)
    noise = float(np.std(globally))
    from scipy.signal import find_peaks
    idx, _ = find_peaks(globally, height=3.0 * noise, distance=3)

    assert len(good) > len(idx), (
        f"rolling baseline found {len(good)}, global found {len(idx)} -- "
        f"the global baseline was expected to find fewer"
    )


def test_min_snr_controls_sensitivity():
    r, prof, _ = _profile()
    assert len(find_rings(r, prof, min_snr=3.0)) >= len(
        find_rings(r, prof, min_snr=30.0))


def test_close_peaks_are_not_merged_when_separation_is_small():
    r, prof, _ = _profile(peaks=((200, 30.0), (208, 25.0)))
    merged = find_rings(r, prof, min_snr=3.0, min_separation_px=20.0)
    resolved = find_rings(r, prof, min_snr=3.0, min_separation_px=3.0)
    assert len(resolved) > len(merged), (
        "a wide min_separation silently merged a doublet, which turns two "
        "real rings into one mis-measured one"
    )


def test_rings_come_back_sorted_by_radius():
    r, prof, _ = _profile()
    rings = find_rings(r, prof, min_snr=3.0)
    assert [x.radius_px for x in rings] == sorted(x.radius_px for x in rings)


def test_max_rings_keeps_the_strongest():
    r, prof, _ = _profile()
    rings = find_rings(r, prof, min_snr=3.0, max_rings=2)
    assert len(rings) == 2
    assert min(x.snr for x in rings) > 5.0


def test_flat_profile_reports_nothing():
    r = np.arange(100.0)
    assert find_rings(r, np.ones_like(r), min_snr=3.0) == []


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="must match"):
        find_rings(np.arange(10.0), np.arange(9.0))
