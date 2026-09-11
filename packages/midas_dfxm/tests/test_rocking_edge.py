"""Edge-safe rocking centres (v3): centre delegates exactly to reduce_rocking(window='peak') (v1
and v2 both failed their own benchmarks trying to invent a better one -- see the module
docstring); this file tests the calibrated truncation bound and the margin-gated fit built on top
of it, plus that delegation itself."""
import numpy as np

from midas_dfxm.rocking import example_rocking_scan, reduce_rocking
from midas_dfxm.rocking_edge import calibrate_flank_ratio, edge_centres, edge_centres_scan

STEP = 0.04


def _gauss(centres, *, M=25, fwhm=2.5, amp=1000.0, bg=100.0, noise=10.0, seed=0):
    """Gaussian peaks at ``centres`` (in frame-index units) on a flat background, white noise."""
    x = 20.0 + STEP * np.arange(M)
    sig = fwhm / 2.3548
    idx = np.arange(M)[:, None]
    c = np.asarray(centres, float)[None]
    clean = bg + amp * np.exp(-0.5 * ((idx - c) / sig) ** 2)
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, noise, clean.shape), x


def test_centre_matches_reduce_rocking_window_peak_exactly():
    """The whole point of v3: don't reinvent this, delegate to it."""
    scan = example_rocking_scan("single", shape=(24, 24), n_points=101, seed=11)
    maps = reduce_rocking(scan, window="peak", split_half=False,
                          lit=np.ones(scan.frames.shape[1:], bool))
    r = edge_centres_scan(scan, fit="none", flank_ratio=1.5)
    d = np.abs(r.centre - maps.centre_deg) * 1000.0       # mdeg
    assert np.nanmax(d) < 1e-9, d.max()
    assert np.array_equal(np.nan_to_num(r.truncated_pkg), np.nan_to_num(maps.truncated))


def test_truncation_flags():
    centres = np.repeat([-1.0, 0.0, 1.0, 1.5, 5.0, 12.0, 19.0, 23.0, 24.0, 25.0], 50)
    I, x = _gauss(centres, seed=4)
    r = edge_centres(I, x, fit="none", flank_ratio=1.5)
    low, high = centres <= 1.5, centres >= 23.0
    assert r.truncated_low[low].all() and not r.truncated_low[~low].any()
    assert r.truncated_high[high].all() and not r.truncated_high[~high].any()


def test_bound_contains_the_complete_centroid_when_the_maximum_is_recorded():
    rng = np.random.default_rng(5)
    c = rng.uniform(0.5, 2.0, 600)
    I, x = _gauss(c, seed=6)
    r = edge_centres(I, x, fit="none", flank_ratio=1.5)
    truth = 20.0 + STEP * c
    assert r.truncated_low.all()
    assert np.mean((r.lower <= truth) & (truth <= r.upper)) >= 0.95


def test_bound_is_open_when_the_peak_can_lie_outside():
    c = np.linspace(-1.5, -0.2, 200)
    I, x = _gauss(c, seed=7)
    r = edge_centres(I, x, fit="none", flank_ratio=1.5)
    truth = 20.0 + STEP * c
    assert np.isneginf(r.lower).all() and np.all(r.upper >= truth)


def test_bound_collapses_where_nothing_is_cut():
    I, x = _gauss(np.repeat([6.0, 12.0, 18.0], 100), seed=8)
    r = edge_centres(I, x, fit="none", flank_ratio=1.5)
    assert not r.truncated.any() and np.allclose(r.lower, r.upper)


def _skewed_gauss(centres, *, M=50, sig_near=1.2, ratio=2.0, amp=1000.0, bg=100.0, noise=8.0, seed=0):
    """A single peak with a real, known-ratio asymmetry: sigma is ``sig_near`` on the low-index
    side and ``ratio * sig_near`` on the high-index side."""
    x = 20.0 + STEP * np.arange(M)
    idx = np.arange(M)[:, None]
    c = np.asarray(centres, float)[None]
    d = idx - c
    sig = np.where(d < 0, sig_near, ratio * sig_near)
    clean = bg + amp * np.exp(-0.5 * (d / sig) ** 2)
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, noise, clean.shape), x


def test_calibrate_flank_ratio_orders_by_the_true_asymmetry():
    """A more asymmetric curve should calibrate to a larger flank_ratio than a less asymmetric one.
    Exact recovery of the planted ratio is not asserted -- calibration targets a coverage bar, not
    the generating parameter -- only that it responds in the right direction."""
    rng = np.random.default_rng(12)
    c = rng.uniform(20.0, 30.0, 4000)
    results = {}
    for ratio_true in (1.0, 4.0):
        I, x = _skewed_gauss(c, ratio=ratio_true, seed=int(ratio_true * 10))
        ratio, report = calibrate_flank_ratio(I, x, target_coverage=0.90, max_drop=6)
        assert report["calibrated"] and report["n_closed"] > 200, report
        assert report["candidates"][0] <= ratio <= report["candidates"][-1]
        results[ratio_true] = (ratio, report)
    assert results[4.0][0] >= results[1.0][0], results
    assert results[4.0][1]["coverage_test"] is None or results[4.0][1]["coverage_test"] >= 0.5, results


def test_calibration_falls_back_when_too_few_closed_pixels():
    I, x = _gauss(np.full(30, 8.0), seed=13)
    ratio, report = calibrate_flank_ratio(I, x, min_closed=200)
    assert not report["calibrated"]
    assert ratio == report["candidates"][0]


def test_calibration_reports_when_the_target_is_not_reached():
    """A curve whose real asymmetry exceeds every candidate ratio should either calibrate to the
    largest candidate with an honest below-target coverage, or (if too few pixels qualify as
    'closed' at that extreme a skew) fall back cleanly -- either way it must not claim the target
    was reached when it was not."""
    rng = np.random.default_rng(21)
    c = rng.uniform(30.0, 40.0, 4000)
    I, x = _skewed_gauss(c, ratio=60.0, sig_near=0.6, M=80, seed=1)   # far past the widest candidate
    ratio, report = calibrate_flank_ratio(I, x, target_coverage=0.90, max_drop=6)
    if report["calibrated"]:
        assert ratio == report["candidates"][-1]
        assert report["target_reached"] is False
        assert report["coverage_calib"] is None or report["coverage_calib"] < 0.90
    else:
        assert ratio == report["candidates"][0]


def test_fit_extrapolates_a_single_peak_past_the_end():
    c = np.repeat([-1.0, -0.5, 0.0, 0.5], 40)
    I, x = _gauss(c, fwhm=3.5, seed=9)
    r = edge_centres(I, x, fit="all", flank_ratio=1.5, fit_min_margin=0)
    truth = 20.0 + STEP * c
    assert r.fit_ok.mean() >= 0.8
    assert np.median(np.abs(r.fit_centre[r.fit_ok] - truth[r.fit_ok])) / STEP < 0.15


def test_fit_does_not_attempt_a_peak_that_sits_on_the_end_frame():
    """A peak exactly on the terminal frame is a missing peak, not a missing flank, and should
    never be accepted."""
    c = np.full(200, 0.0)                                   # maximum on frame 0
    I, x = _gauss(c, fwhm=3.5, seed=14)
    r = edge_centres(I, x, fit="all", flank_ratio=1.5, fit_min_margin=1)
    assert not r.fit_ok.any()


def test_fit_is_not_accepted_for_a_curve_that_is_not_one_peak():
    M = 60
    idx = np.arange(M)[:, None]
    from math import erf
    verf = np.vectorize(erf)
    box = 0.5 * (verf((idx - 15.0) / np.sqrt(2)) - verf((idx - 45.0) / np.sqrt(2)))
    horns = 2.5 * (np.exp(-0.5 * ((idx - 16.0) / 1.0) ** 2) + np.exp(-0.5 * ((idx - 44.0) / 1.0) ** 2))
    rng = np.random.default_rng(10)
    I = 100.0 + 400.0 * np.repeat(box + horns, 30, axis=1) + rng.normal(0, 5.0, (M, 30))
    x = 20.0 + STEP * np.arange(M)
    r = edge_centres(I, x, fit="all", flank_ratio=1.5)
    assert not r.fit_ok.any()


def test_status_categorises_ok_recovered_undetermined():
    c = np.concatenate([np.repeat([12.0], 100),                 # ok: peak well inside
                        np.repeat([0.0], 100)])                  # cut: peak on the terminal frame
    I, x = _gauss(c, fwhm=2.5, seed=15)
    r = edge_centres(I, x, fit="all", flank_ratio=1.5, fit_min_margin=1)
    s = r.status
    assert set(np.unique(s)) <= {"ok", "recovered", "undetermined"}
    assert (s[:100] == "ok").all()
    assert (s[100:] == "undetermined").all()          # margin 0 -> fit_min_margin=1 rejects it
