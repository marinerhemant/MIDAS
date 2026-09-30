"""Tests for streaming.snapshot_profile (synthetic data only)."""
import numpy as np

from midas_integrate_v2.streaming.snapshot_profile import (
    band_excess,
    fit_matrix_scale,
    ring_profile,
    windows_from_trace,
)

LAM = 0.124
D_REF = np.array([2.08, 1.80, 1.27, 1.085, 1.04])    # a generic reference set


def _image(scale, n=400, level=0.1, ring_amp=3.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    r_px = np.hypot(yy - n / 2, xx - n / 2)
    tth = np.degrees(np.arctan(r_px * 0.172 / 300.0))          # 300 mm, 172 um pixels: all rings on the image
    lam_img = np.full((n, n), level)
    for d in scale * D_REF:
        t = np.degrees(2 * np.arcsin(LAM / (2 * d)))
        lam_img += ring_amp * np.exp(-0.5 * ((tth - t) / 0.01) ** 2)
    return rng.poisson(lam_img).astype(float), np.ones((n, n), bool), tth


def test_median_would_rail_mean_does_not():
    img, ok, tth = _image(1.0, level=0.3, ring_amp=0.0)
    tc, prof = ring_profile(img, ok, tth, 0.05)
    b = np.floor((tth - np.nanmin(tth)) / 0.05).astype(int)
    big = [k for k in range(len(prof)) if (b == k).sum() >= 1000]
    assert len(big) >= 10
    medians = [np.median(img[b == k]) for k in big]
    assert set(np.unique(medians)) <= {0.0, 1.0}                 # the rail
    assert np.all(np.abs(prof[big] - 0.3) < 0.05)                # the mean tracks the level


def test_matrix_scale_recovered():
    for s_true in (1.000, 1.012):
        img, ok, tth = _image(s_true, seed=int(s_true * 1000))
        tc, prof = ring_profile(img, ok, tth, 0.002)
        fit = fit_matrix_scale(tc, prof, D_REF, LAM, window_deg=0.03)
        assert fit.n_rings >= 3
        assert abs(fit.scale - s_true) < 5e-4


def test_weak_but_significant_rings_are_fitted():
    """Attenuated series: rings far below an absolute 0.5 counts/px floor but many sigma
    above the profile noise must still be fitted (averaging 50 frames)."""
    img = sum(_image(1.006, level=0.06, ring_amp=0.15, seed=100 + k)[0] for k in range(50)) / 50
    _, ok, tth = _image(1.006)
    tc, prof = ring_profile(img, ok, tth, 0.002)
    fit = fit_matrix_scale(tc, prof, D_REF, LAM, window_deg=0.03)
    assert fit.n_rings >= 3 and abs(fit.scale - 1.006) < 1e-3
    assert np.isnan(fit_matrix_scale(tc, prof, D_REF, LAM, window_deg=0.03, min_peak=0.5).scale)


def test_matrix_scale_is_nan_without_rings():
    img, ok, tth = _image(1.0, ring_amp=0.0)
    tc, prof = ring_profile(img, ok, tth, 0.002)
    assert np.isnan(fit_matrix_scale(tc, prof, D_REF, LAM).scale)


def test_band_excess_and_windows_rule():
    ff = np.arange(0, 2000, 25)
    trace = np.where((ff >= 700) & (ff < 1300), 0.4, 0.01)
    w = windows_from_trace(ff, trace, 2000)
    assert w.onset == 700 and w.before == (0, 650) and w.after == (1400, 1999)
    assert "onset" in w.rule
    assert windows_from_trace(ff, np.full(len(ff), 0.01), 2000) is None
    tc = np.linspace(1, 10, 500)
    prof = np.where((tc > 3) & (tc < 4), 1.0, 0.2)
    assert abs(band_excess(tc, prof, (3.1, 3.9), (1.5, 2.5)) - 0.8) < 1e-9


def test_spread_separates_real_phase_from_neighbour_catching_pattern():
    """The true reference set gives consistent per-ring scales; a wrong set whose windows
    catch the true rings at different relative offsets does not."""
    img, ok, tth = _image(1.004, seed=11)
    tc, prof = ring_profile(img, ok, tth, 0.002)
    good = fit_matrix_scale(tc, prof, D_REF, LAM, window_deg=0.08)
    wrong = D_REF * np.array([1.0, 1.006, 0.995, 1.007, 0.994])   # same rings, scattered offsets
    bad = fit_matrix_scale(tc, prof, wrong, LAM, window_deg=0.08)
    assert good.consistent() and good.completeness == 1.0
    assert not bad.consistent()          # off-line rings dropped (or spread too large)


def _halo_profile(lines_deg, halo_c=3.4, halo_fwhm=0.5, halo_amp=30.0, ring_amp=0.0, seed=5):
    rng = np.random.default_rng(seed)
    tc = np.arange(1.0, 8.0, 0.005)
    prof = 2.0 + halo_amp * np.exp(-0.5 * ((tc - halo_c) / (halo_fwhm / 2.355)) ** 2)
    for t in lines_deg:
        prof = prof + ring_amp * np.exp(-0.5 * ((tc - t) / 0.01) ** 2)
    return tc, prof + rng.normal(0, 0.02, tc.size)


def test_halo_top_is_not_a_ring():
    """A reference whose lines sit on a broad liquid halo must not fit: the halo top rises
    above a straight line through the window edges, but not above a quadratic from the flanks."""
    d = np.array([2.10, 1.212])                         # first line on the halo top
    t = np.degrees(2 * np.arcsin(LAM / (2 * d)))
    tc, prof = _halo_profile([], halo_c=float(t[0]))
    fit = fit_matrix_scale(tc, prof, d, LAM, scale_grid=np.array([1.0]), window_deg=0.124, min_rings=1)
    assert fit.n_rings == 0
    tc, prof = _halo_profile(t, halo_c=float(t[0]), ring_amp=3.0)   # a sharp ring on the halo still counts
    fit = fit_matrix_scale(tc, prof, d, LAM, scale_grid=np.array([1.0]), window_deg=0.124, min_rings=1)
    assert 0 in fit.per_ring_scale and abs(fit.per_ring_scale[0] - 1) < 1e-3


def test_off_line_ring_is_dropped_not_averaged_away():
    """Three rings, one caught 1.5 % off: with a MAD spread it would pass; it must be dropped,
    leaving too few rings for a scale."""
    d = np.array([2.3, 2.0, 1.4])
    t = np.degrees(2 * np.arcsin(LAM / (2 * d * np.array([1.0, 1.0, 1.015]))))
    tc, prof = _halo_profile(t, halo_amp=0.0, ring_amp=3.0)
    fit = fit_matrix_scale(tc, prof, d, LAM, scale_grid=np.array([1.0]), window_deg=0.12)
    assert set(fit.per_ring_scale) == {0, 1} and np.isnan(fit.scale)
    fit4 = fit_matrix_scale(tc, prof, np.r_[d, 1.2], LAM, scale_grid=np.array([1.0]), window_deg=0.12, min_rings=2)
    assert abs(fit4.scale - 1.0) < 1e-3


def test_sharp_peaks_skip_halo():
    t = np.degrees(2 * np.arcsin(LAM / (2 * np.array([2.0, 1.5]))))
    from midas_integrate_v2.streaming.snapshot_profile import sharp_peaks
    tc, prof = _halo_profile(t, halo_c=5.0, ring_amp=2.0)
    pk = sharp_peaks(tc, prof)
    assert len(pk) == 2 and all(min(abs(p["tth"] - x) for x in t) < 0.01 for p in pk)
