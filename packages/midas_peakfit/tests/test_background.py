"""Local background subtraction + the RingThresh calculator's primitives.

The load-bearing test here is ``test_preprocess_default_is_bit_identical``:
background subtraction is opt-in, and with ``BgSubtract 0`` (the default) the
production peak search must produce EXACTLY the bytes it did before, or every
existing reconstruction silently shifts.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_peakfit.background import (
    build_background_bins,
    estimate_cell_stats,
    local_background,
    subtract_local_background,
)
from midas_peakfit.preprocess import preprocess_frame
from midas_peakfit.ring_thresh import blob_snr, expected_false_blobs

N = 128
BC = N / 2.0


def _rt_eta(n=N, bc=BC):
    zz, yy = np.mgrid[0:n, 0:n]
    rt = np.sqrt((zz - bc) ** 2 + (yy - bc) ** 2)
    eta = np.degrees(np.arctan2(yy - bc, zz - bc))
    return rt, eta


def _bins(radii=(30.0,), width=5.0, n_sectors=8):
    rt, eta = _rt_eta()
    return build_background_bins(rt, eta, np.asarray(radii), width,
                                 n_sectors=n_sectors)


# ── binning ────────────────────────────────────────────────────────────────
def test_bins_cover_only_the_band():
    b = _bins()
    rt, _ = _rt_eta()
    in_band = (rt > 25.0) & (rt < 35.0)
    assert np.array_equal(b.in_band, in_band)
    assert b.labels[~in_band].max(initial=-1) == -1


def test_bins_split_into_sectors():
    b = _bins(n_sectors=8)
    used = np.unique(b.labels[b.in_band])
    assert len(used) == 8, "one cell per azimuthal sector expected"
    assert b.counts[used].min() > 0


def test_overlapping_bands_last_ring_wins():
    """Must match geometry.compute_good_coords, which has no ``break``."""
    b = _bins(radii=(30.0, 32.0), width=5.0, n_sectors=4)
    rt, _ = _rt_eta()
    both = (np.abs(rt - 30.0) < 5.0) & (np.abs(rt - 32.0) < 5.0)
    assert both.any(), "test needs genuinely overlapping bands"
    assert (b.labels[both] >= 4).all(), "later ring must win in the overlap"


def test_n_sectors_must_be_positive():
    rt, eta = _rt_eta()
    with pytest.raises(ValueError):
        build_background_bins(rt, eta, np.array([30.0]), 5.0, n_sectors=0)


# ── estimation ─────────────────────────────────────────────────────────────
def test_flat_field_gives_its_level_and_zero_sigma():
    b = _bins()
    img = np.full((N, N), 7.0)
    med, sig = estimate_cell_stats(img, b)
    live = b.counts > 0
    assert np.allclose(med[live], 7.0)
    assert np.allclose(sig[live], 0.0)


def test_sigma_recovers_gaussian_noise():
    b = _bins(n_sectors=4)
    rng = np.random.default_rng(0)
    img = rng.normal(100.0, 3.0, size=(N, N))
    med, sig = estimate_cell_stats(img, b)
    live = b.counts > 0
    assert np.median(med[live]) == pytest.approx(100.0, abs=0.6)
    assert np.median(sig[live]) == pytest.approx(3.0, rel=0.25)


def test_azimuthal_ramp_is_removed():
    """The whole point: a background that varies around the ring."""
    b = _bins(n_sectors=12)
    rt, eta = _rt_eta()
    img = np.zeros((N, N))
    img[b.in_band] = 500.0 * (eta[b.in_band] + 180.0) / 360.0   # 0..500 ramp
    before = img[b.in_band]
    resid, _ = subtract_local_background(img, b)
    after = resid[b.in_band]
    assert after.std() < 0.25 * before.std(), (
        f"ramp not flattened: {before.std():.1f} -> {after.std():.1f}")


def test_bright_spot_does_not_inflate_its_own_background():
    """A mean/std estimator would fail this; median/MAD must not."""
    b = _bins(n_sectors=4)
    img = np.zeros((N, N))
    img[b.in_band] = 10.0
    rt, _ = _rt_eta()
    spot = (np.abs(rt - 30.0) < 2.0)
    spot[:, : N // 2] = False
    img[spot] = 10000.0
    med, _ = estimate_cell_stats(img, b)
    live = b.counts > 0
    assert np.allclose(med[live], 10.0), "spot leaked into the background"


def test_spot_survives_subtraction():
    b = _bins(n_sectors=8)
    img = np.zeros((N, N))
    img[b.in_band] = 40.0
    img[30, int(BC)] = 40.0 + 900.0 if b.in_band[30, int(BC)] else 0.0
    resid, _ = subtract_local_background(img, b)
    if b.in_band[30, int(BC)]:
        assert resid[30, int(BC)] == pytest.approx(900.0, abs=1.0)


def test_out_of_band_untouched():
    b = _bins()
    rng = np.random.default_rng(1)
    img = rng.normal(0.0, 1.0, size=(N, N))
    resid, _ = subtract_local_background(img, b)
    assert np.array_equal(resid[~b.in_band], img[~b.in_band])


def test_thin_cells_fall_back_to_the_ring_median():
    """A band clipped by the detector edge must not get a wild median."""
    b = _bins(radii=(30.0,), width=5.0, n_sectors=8)
    thin = int(np.argmin(np.where(b.counts > 0, b.counts, 1 << 30)))
    b.counts[thin] = 3                       # pretend this cell is nearly empty
    img = np.zeros((N, N))
    img[b.in_band] = 25.0
    img[b.labels == thin] = 9999.0
    bg, _ = local_background(img, b, min_pixels=64)
    assert bg[b.labels == thin].max() == pytest.approx(25.0), \
        "thin cell should inherit the ring median, not its own"


# ── the parity guarantee ───────────────────────────────────────────────────
def _preprocess_kwargs(good_coords):
    return dict(
        NrPixels=N, NrPixelsY=N, NrPixelsZ=N, transform_options=[],
        dark=np.zeros((N, N)), flood=np.ones((N, N)),
        good_coords=good_coords, bc=1.0, bad_px_intensity=0.0, make_map=0,
    )


def test_preprocess_default_is_bit_identical():
    """BgSubtract 0 (bg_bins=None) must not perturb the legacy path at all."""
    rng = np.random.default_rng(7)
    raw = rng.normal(200.0, 10.0, size=(N, N))
    b = _bins()
    gc = np.where(b.in_band, 5.0, 0.0)
    a = preprocess_frame(raw, **_preprocess_kwargs(gc))
    c = preprocess_frame(raw, **_preprocess_kwargs(gc), bg_bins=None)
    assert np.array_equal(a, c)
    assert a.dtype == np.float64


def test_preprocess_with_bg_changes_result_and_lowers_survivors():
    """With a raised background, subtraction should stop it clearing threshold."""
    b = _bins(n_sectors=8)
    raw = np.zeros((N, N))
    raw[b.in_band] = 60.0                    # uniform background well over thr
    gc = np.where(b.in_band, 20.0, 0.0)      # threshold 20
    plain = preprocess_frame(raw, **_preprocess_kwargs(gc))
    subbed = preprocess_frame(raw, **_preprocess_kwargs(gc), bg_bins=b)
    assert (plain > 0).sum() > 0, "background alone should clear a raw threshold"
    assert (subbed > 0).sum() == 0, (
        "after removing a flat 60-count background nothing should clear 20")


# ── calculator primitives ──────────────────────────────────────────────────
def test_blob_snr_high_for_spot_low_for_noise():
    rng = np.random.default_rng(3)
    img = rng.normal(0.0, 1.0, size=(N, N))
    rows = np.array([64, 64, 65, 65])
    cols = np.array([64, 65, 64, 65])
    noise_snr = blob_snr(img, rows, cols)
    img[rows, cols] = 500.0
    spot_snr = blob_snr(img, rows, cols)
    assert spot_snr > 50.0
    assert noise_snr < 10.0
    assert spot_snr > noise_snr


def test_expected_false_blobs_falls_with_threshold():
    v = [expected_false_blobs(t, 5.0, 300_000, 1440) for t in (10, 20, 40, 80)]
    assert all(a > b for a, b in zip(v, v[1:])), v
    assert v[0] > v[-1] * 1e3


def test_expected_false_blobs_degenerate_inputs():
    assert expected_false_blobs(10.0, 0.0, 100, 10) == float("inf")
    assert expected_false_blobs(0.0, 5.0, 100, 10) == float("inf")
    assert expected_false_blobs(1e6, 5.0, 100, 10) == 0.0


def test_size_filter_makes_false_positives_rarer():
    """Requiring >=2 adjacent pixels must be far stricter than 1 pixel."""
    one = expected_false_blobs(30.0, 5.0, 300_000, 1440, min_n_px=0)
    two = expected_false_blobs(30.0, 5.0, 300_000, 1440, min_n_px=1)
    assert two < one


def test_blob_snr_needs_the_valid_mask_in_a_narrow_band():
    """Unrestricted annulus in a thin band is all zeros -> SNR collapses to 0."""
    b = _bins(radii=(30.0,), width=5.0, n_sectors=8)
    img = np.zeros((N, N))
    rng = np.random.default_rng(5)
    img[b.in_band] = rng.normal(20.0, 2.0, size=int(b.in_band.sum()))
    rt, _ = _rt_eta()
    blob = (np.abs(rt - 30.0) < 1.5) & (np.arange(N)[:, None] > 60) \
        & (np.arange(N)[:, None] < 64)
    rows, cols = np.where(blob & b.in_band)
    if rows.size == 0:
        pytest.skip("no blob pixels in band for this geometry")
    img[rows, cols] = 5000.0
    with_mask = blob_snr(img, rows, cols, valid=b.in_band)
    without = blob_snr(img, rows, cols)
    assert with_mask > 20.0, "in-band annulus should give a real SNR"
    assert with_mask > without


# ── per-spot SNR filter ────────────────────────────────────────────────────
def _region(rows, cols):
    from midas_peakfit.connected import Region
    return Region(id=1, pixel_rows=np.asarray(rows), pixel_cols=np.asarray(cols),
                  intensities=np.ones(len(rows)), raw_sum=1.0, threshold=0.0)


def _noisy_band(bins, level=100.0, sigma=4.0, seed=2):
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N))
    img[bins.in_band] = rng.normal(level, sigma, size=int(bins.in_band.sum()))
    return img


def test_region_snr_separates_spot_from_noise():
    from midas_peakfit.background import estimate_cell_stats, region_snr
    b = _bins(n_sectors=8)
    img = _noisy_band(b)
    rows, cols = np.where(b.in_band)
    r, c = rows[len(rows) // 2], cols[len(cols) // 2]
    med, sig = estimate_cell_stats(img, b)
    noise_snr = region_snr(_region([r], [c]), img, b, med, sig)
    img[r, c] = 100.0 + 40.0 * 4.0          # a 40-sigma peak
    med, sig = estimate_cell_stats(img, b)
    spot_snr = region_snr(_region([r], [c]), img, b, med, sig)
    assert spot_snr > 20.0, spot_snr
    assert abs(noise_snr) < 6.0, noise_snr


def test_region_snr_zero_outside_bands():
    from midas_peakfit.background import estimate_cell_stats, region_snr
    b = _bins()
    img = _noisy_band(b)
    med, sig = estimate_cell_stats(img, b)
    assert region_snr(_region([0], [0]), img, b, med, sig) == 0.0


def test_snr_filter_rejects_noise_keeps_spot():
    from midas_peakfit.background import filter_regions_by_snr
    b = _bins(n_sectors=8)
    img = _noisy_band(b)
    rows, cols = np.where(b.in_band)
    i1, i2 = len(rows) // 3, 2 * len(rows) // 3
    img[rows[i1], cols[i1]] = 100.0 + 40.0 * 4.0
    regs = [_region([rows[i1]], [cols[i1]]), _region([rows[i2]], [cols[i2]])]
    kept, snrs = filter_regions_by_snr(regs, img, b, min_snr=10.0)
    assert len(kept) == 1, [f"{s:.1f}" for s in snrs]
    assert snrs[0] > 10.0


def test_snr_filter_off_by_default_keeps_everything():
    """min_snr <= 0 must be a no-op -- MinPeakSNR defaults to 0."""
    from midas_peakfit.background import filter_regions_by_snr
    b = _bins(n_sectors=8)
    img = _noisy_band(b)
    rows, cols = np.where(b.in_band)
    regs = [_region([rows[k]], [cols[k]]) for k in (10, 200, 400)]
    kept, snrs = filter_regions_by_snr(regs, img, b, min_snr=0.0)
    assert len(kept) == len(regs)
    assert len(snrs) == len(regs)


def test_snr_filter_no_bins_is_a_noop():
    from midas_peakfit.background import filter_regions_by_snr
    regs = [_region([1], [1])]
    kept, snrs = filter_regions_by_snr(regs, np.zeros((N, N)), None, min_snr=99.0)
    assert kept == regs and snrs == [0.0]


def test_snr_would_keep_a_single_pixel_bright_spot():
    """The point of SNR over NImgs: a lone bright peak must survive."""
    from midas_peakfit.background import filter_regions_by_snr
    b = _bins(n_sectors=8)
    img = _noisy_band(b)
    rows, cols = np.where(b.in_band)
    r, c = rows[len(rows) // 2], cols[len(cols) // 2]
    img[r, c] = 100.0 + 500.0 * 4.0
    kept, snrs = filter_regions_by_snr([_region([r], [c])], img, b, min_snr=5.0)
    assert len(kept) == 1 and snrs[0] > 100.0
