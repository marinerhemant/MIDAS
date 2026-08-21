"""Gates on per-azimuth ring extraction.

Each test here corresponds to a specific way a real extraction went wrong. The
synthetic cakes are built to reproduce the *conditions* that caused each failure
-- peaks at a few percent above a background that varies in both R and eta, an 8
px FWHM, a doublet, dead azimuthal bins, and a peak that moves with azimuth --
rather than to be easy to pass.

The controlling fact throughout: the area is a **difference** of large numbers
and the centroid is a **ratio**, so at low contrast the centroid survives and the
area does not. Several tests assert exactly that asymmetry, because it is the
property that decides what a dataset can support.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_dt import azimuthal
from midas_dt.azimuthal import (
    area_and_centroid,
    azimuthal_rebin,
    background_from_ring_free,
    radial_half_correlation,
    count_maxima,
    extract_ring,
    mad_filter,
    refine_ring_centres,
    ring_free_mask,
    ring_windows,
    snr_per_eta,
    strain_from_centroid,
)

N_R, N_ETA = 600, 120
FWHM_BINS = 8.0
SIGMA = FWHM_BINS / 2.3548


def _gauss(r, centre, sigma, amp):
    return amp * np.exp(-0.5 * ((r - centre) / sigma) ** 2)


def make_cake(centres=(150.0, 300.0, 450.0), contrast=0.10, bg_level=220.0,
              bg_slope=-0.15, eta_bg_amp=0.25, shift_amp=0.0, seed=0,
              poisson=True, n_eta=N_ETA):
    """A cake with a realistic background: falling in R, varying in eta.

    ``contrast`` is peak amplitude / **local** background at each ring -- the
    number that governs everything, and defined against the background where the
    ring actually sits rather than against the R=0 intercept, so it matches what
    :attr:`RingExtraction.contrast` reports back. ``shift_amp`` moves the peak
    centre with azimuth as ``cos(2 eta)``, which is Singh's lattice-strain
    relation in miniature.
    """
    rng = np.random.default_rng(seed)
    r = np.arange(N_R, dtype=float)
    eta = np.linspace(-np.pi, np.pi, n_eta, endpoint=False)
    bg = (bg_level + bg_slope * r)[:, None] * (
        1.0 + eta_bg_amp * np.cos(eta)[None, :])
    ideal = np.zeros((N_R, n_eta))
    for c in centres:
        cen = c + shift_amp * np.cos(2 * eta)                # (n_eta,)
        local_bg = bg_level + bg_slope * c                   # eta-mean at the ring
        ideal += _gauss(r[:, None], cen[None, :], SIGMA, contrast * local_bg)
    total = bg + ideal
    if poisson:
        total = rng.poisson(np.clip(total, 0, None)).astype(float)
    return total, r, eta, bg, ideal


# ------------------------------------------------------------------ windows
def test_ring_windows_are_in_pixels_not_bins():
    """A cake binned finer than a pixel must still get a pixel-sized window.

    Writing the window in bins is a ~4x under-width at 0.25 px/bin, and the lost
    part is the tails, which is where the area lives.
    """
    r_fine = np.arange(0.0, 150.0, 0.25)                 # 0.25 px per bin
    idx, half = ring_windows(r_fine, [50.0, 100.0], max_half_px=16.0)
    # 16 px at 0.25 px/bin is 64 bins, not 16
    assert half[50.0] == pytest.approx(64, abs=1)
    r_coarse = np.arange(0.0, 150.0, 1.0)
    _, half_c = ring_windows(r_coarse, [50.0, 100.0], max_half_px=16.0)
    assert half_c[50.0] == pytest.approx(16, abs=1)


def test_ring_windows_never_overlap_a_neighbour():
    r = np.arange(N_R, dtype=float)
    centres = [150.0, 165.0, 450.0]                      # first two are close
    idx, half = ring_windows(r, centres, gap_frac=0.45)
    assert half[150.0] <= 0.45 * 15 + 6                  # bounded by the gap
    lo1, hi1 = idx[150.0] - half[150.0], idx[150.0] + half[150.0]
    lo2 = idx[165.0] - half[165.0]
    assert hi1 <= lo2 + 1 or half[150.0] == 6            # min width can force it


def test_ring_windows_never_run_off_the_axis():
    r = np.arange(N_R, dtype=float)
    idx, half = ring_windows(r, [3.0, 597.0], max_half_px=40.0)
    assert idx[3.0] - half[3.0] >= 0
    assert idx[597.0] + half[597.0] <= N_R - 1


def test_ring_windows_rejects_a_decreasing_axis():
    with pytest.raises(ValueError, match="increasing"):
        ring_windows(np.arange(100.0)[::-1], [50.0])


def test_ring_free_mask_widens_beyond_the_window():
    idx, half = {150.0: 150}, {150.0: 10}
    m = ring_free_mask(N_R, idx, half, widen=1.6)
    assert m[150 - 15] and m[150 + 15]                    # inside the widened band
    assert not m[150 - 20]


# --------------------------------------------------------------- background
def test_background_recovers_the_truth_under_the_peaks():
    """The whole point: masked-and-interpolated must not ride up on the flank."""
    cake, r, eta, bg_true, _ = make_cake(contrast=0.10, poisson=False)
    idx, half = ring_windows(r, [150.0, 300.0, 450.0])
    mask = ring_free_mask(N_R, idx, half)
    net, bg = background_from_ring_free(cake, mask, block_bins=30)
    at_peak = [idx[c] for c in (150.0, 300.0, 450.0)]
    rel = np.abs(bg[at_peak] - bg_true[at_peak]) / bg_true[at_peak]
    assert rel.max() < 0.03, f"background biased {rel.max()*100:.1f}% under a peak"


def test_background_tracks_variation_in_both_r_and_eta():
    """A radial-only background leaves an eta pattern that mimics texture."""
    cake, r, eta, bg_true, _ = make_cake(contrast=0.05, eta_bg_amp=0.4,
                                         poisson=False)
    idx, half = ring_windows(r, [150.0, 300.0, 450.0])
    net, bg = background_from_ring_free(cake, ring_free_mask(N_R, idx, half),
                                        block_bins=30)
    # correlation of the recovered background's eta profile with the true one
    prof_rec = bg.mean(axis=0)
    prof_true = bg_true.mean(axis=0)
    assert np.corrcoef(prof_rec, prof_true)[0, 1] > 0.99


def test_background_does_not_clip_the_net_at_zero():
    """Clipping here would bias the centroid; it is the caller's decision."""
    cake, r, _, _, _ = make_cake(contrast=0.02, seed=1)
    idx, half = ring_windows(r, [150.0, 300.0, 450.0])
    net, _ = background_from_ring_free(cake, ring_free_mask(N_R, idx, half),
                                        block_bins=30)
    assert net.min() < 0.0


def test_background_warns_when_there_is_too_little_ring_free_radius(caplog):
    cake, r, _, _, _ = make_cake(poisson=False)
    mask = np.ones(N_R, dtype=bool)                       # everything is "ring"
    mask[:10] = False
    with caplog.at_level("WARNING"):
        background_from_ring_free(cake, mask, block_bins=30)
    assert "ring-free" in caplog.text


def test_background_rejects_a_mismatched_mask():
    cake, r, _, _, _ = make_cake(poisson=False)
    with pytest.raises(ValueError, match="peak_mask has"):
        background_from_ring_free(cake, np.zeros(7, dtype=bool), block_bins=30)


# ----------------------------------------------------------------- vetting
def test_count_maxima_finds_a_singlet():
    r = np.arange(N_R, dtype=float)
    assert count_maxima(_gauss(r, 300.0, SIGMA, 100.0)) == 1


def test_count_maxima_catches_a_doublet():
    """The real case: hcp Ti (101), maxima 12 px apart, matched as one line."""
    r = np.arange(N_R, dtype=float)
    prof = _gauss(r, 294.0, SIGMA, 100.0) + _gauss(r, 306.0, SIGMA, 95.0)
    assert count_maxima(prof) == 2


def test_count_maxima_ignores_noise_below_half_height():
    r = np.arange(N_R, dtype=float)
    prof = _gauss(r, 300.0, SIGMA, 100.0) + _gauss(r, 260.0, SIGMA, 20.0)
    assert count_maxima(prof) == 1                        # 20% is below half


def test_count_maxima_on_an_empty_window_is_zero():
    assert count_maxima(np.zeros(50)) == 0
    assert count_maxima(np.full(50, np.nan)) == 0


# --------------------------------------------------------------------- SNR
def test_snr_rises_with_contrast():
    snrs = []
    for contrast in (0.02, 0.05, 0.20):
        cake, r, _, _, _ = make_cake(contrast=contrast, seed=5)
        idx, half = ring_windows(r, [150.0, 300.0, 450.0])
        net, _ = background_from_ring_free(cake, ring_free_mask(N_R, idx, half),
                                            block_bins=30)
        lo, hi = idx[300.0] - half[300.0], idx[300.0] + half[300.0] + 1
        snrs.append(float(np.median(snr_per_eta(cake[lo:hi], net[lo:hi]))))
    assert snrs == sorted(snrs), snrs


def test_snr_is_per_azimuth_not_a_scalar():
    """Gating on a median lets dead bins through; the spread then blows up."""
    cake, r, _, _, _ = make_cake(contrast=0.10, seed=7)
    cake[:, :20] = 1.0                                     # 20 dead azimuths
    idx, half = ring_windows(r, [150.0, 300.0, 450.0])
    net, _ = background_from_ring_free(cake, ring_free_mask(N_R, idx, half),
                                        block_bins=30)
    lo, hi = idx[300.0] - half[300.0], idx[300.0] + half[300.0] + 1
    s = snr_per_eta(cake[lo:hi], net[lo:hi])
    assert s.shape == (N_ETA,)
    assert np.median(s[:20]) < np.median(s[20:])           # dead bins are visible


# ------------------------------------------------------------- MAD filtering
def test_mad_filter_rejects_a_spotty_outlier():
    """Needs a baseline with real scatter -- see the degenerate case below."""
    rng = np.random.default_rng(2)
    x = 10.0 + rng.normal(scale=0.5, size=64)
    x[7] = 400.0                                           # one large crystallite
    keep = mad_filter(x)
    assert not keep[7]
    assert keep.sum() >= 60                                # the rest survive


def test_mad_filter_cannot_reject_from_a_noiseless_baseline():
    """A real property worth pinning down, not a limitation to route around.

    With identical values the MAD is exactly zero, so *every* deviation is
    infinitely many MADs and the filter would reject the entire window. It
    deliberately passes everything through instead. Synthetic tests that forget
    this conclude the filter is broken.
    """
    x = np.full(64, 10.0)
    x[7] = 400.0
    assert mad_filter(x).all()


def test_mad_filter_passes_a_degenerate_window_through():
    """Zero MAD must not empty the window."""
    assert mad_filter(np.full(20, 5.0)).all()
    assert mad_filter(np.zeros(20)).all()


# ----------------------------------------------------------- area vs centroid
def test_centroid_survives_low_contrast_where_area_does_not():
    """The asymmetry that decides what a dataset supports.

    Both are relative scatters over azimuth on a sample with **no** planted
    azimuthal structure, so everything measured is extraction error. Measured on
    this synthetic at 2 % contrast: area 36 %, centroid 0.85 %. The area is
    unusable for texture at a contrast where the centroid is still good to ~1 %,
    which is exactly what was seen on the real DAC Ti scan.
    """
    errs = {}
    for contrast in (0.50, 0.02):
        cake, r, eta, _, _ = make_cake(contrast=contrast, seed=11)
        idx, half = ring_windows(r, [150.0, 300.0, 450.0])
        net, _ = background_from_ring_free(cake, ring_free_mask(N_R, idx, half),
                                           block_bins=30)
        area, cen = area_and_centroid(net, r, idx[300.0], half[300.0])
        errs[contrast] = (float(np.std(area) / np.median(area)),
                          float(np.std(cen) / np.median(cen)))

    for contrast, (area_err, cen_err) in errs.items():
        assert cen_err < 0.1 * area_err, (
            f"contrast {contrast}: centroid scatter {cen_err:.2e} is not an order "
            f"of magnitude better than area {area_err:.2e} -- the "
            "ratio-vs-difference argument would be wrong")

    area_lo, cen_lo = errs[0.02]
    assert area_lo > 0.25, "area should be badly degraded at 2% contrast"
    assert cen_lo < 0.02, "centroid should still be good to ~1% at 2% contrast"
    # NOT asserted: that the area degrades faster in RELATIVE terms as contrast
    # falls. It does not -- measured 2.7x for the area against 3.3x for the
    # centroid -- because the area is already at 14 % scatter at high contrast and
    # has little headroom left. The claim that survives is about the absolute
    # magnitudes above, which is the one that decides usability.


def test_centroid_tracks_a_planted_azimuthal_peak_shift():
    """Singh's relation in miniature: d(psi) varies, and the centroid must see it."""
    shift = 3.0
    cake, r, eta, _, _ = make_cake(contrast=0.30, shift_amp=shift, seed=13,
                                   poisson=False)
    idx, half = ring_windows(r, [150.0, 300.0, 450.0])
    net, _ = background_from_ring_free(cake, ring_free_mask(N_R, idx, half),
                                        block_bins=30)
    _, cen = area_and_centroid(net, r, idx[300.0], half[300.0])
    planted = 300.0 + shift * np.cos(2 * eta)
    assert np.corrcoef(cen, planted)[0, 1] > 0.99
    assert np.ptp(cen) == pytest.approx(2 * shift, rel=0.15)


def test_a_moving_peak_in_a_narrow_window_fakes_azimuthal_intensity():
    """The trap this module exposes rather than fixes.

    A window too narrow to hold a shifting peak converts *movement* into
    *intensity*, i.e. fake texture.
    """
    cake, r, eta, _, _ = make_cake(contrast=0.50, shift_amp=8.0, seed=17,
                                   poisson=False)
    idx, _ = ring_windows(r, [150.0, 300.0, 450.0])
    net, _ = background_from_ring_free(
        cake, ring_free_mask(N_R, idx, {c: 20 for c in idx}), block_bins=30)
    narrow, _ = area_and_centroid(net, r, idx[300.0], 4)
    wide, _ = area_and_centroid(net, r, idx[300.0], 30)
    narrow_var = float(np.std(narrow) / np.mean(narrow))
    wide_var = float(np.std(wide) / np.mean(wide))
    assert narrow_var > 5 * wide_var                       # the fake signal


def _half_corr_for(shift_amp, amp_mod, contrast=0.26, seed=3):
    """Half-correlation for a ring that is moving and/or changing amplitude."""
    rng = np.random.default_rng(seed)
    r = np.arange(N_R, dtype=float)
    eta = np.linspace(-np.pi, np.pi, N_ETA, endpoint=False)
    bg = (220.0 - 0.15 * r)[:, None] * (1 + 0.25 * np.cos(eta))[None, :]
    cake = bg.copy()
    for c in (150.0, 300.0, 450.0):
        cen = c + shift_amp * np.cos(2 * eta)
        amp = contrast * 175.0 * (1 + amp_mod * np.cos(2 * eta))
        cake += amp[None, :] * np.exp(
            -0.5 * ((r[:, None] - cen[None, :]) / SIGMA) ** 2)
    cake = rng.poisson(cake).astype(float)
    idx, half = ring_windows(r, [150.0, 300.0, 450.0])
    net, _ = background_from_ring_free(cake, ring_free_mask(N_R, idx, half),
                                       block_bins=30)
    lo = idx[300.0] - half[300.0]
    hi = idx[300.0] + half[300.0] + 1
    return radial_half_correlation(net[lo:hi])


def test_half_correlation_goes_negative_for_a_moving_peak():
    """The measured discriminator: −0.72 on a real CeO2 standard.

    Intensity leaves one radial half as it enters the other, so the halves
    anti-correlate. Works **with** Poisson noise, which is the whole point — it is
    a ratio over many bins, not an edge level.
    """
    assert _half_corr_for(shift_amp=6.0, amp_mod=0.0) < -0.25
    assert _half_corr_for(shift_amp=3.0, amp_mod=0.0) < 0.0


def test_half_correlation_stays_positive_for_amplitude_modulation():
    """What a genuine pole figure looks like: position holds, amplitude varies."""
    assert _half_corr_for(shift_amp=0.0, amp_mod=0.4) > 0.4
    assert _half_corr_for(shift_amp=0.0, amp_mod=0.2) > 0.2


def test_half_correlation_separates_movement_from_amplitude_by_SIGN():
    moving = _half_corr_for(shift_amp=6.0, amp_mod=0.0)
    amplitude = _half_corr_for(shift_amp=0.0, amp_mod=0.4)
    assert moving < 0.0 < amplitude
    assert amplitude - moving > 0.8


def test_half_correlation_needs_enough_bins_to_split():
    with pytest.raises(ValueError, match="at least 4"):
        radial_half_correlation(np.zeros((3, 10)))


def test_half_correlation_on_a_degenerate_ring_is_nan():
    assert np.isnan(radial_half_correlation(np.zeros((10, 20))))


# ------------------------------------------------------------- extract_ring
def test_extract_ring_reports_gates_and_a_multiplet():
    cake, r, eta, _, _ = make_cake(contrast=0.20, seed=19)
    idx, half = ring_windows(r, [150.0, 300.0, 450.0])
    net, bg = background_from_ring_free(cake, ring_free_mask(N_R, idx, half),
                                         block_bins=30)
    ext = extract_ring(cake, net, bg, r, idx[300.0], half[300.0])
    assert ext.is_singlet and ext.usable(min_snr=3.0)
    assert ext.contrast == pytest.approx(0.20, rel=0.35)
    assert ext.area.shape == (N_ETA,) and ext.snr.shape == (N_ETA,)
    assert "SINGLET" in ext.describe()

    # now a genuine doublet at the same place
    r2 = np.arange(N_R, dtype=float)
    bg2 = np.full((N_R, N_ETA), 200.0)
    dbl = bg2 + (_gauss(r2, 294.0, SIGMA, 60.0) +
                 _gauss(r2, 306.0, SIGMA, 55.0))[:, None]
    net2, bgm = background_from_ring_free(
        dbl, ring_free_mask(N_R, {300.0: 300}, {300.0: 20}), block_bins=30)
    ext2 = extract_ring(dbl, net2, bgm, r2, 300, 20)
    assert not ext2.is_singlet and ext2.n_maxima == 2
    assert not ext2.usable()
    assert "multiplet" in ext2.describe()


def test_live_mask_is_stricter_than_usable():
    cake, r, _, _, _ = make_cake(contrast=0.10, seed=23)
    cake[:, :30] = 1.0                                     # dead azimuths
    idx, half = ring_windows(r, [150.0, 300.0, 450.0])
    net, bg = background_from_ring_free(cake, ring_free_mask(N_R, idx, half),
                                         block_bins=30)
    ext = extract_ring(cake, net, bg, r, idx[300.0], half[300.0])
    assert ext.usable(min_snr=3.0)                         # median is fine
    assert not ext.live_mask(min_snr=3.0)[:30].all()       # but these are not


# ------------------------------------------------------------------ rebinning
def test_azimuthal_rebin_averages_and_drops_the_remainder():
    x = np.arange(10.0)
    out = azimuthal_rebin(x, 3)
    assert out.shape == (3,)                               # last point dropped
    assert out[0] == pytest.approx(1.0)


def test_azimuthal_rebin_honours_a_keep_mask():
    keep = np.array([True, False, True, True])
    out = azimuthal_rebin(np.array([1.0, 100.0, 1.0, 1.0]), 4, keep=keep)
    assert out[0] == pytest.approx(1.0)                    # outlier excluded


def test_azimuthal_rebin_with_nothing_kept_returns_zero():
    out = azimuthal_rebin(np.arange(4.0), 4, keep=np.zeros(4, dtype=bool))
    assert out[0] == 0.0


def test_azimuthal_rebin_quadrature_combines_uncertainties():
    sig = np.full(4, 2.0)
    out = azimuthal_rebin(sig, 4, how="quadrature")
    assert out[0] == pytest.approx(np.sqrt(4 * 4.0) / 4)


def test_azimuthal_rebin_rejects_a_too_large_factor():
    with pytest.raises(ValueError, match="exceeds"):
        azimuthal_rebin(np.zeros(4), 8)


def test_azimuthal_rebin_works_on_stacked_axes():
    x = np.zeros((3, 5, 12))
    assert azimuthal_rebin(x, 4).shape == (3, 5, 3)


# --------------------------------------------------------------------- strain
def test_strain_from_centroid_is_relative_by_default():
    r = np.arange(100.0, 400.0)
    tt = np.linspace(4.0, 12.0, r.size)                    # degrees
    cen = np.full(64, 250.0)
    cen[:8] += 1.0
    eps = strain_from_centroid(cen, r, tt, 0.17297867)
    assert np.median(eps) == pytest.approx(0.0, abs=1e-12)  # median is the ref
    assert eps[:8].mean() < 0                              # larger 2theta, smaller d


def test_strain_from_centroid_accepts_an_explicit_reference():
    r = np.arange(100.0, 400.0)
    tt = np.linspace(4.0, 12.0, r.size)
    cen = np.full(8, 250.0)
    d0 = 0.17297867 / (2 * np.sin(np.radians(np.interp(250.0, r, tt)) / 2))
    eps = strain_from_centroid(cen, r, tt, 0.17297867, reference_d=d0 * 1.001)
    assert eps.mean() == pytest.approx(-1e-3, rel=1e-3)


def test_strain_from_centroid_rejects_all_nan_centroids():
    r = np.arange(100.0, 400.0)
    tt = np.linspace(4.0, 12.0, r.size)
    with pytest.raises(ValueError, match="reference d-spacing"):
        strain_from_centroid(np.full(8, np.nan), r, tt, 0.17297867)


def test_strain_magnitude_is_plausible_for_a_planted_shift():
    """A 1-px shift at a typical DT geometry is thousands of microstrain."""
    r = np.arange(100.0, 400.0)
    tt = np.linspace(4.0, 12.0, r.size)
    cen = np.array([250.0, 251.0])
    eps = strain_from_centroid(cen, r, tt, 0.17297867)
    assert 1e-4 < abs(eps[1] - eps[0]) < 1e-1


# --------------------------------------------------------------------------
# Window centring. An off-centre radial window converts peak WIDTH changes into
# apparent CENTROID changes -- i.e. into strain that is not there -- and because
# the offset is a fixed fraction of the window it does NOT dilute as the window
# widens, so a window-width sweep cannot detect it. Measured on an APS 1-ID DAC
# Ti scan: catalogue-derived centres were off by up to 1.524 px (19% of that
# ring's FWHM); re-centring moved one ring's apparent strain by 55% and flipped
# another ring's sign, while the ring already centred to +0.044 px was unchanged.
# --------------------------------------------------------------------------
def _gauss_cake(n_r=400, n_eta=24, centre=200.0, sigma=8.0, amp=1000.0):
    r = np.arange(n_r, dtype=float)
    prof = amp * np.exp(-0.5 * ((r - centre) / sigma) ** 2)
    return np.repeat(prof[:, None], n_eta, axis=1)


def test_centre_offset_is_reported_even_when_not_corrected():
    """The defect must be visible without the caller opting in."""
    net = _gauss_cake(centre=200.0)
    bg = np.ones_like(net)
    r_axis = np.arange(net.shape[0], dtype=float)
    res = azimuthal.extract_ring(net + bg, net, bg, r_axis,
                                 centre_bin=194, half_bins=40)
    assert res.centre_offset_bins == pytest.approx(6.0, abs=0.2)
    assert "OFFSET" in res.describe()


def _fake_shift(offset_bins, half_bins, sigma=8.0, grow=1.45):
    """Centroid shift produced by a pure WIDTH change through an off-centre window."""
    r_axis = np.arange(400, dtype=float)
    bg = np.ones((400, 24))
    narrow = _gauss_cake(centre=200.0, sigma=sigma)
    wide = _gauss_cake(centre=200.0, sigma=sigma * grow)
    c = 200 - offset_bins
    a = azimuthal.extract_ring(narrow + bg, narrow, bg, r_axis, c, half_bins)
    b = azimuthal.extract_ring(wide + bg, wide, bg, r_axis, c, half_bins)
    return float(np.nanmedian(b.centroid) - np.nanmedian(a.centroid))


def test_centred_window_is_immune_to_a_pure_width_change():
    """A width change with the window ON the peak must not move the centroid."""
    assert _fake_shift(0, 40) == pytest.approx(0.0, abs=0.02)
    assert _fake_shift(0, 20) == pytest.approx(0.0, abs=0.02)


def test_offcentre_artefact_grows_steeply_as_the_window_narrows():
    """Magnitude matters, and it is NOT a large effect for a wide window.

    Measured here (sigma=8, +45% width change, offsets 0.38-1.5 sigma):

        half/sigma 5.00  ->  0.02-0.22 bins
        half/sigma 2.50  ->  0.72-2.38 bins
        half/sigma 1.75  ->  0.82-2.42 bins

    So an off-centre window is a real but SECOND-ORDER error where the window is
    comfortably wide, and a first-order one near or below the ~2.3x FWHM safety
    line. On the DAC Ti S1 rings (half/sigma 2.96-6.73) this mechanism accounts
    for the ring that moved most on re-centring and NOT for the one that moved
    585 ue -- it is 1-2 orders too small there, so non-Gaussian content in the
    window (background residual, neighbour tails), not Gaussian tail truncation,
    is what makes re-centring matter on real data. Do not over-claim it.
    """
    wide_window = abs(_fake_shift(6, 40))       # half/sigma = 5.0
    narrow_window = abs(_fake_shift(6, 20))     # half/sigma = 2.5
    assert wide_window < 0.15, f"wide-window artefact should be small, got {wide_window}"
    assert narrow_window > 5 * wide_window, (
        f"artefact should grow steeply as the window narrows: "
        f"{narrow_window} vs {wide_window}")


def test_recentre_removes_the_offcentre_artefact():
    r_axis = np.arange(400, dtype=float)
    bg = np.ones((400, 24))
    narrow = _gauss_cake(centre=200.0, sigma=8.0)
    wide = _gauss_cake(centre=200.0, sigma=11.6)
    fake = abs(_fake_shift(6, 20))
    a = azimuthal.extract_ring(narrow + bg, narrow, bg, r_axis, 194, 20,
                               recentre=True)
    b = azimuthal.extract_ring(wide + bg, wide, bg, r_axis, 194, 20,
                               recentre=True)
    fixed = abs(float(np.nanmedian(b.centroid) - np.nanmedian(a.centroid)))
    assert fixed < fake / 3.0, f"recentre did not help: {fixed} vs {fake}"


def test_refine_ring_centres_finds_the_peak_and_refuses_a_wild_shift():
    net = _gauss_cake(centre=207.0)
    r_axis = np.arange(net.shape[0], dtype=float)
    index, half = {207.0: 200}, {207.0: 40}
    out = azimuthal.refine_ring_centres(net, r_axis, index, half)
    assert out[207.0] == pytest.approx(207, abs=1)

    # a shift beyond max_shift_frac is refused, not silently applied
    index2, half2 = {207.0: 180}, {207.0: 10}
    out2 = azimuthal.refine_ring_centres(net, r_axis, index2, half2,
                                         max_shift_frac=0.5)
    assert out2[207.0] == 180


def test_recentre_default_is_off_so_existing_numbers_do_not_move():
    net = _gauss_cake(centre=200.0)
    bg = np.ones_like(net)
    r_axis = np.arange(net.shape[0], dtype=float)
    off = azimuthal.extract_ring(net + bg, net, bg, r_axis, 194, 40)
    assert off.centre_bin == 194          # unchanged unless asked
