"""Fixed-support integration against spots of known total intensity."""
from __future__ import annotations

import numpy as np
import pytest

from midas_sx.integrate import (ProfileModel, choose_support_scale, fit_profile_model, integrate_box)


def _volume(shape, spots, bg, rng, poisson=True):
    """Gaussian spots (frame, row, col, total, sig_f, sig_p) integrated over voxels, on a flat bg."""
    f, r, c = np.indices(shape, dtype=float)
    sig = np.zeros(shape)
    for (cf, cr, cc, tot, sf, sp) in spots:
        # integral of a Gaussian over each voxel = difference of erfs; use the exact cell integral
        from scipy.special import erf

        def cell(x, m, s):
            return 0.5 * (erf((x + 0.5 - m) / (np.sqrt(2) * s)) - erf((x - 0.5 - m) / (np.sqrt(2) * s)))
        sig += tot * cell(f, cf, sf) * cell(r, cr, sp) * cell(c, cc, sp)
    lam = sig + bg
    return (rng.poisson(lam) if poisson else lam).astype(np.float32), sig


def test_box_sum_recovers_total_when_noiseless():
    rng = np.random.default_rng(0)
    raw, sig = _volume((30, 40, 40), [(15, 20, 20, 8000.0, 1.3, 1.0)], 50.0, rng, poisson=False)
    res = integrate_box(raw, (15, 20, 20), (6, 6))
    assert res.ok and not res.clipped_frames and not res.clipped_detector
    # rel 1e-6, not 1e-9: the Gaussian's far tail leaks ~1e-6 counts/voxel into the background shell,
    # which the box then over-subtracts (2197 voxels * 1e-6 = 2.7e-3 counts out of 8000).
    assert res.intensity == pytest.approx(sig[9:22, 14:27, 14:27].sum(), rel=1e-6)
    assert res.intensity == pytest.approx(8000.0, rel=2e-3)          # box of 4.6 sigma_f, 6 sigma_p
    assert res.background == pytest.approx(50.0, rel=1e-6)


@pytest.mark.parametrize("total", [300.0, 3000.0])
def test_poisson_pull_is_unbiased_and_calibrated(total):
    """Over many noise draws: mean pull ~ 0 and sd(pull) ~ 1 (sigma is the honest Poisson one)."""
    pulls = []
    for seed in range(300):
        rng = np.random.default_rng(seed)
        raw, sig = _volume((26, 36, 36), [(13, 18, 18, total, 1.2, 1.0)], 40.0, rng)
        res = integrate_box(raw, (13, 18, 18), (5, 5))
        truth = sig[8:19, 13:24, 13:24].sum()
        pulls.append((res.intensity - truth) / res.sigma)
    pulls = np.array(pulls)
    assert abs(pulls.mean()) < 0.2                   # SE of the mean ~ 0.058
    assert 0.85 < pulls.std(ddof=1) < 1.2


def test_background_uses_a_robust_mean_not_a_biased_trim():
    """A neighbour inside the shell must not drag the estimate, and a plain Poisson shell must not be
    biased low (a percentile trim would be, and is multiplied by the box volume)."""
    rng = np.random.default_rng(1)
    raw, _ = _volume((26, 40, 40), [(13, 20, 20, 5000.0, 1.2, 1.0), (13, 20, 29, 20000.0, 1.2, 1.0)], 40.0, rng)
    res = integrate_box(raw, (13, 20, 20), (4, 4), shell=(2, 5))
    assert abs(res.background - 40.0) < 0.6
    flat = np.random.default_rng(2).poisson(40.0, (26, 40, 40)).astype(np.float32)
    bgs = [integrate_box(flat, (13, 20, 20), (4, 4)).background for _ in range(1)]
    assert abs(bgs[0] - 40.0) < 0.2


def test_clipping_flags():
    raw = np.full((10, 20, 20), 5.0, dtype=np.float32)
    assert integrate_box(raw, (1, 10, 10), (3, 3)).clipped_frames
    assert integrate_box(raw, (5, 2, 10), (3, 3)).clipped_detector
    r = integrate_box(raw, (5, 10, 10), (2, 2))
    assert not r.clipped_frames and not r.clipped_detector


def test_masked_voxels_are_counted_and_excluded():
    raw = np.full((10, 30, 30), 20.0, dtype=np.float32)
    mask = np.zeros((30, 30), bool)
    mask[14, 14] = True
    r = integrate_box(raw, (5, 15, 15), (2, 3), mask=mask)
    assert r.n_masked == 5                           # one pixel through 5 frames
    assert r.n_box == 5 * 7 * 7 - 5


def test_profile_model_recovers_the_widths_and_k_converges():
    rng = np.random.default_rng(3)
    spots, centres, Ls = [], [], []
    for i, L in enumerate([1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 2.5]):
        cr, cc = 20 + 32 * (i % 3), 20 + 32 * (i // 3)   # >= 32 px apart: a neighbour inside the shell biases the background
        sf = 0.6 + 0.25 * L
        spots.append((15, cr, cc, 40000.0, sf, 1.1))
        centres.append((15.0, float(cr), float(cc))); Ls.append(L)
    raw, _ = _volume((32, 100, 100), spots, 30.0, rng)
    model, info = fit_profile_model(raw, np.array(centres), np.array(Ls), window=(9, 8))
    assert info["n_strong"] == 7
    # point-sampled Gaussian fit to voxel-integrated data: widths carry +1/12 in quadrature
    assert model.sigma_px == pytest.approx(np.sqrt(1.1 ** 2 + 1 / 12), rel=0.03)
    assert model.sf0 == pytest.approx(0.6, abs=0.15) and model.sf1 == pytest.approx(0.25, abs=0.06)
    tuned, sweep = choose_support_scale(raw, np.array(centres), np.array(Ls), model)
    assert sweep["converged"] and 2.0 <= tuned.k <= 5.0


def test_profile_model_refuses_too_few_strong_spots():
    raw = np.random.default_rng(4).poisson(30.0, (20, 40, 40)).astype(np.float32)
    with pytest.raises(RuntimeError, match="strong"):
        fit_profile_model(raw, np.array([[10.0, 20.0, 20.0]]), np.array([2.0]))


def test_half_widths_grow_with_lorentz_factor():
    m = ProfileModel(sigma_px=1.0, sf0=0.5, sf1=0.2, k=3.5)
    assert m.half_widths(1.0)[0] < m.half_widths(10.0)[0]
    assert m.half_widths(3.0)[1] == m.half_widths(30.0)[1]


# ---------------------------------------------------------------- v2: measured extents (PREREGISTER A11)
def _flat_top_volume(shape, spots, bg, rng, poisson=True):
    """Spots whose omega profile is a flat top of half-width h_f frames (soft edges), Gaussian in-plane."""
    from scipy.special import erf
    f, r, c = np.indices(shape, dtype=float)
    sig = np.zeros(shape)

    def cell(x, m, s):
        return 0.5 * (erf((x + 0.5 - m) / (np.sqrt(2) * s)) - erf((x - 0.5 - m) / (np.sqrt(2) * s)))
    for (cf, cr, cc, tot, hflat, sp) in spots:
        w_f = 0.5 * (erf((f + 0.5 - cf + hflat) / (np.sqrt(2) * 0.6)) - erf((f + 0.5 - cf - hflat) / (np.sqrt(2) * 0.6)))
        w_f = w_f / w_f[:, 0, 0].sum() if False else w_f
        prof = np.array([0.5 * (erf((k + 0.5 - cf + hflat) / (np.sqrt(2) * 0.6)) - erf((k + 0.5 - cf - hflat) / (np.sqrt(2) * 0.6)))
                         for k in range(shape[0])])
        prof = prof / prof.sum()
        sig += tot * prof[:, None, None] * cell(r, cr, sp) * cell(c, cc, sp)
    lam = sig + bg
    return (rng.poisson(lam) if poisson else lam).astype(np.float32), sig


def test_v2_support_keeps_high_L_flat_top_spots_with_an_off_centre_blob_centroid():
    from midas_sx.integrate import build_profile
    rng = np.random.default_rng(5)
    Ls = [1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 24.0, 36.0, 48.0, 60.0, 3.0, 8.0]
    spots, cen = [], []
    for i, L in enumerate(Ls):
        cr, cc = 20 + 34 * (i % 4), 20 + 34 * (i // 4)
        spots.append((60, cr, cc, 60000.0, 0.12 * L + 0.6, 1.0))
        # the blob centroid of a skewed wide profile is NOT its centre; model that as an offset growing with L
        cen.append((60.0 + 0.04 * L, float(cr), float(cc)))
    raw, sig = _flat_top_volume((120, 140, 110), spots, 30.0, rng)
    cen, Ls = np.array(cen), np.array(Ls)
    # v1: Gaussian rule fitted on the low-L spots only (as the reduction did)
    m1, _ = fit_profile_model(raw, cen, Ls, window=(8, 8))
    m1, _ = choose_support_scale(raw, cen, Ls, m1)
    truth_total = 60000.0
    def loss(model):
        out = []
        for (cf, cr, cc), L in zip(cen, Ls):
            r = integrate_box(raw, (cf, cr, cc), model.half_widths(L), shell=(2, 3))
            out.append(r.intensity / truth_total)
        return np.array(out)
    m2, info = build_profile(raw, cen, Ls)
    l1, l2 = loss(m1), loss(m2)
    # (the old Gaussian rule is NOT asserted to fail here: on this synthetic it does not lose much, 98.8 % at worst; its real-frame
    # failure, 5-17 % at L = 11-62, is shown by validation/diag_integration_bias.py on the simulated S2 frames)
    assert l2.min() > 0.985 and abs(np.median(l2) - 1) < 0.01              # the measured rule keeps them, centroid offset included
    assert info["extents"]["L_max"] > 55 and info["extents"]["n_high_L"] >= 4


def test_v2_extent_window_grows_until_the_spot_does_not_fill_it():
    from midas_sx.integrate import measure_extents
    rng = np.random.default_rng(6)
    spots = [(70, 25, 25, 80000.0, 14.0, 1.0)]          # flat half-width 14 frames: wider than 0.8 x the first window (11 frames)
    for i in range(7):
        spots.append((70, 25 + 34 * ((i + 1) % 3), 25 + 34 * ((i + 1) // 3), 80000.0, 0.12 * (2 + i) + 0.6, 1.0))
    raw, _ = _flat_top_volume((140, 120, 130), spots, 30.0, rng)
    cen = np.array([(70.0, float(s[1]), float(s[2])) for s in spots])
    Ls = np.array([55.0] + [2.0 + i for i in range(7)])
    model, info = measure_extents(raw, cen, Ls)
    assert info["half_frames"][0] >= 14                                        # the first window (15) was filled, so it was grown and re-measured
    assert model.half_widths(55.0)[0] >= 12
