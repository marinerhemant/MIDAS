"""integrate_box(background=...): the shell-mean estimator is exact for a flat background only; the quadratic shell fit is exact for a background that
is quadratic in the detector plane and linear in omega, has calibrated variance, is not biased on a flat background, and keeps a neighbouring spot out."""
from __future__ import annotations

import numpy as np
import pytest

from midas_sx.integrate import BACKGROUNDS, DEFAULT_BACKGROUND, ProfileModel, integrate_box

SHAPE = (40, 80, 80)
CENTRE = (20, 40, 40)
HALF = (6, 7)


def curved(a_r=0.012, a_c=0.008, a_rc=0.004, g_f=0.1, b0=20.0):
    f, r, c = np.indices(SHAPE).astype(float)
    r, c = r - CENTRE[1], c - CENTRE[2]
    return b0 + a_r * r * r + a_c * c * c + a_rc * r * c + g_f * (f - CENTRE[0]) + 0.05 * r


def test_registered_choices():
    assert BACKGROUNDS == ("mean", "quadratic")
    assert DEFAULT_BACKGROUND == "mean"                      # PREREGISTER A36: flips only if T1, T2 and T3 are all met (T1 read INCONCLUSIVE)
    raw = curved()
    a, b = integrate_box(raw, CENTRE, HALF), integrate_box(raw, CENTRE, HALF, background="mean")
    assert (a.intensity, a.sigma, a.background, a.background_kind) == (b.intensity, b.sigma, b.background, "mean")
    assert ProfileModel(sigma_px=1, sf0=1, sf1=0).background == DEFAULT_BACKGROUND
    with pytest.raises(ValueError):
        integrate_box(curved(), CENTRE, HALF, background="plane")


def test_quadratic_is_exact_on_a_curved_noise_free_background_and_the_mean_is_not():
    bg = curved()
    q = integrate_box(bg, CENTRE, HALF, background="quadratic")
    m = integrate_box(bg, CENTRE, HALF, background="mean")
    assert q.background_kind == "quadratic" and m.background_kind == "mean"
    assert abs(q.intensity) < 1e-6 * q.n_box, q.intensity
    assert m.intensity < -20.0, m.intensity                       # the mean over-subtracts: shell mean > box mean for this curvature


def test_the_mean_deficit_grows_with_the_box_volume_and_the_quadratic_does_not():
    bg = curved()
    mean_def, quad_def = [], []
    for hp in (3, 5, 7, 9):
        mean_def.append(integrate_box(bg, CENTRE, (6, hp), background="mean").intensity)
        quad_def.append(integrate_box(bg, CENTRE, (6, hp), background="quadratic").intensity)
    assert all(a > b for a, b in zip(mean_def, mean_def[1:])), mean_def          # more and more negative
    assert max(abs(x) for x in quad_def) < 1e-5, quad_def


def test_flat_background_is_unbiased_with_calibrated_variance():
    """A Poisson flat background of 12 counts/voxel: no signal, so I should be 0 +- sigma. The 1/y-weighted fit would bias this by -1 count per voxel."""
    rng = np.random.default_rng(3)
    z, bg_mean = [], []
    for _ in range(300):
        raw = rng.poisson(12.0, size=SHAPE).astype(float)
        r = integrate_box(raw, CENTRE, HALF, background="quadratic")
        assert r.background_kind == "quadratic"
        z.append(r.intensity / r.sigma)
        bg_mean.append(r.background)
    z = np.array(z)
    assert abs(np.mean(bg_mean) - 12.0) < 0.02, np.mean(bg_mean)                    # per-voxel background, no -1 count bias
    assert abs(z.mean()) < 3 * z.std() / np.sqrt(len(z)) + 0.02, z.mean()
    assert 0.85 < z.std() < 1.2, z.std()


def test_curved_background_with_noise_is_calibrated():
    rng = np.random.default_rng(4)
    bg = curved()
    z = []
    for _ in range(200):
        r = integrate_box(rng.poisson(bg).astype(float), CENTRE, HALF, background="quadratic")
        z.append(r.intensity / r.sigma)
    z = np.array(z)
    assert abs(z.mean()) < 0.3 and 0.8 < z.std() < 1.25, (z.mean(), z.std())


def test_a_neighbouring_spot_at_the_shell_edge_is_mostly_clipped():
    """A bright spot whose tail overlaps the shell shifts the result by less than 10 % of its own integrated intensity (about 5 % measured; the shell mean
    shifts by about 3 %: the polynomial extrapolates, so it is somewhat more sensitive than the mean to unclipped tails; reduce_spots keeps crowded spots out)."""
    bg = curved()
    f, r, c = np.indices(SHAPE).astype(float)
    spot = 400.0 * np.exp(-0.5 * (((f - 20) / 1.0) ** 2 + ((r - 40) / 1.2) ** 2 + ((c - 40 - 11.0) / 1.2) ** 2))      # centred 11 px away, box half-width 7, shell to 10
    total = 400.0 * (2 * np.pi) ** 1.5 * 1.0 * 1.2 * 1.2
    shift = []
    for seed in range(60):
        a = integrate_box(np.random.default_rng(seed).poisson(bg + spot).astype(float), CENTRE, HALF, background="quadratic").intensity
        b = integrate_box(np.random.default_rng(seed).poisson(bg).astype(float), CENTRE, HALF, background="quadratic").intensity
        shift.append(a - b)
    assert abs(np.mean(shift)) < 0.10 * total, (np.mean(shift), total)


def test_fallback_to_the_mean():
    raw = np.random.default_rng(1).poisson(10.0, size=SHAPE).astype(float)
    assert integrate_box(raw, (1, 40, 40), HALF, background="quadratic").background_kind == "mean"          # clipped in frames
    assert integrate_box(raw, (20, 3, 40), HALF, background="quadratic").background_kind == "mean"          # clipped on the detector


def test_a_shell_too_small_for_seven_terms_returns_none():
    from midas_sx.integrate import _fit_shell
    rng = np.random.default_rng(2)
    n = 30                                                                                                   # < 5 voxels per term x 7 terms
    xyz = tuple(rng.integers(0, 20, n).astype(float) for _ in range(3))
    assert _fit_shell(xyz, rng.poisson(10.0, n).astype(float), xyz, CENTRE, HALF) is None
    xyz = tuple(np.zeros(60) for _ in range(3))                                                              # singular: all voxels at one point
    assert _fit_shell(xyz, np.full(60, 10.0), xyz, CENTRE, HALF) is None


def test_masked_voxels_are_excluded_from_the_fit():
    bg = curved()
    mask = np.zeros(SHAPE[1:], bool)
    mask[33:36, 30:60] = True
    bg_bad = bg.copy()
    bg_bad[:, mask] = 1e6                                                                                    # garbage under the mask must not reach the fit
    r = integrate_box(bg_bad, CENTRE, HALF, mask=mask, background="quadratic")
    assert r.background_kind == "quadratic" and abs(r.intensity) < 1e-3 * r.n_box


def test_low_count_background_is_unbiased():
    """1 count/voxel, large box: the Gaussian clip fit + 4 sqrt(fit) cut 6e-4 of the shell and biased the intensity high (z mean +0.3 at (6,7), +0.6 at (10,12),
    found by the statistics lens of /verify claim 97f96459b1e5). The Poisson-quantile clip leaves it within noise."""
    shape, centre = (36, 90, 90), (18, 45, 45)
    for lam, half in ((1.0, (6, 7)), (1.0, (10, 12)), (3.0, (10, 12))):
        z = []
        for seed in range(300):
            raw = np.random.default_rng(seed).poisson(lam, size=shape).astype(float)
            r = integrate_box(raw, centre, half, background="quadratic")
            assert r.background_kind == "quadratic"
            z.append(r.intensity / r.sigma)
        z = np.array(z)
        assert abs(z.mean()) < 4 * z.std() / np.sqrt(len(z)), (lam, half, z.mean(), z.std())


def test_flat_top_fit_survives_a_profile_whose_centroid_falls_outside_the_bounds():
    """A near-zero background (Pilatus) gives background-subtracted profiles with negative bins: the centroid c0 leaves [0, n-1] and scipy refused the start. The start is now clipped
    inside the bounds and the fit returns finite parameters."""
    from midas_sx.integrate import _fit_flat_top
    p = np.array([-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 40.0, 3.0, -2.0, -4.0, -6.0, -5.0, -9.0])
    q, chi2 = _fit_flat_top(p, np.full(len(p), 49.0))
    assert np.all(np.isfinite(q)) and np.isfinite(chi2) and 0.0 <= q[1] <= len(p) - 1
    p2 = np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -1.0, 1.0])                      # total <= 0: centroid fallback
    assert np.all(np.isfinite(_fit_flat_top(p2, np.full(len(p2), 49.0))[0]))
