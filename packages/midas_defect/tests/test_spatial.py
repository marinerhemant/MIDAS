import numpy as np
import pytest

from midas_defect.spatial import (
    epsilon_autocorrelation,
    hall_petch_slope,
    stress_spatial_gradient_per_grain,
)


# -- Autocorrelation --------------------------------------------------------

def test_autocorrelation_uniform_field_gives_nan_or_zero():
    # Constant field: variance=0, Pearson undefined -> NaN.
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, size=(50, 3))
    eps = np.ones(50)
    out = epsilon_autocorrelation(eps, pos)
    assert np.isnan(out["pearson_r_per_bin"]).all()


def test_autocorrelation_short_distance_high_long_distance_low():
    # Field with a smooth spatial gradient: nearby pairs have similar eps so r ~ 1
    # in the smallest-distance bin; opposite-end pairs anti-correlate so r is
    # lower (or even negative) at the largest bin in a bounded domain.
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 50, size=(300, 3))
    eps = pos[:, 0] + rng.normal(scale=0.01, size=300)
    bins = np.array([0.0, 10.0, 30.0, 80.0, 200.0])
    out = epsilon_autocorrelation(eps, pos, distance_bins=bins)
    r = out["pearson_r_per_bin"]
    assert r[0] > 0.7  # smallest distance bin highly correlated
    finite = np.isfinite(r)
    last_finite = np.where(finite)[0][-1]
    assert r[0] > r[last_finite]  # decays with distance


def test_autocorrelation_variant_filter_drops_cross_variant_pairs():
    # 2 variants with anti-correlated fields: filtering by variant should
    # eliminate cancellation in the autocorr.
    rng = np.random.default_rng(2)
    n = 80
    pos = rng.uniform(0, 100, size=(n, 3))
    var = (rng.uniform(size=n) > 0.5).astype(int)
    eps = np.where(var == 0, 1.0, -1.0) + rng.normal(scale=0.05, size=n)
    bins = np.array([0.0, 50.0, 200.0])
    full = epsilon_autocorrelation(eps, pos, distance_bins=bins)
    filtered = epsilon_autocorrelation(eps, pos, distance_bins=bins, variant_labels=var)
    # Intra-variant correlation should be more positive than the mixed one.
    assert filtered["pearson_r_per_bin"][0] > full["pearson_r_per_bin"][0]


def test_autocorrelation_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        epsilon_autocorrelation(np.zeros(5), np.zeros((6, 3)))


# -- Stress gradient --------------------------------------------------------

def test_stress_gradient_uniform_field_is_zero():
    rng = np.random.default_rng(0)
    pos = rng.uniform(size=(20, 3))
    sigma = np.ones(20) * 5.0
    grad = stress_spatial_gradient_per_grain(sigma, pos, k_NN=4)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


def test_stress_gradient_linear_field_recovers_local_slope():
    # Sigma = x_position (1D); local mean-abs-diff over kNN ~ mean(|x_self - x_NN|).
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, size=(50, 3))
    sigma = pos[:, 0]
    grad = stress_spatial_gradient_per_grain(sigma, pos, k_NN=4)
    # Should be positive and finite.
    assert (grad > 0).all()
    # Median local separation in x for kNN over uniform-in-cube is small relative to 100.
    assert np.median(grad) < 30.0


# -- Hall-Petch -------------------------------------------------------------

def test_hall_petch_recovers_planted_slope():
    # Plant sigma = sigma_0 + k / sqrt(d).
    rng = np.random.default_rng(0)
    d = rng.uniform(1.0, 100.0, size=200)
    r = d / 2.0
    sigma_0 = 200.0
    k = 6000.0
    sigma = sigma_0 + k / np.sqrt(d) + rng.normal(scale=2.0, size=200)
    out = hall_petch_slope(sigma, r)
    assert out["k_HP_per_variant"][0] == pytest.approx(k, rel=0.05)
    assert out["sigma_0_per_variant"][0] == pytest.approx(sigma_0, rel=0.05)
    assert out["R_squared"][0] > 0.9
    assert out["n_per_variant"][0] == 200


def test_hall_petch_per_variant():
    rng = np.random.default_rng(1)
    n = 100
    d = rng.uniform(1, 100, size=n)
    r = d / 2.0
    var = (np.arange(n) >= n // 2).astype(int)
    sigma = np.empty(n)
    sigma[:n // 2] = 200.0 + 6000.0 / np.sqrt(d[:n // 2]) + rng.normal(scale=2.0, size=n // 2)
    sigma[n // 2:] = 100.0 + 3000.0 / np.sqrt(d[n // 2:]) + rng.normal(scale=2.0, size=n // 2)
    out = hall_petch_slope(sigma, r, variant_labels=var)
    assert out["k_HP_per_variant"][0] == pytest.approx(6000.0, rel=0.05)
    assert out["k_HP_per_variant"][1] == pytest.approx(3000.0, rel=0.05)


def test_hall_petch_too_few_grains_returns_nan():
    out = hall_petch_slope(np.array([100.0, 110.0]), np.array([5.0, 6.0]))
    assert np.isnan(out["k_HP_per_variant"][0])
    assert out["n_per_variant"][0] == 2
