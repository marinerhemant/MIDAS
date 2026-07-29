import os

import numpy as np
import pytest

from midas_defect.bootstrap import (
    ProfileBand,
    bootstrap_population_stat,
    bootstrap_pool,
    bootstrap_profile_band,
    bootstrapped,
    grain_resample,
    pair_resample,
    percentiles_with_nans,
    reflection_within_grain_resample,
    voxel_resample,
)
from midas_defect.types import BootUnit


def test_resamplers_return_in_range_and_correct_length():
    rng = np.random.default_rng(0)
    for fn, n in ((voxel_resample, 1000), (grain_resample, 50), (pair_resample, 20)):
        idx = fn(n, rng)
        assert idx.shape == (n,)
        assert idx.min() >= 0 and idx.max() < n


def test_reflection_within_grain_resample_preserves_per_grain_counts():
    per_grain = [
        np.array([0, 1, 2, 3]),
        np.array([10, 11]),
        np.array([], dtype=np.intp),
        np.array([20, 21, 22, 23, 24, 25]),
    ]
    rng = np.random.default_rng(0)
    out = reflection_within_grain_resample(per_grain, rng)
    assert [len(x) for x in out] == [4, 2, 0, 6]
    for orig, drawn in zip(per_grain, out):
        if len(orig):
            assert set(drawn).issubset(set(orig.tolist()))


def test_bootstrap_population_stat_recovers_mean_within_tolerance():
    rng = np.random.default_rng(0)
    truth_mu, truth_sigma, n = 5.0, 1.0, 1000
    vals = rng.normal(truth_mu, truth_sigma, size=n)
    boot, median, (p16, p84) = bootstrap_population_stat(
        vals, stat_fn=np.mean, n_boot=500, rng_seed=42
    )
    assert boot.shape == (500,)
    assert abs(median - truth_mu) < 0.1
    se = truth_sigma / np.sqrt(n)
    assert 0.7 * se < (p84 - p16) / 2 < 1.3 * se


def test_bootstrap_population_stat_default_is_nanmedian():
    vals = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    boot, median, _ = bootstrap_population_stat(vals, n_boot=200, rng_seed=0)
    assert np.isfinite(boot).all()
    assert 1.5 <= median <= 3.5


def test_bootstrap_population_stat_rejects_empty_and_zero_nboot():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_population_stat(np.array([]))
    with pytest.raises(ValueError, match="n_boot must be positive"):
        bootstrap_population_stat(np.arange(5.0), n_boot=0)


def test_percentiles_with_nans_drops_nans():
    samples = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan])
    p16, p50, p84 = percentiles_with_nans(samples, p=(16, 50, 84))
    assert p16 < p50 < p84
    assert np.isclose(p50, np.median([1.0, 2.0, 3.0, 4.0]))


def test_percentiles_with_nans_raises_when_all_nan():
    with pytest.raises(ValueError, match="all samples are NaN"):
        percentiles_with_nans(np.array([np.nan, np.nan]))


def test_bootstrapped_decorator_lifts_to_analysisresult():
    @bootstrapped(BootUnit.GRAIN, name="rho_dummy", units="m^-2")
    def per_grain(values):
        return np.asarray(values, dtype=float)

    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 6.0, 7.0])
    result = per_grain(vals, n_boot=200, rng_seed=0, metadata={"k_NN": 5})

    assert result.name == "rho_dummy"
    assert result.units == "m^-2"
    assert result.boot_unit is BootUnit.GRAIN
    assert result.n_boot == 200
    assert result.bootstrap_samples.shape == (200,)
    assert result.population_ci[0] <= result.population_median <= result.population_ci[1]
    assert result.per_grain is not None and result.per_grain.shape == (8,)
    assert np.isnan(result.per_grain[5])
    assert result.metadata == {"k_NN": 5}


def test_bootstrapped_decorator_pair_unit_populates_per_pair():
    @bootstrapped(BootUnit.PAIR, name="dEps_twin", units="strain")
    def per_pair(values):
        return np.asarray(values, dtype=float)

    result = per_pair(np.linspace(-1, 1, 20), n_boot=50)
    assert result.per_pair is not None
    assert result.per_grain is None
    assert result.boot_unit is BootUnit.PAIR


def test_bootstrapped_decorator_rejects_voxel_unit():
    with pytest.raises(ValueError, match="GRAIN / PAIR / REFLECTION"):

        @bootstrapped(BootUnit.VOXEL, name="x", units="")
        def _f(x):
            return x


def _pool_worker_blas_check(_):
    return [os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")]


def test_bootstrap_pool_initializer_pins_blas_threads():
    with bootstrap_pool(n_workers=2) as pool:
        results = pool.map(_pool_worker_blas_check, range(2))
    for row in results:
        assert row == ["1", "1"]


def _pool_worker_uses_shared_global(_):
    import builtins

    return getattr(builtins, "SHARED_VALUE", None)


def test_bootstrap_pool_injects_init_globals():
    with bootstrap_pool(n_workers=2, init_globals={"SHARED_VALUE": 7}) as pool:
        results = pool.map(_pool_worker_uses_shared_global, range(2))
    assert results == [7, 7]


def test_bootstrap_profile_band_recovers_planted_median():
    """Per-bin grain-resampling bootstrap recovers a known profile median."""
    rng = np.random.default_rng(0)
    n_grains = 80
    r = np.linspace(0.0, 10.0, 64)
    # Truth: median profile is a Gaussian bump at r=5; per-grain noise is ±0.1.
    truth = np.exp(-0.5 * ((r - 5.0) / 0.7) ** 2)
    per_grain = truth[None, :] + rng.normal(scale=0.1, size=(n_grains, 64))
    band = bootstrap_profile_band(per_grain, r, n_boot=200, rng_seed=0)
    assert isinstance(band, ProfileBand)
    assert band.r.shape == (64,)
    assert band.median.shape == (64,)
    assert band.ci_lo.shape == (64,)
    assert band.ci_hi.shape == (64,)
    assert band.n_grains == n_grains
    assert band.n_boot == 200
    assert band.boot_unit == "grain"
    # ci_lo <= median <= ci_hi everywhere.
    assert np.all(band.ci_lo <= band.median + 1e-12)
    assert np.all(band.median <= band.ci_hi + 1e-12)
    # The bootstrap median tracks the truth within a few σ/√N.
    max_err = float(np.max(np.abs(band.median - truth)))
    assert max_err < 5.0 * 0.1 / np.sqrt(n_grains), (
        f"median deviates from truth by {max_err:.3f}"
    )


def test_bootstrap_profile_band_rejects_bad_shapes():
    """Argument validation: 2-D profile, n_boot>=1, matching r length."""
    arr = np.zeros((4, 8))
    r_good = np.linspace(0, 1, 8)
    # Bad n_boot.
    with pytest.raises(ValueError, match="n_boot must be positive"):
        bootstrap_profile_band(arr, r_good, n_boot=0)
    # 1-D profile.
    with pytest.raises(ValueError, match="per_grain_profiles must be 2-D"):
        bootstrap_profile_band(np.zeros(8), r_good, n_boot=10)
    # Mismatched r length.
    with pytest.raises(ValueError, match="r shape"):
        bootstrap_profile_band(arr, np.zeros(7), n_boot=10)
    # Empty grain population.
    with pytest.raises(ValueError, match="empty"):
        bootstrap_profile_band(np.zeros((0, 8)), r_good, n_boot=10)


def test_bootstrap_profile_band_tolerates_nan_rows():
    """A grain with all-NaN profile must not poison the population median."""
    rng = np.random.default_rng(0)
    r = np.linspace(0.0, 1.0, 32)
    truth = np.cos(2 * np.pi * r)
    per_grain = truth[None, :] + 0.05 * rng.normal(size=(20, 32))
    per_grain[3, :] = np.nan
    per_grain[11, :] = np.nan
    band = bootstrap_profile_band(per_grain, r, n_boot=100, rng_seed=1)
    assert np.all(np.isfinite(band.median))
