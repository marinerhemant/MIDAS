"""Bootstrap aggregation and percentile helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .samplers import grain_resample


@dataclass
class ProfileBand:
    """Bootstrap CI band of a 1-D profile across a grain population.

    Attributes
    ----------
    r : (n_r,)
        Profile abscissa (typically the real-space r-axis in Å for ΔPDF).
    median : (n_r,)
        Per-bin median of ``stat_fn`` across bootstrap resamples.
    ci_lo, ci_hi : (n_r,)
        Per-bin 16th/84th percentiles (±1σ) of ``stat_fn`` across the
        bootstrap. Always ordered ``ci_lo <= ci_hi``.
    n_grains : int
        Population size before resampling.
    n_boot : int
        Number of bootstrap draws.
    boot_unit : str
        Resampling unit label (default ``"grain"``); kept for traceability
        when results are stored.
    """
    r: NDArray[np.floating]
    median: NDArray[np.floating]
    ci_lo: NDArray[np.floating]
    ci_hi: NDArray[np.floating]
    n_grains: int
    n_boot: int
    boot_unit: str = "grain"


def bootstrap_profile_band(
    per_grain_profiles: NDArray[np.floating],
    r: NDArray[np.floating],
    *,
    stat_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]] = np.nanmedian,
    n_boot: int = 500,
    rng_seed: int = 0,
    boot_unit: str = "grain",
) -> ProfileBand:
    """Per-bin grain-resampling bootstrap of a 1-D profile.

    Parameters
    ----------
    per_grain_profiles : (n_grains, n_r)
        Profile per grain. NaNs (failed/empty grains at that r) are
        tolerated; ``stat_fn`` should ignore them (default ``np.nanmedian``).
    r : (n_r,)
        Profile abscissa, stored unchanged on the result.
    stat_fn
        Population statistic applied along axis 0 (grains). Default
        ``np.nanmedian``.
    n_boot
        Number of grain-resampling draws.

    Returns
    -------
    ProfileBand
        median(r), ci_lo(r) = 16th percentile, ci_hi(r) = 84th percentile.
    """
    arr = np.asarray(per_grain_profiles, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"per_grain_profiles must be 2-D (n_grains, n_r); got {arr.shape}"
        )
    n_grains, n_r = arr.shape
    if n_grains == 0:
        raise ValueError("per_grain_profiles is empty (n_grains=0)")
    if r.shape != (n_r,):
        raise ValueError(
            f"r shape {r.shape} does not match per_grain_profiles axis 1 ({n_r})"
        )
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive; got {n_boot}")

    rng = np.random.default_rng(rng_seed)
    samples = np.empty((n_boot, n_r), dtype=float)
    for i in range(n_boot):
        idx = grain_resample(n_grains, rng)
        samples[i] = stat_fn(arr[idx], axis=0)

    ci_lo, median, ci_hi = (
        np.nanpercentile(samples, 16, axis=0),
        np.nanpercentile(samples, 50, axis=0),
        np.nanpercentile(samples, 84, axis=0),
    )
    return ProfileBand(
        r=np.asarray(r, dtype=float),
        median=median,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_grains=int(n_grains),
        n_boot=int(n_boot),
        boot_unit=boot_unit,
    )


def bootstrap_population_stat(
    per_grain_values: NDArray[np.floating],
    stat_fn: Callable[[NDArray[np.floating]], float] = np.nanmedian,
    n_boot: int = 500,
    rng_seed: int = 0,
) -> tuple[NDArray[np.floating], float, tuple[float, float]]:
    """Resample grains with replacement and evaluate ``stat_fn`` per draw.

    Returns
    -------
    bootstrap_samples : (n_boot,)
        ``stat_fn`` applied to each resample.
    median_estimate : float
        Median of ``bootstrap_samples`` (the population point estimate).
    (p16, p84) : tuple[float, float]
        +/- 1 sigma interval from ``bootstrap_samples``.

    Notes
    -----
    Default ``stat_fn`` is ``np.nanmedian`` because failed-fit grains carry
    NaN by convention (see :class:`midas_defect.types.AnalysisResult`).
    """
    vals = np.asarray(per_grain_values, dtype=float)
    if vals.ndim != 1:
        raise ValueError(f"per_grain_values must be 1-D; got shape {vals.shape}")
    if vals.size == 0:
        raise ValueError("per_grain_values is empty")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive; got {n_boot}")

    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_boot, dtype=float)
    n = vals.size
    for i in range(n_boot):
        idx = grain_resample(n, rng)
        boot[i] = stat_fn(vals[idx])

    p16, median, p84 = percentiles_with_nans(boot, p=(16, 50, 84))
    return boot, float(median), (float(p16), float(p84))


def percentiles_with_nans(
    samples: NDArray[np.floating],
    p: tuple[float, ...] = (16, 50, 84),
) -> tuple[float, ...]:
    """Drop NaNs, then return percentiles. Raises if everything is NaN."""
    arr = np.asarray(samples, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("all samples are NaN/inf; cannot compute percentiles")
    return tuple(float(x) for x in np.percentile(finite, p))


__all__ = [
    "ProfileBand",
    "bootstrap_population_stat",
    "bootstrap_profile_band",
    "percentiles_with_nans",
]
