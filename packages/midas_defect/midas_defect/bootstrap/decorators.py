"""Decorators that lift per-element compute functions to :class:`AnalysisResult`."""

from __future__ import annotations

import functools
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..types import AnalysisResult, BootUnit
from .aggregators import bootstrap_population_stat


def bootstrapped(
    boot_unit: BootUnit,
    name: str,
    units: str,
    n_boot_default: int = 500,
    stat_fn: Callable[[NDArray[np.floating]], float] = np.nanmedian,
):
    """Lift a per-element compute into a function that returns an AnalysisResult.

    The decorated function must return a 1-D array of per-element values (per
    grain, per pair, or per reflection). The wrapper bootstraps that array
    over the chosen ``boot_unit`` and returns the canonical result.

    The decorated function is called *once*. This is valid when the per-element
    compute does not depend on the resample (e.g. per-grain modified-WH fits,
    per-grain Wilson plots). Analyses where the resample changes the inner
    compute (reflection-within-grain bootstrapping of a per-grain fit) must
    drive ``bootstrap_population_stat`` directly with a custom ``stat_fn``.

    Example
    -------
    >>> @bootstrapped(BootUnit.GRAIN, name="rho_total", units="m^-2")
    ... def rho_modified_wh_per_grain(per_grain_refl, crystal, **kw):
    ...     return np.array([_fit_one(g) for g in per_grain_refl])
    """
    if boot_unit not in (BootUnit.GRAIN, BootUnit.PAIR, BootUnit.REFLECTION):
        raise ValueError(
            "bootstrapped decorator currently supports GRAIN / PAIR / REFLECTION "
            f"bootstrap units (got {boot_unit}); for voxel- or "
            "reflection-within-grain resampling, call bootstrap_population_stat "
            "directly with a custom stat_fn."
        )

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(
            *args,
            n_boot: int = n_boot_default,
            rng_seed: int = 0,
            metadata: dict | None = None,
            **kwargs,
        ) -> AnalysisResult:
            per_element = np.asarray(fn(*args, **kwargs), dtype=float).ravel()
            boot, median, ci = bootstrap_population_stat(
                per_element,
                stat_fn=stat_fn,
                n_boot=n_boot,
                rng_seed=rng_seed,
            )
            per_field = {
                BootUnit.GRAIN: "per_grain",
                BootUnit.PAIR: "per_pair",
                BootUnit.REFLECTION: "per_reflection",
            }[boot_unit]
            kw = {per_field: per_element}
            return AnalysisResult(
                name=name,
                units=units,
                boot_unit=boot_unit,
                n_boot=n_boot,
                population_median=median,
                population_ci=ci,
                bootstrap_samples=boot,
                metadata=dict(metadata or {}),
                **kw,
            )

        return wrapper

    return decorator


__all__ = ["bootstrapped"]
