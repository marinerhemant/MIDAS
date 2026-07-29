"""Bootstrap UQ for midas_defect analyses.

Three layers:
    * :mod:`samplers`     -- index resamplers (voxel / grain / pair / refl).
    * :mod:`aggregators`  -- population-statistic bootstrap + percentile helpers.
    * :mod:`parallel`     -- BLAS-thread-guarded multiprocessing Pool.
    * :mod:`decorators`   -- @bootstrapped to lift per-element computes.
"""

from .aggregators import (
    ProfileBand,
    bootstrap_population_stat,
    bootstrap_profile_band,
    percentiles_with_nans,
)
from .decorators import bootstrapped
from .parallel import bootstrap_pool, init_blas_single_thread
from .samplers import (
    grain_resample,
    pair_resample,
    reflection_within_grain_resample,
    voxel_resample,
)

__all__ = [
    "ProfileBand",
    "bootstrap_population_stat",
    "bootstrap_pool",
    "bootstrap_profile_band",
    "bootstrapped",
    "grain_resample",
    "init_blas_single_thread",
    "pair_resample",
    "percentiles_with_nans",
    "reflection_within_grain_resample",
    "voxel_resample",
]
