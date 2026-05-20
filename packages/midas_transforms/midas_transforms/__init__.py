"""midas-transforms: Pure-Python/PyTorch FF-HEDM transforms.

Drop-in replacement for the four C binaries that sit between peak-fitting
(``midas-peakfit``) and indexing (``midas-index``):

- ``MergeOverlappingPeaksAllZarr`` -> ``midas_transforms.merge_overlapping_peaks``
- ``CalcRadiusAllZarr``            -> ``midas_transforms.calc_radius``
- ``FitSetupZarr``                 -> ``midas_transforms.fit_setup``
- ``SaveBinData``                  -> ``midas_transforms.bin_data``

Two equally-supported usage modes:

**Mode 1 - per-stage (round-trips through disk, like the C binaries):**

    from midas_transforms import merge_overlapping_peaks, calc_radius, fit_setup, bin_data
    merge_overlapping_peaks(zarr_path="...", result_folder="...", device="cuda")
    calc_radius(result_folder="...", device="cuda")
    fit_setup(result_folder="...", device="cuda")
    bin_data(result_folder="...", device="cuda")

**Mode 2 - chained Pipeline (intermediates stay on GPU, only final outputs written):**

    from midas_transforms import Pipeline
    pipe = Pipeline.from_zarr(zarr_path, device="cuda")
    result = pipe.run()
    pipe.dump(out_dir)

See ``dev/implementation_plan.md`` for design and roadmap.
"""

__version__ = "0.6.0"

from .merge import merge_overlapping_peaks
from .radius import calc_radius
from .fit_setup import fit_setup
from .bin_data import bin_data, bin_data_scanning, bin_data_unified
from .pipeline import Pipeline

__all__ = [
    "merge_overlapping_peaks",
    "calc_radius",
    "fit_setup",
    "bin_data",
    "bin_data_scanning",
    "bin_data_unified",
    "Pipeline",
    "__version__",
]
