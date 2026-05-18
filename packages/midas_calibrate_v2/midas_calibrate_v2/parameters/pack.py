"""Re-export shim — pack/unpack now live in ``midas_peakfit.pack``.  Kept
here for backwards compatibility with paper-3 reproducibility runners.
"""
from midas_peakfit.pack import (
    MultiPackInfo,
    PackInfo,
    pack_multi,
    pack_spec,
    refined_bounds,
    refined_indices,
    refined_subset,
    unpack_spec,
    write_refined_back,
)

__all__ = [
    "PackInfo", "pack_spec", "unpack_spec",
    "refined_indices", "refined_bounds", "refined_subset",
    "write_refined_back", "MultiPackInfo", "pack_multi",
]
