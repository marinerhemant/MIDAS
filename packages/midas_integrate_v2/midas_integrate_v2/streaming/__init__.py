"""Streaming + multi-frame integration for sweep-mode beamtime.

Sweep-mode HEDM produces hundreds-to-thousands of frames per scan.
Loading them all into RAM is unrealistic; this module provides:

- :class:`TIFFGlobSource`, :class:`HDF5FrameSource`, :class:`ZarrFrameSource`:
  iterators that yield frames one at a time without holding the full
  stack in memory.
- :func:`integrate_stream`: build the geometry once, integrate every
  frame in the source, write profiles as they're produced.
- :func:`reject_cosmic_rays`: per-pixel-across-stack sigma-clip
  outlier rejection.
- :class:`FrameNormalizer`: monitor / exposure / transmission factor
  normalisation at integrate time.
"""
from .frame_source import (
    FrameSource,
    TIFFGlobSource,
    HDF5FrameSource,
    ZarrFrameSource,
    NumpyArraySource,
    GEBinaryFrameSource,
    EDFFrameSource,
)
from .normalize import FrameNormalizer
from .outlier import (
    reject_cosmic_rays,
    reject_spatial_spikes,
    azimuthal_sigma_clip,
    azimuthal_sigma_clip_multi_panel,
)
from .integrate_stream import integrate_stream
from .multi_detector import integrate_multi_detector
from .quality import compute_quality_flags

__all__ = [
    "FrameSource",
    "TIFFGlobSource",
    "HDF5FrameSource",
    "ZarrFrameSource",
    "NumpyArraySource",
    "GEBinaryFrameSource",
    "EDFFrameSource",
    "FrameNormalizer",
    "reject_cosmic_rays",
    "reject_spatial_spikes",
    "azimuthal_sigma_clip",
    "azimuthal_sigma_clip_multi_panel",
    "integrate_stream",
    "integrate_multi_detector",
    "compute_quality_flags",
]
