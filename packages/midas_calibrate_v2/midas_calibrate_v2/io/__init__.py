"""I/O helpers — image readers and diagnostic CSV writers.

- :func:`read_image` — TIFF / HDF5 / GE binary / CBF (auto-detected by ext).
- :class:`BadPixelSentinelWarning` — fires when a frame carries out-of-band
  bad-pixel values (the EIGER ``2**32-1`` convention, which no ``< 0`` guard
  catches).
- :func:`write_calibrant_screen_csv` — per-(ring × η-bin) strain map.
- :func:`write_iteration_trace_csv` — per-iteration history dump.
"""
from .readers import read_image, read_dark, BadPixelSentinelWarning
from .csvs import write_calibrant_screen_csv, write_iteration_trace_csv

__all__ = ["read_image", "read_dark", "BadPixelSentinelWarning",
           "write_calibrant_screen_csv", "write_iteration_trace_csv"]
