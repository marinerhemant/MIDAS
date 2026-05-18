"""I/O helpers — image readers and diagnostic CSV writers.

- :func:`read_image` — TIFF / HDF5 / GE binary / CBF (auto-detected by ext).
- :func:`write_calibrant_screen_csv` — per-(ring × η-bin) strain map.
- :func:`write_iteration_trace_csv` — per-iteration history dump.
"""
from .readers import read_image, read_dark
from .csvs import write_calibrant_screen_csv, write_iteration_trace_csv

__all__ = ["read_image", "read_dark",
           "write_calibrant_screen_csv", "write_iteration_trace_csv"]
