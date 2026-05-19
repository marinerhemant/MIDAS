"""Diagnostic figure + table generators.

Each submodule renders artifacts for a single workflow.  Today:

* :mod:`midas_pipeline.diagnostics.vmap` — V-map refinement plots
  (per-voxel V overlay, spot residuals, loss history, per-grain
  histograms, K-per-ring table).

All matplotlib code lives behind a lazy import so the base
``midas_pipeline`` install doesn't pay the matplotlib startup cost.
"""

from .compare_modes import CompareModesResult, run_compare_modes
from .vmap import (
    plot_loss_history,
    plot_per_grain_v_histograms,
    plot_spot_residuals,
    plot_v_map_overlay,
    write_k_per_ring_table,
    write_v_map_tif,
)

__all__ = [
    "CompareModesResult",
    "run_compare_modes",
    "plot_loss_history",
    "plot_per_grain_v_histograms",
    "plot_spot_residuals",
    "plot_v_map_overlay",
    "write_k_per_ring_table",
    "write_v_map_tif",
]
