"""Diamond-Anvil-Cell-aware integration helpers.

DAC datasets have two structural quirks ordinary powder integration
ignores:

- The gasket blocks signal in two opposing wedges of η; we should
  integrate only on the open wedges to avoid contaminating I(Q) with
  gasket fluorescence + edge scatter.
- η-coverage on each ring may drop below 50% (or worse near the gasket
  shadow), reducing statistics in a way the user should be told about.

This module provides :func:`build_gasket_mask` and
:func:`eta_coverage_per_ring`. The CLI (Item 22) emits a stderr WARNING
when any ring's coverage drops below a threshold.
"""
from .gasket_mask import build_gasket_mask, eta_coverage_per_ring

__all__ = ["build_gasket_mask", "eta_coverage_per_ring"]
