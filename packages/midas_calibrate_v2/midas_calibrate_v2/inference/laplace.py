"""Re-export shim — Laplace covariance and Fisher-info approximation now
live in ``midas_peakfit.laplace``.  Kept here for backwards compatibility.
"""
from midas_peakfit.laplace import (
    LaplaceResult,
    fisher_at_map,
    laplace_at_map,
    report_laplace,
)

__all__ = ["LaplaceResult", "laplace_at_map", "fisher_at_map", "report_laplace"]
