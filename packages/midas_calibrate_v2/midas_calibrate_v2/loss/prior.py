"""Re-export shim — ``sum_log_prior`` now lives in
``midas_peakfit.prior``.  Kept here for backwards compatibility.
"""
from midas_peakfit.prior import sum_log_prior

__all__ = ["sum_log_prior"]
