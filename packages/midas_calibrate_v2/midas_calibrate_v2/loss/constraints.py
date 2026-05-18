"""Re-export shim — generic block-sum-zero gauge and Gaussian-prior
residuals now live in ``midas_peakfit.constraints``.  Kept here for
backwards compatibility with paper-3 reproducibility runners.
"""
from midas_peakfit.constraints import (
    gaussian_prior_residual,
    zero_sum_residual,
)

__all__ = ["zero_sum_residual", "gaussian_prior_residual"]
