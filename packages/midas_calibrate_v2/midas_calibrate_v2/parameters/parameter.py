"""Re-export shim — :class:`Parameter` and prior types now live in
``midas_peakfit.parameter``.  Kept here for backwards compatibility with
paper-3 reproducibility runners and external consumers.
"""
from midas_peakfit.parameter import (
    GaussianPrior,
    HalfCauchyPrior,
    Parameter,
    Prior,
    UniformPrior,
)

__all__ = ["Parameter", "Prior", "GaussianPrior", "HalfCauchyPrior", "UniformPrior"]
