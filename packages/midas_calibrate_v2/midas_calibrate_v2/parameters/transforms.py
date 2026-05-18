"""Re-export shim — bijective parameter-space transforms now live in
``midas_peakfit.transforms``.  Kept here for backwards compatibility.
"""
from midas_peakfit.transforms import (
    Identity,
    Log,
    Logit,
    Scaled,
    Transform,
)

__all__ = ["Transform", "Identity", "Log", "Logit", "Scaled"]
