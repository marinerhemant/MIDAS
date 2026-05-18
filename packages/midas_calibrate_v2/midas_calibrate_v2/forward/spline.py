"""Re-export shim — TPS spline now lives in ``midas_peakfit.spline``.

Promoted to the shared substrate so HEDM grain refinement and joint
calibration can apply the same per-detector spatial-distortion spline that
paper-3 introduced for Stage-4 residual coupling.
"""
from midas_peakfit.spline import TPSpline, fit_tps, fit_tps_refinable

__all__ = ["TPSpline", "fit_tps", "fit_tps_refinable"]
