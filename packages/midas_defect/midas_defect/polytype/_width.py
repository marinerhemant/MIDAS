"""Width estimation on sampled profiles, done so it does not saturate.

The obvious estimator for a FWHM -- take the samples sitting above half max and
report their extent -- is wrong in two ways that compound.  It is biased LOW by
up to one sample spacing (it measures the outermost samples above half max, not
where the curve actually crosses half max), and it is QUANTISED to the sample
spacing, so for a peak only a few samples wide it returns a small set of
discrete values.  Below about two spacings it saturates completely: it returns
one spacing, or NaN when a single sample clears half max.  Six such
measurements agreeing exactly is a rail, not precision.

Interpolating the crossings removes both problems and costs nothing.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["fwhm_half_max"]


def fwhm_half_max(
    centers: NDArray[np.floating],
    values: NDArray[np.floating],
    *,
    baseline: float = 0.0,
) -> float:
    """Full width at half maximum of a sampled single-peaked curve.

    The half-max level is ``baseline + 0.5 * (peak - baseline)``, and the width
    is the distance between the two points where the curve crosses it, found by
    linear interpolation between the bracketing samples.

    Parameters
    ----------
    centers
        Sample positions, strictly increasing.
    values
        Sampled curve, same length as ``centers``.
    baseline
        Level the curve returns to away from the peak.  Defaults to 0, which is
        the right convention for a histogram of a distribution.

    Returns
    -------
    float
        The width in the units of ``centers``, or NaN if the peak is not
        bracketed on both sides (peak at an edge, flat curve, fewer than three
        samples).
    """
    x = np.asarray(centers, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.size < 3 or x.size != y.size:
        return float("nan")

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    x, y = x[finite], y[finite]

    ipk = int(np.argmax(y))
    half = baseline + 0.5 * (y[ipk] - baseline)
    if not np.isfinite(half) or y[ipk] <= baseline:
        return float("nan")

    # A peak clearing half max at a single sample is not resolved: interpolating
    # it would return roughly the sample spacing whatever the true width is.
    # Report that it cannot be measured rather than a number that looks like one.
    if np.count_nonzero(y > half) < 2:
        return float("nan")

    def _crossing(step: int) -> float:
        j = ipk
        while 0 <= j + step < y.size and y[j + step] >= half:
            j += step
        k = j + step
        if not (0 <= k < y.size):
            return float("nan")          # peak runs off this edge, cannot measure
        y0, y1 = y[j], y[k]
        if y0 == y1:
            return float(x[j])
        return float(x[j] + (x[k] - x[j]) * (y0 - half) / (y0 - y1))

    lo, hi = _crossing(-1), _crossing(+1)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    return float(abs(hi - lo))
