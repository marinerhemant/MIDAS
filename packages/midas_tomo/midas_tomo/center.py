"""Finding the rotation-axis offset.

The engine can reconstruct the same slice at a range of assumed axis shifts
in one call. Picking the right one is then an image-quality question, which
:func:`find_center` answers with a sharpness criterion.

Two independent criteria are offered because they fail differently: image
variance is fooled by a noisy reconstruction, and total variation is fooled by
streaks. When they disagree, the shift is not well determined and you should
look at the stack.
"""

from __future__ import annotations

import logging

import numpy as np

__all__ = ["sharpness", "find_center", "shift_values_for_search"]

log = logging.getLogger(__name__)


def shift_values_for_search(half_width: float, step: float = 0.25) -> tuple[float, float, float]:
    """A symmetric ``(start, end, step)`` sweep spanning +/- *half_width*."""
    if half_width <= 0:
        raise ValueError(f"half_width must be positive, got {half_width}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return (-float(half_width), float(half_width), float(step))


def sharpness(img: np.ndarray, *, method: str = "variance") -> float:
    """Focus score for one reconstructed slice. Higher is sharper.

    ``variance``
        Image variance. A correctly centred reconstruction concentrates
        density into real features; a mis-centred one smears it, lowering the
        spread. Cheap and usually decisive, but rewards noise, so on a noisy
        stack it can prefer the grainiest slice rather than the sharpest.
    ``tv``
        Negative total variation, i.e. maximising this minimises TV.

        Note what this does and does not measure. For a monotonic edge,
        ``∫|∇f|`` equals the total height change no matter how wide the ramp,
        so TV is close to **blur-invariant** — it is not a defocus metric.
        What it does detect is *added* variation: the doubled edges, paired
        crescents and streaks that mis-centring produces. That makes it the
        better choice when the failure mode is artefacts and the worse choice
        when it is smearing.

    The two therefore fail differently on purpose. If they disagree, the shift
    is not well determined by either and the stack needs looking at.
    """
    img = np.asarray(img, dtype=np.float64)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return float("-inf")
    if method == "variance":
        return float(np.var(finite))
    if method == "tv":
        gy, gx = np.gradient(np.nan_to_num(img))
        return -float(np.mean(np.hypot(gy, gx)))
    raise ValueError(f"unknown method {method!r}; use 'variance' or 'tv'")


def find_center(
    cube: np.ndarray,
    shift_values: tuple[float, float, float],
    *,
    slice_idx: int | None = None,
    method: str = "variance",
    crop: float = 0.5,
) -> dict:
    """Pick the best rotation-axis shift from a multi-shift reconstruction.

    Parameters
    ----------
    cube : ndarray, shape (n_shifts, n_slices, X, X)
        Output of ``run_tomo``/``run_tomo_from_sinos`` with a shift range.
    shift_values : (start, end, step)
        The same range that produced *cube*.
    slice_idx : int | None
        Which slice to score. None uses the middle one.
    method : {'variance', 'tv'}
        See :func:`sharpness`.
    crop : float
        Fraction of the field of view to score, centred. The engine pads to a
        power of two, so the outer region is mostly empty padding whose
        variance would dilute the signal. 0.5 keeps the central half.

    Returns
    -------
    dict
        ``best_shift``, ``best_idx``, ``shifts``, ``scores``, and
        ``well_determined`` — False when the winning score is within 1% of the
        median, which means the criterion could not really separate the
        candidates.
    """
    cube = np.asarray(cube)
    if cube.ndim != 4:
        raise ValueError(f"cube must be 4-D (shift, slice, y, x); got {cube.shape}")

    n_shifts = cube.shape[0]
    start, end, step = (float(v) for v in shift_values)
    shifts = (np.array([start]) if n_shifts == 1
              else np.linspace(start, end, n_shifts))

    if slice_idx is None:
        slice_idx = cube.shape[1] // 2

    x = cube.shape[-1]
    if not 0 < crop <= 1:
        raise ValueError(f"crop must be in (0, 1], got {crop}")
    half = int(x * crop / 2)
    lo, hi = x // 2 - half, x // 2 + half

    scores = np.array([
        sharpness(cube[i, slice_idx, lo:hi, lo:hi], method=method)
        for i in range(n_shifts)
    ])

    best_idx = int(np.argmax(scores))
    median = float(np.median(scores))
    spread = abs(scores[best_idx] - median)
    well_determined = bool(n_shifts > 2 and median != 0 and spread > 0.01 * abs(median))

    if not well_determined and n_shifts > 2:
        log.warning(
            "rotation-axis shift is poorly determined: best score %.6g is within "
            "1%% of the median %.6g across %d candidates. Inspect the stack rather "
            "than trusting best_shift=%.3f.",
            scores[best_idx], median, n_shifts, shifts[best_idx],
        )

    return {
        "best_shift": float(shifts[best_idx]),
        "best_idx": best_idx,
        "shifts": shifts,
        "scores": scores,
        "method": method,
        "slice_idx": slice_idx,
        "well_determined": well_determined,
    }
