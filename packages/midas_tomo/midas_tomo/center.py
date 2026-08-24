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

__all__ = ["sharpness", "find_center", "find_center_consensus",
           "shift_values_for_search", "slices_with_signal"]

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


def find_center_consensus(
    cube: np.ndarray,
    shift_values: tuple[float, float, float],
    *,
    slices: "list[int] | None" = None,
    crop: float = 0.5,
    tol: float | None = None,
) -> dict:
    """Pick the rotation-axis shift, and say when it should not be trusted.

    :func:`find_center` scores with one criterion. This runs **both**, over
    several slices, and reports a shift only when they agree — which is the
    whole point of having two criteria that fail differently (variance is
    fooled by noise, TV by streaks; see :func:`sharpness`).

    Automating the choice is what makes this usable without a human squinting
    at a 501-panel contact sheet, and it is also where an automated pipeline
    can go quietly wrong: ``argmax`` over any curve always returns something.
    So the return carries ``trustworthy``, and it is False when

    * the two criteria pick shifts further apart than ``tol`` (default: two
      sweep steps), or
    * either criterion reports ``well_determined = False``, meaning its
      winning score was within 1 % of the median and it could not really
      separate the candidates, or
    * the per-slice picks for one criterion disagree by more than ``tol``,
      which usually means the sample drifted or the sweep is too coarse.

    ``best_shift`` is still populated when ``trustworthy`` is False — for
    inspection, not for use. Callers must check the flag; the CLI refuses to
    print a single answer without it.

    Returns a dict with ``best_shift``, ``trustworthy``, ``reason``,
    ``per_method`` and ``spread``.
    """
    cube = np.asarray(cube)
    if cube.ndim != 4:
        raise ValueError(f"cube must be 4-D (shift, slice, y, x); got {cube.shape}")
    n_shifts, n_slices = cube.shape[0], cube.shape[1]
    step = abs(float(shift_values[2])) or 1.0
    if tol is None:
        tol = 2.0 * step

    if slices is None:
        # Spread the probes through the stack rather than trusting one slice:
        # a single slice can be empty, or all sample, or sitting on a defect.
        k = min(4, n_slices)
        slices = [int(round((i + 0.5) * n_slices / k)) for i in range(k)]
        slices = sorted({min(max(s, 0), n_slices - 1) for s in slices})

    start, end, _ = (float(v) for v in shift_values)
    shifts_axis = (np.array([start]) if n_shifts == 1
                   else np.linspace(start, end, n_shifts))
    per_method: dict = {}
    reasons: list[str] = []
    for method in ("variance", "tv"):
        picks, determined = [], []
        for s in slices:
            r = find_center(cube, shift_values, slice_idx=s, method=method,
                            crop=crop)
            picks.append(r["best_shift"])
            determined.append(r["well_determined"])
        picks_arr = np.array(picks, dtype=float)
        # A pick sitting on the first or last candidate is not an interior
        # optimum: either the true shift is outside the sweep, or that slice
        # has nothing to focus. Either way it is not evidence.
        edges = {float(np.min(shifts_axis)), float(np.max(shifts_axis))}
        interior = np.array([p for p in picks if float(p) not in edges],
                            dtype=float)
        n_edge = int(picks_arr.size - interior.size)
        if n_edge:
            reasons.append(
                f"{method}: {n_edge} of {picks_arr.size} slices picked the "
                "edge of the sweep, which is not an interior optimum - widen "
                "the range, or score slices that contain sample "
                "(see slices_with_signal)"
            )
        if interior.size:
            picks_arr = interior
        spread = float(picks_arr.max() - picks_arr.min()) if picks_arr.size else 0.0
        per_method[method] = {
            "picks": picks, "slices": list(slices),
            "n_edge_picks": n_edge,
            "median": float(np.median(picks_arr)),
            "spread": spread,
            "well_determined": bool(all(determined)),
        }
        if not all(determined):
            reasons.append(
                f"{method}: at least one slice could not separate the "
                "candidates (winning score within 1 % of the median)"
            )
        if spread > tol:
            reasons.append(
                f"{method}: per-slice picks span {spread:.3f} > tol {tol:.3f}"
            )

    a, b = per_method["variance"]["median"], per_method["tv"]["median"]
    disagreement = abs(a - b)
    if disagreement > tol:
        reasons.append(
            f"variance picked {a:.3f} but total variation picked {b:.3f}, "
            f"a gap of {disagreement:.3f} > tol {tol:.3f}. They fail "
            "differently on purpose, so disagreement means the shift is not "
            "determined by either - look at the stack."
        )

    trustworthy = not reasons
    best = 0.5 * (a + b) if trustworthy else a

    if not trustworthy:
        log.warning("rotation-axis shift is NOT trustworthy: %s",
                    "; ".join(reasons))

    return {
        "best_shift": float(best),
        "trustworthy": trustworthy,
        "reason": "; ".join(reasons) if reasons else "both criteria agree",
        "per_method": per_method,
        "disagreement": float(disagreement),
        "tol": float(tol),
        "slices": list(slices),
    }


def slices_with_signal(
    data: np.ndarray,
    dark: np.ndarray,
    whites: np.ndarray,
    *,
    k: int = 4,
    min_contrast: float = 0.01,
    min_illumination: float = 0.2,
) -> "list[int]":
    """Detector rows that actually see the specimen.

    Centring on a slice with no sample in it is meaningless, and it does not
    fail quietly: the sharpness curve has no interior optimum, so ``argmax``
    returns whichever end of the sweep happens to score highest.

    Measured on bt_1id_jun25b NMC811 s5 — a ~29 um specimen in a 91 um field of
    view. Scoring four evenly spaced slices, two of them found the right shift
    (+13.00, matching the human's pick from a 501-panel contact sheet) and the
    other two returned -25.00 and -23.00, the bottom of the sweep. Those two
    rows were empty. Spacing probes evenly through the stack is the wrong
    default whenever the specimen does not fill it.

    Ranks rows by attenuation **contrast** — the p99 minus the median across
    the row — not by mean attenuation. The mean is the wrong statistic and
    measurably so: on this dataset a 29 um specimen occupies a fraction of a
    128-pixel row at ~5 % attenuation, so the row mean came out at
    **-0.0005**, slightly negative, because front-to-back flat-field drift is
    larger than the specimen's contribution to the average. The p99-minus-
    median is immune to that constant offset and reports the specimen.

    Rows below ``min_contrast`` are excluded outright; the remaining ``k`` are
    spread across the surviving range rather than taken as the strongest,
    which would cluster them on one feature.

    **The illuminated region is taken from the flat field, never from the
    attenuation.** Outside the beam ``white - dark`` is ~0, so the transmission
    ratio is noise, the clip floor turns it into ``-log(1e-6) = 13.8``, and an
    unilluminated row scores as the strongest signal on the detector.
    Measured: on the Ce scan this let all 2048 rows through a contrast filter
    and left the row span at 2047 px, so the rotation-axis fit still ran over
    ~700 rows of pure garbage. ``min_illumination`` is a fraction of the peak
    row-mean of ``white - dark``.
    """
    data = np.asarray(data)
    dark = np.asarray(dark, dtype=np.float64)
    whites = np.asarray(whites, dtype=np.float64)
    white = whites.mean(axis=0) if whites.ndim == 3 else whites

    denom = white - dark
    bad = denom <= 0
    if bad.all():
        raise ValueError(
            "the white field is nowhere brighter than the dark field; the "
            "calibration blocks are mis-assigned"
        )
    denom = np.where(bad, np.nan, denom)

    # Mean over projections first: cheap, and averaging before the log keeps
    # a few negative pixels from poisoning a whole row.
    mean_proj = np.asarray(data, dtype=np.float64).mean(axis=0)
    trans = (mean_proj - dark) / denom
    with np.errstate(invalid="ignore", divide="ignore"):
        atten = -np.log(np.clip(trans, 1e-6, None))
    atten = np.where(np.isfinite(atten), atten, np.nan)
    hi = np.nanpercentile(atten, 99, axis=1)
    mid = np.nanpercentile(atten, 50, axis=1)
    row_signal = np.nan_to_num(hi - mid, nan=0.0)

    denom_rowmean = np.nan_to_num(
        np.nanmean(np.where(bad, np.nan, white - dark), axis=1), nan=0.0)
    illuminated = denom_rowmean >= min_illumination * float(denom_rowmean.max())
    good = np.nonzero((row_signal >= min_contrast) & illuminated)[0]
    if good.size == 0:
        raise ValueError(
            f"no illuminated detector row reaches {min_contrast} attenuation "
            f"contrast. {int(illuminated.sum())} of {illuminated.size} rows "
            f"are illuminated at all, and the strongest contrast among them "
            f"is {row_signal[illuminated].max() if illuminated.any() else 0:.4g}. "
            "Either the specimen is not in the field of view, the flat/dark "
            "blocks are wrong, or the specimen is too weakly absorbing to "
            "locate this way - at mu*D ~ 0.05 there may genuinely be nothing "
            "to find."
        )

    k = int(min(k, good.size))
    # Spread the picks across the rows that have signal, rather than taking
    # the k strongest, which would cluster on one feature.
    idx = np.linspace(0, good.size - 1, k).round().astype(int)
    return sorted({int(good[i]) for i in idx})
