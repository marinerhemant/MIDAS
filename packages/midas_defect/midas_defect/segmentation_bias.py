"""Threshold-segmentation bias: how a label's size depends on where the cut falls.

For a segmentation that keeps voxels at or above a fixed absolute value ``T``,
two features of identical shape whose amplitudes differ by a factor ``k`` give
label volumes in the ratio

    V_A / V_B = k ** |s|,        s = d log V / d log T  at the operating threshold

-- not 1, and not ``k``. ``|s|`` is set by where ``T`` sits on each feature's
own intensity distribution, so it grows as a feature gets weaker relative to
the cut. On the demk L5 reference sample (``manuals/defect/ENVELOPE.md``
section 16) it ran from 0.8 to 5.4 across components, and it made Friedel pairs
that must be equal come out 0.32-1.82 apart in label sum while their intensity
per voxel held at 0.965.

Two tools for avoiding that trap:

* :func:`label_volume_slope` measures ``s`` for one feature, and
  :func:`predicted_volume_ratio` turns it into the volume ratio a pure
  segmentation effect would produce, so an observed ratio can be checked
  against that null before it is read as physics.
* :func:`mirrored_dead_masks` censors the detector dead regions of two
  mirror-related features *identically*, including the mirror image of the
  partner's dead region, so censoring cannot itself create an asymmetry.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


def label_volume_slope(
    values: ArrayLike,
    threshold: float,
    *,
    rel_step: float = 0.1,
) -> dict:
    """Local slope ``s = d log V / d log T`` of a feature's above-threshold voxel count.

    Parameters
    ----------
    values
        Voxel intensities of ONE feature, any shape (flattened). Pass the raw
        values around the feature, not only the labelled voxels: the count at
        ``T * (1 - rel_step)`` needs the voxels just below the cut.
    threshold
        The operating threshold ``T`` (> 0) the segmentation used.
    rel_step
        Half-width of the symmetric step in threshold, as a fraction. The slope
        is a central difference between ``T * (1 - rel_step)`` and
        ``T * (1 + rel_step)``. Must lie in (0, 1).

    Returns
    -------
    dict with keys
        ``s``          float, normally negative (raising ``T`` shrinks the label)
        ``abs_s``      float, ``|s|``
        ``volume``     int, voxel count at ``T``
        ``volume_lo``  int, count at ``T * (1 - rel_step)``
        ``volume_hi``  int, count at ``T * (1 + rel_step)``

    Raises
    ------
    ValueError
        If ``threshold`` is not positive and finite, ``rel_step`` is outside
        (0, 1), or no voxel reaches ``T * (1 + rel_step)``.
    """
    v = np.asarray(values, dtype=float).ravel()
    T = float(threshold)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"threshold must be positive and finite, got {threshold!r}")
    if not 0.0 < rel_step < 1.0:
        raise ValueError(f"rel_step must lie in (0, 1), got {rel_step!r}")
    t_lo, t_hi = T * (1.0 - rel_step), T * (1.0 + rel_step)
    n_lo = int(np.count_nonzero(v >= t_lo))
    n_0 = int(np.count_nonzero(v >= T))
    n_hi = int(np.count_nonzero(v >= t_hi))
    if n_hi == 0:
        raise ValueError(
            f"no voxel reaches T*(1+rel_step) = {t_hi:g}; the slope is undefined here"
        )
    s = (np.log(n_hi) - np.log(n_lo)) / (np.log(t_hi) - np.log(t_lo))
    return {
        "s": float(s),
        "abs_s": float(abs(s)),
        "volume": n_0,
        "volume_lo": n_lo,
        "volume_hi": n_hi,
    }


def predicted_volume_ratio(amplitude_ratio: float, s: float) -> float:
    """Label-volume ratio a pure threshold effect produces: ``amplitude_ratio ** |s|``.

    Use it as the null for an observed volume ratio between two features
    segmented at the same threshold. On demk L5 a Friedel pair with amplitude
    ratio 1.172 and ``|s|`` = 1.24 predicts 1.172 ** 1.24 = 1.22, against 1.191
    observed.
    """
    k = float(amplitude_ratio)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"amplitude_ratio must be positive and finite, got {amplitude_ratio!r}")
    return float(k ** abs(float(s)))


def mirrored_dead_masks(
    center_a: ArrayLike,
    center_b: ArrayLike,
    dead_rows: ArrayLike,
    dead_cols: ArrayLike,
    mirror_axis: Literal["col", "row"],
) -> dict:
    """Dead rows and columns to censor in BOTH members of a mirror-related pair.

    Two spots of one reflection are mirror images on the detector:

    * the two **Ewald crossings** of one q share a row and are mirrored in
      COLUMN about the beam centre -- ``mirror_axis='col'``;
    * **Friedel mates** share a column and are mirrored in ROW --
      ``mirror_axis='row'``.

    Censoring each member only at its own dead pixels makes the two supports
    differ whenever a dead strip clips one and not the other, and that alone
    shifts both intensity and centroid. Here each member is censored at every
    offset from its own centre at which EITHER member has a dead pixel, with the
    offset reflected along the mirror axis for the partner, so both supports lose
    the same pixels in their own frames.

    Parameters
    ----------
    center_a, center_b
        ``(row, col)`` detector centres of the two members, in pixels, rounded to
        the nearest pixel.
    dead_rows, dead_cols
        1-D arrays of absolute dead row and column indices.
    mirror_axis
        ``'col'`` for a crossing pair, ``'row'`` for a Friedel pair.

    Returns
    -------
    dict with keys ``rows_a``, ``cols_a``, ``rows_b``, ``cols_b``: sorted unique
    integer arrays of absolute indices to censor. They can fall outside the
    detector; clip to your own bounds.
    """
    if mirror_axis not in ("col", "row"):
        raise ValueError(f"mirror_axis must be 'col' or 'row', got {mirror_axis!r}")
    ca_ = np.asarray(center_a, dtype=float).ravel()
    cb_ = np.asarray(center_b, dtype=float).ravel()
    if ca_.size != 2 or cb_.size != 2:
        raise ValueError("center_a and center_b must each be (row, col)")
    ra, ca = int(round(ca_[0])), int(round(ca_[1]))
    rb, cb = int(round(cb_[0])), int(round(cb_[1]))
    dr = np.unique(np.asarray(dead_rows, dtype=np.int64).ravel())
    dc = np.unique(np.asarray(dead_cols, dtype=np.int64).ravel())
    if mirror_axis == "col":
        u = np.union1d(dc - ca, -(dc - cb))
        cols_a, cols_b = ca + u, cb - u
        w = np.union1d(dr - ra, dr - rb)
        rows_a, rows_b = ra + w, rb + w
    else:
        w = np.union1d(dr - ra, -(dr - rb))
        rows_a, rows_b = ra + w, rb - w
        u = np.union1d(dc - ca, dc - cb)
        cols_a, cols_b = ca + u, cb + u
    return {
        "rows_a": np.unique(rows_a),
        "cols_a": np.unique(cols_a),
        "rows_b": np.unique(rows_b),
        "cols_b": np.unique(cols_b),
    }


__all__ = ["label_volume_slope", "mirrored_dead_masks", "predicted_volume_ratio"]
