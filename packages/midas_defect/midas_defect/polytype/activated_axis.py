"""Identify the activated <111> (FCC) axis carrying polytype satellite peaks.

Of the four cubic <111> directions, only the variant carrying the active
9R / 3R / 12R polytype lamellae shows a coherent satellite-peak signature
at q-magnitude ``|G|/3``. We pick the activated direction by integrating
voxel intensity in a tube around each candidate axis at the satellite radius.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_FCC_111_AXES = np.array(
    [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
) / np.sqrt(3.0)


def _integrate_along_axis(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    axis: NDArray[np.floating],
    q_target: float,
    tube_radius_parallel: float,
    tube_radius_perpendicular: float,
) -> float:
    """Sum intensity inside a q-space tube around the satellite radius along axis."""
    proj = qs @ axis  # projection onto axis
    perp = qs - proj[:, None] * axis
    perp_norm = np.linalg.norm(perp, axis=1)
    mask = (
        (np.abs(proj - q_target) < tube_radius_parallel)
        & (perp_norm < tube_radius_perpendicular)
    )
    return float(vals[mask].sum())


def detect_activated_111_axis(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    G_111_magnitude: float,
    crystal_to_sample_OM: NDArray[np.floating] | None = None,
    tube_radius_parallel: float = 0.10,
    tube_radius_perpendicular: float = 0.06,
) -> dict:
    """Pick the <111> direction with the strongest satellite-peak intensity.

    Parameters
    ----------
    qs, vals : (n_voxels, 3), (n_voxels,)
    G_111_magnitude : float
        |G_111| in the same units as ``qs``; tube target is |G|/3.
    crystal_to_sample_OM : (3, 3) or None
        If supplied, candidate axes are rotated into the sample frame via
        ``OM @ axis_crystal``. If None, the axes are interpreted as already
        sample-frame.
    tube_radius_parallel, tube_radius_perpendicular : tube selection radii.

    Returns
    -------
    dict with
        a_sample             (3,) chosen unit vector in the sample frame
        all_intensities      (4,) integrated intensity for each candidate
        argmax_index         int  index into the four candidate axes
    """
    qs = np.asarray(qs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if crystal_to_sample_OM is None:
        axes = _FCC_111_AXES.copy()
    else:
        OM = np.asarray(crystal_to_sample_OM, dtype=float)
        axes = (OM @ _FCC_111_AXES.T).T

    q_target = G_111_magnitude / 3.0
    intensities = np.array(
        [
            _integrate_along_axis(
                qs, vals, axes[i], q_target, tube_radius_parallel, tube_radius_perpendicular
            )
            for i in range(4)
        ]
    )
    k = int(np.argmax(intensities))
    return {
        "a_sample": axes[k],
        "all_intensities": intensities,
        "argmax_index": k,
    }


__all__ = ["detect_activated_111_axis"]
