"""Quantify the polytype satellite-peak enhancement.

Integrate intensity at q = G/3 and q = 2G/3 along the activated axis, and
compare to the mean intensity at the same q-magnitude along a set of random
directions sampled isotropically over the unit sphere. The ratio is a
dimensionless measure of how much the polytype stacking concentrates
intensity into those satellites.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _random_unit_directions(n: int, rng: np.random.Generator) -> NDArray[np.floating]:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def polytype_satellite_enhancement(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    sample_axis: NDArray[np.floating],
    G_magnitude: float,
    integration_radius: float = 0.06,
    n_random_directions: int = 300,
    rng_seed: int = 0,
    *,
    allow_deprecated: bool = False,
) -> dict:
    """DEPRECATED — use ``polytype.satellite_excess.satellite_radial_excess``.

    The "enhancement" (summed I along axis / mean summed I over random directions) is
    dominated by VOXEL COUNT, not intensity: the satellite tube is densely populated,
    random directions at |q|=G/3 hit near-empty space. On demk this inflated a real
    ~5x per-voxel excess to 700-1600x (AUDIT_2026-06-23.md). It raises by default;
    pass ``allow_deprecated=True`` only to reproduce the historical artifact.

    Satellite enhancement at |G|/3 and 2|G|/3 along ``sample_axis``.

    Parameters
    ----------
    qs, vals : (n_voxels, 3), (n_voxels,)
    sample_axis : (3,) the activated axis in the sample frame
    G_magnitude : |G| of the parent Bragg
    integration_radius : sphere radius for intensity sums (same q-units)
    n_random_directions : sample size for the background reference

    Returns
    -------
    dict with
        enhancement_G_over_3   I(along) / I(bg)  at q = |G|/3
        enhancement_2G_over_3  same at q = 2|G|/3
        along_integrated_G3    raw I at |G|/3 along axis
        along_integrated_2G3   raw I at 2|G|/3 along axis
        bg_baseline_G3         mean I per random direction at |G|/3
        bg_baseline_2G3        mean I per random direction at 2|G|/3
    """
    if not allow_deprecated:
        raise RuntimeError(
            "polytype_satellite_enhancement is DEPRECATED: its enhancement ratio is a "
            "voxel-count artifact (700-1600x vs real ~5x; AUDIT_2026-06-23.md). Use "
            "polytype.satellite_excess.satellite_radial_excess (texture-safe per-voxel "
            "excess + discrete-vs-relrod controls). Pass allow_deprecated=True only to "
            "reproduce the historical number."
        )
    qs = np.asarray(qs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    axis = np.asarray(sample_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    rng = np.random.default_rng(rng_seed)
    rand_dirs = _random_unit_directions(n_random_directions, rng)

    def integrate(direction: NDArray[np.floating], q_target: float) -> float:
        anchor = direction * q_target
        d = np.linalg.norm(qs - anchor, axis=1)
        return float(vals[d < integration_radius].sum())

    I_along_G3 = integrate(axis, G_magnitude / 3.0)
    I_along_2G3 = integrate(axis, 2.0 * G_magnitude / 3.0)
    bg_G3 = float(np.mean([integrate(d, G_magnitude / 3.0) for d in rand_dirs]))
    bg_2G3 = float(np.mean([integrate(d, 2.0 * G_magnitude / 3.0) for d in rand_dirs]))

    enh_G3 = (I_along_G3 / bg_G3) if bg_G3 > 0 else float("inf")
    enh_2G3 = (I_along_2G3 / bg_2G3) if bg_2G3 > 0 else float("inf")

    return {
        "enhancement_G_over_3": enh_G3,
        "enhancement_2G_over_3": enh_2G3,
        "along_integrated_G3": I_along_G3,
        "along_integrated_2G3": I_along_2G3,
        "bg_baseline_G3": bg_G3,
        "bg_baseline_2G3": bg_2G3,
    }


__all__ = ["polytype_satellite_enhancement"]
