"""Random-misorientation reference distributions.

The cubic and hexagonal random-pair disorientation densities (Mackenzie 1958
for cubic; Handscomb 1958 / Morawiec 1995 for hexagonal) are obtained by
Monte-Carlo over random Haar-uniform orientation pairs, with the symmetry
reduction supplied by :func:`midas_stress.orientation.misorientation_om_batch`.

The MC result is cached lazily on first call per ``(phase, n_samples)``;
subsequent calls hit the cache and interpolate to the requested angles.

This is preferred over a closed-form piecewise expression because
    * it is robust at branch boundaries without ad-hoc smoothing,
    * symmetry semantics match the canonical ``midas_stress`` reduction
      everywhere else in the package (Cubic m-3m, Hexagonal 6/mmm), and
    * a single code path serves all phases (HCP, BCC, FCC).

The cost is a one-time MC draw (~ tens of ms for n=200_000); negligible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import CrystalPhase

_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi

# Representative space-group numbers; cubic m-3m and hexagonal 6/mmm
# disorientation FZs are not sensitive to the specific SG within the Laue
# class, so any centrosymmetric SG in the class suffices.
_PHASE_TO_SPACE_GROUP = {
    CrystalPhase.FCC: 225,
    CrystalPhase.BCC: 229,
    CrystalPhase.HCP: 194,
}

# (phase, n_samples, n_bins) -> (bin_centers_deg, density_per_radian)
_CACHE: dict[tuple[CrystalPhase, int, int], tuple[NDArray[np.floating], NDArray[np.floating]]] = {}


def _random_so3_quaternions(n: int, rng: np.random.Generator) -> NDArray[np.floating]:
    """Haar-uniform unit quaternions, shape (n, 4)."""
    q = rng.standard_normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def _quat_to_om_flat(q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert (n, 4) quaternions to (n, 9) row-major orientation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    om = np.empty((n, 9), dtype=float)
    om[:, 0] = 1 - 2 * (y * y + z * z)
    om[:, 1] = 2 * (x * y - z * w)
    om[:, 2] = 2 * (x * z + y * w)
    om[:, 3] = 2 * (x * y + z * w)
    om[:, 4] = 1 - 2 * (x * x + z * z)
    om[:, 5] = 2 * (y * z - x * w)
    om[:, 6] = 2 * (x * z - y * w)
    om[:, 7] = 2 * (y * z + x * w)
    om[:, 8] = 1 - 2 * (x * x + y * y)
    return om


def _mc_density(
    phase: CrystalPhase,
    n_samples: int,
    n_bins: int,
    rng_seed: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Monte-Carlo disorientation-angle density for ``phase``."""
    import midas_stress.orientation as o

    space_group = _PHASE_TO_SPACE_GROUP[phase]
    rng = np.random.default_rng(rng_seed)
    q1 = _random_so3_quaternions(n_samples, rng)
    q2 = _random_so3_quaternions(n_samples, rng)
    oms1 = _quat_to_om_flat(q1)
    oms2 = _quat_to_om_flat(q2)
    angles_rad = np.asarray(
        o.misorientation_om_batch(oms1, oms2, space_group=space_group), dtype=float
    )
    angles_deg = angles_rad * _RAD2DEG

    # Bin range: 0 to a generous upper bound; mass beyond the cutoff is
    # exactly zero by construction (no samples land there).
    upper = 95.0 if phase is CrystalPhase.HCP else 65.0
    edges = np.linspace(0.0, upper, n_bins + 1)
    counts, _ = np.histogram(angles_deg, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width_rad = (edges[1] - edges[0]) * _DEG2RAD
    # density per radian
    density = counts / (n_samples * width_rad)
    return centers, density


def mackenzie_pdf(
    angle_deg: NDArray[np.floating],
    phase: CrystalPhase = CrystalPhase.FCC,
    *,
    n_mc_samples: int = 200_000,
    n_bins: int = 200,
    rng_seed: int = 0,
) -> NDArray[np.floating]:
    """Random-misorientation probability density at the given angles.

    Parameters
    ----------
    angle_deg
        Query disorientation angles in degrees.
    phase
        Crystal phase; sets the symmetry group for the FZ reduction.
    n_mc_samples
        Monte-Carlo sample count (cached, so cost is one-shot per phase).
    n_bins
        Bin resolution of the cached MC histogram.
    rng_seed
        RNG seed; combined with ``(phase, n_mc_samples, n_bins)`` to key
        the cache.

    Returns
    -------
    pdf : ndarray (same shape as input)
        Density in **per-radian** units. Multiply by ``pi/180`` for density
        per degree. Mass beyond the cubic / hexagonal disorientation cutoff
        is exactly zero by construction.
    """
    if phase not in _PHASE_TO_SPACE_GROUP:
        raise ValueError(f"unknown phase {phase!r}")
    key = (phase, n_mc_samples, n_bins)
    if key not in _CACHE:
        _CACHE[key] = _mc_density(phase, n_mc_samples, n_bins, rng_seed)
    centers_deg, density = _CACHE[key]

    out = np.interp(
        np.asarray(angle_deg, dtype=float),
        centers_deg,
        density,
        left=0.0,
        right=0.0,
    )
    return out


__all__ = ["mackenzie_pdf"]
