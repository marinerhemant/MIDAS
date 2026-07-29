"""Per-grain polytype lamella thickness from satellite radial FWHM.

For grain g, identify its sample-frame activated axis a_sample = OM_g @ a_crystal,
collect voxels in a q-space tube along that axis near |G|/3, compute the
radial FWHM in q (the broadening along the satellite radial profile), then
invert Scherrer:

    L_9R = 2 pi / FWHM_q     (length in q^{-1} units)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _fwhm_radial(qs: NDArray[np.floating], vals: NDArray[np.floating], axis: NDArray[np.floating], q_target: float) -> float:
    """1-D radial FWHM along ``axis`` near ``q_target`` of the intensity profile."""
    proj = qs @ axis
    if proj.size < 5:
        return float("nan")
    # Local 1-D histogram around q_target.
    bin_lo, bin_hi = q_target - 0.20, q_target + 0.20
    edges = np.linspace(bin_lo, bin_hi, 80)
    counts, _ = np.histogram(proj, bins=edges, weights=vals)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.sum() <= 0:
        return float("nan")
    peak = float(counts.max())
    half = 0.5 * peak
    above = counts > half
    if above.sum() < 2:
        return float("nan")
    where = np.where(above)[0]
    return float(centers[where[-1]] - centers[where[0]])


def per_grain_lamella_thickness(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    OM: NDArray[np.floating],
    a_crystal: NDArray[np.floating],
    G_3_magnitude: float,
    grain_of_voxel: NDArray[np.intp],
    n_grains: int,
    tube_radius_parallel: float = 0.15,
    tube_radius_perpendicular: float = 0.10,
    min_voxels: int = 20,
    allow_deprecated: bool = False,
) -> NDArray[np.floating]:
    """DEPRECATED — use ``polytype.aggregate_thickness.aggregate_lamella_thickness``.

    Per-grain L by projecting onto each grain's OM@a_crystal is a projection-geometry
    artifact for a shared-plane satellite: on demk corr(theta, L)=-0.62 and L swung
    50->22 A with the projection angle alone, manufacturing a parent/twin gap
    (AUDIT_2026-06-23.md). FF cannot give a per-grain/per-variant L for a shared-plane
    feature. Raises by default; pass ``allow_deprecated=True`` to reproduce the artifact.

    Per-grain Scherrer lamella thickness from the satellite radial FWHM.

    Parameters
    ----------
    qs, vals : (n_voxels, 3), (n_voxels,)
    OM : (n_grains, 3, 3)
    a_crystal : (3,) crystal-frame activated axis (e.g. [1,1,1]/sqrt(3))
    G_3_magnitude : satellite radial target q (= |G|/3)
    grain_of_voxel : (n_voxels,)
    n_grains
    tube_radius_parallel, tube_radius_perpendicular : selection radii
    min_voxels : NaN-out grains with fewer voxels in the tube

    Returns
    -------
    L_per_grain : (n_grains,)  thickness in 1/q-units (e.g. Angstrom if q is 1/A);
                  NaN where the fit could not be performed.
    """
    if not allow_deprecated:
        raise RuntimeError(
            "per_grain_lamella_thickness is DEPRECATED: per-grain L on a shared-plane "
            "satellite is a projection-geometry artifact (corr(theta,L)=-0.62; "
            "AUDIT_2026-06-23.md). Use polytype.aggregate_thickness.aggregate_lamella_"
            "thickness (one aggregate L along the measured n_sat). Pass "
            "allow_deprecated=True only to reproduce the historical artifact."
        )
    qs = np.asarray(qs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    OM = np.asarray(OM, dtype=float)
    a_crystal = np.asarray(a_crystal, dtype=float)
    a_crystal = a_crystal / np.linalg.norm(a_crystal)
    g_of_v = np.asarray(grain_of_voxel, dtype=int)

    L = np.full(n_grains, np.nan, dtype=float)
    for gi in range(n_grains):
        mask = (g_of_v == gi) & np.isfinite(vals) & (vals > 0)
        if int(mask.sum()) < min_voxels:
            continue
        a_sample = OM[gi] @ a_crystal
        a_sample /= np.linalg.norm(a_sample)
        qs_g = qs[mask]
        proj = qs_g @ a_sample
        perp = qs_g - proj[:, None] * a_sample
        perp_norm = np.linalg.norm(perp, axis=1)
        in_tube = (
            (np.abs(proj - G_3_magnitude) < tube_radius_parallel)
            & (perp_norm < tube_radius_perpendicular)
        )
        if int(in_tube.sum()) < min_voxels:
            continue
        fwhm = _fwhm_radial(qs_g[in_tube], vals[mask][in_tube], a_sample, G_3_magnitude)
        if not np.isfinite(fwhm) or fwhm <= 0:
            continue
        L[gi] = 2.0 * np.pi / fwhm
    return L


__all__ = ["per_grain_lamella_thickness"]
