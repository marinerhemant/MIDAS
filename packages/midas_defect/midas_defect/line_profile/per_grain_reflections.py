"""Collect per-grain, per-reflection radial-profile moments from a voxel cloud.

For each grain ``g`` and each predicted Bragg reflection ``hkl`` with
crystal-frame g-vector ``G``, the corresponding sample-frame target is
``q_target = OM_g @ G``. Voxels within ``query_radius`` of that point are
collected and a 1-D radial profile is built along the ``q_target`` direction;
the intensity-weighted centroid, FWHM, and (optionally) third-moment skewness
of that profile are returned.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _profile_moments(
    proj: NDArray[np.floating],
    vals: NDArray[np.floating],
    centre_target: float,
    n_bins: int = 60,
    half_window: float = 0.30,
    return_skewness: bool = True,
) -> tuple[float, float, float, float]:
    """Return (centroid, fwhm, skewness, integrated intensity) of a 1-D profile."""
    edges = np.linspace(centre_target - half_window, centre_target + half_window, n_bins + 1)
    counts, _ = np.histogram(proj, bins=edges, weights=vals)
    if counts.sum() <= 0:
        return float("nan"), float("nan"), float("nan"), 0.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = counts / counts.sum()
    centroid = float((w * centers).sum())
    second = float((w * (centers - centroid) ** 2).sum())
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0) * max(second, 0.0))  # Gaussian FWHM
    skew = float("nan")
    if return_skewness and second > 0:
        third = float((w * (centers - centroid) ** 3).sum())
        skew = third / second ** 1.5
    integrated = float(counts.sum())
    return centroid, fwhm, skew, integrated


def collect_per_grain_reflections(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    grain_of_voxel: NDArray[np.intp],
    OM: NDArray[np.floating],
    G_arr: NDArray[np.floating],
    query_radius: float = 0.08,
    min_voxels_per_refl: int = 6,
    return_skewness: bool = True,
) -> list[dict]:
    """Per-grain, per-reflection profile moments.

    Parameters
    ----------
    qs : (n_voxels, 3)
        Voxel q-positions in the sample frame.
    vals : (n_voxels,)
    grain_of_voxel : (n_voxels,)
        Pre-computed grain assignment; entries with index >= n_grains are skipped.
    OM : (n_grains, 3, 3)
    G_arr : (n_hkls, 3)
        Crystal-frame g-vectors for each predicted (h, k, l).
    query_radius
        Sphere radius around the predicted Bragg position; voxels within this
        radius participate in the per-reflection profile.
    min_voxels_per_refl
        Reflections with fewer voxels are skipped.
    return_skewness
        Whether to compute the third-moment skewness (slightly more expensive).

    Returns
    -------
    A list of length ``n_grains``. Each entry is a dict with keys
        refl_indices : (n_refl_g,) which hkls had usable voxels for this grain
        G_magnitude  : (n_refl_g,) |G| per reflection
        centroid     : (n_refl_g,) intensity-weighted centroid along q_target
        fwhm         : (n_refl_g,)
        skewness     : (n_refl_g,) (NaN if return_skewness=False)
        intensity    : (n_refl_g,) integrated intensity in the sphere
        n_voxels     : (n_refl_g,) voxel count per profile
    """
    from scipy.spatial import cKDTree

    qs = np.asarray(qs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    g_of_v = np.asarray(grain_of_voxel, dtype=int)
    OM = np.asarray(OM, dtype=float)
    G_arr = np.asarray(G_arr, dtype=float)

    n_grains = OM.shape[0]
    n_hkls = G_arr.shape[0]
    G_mag = np.linalg.norm(G_arr, axis=1)

    out: list[dict] = []
    tree = cKDTree(qs)

    for gi in range(n_grains):
        grain_mask = (g_of_v == gi) & np.isfinite(vals) & (vals > 0)
        if not grain_mask.any():
            out.append(_empty_grain_entry())
            continue
        grain_idx = np.where(grain_mask)[0]
        idx_set = set(grain_idx.tolist())

        targets = (OM[gi] @ G_arr.T).T  # (n_hkls, 3)
        refl_indices: list[int] = []
        mags: list[float] = []
        centroids: list[float] = []
        fwhms: list[float] = []
        skews: list[float] = []
        intensities: list[float] = []
        counts: list[int] = []
        for hi in range(n_hkls):
            near = tree.query_ball_point(targets[hi], r=query_radius)
            sel = [k for k in near if k in idx_set]
            if len(sel) < min_voxels_per_refl:
                continue
            sel = np.asarray(sel, dtype=int)
            target_norm = float(np.linalg.norm(targets[hi]))
            if target_norm < 1e-15:
                continue
            target_hat = targets[hi] / target_norm
            proj = qs[sel] @ target_hat
            v = vals[sel]
            centroid, fwhm, skew, integ = _profile_moments(
                proj, v, target_norm, return_skewness=return_skewness
            )
            refl_indices.append(hi)
            mags.append(float(G_mag[hi]))
            centroids.append(centroid)
            fwhms.append(fwhm)
            skews.append(skew)
            intensities.append(integ)
            counts.append(int(sel.size))

        out.append(
            {
                "refl_indices": np.asarray(refl_indices, dtype=int),
                "G_magnitude": np.asarray(mags, dtype=float),
                "centroid": np.asarray(centroids, dtype=float),
                "fwhm": np.asarray(fwhms, dtype=float),
                "skewness": np.asarray(skews, dtype=float),
                "intensity": np.asarray(intensities, dtype=float),
                "n_voxels": np.asarray(counts, dtype=int),
            }
        )

    return out


def _empty_grain_entry() -> dict:
    return {
        "refl_indices": np.zeros(0, dtype=int),
        "G_magnitude": np.zeros(0),
        "centroid": np.zeros(0),
        "fwhm": np.zeros(0),
        "skewness": np.zeros(0),
        "intensity": np.zeros(0),
        "n_voxels": np.zeros(0, dtype=int),
    }


__all__ = ["collect_per_grain_reflections"]
