"""Per-grain asterism: per-spot-local radial(strain) vs tangential(rotation) split.

Replaces the radial/azimuthal split derived from ``second_moment.per_grain_asterism_tensor``
+ ``eigenvalue_spectrum.asterism_anisotropy_per_grain``, which built a single q-space
second-moment tensor in the GLOBAL frame by pooling a grain's voxels across all its
spots (each pointing a different way) and then projected onto one mean-q direction.
For multi-spot grains that projection mixes spot frames and mislabels strain vs
rotation (it produced a spurious matrix-edge/twin-screw asymmetry; AUDIT_2026-06-23.md).

Correct decomposition: resolve EACH voxel's offset dq = q - q_B in ITS OWN spot's
local frame before aggregating:

    radial     r = dq . qB_hat            (changes |q|  -> elastic/plastic strain)
    tangential t = |dq - r qB_hat|        (azimuthal arc -> lattice rotation / GND)

then per grain report intensity-weighted RMS sigma_r, sigma_t. A grain dominated by
edge content broadens tangentially (sigma_t > sigma_r); screw/strain broadens radially.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["per_grain_asterism_local"]


def per_grain_asterism_local(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    grain_of_voxel: NDArray[np.intp],
    P_all_nearest: NDArray[np.floating],
    asterism_mask: NDArray[np.bool_],
    n_grains: int,
    *,
    min_voxels_per_grain: int = 30,
) -> dict:
    """Per-grain intensity-weighted radial/tangential asterism widths (per-spot-local).

    Parameters
    ----------
    qs : (N,3) voxel q-positions.        vals : (N,) intensities (weights).
    grain_of_voxel : (N,) grain id per voxel (>= n_grains or <0 ignored).
    P_all_nearest : (N,3) nearest predicted Bragg q for each voxel (defines the
        local radial direction qB_hat for that voxel).
    asterism_mask : (N,) True for asterism-tail voxels.
    min_voxels_per_grain : NaN-out grains with fewer voxels in the mask.

    Returns
    -------
    dict with (each (n_grains,))
        sigma_r           radial RMS offset (1/A)  -> strain broadening
        sigma_t           tangential RMS offset (1/A) -> rotation/mosaic
        ratio             sigma_t / sigma_r  (>1 rotation-dominated)
        frac_tangential   intensity fraction with t > |r| (rotation-dominated)
        n_voxels          voxels used per grain
    """
    qs = np.asarray(qs, dtype=float)
    vals = np.asarray(vals, dtype=float)
    g = np.asarray(grain_of_voxel, dtype=int)
    Pn = np.asarray(P_all_nearest, dtype=float)
    mask = np.asarray(asterism_mask, dtype=bool)

    sel = mask & (g >= 0) & (g < n_grains) & np.isfinite(vals) & (vals > 0)
    sigma_r = np.full(n_grains, np.nan)
    sigma_t = np.full(n_grains, np.nan)
    frac_t = np.full(n_grains, np.nan)
    nvox = np.zeros(n_grains, dtype=int)
    if not sel.any():
        return dict(sigma_r=sigma_r, sigma_t=sigma_t, ratio=sigma_t / sigma_r,
                    frac_tangential=frac_t, n_voxels=nvox)

    gi = g[sel]
    w = vals[sel]
    Ph = Pn[sel]
    norm = np.linalg.norm(Ph, axis=1, keepdims=True)
    Ph = Ph / np.maximum(norm, 1e-12)
    dq = qs[sel] - Pn[sel]
    r = np.einsum("ij,ij->i", dq, Ph)
    perp = dq - r[:, None] * Ph
    t = np.linalg.norm(perp, axis=1)

    wsum = np.zeros(n_grains)
    sr = np.zeros(n_grains)
    st = np.zeros(n_grains)
    wt = np.zeros(n_grains)        # tangential-dominated weight
    np.add.at(wsum, gi, w)
    np.add.at(sr, gi, w * r * r)
    np.add.at(st, gi, w * t * t)
    np.add.at(wt, gi, w * (t > np.abs(r)))
    np.add.at(nvox, gi, 1)

    ok = (nvox >= min_voxels_per_grain) & (wsum > 0)
    sigma_r[ok] = np.sqrt(sr[ok] / wsum[ok])
    sigma_t[ok] = np.sqrt(st[ok] / wsum[ok])
    frac_t[ok] = wt[ok] / wsum[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = sigma_t / sigma_r
    return dict(sigma_r=sigma_r, sigma_t=sigma_t, ratio=ratio,
                frac_tangential=frac_t, n_voxels=nvox)
