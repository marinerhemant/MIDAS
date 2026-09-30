"""Own-grain density check: does a voxel's per-voxel-map grain have reconstructed density at that voxel?

For each grain k, ``rho_v = R_k[v] / P90_k`` where ``R_k`` is the MLEM reconstruction of grain k's OWN sinogram and
``P90_k`` its 90th percentile over the voxels the per-voxel (PBP) map gave to k. A voxel with ``rho_v < tau`` is flagged
NOT-THIS-GRAIN: the grain's own rays put (almost) nothing there. That is what vacuum that inherited a neighbour's
orientation looks like (the point-by-point map scores it at completeness ~0.5, above the 0.4 gate, because it shares
ray lines with real material). The flag is NOT proof of vacuum: material with no density of its own grain is flagged
too, and a PBP wrong-grain voxel is mostly NOT flagged (phantom: 3-6 % of them; the grain it was given has density
elsewhere on the same rays).

Where it may be used (bt_20id_sep26b 20-ID-E Fe9Cr, LAB_NOTEBOOK section 10; preregistered S9b and S11 on spot-level phantoms; S9c was VOID and is not relied on):

- only for grains with ``nr >= 30`` sinogram rows. With an intragranular orientation gradient the phantom calibrated
  this stratum: correctly assigned material flagged 0.1-0.4 %, inherited vacuum flagged 81-89 %, a real material
  protrusion kept and its inherited halo flagged 84-94 %. For 10 <= nr < 30 the false-flag rate of material was 3.0 %
  (over the 2 % bar); nr < 10 was not measurable. Grains below ``min_rows`` are therefore returned as NaN, not scored.
- ``nr`` is low on real data mostly because near-duplicate orientation "sibling" grains lose their shared spots to
  each other in find_grains (``--sibling-merge-deg``); a sibling's density is starved by construction.
- rho is normalised by each grain's own P90 over its PBP voxels, so a grain that is more than ~90 % vacuum has all of its
  voxels flagged (including real ones) and a grain of 1-3 voxels is flagged only if every voxel is vacuum. The labels are
  find_grains' assignment (Output/voxel_grid.csv), not the min_conf-filtered voxel-map labels of ReconQuality.
- Phantom-calibrated only: the beam tail (5 % and 15 % of the flux in a 60 um FWHM tail were tried) and voxel blur of the
  real data are unmeasured. Voxels within ~20 um of a material boundary were not scored in the calibration.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = ["own_grain_density", "TAU_DEFAULT", "MIN_ROWS_DEFAULT"]

TAU_DEFAULT = 0.15
MIN_ROWS_DEFAULT = 30


def own_grain_density(
    mlem_stack: np.ndarray,
    pbp_grain: np.ndarray,
    nr_hkls: np.ndarray,
    *,
    tau: float = TAU_DEFAULT,
    min_rows: int = MIN_ROWS_DEFAULT,
) -> Dict:
    """rho per voxel, the NOT-THIS-GRAIN flag, and a per-grain summary.

    Parameters
    ----------
    mlem_stack : (n_grains, n, n) per-grain MLEM images (grain g = the layer's Output/ grain list).
    pbp_grain : (n, n) or (n*n,) per-voxel grain index (-1 = unsolved), same grain list.
    nr_hkls : (n_grains,) sinogram rows per grain.
    tau, min_rows : threshold (frozen by the phantom calibration) and the row-count floor.

    Returns
    -------
    dict with ``rho`` (n*n float32, NaN where unsolved or the grain has < min_rows rows), ``flag`` (n*n bool, False where
    NaN), ``scored`` (n*n bool), ``per_grain`` (list of dicts for scored grains) and the totals.
    """
    R = np.asarray(mlem_stack)
    n_g = R.shape[0]
    Rf = R.reshape(n_g, -1)
    lab = np.asarray(pbp_grain).ravel().astype(np.int64)
    if lab.size != Rf.shape[1]:
        raise ValueError(f"pbp_grain has {lab.size} voxels, the stack has {Rf.shape[1]}")
    nr = np.asarray(nr_hkls).ravel()
    rho = np.full(lab.size, np.nan, dtype=np.float32)
    per_grain = []
    n_skipped = 0
    for k in np.unique(lab[lab >= 0]):
        if k >= n_g:
            continue
        m = lab == k
        if int(nr[k]) < min_rows:
            n_skipped += 1
            continue
        vals = Rf[k, m].astype(np.float64)
        p90 = float(np.percentile(vals, 90))
        if p90 <= 0.0:                                  # no density anywhere: every voxel is NOT-THIS-GRAIN
            rho[m] = 0.0
        else:
            rho[m] = (vals / p90).astype(np.float32)
        per_grain.append({"grain": int(k), "nr": int(nr[k]), "voxels": int(m.sum()),
                          "flagged": int((rho[m] < tau).sum())})
    scored = ~np.isnan(rho)
    flag = np.zeros(lab.size, dtype=bool)
    flag[scored] = rho[scored] < tau
    return {"rho": rho, "flag": flag, "scored": scored, "per_grain": per_grain, "tau": tau, "min_rows": min_rows,
            "scored_voxels": int(scored.sum()), "flagged_voxels": int(flag.sum()),
            "unscored_solved_voxels": int(((lab >= 0) & ~scored).sum()), "grains_below_min_rows": n_skipped}
