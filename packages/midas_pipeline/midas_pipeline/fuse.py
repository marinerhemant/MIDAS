"""Bayesian fusion + sino mask helpers (port of pf_MIDAS.py:100-260).

This module ports three functions:

- ``_orient_score_per_grain`` — per-(voxel, grain) likelihood from the
  consolidated indexer output. Uses ``midas_stress.orientation.misorientation_om_batch``
  in place of the legacy ``calcMiso.GetMisOrientationAngleOMBatch``.
- ``bayesian_fusion`` — fuses per-grain tomographic shape with the
  per-voxel orientation likelihood; returns a posterior tensor.
- ``mask_sino_by_assignment`` — drops sinogram cells that no
  assigned-to-grain-g voxel can produce (Friedel-symmetric form).

The functions are numpy-only; the indexer output and grain CSV are
already numpy on the read path. Differentiability is not needed here
because there is no learning signal flowing through Bayes/fusion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from midas_stress.orientation import misorientation_om_batch

from ._grain_list import require_grain_list
from .recon.voxelmap import _read_indexbest

LOG = logging.getLogger("midas_pipeline.fuse")


def _orient_score_per_grain(
    topdir: Union[str, Path],
    sgnum: int,
    n_scans: int,
    n_grains: int,
    max_ang_deg: float,
    min_conf: float,
) -> np.ndarray:
    """Per-(voxel, grain) orientation score.

    For each (voxel V, grain g), the score is the highest completeness
    among V's indexer candidates whose orientation matches grain g
    within ``max_ang_deg``. Voxels at grain boundaries can legitimately
    score non-zero for multiple grains.

    Returns
    -------
    ndarray, shape (n_grains, n_scans, n_scans), float32.
    """
    topdir = Path(topdir)
    n_vox, n_sol_arr, off_arr, header, all_vals = _read_indexbest(topdir)
    n_vals_cols = 16

    cand_oms = []
    cand_confs = []
    cand_voxels = []
    for v in range(n_vox):
        if n_sol_arr[v] == 0:
            continue
        data_off = int((off_arr[v] - header) // 8)
        sols = all_vals[data_off : data_off + n_sol_arr[v] * n_vals_cols].reshape(
            n_sol_arr[v], n_vals_cols
        )
        confs = sols[:, 15] / np.maximum(sols[:, 14], 1)
        keep = confs >= min_conf
        if not keep.any():
            continue
        for ci in np.where(keep)[0]:
            cand_oms.append(sols[ci, 2:11])
            cand_confs.append(confs[ci])
            cand_voxels.append(v)
    if not cand_oms:
        return np.zeros((n_grains, n_scans, n_scans), dtype=np.float32)

    cand_oms_arr = np.asarray(cand_oms, dtype=np.float64)
    cand_confs_arr = np.asarray(cand_confs, dtype=np.float64)
    cand_voxels_arr = np.asarray(cand_voxels, dtype=np.int64)

    grain_oms = np.genfromtxt(require_grain_list(topdir), delimiter=" ")
    if grain_oms.ndim == 1:
        grain_oms = grain_oms.reshape(1, -1)
    grain_oms = grain_oms[:, 5:14]

    max_ang_rad = np.deg2rad(max_ang_deg)
    orient = np.zeros((n_grains, n_scans * n_scans), dtype=np.float32)
    n_cands = cand_oms_arr.shape[0]
    for g in range(n_grains):
        angs = misorientation_om_batch(
            cand_oms_arr, np.tile(grain_oms[g], (n_cands, 1)), sgnum
        ).flatten()
        ok = angs < max_ang_rad
        if not ok.any():
            continue
        for cand_idx in np.where(ok)[0]:
            v = int(cand_voxels_arr[cand_idx])
            c = float(cand_confs_arr[cand_idx])
            if c > orient[g, v]:
                orient[g, v] = c
    return orient.reshape(n_grains, n_scans, n_scans)


def bayesian_fusion(
    all_recons: np.ndarray,
    topdir: Union[str, Path],
    sgnum: int,
    n_grains: int,
    *,
    max_ang_deg: float = 1.0,
    min_conf: float = 0.5,
) -> np.ndarray:
    """Fuse per-grain shape (tomographic) with per-voxel orientation likelihood.

    ``posterior(g, V) ∝ shape_score(g, V) × orient_score(g, V)``

    ``shape_score`` is the per-grain recon normalized to ``[0, 1]``
    across its own voxels (so a small grain isn't drowned by a large
    one). ``orient_score(g, V)`` is the highest completeness among
    V's indexer candidates whose OM is within ``max_ang_deg`` of
    grain g's OM.

    Parameters
    ----------
    all_recons : ndarray, shape (n_grains, n_scans, n_scans), float
        Per-grain tomographic reconstruction.
    topdir : path
        Layer directory containing ``Output/IndexBest_all.bin`` and
        ``Output/UniqueOrientations.csv`` (the find_grains grain list).
    sgnum : int
    n_grains : int
    max_ang_deg, min_conf : float

    Returns
    -------
    ndarray, shape (n_grains, n_scans, n_scans), float32.
    """
    n_grs, n_scans, _ = all_recons.shape
    if n_grs != n_grains:
        raise ValueError(
            f"bayesian_fusion: all_recons has {n_grs} grains, "
            f"expected {n_grains}"
        )
    LOG.info(
        "bayesian_fusion: n_grains=%d n_scans=%d max_ang=%g min_conf=%g",
        n_grains, n_scans, max_ang_deg, min_conf,
    )

    shape = all_recons.astype(np.float32).copy()
    for g in range(n_grains):
        m = shape[g].max()
        if m > 0:
            shape[g] /= m

    orient = _orient_score_per_grain(
        topdir, sgnum, n_scans, n_grains, max_ang_deg, min_conf
    )

    posterior = (shape * orient).astype(np.float32)
    for g in range(n_grains):
        LOG.info(
            "  G%d: shape>0 voxels=%d, orient>0 voxels=%d, posterior>0 voxels=%d",
            g,
            int((shape[g] > 0).sum()),
            int((orient[g] > 0).sum()),
            int((posterior[g] > 0).sum()),
        )
    return posterior


def mask_sino_by_assignment(
    sinos_main: np.ndarray,
    omegas_arr: np.ndarray,
    nr_hkls_arr: np.ndarray,
    max_id: np.ndarray,
    n_grains: int,
    n_scans: int,
    spatial_pos: np.ndarray,
    *,
    scan_tol: float = 1.5,
) -> np.ndarray:
    """Hard-assignment sino cleanup using a Friedel-symmetric scan filter.

    For each grain g and its existing per-grain sino, keep only cells
    ``(omega_hkl_i, scan_j)`` where at least one voxel V assigned to
    grain g (``max_id[V] == g``) projects to that cell:

        ``|s_V(omega_hkl_i) - spatial_pos[scan_j]| < scan_tol`` OR
        ``|s_V(omega_hkl_i) + spatial_pos[scan_j]| < scan_tol`` (Friedel)

    where ``s_V(omega)`` is ``find_grains._geom.scan_projection_um`` (x = spatial_pos[row],
    y = spatial_pos[col]; ``x sin(omega) + y cos(omega)``, the indexer's beam gate). Until
    2026-09-28 this used ``-x cos + y sin`` with x = spatial_pos[col] - y mirrored - and kept
    ~24 % of each grain's own sinogram intensity on real data (~86 % now).

    This drops residual cells the C-side scan-pos filter couldn't
    catch because the hard voxel→grain assignment didn't exist yet;
    after Bayesian fusion we have it. The result feeds FBP again for
    streak-free per-grain density TIFs.

    Parameters
    ----------
    sinos_main : ndarray, shape (n_grains, max_NHKLs, n_scans), float64
    omegas_arr : ndarray, shape (n_grains, max_NHKLs), float64 (degrees)
    nr_hkls_arr : ndarray, shape (n_grains,), int
    max_id : ndarray, shape (n_scans, n_scans), int32
        Per-voxel grain assignment; -1 means unassigned.
    n_grains : int
    n_scans : int
    spatial_pos : ndarray, shape (n_scans,), float
        Sorted scan positions in micrometers.
    scan_tol : float, default 1.5

    Returns
    -------
    ndarray, same shape as ``sinos_main``, masked.
    """
    refined = np.zeros_like(sinos_main)
    gid_2d = max_id
    for g in range(n_grains):
        n_sp = int(nr_hkls_arr[g])
        if n_sp == 0:
            continue
        sino_orig = sinos_main[g, :n_sp, :]
        th = omegas_arr[g, :n_sp]
        rows, cols = np.where(gid_2d == g)
        if len(rows) == 0:
            LOG.info("  step5 G%d: no voxels assigned, sino zeroed", g)
            continue
        x_v = spatial_pos[rows]                  # voxel_grid.csv / indexer layout
        y_v = spatial_pos[cols]
        n_vox = len(x_v)

        from .find_grains._geom import scan_projection_um
        s_proj = scan_projection_um(x_v[None, :], y_v[None, :], th[:, None])  # (nSp, n_vox)

        scan_pos = spatial_pos[None, None, :]
        mask = np.zeros((n_sp, n_scans), dtype=bool)
        batch = 256
        for vstart in range(0, n_vox, batch):
            vend = min(vstart + batch, n_vox)
            sp = s_proj[:, vstart:vend, None]
            ok = (np.abs(sp - scan_pos) < scan_tol) | (
                np.abs(-sp - scan_pos) < scan_tol
            )
            mask |= ok.any(axis=1)
        refined[g, :n_sp, :] = sino_orig * mask
        n_orig = int((sino_orig > 0).sum())
        n_keep = int((refined[g, :n_sp, :] > 0).sum())
        LOG.info(
            "  step5 G%d: %d assigned voxels, sino cells %d → %d (%d%%)",
            g, n_vox, n_orig, n_keep, 100 * n_keep // max(n_orig, 1),
        )
    return refined
