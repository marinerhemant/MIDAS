"""Voxelmap reconstruction: per-grain map from per-voxel indexer scores.

Port of ``FF_HEDM/workflows/pf_MIDAS.py:346 voxelmap_recon``. Replaces
the legacy ``from calcMiso import GetMisOrientationAngleOMBatch``
import with ``midas_stress.orientation.misorientation_om_batch`` per
the global agent rule that orientation math always flows through
midas-stress (see ``feedback_orientation_from_midas_stress.md``).

For each voxel V, we pick the indexer candidate with the highest
completeness ratio (``nMatched / nExpected``). Then we find which of
the unique-grain orientations its rotation matches within
``max_ang_deg`` via the symmetry-aware misorientation. The per-grain
TIF stores ``top_conf`` at every assigned voxel and zero elsewhere.

The function reads:

- ``{topdir}/Output/IndexBest_all.bin``: consolidated indexer output
  (``int32 nVox``, then ``int32 nSolArr[nVox]``, ``int64 offArr[nVox]``,
  ``float64 records[N, 16]``).
- ``{topdir}/Output/UniqueOrientations.csv`` (the find_grains grain list, not the
  layer-level seed list): 14 cols
  ``[grainID, rowNr, nSpots, startRowNr, listStartPos, OM1..OM9]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np

# midas-stress orientation primitives — never calcMiso.
from .._grain_list import require_grain_list
from midas_stress.orientation import misorientation_om_batch


def _read_indexbest(topdir: Union[str, Path]) -> Tuple[int, np.ndarray, np.ndarray, int, np.ndarray]:
    """Read ``Output/IndexBest_all.bin`` consolidated indexer output.

    Layout (matches ``IndexerScanningOMP`` / legacy ``_read_indexbest``):
    ``int32 nVox``, ``int32 nSolArr[nVox]``, ``int64 offArr[nVox]``,
    ``float64 records[N, 16]``.

    Returns ``(nVox, nSolArr, offArr, header, allVals)``.
    """
    consol_path = Path(topdir) / "Output" / "IndexBest_all.bin"
    with open(consol_path, "rb") as f:
        n_vox = np.frombuffer(f.read(4), dtype=np.int32)[0]
        n_sol_arr = np.frombuffer(f.read(4 * n_vox), dtype=np.int32)
        off_arr = np.frombuffer(f.read(8 * n_vox), dtype=np.int64)
        header = 4 + 4 * n_vox + 8 * n_vox
        all_vals = np.frombuffer(f.read(), dtype=np.double)
    return int(n_vox), n_sol_arr, off_arr, header, all_vals


def voxelmap_recon(
    topdir: Union[str, Path],
    sgnum: int,
    n_scans: int,
    n_grains: int,
    *,
    max_ang_deg: float = 1.0,
    min_conf: float = 0.5,
) -> np.ndarray:
    """Build per-grain reconstructions from per-voxel indexer scores.

    For each voxel V, take the highest-confidence indexer candidate,
    find which of the unique-grain orientations its rotation matches
    within ``max_ang_deg`` (using the midas-stress symmetry-aware
    misorientation), and assign the voxel to that grain when the
    candidate's completeness is at least ``min_conf``.

    Parameters
    ----------
    topdir : path
        Work dir containing ``Output/IndexBest_all.bin`` and
        ``Output/UniqueOrientations.csv``.
    sgnum : int
        Crystallographic space group (1-230).
    n_scans : int
        Voxel-grid side length (the recon is ``n_scans × n_scans``).
    n_grains : int
        Number of unique grains.
    max_ang_deg : float, default 1.0
        Maximum misorientation (degrees) for a voxel-to-grain match.
    min_conf : float, default 0.5
        Minimum completeness (matched / expected) for a voxel's
        top candidate to be considered for assignment.

    Returns
    -------
    ndarray, shape (n_grains, n_scans, n_scans), float32
        Per-grain density: ``top_conf[v]`` at voxels assigned to that
        grain, 0 elsewhere.
    """
    topdir = Path(topdir)
    n_vox, n_sol_arr, off_arr, header, all_vals = _read_indexbest(topdir)

    n_vals_cols = 16  # CONSOLIDATED_VALS_COLS in IndexerScanningOMP

    top_oms = np.zeros((n_vox, 9), dtype=np.float64)
    top_conf = np.zeros(n_vox, dtype=np.float64)
    for v in range(n_vox):
        if n_sol_arr[v] == 0:
            continue
        data_off = int((off_arr[v] - header) // 8)
        sols = all_vals[data_off : data_off + n_sol_arr[v] * n_vals_cols].reshape(
            n_sol_arr[v], n_vals_cols
        )
        confs = sols[:, 15] / np.maximum(sols[:, 14], 1)
        bi = int(np.argmax(confs))
        top_oms[v] = sols[bi, 2:11]
        top_conf[v] = confs[bi]

    grain_oms = np.genfromtxt(require_grain_list(topdir), delimiter=" ")
    if grain_oms.ndim == 1:
        grain_oms = grain_oms.reshape(1, -1)
    grain_oms = grain_oms[:, 5:14]
    if len(grain_oms) != n_grains:
        raise ValueError(f"voxelmap_recon: n_grains={n_grains} but the grain list "
                         f"{require_grain_list(topdir)} has {len(grain_oms)} rows")

    max_ang_rad = np.deg2rad(max_ang_deg)
    ang_to_grain = np.zeros((n_vox, n_grains), dtype=np.float64)
    for g in range(n_grains):
        ang_to_grain[:, g] = misorientation_om_batch(
            top_oms, np.tile(grain_oms[g], (n_vox, 1)), sgnum
        ).flatten()
    best_g = np.argmin(ang_to_grain, axis=1)
    best_ang = np.min(ang_to_grain, axis=1)
    voxel_grain = np.where(
        (best_ang < max_ang_rad) & (top_conf >= min_conf), best_g, -1
    ).astype(np.int32)

    gid_map = voxel_grain.reshape(n_scans, n_scans)
    conf_map = top_conf.reshape(n_scans, n_scans).astype(np.float32)

    all_recons = np.zeros((n_grains, n_scans, n_scans), dtype=np.float32)
    for g in range(n_grains):
        all_recons[g] = np.where(gid_map == g, conf_map, 0.0)
    return all_recons
