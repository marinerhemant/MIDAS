"""The PF voxel grid as a file: one source of truth for every stage.

Historically every PF consumer re-derived the grid as sorted(positions) x
sorted(positions), which is only right for a full, axis-centred scan. A half
scan (``ScanGeometry.voxel_grid == "symmetric"``) needs a grid over the whole
[-R, R] disc, so the grid is written once per layer to ``VoxelGrid.txt`` and
read back from there -- by the C indexer (``VoxelGridFile``) and by the
Python stages that need voxel coordinates.

Row ``v`` of the file is voxel ``v`` of ``IndexBest_all.bin``, in the C
indexer's order (``v = i * n + j``, ``x = axis[i]``, ``y = axis[j]``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

VOXEL_GRID_NAME = "VoxelGrid.txt"


def write_voxel_grid(layer_dir: Path, scan) -> Path:
    """Write ``<layer_dir>/VoxelGrid.txt`` from a ``ScanGeometry``."""
    xy = scan.voxel_grid_xy()
    dst = Path(layer_dir) / VOXEL_GRID_NAME
    header = (f"# PF voxel grid: {scan.voxel_grid}, n_grid={scan.n_grid}, "
              f"n_scans={scan.n_scans}. Columns: x_um y_um\n")
    dst.write_text(header + "".join(f"{x:.9f} {y:.9f}\n" for x, y in xy))
    return dst


def read_voxel_grid(layer_dir: Path) -> Optional[np.ndarray]:
    """``(nVoxels, 2)`` voxel centres, or ``None`` when the layer uses the
    historical positions x positions grid (no file)."""
    f = Path(layer_dir) / VOXEL_GRID_NAME
    if not f.is_file():
        return None
    xy = np.loadtxt(f, comments="#", dtype=np.float64, ndmin=2)
    if xy.shape[1] != 2:
        raise ValueError(f"{f}: expected 2 columns (x y), got {xy.shape[1]}")
    return xy


def grid_side(xy: np.ndarray) -> int:
    """Voxels per side of a square grid; raises if it is not square."""
    n = int(round(np.sqrt(xy.shape[0])))
    if n * n != xy.shape[0]:
        raise ValueError(f"voxel grid has {xy.shape[0]} voxels: not square")
    return n


def require_positions_grid(layer_dir: Path, stage: str) -> None:
    """Refuse, loudly, to run a stage that still assumes the historical grid.

    Tomographic stages derive geometry from n_scans and a centred rotation
    axis; on a half-scan grid they would produce plausible-looking garbage.
    """
    if (Path(layer_dir) / VOXEL_GRID_NAME).is_file():
        raise NotImplementedError(
            f"{stage}: not implemented for a non-'positions' voxel grid "
            f"({VOXEL_GRID_NAME} present, e.g. a half scan). Disable this "
            f"stage (tomography stays off for point-by-point maps).")
