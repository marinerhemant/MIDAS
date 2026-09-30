"""Unit tests for midas_pipeline.recon.voxelmap.

Builds a synthetic ``Output/IndexBest_all.bin`` + ``Output/UniqueOrientations.csv``
fixture with 2 grains and a small voxel grid; asserts the recon assigns
each voxel to the grain whose OM matches its top candidate, and zeros
out unassigned voxels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.recon import voxelmap_recon


def _write_indexbest_fixture(tmp_path: Path, n_vox: int, sols_per_vox, all_records):
    """Write IndexBest_all.bin with the layout the reader expects.

    Layout (matches IndexerScanningOMP):
      int32 nVox
      int32 nSolArr[nVox]
      int64 offArr[nVox]   — byte offset into the file at which voxel's
                             records begin (record stride = 16 doubles)
      float64 records[N, 16]
    """
    out_dir = tmp_path / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sol_arr = np.asarray(sols_per_vox, dtype=np.int32)
    header_bytes = 4 + 4 * n_vox + 8 * n_vox
    # offset of v-th voxel's records in bytes
    off_arr = np.empty(n_vox, dtype=np.int64)
    cum = 0
    for v in range(n_vox):
        off_arr[v] = header_bytes + cum * 8
        cum += n_sol_arr[v] * 16

    flat_records = np.asarray(all_records, dtype=np.float64).reshape(-1)

    path = out_dir / "IndexBest_all.bin"
    with open(path, "wb") as f:
        np.asarray([n_vox], dtype=np.int32).tofile(f)
        n_sol_arr.tofile(f)
        off_arr.tofile(f)
        flat_records.tofile(f)
    return path


def _rot_z(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _write_unique_orientations(tmp_path: Path, oms_3x3):
    """14-col format: grainID rowNr nSpots startRowNr listStartPos OM1..OM9."""
    rows = []
    for g, om in enumerate(oms_3x3):
        rows.append(
            [float(g), 0.0, 0.0, 0.0, 0.0] + list(np.asarray(om).flatten())
        )
    data = np.asarray(rows, dtype=np.float64)
    path = tmp_path / "Output" / "UniqueOrientations.csv"   # find_grains' grain list
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt="%.10f", delimiter=" ")
    return path


def test_voxelmap_recon_two_grain_assignment(tmp_path):
    n_scans = 2
    n_vox = n_scans * n_scans  # 4
    # Grain 0 OM: identity. Grain 1 OM: 30° rotation about Z.
    om_g0 = np.eye(3)
    om_g1 = _rot_z(np.deg2rad(30.0))

    # Per-voxel: 1 candidate each. records[i, :] is a length-16 row.
    # Columns 2..10 are OM1..OM9; col 14 is nExpected; col 15 is nMatched.
    # We want completeness = matched / max(expected, 1).
    def _record(om_3x3, conf):
        r = np.zeros(16, dtype=np.float64)
        r[2:11] = np.asarray(om_3x3).flatten()
        r[14] = 10.0    # nExpected
        r[15] = 10.0 * conf
        return r

    # voxel layout (row-major across the 2x2 grid):
    #   v0 = (0,0) -> grain 0  (top-left)
    #   v1 = (0,1) -> grain 0
    #   v2 = (1,0) -> grain 1
    #   v3 = (1,1) -> grain 1
    records = [
        _record(om_g0, 0.9),
        _record(om_g0, 0.8),
        _record(om_g1, 0.95),
        _record(om_g1, 0.85),
    ]
    sols = [1, 1, 1, 1]
    _write_indexbest_fixture(tmp_path, n_vox, sols, records)
    _write_unique_orientations(tmp_path, [om_g0, om_g1])

    all_recons = voxelmap_recon(
        tmp_path, sgnum=225, n_scans=n_scans, n_grains=2,
        max_ang_deg=1.0, min_conf=0.5,
    )
    assert all_recons.shape == (2, 2, 2)

    # Grain 0 voxels are the first row (after reshape); confidences ~0.9, 0.8.
    g0 = all_recons[0]
    g1 = all_recons[1]
    # Each voxel should belong to exactly one grain.
    assigned = (g0 > 0).astype(int) + (g1 > 0).astype(int)
    assert (assigned == 1).all(), "exactly one grain per voxel"
    # First-row voxels go to grain 0, second-row to grain 1.
    assert (g0[0, :] > 0).all()
    assert (g1[1, :] > 0).all()
    assert (g0[1, :] == 0).all()
    assert (g1[0, :] == 0).all()


def test_voxelmap_recon_low_conf_unassigned(tmp_path):
    n_scans = 2
    n_vox = 4
    om = np.eye(3)

    def _record(conf):
        r = np.zeros(16, dtype=np.float64)
        r[2:11] = om.flatten()
        r[14] = 10.0
        r[15] = 10.0 * conf
        return r

    records = [_record(0.3), _record(0.4), _record(0.7), _record(0.9)]
    sols = [1, 1, 1, 1]
    _write_indexbest_fixture(tmp_path, n_vox, sols, records)
    _write_unique_orientations(tmp_path, [om])

    all_recons = voxelmap_recon(
        tmp_path, sgnum=225, n_scans=n_scans, n_grains=1,
        max_ang_deg=1.0, min_conf=0.5,
    )
    # Voxels with conf < 0.5 should be unassigned.
    g0 = all_recons[0].flatten()
    assert (g0[:2] == 0).all(), "low-conf voxels must be unassigned"
    assert (g0[2:] > 0).all(), "high-conf voxels must be assigned"
