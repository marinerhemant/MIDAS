"""End-to-end smoke tests for find_grains_single / find_grains_multiple."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from midas_pipeline.find_grains import (
    find_grains_multiple,
    find_grains_single,
    write_ids_bin,
    write_keys_bin,
    write_vals_bin,
)


def _build_min_workdir(tmp_path, *, n_scans=2, axis_z_om):
    """Build a minimal n_scans×n_scans-voxel work_dir with consolidated files
    and Spots.bin so find_grains_single/multiple can run end-to-end.
    """
    work = tmp_path / "work"
    out_dir = work / "Output"
    out_dir.mkdir(parents=True)

    n_vox = n_scans * n_scans
    # Each voxel: 1 candidate with identity OM, conf=1.0, 2 matched spots.
    OM = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64)
    vals_per_vox: list[np.ndarray] = []
    keys_per_vox: list[np.ndarray] = []
    ids_per_vox: list[np.ndarray] = []
    for v in range(n_vox):
        row = np.zeros(16, dtype=np.float64)
        row[2:11] = OM
        row[14] = 2.0
        row[15] = 2.0
        vals_per_vox.append(row.reshape(1, 16))
        keys_per_vox.append(np.array([[v + 1, 2, 2, 0]], dtype=np.uint64))
        ids_per_vox.append(np.array([1, 2], dtype=np.int32))
    write_vals_bin(out_dir / "IndexBest_all.bin", vals_per_vox)
    write_keys_bin(out_dir / "IndexKey_all.bin", keys_per_vox)
    write_ids_bin(out_dir / "IndexBest_IDs_all.bin", ids_per_vox)

    # Spots.bin: 2 spots, each at one scan position.
    spots = np.zeros((2, 10), dtype=np.float64)
    spots[0] = [10, 20, 30.0, 1000, 1, 1, 15.0, 5.0, 1.0, 0]
    spots[1] = [11, 21, 31.0, 2000, 2, 1, 16.0, 5.0, 1.0, 1]
    (out_dir / "Spots.bin").write_bytes(spots.tobytes())

    # positions.csv: one Y per scan.
    pos = np.arange(n_scans, dtype=np.float64) - n_scans / 2
    (work / "positions.csv").write_text("\n".join(str(p) for p in pos))
    return work, out_dir


def test_find_grains_single_e2e_writes_outputs(tmp_path):
    work, out_dir = _build_min_workdir(tmp_path, n_scans=2, axis_z_om=None)
    art = find_grains_single(
        work_dir=work,
        space_group=225,
        sino_mode="tolerance",
        cluster_misorientation_deg=1.0,
        tol_ome_deg=1.0,
        tol_eta_deg=1.0,
    )
    # All 4 voxels share the same OM → global_cluster collapses to 1 grain.
    assert art.n_unique_grains == 1
    assert (out_dir / "UniqueOrientations.csv").exists()
    assert (out_dir / "UniqueIndexSingleKey.bin").exists()
    # Sinogen artifacts must exist.
    assert art.sinogen is not None
    assert Path(art.sinogen.omegas_path).exists()


def test_find_grains_single_emits_voxel_grid_csv(tmp_path):
    """P9 TODO(a): find_grains_single writes Output/voxel_grid.csv with
    columns (voxel_idx, x_um, y_um, z_um, grain_id).  Used by refine_vmap."""
    work, out_dir = _build_min_workdir(tmp_path, n_scans=2, axis_z_om=None)
    art = find_grains_single(
        work_dir=work, space_group=225,
        sino_mode="tolerance",
        cluster_misorientation_deg=1.0,
        tol_ome_deg=1.0, tol_eta_deg=1.0,
    )
    vg = out_dir / "voxel_grid.csv"
    assert vg.exists()
    arr = np.loadtxt(vg, comments="#", skiprows=1)
    assert arr.shape == (4, 5)
    # All 4 voxels collapse to a single grain (id 0) -- they share the same OM.
    assert (arr[:, 0] == [0, 1, 2, 3]).all()
    # positions.csv has Y values [-1, 0]; lab xy follows (positions[i], positions[j])
    # with i = v // n_scans, j = v % n_scans
    expected_xy = np.array([[-1, -1], [-1, 0], [0, -1], [0, 0]], dtype=np.float64)
    np.testing.assert_allclose(arr[:, 1:3], expected_xy)
    assert (arr[:, 3] == 0).all()
    assert (arr[:, 4] == 0).all()
    # Sanity: also matches the artifact's reported grain count.
    assert art.n_unique_grains == 1


def test_find_grains_multiple_e2e_writes_spotsToIndex(tmp_path):
    work, out_dir = _build_min_workdir(tmp_path, n_scans=2, axis_z_om=None)
    art = find_grains_multiple(
        work_dir=work,
        space_group=225,
        cluster_misorientation_deg=1.0,
    )
    assert art.spots_to_index_csv
    assert (work / "SpotsToIndex.csv").exists()
    # One row per voxel × clusters; here each voxel has 1 cluster.
    n_lines = sum(1 for _ in (work / "SpotsToIndex.csv").read_text().splitlines())
    assert n_lines == 4


