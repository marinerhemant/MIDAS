"""positions.csv parser + scan→spatial mapping."""

from __future__ import annotations

import numpy as np

from midas_pipeline.find_grains import build_scan_grid, read_positions_csv, voxel_to_xy_um


def test_unsorted_positions_argsort_matches_C(tmp_path):
    """Unsorted positions get argsorted ascending; scan_to_spatial is the inverse permutation."""
    arr = np.array([3.0, 1.0, 2.0, 0.0], dtype=np.float64)
    grid = build_scan_grid(arr)
    np.testing.assert_array_equal(grid.spatial_positions, np.array([0.0, 1.0, 2.0, 3.0]))
    # scan_to_spatial[file_idx] = spatial_idx.
    # file_idx 0 has value 3.0 → spatial_idx 3.
    # file_idx 1 has value 1.0 → spatial_idx 1.
    # file_idx 2 has value 2.0 → spatial_idx 2.
    # file_idx 3 has value 0.0 → spatial_idx 0.
    np.testing.assert_array_equal(grid.scan_to_spatial, np.array([3, 1, 2, 0]))


def test_read_positions_csv_round_trip(tmp_path):
    path = tmp_path / "positions.csv"
    path.write_text("0.5\n-0.5\n# comment\n\n1.5\n-1.5\n")
    grid = read_positions_csv(path)
    np.testing.assert_array_equal(grid.file_positions, np.array([0.5, -0.5, 1.5, -1.5]))
    np.testing.assert_array_equal(grid.spatial_positions, np.array([-1.5, -0.5, 0.5, 1.5]))


def test_voxel_to_xy_um_layout():
    grid = build_scan_grid(np.array([0.0, 1.0, 2.0], dtype=np.float64))
    # n_scans=3; voxNr=4 → row=1, col=1 → x=spatial[1]=1.0, y=spatial[1]=1.0.
    x, y = voxel_to_xy_um(4, grid.n_scans, grid.spatial_positions)
    assert x == 1.0 and y == 1.0
    # voxNr=5 → row=1, col=2 → x=spatial[2]=2.0, y=spatial[1]=1.0.
    x, y = voxel_to_xy_um(5, grid.n_scans, grid.spatial_positions)
    assert x == 2.0 and y == 1.0


def test_empty_positions_handled_gracefully():
    grid = build_scan_grid(np.array([], dtype=np.float64))
    assert grid.n_scans == 0
    assert grid.scan_to_spatial.size == 0


def test_scan_projection_is_the_indexer_beam_gate():
    """s = x sin(w) + y cos(w) with x = pos[row], y = pos[col] - the convention verified on real data (the
    sinogram columns of 18 in-field grains within 1.85 scans; y-mirrored forms 12-31 scans off)."""
    import numpy as np
    from midas_pipeline.find_grains._geom import scan_projection_um
    assert np.isclose(scan_projection_um(100.0, -40.0, 90.0), 100.0)     # omega 90: s = x
    assert np.isclose(scan_projection_um(100.0, -40.0, 0.0), -40.0)      # omega 0: s = y
    # the old port's (pos[col], pos[row]) with -x cos + y sin is y-mirrored, not equivalent
    x, y, w = 100.0, -40.0, np.deg2rad(30.0)
    old = -(y) * np.cos(w) + (x) * np.sin(w)
    assert not np.isclose(old, scan_projection_um(x, y, 30.0))
