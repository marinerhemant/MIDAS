"""Half-scan support: the explicit voxel grid and its wiring into the indexer."""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.config import ScanGeometry
from midas_pipeline.stages._voxel_grid import (
    VOXEL_GRID_NAME,
    grid_side,
    read_voxel_grid,
    require_positions_grid,
    write_voxel_grid,
)


def _half_scan(**kw):
    # 9 positions from just across the axis out to +35 um: an ESRF-style half scan
    return ScanGeometry.pf_uniform(n_scans=9, scan_step_um=5.0, beam_size_um=5.0,
                                   start_um=-5.0, **kw)


def test_default_grid_is_positions_x_positions_in_c_order():
    sg = ScanGeometry.pf_uniform(n_scans=5, scan_step_um=2.0, beam_size_um=2.0)
    pos = np.sort(sg.scan_positions)
    xy = sg.voxel_grid_xy()
    assert sg.voxel_grid == "positions" and sg.n_grid == sg.n_scans
    # IndexerUnified.c: v = i * n + j, x = pos[i], y = pos[j]
    for v in range(25):
        i, j = divmod(v, 5)
        assert xy[v, 0] == pos[i] and xy[v, 1] == pos[j]


def test_symmetric_grid_spans_the_disc_of_a_half_scan():
    sg = _half_scan(voxel_grid="symmetric")
    ax = sg.voxel_grid_axis()
    assert ax[0] == -35.0 and ax[-1] == 35.0
    assert np.allclose(np.diff(ax), 5.0)
    assert sg.n_grid == 15 and sg.n_scans == 9
    # the historical grid only covers the scanned square
    assert _half_scan().voxel_grid_axis().min() == -5.0


def test_bad_voxel_grid_name_is_refused():
    with pytest.raises(ValueError):
        _half_scan(voxel_grid="halfdisc")


def test_voxel_grid_file_round_trips(tmp_path):
    sg = _half_scan(voxel_grid="symmetric")
    write_voxel_grid(tmp_path, sg)
    xy = read_voxel_grid(tmp_path)
    assert np.array_equal(xy, sg.voxel_grid_xy())
    assert grid_side(xy) == 15
    assert read_voxel_grid(tmp_path / "nope") is None


def test_tomo_stages_refuse_a_non_positions_grid(tmp_path):
    require_positions_grid(tmp_path, "reconstruct")          # no file: fine
    write_voxel_grid(tmp_path, _half_scan(voxel_grid="symmetric"))
    with pytest.raises(NotImplementedError, match="reconstruct"):
        require_positions_grid(tmp_path, "reconstruct")


def _ctx(scan, layer_dir):
    return types.SimpleNamespace(config=types.SimpleNamespace(scan=scan),
                                 layer_dir=layer_dir)


def test_indexer_paramstest_untouched_when_features_off(tmp_path):
    from midas_pipeline.stages.indexing import _ensure_halfscan_keys_in_paramstest
    pp = tmp_path / "paramstest.txt"
    pp.write_text("Wavelength 0.2;\nScanPosTol 1.0;\n")
    before = pp.read_bytes()
    _ensure_halfscan_keys_in_paramstest(_ctx(_half_scan(), tmp_path), pp, tmp_path)
    assert pp.read_bytes() == before
    assert not (tmp_path / VOXEL_GRID_NAME).exists()


def test_indexer_paramstest_gets_halfscan_keys_once(tmp_path):
    from midas_pipeline.stages.indexing import _ensure_halfscan_keys_in_paramstest
    pp = tmp_path / "paramstest.txt"
    pp.write_text("Wavelength 0.2;\nVoxelGridFile /stale/path\n")
    scan = _half_scan(voxel_grid="symmetric", coverage_aware_completeness=True)
    for _ in range(2):  # idempotent: a resumed run must not stack copies
        _ensure_halfscan_keys_in_paramstest(_ctx(scan, tmp_path), pp, tmp_path)
    lines = pp.read_text().splitlines()
    grid_lines = [l for l in lines if l.startswith("VoxelGridFile ")]
    assert grid_lines == [f"VoxelGridFile {(tmp_path / VOXEL_GRID_NAME).resolve()}"]
    assert lines.count("PFCoverageAwareCompleteness 1") == 1
    assert "Wavelength 0.2;" in lines
    assert read_voxel_grid(tmp_path).shape == (225, 2)
