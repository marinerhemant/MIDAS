"""DetectorConfig: JSON load, paramstest fallback."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from midas_pipeline.detector import DetectorConfig


def test_load_many_from_json(tmp_path: Path):
    payload = [
        {"det_id": 1, "zarr_path": "d1.zip", "lsd": 1e6,
         "y_bc": 1024.0, "z_bc": 1024.0,
         "tx": 0.0, "ty": 0.0, "tz": 0.0,
         "p_distortion": [0.0] * 11},
        {"det_id": 2, "zarr_path": "d2.zip", "lsd": 1.005e6,
         "y_bc": 1024.0, "z_bc": 1024.0,
         "tx": 0.05, "ty": 0.0, "tz": 0.0,
         "p_distortion": [0.0] * 11},
    ]
    f = tmp_path / "detectors.json"
    f.write_text(json.dumps(payload))
    dets = DetectorConfig.load_many(f)
    assert len(dets) == 2
    assert dets[0].det_id == 1 and dets[1].det_id == 2
    assert dets[1].tx == 0.05


def test_load_from_paramstest_detparams_rows(tmp_path: Path):
    ps = tmp_path / "ps.txt"
    ps.write_text(
        "DetParams 1 1000000.0 1024.0 1024.0 0.0 0.0 0.0 0 0 0 0 0 0 0 0 0 0 0\n"
        "DetParams 2 1005000.0 1024.0 1024.0 0.05 0.0 0.0 0 0 0 0 0 0 0 0 0 0 0\n"
    )
    dets = DetectorConfig.load_from_paramstest(ps, zarr_path="shared.zip")
    assert len(dets) == 2
    assert dets[0].lsd == 1000000.0 and dets[1].lsd == 1005000.0
    assert dets[1].tx == 0.05
    assert dets[0].zarr_path == "shared.zip"


def test_single_from_paramstest_global_keys(tmp_path: Path):
    ps = tmp_path / "ps.txt"
    ps.write_text(
        "Lsd 1000000.0;\nBC 1024.0 1024.0;\ntx 0.05;\nty 0.0;\ntz 0.0;\n"
    )
    det = DetectorConfig.single_from_paramstest(ps, zarr_path="x.zip")
    assert det.det_id == 1 and det.lsd == 1000000.0
    assert det.y_bc == 1024.0 and det.z_bc == 1024.0
    assert det.tx == 0.05


def test_single_from_paramstest_missing_lsd(tmp_path: Path):
    ps = tmp_path / "ps.txt"
    ps.write_text("BC 1024 1024\n")
    with pytest.raises(ValueError):
        DetectorConfig.single_from_paramstest(ps)


def test_dump_load_roundtrip(tmp_path: Path):
    dets = [
        DetectorConfig(det_id=1, zarr_path="a.zip", lsd=1e6,
                       y_bc=1024.0, z_bc=1024.0),
        DetectorConfig(det_id=2, zarr_path="b.zip", lsd=1.01e6,
                       y_bc=1024.0, z_bc=1024.0, tx=0.1),
    ]
    f = tmp_path / "out.json"
    DetectorConfig.dump_many(dets, f)
    rt = DetectorConfig.load_many(f)
    assert [d.det_id for d in rt] == [1, 2]
    assert rt[1].tx == 0.1
