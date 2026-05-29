"""Cross-detector InputAll.csv merge logic.

Tests global SpotID renumbering + DetID side-car column + merged
paramstest with DetParams rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from midas_pipeline.config import PipelineConfig, ScanGeometry
from midas_pipeline.detector import DetectorConfig
from midas_pipeline.stages import cross_det_merge
from midas_pipeline.stages._base import StageContext


def _write_input_all(path: Path, n: int, base_id: int = 1) -> None:
    """Write `n` synthetic spot rows in InputAll.csv format with header.

    InputAll.csv columns (whitespace-separated):
      0=YLab 1=ZLab 2=Omega 3=GrainRadius 4=SpotID 5=RingNumber 6=Eta 7=Ttheta
    """
    rng = np.random.default_rng(42)
    with path.open("w") as fp:
        fp.write("YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta\n")
        for i in range(n):
            ylab = rng.uniform(-1e5, 1e5)
            zlab = rng.uniform(-1e5, 1e5)
            ome = rng.uniform(-180, 180)
            rrad = rng.uniform(50, 500)
            sid = base_id + i
            ring = int(rng.integers(1, 6))
            eta = rng.uniform(-180, 180)
            tth = rng.uniform(0, 10)
            fp.write(f"{ylab:.6f} {zlab:.6f} {ome:.6f} {rrad:.6f} {sid} {ring} {eta:.6f} {tth:.6f}\n")


def _make_paramstest(path: Path) -> None:
    path.write_text(
        "Lsd 1000000.0\n"
        "BC 1024.0 1024.0\n"
        "tx 0.0\nty 0.0\ntz 0.0\n"
        "RingNumbers 1\nRingNumbers 2\n"
        "RingRadii 100000.0\nRingRadii 110000.0\n"
        "OutputFolder ignored\nResultFolder ignored\n"
    )


def _make_ctx(layer_dir: Path, detectors) -> StageContext:
    params = layer_dir / "params_template.txt"
    _make_paramstest(params)
    cfg = PipelineConfig(
        result_dir=str(layer_dir.parent),
        params_file=str(params),
        scan=ScanGeometry.ff(),
    )
    return StageContext(
        config=cfg, detectors=detectors,
        layer_nr=1, layer_dir=layer_dir,
        log_dir=layer_dir / "logs",
    )


def test_single_detector_appends_detid_column(tmp_path: Path):
    layer_dir = tmp_path / "LayerNr_1"
    layer_dir.mkdir(parents=True)
    _write_input_all(layer_dir / "InputAll.csv", n=37)
    (layer_dir / "paramstest.txt").write_text("Lsd 1\n")

    ctx = _make_ctx(layer_dir, detectors=[
        DetectorConfig(det_id=1, zarr_path="x.zip", lsd=1e6,
                       y_bc=1024, z_bc=1024),
    ])
    result = cross_det_merge.run(ctx)

    assert result.n_total_spots == 37
    assert result.n_per_detector == [37]
    rows = (layer_dir / "InputAll.csv").read_text().splitlines()
    # Header + 37 data rows.
    assert len(rows) == 38
    assert rows[0].split()[-1] == "DetID"
    assert all(int(r.split()[-1]) == 1 for r in rows[1:])


def test_multi_detector_concat_and_renumber(tmp_path: Path):
    layer_dir = tmp_path / "LayerNr_1"
    layer_dir.mkdir(parents=True)

    detectors = [
        DetectorConfig(det_id=1, zarr_path="d1.zip", lsd=1e6,
                       y_bc=1024, z_bc=1024),
        DetectorConfig(det_id=2, zarr_path="d2.zip", lsd=1.005e6,
                       y_bc=1024, z_bc=1024, tx=0.05),
        DetectorConfig(det_id=3, zarr_path="d3.zip", lsd=1.01e6,
                       y_bc=1024, z_bc=1024, ty=0.05),
    ]
    counts = [12, 18, 7]
    for det, n in zip(detectors, counts):
        det_dir = layer_dir / f"Det_{det.det_id}"
        det_dir.mkdir()
        _write_input_all(det_dir / "InputAll.csv", n)
        _write_input_all(det_dir / "InputAllExtraInfoFittingAll.csv", n)
        (det_dir / "paramstest.txt").write_text(
            "Lsd 1000000.0\nBC 1024.0 1024.0\nRingNumbers 1\nRingRadii 100000.0\n"
            f"OutputFolder {det_dir}/Output\nResultFolder {det_dir}/Results\n"
        )

    ctx = _make_ctx(layer_dir, detectors)
    result = cross_det_merge.run(ctx)

    # Counts add up.
    assert result.n_total_spots == sum(counts) == 37
    assert result.n_per_detector == counts

    # Merged InputAll.csv has header + 37 data rows; DetID is the trailing col.
    rows = (layer_dir / "InputAll.csv").read_text().splitlines()
    assert len(rows) == 38
    assert rows[0].split()[-1] == "DetID"
    data_rows = rows[1:]
    sids = [int(r.split()[4]) for r in data_rows]
    assert sids == list(range(1, 38))
    detids = [int(r.split()[-1]) for r in data_rows]
    assert detids[:12] == [1] * 12
    assert detids[12:30] == [2] * 18
    assert detids[30:] == [3] * 7

    # Merged paramstest has DetParams rows.
    text = (layer_dir / "paramstest.txt").read_text()
    for det in detectors:
        assert f"DetParams {det.det_id}" in text
    assert f"OutputFolder {layer_dir}/Output" in text
    assert f"ResultFolder {layer_dir}/Results" in text
