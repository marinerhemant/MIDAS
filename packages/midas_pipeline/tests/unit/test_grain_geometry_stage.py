"""Stage tests for grain_geometry (FF-only grain-based tx/Wedge refine).

The heavy numerical recovery is proven in
midas_joint_ff_calibrate/tests/test_grain_refine.py. Here we only pin the
stage's dispatch contract: OFF by default, FF-only, and a clean skip when the
grain artifacts are absent.
"""
from __future__ import annotations

from pathlib import Path

from midas_pipeline.config import PipelineConfig, ScanGeometry, GrainGeometryConfig
from midas_pipeline.stages._base import StageContext
from midas_pipeline.stages import grain_geometry


def _ctx(tmp_path: Path, *, run: bool, scan_mode: str = "ff",
         with_files: bool = False) -> StageContext:
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    scan = (ScanGeometry.ff() if scan_mode == "ff"
            else ScanGeometry.pf_uniform(n_scans=4, scan_step_um=5.0, beam_size_um=5.0))
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"), params_file=str(params), scan=scan,
        device="cpu", dtype="float64", n_cpus=2,
        grain_geometry=GrainGeometryConfig(run=run),
    )
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    if with_files:
        (layer_dir / "paramstest.txt").write_text("SpaceGroup 225\n")
        (layer_dir / "Grains.csv").write_text("%NumGrains 0\n")
        (layer_dir / "SpotMatrix.csv").write_text("%header\n")
    return StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir, log_dir=log_dir)


def test_default_off_is_skip(tmp_path):
    res = grain_geometry.run(_ctx(tmp_path, run=False, with_files=True))
    assert res.skipped


def test_pf_mode_skips(tmp_path):
    res = grain_geometry.run(_ctx(tmp_path, run=True, scan_mode="pf", with_files=True))
    assert res.skipped


def test_missing_artifacts_skip(tmp_path):
    """Enabled but no Grains.csv/SpotMatrix.csv → soft skip, not a crash."""
    res = grain_geometry.run(_ctx(tmp_path, run=True, with_files=False))
    assert res.skipped


def test_default_config():
    g = GrainGeometryConfig()
    assert g.run is False
    assert g.refine_params == ("tx",)
    assert g.kind == "angular"
