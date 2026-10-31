"""preflight: a panel's zip and detectors.json must describe the same geometry (#75 follow-up)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from midas_pipeline.config import PipelineConfig, ScanGeometry
from midas_pipeline.detector import DetectorConfig
from midas_pipeline.preflight import check_inputs


def _zip(path: Path, *, lsd, ybc, zbc, tx, ty=0.0, tz=0.0, wedge=0.0, omega_start=-30.0) -> None:
    with zarr.ZipStore(str(path), mode="w") as store:
        root = zarr.group(store=store, overwrite=True)
        ap = root.require_group("analysis/process/analysis_parameters")
        for k, v in dict(Lsd=lsd, YCen=ybc, ZCen=zbc, tx=tx, ty=ty, tz=tz,
                       Wedge=wedge, OmegaStart=omega_start).items():
            ap.create_dataset(k, data=np.array([v], dtype=np.float64))
        ap.create_dataset("SkipFrame", data=np.array([0], dtype=np.int32))
        sp = root.require_group("measurement/process/scan_parameters")
        sp.create_dataset("start", data=np.array([omega_start], dtype=np.float64))
        sp.create_dataset("step", data=np.array([0.25], dtype=np.float64))


def _cfg(tmp_path: Path, geoms, json_geoms, params_text="Lsd 1\n"):
    params = tmp_path / "Parameters.txt"
    params.write_text(params_text)
    dets = []
    for i, (g, j) in enumerate(zip(geoms, json_geoms), start=1):
        z = tmp_path / f"p{i}.MIDAS.zip"
        _zip(z, **g)
        dets.append(DetectorConfig(det_id=i, zarr_path=str(z), **j))
    dj = tmp_path / "detectors.json"
    DetectorConfig.dump_many(dets, dj)
    cfg = PipelineConfig(result_dir=str(tmp_path), params_file=str(params),
                         scan=ScanGeometry.ff())
    cfg.detectors_json = str(dj)
    return cfg


G1 = dict(lsd=3.3e6, ybc=2300.0, zbc=2150.0, tx=15.0)
G2 = dict(lsd=3.42e6, ybc=2290.5, zbc=2160.25, tx=105.0, ty=0.1, tz=-0.2)


def _json(g):
    return dict(lsd=g["lsd"], y_bc=g["ybc"], z_bc=g["zbc"], tx=g["tx"],
                ty=g.get("ty", 0.0), tz=g.get("tz", 0.0))


def test_matching_geometry_passes(tmp_path):
    assert check_inputs(_cfg(tmp_path, [G1, G2], [_json(G1), _json(G2)])) == []


def test_lsd_beam_centre_and_tilt_mismatches_are_named(tmp_path):
    bad = _json(G2)
    bad.update(lsd=3.3e6, y_bc=2300.0, tx=104.0)
    problems = check_inputs(_cfg(tmp_path, [G1, G2], [_json(G1), bad]))
    assert len(problems) == 1 and problems[0].startswith("detector 2:")
    for k in ("Lsd", "y_bc", "tx"):
        assert k in problems[0]
    assert "z_bc" not in problems[0]


def test_small_rounding_is_tolerated(tmp_path):
    j = _json(G2)
    j.update(lsd=G2["lsd"] + 1.0, y_bc=G2["ybc"] + 0.01, tx=G2["tx"] + 0.001)
    assert check_inputs(_cfg(tmp_path, [G1, G2], [_json(G1), j])) == []


def test_unreadable_zip_is_skipped_not_an_error(tmp_path):
    cfg = _cfg(tmp_path, [G1, G2], [_json(G1), _json(G2)])
    Path(DetectorConfig.load_many(cfg.detectors_json)[1].zarr_path).write_bytes(b"not a zarr")
    assert check_inputs(cfg) == []


def _cfg2(tmp_path, params_text, g2=None):
    return _cfg(tmp_path, [G1, g2 or G2], [_json(G1), _json(g2 or G2)], params_text)


def test_wedge_and_omegastart_the_zip_does_not_hold_stop_the_run(tmp_path):
    cfg = _cfg2(tmp_path, "Lsd 1\nWedge -0.296\nOmegaStart -30.65\nOmegaStep 0.25\n")
    problems = check_inputs(cfg)
    assert len(problems) == 1
    msg = problems[0]
    assert "Wedge" in msg and "OmegaStart" in msg
    assert "p1.MIDAS.zip" in msg and "p2.MIDAS.zip" in msg
    assert "midas-refresh-zarr-params" in msg and "--keys" in msg
    assert "--accept-zip-params" in msg and "Nothing was run" in msg


def test_accept_zip_params_runs_on_the_zip_values_and_says_so(tmp_path, caplog):
    cfg = _cfg2(tmp_path, "Lsd 1\nWedge -0.296\n")
    cfg.accept_zip_params = True
    with caplog.at_level("WARNING"):
        assert check_inputs(cfg) == []
    assert "--accept-zip-params" in caplog.text and "Wedge" in caplog.text


def test_file_that_agrees_with_the_zips_or_omits_the_keys_passes(tmp_path):
    assert check_inputs(_cfg2(tmp_path, "Lsd 1\n")) == []
    ok = _cfg2(tmp_path, "Lsd 1\nWedge 0\nOmegaStart -30\nOmegaStep 0.25\n")
    assert check_inputs(ok) == []


def test_only_the_panel_that_differs_is_named(tmp_path):
    cfg = _cfg2(tmp_path, "Lsd 1\nWedge 0\n", g2=dict(G2, wedge=0.1))
    msg = check_inputs(cfg)[0]
    assert "p2.MIDAS.zip" in msg and "p1.MIDAS.zip" not in msg


def test_per_panel_thresholds_warn_but_do_not_stop(tmp_path, caplog):
    cfg = _cfg2(tmp_path, "Lsd 1\nRingThresh 1 55\nMaxRingRad 111111\n")
    with caplog.at_level("WARNING"):
        assert check_inputs(cfg) == []
    assert "RingThresh" in caplog.text and "MaxRingRad" in caplog.text
