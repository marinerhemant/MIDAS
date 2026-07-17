"""N7 + N11 unit tests: binning device override and scan-work-dir split."""

from __future__ import annotations

from pathlib import Path

import pytest

from midas_pipeline._pf_scans import iter_pf_scans


def _params(tmp_path: Path, raw: Path) -> Path:
    p = tmp_path / "P.txt"
    p.write_text(
        "FileStem sam\nStartFileNrFirstLayer 100\nNrFilesPerSweep 1\n"
        f"Padding 6\nRawFolder {raw}\n"
    )
    return p


def test_work_dir_splits_raw_from_work(tmp_path: Path):
    raw = tmp_path / "raw_readonly"
    (raw / "100").mkdir(parents=True)
    layer = tmp_path / "LayerNr_1"
    layer.mkdir()
    (layer / "positions.csv").write_text("0.0\n1.0\n")
    work = tmp_path / "work"

    scans = iter_pf_scans(
        params_file=_params(tmp_path, raw), layer_dir=layer, layer_nr=1,
        work_dir=work,
    )
    # scan_dir (Temp/, CSVs, zips-to-build) lives under the work root...
    assert scans[0].scan_dir == work / "100"
    assert scans[1].scan_dir == work / "101"
    # ...and so does the zip that doesn't exist yet.
    assert scans[0].zip_path.parent == work / "100"


def test_work_dir_honours_prebuilt_raw_zip(tmp_path: Path):
    raw = tmp_path / "raw_readonly"
    (raw / "100").mkdir(parents=True)
    prebuilt = raw / "100" / "sam_000100.MIDAS.zip"
    prebuilt.write_bytes(b"zip")
    layer = tmp_path / "LayerNr_1"
    layer.mkdir()
    (layer / "positions.csv").write_text("0.0\n1.0\n")

    scans = iter_pf_scans(
        params_file=_params(tmp_path, raw), layer_dir=layer, layer_nr=1,
        work_dir=tmp_path / "work",
    )
    assert scans[0].zip_path == prebuilt          # read from raw
    assert scans[0].scan_dir == tmp_path / "work" / "100"  # write to work


def test_default_behaviour_unchanged_without_work_dir(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "100").mkdir(parents=True)
    layer = tmp_path / "LayerNr_1"
    layer.mkdir()
    (layer / "positions.csv").write_text("0.0\n1.0\n")
    scans = iter_pf_scans(
        params_file=_params(tmp_path, raw), layer_dir=layer, layer_nr=1,
    )
    assert scans[0].scan_dir == raw / "100"


def test_binning_device_override():
    from midas_pipeline.stages.binning import _binning_device

    class _Cfg:
        device = "cuda"
        binning_device = ""

    class _Ctx:
        config = _Cfg()

    assert _binning_device(_Ctx()) == "cuda"       # inherit by default
    _Cfg.binning_device = "cpu"
    assert _binning_device(_Ctx()) == "cpu"        # explicit override wins
