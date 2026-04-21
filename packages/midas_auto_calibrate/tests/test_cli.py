"""CLI smoke tests for midas-auto-calibrate.

Unit tests for the Parameters.txt → CalibrationConfig path. End-to-end CLI
tests against the bundled calibrant are skipped without the MIDASCalibrant
binary and would duplicate test_progressive coverage anyway.
"""

from __future__ import annotations

import pytest

from midas_auto_calibrate._config import CalibrationConfig
from midas_auto_calibrate.cli import _config_from_params, _parse_params_file


def test_parse_params_basic(tmp_path):
    p = tmp_path / "ps.txt"
    p.write_text(
        "# comment\n"
        "Wavelength 0.172973\n"
        "px 172\n"
        "Lsd 1000000\n"
        "BC 500 600\n"
        "NrPixelsY 2048\n"
        "NrPixelsZ 2048\n"
        "LatticeConstant 5.4116 5.4116 5.4116 90 90 90\n"
        "ImTransOpt 2\n"
        "tolTilts 2\n"
    )
    out = _parse_params_file(p)
    assert out["Wavelength"] == "0.172973"
    assert out["BC"] == "500 600"
    assert out["LatticeConstant"].startswith("5.4116")


def test_parse_params_list_values(tmp_path):
    p = tmp_path / "ps.txt"
    p.write_text(
        "RingsToExclude 19\n"
        "RingsToExclude 20\n"
        "RingsToExclude 21\n"
    )
    out = _parse_params_file(p)
    assert out["RingsToExclude"] == [["19"], ["20"], ["21"]]


def test_config_from_params_minimal(tmp_path):
    p = tmp_path / "ps.txt"
    p.write_text(
        "Wavelength 0.172\n"
        "px 172\n"
        "Lsd 500000\n"
        "BC 100 200\n"
        "NrPixelsY 1024\n"
        "NrPixelsZ 1024\n"
        "LatticeConstant 5.4116 5.4116 5.4116 90 90 90\n"
        "ImTransOpt 2\n"
        "tolTilts 3\n"
    )
    cfg = _config_from_params(_parse_params_file(p))
    assert isinstance(cfg, CalibrationConfig)
    assert cfg.wavelength == pytest.approx(0.172)
    assert cfg.pixel_size == pytest.approx(172.0)
    assert cfg.ybc == 100.0
    assert cfg.zbc == 200.0
    assert cfg.nr_pixels_y == 1024
    assert cfg.im_trans_opt == [2]
    # Unrecognised keys flow through to extra_params as floats when parseable.
    assert cfg.extra_params["tolTilts"] == pytest.approx(3.0)
