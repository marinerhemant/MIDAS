"""binning must say so when it overwrites a positions.csv that disagrees with --scan-step."""
import logging

import numpy as np

from midas_pipeline.stages.binning import _warn_if_positions_differ


def test_warns_on_descending_file_vs_ascending_geometry(tmp_path, caplog):
    np.savetxt(tmp_path / "positions.csv", [600.0, 590.0, 580.0])
    with caplog.at_level(logging.WARNING):
        assert _warn_if_positions_differ(tmp_path, [-600.0, -590.0, -580.0])
    assert "OVERWRITTEN" in caplog.text


def test_silent_when_file_matches(tmp_path, caplog):
    np.savetxt(tmp_path / "positions.csv", [-10.0, 0.0, 10.0])
    with caplog.at_level(logging.WARNING):
        assert not _warn_if_positions_differ(tmp_path, [-10.0, 0.0, 10.0])
    assert caplog.text == ""


def test_silent_when_no_file(tmp_path):
    assert not _warn_if_positions_differ(tmp_path, [0.0])
