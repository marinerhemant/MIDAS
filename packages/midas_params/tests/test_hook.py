"""Tests for the workflow preflight validation hook."""

from __future__ import annotations

import io
import logging
import textwrap
from contextlib import redirect_stderr

import pytest

from midas_params.hook import preflight_validate


def test_skip_short_circuits(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 999\n")
    # Guaranteed-error file, but skip=True must return True
    assert preflight_validate(str(fn), "ff", skip=True) is True


def test_missing_file_non_strict(tmp_path):
    # Returns True (continue) in non-strict mode
    assert preflight_validate(str(tmp_path / "nope.txt"), "ff", strict=False) is True


def test_missing_file_strict(tmp_path):
    # Returns False (caller should exit) in strict mode
    assert preflight_validate(str(tmp_path / "nope.txt"), "ff", strict=True) is False


def test_unknown_pipeline_still_returns_true(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("# empty\n")
    # Garbage pipeline name — should warn and continue
    assert preflight_validate(str(fn), "garbage", strict=False) is True


def test_valid_file_returns_true(tmp_path):
    """A file with no errors/warnings lets the pipeline continue."""
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    for i in range(1, 11):
        (rawdir / f"exp_{i:06d}.ge3").touch()
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent(f"""
        RawFolder {rawdir}
        FileStem exp
        Ext .ge3
        Padding 6
        StartNr 1
        EndNr 10
        Lsd 1000000
        BC 1022 1022
        px 200
        NrPixels 2048
        Wavelength 0.22291
        LatticeConstant 4.08 4.08 4.08 90 90 90
        SpaceGroup 225
        OmegaStart 180
        OmegaEnd -180
        OmegaStep -0.25
        OmegaRange -180 180
        BoxSize -1000000 1000000 -1000000 1000000
        RingThresh 1 10
        OverAllRingToIndex 1
        Completeness 0.8
        StepSizeOrient 0.2
        StepSizePos 100
    """).strip())
    assert preflight_validate(str(fn), "ff", strict=True) is True


def test_error_with_strict_returns_false(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 999\n")
    assert preflight_validate(str(fn), "ff", strict=True) is False


def test_error_without_strict_returns_true(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 999\n")
    # Errors present but non-strict → true (continue)
    assert preflight_validate(str(fn), "ff", strict=False) is True


def test_logger_is_called(tmp_path):
    """If a logger is provided, .info/.warning/.error are used."""
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 999\n")
    calls = []

    class RecordingLogger:
        def info(self, msg): calls.append(("info", msg))
        def warning(self, msg): calls.append(("warning", msg))
        def error(self, msg): calls.append(("error", msg))

    preflight_validate(str(fn), "ff", strict=True, logger=RecordingLogger())
    # Strict + errors → should log an error
    assert any(level == "error" for level, _ in calls)
