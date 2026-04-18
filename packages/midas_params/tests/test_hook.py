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


# ─── resolve_runtime_defaults ────────────────────────────────────────────────


from midas_params.hook import resolve_runtime_defaults, _min_ringthresh


def test_min_ringthresh_simple(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("RingThresh 1 100\nRingThresh 2 80\nRingThresh 3 120\n")
    assert _min_ringthresh(str(fn)) == 80


def test_min_ringthresh_no_entries(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 225\n")  # no RingThresh
    assert _min_ringthresh(str(fn)) is None


def test_min_ringthresh_malformed_skipped(tmp_path):
    """Malformed RingThresh lines are skipped; remaining ones still used."""
    fn = tmp_path / "p.txt"
    fn.write_text(
        "RingThresh 1 100\n"
        "RingThresh 2\n"          # missing threshold
        "RingThresh 3 bad\n"      # non-numeric
        "RingThresh 4 50\n"
    )
    assert _min_ringthresh(str(fn)) == 50


def test_resolve_runtime_defaults_both_autofill(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("RingThresh 1 100\nRingThresh 2 200\n")
    chunks, thresh = resolve_runtime_defaults(
        param_file=str(fn),
        num_frame_chunks=-1,
        pre_proc_thresh=-1,
        n_cpus=8,
    )
    assert chunks == 32        # 8 × 4
    assert thresh == 100       # min of RingThresh intensities


def test_resolve_runtime_defaults_respects_user_value(tmp_path):
    """Non-`-1` values pass through unchanged."""
    fn = tmp_path / "p.txt"
    fn.write_text("RingThresh 1 100\n")
    chunks, thresh = resolve_runtime_defaults(
        param_file=str(fn),
        num_frame_chunks=16,
        pre_proc_thresh=50,
        n_cpus=8,
    )
    assert chunks == 16        # user's explicit value preserved
    assert thresh == 50


def test_resolve_runtime_defaults_no_ringthresh_leaves_minus_one(tmp_path):
    """No RingThresh entries → preProcThresh stays at -1, with warning."""
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 225\n")

    calls = []

    class RecordingLogger:
        def info(self, msg): calls.append(("info", msg))
        def warning(self, msg): calls.append(("warning", msg))
        def error(self, msg): calls.append(("error", msg))

    chunks, thresh = resolve_runtime_defaults(
        param_file=str(fn),
        num_frame_chunks=-1,
        pre_proc_thresh=-1,
        n_cpus=8,
        logger=RecordingLogger(),
    )
    assert chunks == 32        # still auto-set
    assert thresh == -1        # no RingThresh → stay at -1
    assert any(level == "warning" and "-1" in msg for level, msg in calls)


def test_resolve_runtime_defaults_ncpus_one(tmp_path):
    """nCPUs = 1 still produces valid chunks (4), never 0."""
    fn = tmp_path / "p.txt"
    fn.write_text("RingThresh 1 10\n")
    chunks, _ = resolve_runtime_defaults(
        param_file=str(fn),
        num_frame_chunks=-1,
        pre_proc_thresh=-1,
        n_cpus=1,
    )
    assert chunks == 4


def test_resolve_runtime_defaults_logs_to_logger(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("RingThresh 1 77\n")

    calls = []

    class Rec:
        def info(self, msg): calls.append(("info", msg))
        def warning(self, msg): calls.append(("warning", msg))
        def error(self, msg): calls.append(("error", msg))

    resolve_runtime_defaults(
        param_file=str(fn), num_frame_chunks=-1, pre_proc_thresh=-1,
        n_cpus=2, logger=Rec(),
    )
    msgs = [m for _, m in calls]
    assert any("numFrameChunks auto-set to 8" in m for m in msgs)
    assert any("preProcThresh auto-set to 77" in m for m in msgs)
