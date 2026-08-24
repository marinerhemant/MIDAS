"""``run_indexer(line_cb=...)`` must stream, not buffer.

``IndexerUnified.c`` prints ``progress: N/M voxels`` ~200 times per run and
``fflush``es each one specifically because stdout is a pipe. The default
``capture_output=True`` path holds all of it in this process until the binary
exits, so a PF indexing stage that runs eight hours reports nothing at all
until it is over -- which is exactly as useful as no progress reporting.

These exercise ``_run_streaming`` directly with a stand-in child process, so
they run without the compiled binary present.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from midas_index.backend_c import _run_streaming


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_lines_arrive_in_order():
    seen: list[str] = []
    proc = _run_streaming(
        _py("print('a'); print('b'); print('c')"),
        cwd=".", env={}, line_cb=seen.append,
    )
    assert [s.strip() for s in seen] == ["a", "b", "c"]
    assert proc.returncode == 0


def test_lines_arrive_while_the_child_is_still_running():
    """The whole point: a line must be delivered BEFORE the process exits.

    A buffered implementation passes every other test in this file, so this is
    the one that actually distinguishes streaming from capturing. The child
    flushes, sleeps, then flushes again; if we were buffering, both callbacks
    would fire together after exit and the gap would be ~0.
    """
    stamps: list[float] = []

    def cb(_line: str) -> None:
        stamps.append(time.monotonic())

    _run_streaming(
        _py("import sys,time\n"
            "print('first', flush=True)\n"
            "time.sleep(1.0)\n"
            "print('second', flush=True)\n"),
        cwd=".", env={}, line_cb=cb,
    )
    assert len(stamps) == 2
    assert stamps[1] - stamps[0] > 0.5, (
        "both lines arrived at once -- output is being buffered to exit"
    )


def test_stdout_still_returned_like_the_buffered_path():
    # stages/refinement.py writes proc.stdout to refinement_out.csv, so the
    # streaming path must not swallow it.
    code = "print('x'); print('y')"
    streamed = _run_streaming(_py(code), cwd=".", env={}, line_cb=lambda s: None)
    buffered = subprocess.run(_py(code), capture_output=True, check=False)
    assert streamed.stdout == buffered.stdout
    assert streamed.returncode == buffered.returncode == 0


def test_stderr_is_captured_and_returncode_preserved():
    # Callers build their RuntimeError message out of proc.stderr, and a
    # non-zero exit must still be visible.
    proc = _run_streaming(
        _py("import sys; sys.stderr.write('boom\\n'); sys.exit(3)"),
        cwd=".", env={}, line_cb=lambda s: None,
    )
    assert proc.returncode == 3
    assert b"boom" in proc.stderr


def test_a_raising_callback_does_not_fail_the_run():
    # Progress reporting is never worth losing an eight-hour indexing run over.
    def cb(_line: str) -> None:
        raise ValueError("reporter exploded")

    proc = _run_streaming(_py("print('still fine')"), cwd=".", env={}, line_cb=cb)
    assert proc.returncode == 0
    assert b"still fine" in proc.stdout


def test_large_stderr_does_not_deadlock():
    """stderr is spooled to a file, not a second pipe.

    Draining stdout while stderr fills its 64 KB pipe buffer is the classic
    deadlock; write well past that.
    """
    proc = _run_streaming(
        _py("import sys\n"
            "sys.stderr.write('e' * 500000)\n"
            "print('done')\n"),
        cwd=".", env={}, line_cb=lambda s: None,
    )
    assert proc.returncode == 0
    assert len(proc.stderr) >= 500000


def test_progress_lines_parse_into_the_pipeline_reporter():
    """End to end with the format the C actually emits."""
    parse = pytest.importorskip("midas_pipeline.progress").parse_progress_line
    seen = []
    _run_streaming(
        _py(r"print('  progress: 120/36100 voxels, 3.4 vox/s, elapsed 35.0s')"),
        cwd=".", env={}, line_cb=lambda s: seen.append(parse(s)),
    )
    assert seen and seen[0] is not None
    assert seen[0]["done"] == 120
    assert seen[0]["total"] == 36100
    assert seen[0]["unit"] == "voxels"
