"""``run_refiner(line_cb=...)`` must stream, not buffer.

``FitUnified.c`` prints ``progress: N/M seeds`` ~200 times per run and
``fflush``es each because stdout is a pipe; the default ``capture_output``
path holds it all until exit.

``_run_streaming`` here is a deliberate copy of the one in
``midas_index.backend_c`` -- the two packages are independent and neither
should depend on the other for a twenty-line subprocess helper. These tests
pin the same contract on this copy so the two cannot drift apart silently.
"""

from __future__ import annotations

import sys
import time

from midas_fit_grain.backend_c import _run_streaming


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_lines_arrive_in_order():
    seen: list[str] = []
    proc = _run_streaming(_py("print('a'); print('b')"), cwd=".", env={},
                          line_cb=seen.append)
    assert [s.strip() for s in seen] == ["a", "b"]
    assert proc.returncode == 0


def test_lines_arrive_while_the_child_is_still_running():
    # The test that separates streaming from buffering; everything else here
    # passes either way.
    stamps: list[float] = []
    _run_streaming(
        _py("import time\n"
            "print('first', flush=True)\n"
            "time.sleep(1.0)\n"
            "print('second', flush=True)\n"),
        cwd=".", env={}, line_cb=lambda _l: stamps.append(time.monotonic()),
    )
    assert len(stamps) == 2
    assert stamps[1] - stamps[0] > 0.5, "output is being buffered to exit"


def test_stdout_and_stderr_survive_for_the_caller():
    # stages/refinement.py writes both to refinement_out.csv / _err.csv and
    # builds its error message from stderr.
    proc = _run_streaming(
        _py("import sys; print('out'); sys.stderr.write('err\\n'); sys.exit(2)"),
        cwd=".", env={}, line_cb=lambda _l: None,
    )
    assert proc.returncode == 2
    assert b"out" in proc.stdout
    assert b"err" in proc.stderr


def test_a_raising_callback_does_not_fail_the_run():
    def cb(_line: str) -> None:
        raise RuntimeError("reporter exploded")

    proc = _run_streaming(_py("print('fine')"), cwd=".", env={}, line_cb=cb)
    assert proc.returncode == 0
