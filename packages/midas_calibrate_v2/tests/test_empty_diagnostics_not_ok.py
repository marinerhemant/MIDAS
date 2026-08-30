"""An absent diagnosis must not read as a clean one.

Two silent-success paths, both the ``basin_check`` shape:

1. ``worst_severity([])`` returned ``"ok"``. It is built on
   ``max(..., default=0)`` and 0 maps to "ok", so a run whose gates failed to
   compute reported exactly what a run that passed every gate reports.

2. ``auto.py`` builds the uncertainty block and the gate list inside one
   ``try/except`` whose handler printed only under ``verbose``. With the normal
   ``verbose=False`` an exception there returned a result with empty ``sigma``,
   empty ``unconstrained``, empty ``at_bounds`` and empty ``diagnostics`` — and
   nothing on stdout. Combined with (1), that is a green tick for a run whose
   diagnosis never happened.

Separately, ``sigma`` is recorded for SCALAR parameters only. In the panel
stage every global is frozen and only vector parameters (``panel_delta_*``)
refine, so ``sigma`` comes back empty and ``unconstrained`` is then trivially
empty — measured on the real 48-panel Pilatus: 6 gates ran and were genuinely
ok, but ``sigma`` had 0 entries. That is "nothing was measured", not "nothing
is wrong", and it now says so.
"""
from __future__ import annotations

import warnings

import pytest

from midas_calibrate_v2.pipelines.diagnostics import (
    DiagnosticResult, worst_severity)


def _d(sev):
    return DiagnosticResult(name="g", severity=sev, message="", metrics={})


def test_no_gates_is_unknown_not_ok():
    assert worst_severity([]) == "unknown", (
        "an empty gate list must not report the same thing as a run that "
        "passed every gate")


def test_a_real_gate_list_still_ranks_normally():
    assert worst_severity([_d("ok")]) == "ok"
    assert worst_severity([_d("ok"), _d("warn")]) == "warn"
    assert worst_severity([_d("ok"), _d("warn"), _d("fail")]) == "fail"
    assert worst_severity([_d("fail"), _d("ok")]) == "fail"


def test_unknown_is_distinguishable_from_every_real_severity():
    real = {worst_severity([_d(s)]) for s in ("ok", "warn", "fail")}
    assert worst_severity([]) not in real


def test_callers_testing_for_fail_are_unaffected():
    """`first_time.py` accepts an attempt when no critical gate is "fail".
    An empty list was accepted before and is still accepted — the string
    changed, the decision did not."""
    assert worst_severity([]) != "fail"


# ------------------------------------------------- the silent-exception path

def test_the_uncertainty_block_warns_rather_than_vanishing():
    """The handler must warn unconditionally, not only under ``verbose``.

    Checked at the source rather than by forcing an exception through a full
    calibration: the point is that the handler no longer depends on ``verbose``
    to say anything at all.
    """
    import inspect
    from midas_calibrate_v2.pipelines import auto

    src = inspect.getsource(auto)
    i = src.index("diagnostics must never fail a run")
    handler = src[i:i + 1400]
    assert "warnings.warn(" in handler, (
        "the uncertainty/diagnostics handler still reports only under verbose")
    # and the warning must say what the emptiness means
    assert "clean bill of health" in handler


def test_vector_only_refinement_is_now_reported_not_just_warned_about():
    """Superseded 2026-08-29 by the real fix.

    This used to assert a warning saying "sigma is empty because every refined
    parameter is vector-valued". That warning existed because vector σ was not
    recorded at all. It now IS recorded (`sigma_vector`), so the warning is
    gone by design — what must remain is that an EMPTY result still announces
    itself rather than passing for a clean one.
    """
    import inspect
    from midas_calibrate_v2.pipelines import auto

    src = inspect.getsource(auto)
    assert "sigma_vector[nm]" in src, "vector sigma is no longer recorded"
    # the surviving warning covers the genuinely-empty case
    assert "no per-parameter sigma" in src
    assert "not because everything is\n                    \"fine" in src \
        or "not because everything is" in src
