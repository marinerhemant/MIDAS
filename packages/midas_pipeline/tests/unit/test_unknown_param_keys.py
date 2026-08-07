"""A misspelled parameter key must be reported, not silently ignored.

``MarginOmega 0.6`` (canonical: ``MarginOme``) was ignored by the classical
chain, by the pipeline, and by the shipped reference run of the datasetA Ni
dataset. All three used MarginOme 0.5 while the parameter file said 0.6, and
nothing anywhere said so.

midas_params already parsed unknown keys and the registry already supported
difflib suggestions. Nothing called either.
"""

import logging

import pytest

from midas_pipeline._logging import LOG
from midas_pipeline.pipeline import _report_unknown_param_keys


class _Capture(logging.Handler):
    """Attach to LOG directly.

    caplog relies on propagation, and configure_logging() -- which other tests
    trigger via Pipeline.run -- turns propagation off, so caplog sees nothing
    when the suite runs as a whole while passing when this file runs alone.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _run_capturing(path):
    h = _Capture()
    LOG.addHandler(h)
    prev = LOG.level
    LOG.setLevel(logging.WARNING)
    try:
        _report_unknown_param_keys(str(path))
    finally:
        LOG.removeHandler(h)
        LOG.setLevel(prev)
    return " ".join(h.messages)


def _write(tmp_path, text):
    p = tmp_path / "ps.txt"
    p.write_text(text)
    return p


def test_reports_the_real_typo_with_a_suggestion(tmp_path):
    p = _write(tmp_path, "MarginOmega 0.6\nLsd 958874.75\nSpaceGroup 225\n")
    msg = _run_capturing(p)
    assert "MarginOmega" in msg
    assert "IGNORED" in msg
    assert "MarginOme" in msg, "must suggest the canonical key"


def test_reports_a_key_with_no_close_match(tmp_path):
    p = _write(tmp_path, "Lsd 958874.75\nNotAKeyAtAll 7\n")
    msg = _run_capturing(p)
    assert "NotAKeyAtAll" in msg
    assert "did you mean" not in msg, "no suggestion when nothing is close"


def test_silent_on_a_clean_file(tmp_path):
    p = _write(tmp_path, "MarginOme 0.5\nLsd 958874.75\nSpaceGroup 225\n")
    assert "unrecognised" not in _run_capturing(p)


def test_never_raises_on_a_missing_or_broken_file(tmp_path):
    """A parameter linter must not be able to fail a production run."""
    _run_capturing(tmp_path / "does_not_exist.txt")
    _run_capturing(_write(tmp_path, "\x00\x01 garbage\n"))
    # no exception is the assertion


def test_reports_the_line_number(tmp_path):
    p = _write(tmp_path, "Lsd 958874.75\nSpaceGroup 225\nMarginOmega 0.6\n")
    msg = _run_capturing(p)
    assert "line 3" in msg
