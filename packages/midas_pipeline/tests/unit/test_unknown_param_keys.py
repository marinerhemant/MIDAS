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
    # `MarginOmeg` is a genuine typo with no spelling registered anywhere.
    # This used to be spelled `MarginOmega`, which is now an accepted ALIAS
    # (see test_margin_omega_is_an_accepted_alias) — so the linter must be
    # exercised with a key that is still unknown, or it tests nothing.
    p = _write(tmp_path, "MarginOmeg 0.6\nLsd 958874.75\nSpaceGroup 225\n")
    msg = _run_capturing(p)
    assert "MarginOmeg" in msg
    assert "IGNORED" in msg
    assert "MarginOme" in msg, "must suggest the canonical key"


def test_margin_omega_is_an_accepted_alias(tmp_path):
    """``MarginOmega`` resolves to ``MarginOme`` and must not warn.

    Before the alias (2026-08-21) this key was silently dropped and the run
    used the ``MarginOme`` default — the datasetA Ni recipe said
    ``MarginOmega 0.6`` and its generated paramstest.txt read
    ``MarginOme 0.500000``.
    """
    p = _write(tmp_path, "MarginOmega 0.6\nLsd 958874.75\nSpaceGroup 225\n")
    assert "unrecognised" not in _run_capturing(p)

    from midas_params.parser import parse_typed
    parsed = parse_typed(str(p))
    assert "MarginOmega" not in [k for k, _ in (parsed.unknown_keys or ())]
    assert float(parsed.values["MarginOme"]) == pytest.approx(0.6), (
        "the alias must carry its VALUE onto the canonical key, not merely "
        "be tolerated — silently keeping the default is the original bug"
    )


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
    p = _write(tmp_path, "Lsd 958874.75\nSpaceGroup 225\nMarginOmeg 0.6\n")
    msg = _run_capturing(p)
    assert "line 3" in msg


def test_peakfit_background_keys_are_not_reported_as_ignored(tmp_path):
    """MinPeakSNR / BgSubtract / BgNSectors ARE honoured; saying otherwise is
    a false positive with real consequences.

    They reach the zarr through midas_zipper's allow-list
    (``analysis/process/analysis_parameters/MinPeakSNR``), are mapped by
    ``midas_peakfit.zarr_io`` and consumed in ``_producer_worker``. The
    linter called them unrecognised, which would tell a careful user their
    peak search ran unfiltered when it did not.
    """
    p = _write(tmp_path, "MinPeakSNR 5.0\nBgSubtract 1\nBgNSectors 36\n"
                         "Lsd 958874.75\nSpaceGroup 225\n")
    msg = _run_capturing(p)
    assert "unrecognised" not in msg
    assert "MinPeakSNR" not in msg
