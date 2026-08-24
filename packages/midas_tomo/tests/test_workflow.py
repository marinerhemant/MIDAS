"""The one-command workflow: scan record in, reconstruction out.

The engine is not exercised here (it needs a built binary and real
projections); these cover the wiring, the provenance and — most of all — the
refusals, which is where a workflow silently produces a plausible wrong
answer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_tomo.cli_reconstruct import _build_parser, main
from midas_tomo.workflow import ReconstructionResult, _even
from midas_tomo.scanrecord import read_scan_record

# The scan-record fixture lives in its sibling test module; import it by path
# so the tests do not depend on the tests directory being importable.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_scanrec_fixtures", Path(__file__).with_name("test_scanrecord.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BT_1ID_JUN25B, _write = _mod.BT_1ID_JUN25B, _mod._write


def test_the_shift_sweep_is_forced_to_an_even_count():
    """The engine reconstructs shifts in pairs and rejects an odd count. The
    workflow extends the range rather than silently re-deriving the step."""
    assert _even(-25.0, 25.0, 1.0) == (-25.0, 26.0, 1.0)      # 51 -> 52
    assert _even(-25.0, 26.0, 1.0) == (-25.0, 26.0, 1.0)      # already even
    lo, hi, st = _even(11.0, 15.0, 0.1)                        # 41 -> 42
    assert round(abs(hi - lo) / st) + 1 == 42


def test_the_cli_refuses_a_missing_scan_record(tmp_path, capsys):
    rc = main([str(tmp_path / "nope.dat"), "--root", str(tmp_path),
               "--out", str(tmp_path / "out")])
    assert rc == 2 and "no such scan record" in capsys.readouterr().err


def test_the_cli_refuses_a_root_that_is_not_a_directory(tmp_path, capsys):
    rec = _write(tmp_path, BT_1ID_JUN25B)
    rc = main([str(rec), "--root", str(rec), "--out", str(tmp_path / "out")])
    assert rc == 2 and "--root is not a directory" in capsys.readouterr().err


def test_the_help_names_the_pixel_size_trap():
    """The single most costly mistake available here, so it belongs in --help
    and not only in a docstring."""
    epilog = _build_parser().epilog
    assert "tomocupy_args.yml" in epilog
    assert "different camera" in epilog


def test_phase_retrieval_is_off_by_default():
    a = _build_parser().parse_args(["r.dat", "--root", ".", "--out", "o"])
    assert a.delta_beta == 0.0
    assert not a.no_strict            # strict centring is the default


def test_the_crop_is_not_inferred():
    a = _build_parser().parse_args(["r.dat", "--root", ".", "--out", "o"])
    assert a.crop is None
    help_text = _build_parser().format_help()
    assert "NOT inferred" in help_text


def test_the_sample_shape_hint_marks_what_is_still_unresolved(tmp_path):
    """The hint must not look like a finished call: handedness, threshold and
    the stage vertical are all still the user's to supply."""
    scan = read_scan_record(_write(tmp_path, BT_1ID_JUN25B))
    res = ReconstructionResult(
        scan=scan, ingest=None, shift=13.0, shift_trustworthy=True,
        shift_reason="", recon_path=Path("/tmp/r.h5"),
        recon_shape=(1, 128, 128, 128),
    )
    hint = res.sample_shape_hint()
    assert "0.708" in hint                       # the measured pixel size
    assert "UNRESOLVED" in hint                  # handedness
    assert "sweep it" in hint                    # threshold
    assert "stage vertical" in hint              # slice0_z_um


def test_the_summary_reports_an_uncertified_shift_with_its_reason(tmp_path):
    scan = read_scan_record(_write(tmp_path, BT_1ID_JUN25B))
    res = ReconstructionResult(
        scan=scan, ingest=None, shift=13.0, shift_trustworthy=False,
        shift_reason="criteria disagree by 1.6 px", recon_path=None,
        recon_shape=(),
    )
    s = res.summary()
    assert "trustworthy=False" in s and "criteria disagree" in s
    assert "aero sign applied" in s
