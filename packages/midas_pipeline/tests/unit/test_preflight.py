"""Preflight input validation must fail loudly, before any stage runs.

Regression cover for the failure two users hit: a mistyped --params made
zip_convert exit 1, every downstream FF stage skip with "no zarr/zip available
at None", and the run still print "done. layers processed: 1" and exit 0.
"""

from __future__ import annotations

import textwrap

import pytest

from midas_pipeline.preflight import PreflightError, check_inputs, preflight


class _Cfg:
    """Minimal stand-in for PipelineConfig (preflight only reads these)."""

    def __init__(self, params_file, result_dir, *, zarr_path=None,
                 convert_files=True):
        self.params_file = str(params_file)
        self.result_dir = str(result_dir)
        self.zarr_path = zarr_path
        self.convert_files = convert_files


def _params(raw_dir, stem="ff_scan", start=27, pad=6, ext=".ge5.h5", dark=None):
    body = textwrap.dedent(f"""
        RawFolder {raw_dir}
        FileStem {stem}
        Padding {pad}
        Ext {ext}
        StartFileNrFirstLayer {start}
        NrFilesPerSweep 1
    """).strip()
    if dark is not None:
        body += f"\nDark {dark}"
    return body + "\n"


@pytest.fixture
def good(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ff_scan_000027.ge5.h5").write_bytes(b"x")
    dark = raw / "dark_000026.ge5.h5"
    dark.write_bytes(b"x")
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(raw, dark=dark))
    return _Cfg(p, tmp_path / "results")


def test_good_inputs_pass(good):
    assert check_inputs(good, [1]) == []
    preflight(good, [1])                      # must not raise


def test_missing_params_file_is_fatal_and_names_the_path(tmp_path):
    cfg = _Cfg(tmp_path / "Paramers_typo.txt", tmp_path / "results")
    with pytest.raises(PreflightError) as ei:
        preflight(cfg, [1])
    msg = str(ei.value)
    assert "parameter file not found" in msg
    assert "Paramers_typo.txt" in msg
    assert "--params" in msg                  # points at the likely cause


def test_empty_params_file_is_fatal(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text("")
    with pytest.raises(PreflightError, match="empty"):
        preflight(_Cfg(p, tmp_path / "r"), [1])


def test_missing_raw_file_is_fatal_and_lists_nearby_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ff_scan_000031.ge5.h5").write_bytes(b"x")   # wrong scan number
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(raw))
    problems = check_inputs(_Cfg(p, tmp_path / "r"), [1])
    assert any("raw data file for layer 1 not found" in x for x in problems)
    assert any("ff_scan_000031.ge5.h5" in x for x in problems)  # the hint


def test_missing_raw_folder_is_fatal(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(tmp_path / "nope"))
    problems = check_inputs(_Cfg(p, tmp_path / "r"), [1])
    assert any("RawFolder is not a directory" in x for x in problems)


def test_missing_dark_is_fatal(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ff_scan_000027.ge5.h5").write_bytes(b"x")
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(raw, dark=tmp_path / "no_such_dark.h5"))
    problems = check_inputs(_Cfg(p, tmp_path / "r"), [1])
    assert any("Dark file not found" in x for x in problems)


def test_crlf_line_endings_are_reported(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ff_scan_000027.ge5.h5").write_bytes(b"x")
    p = tmp_path / "Parameters.txt"
    p.write_bytes(_params(raw).replace("\n", "\r\n").encode())
    problems = check_inputs(_Cfg(p, tmp_path / "r"), [1])
    assert any("CRLF" in x for x in problems)


def test_all_problems_reported_together(tmp_path):
    """One run should surface every problem, not just the first."""
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(tmp_path / "nope", dark=tmp_path / "no_dark.h5"))
    problems = check_inputs(_Cfg(p, tmp_path / "r"), [1])
    assert len(problems) >= 2
    with pytest.raises(PreflightError) as ei:
        preflight(_Cfg(p, tmp_path / "r"), [1])
    assert "2 problem(s)" in str(ei.value) or "problem(s)" in str(ei.value)


def test_prebuilt_zarr_skips_raw_checks(tmp_path):
    z = tmp_path / "scan.zarr"
    z.mkdir()
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(tmp_path / "nope"))       # raw missing, but unused
    assert check_inputs(_Cfg(p, tmp_path / "r", zarr_path=str(z)), [1]) == []


def test_missing_zarr_is_reported(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(tmp_path / "nope"))
    cfg = _Cfg(p, tmp_path / "r", zarr_path=str(tmp_path / "absent.zarr"))
    assert any("--zarr given but not found" in x for x in check_inputs(cfg, [1]))


def test_existing_zip_skips_raw_checks(tmp_path):
    ld = tmp_path / "results" / "LayerNr_1"
    ld.mkdir(parents=True)
    (ld / "scan.MIDAS.zip").write_bytes(b"x")
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(tmp_path / "nope"))
    assert check_inputs(_Cfg(p, tmp_path / "results"), [1]) == []


def test_convert_files_off_skips_raw_checks(tmp_path):
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(tmp_path / "nope"))
    cfg = _Cfg(p, tmp_path / "r", convert_files=False)
    assert check_inputs(cfg, [1]) == []


def test_multi_layer_resolves_each_file(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ff_scan_000027.ge5.h5").write_bytes(b"x")   # layer 1 only
    p = tmp_path / "Parameters.txt"
    p.write_text(_params(raw))
    problems = check_inputs(_Cfg(p, tmp_path / "r"), [1, 2])
    assert any("layer 2" in x and "000028" in x for x in problems)
