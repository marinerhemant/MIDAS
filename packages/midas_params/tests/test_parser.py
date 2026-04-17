"""Parser tests against real MIDAS Example files."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path as FsPath

import pytest

from midas_params.parser import parse_raw, parse_typed, _tokenize_line


MIDAS_ROOT = FsPath(__file__).resolve().parents[3]
FF_EXAMPLE = MIDAS_ROOT / "FF_HEDM" / "Example" / "Parameters.txt"
NF_EXAMPLE = MIDAS_ROOT / "NF_HEDM" / "Example" / "ps_au.txt"


# ─── Tokenizer unit tests ────────────────────────────────────────────────────


def test_tokenize_simple():
    assert _tokenize_line("Lsd 1000000") == ["Lsd", "1000000"]


def test_tokenize_strips_inline_comment():
    assert _tokenize_line("OmegaStep -0.25  # rotation step") == ["OmegaStep", "-0.25"]


def test_tokenize_blank_and_comment_lines_empty():
    assert _tokenize_line("") == []
    assert _tokenize_line("   ") == []
    assert _tokenize_line("# pure comment") == []
    assert _tokenize_line("   # leading whitespace then comment") == []


def test_tokenize_multi_value():
    assert _tokenize_line("BC 985.415 17.51") == ["BC", "985.415", "17.51"]


# ─── parse_raw ───────────────────────────────────────────────────────────────


def test_parse_raw_multi_entry(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        # multi-entry test
        RingThresh 1 100
        RingThresh 2 150
        RingThresh 3 200
    """).strip())
    raw, line_of = parse_raw(fn)
    assert raw["RingThresh"] == [["1", "100"], ["2", "150"], ["3", "200"]]
    assert line_of["RingThresh"] == 2  # first occurrence


def test_parse_raw_line_numbers(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("# comment\n\nLsd 1000000\n# comment\nBC 1022 1022\n")
    raw, line_of = parse_raw(fn)
    assert line_of["Lsd"] == 3
    assert line_of["BC"] == 5


# ─── parse_typed ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not FF_EXAMPLE.exists(), reason="FF Example not present")
def test_ff_example_parses_cleanly():
    parsed = parse_typed(FF_EXAMPLE)
    # All type coercions must succeed
    assert len(parsed.issues) == 0, parsed.issues
    # Spot-check canonical values
    assert parsed.values["SpaceGroup"] == 225
    assert parsed.values["Wavelength"] == 0.22291
    assert parsed.values["OmegaStart"] == 180
    assert parsed.values["OmegaStep"] == -0.25
    # LatticeConstant is a FLOAT_LIST
    assert parsed.values["LatticeConstant"] == [4.08, 4.08, 4.08, 90.0, 90.0, 90.0]
    # RingThresh is multi-entry
    assert parsed.values["RingThresh"] == [[1, 10], [2, 10], [3, 10], [4, 10], [5, 10]]


@pytest.mark.skipif(not NF_EXAMPLE.exists(), reason="NF Example not present")
def test_nf_example_parses_cleanly():
    parsed = parse_typed(NF_EXAMPLE)
    assert len(parsed.issues) == 0, parsed.issues
    # Multi-entry Lsd and BC are central to NF
    assert parsed.values["Lsd"] == [8289.154576, 10290.724494]
    assert parsed.values["BC"] == [[985.415831, 17.510494], [985.161497, 24.51121]]
    assert parsed.values["nDistances"] == 2


def test_aliases_resolve_to_canonical(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("LatticeParameter 4.08 4.08 4.08 90 90 90\nDistance 1000000\n")
    parsed = parse_typed(fn)
    # Stored under canonical names, not aliases
    assert "LatticeConstant" in parsed.values
    assert "Lsd" in parsed.values
    assert "LatticeParameter" not in parsed.values
    assert "Distance" not in parsed.values


def test_bad_type_produces_issue(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup not_an_int\n")
    parsed = parse_typed(fn)
    assert any(i.key == "SpaceGroup" and "expects an int" in i.message
               for i in parsed.issues)


def test_unknown_key_goes_to_unknown_list(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("CompletelyMadeUp value\n")
    parsed = parse_typed(fn)
    assert parsed.unknown_keys == [("CompletelyMadeUp", 1)]
    assert "CompletelyMadeUp" not in parsed.values
