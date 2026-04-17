"""Tests for the LLM diagnosis payload."""

from __future__ import annotations

import json
import textwrap

import pytest

from midas_params import Path
from midas_params.diagnose import build_diagnosis_payload, format_diagnosis_prompt
from midas_params.validator import validate


def test_payload_shape(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 999\n")  # guaranteed error
    report = validate(str(fn), Path.FF)
    payload = build_diagnosis_payload(report)
    assert payload["path"] == "ff"
    assert payload["status"] == "errors"
    assert payload["counts"]["errors"] >= 1
    assert "pipeline_primer" in payload
    assert "issues" in payload
    # Every issue should have severity, message, rule
    for issue in payload["issues"]:
        assert "severity" in issue
        assert "message" in issue


def test_payload_is_json_serializable(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 999\nWavelength 0.22\n")
    report = validate(str(fn), Path.FF)
    payload = build_diagnosis_payload(report)
    # This must not raise
    serialized = json.dumps(payload, default=str)
    assert len(serialized) > 100


def test_payload_includes_source_by_default(tmp_path):
    fn = tmp_path / "p.txt"
    content = "# line 1\nSpaceGroup 225\n# line 3\n"
    fn.write_text(content)
    report = validate(str(fn), Path.FF)
    payload = build_diagnosis_payload(report)
    assert payload["source"] is not None
    assert payload["source"][0] == {"line": 1, "text": "# line 1"}


def test_payload_excludes_source_when_asked(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 225\n")
    report = validate(str(fn), Path.FF)
    payload = build_diagnosis_payload(report, include_source=False)
    assert "source" not in payload


def test_payload_includes_spec_context(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 999\n")
    report = validate(str(fn), Path.FF)
    payload = build_diagnosis_payload(report)
    # The SpaceGroup issue should have its ParamSpec attached
    sg_issues = [i for i in payload["issues"] if i.get("key") == "SpaceGroup"]
    assert sg_issues
    assert "spec" in sg_issues[0]
    assert sg_issues[0]["spec"]["description"] == "Space group number."


def test_payload_lists_required_keys(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("# empty\n")
    report = validate(str(fn), Path.NF)
    payload = build_diagnosis_payload(report)
    required_names = {p["name"] for p in payload["required_for_path"]}
    assert "DataDirectory" in required_names
    assert "nDistances" in required_names


def test_format_diagnosis_prompt_produces_text(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        OmegaStart 0
        OmegaEnd 180
        OmegaStep -0.25
    """).strip())
    report = validate(str(fn), Path.FF)
    payload = build_diagnosis_payload(report)
    prompt = format_diagnosis_prompt(payload)
    # Should contain section markers and issue content
    assert "# MIDAS parameter diagnosis" in prompt
    assert "## Issues" in prompt
    assert "OmegaStep" in prompt
    assert "Your job" in prompt  # instruction block
