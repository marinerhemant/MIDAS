"""LLM-ready diagnosis export.

The validator emits structured ValidationIssues. This module packages those
plus enough registry context that an LLM (or any downstream tool) can
reason about them without needing to understand MIDAS internals.

The output is JSON-serializable and includes:
  - the file under review (with line-numbered content)
  - each issue (severity, location, rule name, suggestion)
  - for each issue, the relevant ParamSpec entry from the registry
  - for each cross-field rule, the rule's description
  - a short "pipeline primer" paragraph per path

An LLM given this payload can say: "Your OmegaStep sign is wrong because
the rotation direction at APS 1-ID goes from +180 to -180 by convention,
and your step is positive; here's the one-line fix."
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path as FsPath
from typing import Any

from .crossfield import RULE_SPECS
from .registry import by_name, for_path, required_for
from .schema import Path, ParamSpec, ValidationIssue, ValidationReport


_PIPELINE_PRIMERS = {
    Path.FF: (
        "Far-field HEDM (FF-HEDM): detector ~1 m from sample. Data goes "
        "through ffGenerateZipRefactor.py → Zarr analysis file → "
        "IndexerOMP → FitPosOrStrainsOMP. Key concepts: ring thresholds "
        "filter spots, Completeness is the grain-acceptance fraction, "
        "OverAllRingToIndex generates initial candidate orientations."
    ),
    Path.NF: (
        "Near-field HEDM (NF-HEDM): detector ~1 cm from sample, multiple "
        "distances per scan. Each distance has its own Lsd/BC/OmegaRange/"
        "BoxSize line. ProcessImagesCombined → MakeDiffrSpots (with "
        "SeedOrientations) → FitOrientationOMP. Grain voxels live on a "
        "hexagonal grid defined by GridSize + Rsample."
    ),
    Path.PF: (
        "Point-focus HEDM (PF-HEDM): FF geometry with translated scans. "
        "nScans > 1 with ScanStep between positions; positions.csv is "
        "generated from BeamSize + ScanStep. Otherwise identical to FF."
    ),
    Path.RI: (
        "Radial integration: 2D detector image → 1D lineout along 2θ. "
        "Uses CalibrantIntegratorOMP / IntegratorZarrOMP. Key keys: "
        "RMin/RMax/RBinSize control the radial binning, EtaMin/EtaMax/"
        "EtaBinSize control the azimuthal binning."
    ),
}


def _spec_to_summary(spec: ParamSpec) -> dict[str, Any]:
    """Minimal registry summary for one ParamSpec — what an LLM actually needs."""
    return {
        "name": spec.name,
        "type": spec.type.value,
        "units": spec.units,
        "description": spec.description,
        "default": spec.default,
        "typical": spec.typical,
        "multi_entry": spec.multi_entry,
        "aliases": list(spec.aliases),
        "zarr_rename": spec.zarr_rename,
        "required": True if spec.required_for else False,
        "notes": spec.notes,
    }


def build_diagnosis_payload(
    report: ValidationReport,
    include_source: bool = True,
    include_registry_context: bool = True,
    include_primer: bool = True,
) -> dict[str, Any]:
    """Build an LLM-consumable diagnosis payload.

    Args:
        report: the validator's output.
        include_source: embed the file content (with line numbers).
        include_registry_context: attach the ParamSpec for each issue's key.
        include_primer: attach a pipeline primer paragraph.
    """
    registry = by_name()
    payload: dict[str, Any] = {
        "path": report.path.value,
        "param_file": report.param_file,
        "status": "ok" if report.ok else "errors",
        "counts": {
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "info": len(report.issues) - len(report.errors) - len(report.warnings),
        },
    }

    if include_primer:
        payload["pipeline_primer"] = _PIPELINE_PRIMERS.get(report.path, "")

    # Issues with optional registry + cross-field rule context
    rule_desc = {r.name: r.description for r in RULE_SPECS}
    issues_out = []
    referenced_keys: set[str] = set()
    for issue in report.issues:
        item = {
            "severity": issue.severity.value,
            "key": issue.key,
            "line": issue.line,
            "message": issue.message,
            "suggestion": issue.suggestion,
            "rule": issue.rule,
            "stage": issue.stage.value if issue.stage else None,
        }
        if include_registry_context and issue.key:
            spec = registry.get(issue.key)
            if spec is not None:
                item["spec"] = _spec_to_summary(spec)
                referenced_keys.add(spec.name)
        if issue.rule and issue.rule in rule_desc:
            item["rule_description"] = rule_desc[issue.rule]
        issues_out.append(item)
    payload["issues"] = issues_out

    # Source with line numbers
    if include_source:
        try:
            lines = FsPath(report.param_file).read_text().splitlines()
            payload["source"] = [
                {"line": i + 1, "text": line}
                for i, line in enumerate(lines)
            ]
        except OSError:
            payload["source"] = None

    # Summary of all required keys (whether present or not) for the path
    if include_registry_context:
        payload["required_for_path"] = [
            _spec_to_summary(spec) for spec in required_for(report.path)
        ]

    return payload


def format_diagnosis_prompt(payload: dict[str, Any]) -> str:
    """Turn a diagnosis payload into a text prompt suitable for an LLM.

    This is what you'd send to Claude / GPT. The output is plain text with
    clear section headers — no chat/roles, no system prompts, just the
    facts. Wrap in your own Messages API call.
    """
    lines = [
        f"# MIDAS parameter diagnosis",
        f"",
        f"Path: {payload['path'].upper()}",
        f"Status: {payload['status']}",
        f"Counts: {payload['counts']}",
        f"",
    ]
    if payload.get("pipeline_primer"):
        lines += [f"## Pipeline context", payload["pipeline_primer"], ""]

    lines += ["## Issues"]
    for i, issue in enumerate(payload["issues"], 1):
        lines.append(f"### Issue {i} ({issue['severity']})")
        if issue.get("key"):
            lines.append(f"  Key: {issue['key']}")
        if issue.get("line"):
            lines.append(f"  Line: {issue['line']}")
        lines.append(f"  Message: {issue['message']}")
        if issue.get("suggestion"):
            lines.append(f"  Suggestion: {issue['suggestion']}")
        if issue.get("rule_description"):
            lines.append(f"  Rule: {issue.get('rule')} — {issue['rule_description']}")
        if issue.get("spec"):
            s = issue["spec"]
            lines.append(
                f"  Spec: {s['description']}"
                + (f"  [units: {s['units']}]" if s.get("units") else "")
                + (f"  [default: {s['default']}]" if s.get("default") is not None else "")
                + (f"  [typical: {s['typical']}]" if s.get("typical") is not None else "")
            )
        lines.append("")

    if payload.get("source"):
        lines.append("## File content")
        lines.append("```")
        for row in payload["source"]:
            lines.append(f"{row['line']:>4}  {row['text']}")
        lines.append("```")

    lines += [
        "",
        "## Your job",
        "Explain each error in plain language. For each, propose the "
        "minimal edit to the file (show the exact line to add / change / "
        "remove). Group related errors. Do not hallucinate MIDAS behavior "
        "that isn't in the Pipeline context above or the issue messages.",
    ]
    return "\n".join(lines)
