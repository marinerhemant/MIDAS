"""Validation engine.

Takes a parsed parameter file plus a target pipeline path (FF/NF/PF/RI)
and runs:
  1. Required-key check (every `required_for` key must be present).
  2. Per-key validators (each spec's `validators` tuple, in order).
  3. Cross-field rules (from crossfield.RULE_SPECS).
  4. Unknown-key check (typo detection via edit-distance match).

Returns a ValidationReport that aggregates all findings with severity. The
caller decides whether to raise, print, or route to an LLM.
"""

from __future__ import annotations

import difflib

from .crossfield import RULE_SPECS, RULES
from .parser import ParsedParams, parse_typed
from .registry import by_name, required_for
from .schema import (
    ParamSpec,
    Path,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from .validators import Ctx, VALIDATORS


def validate(param_file: str, path: Path) -> ValidationReport:
    """Top-level entry point — parse and validate.

    Convenience wrapper that runs parser + validator in one call.
    """
    parsed = parse_typed(param_file)
    return validate_parsed(parsed, path)


def validate_parsed(parsed: ParsedParams, path: Path) -> ValidationReport:
    """Run all validation checks against already-parsed values."""
    report = ValidationReport(param_file=parsed.path, path=path)

    # Carry forward parser issues (type coercion errors, duplicate warnings)
    report.issues.extend(parsed.issues)

    ctx = Ctx(
        all_values=parsed.values,
        param_file=parsed.path,
        line_of=parsed.line_of,
        path=path.value,
    )

    registry = by_name()
    applicable_specs = [p for p in _unique_specs(registry) if path in p.applies_to]

    # ── 1. Required-key check ────────────────────────────────────────────────
    for spec in required_for(path):
        if spec.name not in parsed.values:
            # Don't shadow if the user set an alias — the parser stores under canonical.
            report.issues.append(ValidationIssue(
                severity=Severity.ERROR,
                key=spec.name,
                message=f"Required key {spec.name!r} is missing for {path.value.upper()} pipeline.",
                suggestion=_required_key_suggestion(spec),
                rule="required_key_missing",
                stage=_primary_stage(spec),
            ))

    # ── 2. Per-key validators ────────────────────────────────────────────────
    for spec in applicable_specs:
        value = parsed.values.get(spec.name)
        if value is None:
            continue  # optional + absent
        for validator_name in spec.validators:
            fn = VALIDATORS.get(validator_name)
            if fn is None:
                # Should not happen — resolve() runs at registry load time
                report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    key=spec.name,
                    message=f"Unknown validator {validator_name!r} referenced by {spec.name}.",
                    rule="internal",
                ))
                continue
            try:
                issues = fn(value, spec, ctx)
            except Exception as e:
                # A crashing validator should not take down the whole report.
                issues = [ValidationIssue(
                    severity=Severity.WARNING,
                    key=spec.name,
                    message=f"Validator {validator_name} crashed: {e}",
                    rule="internal",
                )]
            report.issues.extend(issues)

    # ── 3. Cross-field rules ─────────────────────────────────────────────────
    for rule in RULE_SPECS:
        if path not in rule.applies_to:
            continue
        check_fn = RULES.get(rule.check)
        if check_fn is None:
            continue
        try:
            issues = check_fn(ctx)
        except Exception as e:
            issues = [ValidationIssue(
                severity=Severity.WARNING,
                message=f"Cross-field rule {rule.name} crashed: {e}",
                rule="internal",
            )]
        # Use the rule's declared severity as a floor (don't let a rule
        # emit ERROR when declared WARNING, but the rule can emit INFO
        # within a WARNING rule — we trust what the rule returns).
        report.issues.extend(issues)

    # ── 4. Unknown-key detection with typo suggestions ───────────────────────
    canonical_names = {p.name for p in applicable_specs}
    alias_names = {a for p in applicable_specs for a in p.aliases}
    known = canonical_names | alias_names
    for unknown_key, line_no in parsed.unknown_keys:
        close = difflib.get_close_matches(unknown_key, known, n=1, cutoff=0.75)
        suggestion = f"Did you mean {close[0]!r}?" if close else None
        report.issues.append(ValidationIssue(
            severity=Severity.WARNING,
            key=unknown_key,
            line=line_no,
            message=f"Unknown key {unknown_key!r} for {path.value.upper()} pipeline.",
            suggestion=suggestion,
            rule="unknown_key",
        ))

    # Stable sort: errors first, then warnings, then info; within a severity, by line
    severity_rank = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    report.issues.sort(key=lambda i: (severity_rank[i.severity], i.line or 0))
    return report


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _unique_specs(registry: dict[str, ParamSpec]) -> list[ParamSpec]:
    """Registry `by_name` returns aliases; dedupe to canonical specs."""
    seen: set[str] = set()
    out: list[ParamSpec] = []
    for spec in registry.values():
        if spec.name in seen:
            continue
        seen.add(spec.name)
        out.append(spec)
    return out


def _primary_stage(spec: ParamSpec):
    """Pick one stage to blame in an error message. Preference: indexing >
    peak-search > calibration > file-discovery > anything else."""
    from .schema import Stage
    order = [Stage.INDEXING, Stage.PEAK_SEARCH, Stage.REFINEMENT,
             Stage.CALIBRATION, Stage.FILE_DISCOVERY, Stage.IMAGE_PREPROC]
    for s in order:
        if s in spec.stages:
            return s
    return next(iter(spec.stages)) if spec.stages else None


def _required_key_suggestion(spec: ParamSpec) -> str:
    """Actionable suggestion for a missing required key."""
    parts = [f"Add a line like: `{spec.name} <value>`"]
    if spec.typical is not None:
        parts = [f"Add a line like: `{spec.name} {spec.typical}`"]
    elif spec.default is not None:
        parts = [f"Add a line like: `{spec.name} {spec.default}`"]
    if spec.units:
        parts.append(f"Units: {spec.units}")
    if spec.description:
        parts.append(spec.description)
    return " — ".join(parts)


def format_report(report: ValidationReport, use_color: bool = True) -> str:
    """Pretty-print a report to stdout-friendly text."""
    if use_color:
        red = "\033[31m"
        yellow = "\033[33m"
        blue = "\033[34m"
        dim = "\033[2m"
        reset = "\033[0m"
    else:
        red = yellow = blue = dim = reset = ""

    lines = []
    header = f"{report.param_file}  [{report.path.value.upper()}]"
    lines.append(header)
    lines.append("=" * len(header))

    if not report.issues:
        lines.append(f"  {blue}OK{reset} — no issues found.")
        return "\n".join(lines)

    lines.append(f"  {len(report.errors)} error(s), {len(report.warnings)} warning(s), "
                 f"{len(report.issues) - len(report.errors) - len(report.warnings)} info")
    lines.append("")

    for issue in report.issues:
        if issue.severity == Severity.ERROR:
            tag, color = "ERROR", red
        elif issue.severity == Severity.WARNING:
            tag, color = "WARN ", yellow
        else:
            tag, color = "INFO ", blue
        loc = f":{issue.line}" if issue.line else ""
        key_str = f"[{issue.key}] " if issue.key else ""
        lines.append(f"  {color}{tag}{reset}{loc}  {key_str}{issue.message}")
        if issue.suggestion:
            lines.append(f"          {dim}→ {issue.suggestion}{reset}")
        if issue.rule:
            lines.append(f"          {dim}rule: {issue.rule}{reset}")

    return "\n".join(lines)
