"""Parse a MIDAS text parameter file.

Format (matching MIDAS_ParamParser.c + ffGenerateZipRefactor.py parse_parameter_file):
  - Each line: `<Key> <value1> [value2 ...]`
  - Inline comments start with `#` and continue to end of line
  - Blank lines and comment-only lines are ignored
  - Some keys may appear multiple times (multi-entry) — values accumulate
  - Leading/trailing whitespace tolerated
  - Keys are case-sensitive

Two-stage parsing:
  1. `parse_raw` — file → dict[str, list[raw_token_list]] + line numbers.
     No registry knowledge; purely lexical.
  2. `parse_typed` — raw → typed dict using the registry to coerce strings to
     int/float/bool/list. Unknown keys pass through as strings; type errors
     become ValidationIssues.

The separation lets the parser be tested standalone and lets the validator
see both "the user wrote 'foo'" (raw) and "we parsed it as 3.14" (typed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path as FsPath
from typing import Any

from .registry import by_name
from .schema import ParamSpec, ParamType, Severity, ValidationIssue


@dataclass
class ParsedParams:
    """Output of parsing a parameter file."""

    path: str                                        # source file path
    values: dict[str, Any] = field(default_factory=dict)
    # For multi-entry keys, values[key] is a list of per-entry values.
    # For single-entry keys, values[key] is the typed value (scalar or list of scalars).

    line_of: dict[str, int] = field(default_factory=dict)
    # First line (1-indexed) where each key appeared. Used for error messages.

    raw_tokens: dict[str, list[list[str]]] = field(default_factory=dict)
    # Per-key list of token lists (preserves every occurrence, unparsed).

    issues: list[ValidationIssue] = field(default_factory=list)
    # Type-conversion errors. Missing-key / semantic errors come from the validator.

    unknown_keys: list[tuple[str, int]] = field(default_factory=list)
    # Keys found in file but not in registry (with line numbers). Useful for
    # catching typos like "Completenes" → "Completeness".


# ─── Stage 1: lexical parse ──────────────────────────────────────────────────


def _tokenize_line(line: str) -> list[str]:
    """Strip inline comment, split on whitespace, return tokens.

    Returns [] for blank / comment-only lines.
    """
    # Strip inline comment
    hash_at = line.find("#")
    if hash_at >= 0:
        line = line[:hash_at]
    return line.split()


def parse_raw(param_file: str | FsPath) -> tuple[dict[str, list[list[str]]], dict[str, int]]:
    """Read a MIDAS param file, return (raw_tokens, line_of).

    raw_tokens[key] = [[value-tokens-from-occurrence-1], [value-tokens-from-occurrence-2], ...]
    line_of[key]   = 1-indexed line number of the FIRST occurrence.
    """
    raw: dict[str, list[list[str]]] = {}
    line_of: dict[str, int] = {}

    path_str = str(param_file)
    with open(path_str, "r") as fh:
        for line_no, line in enumerate(fh, start=1):
            tokens = _tokenize_line(line)
            if not tokens:
                continue
            key, *values = tokens
            raw.setdefault(key, []).append(values)
            line_of.setdefault(key, line_no)

    return raw, line_of


# ─── Stage 2: type conversion ────────────────────────────────────────────────


def _coerce(tokens: list[str], spec: ParamSpec) -> tuple[Any, str | None]:
    """Coerce raw tokens to the type declared by `spec`.

    Returns (value, error_message). error_message is None on success.
    """
    if spec.type == ParamType.INT:
        if len(tokens) < 1:
            return None, f"{spec.name} expects an int, got no value."
        try:
            return int(tokens[0]), None
        except ValueError:
            return None, f"{spec.name} expects an int, got {tokens[0]!r}."

    if spec.type == ParamType.FLOAT:
        if len(tokens) < 1:
            return None, f"{spec.name} expects a float, got no value."
        try:
            return float(tokens[0]), None
        except ValueError:
            return None, f"{spec.name} expects a float, got {tokens[0]!r}."

    if spec.type == ParamType.BOOL:
        if len(tokens) < 1:
            return None, f"{spec.name} expects 0/1, got no value."
        try:
            v = int(tokens[0])
            if v not in (0, 1):
                return None, f"{spec.name} expects 0 or 1, got {v}."
            return bool(v), None
        except ValueError:
            return None, f"{spec.name} expects 0 or 1, got {tokens[0]!r}."

    if spec.type in (ParamType.STR, ParamType.PATH):
        if len(tokens) < 1:
            return None, f"{spec.name} expects a string, got no value."
        # PATH values can contain spaces — rejoin all tokens. STR takes first token only
        # (matches the MIDAS convention where Dark is one token, but path users paste full paths).
        if spec.type == ParamType.PATH:
            return " ".join(tokens), None
        return tokens[0], None

    if spec.type == ParamType.INT_LIST:
        try:
            return [int(t) for t in tokens], None
        except ValueError as e:
            return None, f"{spec.name} expects ints, got {tokens!r}: {e}"

    if spec.type == ParamType.FLOAT_LIST:
        try:
            return [float(t) for t in tokens], None
        except ValueError as e:
            return None, f"{spec.name} expects floats, got {tokens!r}: {e}"

    # Unknown type — shouldn't happen, registry uses ParamType enum
    return tokens, f"Unhandled param type {spec.type!r} for {spec.name}."


def parse_typed(param_file: str | FsPath) -> ParsedParams:
    """Parse a param file into typed values using the registry.

    Unknown keys are collected in `unknown_keys` (useful for typo detection).
    Aliased keys are stored under the canonical name.
    Multi-entry values are stored as `list[value]` where each element is the
    per-occurrence value.
    """
    raw, line_of_raw = parse_raw(param_file)
    out = ParsedParams(path=str(param_file), raw_tokens=raw)

    registry = by_name()

    for key, occurrences in raw.items():
        spec = registry.get(key)
        if spec is None:
            out.unknown_keys.append((key, line_of_raw[key]))
            continue

        canonical = spec.name  # resolve aliases
        out.line_of.setdefault(canonical, line_of_raw[key])

        coerced_values: list[Any] = []
        for occurrence_tokens in occurrences:
            value, err = _coerce(occurrence_tokens, spec)
            if err is not None:
                out.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    key=canonical,
                    line=line_of_raw[key],
                    message=err,
                    rule="type_coercion",
                ))
            else:
                coerced_values.append(value)

        if not coerced_values:
            continue

        # Assignment rule:
        #   - multi_entry: values[key] is a list of all occurrences
        #   - single-entry: last occurrence wins (matches MIDAS_ParamParser.c behavior
        #     where the last value overwrites — except for multi-entry keys handled
        #     specially)
        if spec.multi_entry:
            # If there's already a value (e.g. an alias wrote first), extend.
            existing = out.values.get(canonical)
            if existing is None:
                out.values[canonical] = coerced_values
            elif isinstance(existing, list):
                existing.extend(coerced_values)
            else:
                out.values[canonical] = [existing] + coerced_values
        else:
            out.values[canonical] = coerced_values[-1]
            if len(coerced_values) > 1:
                out.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    key=canonical,
                    line=line_of_raw[key],
                    message=(
                        f"{canonical} appeared {len(coerced_values)} times but is "
                        f"not a multi-entry key — only the last value is kept."
                    ),
                    rule="duplicate_single_entry",
                ))

    return out
