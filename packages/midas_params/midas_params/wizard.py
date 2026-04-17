"""Interactive parameter-file builder.

Design principles:
  - Prompts are grouped by category (Data source, Detector geometry, …).
  - Each prompt shows `[default]` in brackets; Enter accepts.
  - Seed values from: (1) `--from-existing` file, (2) `--from-calibration`
    file, (3) `--dataset` auto-discovery, (4) registry.typical, (5) registry.default.
  - Multi-entry keys loop until the user types blank.
  - Required keys cannot be skipped; optional ones can.
  - Preview + diff at the end; write only on confirmation.
  - `--non-interactive` path validates the merged seed values without prompting —
    useful for CI.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Any

from .discovery import (
    DiscoveryResult,
    discover_from_calibration_file,
    discover_from_file,
    merge,
)
from .registry import for_path, required_for, wizard_visible_for
from .schema import ParamSpec, ParamType, Path, Severity


# ─── Helpers for prompting ───────────────────────────────────────────────────


def _format_value_for_file(spec: ParamSpec, value: Any) -> list[str]:
    """Return one or more text lines to write for a (spec, value) pair.

    Multi-entry keys produce multiple lines (one per occurrence).
    Lists (e.g. LatticeConstant) are written space-separated on one line.
    """
    if value is None:
        return []

    def _fmt_one(v: Any) -> str:
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, list):
            return " ".join(_fmt_one(x) for x in v)
        return str(v)

    if spec.multi_entry:
        # value should be a list — one occurrence per element
        if not isinstance(value, list):
            value = [value]
        return [f"{spec.name} {_fmt_one(v)}" for v in value]

    return [f"{spec.name} {_fmt_one(value)}"]


def _parse_one_entry(text: str, spec: ParamSpec) -> tuple[Any, str | None]:
    """Parse a single user-typed string into the spec's type.
    Returns (value, error_message)."""
    tokens = text.strip().split()
    if not tokens:
        return None, "empty"

    if spec.type == ParamType.INT:
        try:
            return int(tokens[0]), None
        except ValueError:
            return None, f"expected int, got {tokens[0]!r}"
    if spec.type == ParamType.FLOAT:
        try:
            return float(tokens[0]), None
        except ValueError:
            return None, f"expected float, got {tokens[0]!r}"
    if spec.type == ParamType.BOOL:
        if tokens[0].lower() in ("y", "yes", "true", "1"):
            return True, None
        if tokens[0].lower() in ("n", "no", "false", "0"):
            return False, None
        return None, f"expected yes/no or 0/1, got {tokens[0]!r}"
    if spec.type in (ParamType.STR, ParamType.PATH):
        return " ".join(tokens) if spec.type == ParamType.PATH else tokens[0], None
    if spec.type == ParamType.INT_LIST:
        try:
            return [int(t) for t in tokens], None
        except ValueError as e:
            return None, str(e)
    if spec.type == ParamType.FLOAT_LIST:
        try:
            return [float(t) for t in tokens], None
        except ValueError as e:
            return None, str(e)
    return tokens, None


def _default_display(spec: ParamSpec, seed_value: Any) -> str:
    """What to show in brackets as the current/default value."""
    if seed_value is not None:
        if isinstance(seed_value, list):
            return " ".join(str(v) for v in seed_value)
        return str(seed_value)
    if spec.typical is not None:
        return f"typical: {spec.typical}"
    if spec.default is not None:
        return f"default: {spec.default}"
    return ""


# ─── Prompting ───────────────────────────────────────────────────────────────


@dataclass
class WizardState:
    """Running state of the interactive session."""

    values: dict[str, Any]           # final values keyed by canonical spec name
    seed: dict[str, Any]             # starting defaults (from calibration/discovery)
    source: dict[str, str]           # how each seed value was obtained
    path: Path                       # FF / NF / PF / RI


def _prompt_for(spec: ParamSpec, state: WizardState) -> None:
    """Prompt the user for a single parameter; update state.values."""
    seed = state.seed.get(spec.name)
    is_required = state.path in spec.required_for
    prefix = "*" if is_required else " "
    display_default = _default_display(spec, seed)
    src = state.source.get(spec.name, "")
    src_str = f" (from {src})" if src and seed is not None else ""

    # Build the prompt
    hint = spec.description
    if spec.units:
        hint += f"  [{spec.units}]"
    bracket = f" [{display_default}]" if display_default else ""
    prompt_line = f"  {prefix} {spec.name}{bracket}{src_str}: "

    print(f"    {hint}")
    if spec.notes:
        print(f"    ({spec.notes})")

    if spec.multi_entry:
        _prompt_multi_entry(spec, state, seed, prompt_line)
        return

    while True:
        resp = input(prompt_line).strip()
        if not resp:
            # Accept seed / default
            if seed is not None:
                state.values[spec.name] = seed
                return
            if spec.default is not None:
                state.values[spec.name] = spec.default
                return
            if spec.typical is not None:
                state.values[spec.name] = spec.typical
                return
            if is_required:
                print(f"    '{spec.name}' is required. Please enter a value.")
                continue
            return  # optional + no default → skip
        value, err = _parse_one_entry(resp, spec)
        if err:
            print(f"    !! {err}")
            continue
        state.values[spec.name] = value
        return


def _prompt_multi_entry(spec: ParamSpec, state: WizardState, seed, prompt_line: str) -> None:
    """Loop until user enters blank. Accept seed as starting list."""
    entries: list[Any] = []
    if seed is not None:
        # Pre-populate from seed; show as a confirmation prompt
        seed_list = seed if isinstance(seed, list) and (
            not seed or not isinstance(seed[0], list)
        ) else seed
        # seed for multi-entry is itself a list of occurrences
        if isinstance(seed, list):
            entries = list(seed)
            print(f"    (existing entries: {len(entries)})")
            for i, e in enumerate(entries):
                print(f"      {i+1}: {e}")
            resp = input(f"    Keep these? [Y/n]: ").strip().lower()
            if resp in ("n", "no"):
                entries = []
            else:
                state.values[spec.name] = entries
                print(f"    Add more? (blank line to stop)")
                # fall through to entry loop

    print(f"    Enter {spec.name} values one per line (blank to finish).")
    while True:
        resp = input(f"      {spec.name}: ").strip()
        if not resp:
            break
        value, err = _parse_one_entry(resp, spec)
        if err:
            print(f"    !! {err}")
            continue
        entries.append(value)

    if entries:
        state.values[spec.name] = entries
    elif state.path in spec.required_for:
        print(f"    !! {spec.name} requires at least one entry.")


# ─── Top-level entry point ───────────────────────────────────────────────────


def run_wizard(
    path: Path,
    output: str,
    from_calibration: str | None = None,
    from_existing: str | None = None,
    dataset_file: str | None = None,
    non_interactive: bool = False,
) -> int:
    """Run the wizard and write a parameter file. Returns exit code."""

    # ── Gather seeds from all sources (priority: existing > calibration > dataset)
    sources_applied: list[DiscoveryResult] = []
    if from_existing:
        sources_applied.append(discover_from_calibration_file(from_existing))
    if from_calibration:
        sources_applied.append(discover_from_calibration_file(from_calibration))
    if dataset_file:
        sources_applied.append(discover_from_file(dataset_file))
    merged = merge(*sources_applied) if sources_applied else DiscoveryResult()

    state = WizardState(
        values={},
        seed=dict(merged.extracted),
        source=dict(merged.source),
        path=path,
    )

    if non_interactive:
        # Use seeds + defaults; no prompting. Fail if any required key still missing.
        for spec in required_for(path):
            if spec.name in state.seed:
                state.values[spec.name] = state.seed[spec.name]
            elif spec.typical is not None:
                state.values[spec.name] = spec.typical
            elif spec.default is not None:
                state.values[spec.name] = spec.default
        # Fill optionals from seeds
        for spec in wizard_visible_for(path):
            if spec.name not in state.values and spec.name in state.seed:
                state.values[spec.name] = state.seed[spec.name]
        missing_required = [
            s.name for s in required_for(path) if s.name not in state.values
        ]
        if missing_required:
            print(f"Missing required values: {missing_required}", file=sys.stderr)
            return 2
        _write_param_file(state, output)
        return _validate_and_report(output, path)

    # ── Interactive flow ─────────────────────────────────────────────────────
    print()
    print(f"MIDAS parameter wizard — {path.value.upper()} pipeline")
    print("=" * 60)
    if merged.extracted:
        print(f"Loaded {len(merged.extracted)} seed value(s) from "
              f"{len(sources_applied)} source(s).")
    if merged.warnings:
        print("Warnings during seed extraction:")
        for w in merged.warnings:
            print(f"  - {w}")
    print()
    print("For each prompt: press Enter to accept the bracketed value, or type a new one.")
    print("Required keys are marked with *. Optional prompts may be skipped with Enter.")
    print()

    # Group specs by category
    visible = wizard_visible_for(path)
    by_category: dict[str, list[ParamSpec]] = {}
    for spec in visible:
        by_category.setdefault(spec.category, []).append(spec)

    for i, (cat, specs) in enumerate(by_category.items(), start=1):
        print()
        print(f"[{i}/{len(by_category)}] {cat}")
        print("-" * (len(cat) + 8))
        for spec in specs:
            try:
                _prompt_for(spec, state)
            except (KeyboardInterrupt, EOFError):
                print()
                print("Wizard aborted. No file written.")
                return 130

    # Preview
    print()
    print("Preview of parameter file:")
    print("=" * 60)
    preview = _render_param_file(state)
    print(preview)
    print("=" * 60)

    try:
        confirm = input(f"Write to {output}? [Y/n]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        confirm = "n"
    if confirm in ("n", "no"):
        print("Aborted. No file written.")
        return 1

    _write_param_file(state, output)
    print(f"Wrote {output}")

    return _validate_and_report(output, path)


# ─── File writing + post-validate ───────────────────────────────────────────


def _render_param_file(state: WizardState) -> str:
    """Build the text content of the param file from state.values."""
    specs_by_name = {s.name: s for s in for_path(state.path)}
    lines: list[str] = [
        f"# MIDAS parameter file — {state.path.value.upper()} pipeline",
        f"# Generated by midas-params wizard",
        "",
    ]
    # Group by category in registry order
    current_cat: str | None = None
    for spec in for_path(state.path):
        if spec.name not in state.values:
            continue
        if spec.category != current_cat:
            lines.append("")
            lines.append(f"# {spec.category}")
            current_cat = spec.category
        lines.extend(_format_value_for_file(spec, state.values[spec.name]))
    return "\n".join(lines) + "\n"


def _write_param_file(state: WizardState, output: str) -> None:
    content = _render_param_file(state)
    FsPath(output).expanduser().write_text(content)


def _validate_and_report(output: str, path: Path) -> int:
    from .validator import format_report, validate

    report = validate(output, path)
    print()
    print("Post-write validation:")
    print(format_report(report, use_color=sys.stdout.isatty()))
    return 0 if report.ok else 1
