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


def _fmt_value(v: Any) -> str:
    """Render a value for bracketed display in prompts."""
    if isinstance(v, list):
        # For multi-entry values (list of per-occurrence values), show each
        # occurrence space-joined internally and comma-separated across occurrences.
        if v and isinstance(v[0], list):
            return "; ".join(" ".join(str(x) for x in entry) for entry in v)
        return " ".join(str(x) for x in v)
    return str(v)


def _default_display(spec: ParamSpec, seed_value: Any) -> str:
    """What to show in brackets as the current/default value."""
    if seed_value is not None:
        return _fmt_value(seed_value)
    if spec.typical is not None:
        return f"typical: {_fmt_value(spec.typical)}"
    if spec.default is not None:
        return f"default: {_fmt_value(spec.default)}"
    return ""


# ─── Prompting ───────────────────────────────────────────────────────────────


@dataclass
class WizardState:
    """Running state of the interactive session."""

    values: dict[str, Any]           # final values keyed by canonical spec name
    seed: dict[str, Any]             # starting defaults (from calibration/discovery)
    source: dict[str, str]           # how each seed value was obtained
    path: Path                       # FF / NF / PF / RI


def _prompt_for(spec: ParamSpec, state: WizardState) -> str | None:
    """Prompt the user for a single parameter; update state.values.

    Returns 'back' if the user requested to go to the previous prompt.
    Returns None otherwise (prompt completed normally).
    """
    # Show any previously-entered value as the default (so "back" revisits
    # with the old input, not just the seed).
    previously_entered = state.values.get(spec.name)
    seed = state.seed.get(spec.name)
    shown_value = previously_entered if previously_entered is not None else seed

    is_required = state.path in spec.required_for
    prefix = "*" if is_required else " "
    display_default = _default_display(spec, shown_value)
    src = state.source.get(spec.name, "")
    # Mark previously-entered values distinctly from seeds
    if previously_entered is not None:
        src_str = " (you entered)"
    elif src and seed is not None:
        src_str = f" (from {src})"
    else:
        src_str = ""

    # Layout:
    #   * <name> [<default>] (from <src>)
    #         <description>  [units]
    #         (notes, if any)
    #         > _
    hint = spec.description
    if spec.units:
        hint += f"  [{spec.units}]"
    bracket = f" [{display_default}]" if display_default else ""
    header_line = f"  {prefix} {spec.name}{bracket}{src_str}"
    input_prompt = "        > "

    print()                                   # blank separator between keys
    print(header_line)
    print(f"        {hint}")
    if spec.notes:
        print(f"        ({spec.notes})")

    if spec.multi_entry:
        return _prompt_multi_entry(spec, state, shown_value, input_prompt, is_required)

    while True:
        resp = input(input_prompt).strip()
        # Navigation commands
        if resp.lower() in ("back", "b", "!back"):
            return "back"
        if resp.lower() in ("skip", "!skip", "del", "delete", "!del", "drop"):
            state.values.pop(spec.name, None)
            if is_required:
                print(f"        = (deleted — {spec.name} will NOT be written. "
                      f"It's required; validator will flag it after the file is saved.)")
            else:
                print(f"        = (deleted — {spec.name} will NOT be written; "
                      f"MIDAS will use its internal default.)")
            return None
        if resp.lower() in ("?", "help", "!help"):
            _print_nav_help()
            continue

        if not resp:
            # Accept shown value (previously-entered, seed, typical, or default)
            if previously_entered is not None:
                _confirm(spec, previously_entered, "you entered")
                return None  # already in state.values
            if seed is not None:
                state.values[spec.name] = seed
                _confirm(spec, seed, "seed")
                return None
            if spec.typical is not None:
                state.values[spec.name] = spec.typical
                _confirm(spec, spec.typical, "typical")
                return None
            if spec.default is not None:
                state.values[spec.name] = spec.default
                _confirm(spec, spec.default, "default")
                return None
            if is_required:
                print(f"        !! '{spec.name}' is required. Please enter a value "
                      f"(or 'back' to revisit a prior prompt).")
                continue
            return None  # optional + no default → skip
        value, err = _parse_one_entry(resp, spec)
        if err:
            print(f"        !! {err}")
            continue
        state.values[spec.name] = value
        _confirm(spec, value, "you entered")
        return None


def _confirm(spec: ParamSpec, value: Any, origin: str) -> None:
    """Echo the accepted value so the user can verify what was recorded."""
    if value is None:
        print(f"        = (skipped — no value recorded for {spec.name})")
        return
    formatted = _fmt_value(value)
    # Guard against very long list renders
    if len(formatted) > 120:
        formatted = formatted[:117] + "..."
    print(f"        = {spec.name}: {formatted}  ({origin})")


def _print_nav_help() -> None:
    print("        Navigation:")
    print("          <Enter>            accept the bracketed value")
    print("          back / b           go back to the previous prompt")
    print("          skip / del / drop  delete this parameter (don't write it)")
    print("          ?                  this help")
    print("        Deleting a required parameter is allowed but the validator")
    print("        will flag it after save — useful for letting MIDAS fall back")
    print("        to its internal defaults.")


def _derive_seeds(state: WizardState) -> None:
    """Compute values that are derivable from other keys, and add them to
    state.seed so the user sees them as pre-filled.

    Current derivations:
      - OmegaEnd = OmegaStart + OmegaStep × (EndNr − StartNr + 1)

    Only fills in seeds that are NOT already present (user input or an
    explicit seed always wins).
    """
    # Merge current values with seeds for the computation (user input takes priority)
    v = {**state.seed, **{k: val for k, val in state.values.items() if val is not None}}

    # Derive OmegaEnd if possible and not already set
    if "OmegaEnd" not in v:
        needed = ("OmegaStart", "OmegaStep", "StartNr", "EndNr")
        if all(k in v for k in needed):
            try:
                ostart = float(v["OmegaStart"])
                ostep = float(v["OmegaStep"])
                snr = int(v["StartNr"])
                enr = int(v["EndNr"])
                nframes = enr - snr + 1
                omega_end = ostart + ostep * nframes
                state.seed["OmegaEnd"] = omega_end
                state.source["OmegaEnd"] = "derived from OmegaStart+OmegaStep×nFrames"
            except (TypeError, ValueError):
                pass


def _prompt_multi_entry(spec: ParamSpec, state: WizardState, seed,
                         input_prompt: str, is_required: bool) -> str | None:
    """Loop until user enters blank. Accept seed as starting list.

    Returns 'back' if user typed back before adding any entries.

    If there is no explicit seed but the spec has a default, treat the
    default as an implicit seed — so pressing Enter accepts it without
    the user having to type the default manually.
    """
    entries: list[Any] = []
    # Fall back to spec.default (or typical) if no explicit seed.
    effective_seed = seed
    if effective_seed is None and spec.default is not None:
        effective_seed = spec.default
        seed_source_note = "(default)"
    elif effective_seed is None and spec.typical is not None:
        effective_seed = spec.typical
        seed_source_note = "(typical)"
    else:
        seed_source_note = ""

    kept_from_seed = False
    if effective_seed is not None:
        # Pre-populate from seed; show as a confirmation prompt
        if isinstance(effective_seed, list):
            entries = list(effective_seed)
            plural = "entries" if len(entries) != 1 else "entry"
            src_tag = f" {seed_source_note}" if seed_source_note else ""
            print(f"        (pre-filled: {len(entries)} {plural}{src_tag})")
            for i, e in enumerate(entries):
                print(f"          {i+1}: {e}")
            resp = input("        Keep these? [Y/n/skip/back]: ").strip().lower()
            if resp in ("back", "b"):
                return "back"
            if resp in ("skip", "del", "delete", "drop", "!skip", "!del"):
                state.values.pop(spec.name, None)
                if is_required:
                    print(f"        = (deleted — {spec.name} will NOT be written. "
                          f"It's required; validator will flag it.)")
                else:
                    print(f"        = (deleted — {spec.name} will NOT be written; "
                          f"MIDAS will use its internal default.)")
                return None
            if resp in ("n", "no"):
                entries = []
            else:
                state.values[spec.name] = entries
                kept_from_seed = True
                print(f"        Add more? (blank line to stop)")
                # fall through to entry loop

    print(f"        Enter {spec.name} values one per line "
          f"(blank to finish; 'back' to revisit; 'skip' to delete the key).")
    while True:
        resp = input(input_prompt).strip()
        if resp.lower() in ("back", "b"):
            if not entries:
                return "back"
            # Mid-list: break out and keep the entries typed so far
            print("        (keeping entries so far; use 'back' before typing any "
                  "entry to revisit prior prompts)")
            break
        if resp.lower() in ("skip", "del", "delete", "drop", "!skip", "!del"):
            # Drop the key entirely from the output
            state.values.pop(spec.name, None)
            if is_required:
                print(f"        = (deleted — {spec.name} will NOT be written. "
                      f"It's required; validator will flag it.)")
            else:
                print(f"        = (deleted — {spec.name} will NOT be written; "
                      f"MIDAS will use its internal default.)")
            return None
        if not resp:
            break
        value, err = _parse_one_entry(resp, spec)
        if err:
            print(f"        !! {err}")
            continue
        entries.append(value)

    if entries:
        state.values[spec.name] = entries
        origin = "kept seed" if kept_from_seed and len(entries) == len(effective_seed or []) \
                 else ("seed + added" if kept_from_seed else "you entered")
        _confirm_multi(spec, entries, origin)
    elif state.path in spec.required_for:
        print(f"        !! {spec.name} requires at least one entry.")
    else:
        _confirm(spec, None, "skipped")
    return None


def _confirm_multi(spec: ParamSpec, entries: list[Any], origin: str) -> None:
    """Echo a multi-entry value. Lists with one entry render compactly;
    longer lists get a count and one-per-line summary."""
    if len(entries) == 1:
        print(f"        = {spec.name}: {_fmt_value(entries[0])}  ({origin})")
        return
    print(f"        = {spec.name}: {len(entries)} entries  ({origin})")
    for i, e in enumerate(entries, start=1):
        fmt = _fmt_value(e)
        if len(fmt) > 100:
            fmt = fmt[:97] + "..."
        print(f"            {i}: {fmt}")


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
        _derive_seeds(state)
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

    # Derive values that we can compute from others before prompting.
    _derive_seeds(state)

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
    print("Per prompt:  <Enter>            accept bracketed value")
    print("             back / b           go back to previous prompt")
    print("             skip / del / drop  delete this parameter (don't write it)")
    print("             ?                  show this help")
    print("Required keys are marked with *. Deleting a required key is allowed")
    print("but the validator will flag it after save.")
    print()

    # Build a flat list of (category_index, category_name, spec) so we can
    # step forward or backward by one.
    visible = wizard_visible_for(path)
    by_category: dict[str, list[ParamSpec]] = {}
    for spec in visible:
        by_category.setdefault(spec.category, []).append(spec)
    cat_names = list(by_category)
    flat: list[tuple[int, str, ParamSpec]] = []
    for ci, cat in enumerate(cat_names, start=1):
        for spec in by_category[cat]:
            flat.append((ci, cat, spec))

    shown_categories: set[int] = set()
    i = 0
    while i < len(flat):
        ci, cat, spec = flat[i]
        # Print category banner once per new category
        if ci not in shown_categories:
            print()
            print(f"[{ci}/{len(cat_names)}] {cat}")
            print("-" * (len(cat) + 8))
            shown_categories.add(ci)
        try:
            action = _prompt_for(spec, state)
        except (KeyboardInterrupt, EOFError):
            print()
            print("Wizard aborted. No file written.")
            return 130
        if action == "back":
            if i == 0:
                print("        (already at the first prompt)")
                continue
            # Re-derive seeds in case a derived value would change
            _derive_seeds(state)
            i -= 1
            continue
        i += 1

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
