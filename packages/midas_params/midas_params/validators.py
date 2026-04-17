"""Built-in validators referenced by name from ParamSpec.validators.

Each validator is a callable:

    fn(value, spec, report_ctx) -> list[ValidationIssue]

where `value` is the parsed value (type matches `spec.type`), `spec` is the
ParamSpec that declared this validator, and `report_ctx` gives access to
sibling params, the source file line number, and the path (FF/NF/PF/RI)
being validated.

Validators return an empty list on success, or one/more ValidationIssue on
finding problems. They do NOT raise — the engine expects a list.

Validators are looked up by string name in VALIDATORS at the bottom of this
file. ParamSpec entries reference them like:

    validators=("positive", "finite")

To add a new validator: write the function here, add it to the VALIDATORS
dict, and reference it by name from ParamSpec entries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Any, Callable

from .schema import ParamSpec, Severity, ValidationIssue


@dataclass
class Ctx:
    """Read-only context passed to validators.

    Gives each validator access to siblings so cross-field-flavored checks
    ("BC must fit inside NrPixels") can be expressed as a per-key validator
    that peeks at other keys. Full cross-field rules live in crossfield.py
    and have a different signature.
    """

    all_values: dict[str, Any]             # all parsed keys in the file (values post-type-parse)
    param_file: str                        # source file path
    line_of: dict[str, int]                # per-key source line numbers (1-indexed)
    path: str                              # "ff" / "nf" / "pf" / "ri"


Validator = Callable[[Any, ParamSpec, Ctx], list[ValidationIssue]]


# ─── Simple value validators ─────────────────────────────────────────────────


def positive(value: Any, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    if value is None:
        return []
    vals = value if isinstance(value, list) else [value]
    out = []
    for v in vals:
        if isinstance(v, (int, float)) and v <= 0:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"{spec.name} must be positive (got {v}).",
                rule="positive",
            ))
    return out


def non_negative(value: Any, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    if value is None:
        return []
    vals = value if isinstance(value, list) else [value]
    out = []
    for v in vals:
        if isinstance(v, (int, float)) and v < 0:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"{spec.name} must be ≥ 0 (got {v}).",
                rule="non_negative",
            ))
    return out


def finite(value: Any, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    """Reject NaN / ±inf."""
    if value is None:
        return []
    vals = value if isinstance(value, list) else [value]
    out = []
    for v in vals:
        if isinstance(v, float) and not math.isfinite(v):
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"{spec.name}={v} is not finite.",
                rule="finite",
            ))
    return out


def space_group_range(value: int, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    if value is None:
        return []
    if not (1 <= value <= 230):
        return [ValidationIssue(
            severity=Severity.ERROR,
            key=spec.name,
            line=ctx.line_of.get(spec.name),
            message=f"{spec.name}={value} is not a valid space group number (1–230).",
            rule="space_group_range",
        )]
    return []


def wavelength_plausible(value: float, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    if value is None:
        return []
    # Hard X-ray range used in HEDM: roughly 0.1–3 Å
    if not (0.05 <= value <= 5.0):
        return [ValidationIssue(
            severity=Severity.WARNING,
            key=spec.name,
            line=ctx.line_of.get(spec.name),
            message=f"Wavelength={value} Å is outside the typical HEDM range (0.1–3 Å).",
            suggestion="Double-check units; MIDAS expects Angstroms.",
            rule="wavelength_plausible",
        )]
    return []


def lsd_plausible(value: float, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    if value is None:
        return []
    vals = value if isinstance(value, list) else [value]
    out = []
    for v in vals:
        # FF: ~10⁶ µm (1 m). NF: ~10⁴ µm (cm range). Anything < 1000 or > 10⁷ is suspicious.
        if v < 1000 or v > 1e7:
            out.append(ValidationIssue(
                severity=Severity.WARNING,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"Lsd={v} µm is outside the typical range (1e3–1e7 µm). "
                        f"Check units — MIDAS expects microns.",
                rule="lsd_plausible",
            ))
    return out


def space_group_default_smell(value: int, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    """Flag SpaceGroup=225 as possibly accidental: that's the parser default
    for Cu/Au/Ni, and copying an example file often leaves it unchanged."""
    if value == 225:
        return [ValidationIssue(
            severity=Severity.INFO,
            key=spec.name,
            line=ctx.line_of.get(spec.name),
            message="SpaceGroup=225 is the parser default (Fm-3m, Cu/Au/Ni). "
                    "Confirm this matches your sample.",
            rule="space_group_default_smell",
        )]
    return []


def omega_range_arity(value: Any, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    """Each OmegaRange line must have exactly 2 values (ω_min ω_max)."""
    if value is None:
        return []
    entries = value if value and isinstance(value[0], list) else [value]
    out = []
    for i, e in enumerate(entries):
        n = len(e) if isinstance(e, list) else 1
        if n != 2:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"OmegaRange entry {i+1} has {n} values; expected 2 (ω_min ω_max).",
                rule="omega_range_arity",
            ))
    return out


def box_size_arity(value: Any, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    """Each BoxSize line must have exactly 4 values (Ymin Ymax Zmin Zmax)."""
    if value is None:
        return []
    entries = value if value and isinstance(value[0], list) else [value]
    out = []
    for i, e in enumerate(entries):
        n = len(e) if isinstance(e, list) else 1
        if n != 4:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"BoxSize entry {i+1} has {n} values; expected 4 (Ymin Ymax Zmin Zmax).",
                rule="box_size_arity",
            ))
    return out


def omega_range_ordered(value: Any, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    """In each OmegaRange pair, the first value must be less than the second."""
    if value is None:
        return []
    entries = value if value and isinstance(value[0], list) else [value]
    out = []
    for i, e in enumerate(entries):
        if isinstance(e, list) and len(e) == 2 and e[0] >= e[1]:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"OmegaRange entry {i+1} = [{e[0]}, {e[1]}] is empty or reversed; "
                        f"require ω_min < ω_max.",
                rule="omega_range_ordered",
            ))
    return out


def box_size_ordered(value: Any, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    """In each BoxSize, Ymin < Ymax and Zmin < Zmax."""
    if value is None:
        return []
    entries = value if value and isinstance(value[0], list) else [value]
    out = []
    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) == 4):
            continue  # arity check catches this
        ymin, ymax, zmin, zmax = e
        if ymin >= ymax:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"BoxSize entry {i+1}: Ymin={ymin} ≥ Ymax={ymax}.",
                rule="box_size_ordered",
            ))
        if zmin >= zmax:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"BoxSize entry {i+1}: Zmin={zmin} ≥ Zmax={zmax}.",
                rule="box_size_ordered",
            ))
    return out


def bc_in_detector(value: list[float], spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    """BC (Y_px, Z_px) should fall within NrPixels bounds."""
    if value is None:
        return []
    # BC can be multi-entry in NF; normalize to list-of-pairs
    entries = value if value and isinstance(value[0], list) else [value]
    nY = ctx.all_values.get("NrPixelsY") or ctx.all_values.get("NrPixels")
    nZ = ctx.all_values.get("NrPixelsZ") or ctx.all_values.get("NrPixels")
    if not nY or not nZ:
        return []  # can't check without bounds; a different validator will flag missing NrPixels
    out = []
    for i, pair in enumerate(entries):
        if len(pair) < 2:
            continue
        y, z = pair[0], pair[1]
        if not (0 <= y <= nY):
            out.append(ValidationIssue(
                severity=Severity.WARNING,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"BC[{i}]=({y}, {z}) has Y outside detector (0–{nY}).",
                rule="bc_in_detector",
            ))
        if not (0 <= z <= nZ):
            out.append(ValidationIssue(
                severity=Severity.WARNING,
                key=spec.name,
                line=ctx.line_of.get(spec.name),
                message=f"BC[{i}]=({y}, {z}) has Z outside detector (0–{nZ}).",
                rule="bc_in_detector",
            ))
    return out


# ─── Path / file validators ──────────────────────────────────────────────────


def file_exists(value: str, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    if not value:
        return []
    if not FsPath(value).expanduser().exists():
        return [ValidationIssue(
            severity=Severity.ERROR,
            key=spec.name,
            line=ctx.line_of.get(spec.name),
            message=f"{spec.name} points to a missing file: {value}",
            suggestion=f"Check that {value} exists and is readable.",
            rule="file_exists",
        )]
    return []


def directory_exists(value: str, spec: ParamSpec, ctx: Ctx) -> list[ValidationIssue]:
    if not value:
        return []
    p = FsPath(value).expanduser()
    if not p.exists():
        return [ValidationIssue(
            severity=Severity.ERROR,
            key=spec.name,
            line=ctx.line_of.get(spec.name),
            message=f"{spec.name} points to a missing directory: {value}",
            rule="directory_exists",
        )]
    if not p.is_dir():
        return [ValidationIssue(
            severity=Severity.ERROR,
            key=spec.name,
            line=ctx.line_of.get(spec.name),
            message=f"{spec.name}={value} is not a directory.",
            rule="directory_exists",
        )]
    return []


# ─── Lookup registry ─────────────────────────────────────────────────────────


VALIDATORS: dict[str, Validator] = {
    "positive": positive,
    "non_negative": non_negative,
    "finite": finite,
    "space_group_range": space_group_range,
    "space_group_default_smell": space_group_default_smell,
    "wavelength_plausible": wavelength_plausible,
    "lsd_plausible": lsd_plausible,
    "bc_in_detector": bc_in_detector,
    "file_exists": file_exists,
    "directory_exists": directory_exists,
    "omega_range_arity": omega_range_arity,
    "omega_range_ordered": omega_range_ordered,
    "box_size_arity": box_size_arity,
    "box_size_ordered": box_size_ordered,
}


def resolve(name: str) -> Validator:
    """Look up a validator by name. Raises KeyError if unknown — this catches
    registry typos at load time rather than validation time."""
    try:
        return VALIDATORS[name]
    except KeyError:
        raise KeyError(
            f"Unknown validator {name!r}. Known: {sorted(VALIDATORS)}"
        ) from None
