"""Data model for the MIDAS parameter registry.

The registry is a list of `ParamSpec` entries (one per parameter key) plus a
list of `CrossFieldRule` entries (multi-key consistency checks). Validators
and wizards are engines that traverse the registry — the schema here is the
single source of truth the engines consume.

Design constraints:
  - Serializable: a ParamSpec must survive `dataclasses.asdict` → JSON/YAML
    round-trip so it can be consumed by LLMs and external tooling. This means
    validator functions are referenced by NAME (strings), not by function
    object. The engine resolves names against `validators.VALIDATORS`.
  - Extensible: per-path applicability (FF/NF/PF/RI) is a set, not a bool,
    so adding a new path (e.g. DT) is one enum value, not a schema change.
  - No inheritance: everything is flat dataclasses for easy pickling/LLM input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParamType(str, Enum):
    """Storage type of a parameter value."""

    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"            # stored as int 0/1 in the param file
    INT_LIST = "int_list"    # space-separated ints on one line
    FLOAT_LIST = "float_list"  # space-separated floats, fixed or variable arity
    PATH = "path"            # str but validated as a filesystem path


class Path(str, Enum):
    """Which MIDAS analysis pipeline this parameter applies to."""

    FF = "ff"   # far-field HEDM
    NF = "nf"   # near-field HEDM
    PF = "pf"   # point-focus HEDM
    RI = "ri"   # radial integration


class Stage(str, Enum):
    """Pipeline stage where the parameter is consumed.

    Used for error messages like "IndexerOMP needs Completeness; set it or
    pass -doPeakSearch 0 to skip indexing".
    """

    FILE_DISCOVERY = "file-discovery"
    IMAGE_PREPROC = "image-preproc"
    INTEGRATION = "integration"
    PEAK_SEARCH = "peak-search"
    SEED_GEN = "seed-gen"
    SPOT_GEN = "spot-gen"
    INDEXING = "indexing"
    REFINEMENT = "refinement"
    CALIBRATION = "calibration"
    FORWARD_SIM = "forward-sim"
    POST_ANALYSIS = "post-analysis"
    MULTI_PANEL = "multi-panel"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single parameter key.

    Used by both the validator (to check user input) and the wizard (to
    prompt for values).
    """

    name: str
    type: ParamType
    category: str                          # human-readable group header (e.g. "Detector Geometry")
    description: str                       # one-line explanation shown in wizard + validator errors
    applies_to: frozenset[Path]            # which paths use this key
    required_for: frozenset[Path] = frozenset()  # subset of applies_to — fatal if absent
    stages: frozenset[Stage] = frozenset()       # pipeline stages that consume this value
    units: str | None = None               # "um", "deg", "Å", "pixels", "counts", etc.
    default: Any = None                    # authoritative default from source; None = no default
    typical: Any = None                    # recommended starting value (different from default);
                                           # wizard prefers this over default when prompting
    multi_entry: bool = False              # key may appear multiple times; values accumulate into a list
    aliases: tuple[str, ...] = ()          # alternate key names (e.g. "LatticeConstant" → "LatticeParameter")
    zarr_rename: str | None = None         # name in the FF Zarr analysis file (for FF path only)
    validators: tuple[str, ...] = ()       # names looked up in validators.VALIDATORS; each runs
                                           # against the parsed value and returns list[ValidationIssue]
    hidden_in_wizard: bool = False         # calibration-only / forward-sim-only: exclude from wizard prompts
    notes: str | None = None               # longer-form context (shown with `--explain` flag, not in prompts)


@dataclass(frozen=True)
class CrossFieldRule:
    """A consistency check that spans multiple keys.

    Example: `nDistances` count must equal the number of `Lsd` entries. The
    check function takes the full parsed parameter dict and returns issues.
    """

    name: str                              # stable identifier, e.g. "nf_multi_entry_count_matches"
    description: str                       # what this rule checks
    applies_to: frozenset[Path]            # paths where this rule runs
    severity: Severity = Severity.ERROR
    check: str = ""                        # name in crossfield.RULES; engine resolves at runtime


@dataclass(frozen=True)
class ValidationIssue:
    """A single finding from the validator.

    Issues are structured (not free-text) so downstream tools — a CLI
    pretty-printer, an LSP-style IDE plugin, or an LLM diagnosis layer —
    can consume them uniformly.
    """

    severity: Severity
    message: str                           # human-readable primary message
    key: str | None = None                 # key this relates to (None for missing-key / cross-field)
    line: int | None = None                # line number in the source param file, if known
    suggestion: str | None = None          # actionable next step (e.g. "set RingThresh 1 100")
    rule: str | None = None                # validator name or cross-field rule name that fired
    stage: Stage | None = None             # which pipeline stage would fail because of this


@dataclass
class ValidationReport:
    """Aggregated validator output for one parameter file."""

    param_file: str
    path: Path                             # which pipeline this file was validated against
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors
