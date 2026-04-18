"""MIDAS parameter-file registry, validator, and wizard.

Public surface (stable):
  - ParamSpec, CrossFieldRule, ValidationIssue, ValidationReport  (schema)
  - Path, Stage, Severity, ParamType                               (enums)
  - PARAMS, by_name, for_path, required_for                        (registry)
  - RULE_SPECS                                                     (cross-field rules)
  - VALIDATORS, resolve                                            (validator lookup)
"""

__version__ = "0.1.3"

from .schema import (
    CrossFieldRule,
    ParamSpec,
    ParamType,
    Path,
    Severity,
    Stage,
    ValidationIssue,
    ValidationReport,
)
from .registry import PARAMS, by_name, for_path, required_for, wizard_visible_for
from .crossfield import RULE_SPECS, RULES
from .validators import VALIDATORS, Ctx, resolve
from .discovery import DiscoveryResult, discover_from_file, discover_from_calibration_file, merge

__all__ = [
    "CrossFieldRule",
    "Ctx",
    "DiscoveryResult",
    "PARAMS",
    "ParamSpec",
    "ParamType",
    "Path",
    "RULES",
    "RULE_SPECS",
    "Severity",
    "Stage",
    "VALIDATORS",
    "ValidationIssue",
    "ValidationReport",
    "by_name",
    "discover_from_file",
    "discover_from_calibration_file",
    "for_path",
    "merge",
    "required_for",
    "resolve",
    "wizard_visible_for",
]
