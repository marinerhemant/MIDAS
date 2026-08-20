"""MIDAS parameter-file registry, validator, and wizard.

Public surface (stable):
  - ParamSpec, CrossFieldRule, ValidationIssue, ValidationReport  (schema)
  - Path, Stage, Severity, ParamType                               (enums)
  - PARAMS, by_name, for_path, required_for                        (registry)
  - RULE_SPECS                                                     (cross-field rules)
  - VALIDATORS, resolve                                            (validator lookup)
"""

# Register the HDF5 filter plugins (bitshuffle, LZ4, zstd) that Eiger / Dectris
# and ESRF files are written with.  hdf5plugin ships the plugin binaries but
# only registers them with the HDF5 library when it is imported — declaring it
# as a dependency is not enough.  Without this, reading such a dataset fails
# with "can't open directory (/usr/local/lib/plugin)".
try:  # pragma: no cover - environment-dependent
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # HDF5 files with no plugin filter still read fine
    pass

__version__ = "0.10.0"

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
from .notebook import build_paramstest, seeds_from_calibration_result

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
    "build_paramstest",
    "by_name",
    "discover_from_file",
    "discover_from_calibration_file",
    "for_path",
    "merge",
    "required_for",
    "resolve",
    "seeds_from_calibration_result",
    "wizard_visible_for",
]
