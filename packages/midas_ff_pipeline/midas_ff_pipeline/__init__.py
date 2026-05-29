"""midas-ff-pipeline — pure-Python FF-HEDM workflow orchestrator.

.. deprecated:: 0.4.0
   ``midas-ff-pipeline`` is being consolidated into the unified
   ``midas-pipeline`` package — FF is the single-scan degeneracy of PF,
   and one orchestrator covers both. Use
   ``midas-pipeline run --scan-mode ff …`` (CLI) or
   ``midas_pipeline.Pipeline(config_with_ScanGeometry.ff())`` (API)
   instead. This package will be removed in 1.0.0; until then it
   continues to function but emits a deprecation warning on import.

   See ``packages/MIDAS_FF_PIPELINE_DEPRECATION_PLAN.md`` in the MIDAS
   repository for the consolidation plan + migration timeline.

Public API:

    Pipeline           main orchestrator class
    PipelineConfig     configuration dataclass
    DetectorConfig     per-detector geometry dataclass
    LayerSelection     start/end layer range
    MachineConfig      cluster/local execution settings
    LayerResult        per-layer result aggregate
"""

from __future__ import annotations

import warnings as _warnings

__version__ = "0.4.0"

# Emit the deprecation notice once at import time. Use DeprecationWarning
# (default-filtered by Python, but shown by pytest / explicit -W /
# the CLI wrapper below) so library consumers don't get spammed on every
# call but interactive / test users see it loud and clear.
_warnings.warn(
    "midas-ff-pipeline is deprecated as of 0.4.0 and will be removed in "
    "1.0.0. Use `midas-pipeline run --scan-mode ff` (CLI) or "
    "`midas_pipeline.Pipeline(...)` (API) instead — the FF path is the "
    "same code under the hood. See the MIDAS_FF_PIPELINE_DEPRECATION_PLAN.md "
    "in the MIDAS repo for the migration timeline.",
    DeprecationWarning,
    stacklevel=2,
)

from .config import LayerSelection, MachineConfig, PipelineConfig
from .detector import DetectorConfig
from .pipeline import Pipeline
from .results import LayerResult

__all__ = [
    "__version__",
    "Pipeline",
    "PipelineConfig",
    "DetectorConfig",
    "LayerSelection",
    "MachineConfig",
    "LayerResult",
]
