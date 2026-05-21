"""midas-ff-pipeline — pure-Python FF-HEDM workflow orchestrator.

Public API:

    Pipeline           main orchestrator class
    PipelineConfig     configuration dataclass
    DetectorConfig     per-detector geometry dataclass
    LayerSelection     start/end layer range
    MachineConfig      cluster/local execution settings
    LayerResult        per-layer result aggregate
"""

from __future__ import annotations

__version__ = "0.3.2"

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
