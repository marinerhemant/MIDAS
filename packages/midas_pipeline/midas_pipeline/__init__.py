"""midas-pipeline: Unified MIDAS HEDM orchestrator (FF + PF, single source).

FF is the single-scan degeneracy of PF: ``ScanGeometry.ff()`` produces
``scan_mode='ff'`` with ``n_scans=1``; everything else is a regular PF
run with ``n_scans > 1``. One orchestrator, one CLI, two scan modes.
"""

from __future__ import annotations

# Register the HDF5 filter plugins (bitshuffle, LZ4, zstd) that Eiger / Dectris
# and ESRF files are written with.  hdf5plugin ships the plugin binaries but
# only registers them with the HDF5 library when it is imported — declaring it
# as a dependency is not enough.  Without this, reading such a dataset fails
# with "can't open directory (/usr/local/lib/plugin)".
try:  # pragma: no cover - environment-dependent
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # HDF5 files with no plugin filter still read fine
    pass

__version__ = "0.15.2"

from .config import (
    AlignMethod,
    Device,
    Dtype,
    EMConfig,
    FusionConfig,
    LayerSelection,
    MachineConfig,
    PipelineConfig,
    ProcessGrainsMode,
    ReconConfig,
    ReconMethod,
    RefineLoss,
    RefineMode,
    RefinePositionMode,
    RefineSolver,
    RefinementConfig,
    ResumeMode,
    ScanGeometry,
    ScanMode,
    SeedingConfig,
    SeedingMode,
    SinoSource,
    SinoType,
    SoftAttributionConfig,
    VMapConfig,
    VoxelCleanupConfig,
    sniff_scan_mode_from_paramfile,
)
from .detector import DetectorConfig
from .pipeline import Pipeline, all_stage_names, stage_order_for
from .results import (
    LayerResult,
    StageResult,
    BinningResult,
    CalcRadiusResult,
    CalcRadiusVResult,
    ConsolidationResult,
    CrossDetMergeResult,
    EMRefineResult,
    FindGrainsResult,
    FuseResult,
    HKLResult,
    IndexResult,
    MergeOverlapsResult,
    MergeScansResult,
    PeakFitResult,
    PottsResult,
    ProcessGrainsResult,
    ReconResult,
    RefineResult,
    RefineVmapResult,
    SinogenResult,
    TransformsResult,
)

__all__ = [
    "__version__",
    # config
    "AlignMethod", "Device", "Dtype",
    "EMConfig", "FusionConfig", "LayerSelection", "MachineConfig",
    "PipelineConfig", "ProcessGrainsMode", "ReconConfig", "ReconMethod",
    "RefineLoss", "RefineMode", "RefinePositionMode", "RefineSolver",
    "RefinementConfig", "ResumeMode", "ScanGeometry", "ScanMode",
    "SeedingConfig", "SeedingMode", "SinoSource", "SinoType",
    "SoftAttributionConfig", "VMapConfig", "VoxelCleanupConfig",
    "sniff_scan_mode_from_paramfile",
    # detector
    "DetectorConfig",
    # pipeline
    "Pipeline", "all_stage_names", "stage_order_for",
    # results
    "LayerResult", "StageResult",
    "BinningResult", "CalcRadiusResult", "CalcRadiusVResult",
    "ConsolidationResult",
    "CrossDetMergeResult", "EMRefineResult", "FindGrainsResult",
    "FuseResult", "HKLResult", "IndexResult", "MergeOverlapsResult",
    "MergeScansResult", "PeakFitResult", "PottsResult",
    "ProcessGrainsResult", "ReconResult", "RefineResult", "RefineVmapResult",
    "SinogenResult", "TransformsResult",
]
