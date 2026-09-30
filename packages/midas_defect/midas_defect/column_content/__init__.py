"""Column content: which crystal orientations are present in one interaction volume (raster point), each with its
intensity share and orientation spread, and how completely that list explains the detected intensity.

Monochromatic rotation (DAC) implementation of the column-content method (manuals/column-content/). Modules:
forward (differentiable reflection prediction on midas_defect geometry), kernel (measured anisotropic kernel),
fit (joint mixture fit of all orientations on shared voxels), synthetic (columns of known content), pipeline (ingest -> round 0 -> joint fit -> discovery), validate
(synthetic-column validation at YOUR geometry; the recall table is the completeness statement).
"""
from .forward import Observable, guard, observable_reflections, predict_torch, rotvec_to_matrix
from .kernel import GaussKernel3D, estimate_kernel
from .fit import ColumnFit
from .synthetic import ColumnTruth, DomainSpec, synthetic_column
from .validate import DEFAULT_GATES, ValidationSpec, build_column, evaluate, measure_gate, validate
from .pipeline import ColumnResult, Ingested, discovery_gate, ingest_column, misorientation_deg, run_column, spots_to_q

__all__ = ["Observable", "guard", "observable_reflections", "predict_torch", "rotvec_to_matrix",
           "GaussKernel3D", "estimate_kernel", "ColumnFit", "ColumnTruth", "DomainSpec", "synthetic_column",
           "ColumnResult", "Ingested", "discovery_gate", "ingest_column", "misorientation_deg", "run_column", "spots_to_q",
           "DEFAULT_GATES", "ValidationSpec", "build_column", "evaluate", "measure_gate", "validate"]
