"""Pipeline stages.

Each stage exposes ``run(ctx: StageContext) -> StageResult``. P1 ships
thin shells that mark themselves ``skipped=True`` and emit a warning.
Real implementations land in P2–P8 — see the relevant phase in the plan
file for the ownership boundary.

The thin shells exist so:
- ``Pipeline._run_layer`` has something to call per stage.
- Parallel-stream developers (P2–P8) have a clear single file to fill in.
- Resume / provenance integration is testable from day one.
"""

from . import (
    binning,
    calc_radius,
    calc_radius_v,
    consolidation,
    cross_det_merge,
    find_grains_stage,
    global_powder,
    hkl,
    indexing,
    merge_overlaps,
    merge_scans,
    peakfit,
    process_grains,
    refine_vmap,
    refinement,
    seeding,
    sinogen,
    transforms,
    zip_convert,
)

# Optional stages — stream-A relocates these out of pf_MIDAS.py in P2.
# Import softly so the package loads even when those files haven't been
# committed yet (they're cross-stream dependencies).
try:
    from . import em_refine  # noqa: F401
except ImportError:
    em_refine = None  # type: ignore
try:
    from . import fuse  # noqa: F401
except ImportError:
    fuse = None  # type: ignore
try:
    from . import potts  # noqa: F401
except ImportError:
    potts = None  # type: ignore
try:
    from . import reconstruct  # noqa: F401
except ImportError:
    reconstruct = None  # type: ignore

__all__ = [
    "binning",
    "calc_radius",
    "calc_radius_v",
    "consolidation",
    "cross_det_merge",
    "em_refine",
    "find_grains_stage",
    "fuse",
    "global_powder",
    "hkl",
    "indexing",
    "merge_overlaps",
    "merge_scans",
    "peakfit",
    "potts",
    "process_grains",
    "reconstruct",
    "refinement",
    "refine_vmap",
    "seeding",
    "sinogen",
    "transforms",
    "zip_convert",
]
