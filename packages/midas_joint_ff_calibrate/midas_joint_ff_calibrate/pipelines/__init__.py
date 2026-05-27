"""Joint-calibration pipelines.

Three drivers, each operating on the same joint :class:`ParameterSpec` +
joint residual closure:

  - :mod:`alternating` — recommended default.  Outer loop alternates
    between (geometry + grain orientations + grain positions) and (grain
    strains).  Cheap, robust, paper-3 numbers carry over directly.
  - :mod:`full_joint` — refine every refined parameter at once with a
    single LM call.  Use after alternating provides a good init, or when
    Gaussian priors on grain strain make it well-conditioned.
  - :mod:`identifiability` — Fisher-rank diagnostic.  Reports the rank of
    the Fisher block on a user-chosen parameter subset under powder-only,
    HEDM-only, and joint evidence.  This is the headline figure of the
    paper.
"""
from midas_joint_ff_calibrate.pipelines.alternating import (
    AlternatingDriver,
    AlternatingResult,
)
from midas_joint_ff_calibrate.pipelines.full_joint import (
    FullJointDriver,
    FullJointResult,
)
from midas_joint_ff_calibrate.pipelines.identifiability import (
    FisherBlockReport,
    fisher_block_rank,
)

__all__ = [
    "AlternatingDriver",
    "AlternatingResult",
    "FullJointDriver",
    "FullJointResult",
    "FisherBlockReport",
    "fisher_block_rank",
]
