"""Stage: calc_radius.

P1 thin-shell — the pf-pipeline doesn't run a per-detector ``calc_radius``
(that is the FF pipeline's job, owned by
:mod:`midas_ff_pipeline.stages.calc_radius`).  The V-map calc_radius for
PF + FF compact mode lives in :mod:`midas_pipeline.stages.calc_radius_v`.
"""

from __future__ import annotations

from ._base import StageContext
from ._stub import stub_run
from ..results import StageResult


def run(ctx: StageContext) -> StageResult:
    return stub_run("calc_radius", ctx)
