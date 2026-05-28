"""Stage: global_powder.

Aggregates per-scan/per-detector powder rings into a global powder pattern
for an in-pipeline calibration refinement step. Optional — the standalone
``midas-calibrate-v2`` pre-step in notebook 06 already handles calibration
upstream, so this stage is a no-op for the FF demo workflow. Will land
when in-pipeline geometry refinement is enabled.
"""
from __future__ import annotations

from ._base import StageContext
from ._stub import stub_run
from ..results import StageResult


def run(ctx: StageContext) -> StageResult:
    return stub_run("global_powder", ctx,
                    reason="no-op (calibration done upstream via midas-calibrate-v2)")
