"""Stage: cross_det_merge.

Merges fitted peak lists from multiple detector panels into a single
combined spot list. With a single-detector run there is nothing to
merge, so this stage is a no-op. The multi-detector implementation
lives in ``midas_ff_pipeline`` (legacy path) and will be ported once
multi-detector runs are wired into the unified orchestrator.
"""
from __future__ import annotations

from ._base import StageContext
from ._stub import stub_run
from ..results import StageResult


def run(ctx: StageContext) -> StageResult:
    return stub_run("cross_det_merge", ctx,
                    reason="no-op for single-detector runs")
