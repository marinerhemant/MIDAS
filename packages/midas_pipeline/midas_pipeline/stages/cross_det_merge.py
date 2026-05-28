"""Stage: cross_det_merge. P1 thin-shell — implementation lands in a later phase."""
from __future__ import annotations

from ._base import StageContext
from ._stub import stub_run
from ..results import StageResult


def run(ctx: StageContext) -> StageResult:
    return stub_run("cross_det_merge", ctx, reason="P1 stub — implementation pending")
