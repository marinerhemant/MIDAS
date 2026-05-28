"""Stage: global_powder. P1 thin-shell — implementation lands in a later phase."""
from __future__ import annotations

from ._base import StageContext
from ._stub import stub_run
from ..results import StageResult


def run(ctx: StageContext) -> StageResult:
    return stub_run("global_powder", ctx, reason="P1 stub — implementation pending")
