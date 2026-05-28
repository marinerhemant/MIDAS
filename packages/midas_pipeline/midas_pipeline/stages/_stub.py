"""Helper for skipped pipeline stages.

Two reasons a stage can be skipped:

  * **disabled** — the stage has a real implementation, but the run's
    config didn't enable it (e.g. ``grain_geometry.run=False``,
    ``vmap.run=False``). Most "skip" messages in a typical run are of
    this kind.
  * **stub** — the stage is a thin shell with no implementation yet
    (P1 stages still awaiting their P2-P8 owner).

Both paths return the same ``StageResult(skipped=True)`` shape.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .._logging import LOG, stage_timer
from ..results import StageResult

if TYPE_CHECKING:
    from ._base import StageContext


def stub_run(stage_name: str, ctx: "StageContext", *,
             reason: str = "not enabled in this config") -> StageResult:
    """Return a skipped StageResult.

    Parameters
    ----------
    stage_name : str
        Name of the stage being skipped.
    ctx : StageContext
        Stage context (used for scan_mode logging).
    reason : str, optional
        Why the stage is skipped. Defaults to "not enabled in this config"
        — for stages with real implementations that are config-gated.
        Genuine P1 stubs (no implementation) should pass
        ``reason="P1 stub — implementation pending"``.
    """
    LOG.info(
        "stage '%s' skipped (%s; scan_mode=%s).",
        stage_name, reason, ctx.scan_mode,
    )
    now = time.time()
    return StageResult(
        stage_name=stage_name,
        started_at=now,
        finished_at=now,
        duration_s=0.0,
        inputs={},
        outputs={},
        metrics={"skipped_reason": reason, "scan_mode": ctx.scan_mode},
        skipped=True,
    )
