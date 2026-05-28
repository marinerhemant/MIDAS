"""Stage: merge_overlaps.

Merges peaks whose Y/Z/ω boxes span adjacent frames into single spots
(cross-frame deduplication, complementing peakfit's in-frame fits).
The current FF workflow gets its frame-spanning handling from the
peakfit stage's omega-tail logic, so this stage is a no-op here. A
proper cross-frame merge lands when the bounding-box-merge code from
the legacy ``MergeOverlappingPeaks`` C tool is ported.
"""
from __future__ import annotations

from ._base import StageContext
from ._stub import stub_run
from ..results import StageResult


def run(ctx: StageContext) -> StageResult:
    return stub_run("merge_overlaps", ctx,
                    reason="no-op (cross-frame merge handled inside peakfit)")
