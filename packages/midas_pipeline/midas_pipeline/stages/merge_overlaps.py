"""Stage: merge_overlaps — intentional no-op; the merge lives in `transforms`.

N12 clarification (this docstring previously claimed the legacy
``MergeOverlappingPeaks`` port was "pending", which mis-led two campaigns
into auditing a gap that does not exist):

The cross-frame merge IS fully ported and executed — just not in this
stage. ``midas_transforms.merge.merge_overlapping_peaks`` is the Python
port of the C ``MergeOverlappingPeaksAllZarr`` (frame-by-frame
mutual-nearest + optional pixel-overlap matching), validated BYTE-EXACT
against C goldens (``midas_transforms/tests/test_regression_vs_c.py``:
merge row-for-row at float64, plus total-intensity conservation). The
``transforms`` stage calls it per scan (PF) / per layer (FF) as the first
step of its merge → calc_radius → fit_setup chain, exactly mirroring the
C stage order.

This stage slot exists only to mirror the legacy per-stage checkpoint
layout; running it is free and always succeeds.
"""
from __future__ import annotations

from ._base import StageContext
from ._stub import stub_run
from ..results import StageResult


def run(ctx: StageContext) -> StageResult:
    return stub_run(
        "merge_overlaps", ctx,
        reason=(
            "no-op by design: the cross-frame merge runs inside the "
            "'transforms' stage (midas_transforms.merge_overlapping_peaks, "
            "byte-exact C-parity port of MergeOverlappingPeaksAllZarr)"
        ),
    )
