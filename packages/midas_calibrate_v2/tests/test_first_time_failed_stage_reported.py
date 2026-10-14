"""A stage of a first-time attempt that raises used to be dropped without a message.

On the Varex frame with Lsd 67 % too large one of seven attempts died with an IndexError (an empty
fit history); it appeared only in a log nobody printed (si_rerun/C).
"""
from midas_calibrate_v2.pipelines.first_time import _failed_stage_messages


def test_failed_stage_lines_are_picked_out():
    log = ["stage1 Lsd->123 um  strain=5.0 uE", "stage2a FAILED: IndexError: list index out of range", "stage2b strain=4.0"]
    assert _failed_stage_messages(log) == ["stage2a FAILED: IndexError: list index out of range"]


def test_clean_log_has_none():
    assert _failed_stage_messages(["stage1 Lsd->1 um strain=1 uE"]) == []
