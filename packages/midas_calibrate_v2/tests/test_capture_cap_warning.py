"""The capture phase must say so when it ends on its iteration cap with the rings still moving.

On the real 14-degree Varex frame (distortion off, default n_capture_max=10) the blind start
stopped 2.0 degrees short in tz at 793 microstrain and the only trace was a verbose line.
"""
from midas_calibrate_v2.pipelines.single_pv import (
    CaptureNotSettledWarning, _capture_cap_message)


def test_cap_reached_with_rings_still_moving_warns():
    m = _capture_cap_message(shift_px=3.2, capture_shift_px=1.0, n_done=10, n_capture_max=10)
    assert m and "n_capture_max=10" in m and "3.20 px" in m


def test_rings_stopped_moving_is_silent_even_at_the_cap():
    assert _capture_cap_message(0.4, 1.0, 10, 10) is None


def test_stopping_before_the_cap_is_silent():
    assert _capture_cap_message(0.4, 1.0, 3, 10) is None


def test_warning_is_a_userwarning():
    assert issubclass(CaptureNotSettledWarning, UserWarning)
