"""The fine-phase outer loop must not stop on the strain alone.

`autocalibrate_pv` stopped when the kept-set strain changed by less than 1 %, which
watches the objective and not the parameters. On a Pilatus frame whose seed was
shifted by 100 um in Lsd the loop stopped at strain 28.62 -> 28.67 while Lsd was
still climbing ~200 um per iteration, and the run ended 23 sigma from the
unshifted run (statistics lens, 2026-10-12). With `param_tol_px` a stable strain
stops the loop only when the geometry step (a ring shift at the outermost fitted
ring) over the last two iterations is also small. A tolerance of None keeps the
old rule.
"""
from __future__ import annotations

import math
import warnings

from midas_calibrate_v2.pipelines.single_pv import (
    GeometryNotSettledWarning, IterRecord, _geometry_step_px, _outer_loop_decision)

R_OUT, PX = 1157.0, 172.0


def _rec(i, strain, lsd, by=685.4, bz=920.9, ty=0.2, tz=0.45):
    return IterRecord(iteration=i, n_fitted=1000, cost=0.0, rc=1, mean_strain_uE=strain,
                      Lsd=lsd, BC_y=by, BC_z=bz, ty=ty, tz=tz)


def test_lsd_step_is_the_ring_shift():
    a, b = _rec(1, 30.0, 657000.0), _rec(2, 30.0, 657076.0)
    assert math.isclose(_geometry_step_px(a, b, R_OUT, PX), R_OUT * 76.0 / 657076.0, rel_tol=1e-9)


def test_beam_centre_and_tilt_steps_are_on_the_same_scale():
    base = _rec(1, 30.0, 657000.0)
    assert math.isclose(_geometry_step_px(base, _rec(2, 30.0, 657000.0, by=685.4 + 0.03), R_OUT, PX), 0.03)
    t = _geometry_step_px(base, _rec(2, 30.0, 657000.0, ty=0.2 + 0.1), R_OUT, PX)
    tan2t = R_OUT * PX / 657000.0
    assert math.isclose(t, R_OUT * tan2t * math.radians(0.1), rel_tol=1e-9)


def test_stable_strain_but_moving_geometry_does_not_stop():
    """The failure that was measured: strain flat, Lsd +200 um per iteration."""
    h = [_rec(1, 28.62, 657322.0), _rec(2, 28.65, 657566.0), _rec(3, 28.67, 657771.0)]
    stop, why, step = _outer_loop_decision(h, R_OUT, PX, param_tol_px=0.05)
    assert not stop and step > 0.3, (stop, why, step)
    # ... and the legacy rule stops there, which is the bug being fixed
    stop_old, why_old, _ = _outer_loop_decision(h, R_OUT, PX, param_tol_px=None)
    assert stop_old and why_old == "strain_stable"


def test_stable_strain_and_settled_geometry_stops_as_settled():
    h = [_rec(1, 7.36, 895926.15), _rec(2, 7.30, 895925.91), _rec(3, 7.30, 895925.90)]
    stop, why, step = _outer_loop_decision(h, 1500.0, 150.0, param_tol_px=0.05)
    assert stop and why == "settled" and step < 0.05, (stop, why, step)


def test_one_small_step_after_a_big_one_is_not_convergence():
    """Pilatus Lsd steps were 49, 12, 76 um: a lone small step proves nothing, so the
    last TWO steps are both checked."""
    h = [_rec(1, 30.1, 657386.0), _rec(2, 30.12, 657434.7), _rec(3, 30.14, 657447.1)]
    stop, why, step = _outer_loop_decision(h, R_OUT, PX, param_tol_px=0.05)
    assert not stop, (why, step)       # the 49 um step (0.09 px) is still in the window


def test_strain_rising_or_floor_still_stop_but_are_not_settled():
    up = [_rec(1, 10.0, 1.0e6), _rec(2, 11.0, 1.0e6)]
    assert _outer_loop_decision(up, 1000.0, 200.0, param_tol_px=0.05)[:2] == (True, "strain_rising")
    lo = [_rec(1, 1.5, 1.0e6), _rec(2, 0.9, 1.0e6)]
    assert _outer_loop_decision(lo, 1000.0, 200.0, param_tol_px=0.05)[:2] == (True, "strain_floor")


def test_fewer_than_two_iterations_never_stops():
    assert _outer_loop_decision([_rec(1, 5.0, 1e6)], 1000.0, 200.0, param_tol_px=0.05)[0] is False


def test_warning_class_is_a_userwarning():
    assert issubclass(GeometryNotSettledWarning, UserWarning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.warn(GeometryNotSettledWarning("x"))
    assert w and w[0].category is GeometryNotSettledWarning
