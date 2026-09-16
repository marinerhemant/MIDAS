"""The capture-phase tilt box slides with the fit instead of staying pinned to the seed.

``compat.from_v1.spec_from_v1_params`` sets each tilt bound to ``init ± tolTilts`` -- centred on the
SEED. That caps how far capture can travel however wide its window is: measured on a 14 deg Varex
frame, a tz 0 seed with the default ``tolTilts`` of 3 rails at 3.0 and ends 11 deg away, while the
same run with ``tolTilts`` widened by hand to 16 reaches the rings.

The assertion that matters here is the transform one. ``Parameter.__post_init__`` builds the logit
transform only when it is None, so moving ``bounds`` without rebuilding it leaves the LM mapping
into the OLD box -- a silently wrong answer, not an error. ``four_stage`` rebuilds it for the same
reason.
"""
from __future__ import annotations

import inspect

import torch

from midas_calibrate_v2.parameters.parameter import Parameter
from midas_calibrate_v2.parameters.spec import CalibrationSpec
from midas_calibrate_v2.parameters.transforms import Logit
from midas_calibrate_v2.pipelines.single_pv import _recentre_tilt_bounds, autocalibrate_pv

TOL = 3.0


def _spec(ty=0.0, tz=0.0, *, refined=True, bounds=True):
    s = CalibrationSpec()
    for n, v in (("ty", ty), ("tz", tz)):
        s.add(Parameter(name=n, init=v, refined=refined,
                        bounds=((v - TOL, v + TOL) if bounds else None)))
    return s


def _unpacked(ty, tz):
    return {"ty": torch.tensor(ty, dtype=torch.float64),
            "tz": torch.tensor(tz, dtype=torch.float64)}


def test_the_box_slides_onto_the_fit_and_keeps_its_width():
    s = _spec()
    centres = _recentre_tilt_bounds(s, _unpacked(1.5, 2.9))
    assert centres == {"ty": 1.5, "tz": 2.9}
    assert s.parameters["tz"].bounds == (2.9 - TOL, 2.9 + TOL)
    assert s.parameters["ty"].bounds == (1.5 - TOL, 1.5 + TOL)
    for n in ("ty", "tz"):
        lo, hi = s.parameters[n].bounds
        assert (hi - lo) == 2 * TOL, n


def test_the_transform_is_rebuilt_on_the_new_box():
    """The stale-transform bug: bounds moved, transform still mapping into the old box."""
    s = _spec()
    old = s.parameters["tz"].transform
    _recentre_tilt_bounds(s, _unpacked(0.0, 2.9))
    p = s.parameters["tz"]
    assert isinstance(p.transform, Logit)
    assert p.transform is not old
    assert (p.transform.lo, p.transform.hi) == p.bounds
    # and it really maps into the new box: the midpoint of the box is u = 0
    assert float(p.transform.inverse(torch.zeros((), dtype=torch.float64))) == 2.9


def test_this_is_what_lifts_the_seed_cap():
    """A tz 0 seed with tolTilts 3 rails at 3.0. After re-centring the box reaches 6.0,
    so the next capture step can keep going -- the mechanism the A6b failure needs."""
    s = _spec(tz=0.0)
    assert s.parameters["tz"].bounds == (-TOL, TOL)
    _recentre_tilt_bounds(s, _unpacked(0.0, 3.0))          # railed at its bound
    assert s.parameters["tz"].bounds == (0.0, 6.0)


def test_nothing_moves_when_the_fit_has_not_moved():
    s = _spec(ty=1.0, tz=2.0)
    assert _recentre_tilt_bounds(s, _unpacked(1.0, 2.0)) is None
    assert s.parameters["tz"].bounds == (2.0 - TOL, 2.0 + TOL)


def test_fixed_and_unbounded_tilts_are_left_alone():
    fixed = _spec(refined=False)
    assert _recentre_tilt_bounds(fixed, _unpacked(1.5, 2.9)) is None
    assert fixed.parameters["tz"].bounds == (-TOL, TOL)

    free = _spec(bounds=False)
    assert _recentre_tilt_bounds(free, _unpacked(1.5, 2.9)) is None
    assert free.parameters["tz"].bounds is None


def test_the_knob_is_exposed_and_defaults_on():
    assert inspect.signature(autocalibrate_pv).parameters["recentre_capture_bounds"].default is True
