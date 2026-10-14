"""Tolerances bound the whole fit, and the distortion terms can be frozen.

Walk (found on a real CCD frame set, 2026-10-12): ``bounds`` centred the
``tol*`` box on the CURRENT values and the orchestrator called it every
iteration, so a parameter pinned at its bound moved by one full tolerance per
iteration. ``tolDistortion 0.001`` over 5 iterations let the largest p reach
5e-3 (his "whole multiples of tolDistortion"), and ``tolLsd`` was not a limit
at all. The tolerances are now measured from the geometry the fit STARTED from.

Freezing: ``FitDistortion 0`` in a parameter file, ``params.freeze_distortion()``
and ``midas-autocalibrate --freeze-distortion`` all hold p0..p14 at their
starting values, for detectors whose software already corrects distortion.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from test_clipped_bin_rejection import _params, _ring_image

from midas_calibrate import CalibrationParams, autocalibrate


def _distorted_start(tol=1e-3, n_iter=4):
    """Image rendered through p2 = p4 = 0.01, i.e. 10x further than ``tol`` allows."""
    truth = _params(N=512, Lsd=400_000.0, ty=0.30, tz=-0.45, p2=0.01, p4=0.01)
    img = _ring_image(truth)
    start = _params(N=512, Lsd=400_000.0, ty=0.30, tz=-0.45)
    start.tolDistortion = tol
    start.nIterations = n_iter
    start.RemoveOutliersBetweenIters = False
    return start, img


def _max_p(params):
    return max(abs(getattr(params, f"p{i}")) for i in range(15))


def test_tolerance_bounds_the_whole_fit_not_each_iteration():
    start, img = _distorted_start(tol=1e-3, n_iter=4)
    got = autocalibrate(start, img, verbose=False).params
    # Data want p ~ 0.01. With the box re-centred every iteration the fit
    # walked to ~4e-3 here; anchored at the start it cannot leave +-1e-3.
    assert _max_p(got) <= 1e-3 + 1e-9, f"max |p| = {_max_p(got):.2e} after 4 iterations"
    assert _max_p(got) > 0.5e-3, "the data should have pushed p to its bound"


