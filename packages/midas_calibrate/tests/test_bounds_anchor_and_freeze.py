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


def test_frozen_distortion_stays_at_the_starting_values():
    start, img = _distorted_start(n_iter=3)
    start.freeze_distortion()
    lsd0 = start.Lsd
    got = autocalibrate(start, img, verbose=False).params
    assert _max_p(got) < 1e-7
    # The rest of the geometry is still refined. (The image carries distortion a
    # frozen model cannot absorb, so Lsd takes it up and moves -- which is the
    # Lsd/distortion degeneracy, and is exactly why this test uses that image.)
    assert abs(got.Lsd - lsd0) > 1.0


def _roundtrip(p, tmp_path):
    f = tmp_path / "p.txt"
    f.write_text(p.to_text())
    return f


def test_fit_distortion_key_in_a_parameter_file(tmp_path):
    p = _params(N=512)
    base = _roundtrip(p, tmp_path).read_text()

    def parse(extra):
        f = tmp_path / "q.txt"
        f.write_text(base + extra)
        return CalibrationParams.from_file(f)

    assert all(parse("").Refine[f"p{i}"] for i in range(15))          # default: refined
    assert all(parse("FitDistortion 1\n").Refine[f"p{i}"] for i in range(15))
    q = parse("FitDistortion 0\n")
    assert not any(q.Refine[f"p{i}"] for i in range(15))
    assert q.Refine["Lsd"] and q.Refine["BC"] and q.Refine["ty"] and q.Refine["tz"]


def test_freeze_distortion_method_round_trips_through_the_writer(tmp_path):
    p = _params(N=512)
    assert "FitDistortion" not in p.to_text()                          # free: nothing emitted
    assert p.freeze_distortion() is p
    assert "FitDistortion 0" in p.to_text()
    back = CalibrationParams.from_file(_roundtrip(p, tmp_path))
    assert not any(back.Refine[f"p{i}"] for i in range(15))


def test_cli_freeze_distortion_flag(tmp_path, monkeypatch):
    from midas_calibrate import cli, orchestrator

    pf = _roundtrip(_params(N=512), tmp_path)
    np.save(tmp_path / "img.npy", np.zeros((512, 512)))
    seen = {}

    def fake_autocalibrate(params, image, **kw):
        seen["refine"] = dict(params.Refine)
        return types.SimpleNamespace(
            params=types.SimpleNamespace(write=lambda path: None),
            history=[types.SimpleNamespace(mean_strain_uE=0.0)])

    monkeypatch.setattr(orchestrator, "autocalibrate", fake_autocalibrate)
    cli.autocalibrate_main([str(pf), "--image", str(tmp_path / "img.npy"), "--quiet"])
    assert all(seen["refine"][f"p{i}"] for i in range(15))             # off by default
    cli.autocalibrate_main([str(pf), "--image", str(tmp_path / "img.npy"), "--quiet",
                            "--freeze-distortion"])
    assert not any(seen["refine"][f"p{i}"] for i in range(15))
