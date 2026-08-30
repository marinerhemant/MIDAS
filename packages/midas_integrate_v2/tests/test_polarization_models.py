"""Polarization correction: the two models, and what separates them.

``polarization_factor`` ships v1's form ``1 - PF sin^2(2th) cos^2(eta)``. That
is the standard result only for a FULLY polarized beam; below PF = 1 it scales
the correction rather than mixing the two orthogonal polarization states, which
is not what a partially polarized beam does.

These tests pin both models, pin that they coincide at PF = 1, and pin the
physical property that separates them: for an unpolarized beam the correction
cannot depend on azimuth.
"""

from __future__ import annotations

import math

import pytest
import torch

from midas_integrate_v2.corrections.intensity import polarization_factor

DT = torch.float64
LSD = torch.tensor(1_000_000.0, dtype=DT)
PX = torch.tensor(200.0, dtype=DT)


def _R_of(two_theta_deg):
    return torch.tensor(math.tan(math.radians(two_theta_deg)) * float(LSD) / float(PX),
                        dtype=DT)


def P(two_theta_deg, eta_deg, pf, model="midas", plane=0.0):
    return float(polarization_factor(
        _R_of(two_theta_deg), torch.tensor(float(eta_deg), dtype=DT),
        Lsd=LSD, px=PX,
        pol_fraction=torch.tensor(float(pf), dtype=DT),
        pol_plane_eta_deg=torch.tensor(float(plane), dtype=DT),
        model=model))


@pytest.mark.parametrize("model", ["midas", "mixture"])
def test_unity_at_zero_two_theta(model):
    for pf in (0.0, 0.5, 0.95, 1.0):
        assert abs(P(0.0, 0.0, pf, model) - 1.0) < 1e-12
        assert abs(P(0.0, 90.0, pf, model) - 1.0) < 1e-12


@pytest.mark.parametrize("two_theta", [10.0, 30.0, 60.0, 80.0])
@pytest.mark.parametrize("eta", [0.0, 37.0, 90.0, 143.0, -60.0])
def test_the_two_models_agree_exactly_when_fully_polarized(two_theta, eta):
    """At PF = 1 the MIDAS form IS the standard form — algebraically, so the
    agreement should be to machine precision, not merely close."""
    a = P(two_theta, eta, 1.0, "midas")
    b = P(two_theta, eta, 1.0, "mixture")
    assert abs(a - b) < 1e-12, f"{a} vs {b}"


def test_mixture_is_azimuth_independent_for_an_unpolarized_beam():
    """The physical property that separates the two models.

    An unpolarized beam has no preferred azimuth, so its correction cannot
    have one either. ``mixture`` at PF=0 satisfies this; ``midas`` does not,
    at any PF it can express.
    """
    for two_theta in (10.0, 30.0, 60.0):
        vals = [P(two_theta, e, 0.0, "mixture")
                for e in (0.0, 45.0, 90.0, 135.0, 180.0)]
        assert max(vals) - min(vals) < 1e-12
        # and it equals the textbook (1 + cos^2 2th)/2
        tt = math.radians(two_theta)
        assert abs(vals[0] - 0.5 * (1 + math.cos(tt) ** 2)) < 1e-12


def test_midas_model_is_not_azimuth_independent_at_half_fraction():
    """Documents the known limitation rather than asserting it away."""
    vals = [P(60.0, e, 0.5, "midas") for e in (0.0, 45.0, 90.0)]
    assert max(vals) - min(vals) > 0.3


@pytest.mark.parametrize("model", ["midas", "mixture"])
def test_bounded_between_zero_and_one(model):
    for pf in (0.0, 0.5, 1.0):
        for tt in (0.0, 15.0, 45.0, 75.0, 89.0):
            for e in range(-180, 181, 30):
                v = P(tt, e, pf, model)
                assert -1e-12 <= v <= 1.0 + 1e-12, (model, pf, tt, e, v)


def test_fully_polarized_vanishes_in_plane_and_is_unity_out_of_plane():
    for model in ("midas", "mixture"):
        assert P(89.999, 0.0, 1.0, model) < 1e-8
        assert abs(P(89.999, 90.0, 1.0, model) - 1.0) < 1e-8


def test_polarization_plane_rotates_the_pattern():
    """A plane at 90 deg must move the minimum to eta = 90."""
    for model in ("midas", "mixture"):
        assert P(89.999, 90.0, 1.0, model, plane=90.0) < 1e-8
        assert abs(P(89.999, 0.0, 1.0, model, plane=90.0) - 1.0) < 1e-8


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="model must be"):
        P(30.0, 0.0, 0.99, "kahn")


def test_default_is_the_mixture_model():
    """Switched 2026-08-29. The old form stays reachable so historical output
    can be reproduced exactly."""
    a = float(polarization_factor(
        _R_of(45.0), torch.tensor(30.0, dtype=DT), Lsd=LSD, px=PX,
        pol_fraction=torch.tensor(0.99, dtype=DT),
        pol_plane_eta_deg=torch.tensor(0.0, dtype=DT)))
    assert abs(a - P(45.0, 30.0, 0.99, "mixture")) < 1e-15
    assert abs(a - P(45.0, 30.0, 0.99, "midas")) > 1e-4   # genuinely different


def test_v1_and_v2_use_the_same_polarization_model():
    """The parity that the four implementations must keep.

    v1 has three copies of this (detector_mapper plus two numba kernels) and
    v2 has one. If they drift apart the distortion/panel/residual parity tests
    fail -- but only for specs that switch polarization on, so it is worth
    pinning the formula itself here.
    """
    import math as _m
    for pf in (1.0, 0.99, 0.5, 0.0):
        for tt in (5.0, 30.0, 60.0):
            for e in (0.0, 45.0, 90.0):
                tr, er = _m.radians(tt), _m.radians(e)
                v1 = 0.5 * (1.0 + _m.cos(tr) ** 2
                            - pf * _m.cos(2.0 * er) * _m.sin(tr) ** 2)
                assert abs(v1 - P(tt, e, pf, "mixture")) < 1e-12
