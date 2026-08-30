"""The polarization plane must sit on the HORIZONTAL, and MIDAS η is vertical-zero.

``PolarizationPlaneEtaDeg`` defaulted to 0 until 2026-08-29, with a docstring
claiming ``0 = horizontal at η = 0 (pyFAI convention)``. That premise is false
and the bug follows directly from it: pyFAI measures its azimuth ``chi`` from
the horizontal detector axis, MIDAS measures η from the **vertical**
(``atan2(-y, z)``). The number was carried over without the axis, so the
correction was applied a quarter turn from the beam's actual polarization —
the right functional form on the wrong axis, which *adds* the azimuthal
modulation it exists to remove (33 % per-pixel error in P at 2θ = 30°).

These tests pin each link of the chain separately, so a future change that
breaks one of them says which one:

  1. MIDAS η = 0 is vertical and η = ±90 is horizontal;
  2. the polarization factor's node sits ON ``pol_plane_eta_deg``;
  3. therefore the default must put the node on the horizontal;
  4. the two integrator packages agree on the default;
  5. an independent check against pyFAI, when it is installed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

DT = torch.float64


# ------------------------------------------------- 1. what η = 0 means

def test_midas_eta_zero_is_vertical_and_ninety_is_horizontal():
    """The premise the whole bug rests on. If this ever flips, the default
    must flip with it."""
    from midas_integrate.geometry import calc_eta_angle

    # +z is "up" on the detector, +y is the other in-plane axis.
    assert calc_eta_angle(y=0.0, z=1.0) == pytest.approx(0.0, abs=1e-9)
    assert abs(calc_eta_angle(y=0.0, z=-1.0)) == pytest.approx(180.0, abs=1e-9)
    # η = ±90 lies on the y axis — the horizontal one.
    assert abs(calc_eta_angle(y=1.0, z=0.0)) == pytest.approx(90.0, abs=1e-9)
    assert abs(calc_eta_angle(y=-1.0, z=0.0)) == pytest.approx(90.0, abs=1e-9)


def test_the_eta_the_kernels_use_is_the_same_convention():
    """The correction consumes the *bin* η built inside the mapper, not
    ``calc_eta_angle``. They must be the same function."""
    from midas_integrate.geometry import calc_eta_angle
    from midas_integrate._mapper_numba import _calc_eta as numba_eta

    for y, z in ((0.0, 1.0), (1.0, 0.0), (-1.0, 0.0), (0.7, -0.7), (-0.3, 0.9)):
        assert numba_eta(y, z) == pytest.approx(calc_eta_angle(y, z), abs=1e-12)


# ------------------------------------------- 2. where the node actually is

def _P(eta_deg, two_theta_deg, plane_deg, pf=1.0, model="mixture"):
    from midas_integrate_v2.corrections.intensity import polarization_factor

    Lsd, px = torch.tensor(1e6, dtype=DT), torch.tensor(200.0, dtype=DT)
    R = 1e6 * math.tan(math.radians(two_theta_deg)) / 200.0
    eta = np.atleast_1d(np.asarray(eta_deg, dtype=float))
    return polarization_factor(
        torch.tensor(np.full(eta.shape, R), dtype=DT),
        torch.tensor(eta, dtype=DT),
        Lsd=Lsd, px=px,
        pol_fraction=torch.tensor(pf, dtype=DT),
        pol_plane_eta_deg=torch.tensor(float(plane_deg), dtype=DT),
        model=model).numpy()


@pytest.mark.parametrize("plane", [0.0, 30.0, 90.0, 135.0])
def test_the_node_sits_on_the_polarization_plane(plane):
    """Scattering is suppressed ALONG the E-vector, so P is minimal at
    η = plane and maximal a quarter turn away."""
    eta = np.linspace(-180.0, 180.0, 3601)
    P = _P(eta, 90.0, plane, pf=1.0)          # 2θ = 90° makes the node deepest
    node = eta[int(np.argmin(P))]
    peak = eta[int(np.argmax(P))]
    # node is at plane, modulo 180 (the factor has period 180 in η)
    assert min(abs((node - plane) % 180.0), 180.0 - abs((node - plane) % 180.0)) < 0.5
    assert min(abs((peak - plane - 90.0) % 180.0),
               180.0 - abs((peak - plane - 90.0) % 180.0)) < 0.5
    assert P.min() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------- 3./4. the defaults, in both packages

def test_the_default_plane_is_horizontal():
    from midas_integrate_v2.corrections.intensity import (
        POL_PLANE_HORIZONTAL_ETA_DEG)

    assert POL_PLANE_HORIZONTAL_ETA_DEG == 90.0, (
        "MIDAS eta is measured from the VERTICAL, so a horizontally polarized "
        "beam — every storage ring — has its plane at eta = 90, not 0")

    # and the node it produces really is on the horizontal axis
    P_vertical = _P([0.0], 90.0, POL_PLANE_HORIZONTAL_ETA_DEG, pf=1.0)[0]
    P_horizontal = _P([90.0], 90.0, POL_PLANE_HORIZONTAL_ETA_DEG, pf=1.0)[0]
    assert P_horizontal < 1e-9, "the node must be on the horizontal (eta=90)"
    assert P_vertical == pytest.approx(1.0, abs=1e-9), (
        "the vertical (eta=0) must be unattenuated for a horizontal beam")


def test_both_integrator_packages_default_to_the_same_plane():
    from midas_integrate.params import IntegrationParams
    from midas_integrate_v2.spec import IntegrationSpec
    from midas_integrate_v2.corrections.intensity import (
        PolarizationCorrection, POL_PLANE_HORIZONTAL_ETA_DEG)

    assert IntegrationParams().PolarizationPlaneEtaDeg == POL_PLANE_HORIZONTAL_ETA_DEG
    assert IntegrationSpec().PolarizationPlaneEtaDeg == POL_PLANE_HORIZONTAL_ETA_DEG
    assert float(PolarizationCorrection().pol_plane_eta_deg) == \
        POL_PLANE_HORIZONTAL_ETA_DEG


def test_the_mapper_fallback_does_not_reintroduce_the_old_default():
    """``detector_mapper`` reads the angle with ``getattr(..., default)``. A
    params object without the attribute must not silently get the vertical
    plane back."""
    import inspect
    from midas_integrate import detector_mapper

    src = inspect.getsource(detector_mapper)
    assert 'getattr(params, "PolarizationPlaneEtaDeg", 0.0)' not in src, (
        "the getattr fallback still defaults to 0.0 (vertical)")
    assert 'getattr(params, "PolarizationPlaneEtaDeg", 90.0)' in src


# -------------------------------------------------- 5. independent: pyFAI

def test_matches_pyfai_for_a_horizontally_polarized_beam():
    """pyFAI is the independent authority on the physics here, and the package
    the old docstring claimed to follow.

    pyFAI's ``chi`` is measured from the horizontal detector axis and MIDAS's
    η from the vertical, so the SAME physical beam is ``axis_offset = 0`` there
    and ``plane = 90`` here. Compare the factor as a function of physical
    azimuth measured from the horizontal.
    """
    pyFAI = pytest.importorskip("pyFAI")
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.detectors import Detector

    N, ps = 401, 100e-6
    det = Detector(pixel1=ps, pixel2=ps, max_shape=(N, N))
    ai = AzimuthalIntegrator(dist=0.001, poni1=(N // 2) * ps,
                             poni2=(N // 2) * ps, detector=det,
                             wavelength=1e-10)
    chi = np.degrees(ai.chiArray((N, N)))
    tth = np.degrees(ai.twoThetaArray((N, N)))
    P_ref = ai.polarization(shape=(N, N), factor=1.0, axis_offset=0.0)

    band = np.abs(tth - 70.0) < 2.0
    assert band.sum() > 500

    # MIDAS eta = 90 - chi puts both on the same physical azimuth (both are
    # in-plane angles; the sign is irrelevant because the factor is even in
    # (eta - plane)).
    eta_midas = 90.0 - chi[band]
    P_mine = _P(eta_midas, 0.0, 90.0, pf=1.0)   # 2theta filled in below
    # recompute with the true per-pixel 2theta rather than a single value
    from midas_integrate_v2.corrections.intensity import polarization_factor
    Lsd, px = torch.tensor(1e6, dtype=DT), torch.tensor(200.0, dtype=DT)
    R = 1e6 * np.tan(np.radians(tth[band])) / 200.0
    P_mine = polarization_factor(
        torch.tensor(R, dtype=DT), torch.tensor(eta_midas, dtype=DT),
        Lsd=Lsd, px=px, pol_fraction=torch.tensor(1.0, dtype=DT),
        pol_plane_eta_deg=torch.tensor(90.0, dtype=DT),
        model="mixture").numpy()

    # pyFAI returns its polarization array in FLOAT32 (eps = 1.19e-7), so
    # agreement to a fraction of one ULP is exact agreement, not an
    # approximation. Measured: 2.96e-08, a quarter of one ULP.
    assert np.abs(P_mine - P_ref[band]).max() < 2e-7, (
        f"max |dP| = {np.abs(P_mine - P_ref[band]).max():.3e}")

    # and the OLD default disagrees with pyFAI, so the test discriminates
    P_old = polarization_factor(
        torch.tensor(R, dtype=DT), torch.tensor(eta_midas, dtype=DT),
        Lsd=Lsd, px=px, pol_fraction=torch.tensor(1.0, dtype=DT),
        pol_plane_eta_deg=torch.tensor(0.0, dtype=DT),
        model="mixture").numpy()
    assert np.abs(P_old - P_ref[band]).max() > 0.5


# ------------------------------------------------- who is actually affected

def test_a_full_ring_one_d_pattern_is_unchanged_by_the_plane():
    """The scoping claim in the docstring, kept honest.

    Over a full ring the two planes give the identical mean, because a quarter
    turn only relabels which azimuth gets which factor. So 1-D patterns did not
    move; η-resolved products did.
    """
    eta = np.linspace(-180.0, 180.0, 4001)[:-1]
    for tt in (10.0, 30.0, 60.0):
        a = np.mean(1.0 / _P(eta, tt, 0.0, pf=0.99))
        b = np.mean(1.0 / _P(eta, tt, 90.0, pf=0.99))
        assert abs(a - b) / b < 1e-6, f"2theta={tt}: {a} vs {b}"


def test_an_eta_resolved_product_carries_the_full_error():
    """The other half: per-η the two planes differ enormously, which is what
    made this worth fixing."""
    eta = np.linspace(-180.0, 180.0, 721)
    worst = {}
    for tt in (10.0, 20.0, 30.0):
        P0 = _P(eta, tt, 0.0, pf=0.99)
        P9 = _P(eta, tt, 90.0, pf=0.99)
        worst[tt] = float(np.abs(1 / P0 - 1 / P9).max() / (1 / P9).min())
    assert worst[10.0] > 0.02
    assert worst[20.0] > 0.10
    assert worst[30.0] > 0.25
