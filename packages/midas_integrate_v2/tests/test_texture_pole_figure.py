"""Item 25 — Pole-figure exporter.

Rewritten 2026-08-29. The previous version of this file asserted the DEFECT:

    # The non-zero stripe should sit at α=0 (chi=0 default)
    nz = (intensity != 0).any(axis=0)
    assert nz[0]

``cake_to_pole_figure`` set the declination from ``sample_rotation_chi_deg``
and used ``hkl_R_px`` only to pick the η stripe, so with the default χ = 0
every ring — at any 2θ — landed in the α = 0 bin at the centre of the pole
figure. The test agreed with the code, which is why it survived: measured
errors of 85–89° against the correct declination, on output that goes straight
into a real POPLA ``.pol`` file.

The physics, derived rather than recalled. With ``k_i = (0,0,1)`` and
``k_f = (sin2θ cosη, sin2θ sinη, cos2θ)``::

    Q ∝ k_f − k_i = 2 sinθ · (cosθ cosη, cosθ sinη, −sinθ)

so the pole direction is ``(cosθ cosη, cosθ sinη, −sinθ)``, at declination
``90° − θ`` from the back-along-beam axis, azimuth ``η``. These tests check the
declination against that formula independently, and check that the two cases
the function cannot do honestly (no 2θ given, sample tilt) now raise instead of
guessing.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from midas_integrate_v2.texture import cake_to_pole_figure, write_popla_pol


def _synthetic_cake(ring_R=20.0, n_eta=360, n_r=64, R_max=64.0):
    eta = np.linspace(-180.0, 180.0, n_eta, endpoint=False)
    R = np.linspace(0.0, R_max, n_r)
    int2d = np.zeros((n_eta, n_r))
    int2d[:, int(np.argmin(np.abs(R - ring_R)))] = (
        100.0 + 50.0 * np.cos(np.deg2rad(eta) * 2.0))
    return int2d, eta, R


def _populated_alpha(int2d, eta, R, **kw):
    a, _b, I = cake_to_pole_figure(int2d, eta, R, **kw)
    nz = np.nonzero(I.sum(axis=0))[0]
    assert nz.size == 1, f"a single ring must fill one declination, got {nz.size}"
    return float(a[nz[0]])


# ------------------------------------------------ the declination is the physics

@pytest.mark.parametrize("two_theta", [1.0, 5.0, 12.0, 30.0, 60.0, 120.0])
def test_declination_is_ninety_minus_theta(two_theta):
    int2d, eta, R = _synthetic_cake()
    got = _populated_alpha(int2d, eta, R, hkl_R_px=20.0, capture_radius_px=1.0,
                           two_theta_deg=two_theta, output_grid=(901, 181))
    expected = 90.0 - 0.5 * two_theta
    # grid resolution is 90/901 = 0.0999 deg; allow one bin
    assert abs(got - expected) <= 0.11, f"{got} vs {expected}"


def test_declination_depends_on_the_ring_not_the_stage():
    """The specific regression: two different rings must land at two different
    declinations. Before the fix both landed at 0."""
    int2d, eta, R = _synthetic_cake(ring_R=20.0)
    a_inner = _populated_alpha(int2d, eta, R, hkl_R_px=20.0,
                               capture_radius_px=1.0, two_theta_deg=4.0,
                               output_grid=(901, 181))
    a_outer = _populated_alpha(int2d, eta, R, hkl_R_px=20.0,
                               capture_radius_px=1.0, two_theta_deg=40.0,
                               output_grid=(901, 181))
    assert a_inner != a_outer
    assert a_inner > a_outer          # smaller 2theta -> larger declination
    assert abs((a_inner - a_outer) - 18.0) < 0.25


def test_a_powder_ring_lands_near_the_rim_not_the_centre():
    """Sanity in words: at a typical powder 2θ the poles are close to the
    equator of the stereographic projection, nowhere near α = 0."""
    int2d, eta, R = _synthetic_cake()
    got = _populated_alpha(int2d, eta, R, hkl_R_px=20.0, capture_radius_px=1.0,
                           two_theta_deg=10.0, output_grid=(901, 181))
    assert got > 80.0, f"declination {got} — that is the centre, not the rim"


# ------------------------------------------------ what it refuses to guess

def test_missing_two_theta_raises_rather_than_guessing():
    int2d, eta, R = _synthetic_cake()
    with pytest.raises(ValueError, match="two_theta_deg"):
        cake_to_pole_figure(int2d, eta, R, hkl_R_px=20.0, capture_radius_px=1.0)


def test_sample_tilt_raises_rather_than_approximating():
    int2d, eta, R = _synthetic_cake()
    with pytest.raises(NotImplementedError, match="chi"):
        cake_to_pole_figure(int2d, eta, R, hkl_R_px=20.0, capture_radius_px=1.0,
                            two_theta_deg=10.0, sample_rotation_chi_deg=30.0)


def test_impossible_two_theta_raises():
    int2d, eta, R = _synthetic_cake()
    with pytest.raises(ValueError, match="declination"):
        cake_to_pole_figure(int2d, eta, R, hkl_R_px=20.0, capture_radius_px=1.0,
                            two_theta_deg=200.0)


# ------------------------------------------------ phi, shapes, and the writer

def test_phi_is_a_pure_azimuthal_rotation():
    """φ about the beam only relabels β: total intensity is conserved and the
    declination does not move."""
    int2d, eta, R = _synthetic_cake()
    kw = dict(hkl_R_px=20.0, capture_radius_px=1.0, two_theta_deg=10.0,
              output_grid=(91, 181))
    a0, _b0, I0 = cake_to_pole_figure(int2d, eta, R, **kw)
    a1, _b1, I1 = cake_to_pole_figure(int2d, eta, R,
                                      sample_rotation_phi_deg=90.0, **kw)
    # Not exact: the η stripe is RESAMPLED onto the β grid with ``np.interp``,
    # which samples rather than integrates, so a rotation that misaligns the
    # two grids changes the sum at the 1e-5 level (measured 3.1e-5 for 360 η
    # bins onto 181 β bins). Fine for a pole figure, which is normalised
    # downstream — but it is a resampling, not a conservative rebin, and a
    # much coarser β grid would lose more.
    assert np.allclose(I0.sum(), I1.sum(), rtol=1e-3)
    assert np.array_equal(np.nonzero(I0.sum(axis=0))[0],
                          np.nonzero(I1.sum(axis=0))[0])
    assert not np.allclose(I0, I1), "phi=90 should move the pattern in beta"


def test_cake_to_pole_basic_shapes():
    int2d, eta, R = _synthetic_cake()
    a, b, intensity = cake_to_pole_figure(
        int2d, eta, R, hkl_R_px=20.0, capture_radius_px=1.0,
        two_theta_deg=10.0, output_grid=(91, 181))
    assert a.shape == (91,)
    assert b.shape == (181,)
    assert intensity.shape == (181, 91)
    assert intensity.sum() > 0


def test_cake_to_pole_no_ring_raises():
    int2d, eta, R = _synthetic_cake()
    with pytest.raises(ValueError):
        cake_to_pole_figure(int2d, eta, R, hkl_R_px=10000.0,
                            capture_radius_px=0.5, two_theta_deg=10.0)


def test_popla_writer(tmp_path: Path):
    int2d, eta, R = _synthetic_cake()
    a, b, intensity = cake_to_pole_figure(
        int2d, eta, R, hkl_R_px=20.0, capture_radius_px=1.0,
        two_theta_deg=10.0, output_grid=(91, 181))
    out = tmp_path / "ring111.pol"
    write_popla_pol(out, a, b, intensity, hkl=(1, 1, 1))
    text = out.read_text()
    assert "hkl=1 1 1" in text
    data_rows = [ln for ln in text.splitlines()
                 if not ln.startswith("#") and ln.strip()]
    assert len(data_rows) == 181
