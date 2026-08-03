"""Extinction, refraction, and the kinematical-validity boundary.

These are scalar corrections *to* a kinematical calculation, not dynamical
diffraction theory, so the tests are about limits, scalings and units -- the
things a wrong implementation gets wrong -- plus agreement between the numpy and
torch paths.
"""
import math

import pytest

from midas_hkls import (
    CLASSICAL_ELECTRON_RADIUS_A,
    extinction_length_um,
    kinematical_path_limit_um,
    primary_extinction_factor,
    refraction_shift_deg,
)
from midas_hkls.crystal import Atom, Crystal
from midas_hkls.lattice import Lattice
from midas_hkls.space_group import SpaceGroup

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

BASE = dict(structure_factor_mag=100.0, unit_cell_volume_A3=48.0,
            wavelength_A=0.172979, theta_deg=2.36)


def _lam(**kw):
    return float(extinction_length_um(**{**BASE, **kw}))


# ---------------------------------------------------------------------------
# extinction length
# ---------------------------------------------------------------------------
def test_extinction_length_scalings():
    """Lambda ~ V / (lambda |F|): one dependence at a time."""
    base = _lam()
    assert abs(_lam(unit_cell_volume_A3=96.0) / base - 2.0) < 1e-12
    assert abs(_lam(wavelength_A=2 * BASE["wavelength_A"]) / base - 0.5) < 1e-12
    assert abs(_lam(structure_factor_mag=200.0) / base - 0.5) < 1e-12


def test_extinction_length_matches_the_closed_form():
    want = (math.pi * 48.0 * math.cos(math.radians(2.36))
            / (CLASSICAL_ELECTRON_RADIUS_A * 0.172979 * 100.0)) / 1e4
    assert abs(_lam() - want) / want < 1e-15


def test_extinction_length_is_a_plausible_number_of_micrometers():
    """Unit check: a strong reflection in a light metal is tens of microns.

    Catches an Angstrom/micrometer slip, which is a factor of 1e4.
    """
    assert 1.0 < _lam() < 1000.0


def test_polarization_factor_enters_inversely():
    assert abs(_lam(polarization_factor=0.5) / _lam() - 2.0) < 1e-12


def test_forbidden_reflection_has_no_extinction_length():
    with pytest.raises(ValueError, match="forbidden"):
        extinction_length_um(0.0, unit_cell_volume_A3=48.0, wavelength_A=0.17,
                             theta_deg=2.4)


# ---------------------------------------------------------------------------
# the factor
# ---------------------------------------------------------------------------
def test_factor_is_one_in_the_kinematical_limit():
    assert abs(primary_extinction_factor(0.0, 50.0) - 1.0) < 1e-15


def test_factor_matches_tanh_away_from_zero():
    for t in (5.0, 25.0, 100.0, 400.0):
        x = t / 50.0
        assert abs(primary_extinction_factor(t, 50.0) - math.tanh(x) / x) < 1e-14


def test_factor_is_monotonic_and_bounded():
    ys = [primary_extinction_factor(t, 50.0) for t in (1.0, 10.0, 50.0, 200.0)]
    assert ys == sorted(ys, reverse=True)
    assert all(0.0 < y <= 1.0 for y in ys)


def test_factor_saturates_as_one_over_x():
    """Dynamical limit: a thick perfect crystal cannot reflect more."""
    assert abs(primary_extinction_factor(1000.0, 10.0) - 10.0 / 1000.0) < 1e-9


def test_series_branch_is_continuous_with_the_tanh_branch():
    L, t = 50.0, 50.0 * 1e-6 * 1.0000001
    x = t / L
    assert abs(primary_extinction_factor(t, L) - math.tanh(x) / x) < 1e-12


# ---------------------------------------------------------------------------
# validity boundary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tol", (0.01, 0.05, 0.1, 0.2, 0.5))
def test_path_limit_costs_exactly_the_stated_tolerance(tol):
    L = 40.0
    y = primary_extinction_factor(float(kinematical_path_limit_um(L, tolerance=tol)), L)
    assert abs((1.0 - y) - tol) < 1e-12


def test_ten_percent_limit_is_a_bit_over_half_an_extinction_length():
    """x = 0.5838. The small-x expansion would say 0.5477 -- 6% low, and 20% low
    by tolerance = 0.2, which is why this is solved rather than expanded."""
    assert abs(float(kinematical_path_limit_um(40.0, tolerance=0.1)) / 40.0
               - 0.583811) < 1e-5


def test_bad_tolerance_rejected():
    for tol in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            kinematical_path_limit_um(40.0, tolerance=tol)


# ---------------------------------------------------------------------------
# numpy / torch parity -- torch is optional here, so both paths must agree
# ---------------------------------------------------------------------------
def test_torch_and_python_paths_agree_for_extinction_length():
    a = _lam()
    b = float(extinction_length_um(torch.tensor(100.0, dtype=torch.float64),
                                   unit_cell_volume_A3=48.0,
                                   wavelength_A=0.172979, theta_deg=2.36))
    assert abs(a - b) / a < 1e-15


def test_torch_and_python_paths_agree_for_the_factor():
    for t in (0.0, 5.0, 120.0):
        a = primary_extinction_factor(t, 40.0)
        b = float(primary_extinction_factor(torch.tensor(t, dtype=torch.float64), 40.0))
        assert abs(a - b) < 1e-14


def test_python_scalars_do_not_silently_become_float32():
    """torch.as_tensor(<python float>) is float32; that costs ~1e-8 relative."""
    out = extinction_length_um(torch.tensor(100.0, dtype=torch.float64),
                               unit_cell_volume_A3=48.0, wavelength_A=0.172979,
                               theta_deg=2.36)
    assert out.dtype == torch.float64


def test_gradcheck_factor_and_length():
    t = torch.tensor([5.0, 25.0, 120.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: primary_extinction_factor(x, 40.0), (t,))
    F = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda f: extinction_length_um(f, unit_cell_volume_A3=48.0,
                                       wavelength_A=0.172979, theta_deg=2.36), (F,))


def test_gradient_is_finite_at_zero_path():
    t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    primary_extinction_factor(t, 40.0).backward()
    assert torch.isfinite(t.grad).all()


# ---------------------------------------------------------------------------
# chi0 and refraction
# ---------------------------------------------------------------------------
def _copper():
    return Crystal(lattice=Lattice.for_system("cubic", a=3.615),
                   space_group=SpaceGroup.from_number(225),
                   atoms=[Atom("Cu", (0.0, 0.0, 0.0), B_iso=0.5)])


def test_chi0_has_the_right_order_of_magnitude():
    """For hard X-rays chi0 is ~1e-5 to 1e-6. Wrong by 1e4 if V or r_e slips."""
    from midas_hkls import susceptibility_chi0
    chi0 = susceptibility_chi0(_copper().to_torch(), 1.0)
    mag = float(abs(chi0))
    assert 1e-7 < mag < 1e-3
    # chi0 is negative-real for X-rays (electrons refract the "wrong" way)
    assert float(chi0.real) < 0.0


def test_chi0_scales_as_wavelength_squared():
    from midas_hkls import susceptibility_chi0
    c = _copper().to_torch()
    a = float(abs(susceptibility_chi0(c, 1.0, anomalous=False)))
    b = float(abs(susceptibility_chi0(c, 2.0, anomalous=False)))
    assert abs(b / a - 4.0) < 0.02


def test_refraction_shift_is_millidegrees():
    """Real DFXM/TT cases sit at 0.4-3.6 mdeg."""
    shift = refraction_shift_deg(2.2e-5, 20.0)
    assert 1e-4 < shift < 1e-2
    assert abs(shift - math.degrees(2.2e-5 / math.sin(math.radians(20.0)))) < 1e-15


def test_refraction_shift_grows_at_small_two_theta():
    """1/sin(2 theta): the correction is worst for low-angle reflections."""
    assert refraction_shift_deg(2.2e-5, 5.0) > refraction_shift_deg(2.2e-5, 40.0)


def test_refraction_shift_accepts_a_complex_chi0():
    """chi0 is complex with anomalous scattering; the shift uses its modulus."""
    a = refraction_shift_deg(complex(-2.2e-5, 1e-7), 20.0)
    b = refraction_shift_deg(2.2e-5, 20.0)
    assert abs(a - b) / b < 1e-3
