"""Phase 1 tests: primary extinction and the direct-beam channel.

This module exists to make the kinematical assumption *checkable* rather than
tacit, so the tests are mostly about limits and scalings: does the factor go to 1
where kinematical theory is valid, does it saturate where dynamical theory takes
over, and does the stated validity boundary mean what it says.
"""
import math

import pytest
import torch

from midas_dct_tt import (
    direct_beam_transmission,
    extinction_length_um,
    kinematical_path_limit_um,
    primary_extinction_factor,
)
import midas_hkls.extinction as hkls_ext

DT = torch.float64


def _lambda_um(**kw):
    base = dict(structure_factor_mag=100.0, unit_cell_volume_A3=48.0,
                wavelength_A=0.172979, theta_deg=2.36)
    base.update(kw)
    return float(extinction_length_um(**base))


# ---------------------------------------------------------------------------
# extinction length
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_extinction_length_scalings():
    """Lambda ~ V / (lambda |F|): each dependence checked one at a time."""
    base = _lambda_um()
    assert abs(_lambda_um(unit_cell_volume_A3=96.0) / base - 2.0) < 1e-12
    assert abs(_lambda_um(wavelength_A=2 * 0.172979) / base - 0.5) < 1e-12
    assert abs(_lambda_um(structure_factor_mag=200.0) / base - 0.5) < 1e-12


@pytest.mark.unit
def test_extinction_length_is_micrometers_of_a_plausible_size():
    """Sanity on units: a strong reflection in a light metal is tens of microns.

    A three-order-of-magnitude unit slip (Angstrom vs micrometer) is the failure
    this catches; it is not a claim about a specific material.
    """
    assert 1.0 < _lambda_um() < 1000.0


@pytest.mark.unit
def test_forbidden_reflection_has_no_extinction_length():
    with pytest.raises(ValueError, match="forbidden"):
        extinction_length_um(0.0, unit_cell_volume_A3=48.0,
                             wavelength_A=0.17, theta_deg=2.4)


# ---------------------------------------------------------------------------
# the factor itself
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_factor_is_one_in_the_kinematical_limit():
    """y(0) = 1 exactly, with no 0/0 and no NaN gradient."""
    y = primary_extinction_factor(torch.zeros(1, dtype=DT), 50.0)
    assert abs(float(y) - 1.0) < 1e-15


@pytest.mark.unit
def test_factor_matches_tanh_form_away_from_zero():
    L = 50.0
    for t in (5.0, 25.0, 100.0, 400.0):
        x = t / L
        assert abs(float(primary_extinction_factor(t, L)) - math.tanh(x) / x) < 1e-14


@pytest.mark.unit
def test_factor_decreases_monotonically_with_path():
    L = 50.0
    ys = [float(primary_extinction_factor(t, L)) for t in (1.0, 10.0, 50.0, 200.0)]
    assert ys == sorted(ys, reverse=True)
    assert all(0.0 < y <= 1.0 for y in ys)


@pytest.mark.unit
def test_factor_saturates_as_one_over_x():
    """Dynamical limit: a thick perfect crystal cannot reflect more."""
    L = 10.0
    t = 1000.0
    assert abs(float(primary_extinction_factor(t, L)) - L / t) < 1e-9


@pytest.mark.unit
def test_small_path_expansion_is_continuous_with_the_tanh_branch():
    """The series branch and the exact branch must agree at the switch point."""
    L = 50.0
    t = 1e-6 * L * 1.0000001          # just above the small-x cutoff
    x = float(t) / L
    assert abs(float(primary_extinction_factor(t, L)) - math.tanh(x) / x) < 1e-12


# ---------------------------------------------------------------------------
# the validity boundary
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("tol", (0.01, 0.05, 0.1, 0.2))
def test_path_limit_costs_exactly_the_stated_tolerance(tol):
    """The number this module exists to provide: how thick is too thick."""
    L = 40.0
    t = float(kinematical_path_limit_um(L, tolerance=tol))
    y = float(primary_extinction_factor(t, L))
    assert abs((1.0 - y) - tol) < 1e-12       # exact solve, not an expansion


@pytest.mark.unit
def test_ten_percent_limit_is_a_bit_over_half_an_extinction_length():
    """x = 0.5838, exactly. The small-x expansion would say 0.5477 -- 6% low,
    and 20% low by tolerance = 0.2, which is why the solve is a bisection."""
    L = 40.0
    assert abs(float(kinematical_path_limit_um(L, tolerance=0.1)) / L - 0.583811) < 1e-5


@pytest.mark.unit
def test_bad_tolerance_rejected():
    for tol in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            kinematical_path_limit_um(40.0, tolerance=tol)


# ---------------------------------------------------------------------------
# direct beam
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_transmission_is_beer_lambert_without_diffraction():
    t, mu = 300.0, 1.0e-3
    got = float(direct_beam_transmission(t, mu_per_um=mu))
    assert abs(got - math.exp(-mu * t)) < 1e-15


@pytest.mark.unit
def test_diffraction_depletes_the_direct_beam():
    """DCT's extinction channel: the grain darkens the beam exactly when it flashes."""
    t, mu = 300.0, 1.0e-3
    off = float(direct_beam_transmission(t, mu_per_um=mu, diffracted_fraction=0.0))
    on = float(direct_beam_transmission(t, mu_per_um=mu, diffracted_fraction=0.2))
    assert abs(on / off - 0.8) < 1e-14


@pytest.mark.unit
def test_unphysical_diffracted_fraction_rejected():
    for f in (-0.1, 1.5):
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            direct_beam_transmission(100.0, mu_per_um=1e-3, diffracted_fraction=f)


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------
@pytest.mark.autograd
def test_gradcheck_extinction_factor():
    t = torch.tensor([5.0, 25.0, 120.0], dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: primary_extinction_factor(x, 40.0), (t,))


@pytest.mark.autograd
def test_gradient_is_finite_at_zero_path():
    """The 0/0 branch must not poison the backward pass."""
    t = torch.zeros(1, dtype=DT, requires_grad=True)
    primary_extinction_factor(t, 40.0).backward()
    assert torch.isfinite(t.grad).all()


@pytest.mark.autograd
def test_gradcheck_extinction_length_wrt_structure_factor():
    F = torch.tensor(100.0, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda f: extinction_length_um(f, unit_cell_volume_A3=48.0,
                                       wavelength_A=0.172979, theta_deg=2.36),
        (F,),
    )


# ---------------------------------------------------------------------------
# the physics is shared, not forked
# ---------------------------------------------------------------------------
@pytest.mark.contract
def test_extinction_physics_is_midas_hkls_not_a_local_copy():
    """These must BE the midas_hkls functions, not lookalikes.

    The extinction length, the tanh factor and the validity bound are generic
    X-ray physics shared with DFXM and pink-beam work. An identity check is the
    only test that cannot pass while a fork quietly drifts.
    """
    assert extinction_length_um is hkls_ext.extinction_length_um
    assert primary_extinction_factor is hkls_ext.primary_extinction_factor
    assert kinematical_path_limit_um is hkls_ext.kinematical_path_limit_um


@pytest.mark.contract
def test_direct_beam_channel_stays_local():
    """The DCT-specific piece is ours; it has no midas_hkls counterpart."""
    assert not hasattr(hkls_ext, "direct_beam_transmission")
