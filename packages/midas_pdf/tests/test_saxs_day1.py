"""SAXS Day-1 tests: form factors + Percus-Yevick structure factor.

Invariants:
  1. Sphere: |F(Q → 0)|² = V², V = 4πR³/3.
  2. Sphere: first zero of |F|² at Q R ≈ 4.4934 (first root of tan x = x).
  3. Ellipsoid at a = b reduces exactly to sphere.
  4. Cylinder: |F(Q → 0)|² = V², V = π R² L.
  5. Percus-Yevick S(0) matches the Wertheim closed form
     (1-φ)⁴ / (1 + 2φ)² to 4 decimals at every valid φ.
  6. Percus-Yevick first peak at Qσ ≈ 6-7 (universal hard-sphere feature).
  7. All routines are differentiable in torch.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.saxs.form_factors import (
    sphere_form_factor_squared,
    ellipsoid_form_factor_squared,
    cylinder_form_factor_squared,
    percus_yevick_S,
)


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

def test_sphere_zero_Q_limit_is_V_squared():
    R = 30.0
    V = 4 / 3 * np.pi * R ** 3
    q = torch.tensor([1e-6, 1e-4], dtype=torch.float64)
    F2 = sphere_form_factor_squared(q, R)
    rel = torch.abs(F2 / V ** 2 - 1)
    assert torch.all(rel < 1e-3), f"rel err = {rel}"


def test_sphere_first_zero_at_QR_4_4934():
    """First root of the transcendental tan x = x is at x ≈ 4.4934."""
    R = 50.0
    q = torch.linspace(4.0 / R, 5.0 / R, 500, dtype=torch.float64)
    F2 = sphere_form_factor_squared(q, R)
    q_at_min = float(q[int(F2.argmin())])
    assert abs(q_at_min * R - 4.4934) < 0.01


def test_sphere_monotonic_decrease_between_zeros():
    """From Q=0 to the first zero, |F|² decreases monotonically."""
    R = 40.0
    q = torch.linspace(1e-6, 4.4934 / R * 0.99, 100, dtype=torch.float64)
    F2 = sphere_form_factor_squared(q, R)
    assert torch.all(torch.diff(F2) < 1e-10)


def test_sphere_differentiable_in_R():
    q = torch.tensor([0.05, 0.1, 0.15], dtype=torch.float64)
    R = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    F2 = sphere_form_factor_squared(q, R)
    F2.sum().backward()
    assert R.grad is not None
    assert torch.isfinite(R.grad)


# ---------------------------------------------------------------------------
# Ellipsoid
# ---------------------------------------------------------------------------

def test_ellipsoid_reduces_to_sphere_when_ab_equal():
    R = 40.0
    q = torch.linspace(0.01, 0.5, 30, dtype=torch.float64)
    F2_sphere = sphere_form_factor_squared(q, R)
    F2_ellipsoid = ellipsoid_form_factor_squared(q, R, R)
    assert torch.allclose(F2_sphere, F2_ellipsoid, rtol=1e-8, atol=1e-8)


def test_ellipsoid_prolate_larger_than_sphere_at_low_Q():
    """A prolate ellipsoid (a > b) has slightly larger orientation-average
    |F|² at very low Q than a sphere of the equatorial radius (same volume
    yields the same Q→0 limit)."""
    R = 30.0
    a = 60.0                # rotation axis
    b = R                    # equatorial
    q = torch.tensor([1e-5], dtype=torch.float64)
    F2_ell = ellipsoid_form_factor_squared(q, a, b)
    # V_ell = 4π a b² / 3 = 4π · 60 · 900 / 3 = 4π · 18000
    V_ell = 4 / 3 * np.pi * a * b ** 2
    assert abs(float(F2_ell[0]) / V_ell ** 2 - 1) < 1e-3


def test_ellipsoid_differentiable_in_a_and_b():
    q = torch.tensor([0.05, 0.1], dtype=torch.float64)
    a = torch.tensor(60.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    F2 = ellipsoid_form_factor_squared(q, a, b)
    F2.sum().backward()
    assert torch.isfinite(a.grad) and torch.isfinite(b.grad)


# ---------------------------------------------------------------------------
# Cylinder
# ---------------------------------------------------------------------------

def test_cylinder_zero_Q_limit_is_V_squared():
    R = 25.0
    L = 100.0
    V = np.pi * R ** 2 * L
    q = torch.tensor([1e-8], dtype=torch.float64)
    F2 = cylinder_form_factor_squared(q, R, L)
    assert abs(float(F2[0]) / V ** 2 - 1) < 1e-3


def test_cylinder_differentiable_in_R_and_L():
    q = torch.tensor([0.05, 0.1], dtype=torch.float64)
    R = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
    L = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    F2 = cylinder_form_factor_squared(q, R, L)
    F2.sum().backward()
    assert torch.isfinite(R.grad) and torch.isfinite(L.grad)


# ---------------------------------------------------------------------------
# Percus-Yevick
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
def test_percus_yevick_S0_matches_wertheim(phi):
    """S(Q → 0) = (1-φ)⁴ / (1 + 2φ)² to 4 decimals."""
    q = torch.tensor([1e-5], dtype=torch.float64)
    S = percus_yevick_S(q, radius_A=50.0, volume_fraction=phi)
    S0_theory = (1 - phi) ** 4 / (1 + 2 * phi) ** 2
    assert abs(float(S[0]) - S0_theory) < 1e-4


def test_percus_yevick_first_peak_near_Qsigma_6():
    """For any physical φ, the first S(Q) peak sits at Qσ ≈ 6-7."""
    R = 40.0
    q = torch.linspace(0.001, 0.5, 500, dtype=torch.float64)
    S = percus_yevick_S(q, radius_A=R, volume_fraction=0.3)
    peak_q = float(q[int(S.argmax())])
    assert 5.5 < peak_q * 2 * R < 7.5


def test_percus_yevick_S_positive_everywhere():
    """A physical S(Q) is > 0 everywhere. The PY formula never dips
    below zero for φ < 0.5."""
    R = 40.0
    q = torch.linspace(0.001, 1.0, 500, dtype=torch.float64)
    for phi in (0.05, 0.15, 0.3, 0.4):
        S = percus_yevick_S(q, R, phi)
        assert torch.all(S > 0), f"φ={phi}: S dips below 0"


def test_percus_yevick_rejects_out_of_range_phi():
    q = torch.tensor([0.1], dtype=torch.float64)
    with pytest.raises(ValueError):
        percus_yevick_S(q, radius_A=50.0, volume_fraction=-0.1)
    with pytest.raises(ValueError):
        percus_yevick_S(q, radius_A=50.0, volume_fraction=0.6)


def test_percus_yevick_dilute_limit_returns_S_close_to_1():
    """φ → 0 → S(Q) → 1 everywhere (no correlation)."""
    q = torch.linspace(0.01, 0.5, 30, dtype=torch.float64)
    S = percus_yevick_S(q, radius_A=40.0, volume_fraction=1e-4)
    assert torch.all(torch.abs(S - 1.0) < 5e-3)
