"""Rev-9 tests: core-shell + multi-shell SAXS form factors."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.saxs import (
    core_shell_sphere_form_factor_squared,
    multi_shell_sphere_form_factor_squared,
    sphere_form_factor_squared,
)


# ---------------------------------------------------------------------------
# Uniform-contrast limits reduce to plain sphere
# ---------------------------------------------------------------------------

def test_core_shell_uniform_contrast_equals_sphere():
    """When core and shell have the same contrast, the whole particle is a
    uniform-density sphere of radius R_total."""
    q = torch.linspace(0.001, 0.5, 40, dtype=torch.float64)
    F2_cs = core_shell_sphere_form_factor_squared(
        q, R_core_A=30.0, R_total_A=50.0,
        contrast_core=1.0, contrast_shell=1.0)
    F2_sphere = sphere_form_factor_squared(q, 50.0)
    assert torch.allclose(F2_cs, F2_sphere, atol=1e-9)


def test_multi_shell_single_shell_equals_sphere():
    """A one-shell multi-shell is a plain sphere."""
    q = torch.linspace(0.001, 0.5, 40, dtype=torch.float64)
    F2_ms = multi_shell_sphere_form_factor_squared(q, [50.0], [1.0])
    F2_sphere = sphere_form_factor_squared(q, 50.0)
    assert torch.allclose(F2_ms, F2_sphere, atol=1e-9)


# ---------------------------------------------------------------------------
# Q → 0 limit matches the analytical formula
# ---------------------------------------------------------------------------

def test_core_shell_Q0_limit_matches_theory():
    """|F(Q=0)|² = (Δρ_core V_core + Δρ_shell V_total)²
    where Δρ_core = ρ_core − ρ_shell, Δρ_shell = ρ_shell − ρ_solvent."""
    q = torch.tensor([1e-6], dtype=torch.float64)
    R_c, R_t = 30.0, 50.0
    rho_c, rho_s = 1.0, -0.5
    V_c = 4 / 3 * np.pi * R_c ** 3
    V_t = 4 / 3 * np.pi * R_t ** 3
    F0 = ((rho_c - rho_s) * V_c + rho_s * V_t) ** 2
    F2 = core_shell_sphere_form_factor_squared(q, R_c, R_t, rho_c, rho_s)
    assert abs(float(F2[0]) / F0 - 1.0) < 1e-3


def test_multi_shell_Q0_limit_matches_theory():
    """|F(Q=0)|² = (Σᵢ (ρᵢ − ρᵢ₊₁) Vᵢ)²."""
    q = torch.tensor([1e-6], dtype=torch.float64)
    radii = [20.0, 35.0, 50.0]
    contrasts = [1.0, 0.5, -0.2]
    F_total = 0.0
    for i in range(len(radii)):
        Vi = 4 / 3 * np.pi * radii[i] ** 3
        rho_next = contrasts[i + 1] if i + 1 < len(contrasts) else 0.0
        F_total += (contrasts[i] - rho_next) * Vi
    F2 = multi_shell_sphere_form_factor_squared(q, radii, contrasts)
    assert abs(float(F2[0]) / F_total ** 2 - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Contrast matching gives zero intensity
# ---------------------------------------------------------------------------

def test_core_shell_contrast_matching_zeros_intensity_at_Q0():
    """When the volume-averaged contrast is zero, I(Q=0) = 0
    (contrast-matched suspension: solvent index matches average particle)."""
    # Pick R_c, R_t and (ρ_c, ρ_s) such that Δρ_core V_c + Δρ_shell V_t = 0
    R_c, R_t = 30.0, 50.0
    V_c = 4 / 3 * np.pi * R_c ** 3
    V_t = 4 / 3 * np.pi * R_t ** 3
    # Choose ρ_s = -1, then (ρ_c + 1) V_c + (-1) V_t = 0
    # ρ_c = V_t / V_c - 1
    rho_s = -1.0
    rho_c = V_t / V_c + rho_s
    q = torch.tensor([1e-6], dtype=torch.float64)
    F2 = core_shell_sphere_form_factor_squared(q, R_c, R_t, rho_c, rho_s)
    # Cancellation of two ~10⁹ numbers → 1e-6 tolerance in float64
    assert float(F2[0]) < 1e-6


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_core_shell_differentiable_in_radii_and_contrasts():
    q = torch.linspace(0.01, 0.3, 10, dtype=torch.float64)
    R_c = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    R_t = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    rho_c = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    rho_s = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
    F2 = core_shell_sphere_form_factor_squared(q, R_c, R_t, rho_c, rho_s)
    F2.sum().backward()
    for p in (R_c, R_t, rho_c, rho_s):
        assert torch.isfinite(p.grad)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_multi_shell_length_mismatch_raises():
    q = torch.tensor([0.1], dtype=torch.float64)
    with pytest.raises(ValueError):
        multi_shell_sphere_form_factor_squared(q, [10.0, 20.0], [1.0])


def test_multi_shell_empty_raises():
    q = torch.tensor([0.1], dtype=torch.float64)
    with pytest.raises(ValueError):
        multi_shell_sphere_form_factor_squared(q, [], [])


# ---------------------------------------------------------------------------
# Physical sanity: monotonic at very low Q
# ---------------------------------------------------------------------------

def test_core_shell_monotonic_near_Q0():
    """From Q=0 up to the first minimum, |F|² decreases monotonically."""
    q = torch.linspace(1e-6, 0.05, 40, dtype=torch.float64)
    F2 = core_shell_sphere_form_factor_squared(
        q, R_core_A=30.0, R_total_A=50.0, contrast_core=1.0, contrast_shell=0.3)
    assert torch.all(torch.diff(F2) < 1e-9)
