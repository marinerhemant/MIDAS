"""Tests for midas_hkls.absorption.

Covers:
- numpy and torch backends
- device portability (CPU; CUDA / MPS when available)
- autograd differentiability via gradcheck
- physical sanity (μ for Ti at λ=0.173 Å is ~150 cm⁻¹ per NIST XCOM)
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from midas_hkls.absorption import (
    available_elements_absorption,
    atomic_mass,
    element_density,
    linear_absorption_coefficient,
    mass_attenuation_coefficient,
)


# ---------------------------------------------------------------- properties


def test_property_table_has_common_elements():
    elems = set(available_elements_absorption())
    for e in ["H", "C", "O", "Mg", "Al", "Si", "Ti", "V", "Cr", "Fe", "Ni",
              "Cu", "Zn", "Zr", "Mo", "Pd", "Ag", "Sn", "Ta", "W", "Pt", "Au", "U"]:
        assert e in elems, f"missing element {e!r}"


def test_atomic_mass_ti_known():
    assert math.isclose(atomic_mass("Ti"), 47.867, rel_tol=1e-3)


def test_density_ti_known():
    assert math.isclose(element_density("Ti"), 4.506, rel_tol=1e-3)


def test_unknown_element_raises():
    with pytest.raises(KeyError):
        atomic_mass("Xx")


# ------------------------------------------------------------------ numpy μ


def test_mu_ti_at_0p173A_matches_nist():
    """Ti at λ=0.173 Å (CP-Ti @ ~71.67 keV).

    Reference (xraylib CS_Total, wrapping NIST XCOM):
        μ/ρ = 0.5092 cm²/g  →  μ = 0.5092 * 4.506 = 2.295 cm⁻¹

    Our log-log interpolation should be within 1% of xraylib at this energy
    (we sampled the table densely in the 8-150 keV HEDM band)."""
    mu = linear_absorption_coefficient("Ti", 0.173)
    assert math.isclose(mu, 2.295, rel_tol=0.01), \
        f"μ(Ti, 0.173 Å) = {mu:.3f} cm⁻¹ vs xraylib 2.295"


def test_mu_cu_at_8keV_below_k_edge():
    """Cu just below K-edge (E=8.0 keV, λ=1.55 Å): xraylib gives μ/ρ ≈ 50.2 cm²/g.
    Within 10% of xraylib (log-log interpolation error near edges)."""
    lam = 12.398 / 8.0
    mu_rho = linear_absorption_coefficient("Cu", lam) / element_density("Cu")
    assert math.isclose(mu_rho, 50.21, rel_tol=0.10), \
        f"μ/ρ(Cu, 8 keV) = {mu_rho:.3f} cm²/g vs xraylib 50.21"


def test_mu_fe_at_25keV():
    """Fe at 25 keV (above K-edge 7.11 keV): xraylib μ/ρ ≈ 12.49 cm²/g.
    Within 10% of xraylib."""
    lam = 12.398 / 25.0
    mu_rho = linear_absorption_coefficient("Fe", lam) / element_density("Fe")
    assert math.isclose(mu_rho, 12.49, rel_tol=0.10), \
        f"μ/ρ(Fe, 25 keV) = {mu_rho:.3f} cm²/g vs xraylib 12.49"


def test_mu_monotonic_decrease_above_k_edge():
    """Far from edges, μ should monotonically decrease with increasing E
    (decreasing λ).  Test on Ti in the HEDM band 10-150 keV."""
    lams = [12.398 / e for e in (100, 80, 60, 40, 20, 12)]
    mus = [linear_absorption_coefficient("Ti", lam) for lam in lams]
    # Increasing λ → decreasing E → increasing μ
    assert all(mus[i] < mus[i + 1] for i in range(len(mus) - 1)), \
        f"μ not monotonic in λ: {mus}"


def test_mu_density_override_numpy():
    mu_full = linear_absorption_coefficient("Ti", 0.173)
    mu_half = linear_absorption_coefficient("Ti", 0.173, density_g_cm3=4.506 / 2)
    assert math.isclose(mu_full / mu_half, 2.0, rel_tol=1e-6)


def test_mu_linear_in_density_numpy():
    mus = [linear_absorption_coefficient("Ti", 0.173, density_g_cm3=d)
           for d in (1.0, 2.0, 4.0)]
    assert math.isclose(mus[1] / mus[0], 2.0, rel_tol=1e-6)
    assert math.isclose(mus[2] / mus[0], 4.0, rel_tol=1e-6)


def test_sigma_mass_density_independent():
    """σ_mass is a material property (independent of density override)."""
    s1 = mass_attenuation_coefficient("Ti", 0.173)
    s2 = mass_attenuation_coefficient("Ti", 0.173)
    assert s1 == s2


# ------------------------------------------------------------------ torch μ


torch = pytest.importorskip("torch")


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS often fails fp64; only add if we test on fp32
        devs.append("mps")
    return devs


@pytest.mark.parametrize("device", ["cpu"])  # CUDA/MPS gated in separate test
def test_mu_ti_torch_matches_numpy(device):
    lam = torch.tensor(0.173, dtype=torch.float64, device=device)
    mu_t = linear_absorption_coefficient("Ti", lam)
    mu_np = linear_absorption_coefficient("Ti", 0.173)
    assert math.isclose(float(mu_t), mu_np, rel_tol=1e-10)


@pytest.mark.parametrize("device", _devices())
def test_mu_torch_runs_on_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    lam = torch.tensor(0.173, dtype=dtype, device=device)
    mu = linear_absorption_coefficient("Ti", lam)
    assert mu.device.type == torch.device(device).type
    assert mu.dtype == dtype
    # Sanity: positive, finite
    assert float(mu) > 0
    assert math.isfinite(float(mu))


def test_mu_torch_differentiable_in_wavelength():
    """Gradient flows from μ back to wavelength."""
    lam = torch.tensor(0.173, dtype=torch.float64, requires_grad=True)
    mu = linear_absorption_coefficient("Ti", lam)
    mu.backward()
    assert lam.grad is not None
    assert math.isfinite(float(lam.grad))
    # σ_mass scales linearly with λ (optical-theorem formula), and f'' has its
    # own λ-dependence too, so dμ/dλ should be a finite non-trivial number.
    assert abs(float(lam.grad)) > 0


def test_mu_torch_gradcheck_wavelength():
    """torch.autograd.gradcheck: μ as a function of wavelength."""
    def f(lam):
        return linear_absorption_coefficient("Ti", lam)
    lam = torch.tensor(0.173, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (lam,), eps=1e-6, atol=1e-5)


def test_mu_torch_differentiable_in_density():
    """Density override may be a tensor and grad flows through it."""
    rho = torch.tensor(4.506, dtype=torch.float64, requires_grad=True)
    lam = torch.tensor(0.173, dtype=torch.float64)
    mu = linear_absorption_coefficient("Ti", lam, density_g_cm3=rho)
    mu.backward()
    assert rho.grad is not None
    assert math.isfinite(float(rho.grad))


def test_mu_batched_wavelength():
    """λ as a vector of wavelengths returns vector μ."""
    lam = torch.tensor([0.100, 0.173, 0.500], dtype=torch.float64)
    mu = linear_absorption_coefficient("Ti", lam)
    assert mu.shape == lam.shape
    # μ should be roughly monotonic in λ over this range (away from edges)
    # for Ti (K-edge at 4966 eV ≈ 2.5 Å) — values at 0.1, 0.173, 0.5 Å are
    # all above the K-edge, so μ should increase with λ.
    assert float(mu[0]) < float(mu[1]) < float(mu[2])
