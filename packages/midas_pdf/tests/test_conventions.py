import numpy as np
import torch

from midas_pdf.conventions import (
    pair_distribution_g,
    radial_distribution_R,
    structure_function_F,
    total_correlation_T,
)

RHO = 0.066  # atoms / Å³, ~ a condensed solid


def test_F_is_Q_times_S_minus_one():
    q = torch.linspace(0.5, 25.0, 100, dtype=torch.float64)
    S = 1.0 + 0.3 * torch.sin(2.0 * q)
    F, sigF = structure_function_F(q, S, sigma_S=0.1 * torch.ones_like(q))
    assert torch.allclose(F, q * (S - 1.0))
    assert torch.allclose(sigF, q.abs() * 0.1)


def test_function_family_consistency():
    # T(r) = G + 4πrρ, R(r) = r·G + 4πr²ρ, g = 1 + G/(4πrρ).
    # Check the identities R = r·T and T = 4πrρ·g hold pointwise.
    r = torch.linspace(0.05, 10.0, 500, dtype=torch.float64)
    G = torch.sin(5.0 * r) * torch.exp(-0.2 * r)  # arbitrary smooth G(r)

    g, _ = pair_distribution_g(r, G, number_density=RHO)
    T, _ = total_correlation_T(r, G, number_density=RHO)
    R, _ = radial_distribution_R(r, G, number_density=RHO)

    np.testing.assert_allclose(R.numpy(), (r * T).numpy(), rtol=1e-10)
    np.testing.assert_allclose(
        T.numpy(), (4 * np.pi * r * RHO * g).numpy(), rtol=1e-10
    )


def test_g_tends_to_one_at_large_r():
    r = torch.linspace(0.05, 50.0, 2000, dtype=torch.float64)
    G = torch.zeros_like(r)  # G→0 at large r ⇒ g→1
    g, _ = pair_distribution_g(r, G, number_density=RHO)
    assert abs(float(g[-1]) - 1.0) < 1e-9


def test_sigma_scaling_rules():
    r = torch.linspace(0.05, 10.0, 200, dtype=torch.float64)
    G = torch.cos(3.0 * r)
    sg = 0.05 * torch.ones_like(r)
    _, sigT = total_correlation_T(r, G, number_density=RHO, sigma_G=sg)
    _, sigR = radial_distribution_R(r, G, number_density=RHO, sigma_G=sg)
    assert torch.allclose(sigT, sg)            # additive shift ⇒ σ unchanged
    assert torch.allclose(sigR, r * sg)        # R = r·G ⇒ σ_R = r·σ_G
