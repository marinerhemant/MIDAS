"""Item 27 — Cylindrical capillary absorption."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate_v2 import CylindricalAbsorption


def test_zero_mu_R_gives_unit_transmission():
    abs_mod = CylindricalAbsorption(mu_R=0.0)
    twoth = torch.linspace(0.01, 0.5, 50, dtype=torch.float64)
    A = abs_mod(twoth)
    np.testing.assert_allclose(A.numpy(), np.ones_like(twoth.numpy()),
                                rtol=1e-12, atol=1e-12)


def test_thin_limit_decreases_with_mu_R():
    twoth = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    A_low = CylindricalAbsorption(mu_R=0.1)(twoth)
    A_high = CylindricalAbsorption(mu_R=1.0)(twoth)
    assert (A_high < A_low).all()
    # All values must be in (0, 1].
    assert (A_low > 0).all() and (A_low <= 1).all()
    assert (A_high > 0).all() and (A_high <= 1).all()


def test_quadrature_path_thick_mu_R():
    """μR > 1.5 takes the quadrature path; result must be < 1 and finite."""
    twoth = torch.linspace(0.01, 0.5, 8, dtype=torch.float64)
    A = CylindricalAbsorption(mu_R=2.0)(twoth)
    assert torch.isfinite(A).all()
    assert (A < 1.0).all()
    assert (A > 0.0).all()


def test_refinable_mu_R_gradient_flows():
    twoth = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    abs_mod = CylindricalAbsorption(mu_R=0.5, refinable=True)
    A = abs_mod(twoth)
    L = A.sum()
    L.backward()
    assert abs_mod.mu_R.grad is not None
    assert torch.isfinite(abs_mod.mu_R.grad)
    assert float(abs_mod.mu_R.grad) != 0.0
