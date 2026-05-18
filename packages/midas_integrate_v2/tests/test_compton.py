"""Item 28 — Compton subtraction stub."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate_v2 import ComptonSubtraction


def test_compton_increases_with_Z():
    """Heavy elements have larger Compton intensity than light ones."""
    q = torch.linspace(1.0, 30.0, 200, dtype=torch.float64)
    cs_h = ComptonSubtraction({"H": 1.0}, wavelength_A=0.18)
    cs_pb = ComptonSubtraction({"Pb": 1.0}, wavelength_A=0.18)
    I_h = cs_h(q)
    I_pb = cs_pb(q)
    assert (I_pb > I_h).all(), "Pb Compton should exceed H at all Q"


def test_compton_high_Q_plateaus_at_Z():
    """At high Q, S_inc → Z; with Klein-Nishina ≤ 1, total ≤ Z."""
    q = torch.linspace(15.0, 30.0, 50, dtype=torch.float64)
    cs = ComptonSubtraction({"Cu": 1.0}, wavelength_A=0.18)
    I = cs(q)
    Z_cu = 29
    assert (I <= Z_cu).all()
    assert (I.max() > 0.5 * Z_cu)   # plateau is at least half of Z


def test_composition_renormalises():
    q = torch.linspace(1.0, 30.0, 100, dtype=torch.float64)
    cs1 = ComptonSubtraction({"Cu": 0.5, "O": 0.5}, wavelength_A=0.18)
    cs2 = ComptonSubtraction({"Cu": 1.0, "O": 1.0}, wavelength_A=0.18)
    np.testing.assert_allclose(cs1(q).numpy(), cs2(q).numpy(),
                                rtol=1e-10, atol=1e-12)


def test_unknown_element_raises():
    with pytest.raises(KeyError):
        ComptonSubtraction({"Xyzonium": 1.0}, wavelength_A=0.18)(
            torch.tensor([1.0, 2.0])
        )


def test_refinable_scale_gradient():
    q = torch.linspace(1.0, 20.0, 64, dtype=torch.float64)
    cs = ComptonSubtraction({"Ce": 1.0, "O": 2.0}, wavelength_A=0.18,
                              refinable_scale=True)
    L = cs(q).sum()
    L.backward()
    assert cs.scale.grad is not None
    assert torch.isfinite(cs.scale.grad)
    assert float(cs.scale.grad) != 0.0
