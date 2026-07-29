import numpy as np
import torch

from midas_pdf import Composition, faber_ziman_S, i_of_q_to_Gr, lumped_background
from midas_pdf.multiple_scattering import polynomial_basis

WAVELENGTH_A = 0.1665


def test_polynomial_basis_shape_and_values():
    q = torch.linspace(0.0, 10.0, 11, dtype=torch.float64)
    B = polynomial_basis(q, 2, q_max=10.0)
    assert B.shape == (11, 3)
    assert torch.allclose(B[:, 0], torch.ones(11, dtype=torch.float64))   # constant
    assert torch.allclose(B[:, 1], q / 10.0)                              # linear


def test_lumped_background_is_polynomial():
    q = torch.linspace(0.5, 20.0, 50, dtype=torch.float64)
    coef = torch.tensor([2.0, -0.5], dtype=torch.float64)   # 2 - 0.5*(Q/Qmax)
    b = lumped_background(q, coef, q_max=20.0)
    assert torch.allclose(b, 2.0 - 0.5 * (q / 20.0))


def test_background_none_is_identity():
    q = torch.linspace(0.7, 20.0, 300, dtype=torch.float64)
    comp = Composition({"Si": 1, "O": 2})
    I = torch.ones_like(q) * 50.0
    S0, _ = faber_ziman_S(I, q, comp, wavelength_A=WAVELENGTH_A)
    S1, _ = faber_ziman_S(I, q, comp, wavelength_A=WAVELENGTH_A, background=None)
    assert torch.allclose(S0, S1)


def test_background_subtracts_on_measured_scale():
    """S with a background equals S of (I - b) at scale=1 — i.e. the lumped
    term is removed on the measured scale before normalization."""
    q = torch.linspace(0.7, 20.0, 300, dtype=torch.float64)
    comp = Composition({"Si": 1, "O": 2})
    I = torch.linspace(20.0, 200.0, 300, dtype=torch.float64)
    b = 5.0 + 0.3 * q
    S_bg, _ = faber_ziman_S(I, q, comp, wavelength_A=WAVELENGTH_A,
                            scale=1.0, background=b)
    S_sub, _ = faber_ziman_S(I - b, q, comp, wavelength_A=WAVELENGTH_A, scale=1.0)
    assert torch.allclose(S_bg, S_sub, atol=1e-12)


def test_background_differentiable_through_pipeline():
    q = torch.linspace(0.7, 20.0, 600, dtype=torch.float64)
    r = torch.linspace(0.5, 8.0, 300, dtype=torch.float64)
    comp = Composition({"Si": 1, "O": 2})
    f_avg, f2_avg = comp.form_factor_averages(q)
    I = f2_avg * (1.0 + 0.3 * torch.sin(2.0 * q)) + comp.compton(q, wavelength_A=WAVELENGTH_A)

    coef = torch.tensor([1.0, 0.2], dtype=torch.float64, requires_grad=True)
    b = lumped_background(q, coef, q_max=20.0)
    G, _, _ = i_of_q_to_Gr(q, I, comp, r, wavelength_A=WAVELENGTH_A,
                           background=b, compton=True, q_max=18.0)
    G.pow(2).sum().backward()
    assert coef.grad is not None and torch.all(torch.isfinite(coef.grad))
