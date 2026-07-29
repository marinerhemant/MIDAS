import numpy as np
import torch

from midas_pdf import Composition
from midas_pdf.cross_section import (
    differential_cross_section,
    polarization_factor,
    total_cross_section,
)

WAVELENGTH_A = 0.1665


def test_polarization_unpolarized_limits():
    q = torch.linspace(0.0, 20.0, 100, dtype=torch.float64)
    P = polarization_factor(q, wavelength_A=WAVELENGTH_A, polarization_fraction=0.0)
    assert abs(float(P[0]) - 1.0) < 1e-9        # forward scattering: P=1
    assert torch.all((P > 0) & (P <= 1.0 + 1e-9))


def test_dsigma_positive_and_finite():
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.5, 25.0, 300, dtype=torch.float64)
    dsig = differential_cross_section(q, comp, wavelength_A=WAVELENGTH_A)
    assert torch.all(dsig > 0)
    assert torch.all(torch.isfinite(dsig))


def test_dsigma_coherent_dominant_at_low_Q():
    # at low Q coherent ~ <f>^2 ~ <Z>^2 >> incoherent (-> 0); check incoherent
    # toggle changes things little at low Q, a lot at high Q.
    comp = Composition({"Si": 1, "O": 2})
    q = torch.tensor([1.0, 24.0], dtype=torch.float64)
    with_inc = differential_cross_section(q, comp, wavelength_A=WAVELENGTH_A,
                                          include_incoherent=True)
    no_inc = differential_cross_section(q, comp, wavelength_A=WAVELENGTH_A,
                                        include_incoherent=False)
    # low Q: incoherent negligible
    assert abs(float(with_inc[0]) - float(no_inc[0])) / float(no_inc[0]) < 0.05
    # high Q: incoherent is a significant fraction
    assert (float(with_inc[1]) - float(no_inc[1])) / float(no_inc[1]) > 0.1


def test_total_cross_section_positive_and_differentiable():
    comp = Composition({"Si": 1, "O": 2})
    sigma = total_cross_section(comp, wavelength_A=WAVELENGTH_A)
    assert float(sigma) > 0 and np.isfinite(float(sigma))


def test_dsigma_differentiable_in_Q_and_composition():
    comp = Composition({"Si": 1, "O": 1})
    q = torch.linspace(1.0, 20.0, 40, dtype=torch.float64, requires_grad=True)
    frac = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    dsig = differential_cross_section(q, comp, wavelength_A=WAVELENGTH_A, fractions=frac)
    dsig.sum().backward()
    assert q.grad is not None and torch.all(torch.isfinite(q.grad))
    assert frac.grad is not None and torch.all(torch.isfinite(frac.grad))
