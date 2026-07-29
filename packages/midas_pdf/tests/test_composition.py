import numpy as np
import torch

from midas_pdf import Composition


def test_normalizes_fractions():
    comp = Composition({"Si": 1, "O": 2})
    assert comp.elements == ["Si", "O"]
    np.testing.assert_allclose(comp.fractions, [1 / 3, 2 / 3])


def test_f_of_zero_is_Z_weighted():
    # Cromer-Mann f(Q=0) ≈ Z (not exact), so check (a) it is near the Z-weighted
    # value and (b) the averaging is exactly the c-weighted sum of per-element
    # form factors at Q=0.
    from midas_hkls import form_factor

    comp = Composition({"Si": 1, "O": 2})  # Z: Si=14, O=8; c: 1/3, 2/3
    q = torch.zeros(1, dtype=torch.float64)
    f_avg, f2_avg = comp.form_factor_averages(q)

    assert abs(float(f_avg) - (1 / 3 * 14 + 2 / 3 * 8)) < 0.01  # near Z-weighted

    f_si = float(form_factor(torch.zeros(1, dtype=torch.float64), "Si"))
    f_o = float(form_factor(torch.zeros(1, dtype=torch.float64), "O"))
    assert abs(float(f_avg) - (1 / 3 * f_si + 2 / 3 * f_o)) < 1e-12
    assert abs(float(f2_avg) - (1 / 3 * f_si**2 + 2 / 3 * f_o**2)) < 1e-12


def test_monoatomic_laue_is_zero():
    comp = Composition({"Ni": 1})
    q = torch.linspace(0.5, 20.0, 50, dtype=torch.float64)
    laue = comp.laue(q)
    assert torch.allclose(laue, torch.zeros_like(laue), atol=1e-12)


def test_polyatomic_laue_positive():
    # ⟨f²⟩ - ⟨f⟩² >= 0 always (variance of f over composition).
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.5, 20.0, 50, dtype=torch.float64)
    laue = comp.laue(q)
    assert torch.all(laue > 0)


def test_averages_shape_matches_q():
    comp = Composition({"Fe": 1, "O": 1})
    q = torch.linspace(0.5, 18.0, 123, dtype=torch.float64)
    f_avg, f2_avg = comp.form_factor_averages(q)
    assert f_avg.shape == q.shape
    assert f2_avg.shape == q.shape


def test_differentiable_in_fractions():
    comp = Composition({"Si": 1, "O": 1})
    q = torch.linspace(0.5, 10.0, 20, dtype=torch.float64)
    frac = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    f_avg, _ = comp.form_factor_averages(q, fractions=frac)
    f_avg.sum().backward()
    assert frac.grad is not None
    assert torch.all(torch.isfinite(frac.grad))


def test_compton_reuses_integrate_v2():
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.5, 20.0, 40, dtype=torch.float64)
    cmp = comp.compton(q, wavelength_A=0.1665)
    assert cmp.shape == q.shape
    assert torch.all(cmp >= 0)
    # Compton grows with Q (more incoherent scattering at high Q).
    assert float(cmp[-1]) > float(cmp[0])
