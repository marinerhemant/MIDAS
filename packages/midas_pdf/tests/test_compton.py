import numpy as np
import torch

from midas_pdf import Composition
from midas_pdf.compton import breit_dirac_factor, incoherent_scattering

WAVELENGTH_A = 0.1665


def test_incoherent_limits():
    """S_inc -> 0 at Q=0 and rises toward (composition-weighted Z) at high Q."""
    q = torch.tensor([0.0, 30.0], dtype=torch.float64)
    # single element O (Z=8): no Breit-Dirac so we see the bare S_inc limits
    I = incoherent_scattering(q, ["O"], wavelength_A=WAVELENGTH_A, breit_dirac=False)
    assert float(I[0]) < 0.05
    assert 7.0 < float(I[1]) <= 8.01


def test_incoherent_monotonic_and_weighted():
    q = torch.linspace(0.5, 25.0, 200, dtype=torch.float64)
    comp = Composition({"Si": 1, "O": 2})
    I = incoherent_scattering(q, comp.elements, wavelength_A=WAVELENGTH_A,
                              fractions=torch.tensor(comp.fractions), breit_dirac=False)
    assert torch.all(I[1:] >= I[:-1] - 1e-6)          # monotonic non-decreasing
    # approaches the composition-weighted <Z> = (14 + 2*8)/3 = 10 from BELOW
    # (S_inc reaches Z only asymptotically), so expect just under it at Q=25.
    Z_avg = (14 + 2 * 8) / 3
    assert 8.5 < float(I[-1]) < Z_avg


def test_breit_dirac_reduces_high_q():
    q = torch.linspace(0.5, 30.0, 100, dtype=torch.float64)
    R = breit_dirac_factor(q, wavelength_A=WAVELENGTH_A, k=2)
    assert float(R[0]) <= 1.0 and float(R[0]) > 0.99   # ~1 at low Q
    assert float(R[-1]) < float(R[0])                  # damps at high Q
    assert torch.all((R > 0) & (R <= 1.0))


def test_incoherent_differentiable_in_Q_and_fractions():
    q = torch.linspace(1.0, 20.0, 50, dtype=torch.float64, requires_grad=True)
    frac = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    I = incoherent_scattering(q, ["Si", "O"], wavelength_A=WAVELENGTH_A, fractions=frac)
    I.sum().backward()
    assert q.grad is not None and torch.all(torch.isfinite(q.grad))
    assert frac.grad is not None and torch.all(torch.isfinite(frac.grad))


def test_hubbell_vs_it94_same_order():
    """Hubbell and the coarse IT94 should agree to within a factor ~2 (sanity,
    not identity — the whole point is Hubbell is more accurate)."""
    q = torch.linspace(2.0, 18.0, 50, dtype=torch.float64)
    comp = Composition({"Si": 1, "O": 2})
    hub = comp.compton(q, wavelength_A=WAVELENGTH_A, method="hubbell", breit_dirac=False)
    it94 = comp.compton(q, wavelength_A=WAVELENGTH_A, method="it94")
    ratio = (hub / it94.clamp(min=1e-6)).median()
    assert 0.3 < float(ratio) < 3.0
