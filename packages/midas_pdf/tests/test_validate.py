import numpy as np
import torch

from midas_pdf import Composition, i_of_q_to_Gr
from midas_pdf.validate import debye_scattering_intensity, interatomic_distances

WAVELENGTH_A = 0.1665


def test_dimer_gr_peaks_at_bond_length():
    """A two-atom cluster at distance d must give a G(r) peak at ~d."""
    d = 2.50
    elements = ["Ni", "Ni"]
    pos = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float64)
    q = torch.linspace(0.7, 24.0, 3000, dtype=torch.float64)
    r = torch.linspace(0.5, 6.0, 1100, dtype=torch.float64)

    I = debye_scattering_intensity(q, elements, pos, thermal_B=0.3)
    comp = Composition({"Ni": 1})
    G, _, _ = i_of_q_to_Gr(q, I, comp, r, wavelength_A=WAVELENGTH_A,
                           compton=False, q_max=22.0)
    # Peak of G(r) in [1.5, 4] Å should sit at the bond length.
    win = (r >= 1.5) & (r <= 4.0)
    r_peak = float(r[win][torch.argmax(G[win])])
    assert abs(r_peak - d) < 0.10


def test_interatomic_distances_symmetric_zero_diag():
    pos = torch.randn(5, 3, dtype=torch.float64)
    D = interatomic_distances(pos)
    assert torch.allclose(D, D.T)
    assert torch.allclose(torch.diag(D), torch.zeros(5, dtype=torch.float64))


def test_debye_differentiable_in_positions():
    elements = ["Si", "O", "O"]
    pos = torch.tensor([[0., 0., 0.], [1.6, 0., 0.], [0., 1.6, 0.]],
                       dtype=torch.float64, requires_grad=True)
    q = torch.linspace(1.0, 15.0, 200, dtype=torch.float64)
    I = debye_scattering_intensity(q, elements, pos)
    I.sum().backward()
    assert pos.grad is not None and torch.all(torch.isfinite(pos.grad))
