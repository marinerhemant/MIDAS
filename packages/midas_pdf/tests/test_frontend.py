import numpy as np
import pytest
import torch

from midas_integrate_v2.spec import IntegrationSpec

pytestmark = pytest.mark.slow

from midas_pdf import Composition
from midas_pdf.frontend import image_to_Gr, image_to_iq
from midas_pdf.validate import debye_scattering_intensity, synthetic_powder_image

WAVELENGTH_A = 0.1665


def _t(x):
    return torch.tensor(float(x), dtype=torch.float64)


def _spec(n=512):
    s = IntegrationSpec()
    s.NrPixelsY = n
    s.NrPixelsZ = n
    s.pxY = 150.0
    s.pxZ = 150.0
    s.Lsd = _t(70000.0)           # µm (70 mm → Qmax≈18 at 512²); tensors required
    s.BC_y = _t(n / 2.0)
    s.BC_z = _t(n / 2.0)
    s.Wavelength = _t(WAVELENGTH_A)
    s.RMin = 5.0
    s.RMax = n / 2.0 - 5.0
    s.RBinSize = 1.0
    s.EtaBinSize = 5.0
    return s


def test_image_to_iq_recovers_profile_peak():
    """Render an image from a single-peak I(Q), integrate back, check the
    recovered profile peaks at the same Q."""
    s = _spec()
    q_prof = torch.linspace(0.3, 18.0, 1500, dtype=torch.float64)
    q_peak = 3.0
    I_prof = 1.0 + 20.0 * torch.exp(-0.5 * ((q_prof - q_peak) / 0.1) ** 2)
    img = synthetic_powder_image(s, q_prof, I_prof, counts=8e4, seed=3)

    Q, I, sigma_I = image_to_iq(img, s)
    assert Q.shape == I.shape == sigma_I.shape
    assert torch.all(sigma_I >= 0)
    band = (Q > 1.5) & (Q < 5.0)
    q_recovered = float(Q[band][torch.argmax(I[band])])
    assert abs(q_recovered - q_peak) < 0.05


def test_image_to_Gr_dimer():
    """Full pixels → G(r): render a Ni dimer's Debye I(Q) onto a detector,
    integrate + normalize + FT, and recover the bond length."""
    s = _spec()
    d = 2.50
    q_prof = torch.linspace(0.3, 20.0, 2500, dtype=torch.float64)
    I_prof = debye_scattering_intensity(
        q_prof, ["Ni", "Ni"],
        torch.tensor([[0, 0, 0], [d, 0, 0]], dtype=torch.float64),
        thermal_B=0.3,
    )
    img = synthetic_powder_image(s, q_prof, I_prof, counts=2e5, seed=5)

    r = torch.linspace(0.5, 6.0, 1100, dtype=torch.float64)
    comp = Composition({"Ni": 1})
    Q, G, sigma_G, S = image_to_Gr(img, s, comp, r, compton=False,
                                   q_min=0.8, q_max=16.0)
    assert torch.all(sigma_G >= 0)
    win = (r >= 1.5) & (r <= 4.0)
    r_peak = float(r[win][torch.argmax(G[win])])
    assert abs(r_peak - d) < 0.15
