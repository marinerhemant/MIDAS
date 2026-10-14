"""Non-square pixels through the pixels -> I(Q) front end.

The v2 binning returns R in units of the MEAN pixel pitch; midas_pdf converted it to Q with pxY alone, so a peak at a known
Q came out at Q * pxY / mean (7 % at 172 x 200 um, 14 % at 200 x 150 um). The synthetic image is rendered from the physical
radius on the (non-square) pixel grid, so the recovered peak must sit at the true Q.
"""
import numpy as np
import pytest
import torch

from midas_integrate_v2.spec import IntegrationSpec

pytestmark = pytest.mark.slow

from midas_pdf.frontend import image_to_iq
from midas_pdf.validate import synthetic_powder_image

WAVELENGTH_A = 0.1665


def _t(x):
    return torch.tensor(float(x), dtype=torch.float64)


def _spec(pxy, pxz, n=512):
    s = IntegrationSpec()
    s.NrPixelsY = n
    s.NrPixelsZ = n
    s.pxY = pxy
    s.pxZ = pxz
    s.Lsd = _t(70000.0)
    s.BC_y = _t(n / 2.0)
    s.BC_z = _t(n / 2.0)
    s.Wavelength = _t(WAVELENGTH_A)
    s.RMin = 5.0
    s.RMax = n / 2.0 - 5.0
    s.RBinSize = 1.0
    s.EtaBinSize = 5.0
    return s


@pytest.mark.parametrize("pxy,pxz", [(172.0, 200.0), (200.0, 150.0), (150.0, 150.0)])
def test_image_to_iq_peak_at_true_q_for_non_square_pixels(pxy, pxz):
    s = _spec(pxy, pxz)
    q_prof = torch.linspace(0.3, 18.0, 1500, dtype=torch.float64)
    q_peak = 3.0
    I_prof = 1.0 + 20.0 * torch.exp(-0.5 * ((q_prof - q_peak) / 0.1) ** 2)
    img = synthetic_powder_image(s, q_prof, I_prof, counts=8e4, seed=3)
    Q, I, _ = image_to_iq(img, s)
    band = (Q > 1.5) & (Q < 5.0)
    q_recovered = float(Q[band][torch.argmax(I[band])])
    assert abs(q_recovered - q_peak) < 0.05, (q_recovered, q_peak)
