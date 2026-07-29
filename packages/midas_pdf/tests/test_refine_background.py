import numpy as np
import torch

from midas_pdf import Composition
from midas_pdf.refine import refine_normalization

WAVELENGTH_A = 0.1665
RHO = 0.0709


def _synthetic_iq(comp, q, *, r0=1.62, shell=4.0, broadening=0.05):
    f_avg, f2_avg = comp.form_factor_averages(q)
    debye = torch.sin(q * r0) / (q * r0) * torch.exp(-0.5 * (q * broadening) ** 2)
    cmp = comp.compton(q, wavelength_A=WAVELENGTH_A)
    return f2_avg * (1.0 + shell * debye) + cmp


def test_polynomial_background_removes_added_slope():
    """Add a smooth linear baseline (mimicking fluorescence/air scatter) to the
    data and check a bg_order=1 refinement removes it and recovers the scale."""
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.7, 22.0, 2000, dtype=torch.float64)
    r = torch.linspace(0.0, 10.0, 800, dtype=torch.float64)
    I_clean = _synthetic_iq(comp, q)
    baseline = 3.0 + 0.4 * q                      # planted smooth background
    I = I_clean + baseline

    res = refine_normalization(
        q, I, comp, r, wavelength_A=WAVELENGTH_A, number_density=RHO,
        compton=True, q_max=20.0, r_min_phys=1.2,
        init_scale=1.0, fit_background=True, bg_order=1, steps=120,
    )
    assert len(res.bg_coef) == 2
    assert res.background.shape == q.shape
    assert res.history[-1] < res.history[0]
    # low-r slope should still be enforced after background removal
    lowr = (r > 0) & (r < 1.2)
    target = -4 * np.pi * RHO * r[lowr]
    rms = torch.sqrt(((res.G[lowr] - target) ** 2).mean())
    assert float(rms) < 0.6


def test_bg_order_zero_matches_constant_offset():
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.7, 22.0, 1500, dtype=torch.float64)
    r = torch.linspace(0.0, 10.0, 700, dtype=torch.float64)
    I = _synthetic_iq(comp, q)
    res = refine_normalization(
        q, I, comp, r, wavelength_A=WAVELENGTH_A, number_density=RHO,
        compton=True, q_max=20.0, r_min_phys=1.2, bg_order=0, steps=60,
    )
    assert len(res.bg_coef) == 1
    assert abs(res.offset - res.bg_coef[0]) < 1e-12
