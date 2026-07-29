import numpy as np
import torch

from midas_pdf import Composition
from midas_pdf.refine import refine_normalization

WAVELENGTH_A = 0.1665
RHO = 0.0709  # atoms/Å³ for SiO2-like (3 atoms / unit, ~ density)


def _synthetic_iq(comp, q, *, r0=1.62, shell=4.0, broadening=0.05):
    f_avg, f2_avg = comp.form_factor_averages(q)
    debye = torch.sin(q * r0) / (q * r0) * torch.exp(-0.5 * (q * broadening) ** 2)
    S_model = 1.0 + shell * debye
    cmp = comp.compton(q, wavelength_A=WAVELENGTH_A)
    return f2_avg * S_model + cmp


def test_refine_recovers_wrong_scale():
    """Plant data on the correct scale, hand the refiner a wrong initial scale,
    and check it recovers ~1.0 from the physical constraints alone."""
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.7, 22.0, 2000, dtype=torch.float64)
    r = torch.linspace(0.0, 10.0, 800, dtype=torch.float64)
    I = _synthetic_iq(comp, q)  # already on the per-atom electron scale

    res = refine_normalization(
        q, I, comp, r, wavelength_A=WAVELENGTH_A, number_density=RHO,
        compton=True, q_max=20.0, r_min_phys=1.2,
        init_scale=0.5,            # deliberately wrong
        fit_offset=True, steps=80,
    )
    assert abs(res.scale - 1.0) < 0.05          # recovered to a few %
    assert res.history[-1] < res.history[0]      # loss decreased
    assert torch.all(torch.isfinite(res.G))


def test_refine_low_r_slope_enforced():
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.7, 22.0, 2000, dtype=torch.float64)
    r = torch.linspace(0.0, 10.0, 800, dtype=torch.float64)
    I = _synthetic_iq(comp, q)

    res = refine_normalization(
        q, I, comp, r, wavelength_A=WAVELENGTH_A, number_density=RHO,
        compton=True, q_max=20.0, r_min_phys=1.2, init_scale=0.8, steps=80,
    )
    # In the low-r window G(r) should track the line -4π ρ₀ r.
    lowr = (r > 0) & (r < 1.2)
    target = -4 * np.pi * RHO * r[lowr]
    rms = torch.sqrt(((res.G[lowr] - target) ** 2).mean())
    assert float(rms) < 0.5
