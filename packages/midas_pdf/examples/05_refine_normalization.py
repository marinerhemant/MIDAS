"""05 — Differentiable normalization: hand-tuning becomes optimization.

A deliberately wrong intensity scale is recovered by L-BFGS using only the
high-Q asymptote (<S> -> 1) and the low-r slope (G -> -4 pi rho0 r). Also shows
a polynomial background fit (bg_order) to absorb a smooth contaminant.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_pdf import Composition
from midas_pdf.refine import refine_normalization

WL, RHO = 0.1665, 0.0709
comp = Composition({"Si": 1, "O": 2})
q = torch.linspace(0.7, 22.0, 2000, dtype=torch.float64)
r = torch.linspace(0.0, 10.0, 800, dtype=torch.float64)

f_avg, f2_avg = comp.form_factor_averages(q)
debye = torch.sin(q * 1.62) / (q * 1.62) * torch.exp(-0.5 * (q * 0.05) ** 2)
I_clean = f2_avg * (1.0 + 4.0 * debye) + comp.compton(q, wavelength_A=WL)

# (a) recover a wrong scale
res = refine_normalization(q, I_clean, comp, r, wavelength_A=WL, number_density=RHO,
                           q_max=20.0, r_min_phys=1.2, init_scale=0.45, steps=80)
print(f"(a) scale: init 0.45 -> refined {res.scale:.4f} (target 1.0); loss {res.loss:.2e}")

# (b) remove a planted smooth (linear) background with bg_order=1
I_bg = I_clean + (3.0 + 0.4 * q)
res2 = refine_normalization(q, I_bg, comp, r, wavelength_A=WL, number_density=RHO,
                            q_max=20.0, r_min_phys=1.2, bg_order=1, steps=120)
print(f"(b) recovered background coefficients (≈ 3.0, 0.4*Qmax): {[round(c,3) for c in res2.bg_coef]}")
