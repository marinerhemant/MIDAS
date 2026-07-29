"""02 — Faber-Ziman normalization: measured I(Q) -> S(Q), with sigma.

S(Q) -> 1 at high Q. The monoatomic case reduces exactly to the structure
factor; here we show a polyatomic (SiO2-like) example with sigma propagation.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_pdf import Composition, faber_ziman_S

WL = 0.1665
comp = Composition({"Si": 1, "O": 2})
q = torch.linspace(0.7, 22.0, 1500, dtype=torch.float64)

# synthesize a measured coherent intensity with one nearest-neighbour shell
f_avg, f2_avg = comp.form_factor_averages(q)
debye = torch.sin(q * 1.62) / (q * 1.62) * torch.exp(-0.5 * (q * 0.05) ** 2)
I = f2_avg * (1.0 + 4.0 * debye) + comp.compton(q, wavelength_A=WL)
sigma_I = torch.sqrt(I.clamp(min=1.0))

S, sigma_S = faber_ziman_S(I, q, comp, wavelength_A=WL, sigma_intensity=sigma_I)

print(f"S(Q) mean over last 100 points (should be ~1): {float(S[-100:].mean()):.3f}")
print(f"sigma_S median: {float(sigma_S.median()):.4f}")
print(f"S range: [{float(S.min()):.2f}, {float(S.max()):.2f}]")
